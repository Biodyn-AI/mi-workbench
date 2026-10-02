#!/usr/bin/env python
"""Real-backend concurrency of 3-lens reviewer panels (PLOS revision 2, X6b).

For each level K (default 1, 2, 4, 8), K panels are started at once. Each is
one ``ConsensusReviewer.run_panel`` call with three lenses (rigour,
adversarial, biological plausibility), the same panel the engine uses, over a
fixed benchmark artifact. All panels share one Codex adapter instance:

- model ``gpt-5.6-luna``, effort ``low``;
- ``allow_tools=False``, which gives ``--sandbox read-only``;
- web search off (``-c web_search="disabled"``, added by the backend Codex
  adapter through the benchmark's ``make_adapter``), the same condition as X1;
- an empty working directory per panel.

Levels run one after another with a pause between them. Every adapter call
attempt is recorded, including retries made by the panel's ``run_with_retry``:

- wall-clock latency;
- success, error, error category and whether the retry logic treats the error
  as transient;
- provider-reported tokens (input, cached, output, reasoning);
- answering model and CLI version;
- the full raw output.

Per level the script reports: per-panel wall time, per-call latency, tokens,
failures, retries, rate-limit errors, and throughput (panels and calls per
minute).

Rate limits and errors are results. They are recorded, never hidden or
silently retried beyond the panel's own retry policy (``--max-retries``,
exponential backoff from ``--retry-base-delay``).

Outputs:

- ``experiments/orchestration/real_concurrency.json``: config, environment,
  per-level and per-panel summaries, and per-call records without the raw text
  (a SHA-256 and length stand in for it). Rewritten after every level.
- ``experiments/orchestration/raw/real_concurrency_K<k>.jsonl``: one line per
  call attempt, with the raw output, error, stderr log and parsed CLI events.

Resume: a level already present in the JSON with the same configuration
fingerprint is skipped. ``--force`` re-runs every level.

Usage (from automation/mi-workbench, env ``mi_workbench``):

    TMPDIR="/Volumes/Crucial X6/tmp_miw" PYTHONDONTWRITEBYTECODE=1 \\
        python scripts/real_concurrency.py                  # 1+2+4+8 = 15 panels, 45 calls
    python scripts/real_concurrency.py --levels 1 --dry-run # fake adapter, no cost
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "experiments" / "benchmark"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.adapters.base import BaseAdapter  # noqa: E402
from backend.adapters.cli_common import categorize_error  # noqa: E402
from backend.models import AdapterRunRequest, AdapterRunResult, WorkspaceContext  # noqa: E402
from backend.orchestrator.consensus import ConsensusReviewer  # noqa: E402
from backend.orchestrator.retry import error_category, is_transient_error  # noqa: E402

SCHEMA = "miw-orchestration-real-concurrency/1"
DEFAULT_ARTIFACT = BENCH_DIR / "artifacts" / "A01-clean.md"
DEFAULT_OUT = REPO_ROOT / "experiments" / "orchestration" / "real_concurrency.json"
PANEL = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]

_PANEL_ID: contextvars.ContextVar[str] = contextvars.ContextVar("miw_panel_id", default="")
_BENCH_COMMON = None


def bench_common():
    """The X1 benchmark helpers (``experiments/benchmark/common.py``), loaded by path.

    Reused so that prompts, the Codex adapter (web search off) and per-call
    working directories are exactly those of the planted-flaw benchmark.
    Loaded under a private module name to keep ``sys.path`` clean.
    """
    global _BENCH_COMMON
    if _BENCH_COMMON is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("miw_bench_common", BENCH_DIR / "common.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["miw_bench_common"] = mod
        spec.loader.exec_module(mod)
        _BENCH_COMMON = mod
    return _BENCH_COMMON


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    import math
    vals = sorted(values)
    if not vals:
        return 0.0
    k = max(1, math.ceil(q / 100.0 * len(vals)))
    return vals[min(k, len(vals)) - 1]


def summarize(values: Sequence[float], digits: int = 2) -> dict:
    vals = list(values)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(fmean(vals), digits),
        "p50": round(median(vals), digits),
        "p95": round(percentile(vals, 95), digits),
        "min": round(min(vals), digits),
        "max": round(max(vals), digits),
    }


# ── recording adapter ───────────────────────────────────────────────────


class RecordingAdapter(BaseAdapter):
    """Adapter proxy that records every call attempt (one ``run`` = one attempt).

    The panel is identified from a context variable set by the panel task; the
    lens from ``prompt_bundle.variables['role']``. Times are relative to
    ``t_origin`` (perf_counter).
    """

    def __init__(self, inner: BaseAdapter, t_origin: Optional[float] = None):
        self.inner = inner
        self.name = getattr(inner, "name", "adapter")
        self.t_origin = time.perf_counter() if t_origin is None else t_origin
        self.records: list[dict] = []
        self._attempt_no: Counter = Counter()

    def is_available(self) -> bool:
        return self.inner.is_available()

    async def smoke_test(self) -> dict:
        return await self.inner.smoke_test()

    async def cli_version(self) -> str:
        return await self.inner.cli_version()

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        panel = _PANEL_ID.get()
        role = (request.prompt_bundle.variables or {}).get("role", "")
        self._attempt_no[(panel, role)] += 1
        attempt = self._attempt_no[(panel, role)]
        started_utc = utc_now()
        t0 = time.perf_counter()
        exc_text = None
        result: Optional[AdapterRunResult] = None
        try:
            result = await self.inner.run(request)
            return result
        except Exception as exc:  # recorded; the panel's guard turns it into a failure
            exc_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            t1 = time.perf_counter()
            self.records.append(self._record(panel, role, attempt, started_utc, t0, t1,
                                             result, exc_text))

    def _record(self, panel, role, attempt, started_utc, t0, t1, result, exc_text) -> dict:
        r = result
        so = dict(getattr(r, "structured_output", {}) or {}) if r is not None else {}
        raw_usage = dict(getattr(r, "raw_usage", {}) or {}) if r is not None else {}
        error = (getattr(r, "error", None) if r is not None else None) or exc_text
        success = bool(r is not None and r.success)
        output = (getattr(r, "output", "") or "") if r is not None else ""
        exit_code = getattr(r, "exit_code", None) if r is not None else None
        return {
            "panel_id": panel,
            "role": role,
            "attempt": attempt,
            "started_utc": started_utc,
            "t_start_s": round(t0 - self.t_origin, 3),
            "t_end_s": round(t1 - self.t_origin, 3),
            "latency_s": round(t1 - t0, 3),
            "adapter_duration_s": round(float(getattr(r, "duration_seconds", 0.0) or 0.0), 3)
            if r is not None else None,
            "success": success,
            "error": error,
            # The adapter's own [category] prefix when present (re-categorising
            # the formatted text would turn "[rate_limit] Too Many Requests"
            # into "unknown"), else categorised from the text.
            "error_category": (None if success else
                               (error_category(error)
                                or categorize_error(int(exit_code or 0), error or ""))),
            "transient": bool(r is not None and is_transient_error(r)),
            "exit_code": exit_code,
            "model": getattr(r, "model", "") if r is not None else "",
            "reasoning_effort": getattr(r, "reasoning_effort", "") if r is not None else "",
            "cli_version": getattr(r, "cli_version", "") if r is not None else "",
            "input_tokens": int(getattr(r, "input_tokens", 0) or 0) if r is not None else 0,
            "cached_input_tokens": int(getattr(r, "cached_input_tokens", 0) or 0)
            if r is not None else 0,
            "output_tokens": int(getattr(r, "output_tokens", 0) or 0) if r is not None else 0,
            "reasoning_output_tokens": int(raw_usage.get("reasoning_output_tokens", 0) or 0),
            "tool_events": sum(1 for t in so.get("event_types", []) or [] if t == "item.started"),
            "output_chars": len(output),
            "output_sha256": sha256_text(output) if output else None,
            # raw-only fields (stripped from the summary JSON)
            "_raw_output": output,
            "_raw_log": (getattr(r, "raw_log", "") or "")[-20000:] if r is not None else "",
            "_structured_output": {k: v for k, v in so.items() if k != "agent_messages"},
        }


class FakeReviewAdapter(BaseAdapter):
    """Offline stand-in for tests / ``--dry-run``: JSON critiques after a delay."""

    name = "fake"

    def __init__(self, delay: float = 0.05, fail_roles: Sequence[str] = (),
                 transient_failures: int = 0):
        self.delay = delay
        self.fail_roles = set(fail_roles)
        self.transient_left = int(transient_failures)

    def is_available(self) -> bool:
        return True

    async def smoke_test(self) -> dict:
        return {"status": "ok", "adapter": "fake"}

    async def cli_version(self) -> str:
        return "fake-0"

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        await asyncio.sleep(self.delay)
        role = (request.prompt_bundle.variables or {}).get("role", "")
        base = dict(provider="fake", model=request.model or "fake",
                    reasoning_effort=request.reasoning_effort, cli_version="fake-0",
                    duration_seconds=self.delay)
        if self.transient_left > 0:
            self.transient_left -= 1
            return AdapterRunResult(success=False, error="[rate_limit] 429 Too Many Requests",
                                    exit_code=1, **base)
        if role in self.fail_roles:
            return AdapterRunResult(success=False, error="[unknown] simulated failure",
                                    exit_code=1, **base)
        body = {"critiques": [
            {"severity": "medium", "category": "statistics",
             "description": f"{role}: effect sizes are not reported for the main comparison",
             "required_fix": "report effect sizes"},
            {"severity": "low", "category": "reporting",
             "description": f"{role}: decimal places differ between tables",
             "required_fix": "standardise"},
        ], "overall_assessment": "fake"}
        out = "```json\n" + json.dumps(body) + "\n```"
        return AdapterRunResult(success=True, output=out, input_tokens=1000,
                                output_tokens=200, token_usage=1200, **base)


# ── experiment ──────────────────────────────────────────────────────────


def build_prompts(artifact_text: str) -> tuple[str, dict[str, Any]]:
    """User prompt and per-lens system prompts, resolved exactly as in X1."""
    bc = bench_common()
    user = bc.build_user_prompt(artifact_text)
    lenses = {}
    for role, ref in PANEL:
        info = bc.load_system_prompt(ref, role)
        lenses[role] = {"prompt_ref": ref, "text": info.text, "version": info.version,
                        "sha256": info.sha256}
    return user, lenses


async def run_panel_once(panel_id: str, adapter: BaseAdapter, user_prompt: str,
                         lenses: dict[str, Any], cfg: dict, ws_root: Optional[Path]) -> dict:
    bc = bench_common()
    _PANEL_ID.set(panel_id)
    ws = bc.make_workspace(ws_root)
    reviewer = ConsensusReviewer(
        panel=list(PANEL),
        # Pinned to the merge of the recorded results (the platform default at
        # the time); the default is now LLM adjudication, which would add one
        # model call per panel to the measured concurrency.
        similarity_method="jaccard", similarity_threshold=0.5,
        lens_timeout=cfg["lens_timeout"],
        max_retries=cfg["max_retries"],
        retry_base_delay=cfg["retry_base_delay"],
        retry_max_delay=cfg["retry_max_delay"],
        request_kwargs={"model": cfg["model"], "reasoning_effort": cfg["effort"],
                        "allow_tools": False},
    )
    started_utc = utc_now()
    t0 = time.perf_counter()
    error = None
    report: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    try:
        merged, _raw = await reviewer.run_panel(
            user_prompt, adapter, WorkspaceContext(workspace_path=str(ws)),
            system_prompt_builder=lambda role, ref: lenses[role]["text"],
        )
        report = reviewer.compute_consensus_report(merged)
        meta = merged.consensus_meta or {}
    except Exception as exc:  # noqa: BLE001 - recorded
        error = f"{type(exc).__name__}: {exc}"
    finally:
        wall = time.perf_counter() - t0
        bc.remove_workspace(ws)
    return {
        "panel_id": panel_id,
        "started_utc": started_utc,
        "wall_seconds": round(wall, 3),
        "error": error,
        "grade": report.get("grade"),
        "panel_failed": report.get("panel_failed", error is not None),
        "panel_partial": report.get("panel_partial", False),
        "failed_lenses": report.get("failed_lenses", []),
        "lens_status": report.get("lens_status", {}),
        "lens_errors": report.get("lens_errors", {}),
        "attempts": meta.get("attempts", {}),
        "lens_durations_s": meta.get("durations_seconds", {}),
        "raw_critique_counts": report.get("raw_critique_counts", {}),
        "parse_methods": report.get("parse_methods", {}),
        "total_merged": report.get("total_merged"),
        "multi_reviewer": report.get("multi_reviewer"),
        "severity_histogram": report.get("severity_histogram", {}),
    }


def summarize_level(k: int, wall: float, panels: list[dict], calls: list[dict]) -> dict:
    lens_calls = len(panels) * len(PANEL)
    ok_attempts = [c for c in calls if c["success"]]
    failed_attempts = [c for c in calls if not c["success"]]
    final_failed = sum(len(p.get("failed_lenses", [])) for p in panels)
    cats = Counter(c["error_category"] or "none" for c in failed_attempts)
    tok = {
        key: sum(c[key] for c in calls)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens",
                    "reasoning_output_tokens")
    }
    return {
        "k": k,
        "wall_seconds": round(wall, 3),
        "panels": len(panels),
        "panels_complete": sum(1 for p in panels
                               if not p["panel_failed"] and not p["panel_partial"]),
        "panels_partial": sum(1 for p in panels if p["panel_partial"]),
        "panels_failed": sum(1 for p in panels if p["panel_failed"]),
        "lens_calls": lens_calls,
        "attempts": len(calls),
        "retries": len(calls) - lens_calls,
        "failed_attempts": len(failed_attempts),
        "failed_lens_calls_final": final_failed,
        "error_categories": dict(cats),
        "rate_limit_errors": sum(1 for c in failed_attempts if c["error_category"] == "rate_limit"),
        "timeouts": sum(1 for c in failed_attempts if c["error_category"] == "timeout"
                        or "timeout" in (c["error"] or "").lower()),
        "tool_events": sum(c["tool_events"] for c in calls),
        "call_latency_s_successful": summarize([c["latency_s"] for c in ok_attempts]),
        "call_latency_s_all_attempts": summarize([c["latency_s"] for c in calls]),
        "panel_wall_s": summarize([p["wall_seconds"] for p in panels]),
        "throughput_panels_per_min": round(len(panels) / wall * 60.0, 3) if wall else 0.0,
        "throughput_successful_calls_per_min": round(len(ok_attempts) / wall * 60.0, 3)
        if wall else 0.0,
        "tokens": tok,
        "tokens_per_successful_call": {
            key: round(sum(c[key] for c in ok_attempts) / len(ok_attempts), 1)
            for key in tok
        } if ok_attempts else {},
        "models_answering": sorted({c["model"] for c in calls if c["model"]}),
        "cli_versions": sorted({c["cli_version"] for c in calls if c["cli_version"]}),
    }


async def run_level(k: int, adapter: RecordingAdapter, user_prompt: str, lenses: dict,
                    cfg: dict, ws_root: Optional[Path]) -> tuple[dict, list[dict]]:
    first = len(adapter.records)
    started_utc = utc_now()
    t0 = time.perf_counter()
    panels = await asyncio.gather(*(
        run_panel_once(f"K{k}-P{i + 1}", adapter, user_prompt, lenses, cfg, ws_root)
        for i in range(k)
    ))
    wall = time.perf_counter() - t0
    calls = adapter.records[first:]
    summary = summarize_level(k, wall, list(panels), calls)
    level = {
        "k": k,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "summary": summary,
        "panels": list(panels),
        "calls": [{kk: v for kk, v in c.items() if not kk.startswith("_")} for c in calls],
    }
    return level, calls


def config_fingerprint(cfg: dict) -> str:
    keys = ("model", "effort", "provider", "artifact_sha256", "user_prompt_sha256",
            "lens_prompt_sha256", "web_search", "lens_timeout", "max_retries",
            "retry_base_delay", "retry_max_delay", "dry_run")
    return sha256_text(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def make_codex_adapter(model: str, effort: str, web_search: str, codex_home: Optional[str]):
    bc = bench_common()
    return bc.make_adapter(model, effort, codex_home=codex_home,
                           codex_extra=bc.codex_extra_args(web_search))


async def run_experiment(
    levels: Sequence[int],
    *,
    out_path: Path,
    raw_dir: Path,
    artifact: Path,
    model: str,
    effort: str,
    adapter: Optional[BaseAdapter] = None,
    web_search: str = "disabled",
    codex_home: Optional[str] = None,
    lens_timeout: float = 600.0,
    max_retries: int = 2,
    retry_base_delay: float = 10.0,
    retry_max_delay: float = 60.0,
    pause: float = 10.0,
    force: bool = False,
    dry_run: bool = False,
    ws_root: Optional[Path] = None,
) -> dict:
    artifact_text = artifact.read_text()
    user_prompt, lenses = build_prompts(artifact_text)
    if adapter is None:
        adapter = (FakeReviewAdapter() if dry_run
                   else make_codex_adapter(model, effort, web_search, codex_home))
    rec = RecordingAdapter(adapter)
    try:
        rel_artifact = str(artifact.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel_artifact = str(artifact)
    cfg = {
        "provider": getattr(adapter, "name", ""),
        "model": model,
        "effort": effort,
        "allow_tools": False,
        "web_search": web_search,
        "artifact": rel_artifact,
        "artifact_sha256": sha256_text(artifact_text),
        "artifact_chars": len(artifact_text),
        "user_prompt_sha256": sha256_text(user_prompt),
        "panel": [{"role": r, "prompt_ref": lenses[r]["prompt_ref"],
                   "version": lenses[r]["version"], "sha256": lenses[r]["sha256"]}
                  for r, _ in PANEL],
        "lens_prompt_sha256": [lenses[r]["sha256"] for r, _ in PANEL],
        "lens_timeout": lens_timeout,
        "max_retries": max_retries,
        "retry_base_delay": retry_base_delay,
        "retry_max_delay": retry_max_delay,
        "retry_policy": ("backend.orchestrator.retry.run_with_retry inside run_panel: retries "
                         "only errors whose text contains timeout / rate limit / connection / "
                         "429 / 502 / 503 / overloaded; delay base*2^n (+-25% jitter), capped"),
        "pause_between_levels_s": pause,
        "levels": [int(k) for k in levels],
        "dry_run": dry_run,
        "similarity_method": "jaccard, threshold 0.5 (pinned; the ConsensusReviewer default "
                             "when the results were recorded)",
    }
    fp = config_fingerprint(cfg)

    previous: dict[int, dict] = {}
    if out_path.is_file() and not force:
        try:
            old = json.loads(out_path.read_text())
            if old.get("config_fingerprint") == fp:
                previous = {int(lv["k"]): lv for lv in old.get("levels", [])}
        except Exception:  # noqa: BLE001 - unreadable -> start fresh
            previous = {}

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        env["cli_version"] = await rec.cli_version()
    except Exception as exc:  # noqa: BLE001
        env["cli_version_error"] = str(exc)
    if not dry_run and cfg["provider"] in ("codex_cli", "codex"):
        try:
            env["codex_env"] = bench_common().codex_env_fingerprint(codex_home)
        except Exception as exc:  # noqa: BLE001
            env["codex_env_error"] = str(exc)
    try:
        env["loadavg_start"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "config": cfg,
        "config_fingerprint": fp,
        "environment": env,
        "levels": [],
        "summary_by_level": [],
    }

    def flush() -> None:
        lv_sorted = sorted(result["levels"], key=lambda lv: lv["k"])
        result["levels"] = lv_sorted
        base = next((lv["summary"] for lv in lv_sorted if lv["k"] == 1), None)
        rows = []
        for lv in lv_sorted:
            s = dict(lv["summary"])
            if base and base.get("panel_wall_s", {}).get("p50"):
                s["panel_wall_p50_vs_k1"] = round(
                    s["panel_wall_s"].get("p50", 0.0) / base["panel_wall_s"]["p50"], 3)
            if base and base.get("call_latency_s_successful", {}).get("p50"):
                s["call_latency_p50_vs_k1"] = round(
                    s["call_latency_s_successful"].get("p50", 0.0)
                    / base["call_latency_s_successful"]["p50"], 3)
            rows.append(s)
        result["summary_by_level"] = rows
        result["updated_utc"] = utc_now()
        _atomic_write(out_path, json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    ran_any = False
    for idx, k in enumerate(levels):
        k = int(k)
        if k in previous:
            print(f"K={k}: already recorded with this configuration; skipping (use --force)")
            result["levels"].append(previous[k])
            continue
        if ran_any and pause > 0:
            await asyncio.sleep(pause)
        print(f"K={k}: starting {k} panel(s) = {k * len(PANEL)} lens calls ...", flush=True)
        level, calls = await run_level(k, rec, user_prompt, lenses, cfg, ws_root)
        ran_any = True
        raw_path = raw_dir / f"real_concurrency_K{k}.jsonl"
        lines = []
        for c in calls:
            line = {kk: v for kk, v in c.items() if not kk.startswith("_")}
            line["raw_output"] = c["_raw_output"]
            line["raw_log"] = c["_raw_log"]
            line["structured_output"] = c["_structured_output"]
            line["config_fingerprint"] = fp
            lines.append(json.dumps(line, ensure_ascii=False))
        _atomic_write(raw_path, "\n".join(lines) + ("\n" if lines else ""))
        try:
            level["raw_file"] = str(raw_path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            level["raw_file"] = str(raw_path)
        result["levels"].append(level)
        flush()
        s = level["summary"]
        print(f"  wall={s['wall_seconds']}s panels ok/partial/failed="
              f"{s['panels_complete']}/{s['panels_partial']}/{s['panels_failed']} "
              f"attempts={s['attempts']} retries={s['retries']} "
              f"failed_attempts={s['failed_attempts']} rate_limit={s['rate_limit_errors']} "
              f"call p50/p95={s['call_latency_s_successful'].get('p50')}/"
              f"{s['call_latency_s_successful'].get('p95')}s "
              f"panel p50={s['panel_wall_s'].get('p50')}s", flush=True)
    try:
        env["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    flush()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--raw-dir", default=None,
                    help="default: <out dir>/raw")
    ap.add_argument("--web-search", choices=["disabled", "default"], default="disabled")
    ap.add_argument("--codex-home", default=None)
    ap.add_argument("--lens-timeout", type=float, default=600.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--retry-base-delay", type=float, default=10.0)
    ap.add_argument("--retry-max-delay", type=float, default=60.0)
    ap.add_argument("--pause", type=float, default=10.0, help="seconds between levels")
    ap.add_argument("--force", action="store_true", help="re-run levels already recorded")
    ap.add_argument("--dry-run", action="store_true", help="fake adapter, no model calls")
    args = ap.parse_args(argv)

    if args.dry_run and args.out == str(DEFAULT_OUT):
        # Never let a fake run overwrite the real results.
        out = Path(tempfile.gettempdir()) / "miw_real_concurrency_dryrun" / "real_concurrency.json"
    else:
        out = Path(args.out)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out.parent / "raw"
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if args.codex_home:
        # Every codex subprocess (and `codex --version`) must use this home,
        # matching the codex_env fingerprint recorded in the results.
        os.environ["CODEX_HOME"] = args.codex_home
    res = asyncio.run(run_experiment(
        levels, out_path=out, raw_dir=raw_dir, artifact=Path(args.artifact),
        model=args.model, effort=args.effort, web_search=args.web_search,
        codex_home=args.codex_home, lens_timeout=args.lens_timeout,
        max_retries=args.max_retries, retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay, pause=args.pause, force=args.force,
        dry_run=args.dry_run,
    ))
    print(f"\nWrote {out}")
    failed = sum(s["failed_lens_calls_final"] for s in res["summary_by_level"])
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
