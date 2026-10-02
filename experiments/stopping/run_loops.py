"""Run the stopping-rule calibration trajectories (X3) with the REAL engine.

One trajectory per (configuration, flawed item): the platform's
``LoopEngine`` drives the ``reviewer_consensus`` preset (three-lens consensus
panel <-> executor) with the Codex CLI adapter. The original flawed write-up
is passed as run config ``seed_executor_output`` (the executor's iteration-0
output), so the loop starts with a panel review of it:

    P0 E1 P1 E2 P2 E3 P3 E4 P4 E5 P5   (engine iterations 1..11)

Convergence and early stopping are disabled (``convergence_enabled=false``,
no ``grade_at_least``, budgets set far above use), so every trajectory runs to
the fixed horizon (``stop_reason == "max_iterations"``). Executor and reviewer
calls run with agent tools off (Codex strict tools-off: read-only sandbox, no
shell, no web search, no apps / plugins / MCP, user config ignored), in a
fresh empty working directory outside the repository.

Persistence (one directory per unit, ``<data-dir>/<config>/<item>/``):

* ``unit.json``: cache signature (model, effort, horizon, task / seed / prompt
  hashes, run config, Codex AGENTS.md hash) and platform provenance;
* ``seed_executor_output.md``: the original write-up (E0);
* ``runs/<run_id>/iter_NNNN/``: the engine's own artifacts (full step outputs,
  ``CONSENSUS.json``, one ``<lens>_output.md`` per lens, feedback files) and
  ``runs/<run_id>/run_meta.json`` (the runner's ``build_run_meta``);
* ``iterations.jsonl``: every completed ``IterationResult`` (engine order);
* ``calls/``: one JSON record per adapter call (full prompts, raw output,
  usage, command) and ``calls/system_prompts/<sha>.txt``;
* ``events.jsonl``: every engine event; ``attempts.jsonl``: failed attempts;
* ``states/E<k>.md`` and ``trajectory.json``: written when the trajectory is
  complete (per-step grade, Critical/High counts, panel status, executor text,
  executor-output similarity, tokens, wall time, stop events).

Resumable at step level: a failed step (adapter failure after the engine's
retries, a panel still partial after its retry, ...) ends the engine run; the
script archives the failed iteration's files under ``failed_attempts/`` and
resumes the run with the engine's own resume path (``resume_from``), up to
``--max-attempts-per-step`` attempts per step. The last attempt for a step
uses ``consensus_partial_policy=no_stop`` so a persistently partial panel is
kept (flagged ``grade_valid=false``) instead of blocking the trajectory.
Complete units are skipped; a unit whose ``unit.json`` signature differs is
reported as SIGNATURE_MISMATCH and never overwritten.

Units run strictly one at a time (a panel issues three concurrent calls, an
executor step one), so at most three model calls are in flight.

Usage:
    python experiments/stopping/run_loops.py --configs sol-medium --items A01-flawed
    python experiments/stopping/run_loops.py --configs sol-medium,luna-low
    python experiments/stopping/run_loops.py --status
Exit code: 0 all requested units complete, 1 some unit incomplete/failed,
2 fatal (auth / quota error, or the inline fallback prompt was used).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stopping_common as sc  # noqa: E402
from backend.artifacts.writer import write_artifact, write_run_meta  # noqa: E402
from backend.models import (  # noqa: E402
    AdapterRunRequest,
    AdapterRunResult,
    IterationResult,
    IterationStatus,
    ProviderName,
    RunState,
    RunStatus,
    WorkspaceContext,
)
from backend.orchestrator.consensus import ConsensusReviewer, parse_lens_output  # noqa: E402
from backend.orchestrator.convergence import _jaccard_similarity, grade_score  # noqa: E402
from backend.orchestrator.engine import LoopEngine  # noqa: E402
from backend.orchestrator.presets import get_preset  # noqa: E402
from backend.orchestrator.runner import build_run_meta  # noqa: E402
from backend.utils.output_cleaner import strip_thinking_traces  # noqa: E402

DEFAULT_ADAPTER_TIMEOUT = 1500
DEFAULT_MAX_ATTEMPTS_PER_STEP = 3
DEFAULT_BACKOFF = 60.0
MAX_BACKOFF = 600.0
ENGINE_MAX_RETRIES = 3
WORKSPACE_ID = "stopping-x3"
# Consensus merge of every trajectory: the platform default when X3 started
# (legacy jaccard, threshold 0.5; recorded as consensus_similarity_*_default in
# the first units' unit.json). The platform default changed afterwards (LLM
# adjudication, calibrated thresholds; experiments/benchmark/results/
# merge_eval.json), so it is pinned here to keep all configurations, and resumed
# units, on one merge. It is added to the engine run config outside
# ``run_config()``, so the cache signatures of existing units are unchanged.
X3_MERGE = {"consensus_similarity_method": "jaccard",
            "consensus_similarity_threshold": 0.5}
_ITER_RE = re.compile(r"iter_(\d+)")
_LOG_EVENTS = {
    "loop_started", "loop_finished", "adapter_retry", "consensus_panel_partial",
    "consensus_panel_failed", "budget_exceeded", "convergence_detected",
    "stop_condition_met",
}
STOP_EVENT_TYPES = {
    "loop_finished", "budget_exceeded", "convergence_detected", "stop_condition_met",
    "consensus_panel_partial", "consensus_panel_failed", "loop_cancelled",
}


class FatalRunError(RuntimeError):
    """Stop the whole script (auth / quota failure, fallback prompt used)."""


def _is_fatal_error(error: Optional[str]) -> bool:
    e = (error or "").lower()
    return "[auth]" in e or "[quota]" in e


def _fallback_prompt(role: str, system_prompt: str) -> bool:
    """True if the engine used its inline fallback instead of the prompt YAML."""
    return (system_prompt or "").startswith(f"You are the {role} in an automated MI")


# ── run configuration ───────────────────────────────────────────────────


def run_config(effort: str, seed_text: str, *, partial_policy: str = "fail",
               adapter_timeout: int = DEFAULT_ADAPTER_TIMEOUT) -> dict[str, Any]:
    """Run config of one trajectory (the part in the cache signature). The
    consensus merge is not part of it: ``X3_MERGE`` (the platform default when
    X3 started) is added by the engine call and recorded per panel in
    CONSENSUS.json and in unit.json."""
    return {
        "reasoning_effort": effort,
        "seed_executor_output": seed_text,
        "executor_allow_tools": False,
        "reviewer_allow_tools": False,
        "convergence_enabled": False,
        "consensus_gate": False,
        "budget_max_tokens": 10_000_000_000,
        "budget_max_cost": 1_000_000.0,
        "adapter_timeout": int(adapter_timeout),
        "consensus_lens_timeout": int(adapter_timeout),
        "max_retries": ENGINE_MAX_RETRIES,
        "consensus_partial_policy": partial_policy,
    }


def prompt_hashes() -> dict[str, str]:
    """sha256 of every prompt YAML the preset uses (executor + 3 lenses)."""
    loop = get_preset(sc.LOOP_PRESET)
    out: dict[str, str] = {}
    for node in loop.nodes:
        if node.prompt_ref:
            p = sc.REPO_ROOT / "prompts" / f"{node.prompt_ref}.yaml"
            out[node.prompt_ref] = sc.sha256_file(p) if p.is_file() else "MISSING"
    return out


def codex_agents_md_sha() -> Optional[str]:
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    p = home / "AGENTS.md"
    return sc.sha256_file(p) if p.is_file() else None


def unit_signature(config: str, item: str, provider: str, model: str, effort: str,
                   horizon: int, seed_text: str, adapter_timeout: int) -> dict[str, Any]:
    cfg = run_config(effort, "", adapter_timeout=adapter_timeout)
    cfg.pop("seed_executor_output", None)
    cfg.pop("consensus_partial_policy", None)
    return {
        "schema": sc.UNIT_SCHEMA,
        "config": config,
        "item": item,
        "provider": provider,
        "model": model,
        "effort": effort,
        "horizon": int(horizon),
        "loop_preset": sc.LOOP_PRESET,
        "task_sha256": sc.sha256_text(sc.EXECUTOR_TASK),
        "seed_sha256": sc.sha256_text(seed_text),
        "prompt_sha256": prompt_hashes(),
        "run_config": cfg,
        "codex_agents_md_sha256": codex_agents_md_sha() if provider == "codex" else None,
    }


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", str(sc.REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def platform_provenance() -> dict[str, Any]:
    default_merger = ConsensusReviewer.from_config({})
    engine_py = sc.REPO_ROOT / "backend" / "orchestrator" / "engine.py"
    return {
        "git_head": _git_head(),
        "engine_py_sha256": sc.sha256_file(engine_py),
        "consensus_similarity_method_default": getattr(default_merger, "similarity_method", None),
        "consensus_similarity_threshold_default": getattr(default_merger, "similarity_threshold", None),
        "consensus_merge_used": dict(X3_MERGE),
        "python": sys.version.split()[0],
    }


# ── adapters ────────────────────────────────────────────────────────────


def make_platform_adapter(provider: str, model: str, effort: str):
    """Backend adapter with agent tools off (Codex strict tools-off)."""
    if provider == "codex":
        from backend.adapters.codex_cli import CodexCliAdapter
        return CodexCliAdapter(binary="codex", model=model, reasoning_effort=effort,
                               allow_tools=False)
    if provider == "mock":
        from backend.adapters.mock import MockAdapter
        return MockAdapter(min_delay=0, max_delay=0)
    raise ValueError(f"unknown provider {provider!r}")


def make_call_workspace(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ws_", dir=str(root)))


class RecordingAdapter:
    """Wraps the platform adapter used by the engine.

    * every call gets a fresh, empty working directory outside the repository
      (``--cd`` for Codex), so no project file (AGENTS.md, ground truth) is in
      the agent's working tree; the engine's own workspace path (the unit
      directory, used for artifacts and resume) is never handed to the CLI;
    * every call (each retry attempt, each lens) is recorded with its full
      user prompt, system-prompt hash, raw output, usage and command.
    """

    def __init__(self, inner: Any, calls_dir: Path, workspace_root: Path,
                 on_call: Optional[Callable[[dict], None]] = None):
        self.inner = inner
        self.name = getattr(inner, "name", "adapter")
        self.calls_dir = Path(calls_dir)
        self.workspace_root = Path(workspace_root)
        self.on_call = on_call
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        (self.calls_dir / "system_prompts").mkdir(parents=True, exist_ok=True)
        self._seq = len(list(self.calls_dir.glob("call_*.json")))
        self.fallback_prompt_roles: list[str] = []

    def is_available(self) -> bool:
        fn = getattr(self.inner, "is_available", None)
        return bool(fn()) if callable(fn) else True

    async def smoke_test(self) -> dict:
        fn = getattr(self.inner, "smoke_test", None)
        return await fn() if fn else {"status": "ok"}

    async def cli_version(self) -> str:
        fn = getattr(self.inner, "cli_version", None)
        return await fn() if fn else ""

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        self._seq += 1
        seq = self._seq
        bundle = request.prompt_bundle
        role = (bundle.variables or {}).get("role", "") or "unknown"
        m = _ITER_RE.search(request.workspace_context.iteration_dir or "")
        iteration = int(m.group(1)) if m else None
        sys_sha = sc.sha256_text(bundle.system_prompt or "")
        sp_path = self.calls_dir / "system_prompts" / f"{sys_sha}.txt"
        if not sp_path.exists():
            sc.atomic_write_text(sp_path, bundle.system_prompt or "")
        fallback = _fallback_prompt(role, bundle.system_prompt or "")
        if fallback:
            self.fallback_prompt_roles.append(role)
        ws = make_call_workspace(self.workspace_root)
        req = request.model_copy(update={"workspace_context": WorkspaceContext(
            workspace_path=str(ws),
            run_dir=request.workspace_context.run_dir,
            iteration_dir=request.workspace_context.iteration_dir,
        )})
        started = sc.utc_now()
        t0 = time.monotonic()
        try:
            result = await self.inner.run(req)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # recorded, then re-raised to the engine
            self._write_record(seq, iteration, role, request, ws, started, t0, None,
                               f"{type(exc).__name__}: {exc}", sys_sha, fallback)
            raise
        finally:
            shutil.rmtree(ws, ignore_errors=True)
        self._write_record(seq, iteration, role, request, ws, started, t0, result, None,
                           sys_sha, fallback)
        return result

    def _write_record(self, seq, iteration, role, request, ws, started, t0, result,
                      exception, sys_sha, fallback) -> None:
        so = dict(getattr(result, "structured_output", {}) or {}) if result else {}
        rec = {
            "schema": sc.CALL_SCHEMA,
            "seq": seq,
            "iteration": iteration,
            "label": sc.step_label(iteration) if iteration else None,
            "role": role,
            "started_at": started,
            "wall_seconds": round(time.monotonic() - t0, 3),
            "request": {
                "model": request.model,
                "reasoning_effort": request.reasoning_effort,
                "allow_tools": request.allow_tools,
                "timeout_seconds": request.timeout_seconds,
                "adapter_options": dict(request.adapter_options or {}),
                "cli_cwd": str(ws),
                "engine_workspace_path": request.workspace_context.workspace_path,
                "system_prompt_sha256": sys_sha,
                "system_prompt_is_inline_fallback": fallback,
                "user_prompt": request.prompt_bundle.user_prompt,
                "user_prompt_sha256": sc.sha256_text(request.prompt_bundle.user_prompt),
            },
            "exception": exception,
        }
        if result is not None:
            rec["result"] = {
                "success": bool(result.success),
                "error": result.error,
                "exit_code": result.exit_code,
                "output": result.output,
                "duration_seconds": result.duration_seconds,
                "provider": result.provider,
                "model": result.model,
                "reasoning_effort": result.reasoning_effort,
                "cli_version": result.cli_version,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cached_input_tokens": result.cached_input_tokens,
                "token_usage": result.token_usage,
                "raw_usage": dict(result.raw_usage or {}),
                "adapter_meta": so.get("adapter_meta"),
                "tool_counts": so.get("tool_counts"),
            }
        name = f"call_{seq:04d}_iter{iteration or 0:04d}_{role}.json"
        sc.atomic_write_json(self.calls_dir / name, rec)
        if self.on_call:
            try:
                self.on_call(rec)
            except Exception:
                pass


# ── iteration persistence ───────────────────────────────────────────────


def load_iterations(udir: Path) -> list[IterationResult]:
    path = Path(udir) / "iterations.jsonl"
    if not path.exists():
        return []
    out: list[IterationResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(IterationResult.model_validate_json(line))
        except Exception:
            continue
    return sorted(out, key=lambda i: i.iteration_number)


def completed_prefix(iters: list[IterationResult], horizon: int) -> list[IterationResult]:
    """The completed iterations 1..m (contiguous, expected roles)."""
    roles = sc.expected_roles(horizon)
    out: list[IterationResult] = []
    by_num: dict[int, IterationResult] = {}
    for it in iters:
        if it.status == IterationStatus.COMPLETED:
            by_num[it.iteration_number] = it
    for n in range(1, len(roles) + 1):
        it = by_num.get(n)
        if it is None or it.role != roles[n - 1]:
            break
        out.append(it)
    return out


# ── trajectory record ───────────────────────────────────────────────────


def _tokens(it: IterationResult) -> dict[str, int]:
    return {"input": it.input_tokens, "output": it.output_tokens,
            "cached_input": it.cached_input_tokens, "total": it.token_usage}


def _lens_summary(text: str) -> dict[str, Any]:
    parsed = parse_lens_output(strip_thinking_traces(text or ""))
    sev: dict[str, int] = {}
    for c in parsed.critiques:
        sev[c.severity.value] = sev.get(c.severity.value, 0) + 1
    return {"parse_method": parsed.method, "n_critiques": len(parsed.critiques),
            "severity": sev, "critical": sev.get("critical", 0), "high": sev.get("high", 0)}


def read_events(udir: Path) -> list[dict]:
    path = Path(udir) / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def build_trajectory(udir: Path, config: str, item: str, horizon: int,
                     model: str, effort: str) -> dict[str, Any]:
    """Assemble ``trajectory.json`` from the engine's persisted outputs.

    Raises ``ValueError`` if the trajectory is not complete."""
    udir = Path(udir)
    run_id = sc.run_id_for(config, item)
    run_dir = udir / "runs" / run_id
    iters = completed_prefix(load_iterations(udir), horizon)
    total = sc.total_iterations(horizon)
    if len(iters) != total:
        raise ValueError(f"trajectory incomplete: {len(iters)}/{total} iterations")
    seed = (udir / "seed_executor_output.md").read_text(encoding="utf-8")
    meta_path = run_dir / "run_meta.json"
    run_meta = sc.read_json(meta_path) if meta_path.exists() else {}
    engine_sims = list(((run_meta.get("convergence") or {}).get("similarity_scores")) or [])

    states: list[dict[str, Any]] = []
    texts = [seed]
    for k in range(1, horizon + 1):
        p = run_dir / f"iter_{2 * k:04d}" / "executor_output.md"
        texts.append(p.read_text(encoding="utf-8"))
    states_dir = udir / "states"
    for k, text in enumerate(texts):
        sc.atomic_write_text(states_dir / f"E{k}.md", text)
        sim = None if k == 0 else _jaccard_similarity(texts[k - 1], text)
        nn = sc.new_numbers(text, seed)
        states.append({
            "k": k,
            "label": f"E{k}",
            "iteration": 2 * k if k else 0,
            "path": f"states/E{k}.md",
            "sha256": sc.sha256_text(text),
            "chars": len(text),
            "words": len(text.split()),
            "similarity_to_previous": sim,
            "n_new_numbers": len(nn),
            "new_numbers": nn[:60],
            "pending_markers": sc.pending_markers(text),
        })
    sims = [s["similarity_to_previous"] for s in states[1:]]
    sim_match = (len(engine_sims) == len(sims)
                 and all(abs(a - b) < 1e-12 for a, b in zip(engine_sims, sims)))

    panels: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for it in iters:
        n = it.iteration_number
        label = sc.step_label(n)
        step: dict[str, Any] = {
            "iteration": n,
            "label": label,
            "role": it.role,
            "status": it.status.value,
            "started_at": it.started_at.isoformat() if it.started_at else None,
            "completed_at": it.completed_at.isoformat() if it.completed_at else None,
            "duration_seconds": it.duration_seconds,
            "tokens": _tokens(it),
            "model": it.model,
            "reasoning_effort": it.reasoning_effort,
            "cli_version": it.cli_version,
            "agent_tools": it.agent_tools,
        }
        if it.role == sc.EXECUTOR_ROLE:
            k = n // 2
            step.update({"state": f"E{k}", "output_path": states[k]["path"],
                         "similarity_to_previous": states[k]["similarity_to_previous"],
                         "chars": states[k]["chars"]})
        else:
            j = (n - 1) // 2
            iter_dir = run_dir / f"iter_{n:04d}"
            cj_path = iter_dir / "CONSENSUS.json"
            cj = sc.read_json(cj_path) if cj_path.exists() else {}
            report = cj.get("report") or (it.consensus_report or {})
            lens_status = report.get("lens_status") or {}
            lenses = {}
            for role in sc.LENS_ROLES:
                lp = iter_dir / f"{role}_output.md"
                text = lp.read_text(encoding="utf-8") if lp.exists() else ""
                info = _lens_summary(text)
                info.update({"status": lens_status.get(role),
                             "file": str(lp.relative_to(udir)) if lp.exists() else None,
                             "chars": len(text)})
                lenses[role] = info
            panel_partial = bool(report.get("panel_partial", False))
            grade_valid = report.get("grade_valid", not panel_partial)
            hist = report.get("severity_histogram") or {}
            panel = {
                "j": j,
                "label": label,
                "iteration": n,
                "reviews_state": f"E{j}",
                "grade": it.grade,
                "grade_score": grade_score(it.grade),
                "critical": it.critical_count or 0,
                "high": it.high_count or 0,
                "n_critiques": len(cj.get("critiques") or []),
                "severity_histogram": hist,
                "unresolved_critical": int(report.get("unresolved_critical", it.critical_count or 0) or 0),
                "panel_partial": panel_partial,
                "grade_valid": bool(grade_valid),
                "failed_lenses": list(report.get("failed_lenses") or []),
                "panel_status": "partial" if panel_partial else "complete",
                "similarity_method": report.get("similarity_method"),
                "similarity_threshold": report.get("similarity_threshold"),
                "n_groups_multi_lens": report.get("n_groups_multi_lens"),
                "escalated_by_agreement": report.get("escalated_by_agreement"),
                "lenses": lenses,
                "consensus_json": str(cj_path.relative_to(udir)) if cj_path.exists() else None,
                "attempts": len(report.get("attempts") or (it.consensus_report or {}).get("attempts") or []),
                "tokens": _tokens(it),
                "duration_seconds": it.duration_seconds,
            }
            panels.append(panel)
            step.update({k2: panel[k2] for k2 in (
                "grade", "grade_score", "critical", "high", "n_critiques", "panel_status",
                "panel_partial", "grade_valid", "failed_lenses", "unresolved_critical",
                "reviews_state")})
        steps.append(step)

    events = read_events(udir)
    stop_events = [e for e in events if e.get("type") in STOP_EVENT_TYPES]
    attempts = []
    ap = udir / "attempts.jsonl"
    if ap.exists():
        for line in ap.read_text(encoding="utf-8").splitlines():
            try:
                attempts.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    last_finish = next((e for e in reversed(events) if e.get("type") == "loop_finished"), {})
    methods = sorted({p["similarity_method"] for p in panels if p.get("similarity_method")})
    return {
        "schema": sc.TRAJECTORY_SCHEMA,
        "config": config,
        "item": item,
        "model": model,
        "effort": effort,
        "horizon": horizon,
        "run_id": run_id,
        "complete": True,
        "stop_reason": last_finish.get("stop_reason") or run_meta.get("stop_reason"),
        "expected_order": [sc.step_label(n) for n in range(1, total + 1)],
        "observed_order": [s["label"] for s in steps],
        "roles": [s["role"] for s in steps],
        "similarity_methods": methods,
        "states": states,
        "panels": panels,
        "steps": steps,
        "engine_similarity_scores": engine_sims,
        "similarity_matches_engine": sim_match,
        "stop_events": stop_events,
        "failed_attempts": attempts,
        "totals": {
            "tokens": sum(s["tokens"]["total"] for s in steps),
            "input_tokens": sum(s["tokens"]["input"] for s in steps),
            "output_tokens": sum(s["tokens"]["output"] for s in steps),
            "cached_input_tokens": sum(s["tokens"]["cached_input"] for s in steps),
            "duration_seconds": round(sum(s["duration_seconds"] or 0 for s in steps), 3),
            "n_calls": len(list((udir / "calls").glob("call_*.json"))),
        },
        "generated_at": sc.utc_now(),
    }


# ── runner ──────────────────────────────────────────────────────────────


class LoopRunner:
    def __init__(self, data_dir: Path, *, provider: str = "codex",
                 horizon: int = sc.HORIZON,
                 max_attempts_per_step: int = DEFAULT_MAX_ATTEMPTS_PER_STEP,
                 backoff: float = DEFAULT_BACKOFF,
                 adapter_timeout: int = DEFAULT_ADAPTER_TIMEOUT,
                 workspace_root: Optional[Path] = None,
                 adapter_factory: Optional[Callable[[str, str, str], Any]] = None,
                 items: Optional[dict] = None,
                 configs: Optional[dict[str, dict[str, str]]] = None):
        self.data_dir = Path(data_dir)
        self.provider = provider
        self.horizon = int(horizon)
        self.max_attempts_per_step = max(1, int(max_attempts_per_step))
        self.backoff = float(backoff)
        self.adapter_timeout = int(adapter_timeout)
        root = Path(workspace_root) if workspace_root else Path(tempfile.gettempdir()) / "miw_stop_ws"
        self.workspace_root = root.resolve()
        repo = sc.REPO_ROOT.resolve()
        if self.workspace_root == repo or repo in self.workspace_root.parents:
            raise ValueError(f"workspace root {self.workspace_root} is inside the repository; "
                             "set TMPDIR or --workspace-root outside it")
        self.adapter_factory = adapter_factory or make_platform_adapter
        self.items = items if items is not None else sc.flawed_items()
        self.configs = configs or sc.CONFIGS
        self.log_path = self.data_dir / "logs" / "run_loops.log"

    def log(self, msg: str) -> None:
        line = f"{sc.utc_now()} {msg}"
        sc.append_line(self.log_path, line)
        print(line, flush=True)

    # one unit ------------------------------------------------------------

    def _prepare_unit(self, config: str, item_id: str) -> tuple[Path, dict, str]:
        cfg = self.configs[config]
        item = self.items[item_id]
        seed = item.artifact_text
        udir = sc.unit_dir(self.data_dir, config, item_id)
        sig = unit_signature(config, item_id, self.provider, cfg["model"], cfg["effort"],
                             self.horizon, seed, self.adapter_timeout)
        upath = udir / "unit.json"
        if upath.exists():
            old = sc.read_json(upath)
            if old.get("signature") != sig:
                diff = sorted(k for k in set(sig) | set(old.get("signature") or {})
                              if (old.get("signature") or {}).get(k) != sig.get(k))
                raise SignatureMismatch(f"{config}/{item_id}: unit.json signature differs "
                                        f"({', '.join(diff)})")
        else:
            udir.mkdir(parents=True, exist_ok=True)
            sc.atomic_write_text(udir / "seed_executor_output.md", seed)
            sc.atomic_write_json(upath, {
                "signature": sig,
                "created_at": sc.utc_now(),
                "platform": platform_provenance(),
                "task": sc.EXECUTOR_TASK,
                "artifact_path": str(item.artifact_path.relative_to(sc.REPO_ROOT)),
                "planted_flaws": item.flaw_ids,
            })
        return udir, sig, seed

    def _archive_failed(self, udir: Path, run_id: str, first_bad: int, attempt_tag: str) -> None:
        run_dir = udir / "runs" / run_id
        if not run_dir.exists():
            return
        for d in sorted(run_dir.glob("iter_*")):
            try:
                n = int(d.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if n >= first_bad:
                dest = udir / "failed_attempts" / attempt_tag / d.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(str(d), str(dest))

    async def _run_engine_once(self, config: str, item_id: str, udir: Path, seed: str,
                               done: list[IterationResult], partial_policy: str,
                               stop_after: Optional[int] = None):
        cfg = self.configs[config]
        model, effort = cfg["model"], cfg["effort"]
        run_id = sc.run_id_for(config, item_id)
        run_dir = udir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        total = sc.total_iterations(self.horizon)
        provider_enum = ProviderName.CODEX_CLI if self.provider == "codex" else ProviderName.MOCK
        state = RunState(
            run_id=run_id, workspace_id=WORKSPACE_ID, loop_preset=sc.LOOP_PRESET,
            task=sc.EXECUTOR_TASK, provider=provider_enum, model=model,
            max_iterations=min(total, int(stop_after or total)),
            config={**run_config(effort, seed, partial_policy=partial_policy,
                                 adapter_timeout=self.adapter_timeout),
                    **X3_MERGE, "revision_budget": None},
        )
        resume_from = 0
        if done:
            resume_from = len(done)
            state.status = RunStatus.PAUSED
            state.current_iteration = resume_from
            state.iterations = [it.model_copy() for it in done]
            state.total_tokens = sum(it.token_usage for it in done)
            state.total_input_tokens = sum(it.input_tokens for it in done)
            state.total_output_tokens = sum(it.output_tokens for it in done)
            state.total_cached_input_tokens = sum(it.cached_input_tokens for it in done)
            state.started_at = done[0].started_at
        loop_def = get_preset(sc.LOOP_PRESET)
        loop_def.max_iterations = total

        inner = self.adapter_factory(self.provider, model, effort)
        adapter = RecordingAdapter(inner, udir / "calls", self.workspace_root)
        events_path = udir / "events.jsonl"
        iters_path = udir / "iterations.jsonl"
        label = f"{config} {item_id}"

        def on_event(event_type: str, data: dict) -> None:
            rec = {"t": sc.utc_now(), "type": event_type, **data}
            try:
                sc.append_line(events_path, json.dumps(rec, default=str))
            except Exception:
                pass
            if event_type in _LOG_EVENTS:
                brief = {k: v for k, v in data.items()
                         if k in ("iteration", "stop_reason", "error", "attempt",
                                  "failed_lenses", "will_retry")}
                self.log(f"{label} event {event_type} {json.dumps(brief, default=str)[:300]}")

        def writer(n: int, name: str, content: str) -> None:
            write_artifact(run_dir, n, name, content)

        async def on_iteration(it: IterationResult, st: RunState) -> None:
            sc.append_line(iters_path, it.model_dump_json())
            extra = ""
            if it.role == sc.PANEL_ROLE:
                rep = it.consensus_report or {}
                extra = (f" grade={it.grade} C={it.critical_count} H={it.high_count}"
                         f" partial={bool(rep.get('panel_partial'))}")
            self.log(f"{label} {sc.step_label(it.iteration_number)} (iter {it.iteration_number}) "
                     f"{it.role} {it.status.value} tokens={it.token_usage} "
                     f"dur={it.duration_seconds:.0f}s{extra}"
                     + (f" error={(it.error or '')[:200]}" if it.error else ""))

        engine = LoopEngine(adapter=adapter, on_event=on_event, artifact_writer=writer,
                            max_retries=ENGINE_MAX_RETRIES, on_iteration=on_iteration)
        final = await engine.run_loop(state, loop_def, workspace_path=str(udir),
                                      resume_from=resume_from)
        try:
            meta = build_run_meta(final, loop_def, engine._convergence.get_metrics(),
                                  tool_settings=engine.tool_settings())
            write_run_meta(run_dir, meta)
        except Exception as exc:
            self.log(f"{label} WARNING run_meta.json not written: {exc}")
        if adapter.fallback_prompt_roles:
            raise FatalRunError(f"{label}: inline fallback system prompt used for roles "
                                f"{sorted(set(adapter.fallback_prompt_roles))}")
        return final

    async def run_unit(self, config: str, item_id: str) -> str:
        """Run / resume one trajectory; returns 'complete', 'cached',
        'failed' or 'mismatch'."""
        cfg = self.configs[config]
        try:
            udir, _sig, seed = self._prepare_unit(config, item_id)
        except SignatureMismatch as exc:
            self.log(f"SIGNATURE_MISMATCH {exc}")
            return "mismatch"
        traj_path = udir / "trajectory.json"
        if traj_path.exists():
            return "cached"
        total = sc.total_iterations(self.horizon)
        run_id = sc.run_id_for(config, item_id)
        failures_at: dict[int, int] = {}   # failed tries per engine iteration
        while True:
            all_iters = load_iterations(udir)
            done = completed_prefix(all_iters, self.horizon)
            if len(done) == total:
                traj = build_trajectory(udir, config, item_id, self.horizon,
                                        cfg["model"], cfg["effort"])
                sc.atomic_write_json(traj_path, traj)
                self.log(f"{config} {item_id} COMPLETE order={' '.join(traj['observed_order'])} "
                         f"tokens={traj['totals']['tokens']} "
                         f"step_time={traj['totals']['duration_seconds']:.0f}s "
                         f"sim_match={traj['similarity_matches_engine']}")
                return "complete"
            if len(done) != len(all_iters):
                # Keep only the completed prefix; failed iterations go to attempts.jsonl.
                sc.atomic_write_text(udir / "iterations.jsonl",
                                     "".join(it.model_dump_json() + "\n" for it in done))
            nxt = len(done) + 1
            prior = failures_at.get(nxt, 0)
            if prior >= self.max_attempts_per_step:
                self.log(f"{config} {item_id} FAILED: step {sc.step_label(nxt)} (iter {nxt}) "
                         f"failed {prior} times; rerun the script to retry")
                return "failed"
            # Last try of a step: keep a still-partial panel (no_stop) and run
            # only that step, so later steps use the strict policy again.
            last_try = self.max_attempts_per_step > 1 and prior == self.max_attempts_per_step - 1
            policy = "no_stop" if last_try else "fail"
            stop_after = nxt if last_try else total
            tag = f"iter{nxt:04d}_try{prior + 1}_{int(time.time())}"
            self._archive_failed(udir, run_id, nxt, tag)
            self.log(f"{config} {item_id} {'start' if nxt == 1 else 'resume'} at "
                     f"{sc.step_label(nxt)} (iter {nxt}) try {prior + 1} "
                     f"partial_policy={policy}" + (f" stop_after={stop_after}" if last_try else ""))
            final = await self._run_engine_once(config, item_id, udir, seed, done, policy,
                                                stop_after=stop_after)
            reason = final.stop_reason or ""
            if reason == "max_iterations":
                continue
            failed_its = [it for it in final.iterations if it.status != IterationStatus.COMPLETED]
            failed_at = failed_its[-1].iteration_number if failed_its else max(
                nxt, int(final.current_iteration or 0))
            failures_at[failed_at] = failures_at.get(failed_at, 0) + 1
            err = final.error or (failed_its[-1].error if failed_its else "") or ""
            sc.append_line(udir / "attempts.jsonl", json.dumps({
                "t": sc.utc_now(), "iteration": failed_at, "label": sc.step_label(failed_at),
                "try": failures_at[failed_at], "run_started_at": sc.step_label(nxt),
                "partial_policy": policy, "stop_reason": reason, "error": err[:4000],
                "failed_iterations": [it.model_dump(mode="json") for it in failed_its],
            }, default=str))
            self.log(f"{config} {item_id} step {sc.step_label(failed_at)} failed "
                     f"(try {failures_at[failed_at]}): stop_reason={reason} error={err[:300]}")
            if _is_fatal_error(err):
                raise FatalRunError(f"{config} {item_id}: {err[:500]}")
            if not reason.startswith("failed:"):
                self.log(f"{config} {item_id} FAILED: unexpected stop reason {reason!r}")
                return "failed"
            delay = min(MAX_BACKOFF, self.backoff * failures_at[failed_at])
            if delay > 0:
                await asyncio.sleep(delay)

    async def run(self, configs: list[str], items: list[str]) -> dict[str, list[str]]:
        outcome: dict[str, list[str]] = {}
        for config in configs:
            for item_id in items:
                t0 = time.monotonic()
                status = await self.run_unit(config, item_id)
                outcome.setdefault(status, []).append(f"{config}/{item_id}")
                if status != "cached":
                    self.log(f"{config} {item_id} unit status={status} "
                             f"wall={time.monotonic() - t0:.0f}s")
        return outcome


class SignatureMismatch(RuntimeError):
    pass


# ── status ──────────────────────────────────────────────────────────────


def unit_status(data_dir: Path, config: str, item: str, horizon: int = sc.HORIZON) -> dict:
    udir = sc.unit_dir(data_dir, config, item)
    total = sc.total_iterations(horizon)
    if not udir.exists():
        return {"config": config, "item": item, "state": "not_started", "done": 0, "total": total}
    iters = load_iterations(udir)
    done = completed_prefix(iters, horizon)
    failures = 0
    ap = udir / "attempts.jsonl"
    last_error = ""
    if ap.exists():
        lines = [ln for ln in ap.read_text(encoding="utf-8").splitlines() if ln.strip()]
        failures = len(lines)
        if lines:
            try:
                last_error = json.loads(lines[-1]).get("error", "")[:120]
            except json.JSONDecodeError:
                pass
    state = "complete" if (udir / "trajectory.json").exists() else (
        "in_progress" if done else "started")
    grades = [it.grade for it in done if it.role == sc.PANEL_ROLE]
    return {
        "config": config, "item": item, "state": state, "done": len(done), "total": total,
        "next": sc.step_label(len(done) + 1) if len(done) < total else "-",
        "grades": grades, "tokens": sum(it.token_usage for it in done),
        "step_seconds": round(sum(it.duration_seconds for it in done)),
        "failed_attempts": failures, "last_error": last_error,
    }


def print_status(data_dir: Path, configs: list[str], items: list[str]) -> None:
    rows = [unit_status(data_dir, c, i) for c in configs for i in items]
    for r in rows:
        print(f"{r['config']:<11} {r['item']:<11} {r['state']:<11} {r['done']:>2}/{r['total']} "
              f"next={r.get('next', '-'):<3} grades={','.join(g or '?' for g in r.get('grades', []))} "
              f"tokens={r.get('tokens', 0)} step_s={r.get('step_seconds', 0)} "
              f"fails={r.get('failed_attempts', 0)}"
              + (f" last_error={r['last_error']}" if r.get("last_error") else ""))
    n_done = sum(1 for r in rows if r["state"] == "complete")
    print(f"complete units: {n_done}/{len(rows)}")


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", default=sc.PRIMARY_CONFIG,
                    help=f"comma list of {list(sc.CONFIGS)} (default {sc.PRIMARY_CONFIG})")
    ap.add_argument("--items", default="", help="comma list of flawed item ids (default: all 12)")
    ap.add_argument("--data-dir", default=str(sc.DEFAULT_DATA_DIR))
    ap.add_argument("--provider", default="codex", choices=("codex", "mock"))
    ap.add_argument("--horizon", type=int, default=sc.HORIZON)
    ap.add_argument("--max-attempts-per-step", type=int, default=DEFAULT_MAX_ATTEMPTS_PER_STEP)
    ap.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF)
    ap.add_argument("--adapter-timeout", type=int, default=DEFAULT_ADAPTER_TIMEOUT)
    ap.add_argument("--workspace-root", default=None,
                    help="parent of the per-call empty CLI working dirs (default $TMPDIR/miw_stop_ws)")
    ap.add_argument("--status", action="store_true", help="print unit status and exit")
    args = ap.parse_args(argv)

    configs = sc.parse_csv(args.configs)
    unknown = [c for c in configs if c not in sc.CONFIGS]
    if unknown:
        ap.error(f"unknown config(s) {unknown}; known: {list(sc.CONFIGS)}")
    all_items = sc.flawed_items()
    items = sc.parse_csv(args.items) or list(all_items)
    bad = [i for i in items if i not in all_items]
    if bad:
        ap.error(f"unknown or non-flawed item(s) {bad}")
    data_dir = Path(args.data_dir)
    if args.status:
        print_status(data_dir, configs, items)
        return 0
    sc.bench_common().verify_manifest()
    runner = LoopRunner(data_dir, provider=args.provider, horizon=args.horizon,
                        max_attempts_per_step=args.max_attempts_per_step,
                        backoff=args.backoff, adapter_timeout=args.adapter_timeout,
                        workspace_root=Path(args.workspace_root) if args.workspace_root else None,
                        items=all_items)
    runner.log(f"run_loops start configs={configs} items={len(items)} provider={args.provider} "
               f"horizon={args.horizon} pid={os.getpid()}")
    try:
        outcome = asyncio.run(runner.run(configs, items))
    except FatalRunError as exc:
        runner.log(f"FATAL {exc}")
        return 2
    summary = {k: len(v) for k, v in outcome.items()}
    runner.log(f"run_loops end {json.dumps(summary)}")
    incomplete = [u for k, v in outcome.items() if k not in ("complete", "cached") for u in v]
    if incomplete:
        runner.log(f"incomplete units: {incomplete}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
