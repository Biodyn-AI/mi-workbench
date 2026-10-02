"""Judge the benchmark reviews against the planted-flaw ground truth (X1).

For every (model, item, repeat) whose eight reviewer calls all succeeded, one
judge call receives the analysis, the item's planted-flaw list (id, passage,
description, detection criterion; empty for clean items) and the shuffled,
numbered union of all parsed critiques with lens/call identity stripped. The
shuffle is seeded by a hash of ``model|item|repeat``. The judge labels every
critique with a flaw id or ``none`` (+ substantive|generic|incorrect); see
``prompts/judge.md``. Output is validated (every index labelled exactly once,
valid flaw ids, valid none types) and re-asked once when invalid. Cached at

    <data-dir>/judgments/<judge_model>/<model>/<item>/r<k>.json

Usage:
    python experiments/benchmark/judge.py --judge-models gpt-5.6-sol --effort high
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from backend.models import AdapterRunResult  # noqa: E402
from backend.orchestrator.feedback import extract_json_payloads  # noqa: E402

JUDGE_PROMPT_PATH = common.BENCH_DIR / "prompts" / "judge.md"
NONE_TYPES = ("substantive", "generic", "incorrect")
DEFAULT_TIMEOUT = 1800.0
DEFAULT_MAX_ATTEMPTS = 3


def load_judge_system_prompt(path: Path = JUDGE_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").strip() + "\n"


def collect_critiques(records: dict[str, dict], calls=common.ALL_CALLS) -> list[dict]:
    """Union of parsed critiques in canonical call order (before shuffling)."""
    out = []
    for tag in calls:
        rec = records.get(tag) or {}
        for j, c in enumerate(rec.get("critiques", []) or []):
            out.append({
                "tag": tag,
                "critique_index": j,
                "severity": c.get("severity"),
                "category": c.get("category"),
                "description": c.get("description") or "",
                "required_fix": c.get("required_fix") or "",
            })
    return out


def shuffle_critiques(critiques: list[dict], model: str, item: str, repeat: int) -> tuple[list[dict], int]:
    seed = common.stable_seed(common.model_dirname(model), item, repeat)
    order = list(range(len(critiques)))
    random.Random(seed).shuffle(order)
    shuffled = []
    for new_index, old in enumerate(order):
        entry = dict(critiques[old])
        entry["index"] = new_index
        shuffled.append(entry)
    return shuffled, seed


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def build_judge_user_prompt(item: common.Item, shuffled: list[dict]) -> str:
    """Analysis + planted flaws + numbered critiques (no lens/call/severity)."""
    n = len(shuffled)
    lines = ["ANALYSIS", "<<<", item.artifact_text.strip(), ">>>", ""]
    if item.flaws:
        flaws = [
            {
                "flaw_id": f.flaw_id,
                "passage": f.quote,
                "description": f.description,
                "detection_criterion": f.detection_criterion,
            }
            for f in item.flaws
        ]
        lines.append(f"PLANTED FLAWS ({len(flaws)}; valid flaw ids: "
                     f"{', '.join(f.flaw_id for f in item.flaws)})")
        lines.append(json.dumps(flaws, indent=2, ensure_ascii=False))
    else:
        lines.append("PLANTED FLAWS (0): none. This is a control analysis; every label is \"none\".")
    lines += ["", f"CRITIQUES (N = {n})"]
    for c in shuffled:
        lines.append(f"[{c['index']}] {_one_line(c['description'])}")
        if c.get("required_fix"):
            lines.append(f"    Proposed fix: {_one_line(c['required_fix'])}")
    lines += ["", f"Return the JSON block with exactly N = {n} labels (indices 0 to {n - 1})."]
    return "\n".join(lines) + "\n"


def validate_labels(payload: Any, n: int, valid_ids: list[str]) -> tuple[Optional[list[dict]], list[str]]:
    """Check coverage and values. Returns (labels sorted by index, problems)."""
    problems: list[str] = []
    entries = payload.get("labels") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return None, ["no 'labels' list"]
    ids_lower = {fid.lower(): fid for fid in valid_ids}
    by_index: dict[int, dict] = {}
    for e in entries:
        if not isinstance(e, dict):
            problems.append(f"non-object entry {e!r}")
            continue
        try:
            idx = int(str(e.get("index")).strip().strip("[]#"))
        except (TypeError, ValueError):
            problems.append(f"bad index {e.get('index')!r}")
            continue
        if idx < 0 or idx >= n:
            problems.append(f"index {idx} out of range 0..{n - 1}")
            continue
        if idx in by_index:
            problems.append(f"index {idx} labelled more than once")
            continue
        raw_fid = e.get("flaw_id")
        fid_s = "none" if raw_fid is None else str(raw_fid).strip()
        if fid_s.lower() in ("none", "null", ""):
            fid = "none"
        elif fid_s.lower() in ids_lower:
            fid = ids_lower[fid_s.lower()]
        else:
            problems.append(f"index {idx}: invalid flaw_id {raw_fid!r}")
            continue
        nt = e.get("none_type")
        nt_s = None if nt is None else str(nt).strip().lower()
        if fid == "none":
            if nt_s not in NONE_TYPES:
                problems.append(f"index {idx}: none_type must be one of {NONE_TYPES}, got {nt!r}")
                continue
        else:
            nt_s = None  # ignored for matches
        by_index[idx] = {"index": idx, "flaw_id": fid, "none_type": nt_s,
                         "note": str(e.get("note") or "")[:400]}
    missing = [i for i in range(n) if i not in by_index]
    if missing:
        problems.append(f"missing indices {missing[:30]}{'...' if len(missing) > 30 else ''}")
    if problems:
        return None, problems
    return [by_index[i] for i in range(n)], []


def parse_judge_output(text: str, n: int, valid_ids: list[str]) -> tuple[Optional[list[dict]], list[str]]:
    payloads = extract_json_payloads(text or "")
    candidates = [p for p, _ in payloads if (isinstance(p, dict) and "labels" in p)]
    if not candidates:
        candidates = [p for p, _ in payloads if isinstance(p, list)]
    if not candidates:
        return None, ["no JSON block with a 'labels' list found"]
    return validate_labels(candidates[-1], n, valid_ids)


def _failure(error: str) -> AdapterRunResult:
    return AdapterRunResult(success=False, exit_code=-1, error=error)


class Judge:
    def __init__(
        self,
        *,
        judge_model: str,
        effort: str,
        data_dir: Path,
        items: Optional[dict[str, common.Item]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff: float = 15.0,
        adapter_factory: Optional[Callable[[str, str], Any]] = None,
        codex_home: Optional[str] = None,
        workspace_root: Optional[Path] = None,
        system_prompt: Optional[str] = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        stdout: bool = True,
        hard_timeout_grace: Optional[float] = None,
        codex_extra: Optional[list] = None,
    ):
        self.judge_model = judge_model
        self.codex_extra = list(common.CODEX_NO_WEB_ARGS) if codex_extra is None else list(codex_extra)
        self.effort = effort
        self.data_dir = Path(data_dir)
        self.items = items if items is not None else common.load_items()
        self.timeout = float(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.backoff = float(backoff)
        self.codex_home = codex_home
        factory = adapter_factory or (
            lambda model, eff: common.make_adapter(model, eff, codex_home=codex_home,
                                                   codex_extra=self.codex_extra))
        self.adapter = factory(judge_model, effort)
        self.workspace_root = workspace_root
        self.system_prompt = system_prompt or load_judge_system_prompt()
        self.sleep = sleep
        self.stdout = stdout
        self.hard_timeout_grace = (max(30.0, 0.05 * self.timeout) if hard_timeout_grace is None
                                   else float(hard_timeout_grace))
        self.log_path = self.data_dir / "logs" / "judge.log"
        self.stats = {"skipped_cached": 0, "judged": 0, "invalid": 0, "failed": 0,
                      "no_critiques": 0, "incomplete_units": 0}

    def log(self, msg: str) -> None:
        line = f"{common.utc_now()} {self.judge_model} {msg}"
        common.append_log(self.log_path, line)
        if self.stdout:
            print(line, flush=True)

    async def _call(self, user_prompt: str) -> tuple[AdapterRunResult, dict]:
        ws = common.make_workspace(self.workspace_root)
        request = common.make_request(
            self.system_prompt, user_prompt, model=self.judge_model, effort=self.effort,
            timeout_seconds=self.timeout, workspace=str(ws), role="benchmark_judge",
        )
        started = common.utc_now()
        t0 = time.monotonic()
        captured = common.begin_capture()
        hard = self.timeout + self.hard_timeout_grace
        try:
            result = await asyncio.wait_for(self.adapter.run(request), timeout=hard)
        except asyncio.TimeoutError:
            result = _failure(f"[timeout] hard timeout after {hard:.0f}s")
        except asyncio.CancelledError:
            common.remove_workspace(ws)
            raise
        except Exception as exc:  # noqa: BLE001
            result = _failure(f"[exception] {type(exc).__name__}: {exc}")
        finally:
            common.remove_workspace(ws)
        info = {
            "started_utc": started,
            "finished_utc": common.utc_now(),
            "wall_seconds": round(time.monotonic() - t0, 3),
            "success": bool(result.success),
            "error": result.error,
            "tokens": common.usage_tokens(result),
            "tool_events": common.count_tool_events(result),
            "tool_items": common.extract_tool_items(captured),
            "raw_output": result.output,
            "cli_version": result.cli_version,
        }
        return result, info

    async def judge_unit(self, model: str, item_id: str, repeat: int) -> Optional[dict]:
        records = common.unit_records(self.data_dir, model, item_id, repeat)
        if not common.unit_complete(records):
            self.stats["incomplete_units"] += 1
            return None
        item = self.items[item_id]
        critiques = collect_critiques(records)
        shuffled, seed = shuffle_critiques(critiques, model, item_id, repeat)
        user_prompt = build_judge_user_prompt(item, shuffled)
        provider, name = common.resolve_provider(self.judge_model)
        signature = {
            "judge_provider": provider,
            "judge_model": name,
            "reasoning_effort": self.effort,
            "judge_system_prompt_sha256": common.sha256_text(self.system_prompt),
            "user_prompt_sha256": common.sha256_text(user_prompt),
            "cli_extra_args": self.codex_extra if provider == "codex" else [],
            # Codex injects $CODEX_HOME/AGENTS.md into every prompt.
            **common.codex_env_signature(provider, self.codex_home),
        }
        path = common.judgment_path(self.data_dir, self.judge_model, model, item_id, repeat)
        previous = None
        if path.exists():
            try:
                previous = common.read_json(path)
            except (OSError, json.JSONDecodeError):
                previous = None
        if previous is not None:
            same = common.signature_matches(previous.get("request", {}), signature)
            if same and previous.get("valid"):
                self.stats["skipped_cached"] += 1
                return previous
            if not same:
                self.log(f"STALE {model}/{item_id}/r{repeat}: inputs changed, re-judging")

        n = len(shuffled)
        valid_ids = item.flaw_ids
        record: dict[str, Any] = {
            "schema": common.JUDGE_SCHEMA,
            "key": {"judge_model": self.judge_model, "model": model, "item": item_id, "repeat": repeat},
            "request": {
                **signature,
                "judge_prompt_path": "experiments/benchmark/prompts/judge.md",
                "shuffle_seed": seed,
                "n_critiques": n,
                "flaw_ids": valid_ids,
                "timeout_seconds": self.timeout,
                "review_raw_output_sha256": {
                    tag: common.sha256_text((rec or {}).get("raw_output") or "")
                    for tag, rec in records.items()
                },
                "codex_env": (common.codex_env_fingerprint(self.codex_home)
                              if provider == "codex" else None),
            },
            "critiques": shuffled,
            "labels": [],
            "valid": False,
            "problems": [],
            "judge_called": False,
            "attempts": [],
        }
        if n == 0:
            record.update(valid=True, judge_called=False)
            common.atomic_write_json(path, record)
            self.stats["no_critiques"] += 1
            self.log(f"NO_CRITIQUES {model}/{item_id}/r{repeat}")
            return record

        prompt = user_prompt
        reasked = False
        labels = None
        problems: list[str] = []
        failures = 0
        while True:
            result, info = await self._call(prompt)
            info["reask"] = reasked
            record["attempts"].append(info)
            record["judge_called"] = True
            if not result.success:
                failures += 1
                self.log(f"CALL_FAILED {model}/{item_id}/r{repeat} attempt={failures} "
                         f"error={' '.join(str(result.error).split())[:300]}")
                if failures >= self.max_attempts or str(result.error or "").lower().startswith("[auth]"):
                    problems = [f"judge call failed: {result.error}"]
                    break
                await self.sleep(min(240.0, self.backoff * 2 ** (failures - 1)))
                continue
            labels, problems = parse_judge_output(result.output, n, valid_ids)
            info["problems"] = problems
            if labels is not None or reasked:
                break
            reasked = True
            self.log(f"REASK {model}/{item_id}/r{repeat}: {'; '.join(problems)[:300]}")
            prompt = (
                user_prompt
                + "\nYOUR PREVIOUS ANSWER WAS INVALID: " + "; ".join(problems)
                + f"\nReturn the complete JSON block again with exactly N = {n} labels, "
                  f"one per index 0 to {n - 1}.\n"
            )
        record["valid"] = labels is not None
        record["labels"] = labels or []
        record["problems"] = problems
        record["reasked"] = reasked
        tok = {"input": 0, "output": 0, "cached_input": 0, "reasoning_output": 0}
        for a in record["attempts"]:
            for k in tok:
                tok[k] += int(a["tokens"].get(k, 0) or 0)
        record["tokens"] = tok
        record["wall_seconds"] = round(sum(a["wall_seconds"] for a in record["attempts"]), 3)
        common.atomic_write_json(path, record)
        if record["valid"]:
            self.stats["judged"] += 1
            matched = sum(1 for lab in labels if lab["flaw_id"] != "none")
            self.log(f"OK {model}/{item_id}/r{repeat} n={n} matched={matched} reasked={reasked} "
                     f"wall={record['wall_seconds']:.0f}s in={tok['input']} out={tok['output']}")
        elif any(not a["success"] for a in record["attempts"][-1:]):
            self.stats["failed"] += 1
        else:
            self.stats["invalid"] += 1
            self.log(f"INVALID {model}/{item_id}/r{repeat}: {'; '.join(problems)[:300]}")
        return record

    async def run(self, units: list[tuple[str, str, int]], concurrency: int = 2) -> dict:
        queue: asyncio.Queue = asyncio.Queue()
        for u in units:
            queue.put_nowait(u)

        async def worker():
            while True:
                try:
                    u = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await self.judge_unit(*u)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.stats["failed"] += 1
                    self.log(f"HARNESS_ERROR {u}: {type(exc).__name__}: {exc}")

        await asyncio.gather(*(asyncio.create_task(worker()) for _ in range(max(1, concurrency))))
        return dict(self.stats)


def discover_units(data_dir: Path, models: Optional[list[str]] = None,
                   items: Optional[list[str]] = None) -> list[tuple[str, str, int]]:
    """(model, item, repeat) units present in the review cache."""
    seen: dict[tuple[str, str, int], None] = {}
    for rec in common.discover_reviews(data_dir):
        k = rec.get("key", {})
        unit = (k.get("model"), k.get("item"), int(k.get("repeat", 0)))
        if None in unit:
            continue
        if models and unit[0] not in models:
            continue
        if items and unit[1] not in items:
            continue
        seen[unit] = None
    return sorted(seen, key=lambda u: (u[2], u[1], u[0]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judge-models", required=True, help="comma-separated judge models")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--models", default="", help="restrict to these reviewer models")
    ap.add_argument("--items", default="", help="restrict to these items")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--data-dir", default=str(common.DEFAULT_DATA_DIR))
    ap.add_argument("--codex-home", default=None)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--codex-web-search", choices=("disabled", "default"), default="disabled")
    ap.add_argument("--codex-tools-off", choices=common.CODEX_TOOLS_OFF_MODES, default="web_only",
                    help="Codex tools-off condition for the judge (see run_reviews.py)")
    args = ap.parse_args(argv)

    common.verify_manifest()
    common.install_cli_capture()
    if args.codex_home:
        os.environ["CODEX_HOME"] = args.codex_home
    data_dir = Path(args.data_dir)
    items = common.load_items()
    units = discover_units(data_dir, common.parse_csv_arg(args.models) or None,
                           common.parse_csv_arg(args.items) or None)
    print(f"{len(units)} units found in the review cache", flush=True)
    all_stats = {}
    for jm in common.parse_csv_arg(args.judge_models):
        judge = Judge(judge_model=jm, effort=args.effort, data_dir=data_dir, items=items,
                      timeout=args.timeout, max_attempts=args.max_attempts,
                      codex_home=args.codex_home,
                      workspace_root=Path(args.workspace_root) if args.workspace_root else None,
                      codex_extra=common.codex_extra_args(args.codex_web_search,
                                                          args.codex_tools_off))
        t0 = time.monotonic()
        stats = asyncio.run(judge.run(units, concurrency=args.concurrency))
        stats["wall_seconds"] = round(time.monotonic() - t0, 1)
        all_stats[jm] = stats
        print(json.dumps({jm: stats}), flush=True)
    bad = sum(s["invalid"] + s["failed"] for s in all_stats.values())
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
