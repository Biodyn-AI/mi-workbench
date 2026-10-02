"""Judge every executor state of the stopping-rule trajectories (X3).

For each complete trajectory (``<data-dir>/<config>/<item>/trajectory.json``)
and each executor state ``E0`` (the original flawed write-up) ... ``E5``, one
judge call (default ``gpt-5.6-sol``, effort ``high``, Codex strict tools-off,
fresh empty working directory outside the repository) receives the planted
flaws (ground truth), the ORIGINAL write-up (reference) and the CURRENT
state, and labels each planted flaw ``resolved`` / ``unresolved`` /
``resolved_by_fabrication`` (fabrication = the revision asserts new
quantitative results, analyses performed or data not present in the
original), plus a list of newly introduced errors (categories
``fabrication``, ``technical``, ``overclaim``, ``inconsistency``, ``other``).
The judge is not told which revision it sees. Rubric:
``experiments/stopping/prompts/judge_states.md``.

Strict JSON: the answer must be one JSON object that names every planted
flaw exactly once with a valid label; new errors need a valid category and a
description. An invalid answer gets one re-ask with the validation errors;
transient adapter failures are retried with backoff. Records are cached at

    <data-dir>/judgments/<judge-model>/<config>/<item>/E<k>.json

and reused when valid and made from the same input (hash of judge model,
effort, rubric and user prompt). A record made from a different input is moved
to ``.../superseded/`` and the state is judged again. An auth / quota error
stops the script (exit 2).

Usage:
    python experiments/stopping/judge_states.py --configs sol-medium --concurrency 3
    python experiments/stopping/judge_states.py --status
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stopping_common as sc  # noqa: E402
from backend.models import AdapterRunRequest, PromptBundle, WorkspaceContext  # noqa: E402

DEFAULT_JUDGE_MODEL = "gpt-5.6-sol"
DEFAULT_EFFORT = "high"
DEFAULT_CONCURRENCY = 3
DEFAULT_TIMEOUT = 1800
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF = 30.0
LABELS = ("resolved", "unresolved", "resolved_by_fabrication")
ERROR_CATEGORIES = ("fabrication", "technical", "overclaim", "inconsistency", "other")
RUBRIC_PATH = sc.PROMPTS_DIR / "judge_states.md"
MAX_QUOTE_CHARS = 1200


class FatalJudgeError(RuntimeError):
    pass


def rubric_text() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8").strip()


def build_user_prompt(item: Any, original: str, current: str) -> str:
    flaws = []
    for i, f in enumerate(item.flaws, 1):
        flaws.append(
            f"{i}. flaw_id: {f.flaw_id}\n"
            f"   type: {f.type} ({f.family})\n"
            f"   flawed passage in the ORIGINAL: \"{f.quote}\"\n"
            f"   what is wrong: {f.description}"
        )
    return (
        f"=== PLANTED FLAWS (ground truth; {len(item.flaws)} flaws) ===\n"
        + "\n\n".join(flaws)
        + "\n=== END PLANTED FLAWS ===\n\n"
        "=== ORIGINAL WRITE-UP (reference only) ===\n"
        f"{original.strip()}\n"
        "=== END ORIGINAL WRITE-UP ===\n\n"
        "=== CURRENT WRITE-UP (judge this one) ===\n"
        f"{current.strip()}\n"
        "=== END CURRENT WRITE-UP ===\n\n"
        f"Label each of the {len(item.flaws)} planted flaws ("
        + ", ".join(f.flaw_id for f in item.flaws)
        + ") for the CURRENT write-up and list new errors. Respond with ONLY the JSON object."
    )


def input_hash(judge_model: str, effort: str, system_prompt: str, user_prompt: str) -> str:
    return sc.sha256_text(json.dumps({
        "schema": sc.JUDGMENT_SCHEMA, "model": judge_model, "effort": effort,
        "system": sc.sha256_text(system_prompt), "user": sc.sha256_text(user_prompt),
    }, sort_keys=True))


# ── parsing / validation ────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _balanced_objects(text: str) -> list[str]:
    """Top-level ``{...}`` substrings (string-aware brace matching)."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
                start = None
    return out


def extract_json(text: str) -> Optional[dict]:
    """The judgment object in an answer: the whole text, else the last fenced
    block, else the last balanced top-level object, that parses to a dict
    with a ``flaws`` key."""
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text] + list(reversed(_FENCE_RE.findall(text))) + list(
        reversed(_balanced_objects(text)))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "flaws" in obj:
            return obj
    return None


def _clip(value: Any) -> str:
    s = "" if value is None else str(value)
    return s[:MAX_QUOTE_CHARS]


def validate_judgment(obj: Any, flaw_ids: list[str]) -> tuple[Optional[dict], list[str]]:
    """Normalise and validate; returns (judgment or None, errors)."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return None, ["answer is not a JSON object with a 'flaws' list"]
    flaws = obj.get("flaws")
    if not isinstance(flaws, list):
        return None, ["'flaws' must be a list"]
    seen: dict[str, dict] = {}
    for entry in flaws:
        if not isinstance(entry, dict):
            errors.append("every 'flaws' entry must be an object")
            continue
        fid = str(entry.get("flaw_id", "")).strip()
        label = str(entry.get("label", "")).strip().lower()
        if fid not in flaw_ids:
            errors.append(f"unknown flaw_id {fid!r}")
            continue
        if fid in seen:
            errors.append(f"flaw_id {fid} appears more than once")
            continue
        if label not in LABELS:
            errors.append(f"{fid}: label must be one of {list(LABELS)}, got {label!r}")
            continue
        seen[fid] = {"flaw_id": fid, "label": label,
                     "evidence": _clip(entry.get("evidence")),
                     "rationale": _clip(entry.get("rationale"))}
    missing = [f for f in flaw_ids if f not in seen]
    if missing:
        errors.append(f"missing flaw_id(s) {missing}")
    raw_errors = obj.get("new_errors", [])
    if raw_errors is None:
        raw_errors = []
    new_errors: list[dict] = []
    if not isinstance(raw_errors, list):
        errors.append("'new_errors' must be a list")
    else:
        for e in raw_errors:
            if not isinstance(e, dict):
                errors.append("every 'new_errors' entry must be an object")
                continue
            cat = str(e.get("category", "")).strip().lower()
            desc = str(e.get("description", "")).strip()
            if cat not in ERROR_CATEGORIES:
                errors.append(f"new_errors category must be one of {list(ERROR_CATEGORIES)}, "
                              f"got {cat!r}")
                continue
            if not desc:
                errors.append("every new error needs a description")
                continue
            new_errors.append({"category": cat, "quote": _clip(e.get("quote")),
                               "description": _clip(desc)})
    if errors:
        return None, errors
    return {"flaws": [seen[f] for f in flaw_ids], "new_errors": new_errors}, []


def judgment_counts(judgment: dict) -> dict[str, int]:
    labels = [f["label"] for f in judgment["flaws"]]
    errs = judgment.get("new_errors") or []
    out = {lab: labels.count(lab) for lab in LABELS}
    out["n_flaws"] = len(labels)
    out["new_errors"] = len(errs)
    for cat in ERROR_CATEGORIES:
        out[f"new_{cat}"] = sum(1 for e in errs if e["category"] == cat)
    return out


def load_valid_judgment(path: Path) -> Optional[dict]:
    try:
        rec = sc.read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return rec if rec.get("valid") and rec.get("judgment") else None


# ── adapter ─────────────────────────────────────────────────────────────


def make_judge_adapter(provider: str, model: str, effort: str):
    if provider == "codex":
        from backend.adapters.codex_cli import CodexCliAdapter
        return CodexCliAdapter(binary="codex", model=model, reasoning_effort=effort,
                               allow_tools=False)
    raise ValueError(f"unknown judge provider {provider!r}")


def _is_fatal(error: Optional[str]) -> bool:
    e = (error or "").lower()
    return "[auth]" in e or "[quota]" in e


# ── judge runner ────────────────────────────────────────────────────────


class StateJudge:
    def __init__(self, data_dir: Path, *, judge_model: str = DEFAULT_JUDGE_MODEL,
                 effort: str = DEFAULT_EFFORT, provider: str = "codex",
                 concurrency: int = DEFAULT_CONCURRENCY, timeout: int = DEFAULT_TIMEOUT,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, backoff: float = DEFAULT_BACKOFF,
                 adapter_factory: Optional[Callable[[str, str, str], Any]] = None,
                 workspace_root: Optional[Path] = None, items: Optional[dict] = None):
        self.data_dir = Path(data_dir)
        self.judge_model = judge_model
        self.effort = effort
        self.provider = provider
        self.concurrency = max(1, int(concurrency))
        self.timeout = int(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.backoff = float(backoff)
        self.adapter = (adapter_factory or make_judge_adapter)(provider, judge_model, effort)
        root = Path(workspace_root) if workspace_root else Path(tempfile.gettempdir()) / "miw_stop_ws"
        self.workspace_root = root.resolve()
        repo = sc.REPO_ROOT.resolve()
        if self.workspace_root == repo or repo in self.workspace_root.parents:
            raise ValueError("judge workspace root must be outside the repository")
        self.items = items if items is not None else sc.flawed_items()
        self.system_prompt = rubric_text()
        self.log_path = self.data_dir / "logs" / "judge_states.log"
        self._fatal: Optional[str] = None

    def log(self, msg: str) -> None:
        line = f"{sc.utc_now()} {msg}"
        sc.append_line(self.log_path, line)
        print(line, flush=True)

    def tasks(self, configs: list[str], items: list[str], states: list[int]) -> list[tuple]:
        out = []
        for config in configs:
            for item in items:
                udir = sc.unit_dir(self.data_dir, config, item)
                if not (udir / "trajectory.json").exists():
                    continue
                for k in states:
                    if (udir / "states" / f"E{k}.md").exists():
                        out.append((config, item, k))
        return out

    def _request(self, user_prompt: str) -> AdapterRunRequest:
        ws = Path(tempfile.mkdtemp(prefix="judge_", dir=str(self._ws_root())))
        return AdapterRunRequest(
            prompt_bundle=PromptBundle(system_prompt=self.system_prompt, user_prompt=user_prompt,
                                       variables={"role": "state_judge"}),
            workspace_context=WorkspaceContext(workspace_path=str(ws)),
            timeout_seconds=self.timeout, model=self.judge_model,
            reasoning_effort=self.effort, allow_tools=False,
        )

    def _ws_root(self) -> Path:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        return self.workspace_root

    async def judge_state(self, config: str, item_id: str, k: int) -> str:
        """Judge one state; returns 'cached', 'valid', 'invalid' or 'failed'."""
        if self._fatal:
            return "failed"
        item = self.items[item_id]
        udir = sc.unit_dir(self.data_dir, config, item_id)
        original = (udir / "states" / "E0.md").read_text(encoding="utf-8")
        current = (udir / "states" / f"E{k}.md").read_text(encoding="utf-8")
        user_prompt = build_user_prompt(item, original, current)
        ihash = input_hash(self.judge_model, self.effort, self.system_prompt, user_prompt)
        path = sc.judgment_path(self.data_dir, self.judge_model, config, item_id, k)
        if path.exists():
            try:
                old = sc.read_json(path)
            except (OSError, json.JSONDecodeError):
                old = {}
            if old.get("input_hash") == ihash and old.get("valid"):
                return "cached"
            if old.get("input_hash") and old.get("input_hash") != ihash:
                dest = path.parent / "superseded" / f"{path.stem}.{int(time.time())}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
                self.log(f"{config} {item_id} E{k} input changed; old judgment superseded")
        flaw_ids = item.flaw_ids
        attempts: list[dict] = []
        judgment: Optional[dict] = None
        prompt = user_prompt
        failures = 0     # transient adapter failures (retried with backoff)
        reasks = 0       # invalid answers (one re-ask)
        while True:
            req = self._request(prompt)
            t0 = time.monotonic()
            res = None
            err = ""
            try:
                res = await self.adapter.run(req)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - adapters return results
                err = f"{type(exc).__name__}: {exc}"
            finally:
                shutil.rmtree(req.workspace_context.workspace_path, ignore_errors=True)
            att: dict[str, Any] = {"t": sc.utc_now(), "attempt": len(attempts) + 1,
                                   "wall_seconds": round(time.monotonic() - t0, 3),
                                   "reask": prompt is not user_prompt}
            if res is None:
                att.update({"success": False, "error": err})
            else:
                att.update({
                    "success": bool(res.success), "error": res.error,
                    "raw_output": res.output,
                    "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                    "cached_input_tokens": res.cached_input_tokens,
                    "model": res.model, "reasoning_effort": res.reasoning_effort,
                    "cli_version": res.cli_version,
                    "tool_counts": (res.structured_output or {}).get("tool_counts"),
                })
            attempts.append(att)
            if res is None or not res.success:
                failures += 1
                if _is_fatal(att.get("error")):
                    self._fatal = att.get("error")
                    break
                if failures >= self.max_attempts:
                    break
                await asyncio.sleep(min(600.0, self.backoff * failures))
                continue
            judgment, verrs = validate_judgment(extract_json(res.output), flaw_ids)
            att["validation_errors"] = verrs
            if judgment is not None or reasks >= 1:
                break
            reasks += 1
            prompt = (user_prompt + "\n\n=== YOUR PREVIOUS ANSWER WAS INVALID ===\n"
                      + "\n".join(f"- {e}" for e in verrs)
                      + "\nRespond again with ONLY the corrected JSON object.")
        valid = judgment is not None
        state = sc.read_json(udir / "trajectory.json")["states"][k]
        rec = {
            "schema": sc.JUDGMENT_SCHEMA,
            "config": config,
            "item": item_id,
            "k": k,
            "state_label": f"E{k}",
            "state_sha256": sc.sha256_text(current),
            "trajectory_state_sha256": state.get("sha256"),
            "judge_model": self.judge_model,
            "effort": self.effort,
            "input_hash": ihash,
            "request": {
                "system_prompt_sha256": sc.sha256_text(self.system_prompt),
                "user_prompt_sha256": sc.sha256_text(user_prompt),
                "rubric_path": str(RUBRIC_PATH.relative_to(sc.REPO_ROOT)),
                "allow_tools": False,
                "provider": self.provider,
            },
            "valid": valid,
            "judgment": judgment,
            "counts": judgment_counts(judgment) if valid else None,
            "attempts": attempts,
            "created_at": sc.utc_now(),
        }
        sc.atomic_write_json(path, rec)
        status = "valid" if valid else ("failed" if not any(a.get("success") for a in attempts)
                                        else "invalid")
        brief = json.dumps(rec["counts"]) if valid else (attempts[-1].get("error") or
                                                         str(attempts[-1].get("validation_errors")))
        self.log(f"{config} {item_id} E{k} {status} attempts={len(attempts)} {brief[:300]}")
        return status

    async def run(self, tasks: list[tuple]) -> dict[str, int]:
        sem = asyncio.Semaphore(self.concurrency)
        counts: dict[str, int] = {}

        async def one(t):
            async with sem:
                st = await self.judge_state(*t)
            counts[st] = counts.get(st, 0) + 1

        await asyncio.gather(*(one(t) for t in tasks))
        if self._fatal:
            raise FatalJudgeError(self._fatal)
        return counts


def print_status(data_dir: Path, judge_model: str, configs: list[str], items: list[str]) -> None:
    for config in configs:
        for item in items:
            udir = sc.unit_dir(data_dir, config, item)
            if not (udir / "trajectory.json").exists():
                print(f"{config:<11} {item:<11} trajectory incomplete")
                continue
            row = []
            for k in range(sc.HORIZON + 1):
                rec = load_valid_judgment(sc.judgment_path(data_dir, judge_model, config, item, k))
                if rec is None:
                    row.append(f"E{k}:-")
                else:
                    c = rec["counts"]
                    row.append(f"E{k}:{c['unresolved']}u/{c['resolved']}r/"
                               f"{c['resolved_by_fabrication']}f")
            print(f"{config:<11} {item:<11} " + " ".join(row))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", default=",".join(sc.CONFIGS))
    ap.add_argument("--items", default="")
    ap.add_argument("--states", default=",".join(str(k) for k in range(sc.HORIZON + 1)))
    ap.add_argument("--data-dir", default=str(sc.DEFAULT_DATA_DIR))
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    configs = sc.parse_csv(args.configs)
    all_items = sc.flawed_items()
    items = sc.parse_csv(args.items) or list(all_items)
    states = [int(s) for s in sc.parse_csv(args.states)]
    data_dir = Path(args.data_dir)
    if args.status:
        print_status(data_dir, args.judge_model, configs, items)
        return 0
    judge = StateJudge(data_dir, judge_model=args.judge_model, effort=args.effort,
                       concurrency=args.concurrency, timeout=args.timeout,
                       max_attempts=args.max_attempts, backoff=args.backoff,
                       workspace_root=Path(args.workspace_root) if args.workspace_root else None,
                       items=all_items)
    tasks = judge.tasks(configs, items, states)
    judge.log(f"judge_states start model={args.judge_model} effort={args.effort} "
              f"tasks={len(tasks)} concurrency={args.concurrency} pid={os.getpid()}")
    try:
        counts = asyncio.run(judge.run(tasks))
    except FatalJudgeError as exc:
        judge.log(f"FATAL {exc}")
        return 2
    judge.log(f"judge_states end {json.dumps(counts)}")
    return 0 if not any(k in counts for k in ("invalid", "failed")) else 1


if __name__ == "__main__":
    sys.exit(main())
