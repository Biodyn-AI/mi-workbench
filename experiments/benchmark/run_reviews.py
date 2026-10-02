"""Run the reviewer calls of the planted-flaw benchmark (X1), with caching.

For every model x item x repeat, eight stateless calls are made on the same
input (ANALYSIS_PLAN.md): rig1-3 (``reviewer/mi_reviewer``), cmb1-3
(``reviewer/mi_reviewer_combined``), adv (``adversarial_reviewer``) and bio
(``bio_plausibility_checker``). Each call's full record is cached at

    <data-dir>/reviews/<model>/<item>/r<k>/<tag>.json

Resumable: successful cached calls are skipped; failed ones are retried (up to
``--max-attempts`` new attempts per invocation, exponential backoff). A call
whose output has no usable critique format (``parse.method == "none"``, which
includes a JSON block that cannot count as a review) is a failed attempt and
is retried like any other failure; if it never parses the record is saved with
``success=False`` (raw output and parse kept for audit), so the unit is not
analysed or judged. An explicit, complete ``"critiques": []`` is a valid
clean review. A cached record is reused only if it was made with the same
model, effort, prompts, Codex tools-off arguments and Codex ``AGENTS.md``
(``$CODEX_HOME``); otherwise it is reported as CACHE_MISMATCH. Work is
scheduled unit by unit (repeat -> item -> model -> call) so a partial run
covers models and items evenly and completes whole (item, model, repeat) units
for judging. One log line per call goes to ``<data-dir>/logs/run_reviews.log``.

Usage (pilot):
    python experiments/benchmark/run_reviews.py --models gpt-5.6-sol,gpt-5.6-luna \
        --effort medium --repeats 1 --items A01-flawed,A05-clean
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from backend.models import AdapterRunResult  # noqa: E402

DEFAULT_TIMEOUT = 900.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF = 15.0
MAX_BACKOFF = 240.0
RAW_LOG_LIMIT = 20000


@dataclass(frozen=True)
class CallKey:
    model: str
    item: str
    repeat: int
    tag: str

    def label(self) -> str:
        return f"{self.model}/{self.item}/r{self.repeat}/{self.tag}"


def schedule(models: list[str], items: list[str], repeats: int, calls: list[str],
             seed: str = "miw-benchmark") -> list[CallKey]:
    """Balanced order: repeat -> item (seeded order per repeat) -> model -> call."""
    order: list[CallKey] = []
    for r in range(1, repeats + 1):
        item_order = list(items)
        random.Random(common.stable_seed(seed, "items", r)).shuffle(item_order)
        for item in item_order:
            for model in models:
                for tag in calls:
                    order.append(CallKey(model, item, r, tag))
    return order


def request_signature(model: str, effort: str, prompt: common.PromptInfo, user_prompt: str,
                      cli_extra_args: Optional[list] = None,
                      codex_home: Optional[str] = None) -> dict:
    """Fields a cached record must match to be reused (including the hashes
    of the Codex AGENTS.md files the CLI injects into every prompt)."""
    provider, name = common.resolve_provider(model)
    return {
        "provider": provider,
        "model": name,
        "reasoning_effort": effort,
        "system_prompt_sha256": prompt.sha256,
        "user_prompt_sha256": common.sha256_text(user_prompt),
        "cli_extra_args": list(cli_extra_args or []) if provider == "codex" else [],
        **common.codex_env_signature(provider, codex_home),
    }


def cache_matches(record: dict, signature: dict) -> bool:
    return common.signature_matches(record.get("request", {}), signature)


UNPARSED_ERROR = "[unparsed] no recognisable critique format in the reviewer output"


def _failure(error: str) -> AdapterRunResult:
    return AdapterRunResult(success=False, exit_code=-1, error=error)


def _is_auth_error(error: Optional[str]) -> bool:
    return bool(error) and error.lower().startswith("[auth]")


class Runner:
    def __init__(
        self,
        *,
        data_dir: Path,
        effort: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff: float = DEFAULT_BACKOFF,
        adapter_factory: Callable[[str, str], Any] = None,
        codex_home: Optional[str] = None,
        workspace_root: Optional[Path] = None,
        items: Optional[dict[str, common.Item]] = None,
        log_path: Optional[Path] = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        stdout: bool = True,
        hard_timeout_grace: Optional[float] = None,
        codex_extra: Optional[list] = None,
    ):
        self.data_dir = Path(data_dir)
        self.effort = effort
        self.codex_extra = list(common.CODEX_NO_WEB_ARGS) if codex_extra is None else list(codex_extra)
        self.timeout = float(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.backoff = float(backoff)
        self.codex_home = codex_home
        self.adapter_factory = adapter_factory or (
            lambda model, effort: common.make_adapter(model, effort, codex_home=codex_home,
                                                      codex_extra=self.codex_extra))
        self.workspace_root = workspace_root
        self.items = items if items is not None else common.load_items()
        self.log_path = Path(log_path) if log_path else self.data_dir / "logs" / "run_reviews.log"
        self.sleep = sleep
        self.stdout = stdout
        # Extra seconds before the asyncio guard fires, so the adapter's own
        # timeout (which kills the CLI process group) normally triggers first.
        self.hard_timeout_grace = (max(30.0, 0.05 * self.timeout) if hard_timeout_grace is None
                                   else float(hard_timeout_grace))
        self._adapters: dict[str, Any] = {}
        self._versions: dict[str, str] = {}
        self.stats = {"skipped_cached": 0, "succeeded": 0, "failed": 0, "cache_mismatch": 0}

    def adapter(self, model: str):
        if model not in self._adapters:
            self._adapters[model] = self.adapter_factory(model, self.effort)
        return self._adapters[model]

    async def cli_version(self, model: str) -> str:
        if model not in self._versions:
            ad = self.adapter(model)
            fn = getattr(ad, "cli_version", None)
            v = ""
            if fn is not None:
                try:
                    v = await fn()
                except Exception:  # noqa: BLE001
                    v = ""
            self._versions[model] = v or ""
        return self._versions[model]

    def log(self, key: CallKey, status: str, rec: Optional[dict] = None, extra: str = "") -> None:
        parts = [common.utc_now(), status, key.label()]
        if rec is not None:
            tok = rec.get("tokens", {})
            parse = rec.get("parse") or {}
            parts += [
                f"attempts={len(rec.get('attempts', []))}",
                f"wall={rec.get('wall_seconds', 0):.1f}s",
                f"in={tok.get('input', 0)}",
                f"cached={tok.get('cached_input', 0)}",
                f"out={tok.get('output', 0)}",
                f"parse={parse.get('method', '-')}",
                f"n={parse.get('n_critiques', '-')}",
            ]
            if rec.get("tool_events"):
                parts.append(f"tool_events={rec['tool_events']}")
            if rec.get("error"):
                parts.append("error=" + " ".join(str(rec["error"]).split())[:300])
        if extra:
            parts.append(extra)
        line = " ".join(parts)
        common.append_log(self.log_path, line)
        if self.stdout:
            print(line, flush=True)

    async def run_call(self, key: CallKey) -> Optional[dict]:
        role, prompt_ref = common.CALL_SPECS[key.tag]
        prompt = common.load_system_prompt(prompt_ref, role)
        item = self.items[key.item]
        user_prompt = common.build_user_prompt(item.artifact_text)
        signature = request_signature(key.model, self.effort, prompt, user_prompt, self.codex_extra,
                                      codex_home=self.codex_home)
        path = common.review_path(self.data_dir, key.model, key.item, key.repeat, key.tag)

        previous: Optional[dict] = None
        if path.exists():
            try:
                previous = common.read_json(path)
            except (OSError, json.JSONDecodeError):
                previous = None
        if previous is not None:
            if not cache_matches(previous, signature):
                self.stats["cache_mismatch"] += 1
                self.log(key, "CACHE_MISMATCH", None,
                         "cached record was made with a different model/effort/prompt; "
                         "not overwritten (move it away to re-run)")
                return previous
            if previous.get("success"):
                self.stats["skipped_cached"] += 1
                return previous

        adapter = self.adapter(key.model)
        cli_version = await self.cli_version(key.model)
        attempts: list[dict] = list(previous.get("attempts", [])) if previous else []
        hard_timeout = self.timeout + self.hard_timeout_grace
        result: AdapterRunResult = _failure("not run")
        call_started = common.utc_now()
        for k in range(self.max_attempts):
            ws = common.make_workspace(self.workspace_root)
            request = common.make_request(
                prompt.text, user_prompt, model=key.model, effort=self.effort,
                timeout_seconds=self.timeout, workspace=str(ws), role=role,
            )
            started = common.utc_now()
            t0 = time.monotonic()
            captured = common.begin_capture()
            try:
                result = await asyncio.wait_for(adapter.run(request), timeout=hard_timeout)
            except asyncio.TimeoutError:
                result = _failure(f"[timeout] hard timeout after {hard_timeout:.0f}s")
            except asyncio.CancelledError:
                common.remove_workspace(ws)
                raise
            except Exception as exc:  # noqa: BLE001 - recorded as a failed attempt
                result = _failure(f"[exception] {type(exc).__name__}: {exc}")
            finally:
                common.remove_workspace(ws)
            wall = time.monotonic() - t0
            parse: Optional[dict] = None
            call_ok = bool(result.success)
            error = result.error
            if call_ok:
                parse = common.parse_output(result.output)
                if parse["method"] == "none" and (result.output or "").strip():
                    # Unusable output is a failed attempt, never a clean review.
                    call_ok = False
                    error = UNPARSED_ERROR + (
                        f" ({parse['invalid_reason']})" if parse.get("invalid_reason") else "")
            attempts.append({
                "started_utc": started,
                "finished_utc": common.utc_now(),
                "wall_seconds": round(wall, 3),
                "success": call_ok,
                "error": error,
                "exit_code": result.exit_code,
                "tokens": common.usage_tokens(result),
                "tool_items": common.extract_tool_items(captured),
            })
            if call_ok:
                break
            self.log(key, "ATTEMPT_FAILED", None,
                     f"attempt={len(attempts)} error=" + " ".join(str(error).split())[:300])
            if _is_auth_error(error) or k == self.max_attempts - 1:
                break
            delay = min(MAX_BACKOFF, self.backoff * (2 ** k))
            delay *= 1.0 + random.uniform(-0.2, 0.2)
            await self.sleep(delay)

        last = attempts[-1]
        final_ok = bool(last["success"])
        record: dict[str, Any] = {
            "schema": common.REVIEW_SCHEMA,
            "key": {"model": key.model, "item": key.item, "repeat": key.repeat, "tag": key.tag},
            "request": {
                **signature,
                "prompt_ref": prompt_ref,
                "lens_role": role,
                "system_prompt_version": prompt.version,
                "task_statement_sha256": common.sha256_text(common.TASK_STATEMENT),
                "artifact_sha256": common.sha256_text(item.artifact_text),
                "allow_tools": False,
                "timeout_seconds": self.timeout,
                "adapter": type(adapter).__name__,
                "cli_version": result.cli_version or cli_version,
                "sampling": "CLI defaults (temperature not user-settable)",
                "codex_env": (common.codex_env_fingerprint(self.codex_home)
                              if signature["provider"] == "codex" else None),
            },
            "success": final_ok,
            "error": None if final_ok else last["error"],
            "attempts": attempts,
            "started_utc": call_started if previous is None else (previous.get("started_utc") or call_started),
            "finished_utc": last["finished_utc"],
            "last_attempt_started_utc": last["started_utc"],
            "wall_seconds": last["wall_seconds"],
            "answering_model": result.model,
            "reasoning_effort_reported": result.reasoning_effort,
            "tokens": common.usage_tokens(result),
            "tool_events": common.count_tool_events(result),
            "tool_items": last.get("tool_items", []),
            "raw_output": result.output,
            "raw_log": (result.raw_log or "")[-RAW_LOG_LIMIT:],
            "structured_output": result.structured_output,
            "parse": None,
            "critiques": [],
        }
        if parse is not None:
            parse = dict(parse)
            critiques = parse.pop("critiques")
            record["parse"] = parse
            # Critiques count only for a usable review.
            record["critiques"] = critiques if final_ok else []
        if final_ok:
            self.stats["succeeded"] += 1
        else:
            self.stats["failed"] += 1
        common.atomic_write_json(path, record)
        self.log(key, "OK" if final_ok else "FAILED", record)
        return record

    async def run(self, keys: list[CallKey], concurrency: int = 4) -> dict:
        queue: asyncio.Queue = asyncio.Queue()
        for k in keys:
            queue.put_nowait(k)

        async def worker() -> None:
            while True:
                try:
                    key = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await self.run_call(key)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - never kill the pool
                    self.stats["failed"] += 1
                    self.log(key, "HARNESS_ERROR", None, f"{type(exc).__name__}: {exc}")

        workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
        await asyncio.gather(*workers)
        return dict(self.stats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True, help="comma-separated, e.g. gpt-5.6-sol,gpt-5.6-luna")
    ap.add_argument("--effort", required=True, help="reasoning effort for every model (e.g. medium)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--items", default="", help="comma-separated item ids (default: all 18)")
    ap.add_argument("--calls", default=",".join(common.ALL_CALLS))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-call timeout (s)")
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="first retry delay (s)")
    ap.add_argument("--data-dir", default=str(common.DEFAULT_DATA_DIR))
    ap.add_argument("--codex-home", default=None,
                    help="CODEX_HOME for the codex subprocesses (default: inherited / ~/.codex)")
    ap.add_argument("--workspace-root", default=None,
                    help="parent of the empty per-call working dirs (default: $TMPDIR/miw_bench_ws)")
    ap.add_argument("--codex-web-search", choices=("disabled", "default"), default="disabled",
                    help="disable Codex's web-search tool (default) or leave the CLI default")
    ap.add_argument("--codex-tools-off", choices=common.CODEX_TOOLS_OFF_MODES, default="web_only",
                    help="web_only (default, the frozen pilot condition: web search off, user "
                         "config kept) or strict (backend adapter tools-off set: no shell, no "
                         "apps/plugins/MCP, user config.toml ignored). Part of the cache "
                         "signature.")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and cache state only")
    args = ap.parse_args(argv)

    common.verify_manifest()
    common.install_cli_capture()
    items = common.load_items()
    item_ids = common.parse_csv_arg(args.items) or list(items)
    unknown = [i for i in item_ids if i not in items]
    calls = common.parse_csv_arg(args.calls)
    bad_calls = [c for c in calls if c not in common.CALL_SPECS]
    if unknown or bad_calls:
        print(f"unknown items {unknown} / calls {bad_calls}", file=sys.stderr)
        return 2
    models = common.parse_csv_arg(args.models)
    for m in models:
        common.resolve_provider(m)
    if args.codex_home:
        os.environ["CODEX_HOME"] = args.codex_home

    keys = schedule(models, item_ids, args.repeats, calls)
    data_dir = Path(args.data_dir)
    cached = sum(
        1 for k in keys
        if (r := common.load_review(data_dir, k.model, k.item, k.repeat, k.tag)) and r.get("success")
    )
    print(f"{len(keys)} calls scheduled ({len(models)} models x {len(item_ids)} items x "
          f"{args.repeats} repeats x {len(calls)} calls); {cached} already cached; "
          f"concurrency {args.concurrency}; effort {args.effort}", flush=True)
    if args.dry_run:
        for k in keys[:40]:
            print("  ", k.label())
        return 0

    runner = Runner(
        data_dir=data_dir, effort=args.effort, timeout=args.timeout,
        max_attempts=args.max_attempts, backoff=args.backoff, codex_home=args.codex_home,
        workspace_root=Path(args.workspace_root) if args.workspace_root else None, items=items,
        codex_extra=common.codex_extra_args(args.codex_web_search, args.codex_tools_off),
    )
    common.append_log(runner.log_path, f"{common.utc_now()} START models={models} items={item_ids} "
                      f"repeats={args.repeats} calls={calls} effort={args.effort} "
                      f"concurrency={args.concurrency} timeout={args.timeout} "
                      f"codex_extra={runner.codex_extra}")
    t0 = time.monotonic()
    stats = asyncio.run(runner.run(keys, concurrency=args.concurrency))
    stats["wall_seconds"] = round(time.monotonic() - t0, 1)
    common.append_log(runner.log_path, f"{common.utc_now()} END {json.dumps(stats)}")
    print(json.dumps(stats))
    return 0 if stats["failed"] == 0 and stats["cache_mismatch"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
