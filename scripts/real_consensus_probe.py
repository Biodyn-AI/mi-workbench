#!/usr/bin/env python
"""Smoke test of the live consensus path with a real model backend (3 calls).

Runs one three-lens panel (rigour, adversarial, biological plausibility)
through ``ConsensusReviewer.run_panel``, the path the engine uses, over a
benchmark artifact. It records what is needed to audit the call:

- provider, requested and answering model, reasoning effort, CLI version;
- per-lens raw output, tokens, latency, status and parse method;
- the merged critiques and the consensus report.

This is a connectivity and parsing check, not an evaluation. Reviewer quality
is measured on the planted-flaw benchmark (``experiments/benchmark``) and
concurrency by ``scripts/real_concurrency.py``. The previous version reviewed
a hard-coded text whose numbers came from the mock executor; that text is
gone.

Prompts, adapter settings (``allow_tools=False``; for Codex also web search
off) and the empty working directory are those of the X1 benchmark
(``experiments/benchmark/common.py``).

Usage (from automation/mi-workbench, env ``mi_workbench``):

    python scripts/real_consensus_probe.py                       # codex gpt-5.6-luna, effort low
    python scripts/real_consensus_probe.py --model gpt-5.5 --effort medium \\
        --artifact experiments/benchmark/artifacts/A03-flawed.md --out /tmp/probe.json

``--model`` takes ``[provider:]model``: ``gpt-*`` uses Codex, ``claude-*``
Claude Code and ``gemini-*`` Gemini. Exit code 0 means every lens succeeded
and parsed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_concurrency as rc  # noqa: E402  (shared recorder + benchmark helpers)
from backend.models import WorkspaceContext  # noqa: E402
from backend.orchestrator.consensus import ConsensusReviewer  # noqa: E402

SCHEMA = "miw-real-consensus-probe/2"
DEFAULT_OUT = REPO_ROOT / "experiments" / "orchestration" / "smoke" / "real_consensus_probe.json"


async def probe(artifact: Path, model: str, effort: str, timeout: float,
                adapter=None, web_search: str = "disabled") -> dict:
    bc = rc.bench_common()
    text = artifact.read_text()
    user_prompt, lenses = rc.build_prompts(text)
    if adapter is None:
        provider, model_name = bc.resolve_provider(model)
        adapter = bc.make_adapter(model, effort, codex_extra=bc.codex_extra_args(web_search))
    else:  # injected adapter (tests)
        provider, model_name = getattr(adapter, "name", "custom"), model.split(":", 1)[-1]
    rec = rc.RecordingAdapter(adapter)
    reviewer = ConsensusReviewer(
        panel=list(rc.PANEL), lens_timeout=timeout, max_retries=1, retry_base_delay=10.0,
        # Pinned to the merge of the recorded probe (the platform default at the
        # time; the default is now LLM adjudication with one extra model call).
        similarity_method="jaccard", similarity_threshold=0.5,
        request_kwargs={"model": model_name, "reasoning_effort": effort, "allow_tools": False},
    )
    ws = bc.make_workspace()
    rc._PANEL_ID.set("probe")
    t0 = time.perf_counter()
    try:
        merged, raw = await reviewer.run_panel(
            user_prompt, rec, WorkspaceContext(workspace_path=str(ws)),
            system_prompt_builder=lambda role, ref: lenses[role]["text"],
        )
    finally:
        wall = time.perf_counter() - t0
        bc.remove_workspace(ws)
    report = reviewer.compute_consensus_report(merged)
    meta = merged.consensus_meta or {}

    per_lens = []
    for (role, ref), r in zip(rc.PANEL, raw):
        attempts = [c for c in rec.records if c["role"] == role]
        last = attempts[-1] if attempts else {}
        per_lens.append({
            "role": role,
            "prompt_ref": ref,
            "prompt_version": lenses[role]["version"],
            "prompt_sha256": lenses[role]["sha256"],
            "status": meta.get("lens_status", {}).get(role),
            "error": meta.get("lens_errors", {}).get(role),
            "attempts": len(attempts),
            "latency_s": [c["latency_s"] for c in attempts],
            "model": getattr(r, "model", ""),
            "reasoning_effort": getattr(r, "reasoning_effort", ""),
            "cli_version": getattr(r, "cli_version", ""),
            "input_tokens": last.get("input_tokens", 0),
            "cached_input_tokens": last.get("cached_input_tokens", 0),
            "output_tokens": last.get("output_tokens", 0),
            "reasoning_output_tokens": last.get("reasoning_output_tokens", 0),
            "tool_events": last.get("tool_events", 0),
            "parse_method": meta.get("parse_methods", {}).get(role),
            "raw_critiques": meta.get("raw_critique_counts", {}).get(role),
            "raw_output": getattr(r, "output", "") or "",
            "raw_log": (getattr(r, "raw_log", "") or "")[-5000:],
        })
    try:
        rel = str(artifact.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(artifact)
    return {
        "schema": SCHEMA,
        "created_utc": rc.utc_now(),
        "note": "Smoke test of the live consensus path (one panel, 3 calls); not an evaluation.",
        "artifact": rel,
        "artifact_sha256": rc.sha256_text(text),
        "provider": provider,
        "model_requested": model_name,
        "reasoning_effort": effort,
        "allow_tools": False,
        "web_search": web_search if provider == "codex" else None,
        "cli_version": await rec.cli_version(),
        "answering_models": sorted({x["model"] for x in per_lens if x["model"]}),
        "panel_wall_seconds": round(wall, 3),
        "tokens_total": {
            k: sum(c[k] for c in rec.records)
            for k in ("input_tokens", "cached_input_tokens", "output_tokens",
                      "reasoning_output_tokens")
        },
        "lenses": per_lens,
        "report": {k: report[k] for k in (
            "grade", "total_merged", "multi_reviewer", "unresolved_critical",
            "severity_histogram", "panel_failed", "panel_partial", "failed_lenses",
            "similarity_method", "similarity_threshold", "n_groups_multi_lens")},
        "merged_critiques": [
            {"severity": c.severity.value, "raised_by": list(c.raised_by or []),
             "category": c.category, "description": c.description}
            for c in merged.critiques
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default=str(rc.DEFAULT_ARTIFACT))
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--web-search", choices=["disabled", "default"], default="disabled")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    res = asyncio.run(probe(Path(args.artifact), args.model, args.effort, args.timeout,
                            web_search=args.web_search))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    summary = {k: res[k] for k in ("provider", "model_requested", "answering_models",
                                   "cli_version", "panel_wall_seconds", "tokens_total")}
    summary["report"] = res["report"]
    summary["lenses"] = [{k: x[k] for k in ("role", "status", "parse_method", "raw_critiques",
                                            "latency_s", "output_tokens")}
                         for x in res["lenses"]]
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")
    ok = all(x["status"] == "ok" for x in res["lenses"])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
