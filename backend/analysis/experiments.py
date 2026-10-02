"""Orchestration experiments: mock implementation checks and runner-path overhead.

Two groups of experiments live here (PLOS revision 2, work packages E9 and X6a).
Their outputs go to ``experiments/orchestration/``.

1. Mock implementation verification (``mock_verification``)
   The mock adapter returns fixed, hand-written critique sets: three lenses, one
   critique raised word for word by two of them. These experiments only check
   that the merge code does what it is specified to do on known inputs:

   - identical critiques are merged;
   - a critique raised by two lenses is escalated one severity level;
   - critiques from a single lens are left alone;
   - a larger panel never yields fewer distinct critiques;
   - the loop is deterministic.

   They say nothing about whether a reviewer panel finds more real problems.
   That question is answered with real models on the planted-flaw benchmark
   (``experiments/benchmark``).

   - ``consensus_ablation``: distinct and merged critique counts for panel
     sizes 1..3 at a fixed mock tier.
   - ``role_overlap``: pairwise overlap of the lenses' critique sets. Critiques
     are compared by exact string equality after removing the reviewer tag, so
     the field is called ``exact_string_set_overlap``. It is not a semantic
     similarity.
   - ``merge_semantics_checks``: pass/fail checks of the merge rules above.
   - ``convergence_profile``: the ``reviewer_consensus`` loop on the mock with
     the adaptive stopping rule enabled explicitly (``convergence_enabled``
     is off by default since the stopping calibration) and no revision budget
     (``revision_budget: None``), so only the adaptive rule or the iteration
     cap can end it. The mock's outputs depend only on the call index
     (seeding the RNG changes only delays and token jitter). The result is
     therefore one deterministic trajectory (``n_runs = 1``), and repeating it
     with different seeds checks only that it is identical.

2. Control-plane overhead through the real runner path (``control_plane_overhead``)
   Concurrent runs are driven through ``backend.orchestrator.runner.execute_run``
   with a real SQLite database (``backend.database``) and real artifact writing
   in a temporary workspace. Iterations are fixed: convergence is off and the
   revision budget is disabled (``revision_budget: None``), so every run ends
   at its iteration cap. The model backend is the mock adapter with a fixed
   delay:

   - 0 s measures pure orchestration cost;
   - 1 s shows that cost relative to a slow backend.

   Every adapter call is timed. Per-iteration overhead is run wall time minus
   adapter time, divided by iterations. After every batch, ``check_integrity``
   runs ``PRAGMA integrity_check`` and ``foreign_key_check``, looks for orphaned
   rows, and checks that each run's iteration count agrees across the DB rows,
   the run row, the ``iter_NNNN`` directories on disk and ``run_meta.json``.

   Real-backend concurrency (provider latency, rate limits, retries) is measured
   separately by ``scripts/real_concurrency.py``.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import random
import re
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Optional, Sequence

from backend.adapters.base import BaseAdapter
from backend.adapters.mock import MockAdapter
from backend.models import (
    AdapterRunRequest,
    AdapterRunResult,
    ProviderName,
    RunState,
    SeverityLevel,
    WorkspaceConfig,
    WorkspaceContext,
)
from backend.orchestrator.consensus import ConsensusReviewer
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset

FULL_PANEL = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]

MOCK_VERIFICATION_SCHEMA = "miw-orchestration-mock-verification/1"
OVERHEAD_SCHEMA = "miw-orchestration-overhead/1"

# Merge used by these experiments, pinned so their results do not depend on the
# platform default (LLM adjudication since the benchmark calibration): the
# deterministic lexical merge (legacy jaccard at its uncalibrated 0.5), with
# which the published mock verification and overhead results were produced.
# On the mock's hand-written critiques it merges exactly the verbatim
# duplicates. In the overhead runs it also keeps every iteration at one
# adapter round (no extra adjudicator call per consensus step).
EXPERIMENT_MERGE = {"consensus_similarity_method": "jaccard",
                    "consensus_similarity_threshold": 0.5}

_SEVERITY_ORDER = [SeverityLevel.INFO, SeverityLevel.LOW, SeverityLevel.MEDIUM,
                   SeverityLevel.HIGH, SeverityLevel.CRITICAL]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sp(role: str, ref: str) -> str:
    return f"You are the {role}. Review the following artifacts."


def _strip_tag(desc: str) -> str:
    """Drop a leading ``[reviewer, ...]`` tag to get the bare critique text."""
    if desc.startswith("["):
        close = desc.find("]")
        if close != -1:
            return desc[close + 1:].strip()
    return desc.strip()


def _escalated(sev: SeverityLevel) -> SeverityLevel:
    idx = _SEVERITY_ORDER.index(sev)
    return _SEVERITY_ORDER[min(idx + 1, len(_SEVERITY_ORDER) - 1)]


async def _panel_result(panel, tier, consensus_threshold: int = 2):
    MockAdapter.fixed_iteration = tier
    try:
        cr = ConsensusReviewer(
            panel=panel, consensus_threshold=consensus_threshold,
            similarity_method=EXPERIMENT_MERGE["consensus_similarity_method"],
            similarity_threshold=EXPERIMENT_MERGE["consensus_similarity_threshold"])
        merged, _ = await cr.run_panel(
            "EXECUTOR ARTIFACT UNDER REVIEW",
            MockAdapter(min_delay=0, max_delay=0),
            WorkspaceContext(workspace_path=""),
            system_prompt_builder=_sp,
        )
        return merged, cr.compute_consensus_report(merged)
    finally:
        MockAdapter.fixed_iteration = None


# ══════════════════════════════════════════════════════════════════════
# 1. Mock implementation verification
# ══════════════════════════════════════════════════════════════════════


def consensus_ablation(panel_sizes=(1, 2, 3), tier: int = 1) -> dict:
    """Panel-size ablation at a fixed mock tier (implementation check only)."""
    rows = []
    for n in panel_sizes:
        merged, report = asyncio.run(_panel_result(FULL_PANEL[:n], tier))
        rows.append({
            "panel_size": n,
            "lenses": [r for r, _ in FULL_PANEL[:n]],
            "distinct_critiques": report["total_merged"],
            "multi_reviewer": report["multi_reviewer"],
            "unresolved_critical": report["unresolved_critical"],
            "grade": report["grade"],
            "severity_histogram": report["severity_histogram"],
        })
    return {"tier": tier, "rows": rows}


def _lens_critiques(tier: int) -> dict[str, dict[str, SeverityLevel]]:
    """Per-lens critique text -> severity, each lens run alone (no merging)."""
    out: dict[str, dict[str, SeverityLevel]] = {}
    for role, ref in FULL_PANEL:
        merged, _ = asyncio.run(_panel_result([(role, ref)], tier))
        out[role] = {_strip_tag(c.description): c.severity for c in merged.critiques}
    return out


def role_overlap(tier: int = 1) -> dict:
    """Pairwise overlap of the lenses' critique sets by exact string equality.

    ``exact_string_set_overlap`` = |A ∩ B| / |A ∪ B| where critiques are the
    bare critique strings (reviewer tag removed) compared by exact equality.
    It counts only verbatim duplicates; it is not a semantic or token-level
    similarity.
    """
    lens = _lens_critiques(tier)
    sets = {r: set(c) for r, c in lens.items()}
    roles = [r for r, _ in FULL_PANEL]
    pairwise = []
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            a, b = sets[roles[i]], sets[roles[j]]
            union = a | b
            ov = (len(a & b) / len(union)) if union else 0.0
            pairwise.append({
                "roles": [roles[i], roles[j]],
                "exact_string_set_overlap": round(ov, 3),
                "shared_exact": len(a & b),
            })
    union_all = set().union(*sets.values()) if sets else set()
    return {
        "tier": tier,
        "overlap_measure": (
            "exact_string_set_overlap = |A∩B|/|A∪B| over critique strings compared by "
            "exact equality after removing the reviewer tag (verbatim duplicates only)"
        ),
        "per_role_counts": {r: len(s) for r, s in sets.items()},
        "pairwise": pairwise,
        "union_size": len(union_all),
        "sum_individual": sum(len(s) for s in sets.values()),
    }


def merge_semantics_checks(tier: int = 1, consensus_threshold: int = 2) -> dict:
    """Pass/fail checks that the merge implements its specification on mock input.

    Ground truth comes from running each lens alone: a critique string raised
    by k lenses must appear once in the merged panel output, with ``raised_by``
    equal to those lenses. When k >= ``consensus_threshold`` its severity must
    be one level above the highest single-lens severity; otherwise it must keep
    that severity. The number of merged critiques must equal the number of
    distinct strings. Panel-size monotonicity is checked from the ablation.
    """
    lens = _lens_critiques(tier)
    raisers: dict[str, list[str]] = {}
    max_sev: dict[str, SeverityLevel] = {}
    for role, crits in lens.items():
        for text, sev in crits.items():
            raisers.setdefault(text, []).append(role)
            prev = max_sev.get(text)
            if prev is None or _SEVERITY_ORDER.index(sev) > _SEVERITY_ORDER.index(prev):
                max_sev[text] = sev

    merged, report = asyncio.run(_panel_result(FULL_PANEL, tier, consensus_threshold))
    by_text: dict[str, list] = {}
    for c in merged.critiques:
        by_text.setdefault(_strip_tag(c.description), []).append(c)

    problems: list[str] = []
    shared_ok = escalation_ok = single_ok = True
    for text, roles in raisers.items():
        got = by_text.get(text, [])
        if len(got) != 1:
            problems.append(f"{text!r}: expected once in merged output, found {len(got)}")
            if len(roles) >= 2:
                shared_ok = False
            else:
                single_ok = False
            continue
        c = got[0]
        raised_by = sorted(c.raised_by) if c.raised_by else []
        if len(roles) >= 2 and raised_by != sorted(roles):
            shared_ok = False
            problems.append(f"{text!r}: raised_by {raised_by} != {sorted(roles)}")
        expected = (_escalated(max_sev[text]) if len(set(roles)) >= consensus_threshold
                    else max_sev[text])
        if c.severity != expected:
            problems.append(f"{text!r}: severity {c.severity.value} != {expected.value}")
            if len(set(roles)) >= consensus_threshold:
                escalation_ok = False
            else:
                single_ok = False
    extra = sorted(set(by_text) - set(raisers))
    if extra:
        problems.append(f"merged output has critiques no lens raised: {extra}")

    ablation = consensus_ablation(panel_sizes=tuple(range(1, len(FULL_PANEL) + 1)), tier=tier)
    distinct = [r["distinct_critiques"] for r in ablation["rows"]]
    checks = {
        "distinct_equals_exact_union": report["total_merged"] == len(raisers) and not extra,
        "shared_critiques_merged_once_with_all_raisers": shared_ok,
        "agreement_escalates_one_level": escalation_ok,
        "single_lens_critiques_keep_severity": single_ok,
        "panel_size_monotonic": distinct == sorted(distinct),
    }
    n_shared = sum(1 for r in raisers.values() if len(set(r)) >= 2)
    return {
        "tier": tier,
        "consensus_threshold": consensus_threshold,
        "similarity_method": report.get("similarity_method"),
        "similarity_threshold": report.get("similarity_threshold"),
        "n_distinct_exact": len(raisers),
        "n_shared_exact": n_shared,
        "n_merged": report["total_merged"],
        "multi_reviewer": report["multi_reviewer"],
        "checks": checks,
        "all_passed": all(checks.values()),
        "problems": problems,
    }


def _trajectory(final: RunState) -> list[dict]:
    out = []
    for it in sorted(final.iterations, key=lambda i: i.iteration_number):
        out.append({
            "iteration": it.iteration_number,
            "role": it.role,
            "status": it.status.value,
            "grade": it.grade,
            "critical": it.critical_count,
            "high": it.high_count,
        })
    return out


def convergence_profile(max_iterations: int = 40, seeds: Sequence[int] = (0, 1, 2)) -> dict:
    """The mock ``reviewer_consensus`` loop: one deterministic trajectory.

    The mock adapter's outputs are a function of the global call counter only;
    its RNG affects delays and token jitter, never the text. Re-running with
    different seeds therefore cannot produce independent runs. This function
    reports a single trajectory (``n_runs = 1``) and replays it once per seed
    to confirm it is identical (``determinism_check``).

    The adaptive rule is enabled explicitly and the revision budget disabled,
    so the profile shows the adaptive rule alone (``convergence_enabled`` is
    off and ``revision_budget`` is 4 by default).
    """
    async def one_run(seed: int):
        random.seed(seed)
        MockAdapter.reset_iteration_count()
        signal: dict[str, Any] = {"reason": "", "stop_reason": "", "signals": []}

        def on_event(t, d):
            if t == "convergence_detected":
                signal.update(reason=d.get("reason", ""),
                              stop_reason=d.get("stop_reason", ""),
                              signals=list(d.get("signals", [])))

        eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0), on_event=on_event)
        loop = get_preset("reviewer_consensus", max_iterations=max_iterations)
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus",
                       task="probe", provider=ProviderName.MOCK, model="",
                       max_iterations=max_iterations,
                       config={"convergence_enabled": True, "revision_budget": None,
                               **EXPERIMENT_MERGE})
        final = await eng.run_loop(run, loop)
        return final, signal

    seeds = list(seeds) or [0]
    replays = []
    for s in seeds:
        final, signal = asyncio.run(one_run(s))
        replays.append({
            "seed": s,
            "iterations": len(final.iterations),
            "stop_reason": final.stop_reason or "",
            "signals": signal["signals"],
            "reason": signal["reason"],
            "trajectory": _trajectory(final),
        })
    ref = replays[0]
    identical = all(
        (r["iterations"], r["stop_reason"], r["trajectory"])
        == (ref["iterations"], ref["stop_reason"], ref["trajectory"])
        for r in replays[1:]
    )
    return {
        "n_runs": 1,
        "deterministic": True,
        "note": (
            "The mock adapter's outputs depend only on the call index, so the loop has "
            "exactly one trajectory; this is a single deterministic run, not a sample. "
            "Seeds change only mock delays and token jitter and are used to confirm "
            "the trajectory is reproduced exactly."
        ),
        "preset": "reviewer_consensus",
        "max_iterations": max_iterations,
        "iterations": ref["iterations"],
        "stop_reason": ref["stop_reason"],
        "signals": ref["signals"],
        "reason": ref["reason"],
        "trajectory": ref["trajectory"],
        "determinism_check": {
            "seeds": seeds,
            "identical": identical,
            "iterations_per_seed": [r["iterations"] for r in replays],
            "stop_reason_per_seed": [r["stop_reason"] for r in replays],
        },
    }


def mock_verification(tier: int = 1, max_iterations: int = 40,
                      seeds: Sequence[int] = (0, 1, 2)) -> dict:
    """All mock implementation-verification experiments in one record."""
    return {
        "schema": MOCK_VERIFICATION_SCHEMA,
        "created_utc": _utc_now(),
        "purpose": (
            "Implementation verification of the merge semantics and loop control on the "
            "mock adapter, whose critique sets are fixed and hand-written (one critique is "
            "shared verbatim by two lenses). These results check that the code does what "
            "it specifies; they are not evidence about reviewer quality or coverage."
        ),
        "adapter": "mock",
        "merge_semantics": merge_semantics_checks(tier=tier),
        "consensus_ablation": consensus_ablation(tier=tier),
        "role_overlap": role_overlap(tier=tier),
        "convergence_profile": convergence_profile(max_iterations=max_iterations, seeds=seeds),
    }


# ══════════════════════════════════════════════════════════════════════
# 2. Control-plane overhead through the real runner path
# ══════════════════════════════════════════════════════════════════════

_RUN_DIR_RE = re.compile(r"(?:^|/)runs/([^/]+)")
_ITER_DIR_RE = re.compile(r"^iter_(\d{4,})$")


def _run_id_of(request: AdapterRunRequest) -> str:
    ctx = request.workspace_context
    m = _RUN_DIR_RE.search(getattr(ctx, "run_dir", "") or "")
    if m:
        return m.group(1)
    try:
        from backend.logging_config import _current_run_id
        return _current_run_id.get(None) or ""
    except Exception:  # pragma: no cover - defensive
        return ""


class TimedAdapter(BaseAdapter):
    """Adapter proxy that records the wall-clock interval of every call per run.

    The run is identified from ``workspace_context.run_dir``
    (``runs/<run_id>``), which the engine sets on every request, including
    consensus-panel lens calls.
    """

    def __init__(self, inner: BaseAdapter):
        self.inner = inner
        self.name = getattr(inner, "name", "mock")
        self.intervals: dict[str, list[tuple[float, float]]] = {}

    def is_available(self) -> bool:
        return self.inner.is_available()

    async def smoke_test(self) -> dict:
        return await self.inner.smoke_test()

    async def cli_version(self) -> str:
        return await self.inner.cli_version()

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        run_id = _run_id_of(request)
        t0 = time.perf_counter()
        try:
            return await self.inner.run(request)
        finally:
            self.intervals.setdefault(run_id, []).append((t0, time.perf_counter()))


def interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length covered by possibly overlapping intervals (parallel panel calls)."""
    total = 0.0
    cur_s: Optional[float] = None
    cur_e = 0.0
    for s, e in sorted(intervals):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); 0.0 for an empty sequence."""
    vals = sorted(values)
    if not vals:
        return 0.0
    k = max(1, math.ceil(q / 100.0 * len(vals)))
    return vals[min(k, len(vals)) - 1]


def _summary(values: Sequence[float], scale: float = 1.0, digits: int = 3) -> dict:
    vals = [v * scale for v in values]
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


async def check_integrity(db_path: str, workspace_path: str,
                          expect_stop_reason: Optional[str] = "max_iterations") -> dict:
    """Database / artifact consistency check for every run in ``db_path``.

    Checks: ``PRAGMA integrity_check`` == ok; ``PRAGMA foreign_key_check`` is
    empty; no iteration rows without a run, no runs without a workspace; for
    every run, the iteration rows are numbered 1..n without duplicates and are
    all completed, n equals ``runs.current_iteration`` and ``max_iterations``
    (fixed-iteration runs), the run completed (with ``expect_stop_reason``), the
    run directory has exactly the ``iter_NNNN`` directories 1..n, each containing
    ``<role>_output.md`` for the DB row's role, and ``run_meta.json`` agrees
    (``total_iterations`` and number of iteration entries). Run directories
    on disk with no DB row are reported as orphans.
    """
    from backend.database import close_connection, open_connection
    from backend.orchestrator.engine import _safe_name

    db = await open_connection(db_path)
    try:
        cur = await db.execute("PRAGMA integrity_check")
        integrity = [r[0] for r in await cur.fetchall()]
        cur = await db.execute("PRAGMA foreign_key_check")
        fk = await cur.fetchall()
        cur = await db.execute(
            "SELECT COUNT(*) FROM iterations i LEFT JOIN runs r ON r.run_id = i.run_id "
            "WHERE r.run_id IS NULL")
        orphan_iterations = (await cur.fetchone())[0]
        cur = await db.execute(
            "SELECT COUNT(*) FROM runs r LEFT JOIN workspaces w ON w.id = r.workspace_id "
            "WHERE w.id IS NULL")
        orphan_runs = (await cur.fetchone())[0]
        cur = await db.execute(
            "SELECT run_id, status, current_iteration, max_iterations, stop_reason FROM runs")
        runs = await cur.fetchall()
        cur = await db.execute(
            "SELECT run_id, iteration_number, role, status FROM iterations")
        its = await cur.fetchall()
    finally:
        await close_connection(db)

    by_run: dict[str, list[tuple[int, str, str]]] = {}
    for run_id, num, role, status in its:
        by_run.setdefault(run_id, []).append((int(num), role, status))

    runs_root = Path(workspace_path) / "runs"
    problems: list[str] = []
    n_iters = 0
    for run_id, status, current, max_it, stop_reason in runs:
        rows = sorted(by_run.get(run_id, []))
        nums = [r[0] for r in rows]
        n = len(rows)
        n_iters += n
        if nums != list(range(1, n + 1)):
            problems.append(f"{run_id}: iteration numbers {nums} are not 1..{n} (duplicates/gaps)")
        if any(r[2] != "completed" for r in rows):
            problems.append(f"{run_id}: non-completed iteration rows")
        if n != current:
            problems.append(f"{run_id}: {n} iteration rows but runs.current_iteration={current}")
        if n != max_it:
            problems.append(f"{run_id}: {n} iterations, expected fixed {max_it}")
        if status != "completed":
            problems.append(f"{run_id}: status {status}")
        if expect_stop_reason is not None and stop_reason != expect_stop_reason:
            problems.append(f"{run_id}: stop_reason {stop_reason!r} != {expect_stop_reason!r}")
        run_dir = runs_root / run_id
        disk = sorted(
            int(m.group(1)) for d in (run_dir.iterdir() if run_dir.is_dir() else [])
            if d.is_dir() and (m := _ITER_DIR_RE.match(d.name))
        )
        if disk != nums:
            problems.append(f"{run_id}: iteration dirs on disk {disk} != DB {nums}")
        for num, role, _ in rows:
            f = run_dir / f"iter_{num:04d}" / f"{_safe_name(role)}_output.md"
            if not f.is_file() or f.stat().st_size == 0:
                problems.append(f"{run_id}: missing or empty {f.relative_to(runs_root)}")
        meta_path = run_dir / "run_meta.json"
        if not meta_path.is_file():
            problems.append(f"{run_id}: run_meta.json missing")
        else:
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("total_iterations") != n or len(meta.get("iterations", [])) != n:
                    problems.append(
                        f"{run_id}: run_meta total_iterations={meta.get('total_iterations')} "
                        f"entries={len(meta.get('iterations', []))} != {n}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{run_id}: run_meta.json unreadable ({exc})")

    known = {r[0] for r in runs}
    orphan_dirs = sorted(
        d.name for d in (runs_root.iterdir() if runs_root.is_dir() else [])
        if d.is_dir() and d.name not in known
    )
    ok = (integrity == ["ok"] and not fk and orphan_iterations == 0 and orphan_runs == 0
          and not orphan_dirs and not problems)
    return {
        "ok": ok,
        "integrity_check": integrity[0] if len(integrity) == 1 else integrity,
        "foreign_key_violations": len(fk),
        "orphan_iteration_rows": orphan_iterations,
        "orphan_run_rows": orphan_runs,
        "orphan_run_dirs": len(orphan_dirs),
        "runs_checked": len(runs),
        "iterations_checked": n_iters,
        "n_problems": len(problems),
        "problems": problems[:20],
    }


async def _overhead_sweep(
    delay: float,
    levels: Sequence[int],
    iterations: int,
    repeats: int,
    preset: str,
    root: Path,
    warmup: bool = True,
) -> dict:
    from backend.adapters import registry
    from backend.config import config
    from backend.database import create_run, create_workspace, init_db
    from backend.orchestrator import runner

    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "miw.db"
    ws_dir = root / "workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)

    old_db = config.db_path
    old_adapter = registry._ADAPTERS[ProviderName.MOCK]
    old_fixed = MockAdapter.fixed_iteration
    timed = TimedAdapter(MockAdapter(min_delay=delay, max_delay=delay))
    config.db_path = str(db_path)
    registry._ADAPTERS[ProviderName.MOCK] = timed
    MockAdapter.fixed_iteration = 1   # identical per-call work in every run
    try:
        await init_db()
        ws = WorkspaceConfig(name=f"overhead-d{delay:g}", path=str(ws_dir),
                             default_provider=ProviderName.MOCK)
        await create_workspace(ws)
        run_cfg = {
            "convergence_enabled": False,
            "revision_budget": None,   # fixed iterations: only max_iterations ends a run
            "budget_max_tokens": 10 ** 12,
            "budget_max_cost": 1e12,
            **EXPERIMENT_MERGE,
        }

        async def batch(n: int, n_iter: int) -> dict:
            ids = []
            t_create = time.perf_counter()
            for _ in range(n):
                run = RunState(workspace_id=ws.id, loop_preset=preset, task="overhead probe",
                               provider=ProviderName.MOCK, model="", max_iterations=n_iter,
                               config=dict(run_cfg))
                await create_run(run)
                ids.append(run.run_id)
            create_s = time.perf_counter() - t_create

            spans: dict[str, tuple[float, float]] = {}

            async def timed_run(run_id: str) -> None:
                t0 = time.perf_counter()
                try:
                    await runner.execute_run(run_id)
                finally:
                    spans[run_id] = (t0, time.perf_counter())

            cpu0 = time.process_time()
            t0 = time.perf_counter()
            tasks = []
            for rid in ids:
                task = asyncio.create_task(timed_run(rid))
                runner._active_tasks[rid] = task   # as POST /api/runs does
                tasks.append(task)
            await asyncio.gather(*tasks)
            wall = time.perf_counter() - t0
            cpu = time.process_time() - cpu0

            per_run = []
            for rid in ids:
                s, e = spans[rid]
                ivs = timed.intervals.get(rid, [])
                busy = interval_union(ivs)
                per_run.append({
                    "run_id": rid,
                    "latency_s": e - s,
                    "adapter_busy_s": busy,
                    "adapter_calls": len(ivs),
                })
            integrity = await check_integrity(str(db_path), str(ws_dir))
            return {"ids": ids, "wall": wall, "cpu": cpu, "create_s": create_s,
                    "per_run": per_run, "integrity": integrity}

        warm = None
        if warmup:
            w = await batch(1, 2)
            warm = {"wall_seconds": round(w["wall"], 4), "integrity_ok": w["integrity"]["ok"]}

        rows = []
        for n in levels:
            load_start = _loadavg()
            reps = [await batch(n, iterations) for _ in range(repeats)]
            per_run = [r for b in reps for r in b["per_run"]]
            total_iters = [n * iterations for _ in reps]
            lat = [r["latency_s"] for r in per_run]
            ovh = [(r["latency_s"] - r["adapter_busy_s"]) / iterations for r in per_run]
            ovh_upper = [(r["latency_s"] - iterations * delay) / iterations for r in per_run]
            busy_it = [r["adapter_busy_s"] / iterations for r in per_run]
            frac = [((r["latency_s"] - r["adapter_busy_s"]) / r["latency_s"])
                    if r["latency_s"] > 0 else 0.0 for r in per_run]
            rows.append({
                "concurrency": n,
                "repeats": repeats,
                "runs": len(per_run),
                "iterations_per_run": iterations,
                "adapter_calls_per_run": sorted({r["adapter_calls"] for r in per_run}),
                "wall_seconds": [round(b["wall"], 4) for b in reps],
                "throughput_iters_per_s": _summary(
                    [t / b["wall"] for t, b in zip(total_iters, reps)], digits=2),
                "latency_s": _summary(lat, digits=4),
                "overhead_ms_per_iteration": _summary(ovh, scale=1000.0, digits=2),
                "overhead_upper_ms_per_iteration": _summary(ovh_upper, scale=1000.0, digits=2),
                "adapter_ms_per_iteration": _summary(busy_it, scale=1000.0, digits=2),
                "overhead_fraction_of_run_wall": _summary(frac, digits=4),
                "cpu_ms_per_iteration": _summary(
                    [b["cpu"] / t * 1000.0 for t, b in zip(total_iters, reps)], digits=2),
                "run_creation_ms_per_run": _summary(
                    [b["create_s"] / n * 1000.0 for b in reps], digits=2),
                "integrity": [b["integrity"] for b in reps],
                "integrity_ok": all(b["integrity"]["ok"] for b in reps),
                "loadavg_start": load_start,
            })
        return {
            "adapter_delay_s": delay,
            "warmup": warm,
            "rows": rows,
            "integrity_ok": all(r["integrity_ok"] for r in rows)
                            and (warm is None or warm["integrity_ok"]),
        }
    finally:
        config.db_path = old_db
        registry._ADAPTERS[ProviderName.MOCK] = old_adapter
        MockAdapter.fixed_iteration = old_fixed


def _rmtree(path: Path, tries: int = 3) -> None:
    """rmtree that tolerates exFAT AppleDouble ``._*`` files vanishing mid-walk."""
    for _ in range(tries):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return


def _loadavg() -> Optional[list[float]]:
    """1/5/15-minute load averages (other processes perturb wall-clock numbers)."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def _environment() -> dict:
    try:
        import aiosqlite
        aiosqlite_version = getattr(aiosqlite, "__version__", "")
    except Exception:  # pragma: no cover
        aiosqlite_version = ""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "sqlite_version": sqlite3.sqlite_version,
        "aiosqlite_version": aiosqlite_version,
    }


def control_plane_overhead(
    concurrency_levels: Sequence[int] = (1, 2, 5, 10, 20),
    iterations_per_run: int = 6,
    adapter_delays: Sequence[float] = (0.0, 1.0),
    repeats: int = 5,
    preset: str = "reviewer_consensus",
    parent_dir: Optional[str] = None,
    keep_workdir: bool = False,
) -> dict:
    """Orchestration overhead of concurrent runs through ``runner.execute_run``.

    A new temporary directory is created under ``parent_dir`` (default
    ``$TMPDIR``); ``parent_dir`` therefore selects the storage device the
    database and artifacts live on. For each mock delay, a fresh SQLite database
    and workspace are created inside it. The directory is removed afterwards
    unless ``keep_workdir``.
    One untimed warm-up run fills the prompt-template cache. Then, for each
    concurrency level, ``repeats`` batches of N runs are started together. Each
    run is created with ``create_run`` and executed with ``runner.execute_run``,
    registered in the runner's task table as the API does, with convergence off,
    no revision budget and ``iterations_per_run`` fixed.

    Every mock call returns the same tier-1 output, so each run does identical
    work. The integrity check covers every run in the database after each batch.

    Per run:

    - ``latency``: wall time of ``execute_run``;
    - ``adapter busy``: the union of its adapter-call intervals (panel calls
      overlap);
    - overhead per iteration = (latency − adapter busy) / iterations.

    The waiting adapter coroutines' resumption delay is counted as adapter
    time, so ``overhead_upper_ms_per_iteration`` gives the conservative variant
    (latency − iterations × delay) / iterations. Each iteration makes one
    adapter round: one executor call or one parallel panel.
    """
    levels = [int(c) for c in concurrency_levels]
    from backend.orchestrator.runner import MAX_CONCURRENT_RUNS
    if max(levels) > MAX_CONCURRENT_RUNS:
        raise ValueError(f"concurrency {max(levels)} exceeds MAX_CONCURRENT_RUNS={MAX_CONCURRENT_RUNS}")
    if parent_dir:
        Path(parent_dir).mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="miw_overhead_", dir=parent_dir or None))
    started = _utc_now()
    load_start = _loadavg()
    try:
        sweeps = []
        for d in adapter_delays:
            root = base / f"delay_{d:g}s"
            sweeps.append(asyncio.run(_overhead_sweep(
                float(d), levels, int(iterations_per_run), int(repeats), preset, root)))
    finally:
        if not keep_workdir:
            _rmtree(base)
    return {
        "schema": OVERHEAD_SCHEMA,
        "created_utc": started,
        "finished_utc": _utc_now(),
        "backend": "mock adapter (fixed delay); measurement of the orchestration layer only",
        "path": ("backend.orchestrator.runner.execute_run with backend.database (SQLite, WAL) "
                 "and artifact writing to a temporary workspace"),
        "preset": preset,
        "iterations_per_run": int(iterations_per_run),
        "concurrency_levels": levels,
        "repeats": int(repeats),
        "adapter_delays_s": [float(d) for d in adapter_delays],
        "mock_fixed_iteration": 1,
        "convergence_enabled": False,
        "revision_budget": None,
        "merge": dict(EXPERIMENT_MERGE),
        "definitions": {
            "overhead_ms_per_iteration": "(run wall time - union of the run's adapter-call "
                                         "intervals) / iterations",
            "overhead_upper_ms_per_iteration": "(run wall time - iterations x mock delay) / "
                                               "iterations (counts all event-loop contention "
                                               "as overhead)",
            "throughput_iters_per_s": "iterations completed in a batch / batch wall time",
            "latency_s": "wall time of one execute_run call",
            "cpu_ms_per_iteration": "process CPU time (all threads) during the batch / "
                                    "iterations",
            "percentiles": "nearest rank over all runs of all repeats at a level",
        },
        "environment": {**_environment(), "workdir": str(base),
                        "workdir_kept": bool(keep_workdir),
                        "loadavg_start": load_start, "loadavg_end": _loadavg()},
        "sweeps": sweeps,
        "integrity_ok": all(s["integrity_ok"] for s in sweeps),
    }


def run_all(include_overhead: bool = True, **overhead_kwargs: Any) -> dict:
    """Both experiment groups (mock verification and control-plane overhead)."""
    out = {"mock_verification": mock_verification()}
    if include_overhead:
        out["overhead"] = control_plane_overhead(**overhead_kwargs)
    return out
