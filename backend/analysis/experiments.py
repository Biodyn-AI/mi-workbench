"""Reproducible evaluation harness for MI-Workbench orchestration.

These experiments quantify orchestration behaviour deterministically using the
mock adapter (no network, no API cost). They characterise the *mechanism* and
its *scaling*; claims about research quality rest on the separate live-model
case study.

Experiments
-----------
consensus_ablation : vary the reviewer panel size and measure how many distinct
    critiques the panel surfaces and how agreement escalates severity.
role_overlap       : pairwise overlap of the critique sets produced by the three
    reviewer lenses (evidence that the roles are complementary, not redundant).
convergence_profile: iterations-to-converge and which signal fires, per run.
load_benchmark     : concurrent-run throughput and per-run latency (p50/p95) of
    the centralized orchestrator across concurrency levels.
"""
from __future__ import annotations

import asyncio
import time
from statistics import median

from backend.adapters.mock import MockAdapter
from backend.models import ProviderName, RunState, WorkspaceContext
from backend.orchestrator.consensus import ConsensusReviewer
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset

FULL_PANEL = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]


def _sp(role: str, ref: str) -> str:
    return f"You are the {role}. Review the following artifacts."


def _strip_tag(desc: str) -> str:
    """Drop a leading ``[reviewer, ...]`` tag to get the bare critique text."""
    if desc.startswith("["):
        close = desc.find("]")
        if close != -1:
            return desc[close + 1:].strip()
    return desc.strip()


async def _panel_result(panel, tier):
    MockAdapter.fixed_iteration = tier
    try:
        cr = ConsensusReviewer(panel=panel)
        merged, _ = await cr.run_panel(
            "EXECUTOR ARTIFACT UNDER REVIEW",
            MockAdapter(min_delay=0, max_delay=0),
            WorkspaceContext(workspace_path=""),
            system_prompt_builder=_sp,
        )
        return merged, cr.compute_consensus_report(merged)
    finally:
        MockAdapter.fixed_iteration = None


def consensus_ablation(panel_sizes=(1, 2, 3), tier: int = 1) -> dict:
    """Panel-size ablation at a fixed quality tier."""
    rows = []
    for n in panel_sizes:
        merged, report = asyncio.run(_panel_result(FULL_PANEL[:n], tier))
        rows.append({
            "panel_size": n,
            "distinct_critiques": report["total_merged"],
            "multi_reviewer": report["multi_reviewer"],
            "unresolved_critical": report["unresolved_critical"],
            "grade": report["grade"],
            "severity_histogram": report["severity_histogram"],
        })
    return {"tier": tier, "rows": rows}


def role_overlap(tier: int = 1) -> dict:
    """Pairwise overlap of the three reviewer lenses' critique sets."""
    sets: dict[str, set] = {}
    for role, ref in FULL_PANEL:
        merged, _ = asyncio.run(_panel_result([(role, ref)], tier))
        sets[role] = {_strip_tag(c.description) for c in merged.critiques}

    roles = [r for r, _ in FULL_PANEL]
    pairwise = []
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            a, b = sets[roles[i]], sets[roles[j]]
            union = a | b
            jac = (len(a & b) / len(union)) if union else 0.0
            pairwise.append({
                "roles": [roles[i], roles[j]],
                "jaccard": round(jac, 3),
                "shared": len(a & b),
            })
    union_all = set().union(*sets.values()) if sets else set()
    return {
        "tier": tier,
        "per_role_counts": {r: len(s) for r, s in sets.items()},
        "pairwise": pairwise,
        "union_size": len(union_all),
        "sum_individual": sum(len(s) for s in sets.values()),
    }


def convergence_profile(n_runs: int = 20, max_iterations: int = 40) -> dict:
    """Iterations-to-converge and which signal fires, over repeated runs."""
    async def one_run(seed: int):
        MockAdapter.reset_iteration_count()
        signal = {"reason": ""}

        def on_event(t, d):
            if t == "convergence_detected":
                signal["reason"] = d.get("reason", "")

        eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0),
                         on_event=on_event)
        loop = get_preset("reviewer_consensus", max_iterations=max_iterations)
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus",
                       task="probe", provider=ProviderName.MOCK, model="",
                       max_iterations=max_iterations,
                       config={"convergence_enabled": True})
        final = await eng.run_loop(run, loop)
        return len(final.iterations), signal["reason"]

    iters, reasons = [], []
    for s in range(n_runs):
        n, reason = asyncio.run(one_run(s))
        iters.append(n)
        reasons.append(reason.split(" for ")[0] if reason else "max_iterations")

    signal_counts: dict[str, int] = {}
    for r in reasons:
        key = ("grade" if r.startswith("Grade") else
               "zero_critical" if r.startswith("Zero") else
               "similarity" if r.startswith("Executor") else "max_iterations")
        signal_counts[key] = signal_counts.get(key, 0) + 1

    return {
        "n_runs": n_runs,
        "max_iterations": max_iterations,
        "iterations_to_converge": {
            "min": min(iters), "median": median(iters), "max": max(iters),
        },
        "signal_counts": signal_counts,
    }


def load_benchmark(
    concurrency_levels=(1, 2, 5, 10, 20),
    iterations_per_run: int = 6,
    adapter_delay: float = 0.005,
) -> dict:
    """Concurrent-run throughput and per-run latency of the orchestrator."""
    async def one_run():
        adapter = MockAdapter(min_delay=adapter_delay, max_delay=adapter_delay)
        eng = LoopEngine(adapter=adapter)
        loop = get_preset("executor_reviewer", max_iterations=iterations_per_run)
        run = RunState(workspace_id="w", loop_preset="executor_reviewer",
                       task="probe", provider=ProviderName.MOCK, model="",
                       max_iterations=iterations_per_run,
                       config={"convergence_enabled": False})
        t0 = time.perf_counter()
        final = await eng.run_loop(run, loop)
        return time.perf_counter() - t0, len(final.iterations)

    async def level(concurrency):
        MockAdapter.reset_iteration_count()
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one_run() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
        latencies = sorted(r[0] for r in results)
        total_iters = sum(r[1] for r in results)
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        return {
            "concurrency": concurrency,
            "wall_seconds": round(wall, 4),
            "total_iterations": total_iters,
            "throughput_iters_per_s": round(total_iters / wall, 1) if wall else 0.0,
            "latency_p50_s": round(median(latencies), 4),
            "latency_p95_s": round(p95, 4),
        }

    rows = [asyncio.run(level(c)) for c in concurrency_levels]
    return {
        "iterations_per_run": iterations_per_run,
        "adapter_delay_s": adapter_delay,
        "rows": rows,
    }


def run_all() -> dict:
    """Run every experiment and return a single results dictionary."""
    return {
        "consensus_ablation": consensus_ablation(),
        "role_overlap": role_overlap(),
        "convergence_profile": convergence_profile(),
        "load_benchmark": load_benchmark(),
    }
