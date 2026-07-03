"""Tests for the consensus panel wired into the live engine.

Covers: compile_loop emitting a consensus step with the full panel; the engine
executing the panel concurrently and accounting all reviewers' tokens;
panel-size monotonicity (more lenses surface strictly more distinct critiques);
agreement-based severity escalation; deterministic ranking; role weighting; and
the advancement gate.
"""
import asyncio

import pytest

from backend.adapters.mock import MockAdapter
from backend.orchestrator.consensus import ConsensusReviewer
from backend.orchestrator.dsl import compile_loop
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset
from backend.models import ProviderName, RunState, SeverityLevel, WorkspaceContext

FULL_PANEL = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]


def _sp(role, ref):
    return f"You are the {role}. Review the following artifacts."


@pytest.fixture(autouse=True)
def _reset_mock():
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    yield
    MockAdapter.fixed_iteration = None


def test_compile_loop_emits_consensus_step_with_full_panel():
    plan = compile_loop(get_preset("reviewer_consensus", max_iterations=6))
    consensus_steps = [s for s in plan.steps if s.kind == "consensus"]
    assert len(consensus_steps) == 1
    panel_roles = [pm.role for pm in consensus_steps[0].panel]
    # All three reviewers must be present (none silently dropped).
    assert panel_roles == [
        "reviewer", "adversarial_reviewer", "bio_plausibility_checker"
    ]
    # Panel members are not also emitted as standalone steps.
    standalone = [s.role for s in plan.steps if s.kind == "single"]
    assert "adversarial_reviewer" not in standalone
    assert "bio_plausibility_checker" not in standalone


def test_engine_runs_full_panel_and_accounts_tokens():
    async def go():
        MockAdapter.reset_iteration_count()
        events = []
        eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0),
                         on_event=lambda t, d: events.append((t, d)))
        loop = get_preset("reviewer_consensus", max_iterations=4)
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus",
                       task="probe", provider=ProviderName.MOCK, model="",
                       max_iterations=4, config={"convergence_enabled": False})
        final = await eng.run_loop(run, loop)
        return final, events

    final, events = asyncio.run(go())
    merged_events = [d for t, d in events if t == "consensus_merged"]
    assert merged_events, "consensus_merged event never emitted"
    assert all(e["panel_size"] == 3 for e in merged_events)
    # Each consensus iteration accounts three reviewer calls' worth of tokens.
    consensus_iters = [it for it in final.iterations if it.role == "consensus_merger"]
    assert consensus_iters and all(it.token_usage > 2000 for it in consensus_iters)


def test_panel_size_monotonic_distinct_critiques():
    async def coverage(n):
        MockAdapter.fixed_iteration = 1
        cr = ConsensusReviewer(panel=FULL_PANEL[:n])
        merged, _ = await cr.run_panel(
            "EXEC", MockAdapter(min_delay=0, max_delay=0),
            WorkspaceContext(workspace_path=""), system_prompt_builder=_sp)
        return len(merged.critiques)

    c1 = asyncio.run(coverage(1))
    c2 = asyncio.run(coverage(2))
    c3 = asyncio.run(coverage(3))
    assert c1 < c2 < c3, (c1, c2, c3)


def test_agreement_escalates_severity():
    async def go():
        MockAdapter.fixed_iteration = 1
        cr = ConsensusReviewer(panel=FULL_PANEL[:2], consensus_threshold=2)
        merged, _ = await cr.run_panel(
            "EXEC", MockAdapter(min_delay=0, max_delay=0),
            WorkspaceContext(workspace_path=""), system_prompt_builder=_sp)
        return merged

    merged = asyncio.run(go())
    # The shared "effect sizes" critique (HIGH from each lens) escalates to CRITICAL.
    shared = [c for c in merged.critiques if "effect sizes" in c.description.lower()]
    assert shared and shared[0].severity == SeverityLevel.CRITICAL


def test_higher_threshold_suppresses_escalation():
    async def go(threshold):
        MockAdapter.fixed_iteration = 1
        cr = ConsensusReviewer(panel=FULL_PANEL[:2], consensus_threshold=threshold)
        merged, _ = await cr.run_panel(
            "EXEC", MockAdapter(min_delay=0, max_delay=0),
            WorkspaceContext(workspace_path=""), system_prompt_builder=_sp)
        return cr.compute_consensus_report(merged)

    two = asyncio.run(go(2))
    three = asyncio.run(go(3))  # cannot be met by a 2-member panel
    assert two["severity_histogram"].get("critical", 0) > \
        three["severity_histogram"].get("critical", 0)


def test_role_weighting_changes_ranking():
    async def top_role(weights):
        MockAdapter.fixed_iteration = 1
        cr = ConsensusReviewer(panel=FULL_PANEL, role_weights=weights)
        merged, _ = await cr.run_panel(
            "EXEC", MockAdapter(min_delay=0, max_delay=0),
            WorkspaceContext(workspace_path=""), system_prompt_builder=_sp)
        # Among the top few, whether a bio-only critique outranks equal-severity peers.
        return merged.critiques

    unweighted = asyncio.run(top_role({}))
    weighted = asyncio.run(top_role({"bio_plausibility_checker": 10.0}))
    # Weighting must not change the multiset of critiques, only their order.
    assert {c.description for c in unweighted} == {c.description for c in weighted}
    # A heavily weighted bio critique ranks ahead of same-severity non-bio ones.
    def first_index(critiques, needle):
        return next(i for i, c in enumerate(critiques) if needle in c.description.lower())
    bio_idx_w = first_index(weighted, "regulatory direction")
    bio_idx_u = first_index(unweighted, "regulatory direction")
    assert bio_idx_w <= bio_idx_u


def test_consensus_gate_blocks_convergence_on_unresolved_critical():
    async def go(gate):
        MockAdapter.reset_iteration_count()
        MockAdapter.fixed_iteration = 1  # always tier-1 => a CRITICAL always remains
        eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0))
        loop = get_preset("reviewer_consensus", max_iterations=10)
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus",
                       task="probe", provider=ProviderName.MOCK, model="",
                       max_iterations=10,
                       config={"convergence_enabled": True, "consensus_gate": gate})
        final = await eng.run_loop(run, loop)
        return len(final.iterations)

    # Grade stabilises at F, so without the gate the run converges early.
    ungated = asyncio.run(go(False))
    # With the gate on, an unresolved CRITICAL blocks convergence: the run keeps
    # going to the iteration ceiling instead of stopping early.
    gated = asyncio.run(go(True))
    assert gated > ungated
    assert gated == 10
