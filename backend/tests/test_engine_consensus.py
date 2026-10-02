"""E1/E5 wiring of the consensus panel into the live engine.

The engine builds the panel with ``ConsensusReviewer.from_config`` (so every
consensus_* run-config key reaches the merger), passes model / effort /
allow_tools on every lens request, retries a failed panel once and fails the
run on a second failure (INCOMPLETE is never treated as a grade).
"""
from __future__ import annotations

import asyncio

import pytest

from backend.adapters.mock import MockAdapter
from backend.models import (
    AdapterRunResult,
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    RunState,
    RunStatus,
)
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset

LENSES = ("reviewer", "adversarial_reviewer", "bio_plausibility_checker")


@pytest.fixture(autouse=True)
def _reset_mock():
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    yield
    MockAdapter.fixed_iteration = None


class Scripted:
    """Mock-backed adapter whose lens calls can be made to fail per attempt."""

    name = "scripted"

    def __init__(self, fail_lens_calls=0, fail_roles=LENSES):
        self.inner = MockAdapter(min_delay=0, max_delay=0)
        self.fail_lens_calls = fail_lens_calls
        self.fail_roles = set(fail_roles)
        self.lens_calls = 0
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        role = request.prompt_bundle.variables.get("role", "")
        if role in LENSES:
            self.lens_calls += 1
            if self.lens_calls <= self.fail_lens_calls and role in self.fail_roles:
                return AdapterRunResult(success=False, error="auth failed", exit_code=1,
                                        input_tokens=10, output_tokens=0, token_usage=10)
        res = await self.inner.run(request)
        res.provider = "scripted"
        res.model = request.model or "default-model"
        res.cli_version = "9.9.9"
        return res

    async def smoke_test(self):
        return {"status": "ok"}

    def is_available(self):
        return True


def _run(adapter, max_iterations=4, config=None, loop=None, model=""):
    events = []

    async def go():
        eng = LoopEngine(adapter=adapter, on_event=lambda t, d: events.append((t, d)))
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus", task="t",
                       provider=ProviderName.MOCK, model=model, max_iterations=max_iterations,
                       config={"convergence_enabled": False, "max_retries": 0,
                               **(config or {})})
        final = await eng.run_loop(run, loop or get_preset("reviewer_consensus",
                                                           max_iterations=max_iterations))
        return final, eng

    final, eng = asyncio.run(go())
    return final, eng, events


def test_from_config_keys_reach_the_merger():
    final, _, events = _run(Scripted(), 2, config={
        "consensus_similarity_method": "tfidf",
        "consensus_similarity_threshold": 0.42,
        "consensus_threshold": 3,
    })
    report = [d["report"] for t, d in events if t == "consensus_merged"][0]
    assert report["similarity_method"] == "tfidf"
    assert report["similarity_threshold"] == pytest.approx(0.42)
    assert report["consensus_threshold"] == 3
    it = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert it.consensus_report["similarity_method"] == "tfidf"


def test_panel_failure_retried_once_then_run_fails():
    adapter = Scripted(fail_lens_calls=100)
    final, eng, events = _run(adapter, 4)
    assert final.status == RunStatus.FAILED
    assert final.stop_reason == "failed:consensus_panel_failed"
    assert "Consensus panel failed at iteration 2 after 2 attempt(s)" in final.error
    assert "auth failed" in final.error
    # Two panel attempts (3 lenses each), then no further steps.
    assert adapter.lens_calls == 6
    failed_events = [d for t, d in events if t == "consensus_panel_failed"]
    assert [e["will_retry"] for e in failed_events] == [True, False]
    merged = [d for t, d in events if t == "consensus_merged"]
    assert len(merged) == 2 and all(d["report"]["panel_failed"] for d in merged)
    # INCOMPLETE never entered the convergence history.
    metrics = eng._convergence.get_metrics()
    assert metrics["grade_history"] == [] and metrics["critique_counts"] == []
    it = final.iterations[-1]
    assert it.role == "consensus_merger" and it.status.value == "failed"
    assert len(it.consensus_report["attempts"]) == 2
    # Failed attempts' tokens are still accounted.
    assert it.token_usage >= 6 * 10


def test_panel_failure_then_success_continues():
    adapter = Scripted(fail_lens_calls=3)  # first attempt: all three lenses fail
    final, eng, events = _run(adapter, 4)
    assert final.status == RunStatus.COMPLETED
    assert final.stop_reason == "max_iterations"
    it = [i for i in final.iterations if i.role == "consensus_merger"][0]
    attempts = it.consensus_report["attempts"]
    assert [a["panel_failed"] for a in attempts] == [True, False]
    assert it.grade and it.grade != "INCOMPLETE"
    assert eng._convergence.get_metrics()["grade_history"]  # recorded only once, a real grade
    assert len(eng._convergence.get_metrics()["grade_history"]) == 2


def test_partial_panel_proceeds_with_partial_feedback():
    adapter = Scripted(fail_lens_calls=100, fail_roles=("bio_plausibility_checker",))
    final, eng, events = _run(adapter, 3)
    assert final.status == RunStatus.COMPLETED
    it = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert "PANEL STATUS: PARTIAL" in it.feedback
    assert it.consensus_report["panel_partial"] is True
    # A partial panel is retried once, for the failed lens only.
    attempts = it.consensus_report["attempts"]
    assert len(attempts) == 2
    assert attempts[1]["lenses_called"] == ["bio_plausibility_checker"]
    assert adapter.lens_calls == 4  # 3 lenses + 1 retried lens
    # Still partial after the retry: kept as feedback, but the grade is not
    # a review observation (default policy no_stop).
    assert it.consensus_report["grade_valid"] is False
    assert it.consensus_report["stop_eligible"] is False
    assert eng._convergence.get_metrics()["grade_history"] == []


def test_lens_requests_carry_model_effort_and_no_tools():
    adapter = Scripted()
    final, _, _ = _run(adapter, 2, config={"reasoning_effort": "medium"}, model="gpt-5.5")
    lens_reqs = [r for r in adapter.requests if r.prompt_bundle.variables.get("role") in LENSES]
    assert len(lens_reqs) == 3
    for r in lens_reqs:
        assert r.model == "gpt-5.5" and r.reasoning_effort == "medium"
        assert r.allow_tools is False
    it = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert it.provider == "scripted"
    assert it.model == "gpt-5.5"
    assert it.cli_version == "9.9.9"
    assert it.reasoning_effort == "medium"


def test_consensus_token_split_accounted():
    final, _, _ = _run(Scripted(), 2)
    it = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert it.input_tokens > 0 and it.output_tokens > 0
    assert it.token_usage == it.input_tokens + it.output_tokens
    assert final.total_tokens == sum(i.token_usage for i in final.iterations)
    assert final.total_input_tokens == sum(i.input_tokens for i in final.iterations)
    assert final.total_output_tokens == sum(i.output_tokens for i in final.iterations)


def _custom_consensus_loop(merger_config):
    return LoopDefinition(
        name="cons",
        nodes=[
            LoopNode(id="e", role="executor", prompt_ref="executor/mi_executor"),
            LoopNode(id="r", role="reviewer", prompt_ref="reviewer/mi_reviewer"),
            LoopNode(id="b", role="bio_plausibility_checker",
                     prompt_ref="biological_plausibility/bio_plausibility_checker"),
            LoopNode(id="m", role="consensus_merger", config=merger_config),
        ],
        edges=[
            LoopEdge(source="e", target="r"), LoopEdge(source="e", target="b"),
            LoopEdge(source="r", target="m"), LoopEdge(source="b", target="m"),
            LoopEdge(source="m", target="e"),
        ],
    )


def test_merger_node_similarity_threshold_is_honoured():
    loop = _custom_consensus_loop({"similarity_method": "jaccard", "similarity_threshold": 0.77})
    _, _, events = _run(Scripted(), 2, loop=loop)
    report = [d["report"] for t, d in events if t == "consensus_merged"][0]
    assert report["similarity_method"] == "jaccard"
    assert report["similarity_threshold"] == pytest.approx(0.77)


def test_run_config_overrides_merger_node_threshold():
    loop = _custom_consensus_loop({"similarity_method": "jaccard", "similarity_threshold": 0.77})
    _, _, events = _run(Scripted(), 2, loop=loop,
                        config={"consensus_similarity_threshold": 0.31})
    report = [d["report"] for t, d in events if t == "consensus_merged"][0]
    assert report["similarity_threshold"] == pytest.approx(0.31)


def test_node_threshold_dropped_when_run_selects_other_method():
    loop = _custom_consensus_loop({"similarity_method": "jaccard", "similarity_threshold": 0.77})
    _, _, events = _run(Scripted(), 2, loop=loop,
                        config={"consensus_similarity_method": "tfidf"})
    report = [d["report"] for t, d in events if t == "consensus_merged"][0]
    assert report["similarity_method"] == "tfidf"
    assert report["similarity_threshold"] == pytest.approx(0.1)  # tfidf default (calibrated)


def test_bundled_yaml_merger_uses_the_llm_default():
    from backend.orchestrator.presets import BUNDLED_LOOPS_DIR
    loop = get_preset(f"custom:{BUNDLED_LOOPS_DIR / 'reviewer_consensus.yaml'}", max_iterations=2)
    adapter = Scripted()
    _, _, events = _run(adapter, 2, loop=loop)
    report = [d["report"] for t, d in events if t == "consensus_merged"][0]
    assert report["similarity_method"] == "llm"
    assert report["similarity_threshold"] is None
    assert report["similarity_fallback"] is None   # the mock adjudicator answered
    roles = [r.prompt_bundle.variables.get("role") for r in adapter.requests]
    assert roles.count("consensus_adjudicator") == 1


def test_invalid_consensus_config_fails_run():
    final, _, _ = _run(Scripted(), 2, config={"consensus_similarity_method": "nonsense"})
    assert final.status == RunStatus.FAILED
    assert final.stop_reason == "failed:invalid_config"
    assert "consensus" in final.error.lower()


def test_consensus_gate_blocks_grade_stop():
    # Tier-1 mock critiques include CRITICAL items -> the gate holds the loop open.
    MockAdapter.fixed_iteration = 1
    final, _, events = _run(Scripted(), 4, config={"consensus_gate": True,
                                                   "grade_at_least": "F",
                                                   "convergence_enabled": True})
    assert final.stop_reason == "max_iterations"
    final2, _, _ = _run(Scripted(), 4, config={"grade_at_least": "F"})
    assert final2.stop_reason == "stop_condition:grade_at_least"
