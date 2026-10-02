"""E6: configurable stopping rules, persisted stop reasons, stop_conditions,
executor-only similarity, and convergence re-seeding on resume."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from backend.adapters.mock import MockAdapter
from backend.models import (
    IterationResult,
    IterationStatus,
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    ReviewCritique,
    ReviewResult,
    RunState,
    RunStatus,
    SeverityLevel,
)
from backend.orchestrator.convergence import (
    ALL_SIGNALS,
    DEFAULT_STOPPING_RULE,
    ConvergenceDetector,
    StoppingRule,
    grade_at_least,
    replay_stopping_rule,
)
from backend.orchestrator.dsl import parse_loop_yaml, parse_stop_conditions
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset


@pytest.fixture(autouse=True)
def _reset_mock():
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    yield
    MockAdapter.fixed_iteration = None


# ── StoppingRule ─────────────────────────────────────────────────────


def test_default_rule_is_the_current_behaviour():
    r = DEFAULT_STOPPING_RULE
    assert (r.window, r.min_iterations, r.similarity_threshold) == (3, 4, 0.9)
    assert r.signals == ALL_SIGNALS
    assert r.rule == "any" and r.require_no_critical is False
    assert StoppingRule.from_config({}) is DEFAULT_STOPPING_RULE
    assert ConvergenceDetector().rule == DEFAULT_STOPPING_RULE


def test_rule_from_config_parses_all_keys():
    r = StoppingRule.from_config({
        "convergence_window": "2",
        "convergence_min_iterations": 6,
        "convergence_similarity_threshold": "0.8",
        "convergence_signals": "output_similar, grade_stable",
        "convergence_rule": "k_of_n",
        "convergence_k": 2,
        "convergence_require_no_critical": "true",
    })
    assert r.window == 2 and r.min_iterations == 6 and r.similarity_threshold == 0.8
    assert r.signals == ("grade_stable", "output_similar")  # canonical order
    assert r.rule == "k_of_n" and r.k == 2 and r.require_no_critical is True


@pytest.mark.parametrize("cfg", [
    {"convergence_rule": "most"},
    {"convergence_signals": ["grade_stable", "vibes"]},
    {"convergence_signals": []},
    {"convergence_window": 0},
    {"convergence_similarity_threshold": 1.5},
    {"convergence_rule": "k_of_n", "convergence_k": 4},
])
def test_invalid_rules_rejected(cfg):
    with pytest.raises(ValueError):
        StoppingRule.from_config(cfg)


def _review(grade, crit=0, high=0):
    cs = [ReviewCritique(severity=SeverityLevel.CRITICAL, category="x", description=f"c{i}")
          for i in range(crit)]
    cs += [ReviewCritique(severity=SeverityLevel.HIGH, category="x", description=f"h{i}")
           for i in range(high)]
    return ReviewResult(overall_grade=grade, critiques=cs)


def test_rule_all_requires_every_enabled_signal():
    rule = StoppingRule(window=2, min_iterations=0, rule="all",
                        signals=("grade_stable", "no_critical_high"))
    d = ConvergenceDetector(rule=rule)
    d.record_iteration(1, "reviewer", "", _review("B", high=1))
    d.record_iteration(2, "reviewer", "", _review("B", high=1))
    assert not d.evaluate().converged  # grade stable, but HIGH present
    d.record_iteration(3, "reviewer", "", _review("B"))
    d.record_iteration(4, "reviewer", "", _review("B"))
    dec = d.evaluate()
    assert dec.converged and dec.stop_reason == "converged:grade_stable+no_critical_high"


def test_rule_k_of_n():
    rule = StoppingRule(window=2, min_iterations=0, rule="k_of_n", k=2)
    d = ConvergenceDetector(rule=rule)
    for i in range(1, 4):
        d.record_iteration(i, "reviewer", "", _review("A", high=1))
    assert not d.evaluate().converged  # only grade_stable holds
    d.record_iteration(4, "executor", "same text")
    d.record_iteration(5, "executor", "same text")
    d.record_iteration(6, "executor", "same text")
    dec = d.evaluate()
    assert dec.converged and set(dec.signals) == {"grade_stable", "output_similar"}


def test_signal_subset_ignores_disabled_signals():
    rule = StoppingRule(window=2, min_iterations=0, signals=("output_similar",))
    d = ConvergenceDetector(rule=rule)
    for i in range(1, 5):
        d.record_iteration(i, "reviewer", "", _review("A"))
    assert not d.evaluate().converged


def test_require_no_critical_blocks_convergence():
    rule = StoppingRule(window=2, min_iterations=0, require_no_critical=True,
                        signals=("grade_stable",))
    d = ConvergenceDetector(rule=rule)
    d.record_iteration(1, "reviewer", "", _review("D", crit=1))
    d.record_iteration(2, "reviewer", "", _review("D", crit=1))
    dec = d.evaluate()
    assert not dec.converged and dec.blocked_by == "critical"
    d.record_iteration(3, "reviewer", "", _review("D", high=2))
    d.record_iteration(4, "reviewer", "", _review("D", high=2))
    assert d.evaluate().converged


def test_similarity_uses_consecutive_executor_outputs_only():
    d = ConvergenceDetector(window_size=2, min_iterations=0)
    d.record_iteration(1, "executor", "alpha beta gamma")
    d.record_iteration(2, "consensus_merger", "totally different feedback text")
    d.record_iteration(3, "idea_generator", "proposals proposals")
    d.record_iteration(4, "adversarial", "attack attack")
    d.record_iteration(5, "executor", "alpha beta gamma")
    d.record_iteration(6, "executor", "alpha beta gamma")
    assert d.get_metrics()["similarity_scores"] == [1.0, 1.0]
    assert d.evaluate().stop_reason == "converged:output_similar"


def test_incomplete_grade_is_never_recorded():
    d = ConvergenceDetector(window_size=2, min_iterations=0)
    d.record_review("INCOMPLETE", 0, 0)
    d.record_review("INCOMPLETE", 0, 0)
    assert d.get_metrics()["grade_history"] == []
    assert d.get_metrics()["critique_counts"] == []
    assert not d.evaluate().converged


def test_grade_at_least():
    assert grade_at_least("B", "B") and grade_at_least("A-", "B+")
    assert not grade_at_least("B-", "B")
    assert not grade_at_least("INCOMPLETE", "F") and not grade_at_least("", "F")
    assert grade_at_least("F", "F")


def test_replay_stopping_rule_offline():
    events = [
        {"iteration": 1, "role": "executor", "output": "x y z"},
        {"iteration": 2, "role": "consensus_merger", "grade": "C", "critical": 1, "high": 1},
        {"iteration": 3, "role": "executor", "output": "x y z w"},
        {"iteration": 4, "role": "consensus_merger", "panel_failed": True},
        {"iteration": 5, "role": "executor", "output": "x y z w"},
        {"iteration": 6, "role": "consensus_merger", "grade": "B", "critical": 0, "high": 0},
        {"iteration": 7, "role": "executor", "output": "x y z w"},
        {"iteration": 8, "role": "consensus_merger", "grade": "B", "critical": 0, "high": 0},
    ]
    any_rule = replay_stopping_rule(events, StoppingRule(window=2, min_iterations=4))
    assert any_rule["stopped"] and any_rule["iteration"] == 7
    assert any_rule["stop_reason"] == "converged:output_similar"
    all_rule = replay_stopping_rule(events, StoppingRule(window=2, min_iterations=4, rule="all"))
    assert all_rule["stopped"] and all_rule["iteration"] == 8
    never = replay_stopping_rule(events, StoppingRule(window=3, min_iterations=4, rule="all"))
    assert never == {"stopped": False, "iteration": None, "stop_reason": "", "reason": ""}


# ── stop_conditions parsing ──────────────────────────────────────────


def test_parse_stop_conditions():
    sc = parse_stop_conditions([
        "user_stop", "budget_exceeded", "max_iterations:40", "budget_max_tokens=1000",
        "budget_max_cost:2.5", "grade_at_least:b+", "on_flag:done",
        "convergence_rule:all", "convergence_signals:grade_stable,no_critical_high",
        "convergence_require_no_critical:true", "all_artifacts_produced",
    ])
    assert sc.config == {
        "max_iterations": 40, "budget_max_tokens": 1000, "budget_max_cost": 2.5,
        "grade_at_least": "B+", "convergence_rule": "all",
        "convergence_signals": ["grade_stable", "no_critical_high"],
        "convergence_require_no_critical": True,
    }
    assert sc.flags == ["done"]
    assert sc.informational == ["user_stop", "budget_exceeded"]
    assert sc.unknown == ["all_artifacts_produced"]


@pytest.mark.parametrize("bad", ["max_iterations:abc", "grade_at_least:Z", "on_flag:",
                                 "convergence_signals:nope", "budget_max_cost:-1"])
def test_parse_stop_conditions_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        parse_stop_conditions([bad])


def test_yaml_mapping_form_of_stop_conditions():
    loop = parse_loop_yaml("""
name: t
nodes: [{id: e, role: executor}]
edges: [{source: e, target: e}]
stop_conditions:
  max_iterations: 7
  grade_at_least: B
  convergence_signals: [grade_stable]
""")
    assert loop.stop_conditions == ["max_iterations:7", "grade_at_least:B",
                                    "convergence_signals:grade_stable"]
    loop2 = parse_loop_yaml("""
name: t
nodes: [{id: e, role: executor}]
edges: [{source: e, target: e}]
stop_conditions:
  - user_stop
  - grade_at_least: A
""")
    assert loop2.stop_conditions == ["user_stop", "grade_at_least:A"]


# ── Engine: stop reasons ─────────────────────────────────────────────


def _er_loop(stop_conditions=None, max_iterations=50):
    loop = get_preset("executor_reviewer", max_iterations=max_iterations)
    if stop_conditions is not None:
        loop.stop_conditions = stop_conditions
    return loop


def _run(loop, max_iterations=6, config=None, adapter=None, cancel=None, events=None,
         run_state=None, resume_from=0):
    MockAdapter.reset_iteration_count()  # each run starts at mock iteration 1

    async def go():
        eng = LoopEngine(adapter=adapter or MockAdapter(min_delay=0, max_delay=0),
                         on_event=(lambda t, d: events.append((t, d))) if events is not None else None)
        run = run_state or RunState(workspace_id="w", loop_preset="x", task="t",
                                    provider=ProviderName.MOCK, model="",
                                    max_iterations=max_iterations, config=dict(config or {}))
        final = await eng.run_loop(run, loop, cancel_event=cancel, resume_from=resume_from)
        return final, eng

    return asyncio.run(go())


def test_stop_reason_max_iterations():
    final, _ = _run(_er_loop(), 3, {"convergence_enabled": False})
    assert final.status == RunStatus.COMPLETED and final.stop_reason == "max_iterations"


def test_stop_reason_budget_tokens_and_cost():
    final, _ = _run(_er_loop(), 50, {"convergence_enabled": False, "budget_max_tokens": 3000})
    assert final.stop_reason == "budget_tokens"
    assert final.total_tokens >= 3000
    final2, _ = _run(_er_loop(), 50, {"convergence_enabled": False, "budget_max_cost": 0.005})
    assert final2.stop_reason == "budget_cost"


def test_stop_reason_cancelled():
    ev = asyncio.Event()
    ev.set()
    final, _ = _run(_er_loop(), 5, {}, cancel=ev)
    assert final.status == RunStatus.STOPPED and final.stop_reason == "cancelled"


def test_stop_reason_failed_adapter():
    final, _ = _run(_er_loop(), 5, {}, adapter=MockAdapter(failure_rate=1.0, min_delay=0, max_delay=0))
    assert final.status == RunStatus.FAILED and final.stop_reason == "failed:adapter_failed"


def test_stop_reason_converged_and_event():
    events = []
    final, eng = _run(get_preset("reviewer_consensus", max_iterations=40), 40,
                      {"convergence_enabled": True}, events=events)
    assert final.stop_reason.startswith("converged:")
    conv = [d for t, d in events if t == "convergence_detected"][0]
    assert conv["stop_reason"] == final.stop_reason
    assert conv["signals"] and conv["metrics"]["rule"]["rule"] == "any"
    # Consensus loop: the similarity signal compares executor outputs only
    # (one score per pair of consecutive executor steps).
    n_exec = sum(1 for it in final.iterations if it.role == "executor")
    assert len(eng._convergence.get_metrics()["similarity_scores"]) == n_exec - 1
    finished = [d for t, d in events if t == "loop_finished"][0]
    assert finished["stop_reason"] == final.stop_reason


def test_convergence_rule_from_run_config_changes_stop():
    # The adaptive rule is off by default (calibrated default: revision budget).
    base, _ = _run(get_preset("reviewer_consensus", max_iterations=40), 40,
                   {"convergence_enabled": True})
    strict, _ = _run(get_preset("reviewer_consensus", max_iterations=40), 40,
                     {"convergence_enabled": True,
                      "convergence_signals": ["grade_stable"], "convergence_window": 3})
    assert base.stop_reason == "converged:output_similar"
    assert strict.stop_reason == "converged:grade_stable"
    assert len(strict.iterations) > len(base.iterations)


def test_grade_at_least_stop_condition_from_yaml_and_run_override():
    # Mock reviewer grades go C, B, B+, A- (global counter: executor/reviewer alternate).
    loop = _er_loop(["grade_at_least:B"])
    final, _ = _run(loop, 20, {"convergence_enabled": False})
    assert final.stop_reason == "stop_condition:grade_at_least"
    assert final.iterations[-1].role == "reviewer"
    assert final.iterations[-1].grade in ("B", "B+", "A-")
    # Run config overrides the loop YAML value (no revision budget, so the
    # run goes to its iteration cap).
    final2, _ = _run(_er_loop(["grade_at_least:B"]), 20,
                     {"convergence_enabled": False, "grade_at_least": "A+",
                      "revision_budget": None})
    assert final2.stop_reason == "max_iterations"


def test_max_iterations_stop_condition_is_a_cap_and_run_config_overrides():
    final, _ = _run(_er_loop(["max_iterations:3"]), 10, {"convergence_enabled": False})
    assert final.current_iteration == 3 and final.stop_reason == "max_iterations"
    final2, _ = _run(_er_loop(["max_iterations:3"]), 5,
                     {"convergence_enabled": False, "max_iterations": 5})
    assert final2.current_iteration == 5
    # The run's own max_iterations always applies.
    final3, _ = _run(_er_loop(["max_iterations:30"]), 4, {"convergence_enabled": False})
    assert final3.current_iteration == 4


def test_budget_stop_condition_from_loop():
    final, _ = _run(_er_loop(["budget_max_tokens:2500"]), 20, {"convergence_enabled": False})
    assert final.stop_reason == "budget_tokens"


def test_convergence_override_from_loop_stop_conditions():
    loop = get_preset("reviewer_consensus", max_iterations=40)
    loop.stop_conditions = ["convergence_enabled:false"]
    final, _ = _run(loop, 9, {})
    assert final.stop_reason == "max_iterations"
    # ...and the run config wins over the loop.
    final2, _ = _run(loop, 40, {"convergence_enabled": True})
    assert final2.stop_reason.startswith("converged:")


def test_on_flag_stop_condition():
    events = []
    final, _ = _run(_er_loop(["on_flag:halt"]), 10, {"flags": {"halt": True}}, events=events)
    assert final.stop_reason == "stop_condition:on_flag:halt"
    assert final.current_iteration == 0


def test_unknown_stop_conditions_reported_and_ignored():
    events = []
    final, _ = _run(_er_loop(["user_stop", "knowledge_graph_converged"]), 2,
                    {"convergence_enabled": False}, events=events)
    assert final.stop_reason == "max_iterations"
    ignored = [d for t, d in events if t == "stop_conditions_ignored"]
    assert ignored and ignored[0]["conditions"] == ["knowledge_graph_converged"]


def test_invalid_config_fails_run_cleanly():
    final, _ = _run(_er_loop(), 4, {"convergence_rule": "sometimes"})
    assert final.status == RunStatus.FAILED
    assert final.stop_reason == "failed:invalid_config"
    assert "convergence_rule" in final.error
    final2, _ = _run(_er_loop(["grade_at_least:Q"]), 4, {})
    assert final2.stop_reason == "failed:invalid_config"


def test_all_steps_conditional_and_off_terminates():
    loop = LoopDefinition(
        name="dead",
        nodes=[LoopNode(id="a", role="executor"), LoopNode(id="b", role="reviewer")],
        edges=[LoopEdge(source="a", target="b", condition="on_flag:x"),
               LoopEdge(source="b", target="a", condition="on_flag:x")],
    )
    # compile_loop emits a: always, then b gated by on_flag:x -> a runs, b never.
    final, _ = _run(loop, 3, {"convergence_enabled": False})
    assert final.stop_reason == "max_iterations"


# ── Resume re-seeding ────────────────────────────────────────────────


def _it(n, role, grade=None, crit=None, high=None, out="", feedback=None):
    return IterationResult(iteration_number=n, role=role, status=IterationStatus.COMPLETED,
                           started_at=datetime.utcnow(), completed_at=datetime.utcnow(),
                           output_summary=out, grade=grade, critical_count=crit,
                           high_count=high, feedback=feedback)


def test_resume_reseeds_grade_history_and_converges_immediately():
    history = [
        _it(1, "executor", out="draft one"), _it(2, "reviewer", "B", 0, 1, feedback="fb2"),
        _it(3, "executor", out="draft two"), _it(4, "reviewer", "B", 0, 1, feedback="fb4"),
        _it(5, "executor", out="draft three"), _it(6, "reviewer", "B", 0, 0, feedback="fb6"),
    ]
    run = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                   model="", max_iterations=20, status=RunStatus.PAUSED, current_iteration=6,
                   iterations=history, config={"convergence_enabled": True})
    run.started_at = datetime.utcnow()

    calls = []

    class Spy:
        name = "mock"

        def __init__(self):
            self.inner = MockAdapter(min_delay=0, max_delay=0)

        async def run(self, request):
            calls.append(request)
            return await self.inner.run(request)

    final, eng = _run(_er_loop(), run_state=run, resume_from=6, adapter=Spy())
    m = eng._convergence.get_metrics()
    assert m["grade_history"][:3] == ["B", "B", "B"]
    assert m["critique_counts"][:3] == [1, 1, 0]
    # Six stored steps = three executor/reviewer cycles, so the next step is
    # the cycle-3 adversarial review (every_k:3), exactly as in an
    # uninterrupted run.
    seventh = [i for i in final.iterations if i.iteration_number == 7][0]
    assert seventh.role == "adversarial"
    p = calls[0].prompt_bundle.user_prompt
    assert "ARTIFACT UNDER REVIEW" in p and "draft three" in p

    # Without a gated step the resumed step is the executor, and its prompt
    # carries the pending reviewer feedback and its previous submission.
    calls.clear()
    MockAdapter.reset_iteration_count()
    loop2 = LoopDefinition(
        name="er2",
        nodes=[LoopNode(id="e", role="executor"), LoopNode(id="r", role="reviewer")],
        edges=[LoopEdge(source="e", target="r"), LoopEdge(source="r", target="e")],
    )
    run2 = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                    model="", max_iterations=20, status=RunStatus.PAUSED, current_iteration=6,
                    iterations=[h.model_copy() for h in history],
                    config={"convergence_enabled": True})
    run2.started_at = datetime.utcnow()
    final2, _ = _run(loop2, run_state=run2, resume_from=6, adapter=Spy())
    assert final2.stop_reason == "converged:grade_stable"
    p = calls[0].prompt_bundle.user_prompt
    assert "fb6" in p and "draft three" in p and "fb4" not in p


def test_resume_without_stored_grades_reparses_reviews():
    history = [
        _it(1, "executor", out="d1"),
        _it(2, "reviewer", out="Overall Grade: C\n[HIGH] Missing control"),
        _it(3, "executor", out="d2"),
        _it(4, "reviewer", out="random words without any structure"),
    ]
    run = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                   model="", max_iterations=5, status=RunStatus.PAUSED, current_iteration=4,
                   iterations=history, config={"convergence_enabled": False})
    run.started_at = datetime.utcnow()
    final, eng = _run(_er_loop(), run_state=run, resume_from=4)
    m = eng._convergence.get_metrics()
    # The parseable legacy review is restored; the unparseable one is NOT "0 critiques".
    assert m["grade_history"][0] == "C"
    assert m["critique_counts"][0] == 1
    assert m["iteration_count"] >= 5
