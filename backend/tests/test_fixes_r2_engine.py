"""Regression tests: engine stopping, resume, gating, accounting and run
lifecycle (code-review findings engine-1..11, consensus-3, tests-claims-3,
-5, -7, -8)."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from backend.adapters.mock import MockAdapter
from backend.models import (
    AdapterRunResult,
    IterationResult,
    IterationStatus,
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    RunState,
    RunStatus,
)
from backend.orchestrator.convergence import StoppingRule, replay_stopping_rule
from backend.orchestrator.dsl import compile_loop
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset
from backend.tests.fixes_r2_helpers import LENSES, ScriptedReviewAdapter, json_block, run_engine

EMPTY = json_block([])
HIGH = json_block([{"severity": "high", "description": "Missing permutation null for AUROC."}])
CRIT = json_block([{"severity": "critical", "description": "Leakage between folds."}])


def _fail(error="[auth] token expired"):
    return AdapterRunResult(success=False, error=error, exit_code=1)


# ── engine-1 / consensus-3 / tests-claims-3: partial panels ───────────


def _two_lenses_fail(role, req, n):
    return EMPTY if role == "bio_plausibility_checker" else _fail()


def test_partial_panel_never_stops_on_grade_at_least():
    adapter = ScriptedReviewAdapter(_two_lenses_fail, roles=LENSES)
    events: list = []
    final, eng = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=6), 6,
                            {"grade_at_least": "B", "convergence_enabled": False}, events=events)
    assert final.stop_reason == "max_iterations"
    assert eng._convergence.get_metrics()["grade_history"] == []
    cons = [i for i in final.iterations if i.role == "consensus_merger"]
    assert cons and all(i.consensus_report["grade_valid"] is False for i in cons)
    assert all(len(i.consensus_report["attempts"]) == 2 for i in cons)
    # the two failed lenses were retried; the usable one was not called again
    assert adapter.calls["bio_plausibility_checker"] == len(cons)
    assert adapter.calls["reviewer"] == 2 * len(cons)
    assert any(t == "consensus_panel_partial" for t, _ in events)


def test_partial_panel_never_converges_by_default():
    adapter = ScriptedReviewAdapter(_two_lenses_fail, roles=LENSES)
    # review-based signals only (the executor-output signal is unaffected);
    # "by default" = the default partial policy no_stop. Convergence itself is
    # off by default, so it is enabled explicitly.
    final, eng = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=12), 12,
                            {"convergence_enabled": True,
                             "convergence_signals": "grade_stable,no_critical_high"})
    assert not (final.stop_reason or "").startswith("converged")
    assert eng._convergence.get_metrics()["grade_history"] == []


def test_lens_timeout_dropping_the_only_high_cannot_stop_the_run():
    def fn(role, req, n):
        if role == "reviewer":
            return _fail("Timeout after 1s")  # transient, retried once, still fails
        return EMPTY

    adapter = ScriptedReviewAdapter(fn, roles=LENSES)
    final, _ = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=6), 6,
                          {"grade_at_least": "B", "convergence_enabled": False})
    assert final.stop_reason == "max_iterations"


def test_partial_panel_recovered_by_retry_counts():
    def fn(role, req, n):
        if role == "reviewer" and n == 1:
            return _fail()
        return EMPTY

    adapter = ScriptedReviewAdapter(fn, roles=LENSES)
    final, eng = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=6), 6,
                            {"grade_at_least": "B", "convergence_enabled": False})
    assert final.stop_reason == "stop_condition:grade_at_least"
    cons = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert cons.consensus_report["grade_valid"] is True
    assert cons.consensus_report["panel_partial"] is False


def test_partial_policy_count_and_fail():
    adapter = ScriptedReviewAdapter(_two_lenses_fail, roles=LENSES)
    final, _ = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=6), 6,
                          {"grade_at_least": "B", "convergence_enabled": False,
                           "consensus_partial_policy": "count"})
    assert final.stop_reason == "stop_condition:grade_at_least"
    adapter2 = ScriptedReviewAdapter(_two_lenses_fail, roles=LENSES)
    final2, _ = run_engine(adapter2, get_preset("reviewer_consensus", max_iterations=6), 6,
                           {"consensus_partial_policy": "fail"})
    assert final2.stop_reason == "failed:consensus_panel_partial"
    final3, _ = run_engine(ScriptedReviewAdapter(_two_lenses_fail, roles=LENSES),
                           get_preset("reviewer_consensus"), 4,
                           {"consensus_partial_policy": "sometimes"})
    assert final3.stop_reason == "failed:invalid_config"


def test_resume_does_not_reseed_partial_panel_grades():
    hist = [
        IterationResult(iteration_number=1, role="executor", status=IterationStatus.COMPLETED,
                        output_summary="draft"),
        IterationResult(iteration_number=2, role="consensus_merger",
                        status=IterationStatus.COMPLETED, grade="A", critical_count=0,
                        high_count=0, feedback="fb",
                        consensus_report={"panel_partial": True, "grade_valid": False,
                                          "unresolved_critical": 0}),
    ]
    run = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                   model="", max_iterations=3, status=RunStatus.PAUSED, current_iteration=2,
                   iterations=hist, config={"convergence_enabled": False})
    final, eng = run_engine(ScriptedReviewAdapter(lambda r, q, n: EMPTY, roles=LENSES),
                            get_preset("reviewer_consensus"), run_state=run, resume_from=2)
    assert eng._convergence.get_metrics()["grade_history"] == []


def test_replay_skips_partial_panels_and_applies_gate():
    rule = StoppingRule(window=2, min_iterations=0, signals=("grade_stable",))
    events = [
        {"iteration": 1, "role": "executor", "output": "x"},
        {"iteration": 2, "role": "consensus_merger", "grade": "A", "critical": 0, "high": 0},
        {"iteration": 3, "role": "executor", "output": "y"},
        {"iteration": 4, "role": "consensus_merger", "grade": "A", "critical": 0, "high": 0,
         "panel_partial": True},
    ]
    assert replay_stopping_rule(events, rule)["stopped"] is False
    assert replay_stopping_rule(events, rule, include_partial=True)["iteration"] == 4
    gated = [
        {"iteration": 1, "role": "consensus_merger", "grade": "D", "critical": 1, "high": 0},
        {"iteration": 2, "role": "consensus_merger", "grade": "D", "critical": 1, "high": 0},
        {"iteration": 3, "role": "executor", "output": "z"},
    ]
    assert replay_stopping_rule(gated, rule)["iteration"] == 2
    assert replay_stopping_rule(gated, rule, gate=True)["stopped"] is False


# ── engine-3: executor tools off under verified execution ─────────────


def _kwargs(config):
    eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0))
    run = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                   model="gpt-5.5", config=config)
    eng._configure(run, get_preset("executor_reviewer"))
    return eng, eng._request_kwargs(run, executor=True), eng._request_kwargs(run, executor=False)


def test_code_execution_disables_executor_tools_by_default():
    eng, ex, rv = _kwargs({"code_execution_enabled": True})
    assert ex["allow_tools"] is False and rv["allow_tools"] is False
    assert eng.tool_settings()["executor_allow_tools"] is False
    eng, ex, _ = _kwargs({})
    assert ex["allow_tools"] is True
    eng, ex, _ = _kwargs({"code_execution_enabled": True, "executor_allow_tools": True})
    assert ex["allow_tools"] is True
    assert eng.tool_settings()["executor_tools_with_code_execution"] is True


def test_executor_tools_with_code_execution_is_flagged_in_events():
    events: list = []
    run_engine(MockAdapter(min_delay=0, max_delay=0), get_preset("executor_reviewer"), 1,
               {"code_execution_enabled": True, "executor_allow_tools": True,
                "convergence_enabled": False}, events=events)
    assert any(t == "executor_tools_with_code_execution" for t, _ in events)


def test_adapter_options_reach_requests():
    eng, ex, rv = _kwargs({"adapter_options": {"codex_ignore_user_config": False}})
    assert ex["adapter_options"] == rv["adapter_options"] == {"codex_ignore_user_config": False}


# ── engine-4: no second engine on one run ─────────────────────────────


@pytest.mark.asyncio
async def test_resume_refused_while_run_active(client, tmp_path):
    from backend.orchestrator import runner

    ws = await client.post("/api/workspaces", json={"name": "w4", "path": str(tmp_path / "w4")})
    create = await client.post("/api/runs", json={"workspace_id": ws.json()["id"], "task": "t"})
    run_id = create.json()["run_id"]
    await client.post(f"/api/runs/{run_id}/stop")
    task = runner._active_tasks.get(run_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), 30)
    # simulate the stopped engine still finishing its in-flight step
    runner._active_runs[run_id] = asyncio.Event()
    try:
        resp = await client.post(f"/api/runs/{run_id}/resume")
        assert resp.status_code == 409
        # a direct second execute_run is refused as well
        await runner.execute_run(run_id)
        assert runner._active_runs.get(run_id) is not None
    finally:
        runner._active_runs.pop(run_id, None)


# ── engine-5: consensus gate restored on resume ───────────────────────


def _crit_panel():
    return ScriptedReviewAdapter(lambda role, req, n: CRIT, roles=LENSES)


def test_consensus_gate_survives_resume(tmp_path):
    # Convergence on (off by default) so the gate has something to block; no
    # revision budget, so only the iteration cap ends the run.
    cfg = {"consensus_gate": True, "convergence_enabled": True,
           "convergence_signals": "grade_stable", "revision_budget": None}
    full, _ = run_engine(_crit_panel(), get_preset("reviewer_consensus", max_iterations=12), 12, cfg)
    assert full.stop_reason == "max_iterations"
    first, _ = run_engine(_crit_panel(), get_preset("reviewer_consensus", max_iterations=6), 6, cfg)
    assert first.current_iteration == 6
    resumed_state = RunState(workspace_id="w", loop_preset="x", task="t",
                             provider=ProviderName.MOCK, model="", max_iterations=12,
                             status=RunStatus.PAUSED, current_iteration=6,
                             iterations=[i.model_copy() for i in first.iterations], config=cfg)
    final, _ = run_engine(_crit_panel(), get_preset("reviewer_consensus", max_iterations=12),
                          run_state=resumed_state, resume_from=6)
    assert final.stop_reason == "max_iterations"


# ── engine-6 / engine-7: resume position and every_k by cycle ─────────


def _er_roles(n_iter, config=None):
    final, eng = run_engine(MockAdapter(min_delay=0, max_delay=0),
                            get_preset("executor_reviewer", max_iterations=n_iter), n_iter,
                            {"convergence_enabled": False, **(config or {})})
    return final, eng


def test_every_k_runs_every_kth_cycle():
    final, _ = _er_roles(20, {"revision_budget": None})
    adv = [i.iteration_number for i in final.iterations if i.role == "adversarial"]
    assert adv == [7, 14]


@pytest.mark.parametrize("cut,next_role", [(4, "executor"), (6, "adversarial"), (7, "executor")])
def test_resume_continues_the_plan_position(cut, next_role):
    full, full_eng = _er_roles(10)
    first, _ = _er_roles(cut)
    state = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                     model="", max_iterations=10, status=RunStatus.PAUSED,
                     current_iteration=cut, iterations=[i.model_copy() for i in first.iterations],
                     config={"convergence_enabled": False})
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    final, eng = run_engine(MockAdapter(min_delay=0, max_delay=0),
                            get_preset("executor_reviewer", max_iterations=10),
                            run_state=state, resume_from=cut)
    roles = [i.role for i in sorted(final.iterations, key=lambda i: i.iteration_number)]
    assert roles[cut] == next_role
    assert roles == [i.role for i in sorted(full.iterations, key=lambda i: i.iteration_number)]
    nums = [i.iteration_number for i in final.iterations]
    assert len(nums) == len(set(nums))


# ── engine-8: on_flag edges and flags ─────────────────────────────────


def test_conditional_back_edge_is_not_an_extra_executor_step():
    plan = compile_loop(get_preset("custom:loops/research_followups.yaml"))
    assert [(s.role, s.condition) for s in plan.steps] == [
        ("executor", "always"), ("reviewer", "always"),
        ("idea_generator", "on_flag:milestone_complete")]
    assert any("loop-back" in w for w in plan.warnings)
    final, _ = run_engine(MockAdapter(min_delay=0, max_delay=0),
                          get_preset("custom:loops/research_followups.yaml"), 8,
                          {"convergence_enabled": False, "flags": {"needs_revision": True}})
    roles = [i.role for i in final.iterations]
    assert "idea_generator" not in roles
    assert all(not (a == b == "executor") for a, b in zip(roles, roles[1:]))
    final2, _ = run_engine(MockAdapter(min_delay=0, max_delay=0),
                           get_preset("custom:loops/research_followups.yaml"), 6,
                           {"convergence_enabled": False, "flags": {"milestone_complete": True}})
    assert [i.role for i in final2.iterations][:3] == ["executor", "reviewer", "idea_generator"]


def test_flag_strings_parsed_like_booleans():
    loop = LoopDefinition(
        name="f",
        nodes=[LoopNode(id="e", role="executor"), LoopNode(id="r", role="reviewer"),
               LoopNode(id="k", role="knowledge_extractor")],
        edges=[LoopEdge(source="e", target="r"), LoopEdge(source="r", target="e"),
               LoopEdge(source="r", target="k", condition="on_flag:x")],
    )
    final, _ = run_engine(MockAdapter(min_delay=0, max_delay=0), loop, 6,
                          {"convergence_enabled": False, "flags": {"x": "false"}})
    assert "knowledge_extractor" not in [i.role for i in final.iterations]


# ── engine-9: exact pending feedback restored on resume ───────────────


def _disk_writer(root: Path, run_id: str):
    def write(n, name, content):
        d = root / "runs" / run_id / f"iter_{n:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(content)
    return write


def test_resume_restores_long_and_unparsed_feedback(tmp_path):
    long_review = "This review has no structured format. " * 40  # > 500 chars, unparsed
    assert len(long_review) > 500

    def fn(role, req, n):
        return long_review

    loop = get_preset("executor_reviewer", max_iterations=3)
    run_id = "r9"
    a1 = ScriptedReviewAdapter(fn)
    first = RunState(run_id=run_id, workspace_id="w", loop_preset="x", task="t",
                     provider=ProviderName.MOCK, model="", max_iterations=2,
                     config={"convergence_enabled": False})
    f1, _ = run_engine(a1, get_preset("executor_reviewer", max_iterations=2), run_state=first,
                       workspace_path=str(tmp_path),
                       artifact_writer=_disk_writer(tmp_path, run_id))
    assert [i.role for i in f1.iterations] == ["executor", "reviewer"]
    assert f1.iterations[1].feedback is None  # no empty formatter shell stored
    state = RunState(run_id=run_id, workspace_id="w", loop_preset="x", task="t",
                     provider=ProviderName.MOCK, model="", max_iterations=3,
                     status=RunStatus.PAUSED, current_iteration=2,
                     iterations=[i.model_copy() for i in f1.iterations],
                     config={"convergence_enabled": False})
    a2 = ScriptedReviewAdapter(fn)
    run_engine(a2, loop, run_state=state, resume_from=2, workspace_path=str(tmp_path),
               artifact_writer=_disk_writer(tmp_path, run_id))
    prompt = a2.executor_requests()[0].prompt_bundle.user_prompt
    assert long_review.strip() in prompt
    assert "could not be parsed" in prompt


# ── engine-10: workspace budgets and invalid budgets ──────────────────


def test_workspace_budget_is_the_default_and_run_config_wins():
    eng = LoopEngine(adapter=MockAdapter())
    run = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                   model="", config={})
    eng._reset_run_memory()
    eng._workspace_defaults = {"budget_max_tokens": 2_000_000, "budget_max_cost": 50.0}
    eng._configure(run, get_preset("reviewer_consensus"))
    assert eng._limits(run)[:2] == (2_000_000.0, 50.0)
    run.config["budget_max_tokens"] = 1000
    assert eng._limits(run)[0] == 1000.0


def test_malformed_budget_fails_the_run():
    final, _ = run_engine(MockAdapter(min_delay=0, max_delay=0), get_preset("executor_reviewer"),
                          4, {"budget_max_tokens": "2,000,000"})
    assert final.stop_reason == "failed:invalid_config"
    assert "budget_max_tokens" in final.error


# ── engine-11 / tests-claims-8: adjudicator accounting ────────────────


class _LlmMergeAdapter:
    name = "acct"

    def __init__(self):
        self.inner = MockAdapter(min_delay=0, max_delay=0)
        self.adjudicator_calls = 0

    async def run(self, request):
        role = request.prompt_bundle.variables.get("role", "")
        if role == "consensus_adjudicator":
            self.adjudicator_calls += 1
            n = request.prompt_bundle.user_prompt.count("\n[")
            groups = [[i] for i in range(max(n, 1))]
            import json
            return AdapterRunResult(success=True, output=json.dumps({"groups": groups}),
                                    input_tokens=3000, output_tokens=500, token_usage=3500,
                                    provider="acct", model="adj-model", cli_version="7.7.7")
        res = await self.inner.run(request)
        res.input_tokens, res.output_tokens, res.token_usage = 100, 10, 110
        res.model, res.cli_version = "lens-model", "1.0.0"
        return res


def test_llm_adjudicator_usage_in_split_and_provenance():
    adapter = _LlmMergeAdapter()
    final, _ = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=2), 2,
                          {"consensus_similarity_method": "llm", "convergence_enabled": False})
    assert adapter.adjudicator_calls == 1
    cons = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert cons.token_usage == cons.input_tokens + cons.output_tokens
    assert final.total_tokens == final.total_input_tokens + final.total_output_tokens
    assert "adj-model" in cons.model and "7.7.7" in cons.cli_version


# ── tests-claims-5: interrupted runs are recoverable ──────────────────


@pytest_asyncio.fixture
async def fresh_db(tmp_path):
    from backend.config import config
    from backend.database import init_db
    old = config.db_path
    config.db_path = str(tmp_path / "tc5.db")
    await init_db()
    MockAdapter.reset_iteration_count()
    yield
    config.db_path = old


@pytest.mark.asyncio
async def test_interrupted_run_is_recovered_and_resumes_at_the_right_step(tmp_path, fresh_db):
    from backend.database import create_run, create_workspace, get_run
    from backend.models import WorkspaceConfig
    from backend.orchestrator import runner
    from backend.orchestrator.recovery import can_resume_run, recover_interrupted_runs

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    ws = WorkspaceConfig(name="w", path=str(ws_dir), default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    run = RunState(workspace_id=ws.id, loop_preset="executor_reviewer", task="probe",
                   provider=ProviderName.MOCK, model="", max_iterations=6,
                   config={"convergence_enabled": False})
    await create_run(run)
    task = asyncio.create_task(runner.execute_run(run.run_id))
    runner._active_tasks[run.run_id] = task
    for _ in range(400):
        live = await get_run(run.run_id)
        if live and len(live.iterations) >= 3:
            break
        await asyncio.sleep(0.01)
    task.cancel()  # e.g. server shutdown
    with pytest.raises(asyncio.CancelledError):
        await task
    live = await get_run(run.run_id)
    assert live.status == RunStatus.RUNNING and live.iterations
    n_done = live.current_iteration
    assert run.run_id in await recover_interrupted_runs()
    rec = await get_run(run.run_id)
    assert rec.status == RunStatus.PAUSED and rec.stop_reason == "interrupted"
    ok, _ = await can_resume_run(run.run_id)
    assert ok
    await runner.execute_run(run.run_id)
    final = await get_run(run.run_id)
    nums = sorted(i.iteration_number for i in final.iterations)
    assert nums == list(range(1, len(nums) + 1))  # no duplicates, no gaps
    roles = [i.role for i in sorted(final.iterations, key=lambda i: i.iteration_number)]
    expected = ["executor", "reviewer", "executor", "reviewer", "executor", "reviewer"]
    assert roles == expected[:len(roles)] and len(roles) == 6
    assert n_done < 6


# ── adapters-8: per-call telemetry is recorded for real runs ──────────


@pytest.mark.asyncio
async def test_runner_records_one_telemetry_row_per_call(tmp_path, fresh_db):
    from backend.database import create_run, create_workspace, get_run
    from backend.models import WorkspaceConfig
    from backend.orchestrator import runner
    from backend.telemetry.collector import get_collector
    from backend.telemetry.storage import get_records

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    ws = WorkspaceConfig(name="w", path=str(ws_dir), default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    run = RunState(workspace_id=ws.id, loop_preset="reviewer_consensus", task="probe",
                   provider=ProviderName.MOCK, model="", max_iterations=2,
                   config={"convergence_enabled": False})
    await create_run(run)
    await runner.execute_run(run.run_id)
    final = await get_run(run.run_id)
    assert final.stop_reason == "max_iterations"
    mem = get_collector().get_records(run.run_id)
    # executor + 3 lenses + the LLM adjudicator (default similarity method)
    assert len(mem) == 5
    assert sorted(r.role for r in mem) == sorted(["executor", *LENSES, "consensus_adjudicator"])
    rows = await get_records(run_id=run.run_id)
    assert len(rows) == 5
    assert all(r.cli_version == "mock" and r.model == "mock" for r in rows)
    assert {r.iteration for r in rows} == {1, 2}


# ── tests-claims-9: YAML loop variants under their own defaults ───────


def test_yaml_loops_default_behaviour_without_injected_flags():
    # research_followups.yaml: idea_generator is flag-gated (static flag) and
    # never runs without it; the reviewer runs every cycle.
    rf, _ = run_engine(MockAdapter(min_delay=0, max_delay=0),
                       get_preset("custom:loops/research_followups.yaml"), 6,
                       {"convergence_enabled": False})
    assert [i.role for i in rf.iterations] == ["executor", "reviewer"] * 3
    # The Python preset of the same name has no reviewer and always proposes.
    py, _ = run_engine(MockAdapter(min_delay=0, max_delay=0),
                       get_preset("research_followups"), 4, {"convergence_enabled": False})
    assert [i.role for i in py.iterations] == ["executor", "idea_generator"] * 2
    # reviewer_consensus.yaml carries grade_at_least:A; the Python preset does not.
    yaml_rc = get_preset("custom:loops/reviewer_consensus.yaml")
    assert "grade_at_least:A" in yaml_rc.stop_conditions
    assert not any(str(c).startswith("grade_at_least")
                   for c in get_preset("reviewer_consensus").stop_conditions)
    # executor_reviewer.yaml names the escalation role adversarial_reviewer.
    roles = [s.role for s in compile_loop(get_preset("custom:loops/executor_reviewer.yaml")).steps]
    assert roles == ["executor", "reviewer", "adversarial_reviewer"]
