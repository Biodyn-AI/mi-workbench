"""Calibrated default stopping policy: the revision budget.

Run config ``revision_budget`` (default 4; ``None`` disables it) is the maximum
number of executor revisions after the initial executor submission (a
``seed_executor_output`` counts as the initial submission). After the last
allowed revision the remaining non-executor steps run, so the final artifact
is reviewed, and the run stops with ``revision_budget``. ``convergence_enabled``
defaults to false; the adaptive rule stays available.

The default comes from the stopping calibration
(``experiments/stopping/results/``): pooled over 24 real trajectories, the
pre-specified selection rule picks a fixed budget of four revisions. The
equivalence tests below check that the engine's budget stops at the state the
calibration's replay assumed (``fixed_k`` in ``experiments/stopping/replay.py``
and ``replay_stopping_rule(..., revision_budget=k)``).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
import pytest_asyncio

from backend.adapters.mock import MockAdapter
from backend.models import (
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    RunState,
    RunStatus,
)
from backend.orchestrator.convergence import (
    DEFAULT_CONVERGENCE_ENABLED,
    DEFAULT_REVISION_BUDGET,
    DEFAULT_STOPPING_RULE,
    REVISION_BUDGET_STOP_REASON,
    parse_revision_budget,
    replay_stopping_rule,
)
from backend.orchestrator.dsl import parse_loop_yaml, parse_stop_conditions
from backend.orchestrator.presets import BUNDLED_LOOPS_DIR, get_preset, load_loop_file
from backend.orchestrator.runner import build_run_meta
from backend.tests.fixes_r2_helpers import LENSES, ScriptedReviewAdapter, json_block, run_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_PY = REPO_ROOT / "experiments" / "stopping" / "replay.py"

SEED = "# Original write-up\n\nWe found 19 significant heads and conclude they form the circuit."
HIGH = json_block([{"severity": "high",
                    "description": "No multiple-testing correction across 144 heads."}])
CRIT = json_block([{"severity": "critical", "description": "Leakage between folds."}])


@pytest.fixture(autouse=True)
def _reset_mock():
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    yield
    MockAdapter.fixed_iteration = None


def _mock():
    return MockAdapter(min_delay=0, max_delay=0)


def _panel(reply=HIGH):
    return ScriptedReviewAdapter(lambda role, req, n: reply, roles=LENSES)


def _roles(state):
    return [it.role for it in sorted(state.iterations, key=lambda i: i.iteration_number)]


def _short(roles):
    return ["E" if r == "executor" else ("P" if r == "consensus_merger" else r[:3].upper())
            for r in roles]


def _n_exec(state):
    return sum(1 for it in state.iterations if it.role == "executor")


# ── defaults and parsing ─────────────────────────────────────────────


def test_calibrated_defaults():
    assert DEFAULT_REVISION_BUDGET == 4
    assert DEFAULT_CONVERGENCE_ENABLED is False
    # The adaptive rule (used when convergence is enabled) is unchanged.
    r = DEFAULT_STOPPING_RULE
    assert (r.window, r.min_iterations, r.similarity_threshold, r.rule) == (3, 4, 0.9, "any")
    assert REVISION_BUDGET_STOP_REASON == "revision_budget"


@pytest.mark.parametrize("value,expected", [
    (0, 0), (4, 4), ("4", 4), (" 2 ", 2), (3.0, 3), (None, None),
    ("none", None), ("NULL", None), ("off", None),
])
def test_parse_revision_budget_valid(value, expected):
    assert parse_revision_budget(value) == expected


@pytest.mark.parametrize("value", [-1, "-2", 2.5, True, False, "many", "", [4], {"k": 4}])
def test_parse_revision_budget_invalid(value):
    with pytest.raises(ValueError, match="revision_budget"):
        parse_revision_budget(value)


def test_stop_condition_entry_parses():
    assert parse_stop_conditions(["revision_budget:3"]).config == {"revision_budget": 3}
    assert parse_stop_conditions(["revision_budget=0"]).config == {"revision_budget": 0}
    assert parse_stop_conditions(["revision_budget:none"]).config == {"revision_budget": None}
    # YAML mapping forms, including a YAML null.
    loop = parse_loop_yaml(
        "name: x\nnodes: [{id: e, role: executor}]\nstop_conditions:\n  - revision_budget: 2\n")
    assert loop.stop_conditions == ["revision_budget:2"]
    loop2 = parse_loop_yaml(
        "name: x\nnodes: [{id: e, role: executor}]\nstop_conditions: {revision_budget: null}\n")
    assert parse_stop_conditions(loop2.stop_conditions).config == {"revision_budget": None}
    for bad in ("revision_budget:-1", "revision_budget:2.5", "revision_budget:lots"):
        with pytest.raises(ValueError, match="revision_budget"):
            parse_stop_conditions([bad])


def test_bundled_loops_still_load():
    for p in sorted(BUNDLED_LOOPS_DIR.glob("*.yaml")):
        if p.name.startswith("._"):
            continue
        loop = load_loop_file(p)
        assert "revision_budget" not in parse_stop_conditions(loop.stop_conditions).config


# ── the default policy on the bundled presets ───────────────────────


def test_default_policy_is_four_reviewed_revisions_without_convergence():
    events: list = []
    final, eng = run_engine(_mock(), get_preset("reviewer_consensus"), 50, {}, events=events)
    assert final.status == RunStatus.COMPLETED
    assert final.stop_reason == "revision_budget"
    # Initial submission + 4 revisions, each reviewed: E P E P E P E P E P.
    assert _short(_roles(final)) == ["E", "P"] * 5
    assert final.current_iteration == 10
    # The mock converges at iteration 7 when the adaptive rule is on; by
    # default it is off.
    assert not any(t == "convergence_detected" for t, _ in events)
    started = [d for t, d in events if t == "loop_started"][0]
    assert started["revision_budget"] == 4 and started["revision_budget_source"] == "default"
    assert started["convergence_enabled"] is False
    stop = [d for t, d in events if t == "stop_condition_met"][-1]
    assert stop["stop_reason"] == "revision_budget"
    assert stop["executor_submissions"] == 5 and stop["executor_revisions"] == 4
    finished = [d for t, d in events if t == "loop_finished"][0]
    assert finished["stop_reason"] == "revision_budget"
    s = eng.stopping_settings()
    assert s["revision_budget"] == 4 and s["revision_budget_source"] == "default"
    assert s["convergence_enabled"] is False and s["convergence_enabled_source"] == "default"
    assert s["executor_submissions"] == 5 and s["executor_revisions"] == 4
    assert s["stopping_rule"] == DEFAULT_STOPPING_RULE.to_dict()


def test_executor_first_consensus_loop_matches_the_case_study_horizon():
    """revision_budget=4 on executor-first reviewer_consensus: 5 executor steps
    and 5 panels (10 iterations), the case study's E1 P1 ... E5 P5 horizon."""
    final, _ = run_engine(_panel(), get_preset("reviewer_consensus"), 50,
                          {"revision_budget": 4})
    assert final.stop_reason == "revision_budget"
    roles = _roles(final)
    assert len(roles) == 10
    assert roles.count("executor") == 5 and roles.count("consensus_merger") == 5
    assert roles[-1] == "consensus_merger"          # the final artifact was reviewed
    # With the case study's own cap (max_iterations 10) both conditions hold
    # at the same point; the iteration cap is checked first.
    capped, _ = run_engine(_panel(), get_preset("reviewer_consensus"), 10,
                           {"revision_budget": 4})
    assert capped.stop_reason == "max_iterations"
    assert _roles(capped) == roles


# ── equivalence with the stopping calibration's replay ──────────────


def _trajectory_events(final, seed):
    """The replay event sequence of a seeded engine run (E0 = seed at 0)."""
    events = [{"iteration": 0, "role": "executor", "output": seed}]
    for it in sorted(final.iterations, key=lambda i: i.iteration_number):
        if it.role == "executor":
            events.append({"iteration": it.iteration_number, "role": "executor",
                           "output": it.output_summary})
        else:
            events.append({"iteration": it.iteration_number, "role": it.role,
                           "grade": it.grade, "critical": it.critical_count or 0,
                           "high": it.high_count or 0})
    return events


def _load_replay_module():
    if not REPLAY_PY.is_file():
        pytest.skip("experiments/stopping/replay.py is not present")
    spec = importlib.util.spec_from_file_location("miw_stop_replay_for_engine_tests", REPLAY_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_seeded_panel_first_budget_matches_the_calibration_fixed_k_rule():
    """Panel-first seeded loop (the calibration's trajectory structure,
    P0 E1 P1 ... E5 P5): revision_budget=k stops with E_k as the final state,
    exactly the state at which the replay's ``fixed_k`` rule stops, after the
    same k revisions; the engine additionally runs P_k, the review of E_k."""
    rp = _load_replay_module()
    horizon = rp.sc.HORIZON
    assert horizon == 5
    full, _ = run_engine(_panel(), get_preset("reviewer_consensus"),
                         rp.sc.total_iterations(horizon),
                         {"seed_executor_output": SEED, "revision_budget": None})
    assert full.stop_reason == "max_iterations"
    assert _roles(full) == rp.sc.expected_roles(horizon)
    events = _trajectory_events(full, SEED)
    traj = {"item": "synthetic",
            "steps": [{"iteration": it.iteration_number, "role": it.role,
                       "tokens": {"total": it.token_usage}}
                      for it in sorted(full.iterations, key=lambda i: i.iteration_number)]}

    for k in range(1, horizon + 1):
        live, eng = run_engine(_panel(), get_preset("reviewer_consensus"), 50,
                               {"seed_executor_output": SEED, "revision_budget": k})
        assert live.stop_reason == "revision_budget", k
        stop_it = live.current_iteration
        # Same steps as the recorded trajectory up to the stop: P0 E1 ... E_k P_k.
        assert _roles(live) == rp.sc.expected_roles(horizon)[:stop_it], k
        assert stop_it == 2 * k + 1, k
        assert _n_exec(live) == k                       # k revisions of the seed
        assert eng.stopping_settings()["executor_revisions"] == k

        fixed = rp.apply_rule(traj, {"name": f"fixed_k{k}", "family": "fixed",
                                     "description": "", "fixed_k": k}, horizon)
        assert fixed["k_stop"] == k
        assert rp.sc.state_index_at(stop_it) == fixed["k_stop"], k   # same final state E_k
        assert fixed["cycles"] == _n_exec(live), k                  # same revisions
        assert fixed["stopped_before_horizon"] == (k < horizon)
        # The only extra step is the panel reviewing the final state.
        assert stop_it == fixed["iteration"] + 1
        assert rp.sc.step_label(stop_it) == f"P{k}"

        # The platform replay of the budget stops where the live engine does.
        res = replay_stopping_rule(events, convergence=False, revision_budget=k)
        assert res["stopped"] and res["stop_reason"] == "revision_budget"
        assert res["iteration"] == stop_it, k


def test_replay_budget_and_adaptive_rule_whichever_first():
    """``replay_stopping_rule`` with both the adaptive rule and a budget
    reports the earlier stop, like the engine."""
    texts = ["same words here"] * 6
    events = [{"iteration": 0, "role": "executor", "output": texts[0]}]
    for k in range(6):
        events.append({"iteration": 2 * k + 1, "role": "consensus_merger", "grade": "C",
                       "critical": 0, "high": 1})
        if k < 5:
            events.append({"iteration": 2 * k + 2, "role": "executor", "output": texts[k + 1]})
    # grade_stable (window 3, min 4 iterations) holds at P2 (iteration 5).
    adaptive = replay_stopping_rule(events)
    assert adaptive["stopped"] and adaptive["iteration"] == 5
    both = replay_stopping_rule(events, revision_budget=1)
    assert both["iteration"] == 3 and both["stop_reason"] == "revision_budget"
    later = replay_stopping_rule(events, revision_budget=4)
    assert later["iteration"] == 5 and later["stop_reason"].startswith("converged:")
    assert replay_stopping_rule(events, convergence=False)["stopped"] is False
    zero = replay_stopping_rule(events, convergence=False, revision_budget=0)
    assert zero["iteration"] == 1                  # P0 reviews the seed, then stop
    with pytest.raises(ValueError):
        replay_stopping_rule(events, revision_budget=-1)


# ── semantics ────────────────────────────────────────────────────────


def test_remaining_non_executor_steps_of_the_cycle_run():
    # executor -> reviewer -> adversarial (every_k:3): after the 3rd submission
    # (budget 2) the reviewer and the cycle-3 adversarial still run.
    final, _ = run_engine(_mock(), get_preset("executor_reviewer"), 50, {"revision_budget": 2})
    assert final.stop_reason == "revision_budget"
    assert _roles(final) == ["executor", "reviewer"] * 3 + ["adversarial"]
    one, _ = run_engine(_mock(), get_preset("executor_reviewer"), 50, {"revision_budget": 1})
    assert _roles(one) == ["executor", "reviewer"] * 2


def test_budget_zero_reviews_only_the_initial_submission():
    final, _ = run_engine(_mock(), get_preset("executor_reviewer"), 50, {"revision_budget": 0})
    assert final.stop_reason == "revision_budget"
    assert _roles(final) == ["executor", "reviewer"]
    seeded, _ = run_engine(_panel(), get_preset("reviewer_consensus"), 50,
                           {"revision_budget": 0, "seed_executor_output": SEED})
    assert seeded.stop_reason == "revision_budget"
    assert _roles(seeded) == ["consensus_merger"]   # the seed is the initial submission


def test_none_disables_the_budget():
    final, eng = run_engine(_mock(), get_preset("executor_reviewer"), 25,
                            {"revision_budget": None})
    assert final.stop_reason == "max_iterations" and final.current_iteration == 25
    assert _n_exec(final) > 5
    assert eng.stopping_settings()["revision_budget"] is None
    assert eng.stopping_settings()["revision_budget_source"] == "config"
    # Strings accepted from the CLI / YAML.
    final2, _ = run_engine(_mock(), get_preset("executor_reviewer"), 14,
                           {"revision_budget": "none"})
    assert final2.stop_reason == "max_iterations"


def test_loop_stop_condition_and_precedence():
    loop = get_preset("executor_reviewer")
    loop.stop_conditions = ["user_stop", "revision_budget:1"]
    final, eng = run_engine(_mock(), loop, 50, {})
    assert final.stop_reason == "revision_budget" and _n_exec(final) == 2
    assert eng.stopping_settings()["revision_budget_source"] == "loop"
    # The run config wins over the loop, also with None (disabled).
    final2, _ = run_engine(_mock(), loop, 50, {"revision_budget": 2})
    assert final2.stop_reason == "revision_budget" and _n_exec(final2) == 3
    final3, _ = run_engine(_mock(), loop, 12, {"revision_budget": None})
    assert final3.stop_reason == "max_iterations"
    # A loop can disable the default.
    off = get_preset("executor_reviewer")
    off.stop_conditions = ["revision_budget:none"]
    final4, eng4 = run_engine(_mock(), off, 15, {})
    assert final4.stop_reason == "max_iterations"
    assert eng4.stopping_settings()["revision_budget"] is None
    # The loop config block is a lower-priority source than stop_conditions.
    cfg_loop = get_preset("executor_reviewer")
    cfg_loop.config = {**cfg_loop.config, "revision_budget": 3}
    cfg_loop.stop_conditions = ["revision_budget:1"]
    final5, _ = run_engine(_mock(), cfg_loop, 50, {})
    assert _n_exec(final5) == 2


def test_loops_without_an_executor_ignore_the_budget():
    loop = LoopDefinition(
        name="rev", nodes=[LoopNode(id="r", role="reviewer", prompt_ref="reviewer/mi_reviewer")],
        edges=[LoopEdge(source="r", target="r", condition="always")])
    final, _ = run_engine(_mock(), loop, 4, {"revision_budget": 0})
    assert final.stop_reason == "max_iterations" and final.current_iteration == 4


@pytest.mark.parametrize("bad", [-1, "many", True, 2.5])
def test_invalid_budget_fails_the_run_cleanly(bad):
    final, eng = run_engine(_mock(), get_preset("executor_reviewer"), 5, {"revision_budget": bad})
    assert final.status == RunStatus.FAILED
    assert final.stop_reason == "failed:invalid_config"
    assert "revision_budget" in (final.error or "")
    assert final.iterations == [] and eng.stopping_settings() == {}


def test_invalid_loop_stop_condition_fails_the_run_cleanly():
    loop = get_preset("executor_reviewer")
    loop.stop_conditions = ["revision_budget:-2"]
    final, _ = run_engine(_mock(), loop, 5, {})
    assert final.stop_reason == "failed:invalid_config"


# ── whichever condition is reached first stops the run ──────────────


def test_convergence_before_the_budget_and_budget_before_convergence():
    final, _ = run_engine(_mock(), get_preset("reviewer_consensus"), 50,
                          {"convergence_enabled": True})
    assert final.stop_reason == "converged:output_similar" and final.current_iteration == 7
    early, _ = run_engine(_mock(), get_preset("reviewer_consensus"), 50,
                          {"convergence_enabled": True, "revision_budget": 2})
    assert early.stop_reason == "revision_budget" and early.current_iteration == 6


def test_grade_at_least_before_the_budget():
    loop = get_preset("executor_reviewer")
    loop.stop_conditions = ["grade_at_least:A-"]
    # Mock reviewer grades in this loop go B, A-, ...: the second review stops
    # the run, long before the default budget.
    final, _ = run_engine(_mock(), loop, 50, {})
    assert final.stop_reason == "stop_condition:grade_at_least"
    assert final.current_iteration == 4
    # With budget 0 the first review (B) is the last step.
    final2, _ = run_engine(_mock(), loop, 50, {"revision_budget": 0})
    assert final2.stop_reason == "revision_budget" and final2.current_iteration == 2


def test_caps_budgets_flags_and_cancellation_still_win_when_reached_first():
    rc = get_preset("reviewer_consensus")
    capped, _ = run_engine(_mock(), rc, 3, {})
    assert capped.stop_reason == "max_iterations" and capped.current_iteration == 3
    cap_cfg, _ = run_engine(_mock(), get_preset("reviewer_consensus"), 50, {"max_iterations": 5})
    assert cap_cfg.stop_reason == "max_iterations" and cap_cfg.current_iteration == 5
    tokens, _ = run_engine(_mock(), get_preset("reviewer_consensus"), 50,
                           {"budget_max_tokens": 2000})
    assert tokens.stop_reason == "budget_tokens"
    cost, _ = run_engine(_mock(), get_preset("reviewer_consensus"), 50,
                         {"budget_max_cost": 0.005})
    assert cost.stop_reason == "budget_cost"
    flagged = get_preset("reviewer_consensus")
    flagged.stop_conditions = ["on_flag:halt"]
    halted, _ = run_engine(_mock(), flagged, 50, {"flags": {"halt": True}})
    assert halted.stop_reason == "stop_condition:on_flag:halt" and halted.iterations == []
    ev = asyncio.Event()
    ev.set()
    cancelled, _ = run_engine(_mock(), get_preset("reviewer_consensus"), 50, {}, cancel_event=ev)
    assert cancelled.status == RunStatus.STOPPED and cancelled.stop_reason == "cancelled"


def test_consensus_gate_does_not_block_the_budget():
    # Every panel leaves an unresolved CRITICAL: the gate holds convergence and
    # grade_at_least open, but the revision budget (a budget) still applies.
    final, _ = run_engine(_panel(CRIT), get_preset("reviewer_consensus"), 50,
                          {"consensus_gate": True, "convergence_enabled": True,
                           "convergence_signals": "grade_stable", "grade_at_least": "F"})
    assert final.stop_reason == "revision_budget" and final.current_iteration == 10


# ── resume ───────────────────────────────────────────────────────────


def _resume_state(first, max_iterations, config):
    return RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                    model="", max_iterations=max_iterations, status=RunStatus.PAUSED,
                    current_iteration=first.current_iteration,
                    iterations=[i.model_copy() for i in first.iterations], config=dict(config))


@pytest.mark.parametrize("cut", [1, 3, 4, 6])
def test_resume_counts_the_stored_submissions(cut):
    cfg = {"revision_budget": 2}
    full, _ = run_engine(_mock(), get_preset("executor_reviewer"), 50, cfg)
    first, _ = run_engine(_mock(), get_preset("executor_reviewer"), cut, cfg)
    assert first.stop_reason == "max_iterations"
    final, eng = run_engine(_mock(), get_preset("executor_reviewer"),
                            run_state=_resume_state(first, 50, cfg), resume_from=cut)
    assert final.stop_reason == "revision_budget"
    assert _roles(final) == _roles(full)
    assert eng.stopping_settings()["executor_submissions"] == 3


def test_resume_of_a_seeded_run_counts_the_seed():
    cfg = {"revision_budget": 2, "seed_executor_output": SEED}
    full, _ = run_engine(_panel(), get_preset("reviewer_consensus"), 50, cfg)
    assert _short(_roles(full)) == ["P", "E", "P", "E", "P"]
    first, _ = run_engine(_panel(), get_preset("reviewer_consensus"), 2, cfg)
    assert _short(_roles(first)) == ["P", "E"]
    final, eng = run_engine(_panel(), get_preset("reviewer_consensus"),
                            run_state=_resume_state(first, 50, cfg), resume_from=2)
    assert final.stop_reason == "revision_budget"
    assert _short(_roles(final)) == ["P", "E", "P", "E", "P"]
    assert eng.stopping_settings()["executor_submissions"] == 3   # seed + 2 revisions


def test_resume_after_the_budget_was_used_stops_without_a_new_step():
    cfg = {"revision_budget": 1}
    full, _ = run_engine(_mock(), get_preset("executor_reviewer"), 50, cfg)
    assert full.stop_reason == "revision_budget"
    state = _resume_state(full, 50, cfg)
    final, _ = run_engine(_mock(), get_preset("executor_reviewer"), run_state=state,
                          resume_from=full.current_iteration)
    assert final.stop_reason == "revision_budget"
    assert len(final.iterations) == len(full.iterations)


# ── run_meta ─────────────────────────────────────────────────────────


def test_run_meta_records_the_effective_budget():
    final, eng = run_engine(_mock(), get_preset("reviewer_consensus"), 50, {})
    meta = build_run_meta(final, get_preset("reviewer_consensus"),
                          eng._convergence.get_metrics(), tool_settings=eng.tool_settings(),
                          stopping_settings=eng.stopping_settings())
    assert meta.stop_reason == "revision_budget"
    assert meta.config["revision_budget"] == 4
    eff = meta.config["effective_stopping"]
    assert eff["revision_budget_source"] == "default" and eff["executor_revisions"] == 4
    assert eff["convergence_enabled"] is False
    # An explicit run-config value is kept (also None = disabled).
    st = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                  model="", config={"revision_budget": None, "unrelated": 1})
    assert build_run_meta(st).config == {"revision_budget": None}


@pytest_asyncio.fixture
async def runner_db(tmp_path):
    from backend.config import config
    from backend.database import init_db

    old = config.db_path
    config.db_path = str(tmp_path / "rb.db")
    await init_db()
    MockAdapter.reset_iteration_count()
    yield
    config.db_path = old


@pytest.mark.asyncio
async def test_runner_persists_revision_budget_stop_and_meta(tmp_path, runner_db):
    from backend.database import create_run, create_workspace, get_run
    from backend.models import WorkspaceConfig
    from backend.orchestrator.runner import execute_run

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    ws = WorkspaceConfig(name="w", path=str(ws_dir), default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    run = RunState(workspace_id=ws.id, loop_preset="executor_reviewer", task="probe",
                   provider=ProviderName.MOCK, model="", max_iterations=50,
                   config={"revision_budget": 1})
    await create_run(run)
    await execute_run(run.run_id)
    final = await get_run(run.run_id)
    assert final.status == RunStatus.COMPLETED and final.stop_reason == "revision_budget"
    assert [i.role for i in sorted(final.iterations, key=lambda i: i.iteration_number)] == \
        ["executor", "reviewer", "executor", "reviewer"]
    assert final.config.get("revision_budget") == 1
    meta = json.loads((ws_dir / "runs" / run.run_id / "run_meta.json").read_text())
    assert meta["stop_reason"] == "revision_budget"
    assert meta["config"]["revision_budget"] == 1
    assert meta["config"]["effective_stopping"]["revision_budget_source"] == "config"
    assert meta["config"]["effective_stopping"]["executor_revisions"] == 1
    assert meta["config"]["effective_stopping"]["convergence_enabled"] is False
