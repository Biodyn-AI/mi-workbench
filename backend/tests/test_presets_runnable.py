"""E8: all four presets (and every YAML loop in loops/) are runnable through
get_preset / the runner, and every node refers to an existing prompt."""
from __future__ import annotations

import asyncio
import pytest
import pytest_asyncio

from backend.adapters.mock import MockAdapter
from backend.config import config
from backend.database import create_run, create_workspace, get_run, init_db
from backend.models import ProviderName, RunState, RunStatus, WorkspaceConfig
from backend.orchestrator.dsl import CONSENSUS_ROLE, compile_loop, parse_stop_conditions
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import (
    BUNDLED_LOOPS_DIR,
    find_loop_yaml,
    get_preset,
    list_presets,
    load_loop_file,
)
from backend.orchestrator.runner import _active_runs, _active_tasks, execute_run
from backend.registry.prompts import PromptRegistry

FOUR_PRESETS = ["executor_reviewer", "reviewer_consensus", "research_followups", "example_custom"]
YAML_FILES = sorted(p for p in BUNDLED_LOOPS_DIR.glob("*.yaml") if not p.name.startswith("._"))
# Flags that switch on every conditional edge of the YAML loops.
ALL_FLAGS = {"needs_revision": True, "high_risk_claims": True, "milestone_complete": True,
             "weak_claims_found": True, "auto_run_top_1": True}


@pytest.fixture(autouse=True)
def _reset_mock():
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    yield


def test_four_presets_listed_and_resolvable():
    names = list_presets()
    for name in FOUR_PRESETS:
        assert name in names
        loop = get_preset(name)
        assert loop.name == name
        assert loop.source
    assert get_preset("executor_reviewer").source == "python"
    assert get_preset("example_custom").source.endswith("example_custom.yaml")
    assert len(YAML_FILES) == 4


@pytest.mark.parametrize("name", FOUR_PRESETS)
def test_every_node_prompt_exists_and_loop_compiles(name):
    reg = PromptRegistry()
    loop = get_preset(name)
    for node in loop.nodes:
        if node.role == CONSENSUS_ROLE:
            continue
        group, prompt = node.prompt_ref.split("/", 1)
        tpl = reg.load_prompt(group, prompt)  # raises if dangling
        assert tpl.system_prompt.strip()
    plan = compile_loop(loop)
    assert plan.cycle_length >= 2
    parse_stop_conditions(loop.stop_conditions)  # valid


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: p.name)
def test_every_yaml_prompt_exists(path):
    reg = PromptRegistry()
    loop = load_loop_file(path)
    for node in loop.nodes:
        if node.role != CONSENSUS_ROLE:
            reg.load_prompt(*node.prompt_ref.split("/", 1))
    assert parse_stop_conditions(loop.stop_conditions).unknown == []


def test_python_preset_prompt_refs_fixed():
    er = get_preset("executor_reviewer")
    adv = [n for n in er.nodes if n.id == "adversarial"][0]
    assert adv.prompt_ref == "adversarial_reviewer/adversarial_reviewer"
    rf = get_preset("research_followups")
    idea = [n for n in rf.nodes if n.id == "idea_generator"][0]
    assert idea.prompt_ref == "idea_generator/follow_up_proposer"


def test_custom_prefix_and_errors(tmp_path):
    p = BUNDLED_LOOPS_DIR / "example_custom.yaml"
    loop = get_preset(f"custom:{p}", max_iterations=7)
    assert loop.name == "example_custom" and loop.max_iterations == 7
    # Relative to a loops directory works too.
    assert get_preset("custom:example_custom.yaml").name == "example_custom"
    with pytest.raises(ValueError, match="Loop file not found"):
        get_preset(f"custom:{tmp_path / 'missing.yaml'}")
    bad = tmp_path / "loop.txt"
    bad.write_text("name: x")
    with pytest.raises(ValueError, match="yaml"):
        get_preset(f"custom:{bad}")
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: b\nnodes: [{id: a, role: executor}]\nedges: [{source: a, target: zz}]\n")
    with pytest.raises(ValueError, match="Invalid loop file"):
        get_preset(f"custom:{broken}")
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("nope_not_a_preset")
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("../loops/example_custom")  # no traversal through names
    assert find_loop_yaml("../etc/passwd") is None
    with pytest.raises(ValueError, match="accepts only max_iterations"):
        get_preset("example_custom", adversarial_every_k=2)


def test_configured_loops_dir_takes_precedence(tmp_path, monkeypatch):
    d = tmp_path / "loops"
    d.mkdir()
    (d / "my_loop.yaml").write_text(
        "name: my_loop\nnodes:\n  - {id: e, role: executor, prompt_ref: executor/mi_executor}\n"
        "  - {id: r, role: reviewer, prompt_ref: reviewer/mi_reviewer}\n"
        "edges: [{source: e, target: r}, {source: r, target: e}]\n")
    monkeypatch.setattr(config, "loops_dir", str(d))
    assert get_preset("my_loop").name == "my_loop"
    assert "my_loop" in list_presets()


def _run_engine(loop, iterations, extra=None):
    async def go():
        eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0))
        run = RunState(workspace_id="w", loop_preset=loop.name, task="probe",
                       provider=ProviderName.MOCK, model="", max_iterations=iterations,
                       config={"convergence_enabled": False, **(extra or {})})
        return await eng.run_loop(run, loop)
    return asyncio.run(go())


@pytest.mark.parametrize("name", FOUR_PRESETS)
def test_every_preset_runs_end_to_end_with_mock(name):
    loop = get_preset(name, max_iterations=6)
    loop.stop_conditions = [c for c in loop.stop_conditions if not c.startswith("grade_at_least")]
    final = _run_engine(loop, 6, {"flags": ALL_FLAGS} if name == "example_custom" else None)
    assert final.status == RunStatus.COMPLETED, final.error
    assert final.stop_reason == "max_iterations"
    assert final.current_iteration == 6
    assert all(it.status.value == "completed" for it in final.iterations)
    roles = {it.role for it in final.iterations}
    assert "executor" in roles and len(roles) >= 2


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: p.name)
def test_every_yaml_loop_runs_via_custom_path(path):
    loop = get_preset(f"custom:{path}", max_iterations=8)
    final = _run_engine(loop, 8, {"flags": ALL_FLAGS})
    assert final.status == RunStatus.COMPLETED, final.error
    assert final.current_iteration >= 2


def test_example_custom_exercises_all_its_roles():
    loop = get_preset("example_custom", max_iterations=12)
    loop.stop_conditions = ["user_stop"]
    final = _run_engine(loop, 12, {"flags": ALL_FLAGS})
    roles = {it.role for it in final.iterations}
    assert {"executor", "reviewer", "adversarial_reviewer", "knowledge_extractor"} <= roles


def test_example_custom_default_stop_conditions():
    # grade_at_least:B from the YAML stops the mock run on the first B review.
    final = _run_engine(get_preset("example_custom", max_iterations=20), 20)
    assert final.stop_reason == "stop_condition:grade_at_least"


# ── Through the runner (DB + artifacts) ──────────────────────────────


@pytest_asyncio.fixture
async def db(tmp_path):
    config.db_path = str(tmp_path / "presets.db")
    await init_db()
    yield


async def _runner_run(tmp_path, preset, max_iterations=4, cfg=None):
    ws_dir = tmp_path / f"ws_{abs(hash(preset)) % 10_000}"
    ws_dir.mkdir()
    ws = WorkspaceConfig(name="w", path=str(ws_dir), default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    run = RunState(workspace_id=ws.id, loop_preset=preset, task="probe",
                   provider=ProviderName.MOCK, model="", max_iterations=max_iterations,
                   config={"convergence_enabled": False, **(cfg or {})})
    await create_run(run)
    await execute_run(run.run_id)
    return await get_run(run.run_id), ws_dir / "runs" / run.run_id


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", [
    "example_custom",
    f"custom:{BUNDLED_LOOPS_DIR / 'reviewer_consensus.yaml'}",
    f"custom:{BUNDLED_LOOPS_DIR / 'research_followups.yaml'}",
])
async def test_yaml_presets_run_through_runner(db, tmp_path, preset):
    # grade_at_least:A+ (run config) overrides example_custom's grade_at_least:B.
    final, run_dir = await _runner_run(tmp_path, preset, 3,
                                       {"flags": ALL_FLAGS, "grade_at_least": "A+"})
    assert final.status == RunStatus.COMPLETED, final.error
    assert final.current_iteration == 3 and len(final.iterations) == 3
    assert (run_dir / "run_meta.json").is_file()
    assert final.run_id not in _active_runs and final.run_id not in _active_tasks


@pytest.mark.asyncio
async def test_unknown_preset_fails_with_stop_reason(db, tmp_path):
    final, _ = await _runner_run(tmp_path, "custom:/definitely/not/here.yaml")
    assert final.status == RunStatus.FAILED
    assert final.stop_reason == "failed:unknown_preset"
    assert "Loop file not found" in final.error
    assert final.run_id not in _active_runs and final.run_id not in _active_tasks
