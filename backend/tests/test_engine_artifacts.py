"""E3 artifact integrity: full outputs persisted, no 500-char overwrite,
one iteration-directory naming scheme, code-execution reports persisted."""
from __future__ import annotations

import json
import pytest
import pytest_asyncio

from backend.adapters.mock import MockAdapter
from backend.artifacts.writer import write_artifact
from backend.config import config
from backend.database import create_run, create_workspace, get_run, init_db
from backend.models import ProviderName, RunState, WorkspaceConfig
from backend.orchestrator.engine import ITERATION_DIR_FMT, LoopEngine, iteration_dir_name
from backend.orchestrator.presets import get_preset
from backend.orchestrator.runner import execute_run


@pytest_asyncio.fixture(autouse=True)
async def setup_db(tmp_path):
    config.db_path = str(tmp_path / "test.db")
    await init_db()
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    yield


async def _run_via_runner(tmp_path, preset="executor_reviewer", max_iterations=4, cfg=None):
    ws_path = tmp_path / "ws"
    ws_path.mkdir()
    ws = WorkspaceConfig(name="ws", path=str(ws_path), default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    run = RunState(workspace_id=ws.id, loop_preset=preset, task="Analyse attention",
                   provider=ProviderName.MOCK, model="", max_iterations=max_iterations,
                   config={"convergence_enabled": False, **(cfg or {})})
    await create_run(run)
    await execute_run(run.run_id)
    final = await get_run(run.run_id)
    return final, ws_path / "runs" / run.run_id


def test_iteration_dir_naming_matches_writer(tmp_path):
    assert iteration_dir_name(7) == "iter_0007" == ITERATION_DIR_FMT.format(7)
    p = write_artifact(tmp_path, 7, "X.md", "x")
    assert p.parent.name == iteration_dir_name(7)


@pytest.mark.asyncio
async def test_full_outputs_persisted_and_not_overwritten(tmp_path):
    final, run_dir = await _run_via_runner(tmp_path)
    assert final.status.value == "completed"
    it1 = run_dir / "iter_0001"
    full = (it1 / "executor_output.md").read_text()
    # The mock executor output is ~1.4k characters; the DB summary is 500.
    assert len(full) > 1000
    ex_iter = [i for i in final.iterations if i.iteration_number == 1][0]
    assert len(ex_iter.output_summary) == 500
    assert full.startswith(ex_iter.output_summary)
    # Parsed artifacts keep their full section content (no re-parse of the summary).
    mech = (it1 / "MECH.md").read_text()
    xp = (it1 / "XP.md").read_text()
    assert len(mech) > 500
    assert "auroc_attention" in xp  # the code block lives at the very end of the output
    assert "executor_output.md" in ex_iter.artifacts_produced
    assert {"MECH.md", "EVAL.md", "XP.md"} <= set(ex_iter.artifacts_produced)
    # Reviewer iteration: full review output persisted too.
    rev = (run_dir / "iter_0002" / "reviewer_output.md").read_text()
    assert "Overall Grade:" in rev and "Reproducibility Gaps" in rev
    # Only iter_NNNN directories exist (no iter_NNN twins).
    names = sorted(d.name for d in run_dir.iterdir() if d.is_dir())
    assert names == [iteration_dir_name(n) for n in range(1, 5)]


@pytest.mark.asyncio
async def test_consensus_iteration_persists_lens_outputs(tmp_path):
    final, run_dir = await _run_via_runner(tmp_path, preset="reviewer_consensus", max_iterations=2)
    d = run_dir / "iter_0002"
    for name in ("EVAL.md", "consensus_merger_output.md", "CONSENSUS.json",
                 "reviewer_output.md", "adversarial_reviewer_output.md",
                 "bio_plausibility_checker_output.md"):
        assert (d / name).is_file(), name
    payload = json.loads((d / "CONSENSUS.json").read_text())
    assert payload["report"]["panel_failed"] is False
    assert payload["critiques"]
    it = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert it.consensus_report and it.consensus_report["attempts"][0]["attempt"] == 1
    assert "reviewer_output.md" in it.artifacts_produced


@pytest.mark.asyncio
async def test_code_execution_report_persisted(tmp_path):
    final, run_dir = await _run_via_runner(
        tmp_path, max_iterations=2, cfg={"code_execution_enabled": True})
    d = run_dir / "iter_0001"
    data = json.loads((d / "CODE_EXECUTION.json").read_text())
    assert data["iteration"] == 1
    assert data["executed"] == 1 and data["passed"] == 1
    assert "incremental_auroc" in data["blocks"][0]["stdout"]
    assert "incremental_auroc" in (d / "RUN_LOG.md").read_text()
    # Summary stored on the iteration row (additive DB column).
    ex = [i for i in final.iterations if i.iteration_number == 1][0]
    assert ex.code_execution["executed"] == 1
    assert ex.code_execution["backend"] == "subprocess"
    assert "CODE_EXECUTION.json" in ex.artifacts_produced


@pytest.mark.asyncio
async def test_engine_workspace_context_uses_same_dir(tmp_path):
    seen = []

    class Spy:
        name = "mock"

        def __init__(self):
            self.inner = MockAdapter(min_delay=0, max_delay=0)

        async def run(self, request):
            seen.append(request.workspace_context.iteration_dir)
            return await self.inner.run(request)

    run_dir = tmp_path / "runs" / "r1"
    eng = LoopEngine(adapter=Spy(),
                     artifact_writer=lambda n, name, c: write_artifact(run_dir, n, name, c))
    run = RunState(run_id="r1", workspace_id="w", loop_preset="executor_reviewer", task="t",
                   provider=ProviderName.MOCK, model="", max_iterations=2,
                   config={"convergence_enabled": False})
    await eng.run_loop(run, get_preset("executor_reviewer", max_iterations=2),
                       workspace_path=str(tmp_path))
    assert seen == ["runs/r1/iter_0001", "runs/r1/iter_0002"]
    for rel in seen:
        assert (tmp_path / rel).is_dir()


@pytest.mark.asyncio
async def test_context_section_reads_prior_artifacts_from_disk(tmp_path):
    """Without an in-memory previous submission (e.g. resume), the executor
    prompt falls back to the latest canonical artifacts on disk."""
    run_dir = tmp_path / "runs" / "r2"
    write_artifact(run_dir, 3, "MECH.md", "# MECH.md\nprior mechanistic finding XYZ")
    write_artifact(run_dir, 3, "executor_output.md", "full output (not used for context)")
    eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0))
    from backend.orchestrator.context import WorkspaceContextReader
    eng._context_reader = WorkspaceContextReader(str(tmp_path), "r2")
    eng._pending_feedback = [("reviewer", "=== REVIEWER FEEDBACK ===\nfix it")]
    run = RunState(run_id="r2", workspace_id="w", loop_preset="x", task="the task",
                   provider=ProviderName.MOCK, model="")
    prompt = eng._build_executor_user_prompt(run, 4)
    assert "### MECH.md (from iteration 3)" in prompt
    assert "prior mechanistic finding XYZ" in prompt
    assert "full output (not used for context)" not in prompt
