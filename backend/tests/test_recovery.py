"""Tests for crash recovery module."""
import os
import tempfile

import pytest
import pytest_asyncio

from backend.config import config
from backend.database import (
    create_iteration,
    create_run,
    create_workspace,
    get_run,
    init_db,
    update_run,
)
from backend.models import (
    IterationResult,
    IterationStatus,
    ProviderName,
    RunState,
    RunStatus,
    WorkspaceConfig,
)
from backend.orchestrator.recovery import can_resume_run, recover_interrupted_runs


@pytest_asyncio.fixture(autouse=True)
async def setup_db(tmp_path):
    """Initialize a fresh temporary database for each test."""
    db_path = str(tmp_path / "test_recovery.db")
    config.db_path = db_path
    await init_db()
    yield


@pytest.fixture
def workspace_path(tmp_path):
    ws_path = tmp_path / "workspace"
    ws_path.mkdir()
    return str(ws_path)


async def _create_ws(path: str) -> WorkspaceConfig:
    ws = WorkspaceConfig(name="test-ws", path=path, default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    return ws


async def _create_run_with_status(
    workspace_id: str, status: RunStatus, iterations: int = 0
) -> RunState:
    run = RunState(
        workspace_id=workspace_id,
        loop_preset="executor_reviewer",
        task="test task",
        provider=ProviderName.MOCK,
        model="",
        status=status,
    )
    await create_run(run)
    # Manually set status in DB since create_run always creates with initial status
    await update_run(run.run_id, status=status)
    for i in range(iterations):
        it = IterationResult(
            iteration_number=i + 1,
            role="executor",
            status=IterationStatus.COMPLETED,
        )
        await create_iteration(it, run.run_id)
    return run


# ── Recovery tests ───────────────────────────────────────────────────


class TestRecoverInterruptedRuns:
    @pytest.mark.asyncio
    async def test_recover_running_with_iterations_becomes_paused(self, workspace_path):
        ws = await _create_ws(workspace_path)
        run = await _create_run_with_status(ws.id, RunStatus.RUNNING, iterations=3)

        recovered = await recover_interrupted_runs()
        assert run.run_id in recovered

        reloaded = await get_run(run.run_id)
        assert reloaded.status == RunStatus.PAUSED

    @pytest.mark.asyncio
    async def test_recover_running_without_iterations_becomes_failed(self, workspace_path):
        ws = await _create_ws(workspace_path)
        run = await _create_run_with_status(ws.id, RunStatus.RUNNING, iterations=0)

        recovered = await recover_interrupted_runs()
        assert run.run_id in recovered

        reloaded = await get_run(run.run_id)
        assert reloaded.status == RunStatus.FAILED
        assert "interrupted by server restart" in reloaded.error

    @pytest.mark.asyncio
    async def test_no_recovery_needed_for_completed(self, workspace_path):
        ws = await _create_ws(workspace_path)
        await _create_run_with_status(ws.id, RunStatus.COMPLETED, iterations=5)

        recovered = await recover_interrupted_runs()
        assert len(recovered) == 0

    @pytest.mark.asyncio
    async def test_no_recovery_for_pending(self, workspace_path):
        ws = await _create_ws(workspace_path)
        await _create_run_with_status(ws.id, RunStatus.PENDING)

        recovered = await recover_interrupted_runs()
        assert len(recovered) == 0

    @pytest.mark.asyncio
    async def test_recovery_multiple_runs(self, workspace_path):
        ws = await _create_ws(workspace_path)
        run1 = await _create_run_with_status(ws.id, RunStatus.RUNNING, iterations=2)
        run2 = await _create_run_with_status(ws.id, RunStatus.RUNNING, iterations=0)
        await _create_run_with_status(ws.id, RunStatus.COMPLETED)

        recovered = await recover_interrupted_runs()
        assert len(recovered) == 2
        assert run1.run_id in recovered
        assert run2.run_id in recovered


# ── Can resume tests ────────────────────────────────────────────────


class TestCanResumeRun:
    @pytest.mark.asyncio
    async def test_can_resume_paused_run(self, workspace_path):
        ws = await _create_ws(workspace_path)
        run = await _create_run_with_status(ws.id, RunStatus.PAUSED, iterations=2)

        can_resume, reason = await can_resume_run(run.run_id)
        assert can_resume is True
        assert reason == "ok"

    @pytest.mark.asyncio
    async def test_cannot_resume_completed_run(self, workspace_path):
        ws = await _create_ws(workspace_path)
        run = await _create_run_with_status(ws.id, RunStatus.COMPLETED, iterations=5)

        can_resume, reason = await can_resume_run(run.run_id)
        assert can_resume is False
        assert "completed" in reason

    @pytest.mark.asyncio
    async def test_cannot_resume_without_iterations(self, workspace_path):
        ws = await _create_ws(workspace_path)
        run = await _create_run_with_status(ws.id, RunStatus.PAUSED, iterations=0)

        can_resume, reason = await can_resume_run(run.run_id)
        assert can_resume is False
        assert "no completed iterations" in reason

    @pytest.mark.asyncio
    async def test_cannot_resume_nonexistent_run(self):
        can_resume, reason = await can_resume_run("nonexistent-id")
        assert can_resume is False
        assert "not found" in reason.lower()

    @pytest.mark.asyncio
    async def test_can_resume_stopped_run(self, workspace_path):
        ws = await _create_ws(workspace_path)
        run = await _create_run_with_status(ws.id, RunStatus.STOPPED, iterations=3)

        can_resume, reason = await can_resume_run(run.run_id)
        assert can_resume is True
        assert reason == "ok"
