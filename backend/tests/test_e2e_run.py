"""End-to-end tests for run execution pipeline."""
import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from backend.database import create_run, create_workspace, get_run, init_db
from backend.config import config
from backend.models import (
    ProviderName,
    RunState,
    RunStatus,
    WorkspaceConfig,
)
from backend.orchestrator.runner import (
    _active_runs,
    cancel_run,
    execute_run,
    get_active_run_ids,
)


@pytest_asyncio.fixture(autouse=True)
async def setup_db(tmp_path):
    """Initialize a fresh temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    config.db_path = db_path
    await init_db()
    yield


@pytest.fixture
def workspace_path(tmp_path):
    """Create a temporary workspace directory."""
    ws_path = tmp_path / "workspace"
    ws_path.mkdir()
    return str(ws_path)


async def _create_test_workspace(ws_path: str) -> WorkspaceConfig:
    """Helper to create a workspace in the DB."""
    ws = WorkspaceConfig(
        name="test-ws",
        path=ws_path,
        default_provider=ProviderName.MOCK,
    )
    await create_workspace(ws)
    return ws


async def _create_test_run(
    workspace_id: str,
    max_iterations: int = 4,
    provider: ProviderName = ProviderName.MOCK,
) -> RunState:
    """Helper to create a run in the DB."""
    run = RunState(
        workspace_id=workspace_id,
        loop_preset="executor_reviewer",
        task="Test task: analyze attention patterns",
        provider=provider,
        model="",
        max_iterations=max_iterations,
    )
    await create_run(run)
    return run


@pytest.mark.asyncio
async def test_run_executes_to_completion(workspace_path):
    """Create workspace+run, verify iterations appear in DB after completion."""
    ws = await _create_test_workspace(workspace_path)
    run = await _create_test_run(ws.id, max_iterations=4)

    await execute_run(run.run_id)

    # Reload from DB
    final = await get_run(run.run_id)
    assert final is not None
    assert final.status in (RunStatus.COMPLETED, RunStatus.FAILED)
    # With max_iterations=4, should have exactly 4 iterations
    assert final.current_iteration == 4
    assert len(final.iterations) == 4
    # Each iteration should be completed
    for it in final.iterations:
        assert it.status.value == "completed"
        assert it.role in ("executor", "reviewer", "adversarial")


@pytest.mark.asyncio
async def test_run_writes_artifacts(workspace_path):
    """Verify artifact files exist on disk after run."""
    ws = await _create_test_workspace(workspace_path)
    run = await _create_test_run(ws.id, max_iterations=4)

    await execute_run(run.run_id)

    # Check that run directory was created
    run_dir = Path(workspace_path) / "runs" / run.run_id
    assert run_dir.exists(), f"Run dir not found: {run_dir}"

    # Check that run_meta.json exists
    meta_path = run_dir / "run_meta.json"
    assert meta_path.exists(), "run_meta.json not written"
    meta = json.loads(meta_path.read_text())
    assert meta["run_id"] == run.run_id
    assert meta["total_iterations"] == 4

    # Check that iteration artifact dirs exist
    iter_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")])
    assert len(iter_dirs) > 0, "No iteration artifact directories created"


@pytest.mark.asyncio
async def test_run_can_be_cancelled(workspace_path):
    """Start run, cancel it, verify stopped status."""
    ws = await _create_test_workspace(workspace_path)
    # Use more iterations so we have time to cancel
    run = await _create_test_run(ws.id, max_iterations=100)

    # Start execution as a task
    task = asyncio.create_task(execute_run(run.run_id))
    # Wait briefly for the run to start
    await asyncio.sleep(0.3)

    # Cancel it
    cancelled = await cancel_run(run.run_id)
    assert cancelled is True

    # Wait for the task to finish
    await asyncio.wait_for(task, timeout=5.0)

    # Check the final state
    final = await get_run(run.run_id)
    assert final is not None
    assert final.status == RunStatus.STOPPED
    # Should have done fewer than 100 iterations
    assert final.current_iteration < 100


@pytest.mark.asyncio
async def test_run_updates_totals(workspace_path):
    """Verify token/cost totals accumulate across iterations."""
    ws = await _create_test_workspace(workspace_path)
    run = await _create_test_run(ws.id, max_iterations=4)

    await execute_run(run.run_id)

    final = await get_run(run.run_id)
    assert final is not None
    # Mock adapter generates 800-4000 tokens per call at $3/1M tokens
    assert final.total_tokens > 0, "Token total should be positive"
    assert final.total_cost > 0.0, "Cost total should be positive"
    # With 4 iterations of 800-4000 tokens each, total should be >= 3200
    assert final.total_tokens >= 800 * 4, f"Expected at least 3200 tokens, got {final.total_tokens}"
    # Cost should match: tokens * 0.000003
    # Allow some float tolerance
    expected_min_cost = 800 * 4 * 0.000003
    assert final.total_cost >= expected_min_cost * 0.9


@pytest.mark.asyncio
async def test_run_handles_failure(workspace_path):
    """Configure mock to fail, verify FAILED status."""
    ws = await _create_test_workspace(workspace_path)
    run = await _create_test_run(ws.id, max_iterations=4)

    # Monkey-patch the mock adapter to always fail
    from backend.adapters.registry import get_adapter
    adapter = get_adapter(ProviderName.MOCK)
    original_failure_rate = adapter.failure_rate
    adapter.failure_rate = 1.0  # 100% failure

    try:
        await execute_run(run.run_id)

        final = await get_run(run.run_id)
        assert final is not None
        assert final.status == RunStatus.FAILED
        assert final.error is not None
        assert len(final.error) > 0
    finally:
        # Restore original failure rate
        adapter.failure_rate = original_failure_rate


@pytest.mark.asyncio
async def test_cancel_nonexistent_run():
    """Cancelling a run that doesn't exist returns False."""
    result = await cancel_run("nonexistent-id")
    assert result is False


@pytest.mark.asyncio
async def test_active_run_tracking(workspace_path):
    """Verify run appears in active list during execution and is removed after."""
    ws = await _create_test_workspace(workspace_path)
    run = await _create_test_run(ws.id, max_iterations=100)

    task = asyncio.create_task(execute_run(run.run_id))
    await asyncio.sleep(0.3)

    # Should be active
    assert run.run_id in get_active_run_ids()

    # Cancel and wait
    await cancel_run(run.run_id)
    await asyncio.wait_for(task, timeout=5.0)

    # Should no longer be active
    assert run.run_id not in get_active_run_ids()
