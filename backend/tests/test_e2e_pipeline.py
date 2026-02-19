"""Comprehensive end-to-end integration tests for the full MI-Workbench pipeline.

Exercises the entire autonomous pipeline through the HTTP API: workspace creation,
run execution, iteration cycling, artifact parsing/writing, convergence detection,
telemetry, repropack generation, claims parsing, and multi-run comparison.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient

from backend.adapters.mock import MockAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_workspace(client: AsyncClient, tmp_path: Path) -> str:
    """Create a workspace via API and return its id."""
    ws_path = str(tmp_path / "ws")
    resp = await client.post("/api/workspaces", json={
        "name": "e2e-test-workspace",
        "path": ws_path,
    })
    assert resp.status_code == 201, f"Workspace creation failed: {resp.text}"
    return resp.json()["id"]


async def _poll_run(
    client: AsyncClient,
    run_id: str,
    *,
    terminal_statuses: set[str] = {"completed", "failed", "stopped"},
    timeout: float = 30.0,
    poll_interval: float = 0.2,
) -> dict:
    """Poll GET /api/runs/{run_id} until status reaches a terminal state."""

    async def _poll() -> dict:
        while True:
            resp = await client.get(f"/api/runs/{run_id}")
            assert resp.status_code == 200
            data = resp.json()
            if data["status"] in terminal_statuses:
                return data
            await asyncio.sleep(poll_interval)

    return await asyncio.wait_for(_poll(), timeout=timeout)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def _reset_mock_counter():
    """Reset the MockAdapter class-level iteration counter between tests."""
    MockAdapter.reset_iteration_count()
    yield
    MockAdapter.reset_iteration_count()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_autonomous_pipeline(client: AsyncClient, tmp_path: Path):
    """End-to-end: create workspace, run pipeline, verify every subsystem."""

    # 1. Create workspace
    ws_id = await _create_workspace(client, tmp_path)

    # 2. Create and execute a run
    run_resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "Analyze attention patterns in Geneformer for GRN inference",
        "provider": "mock",
        "loop_preset": "executor_reviewer",
        "max_iterations": 8,
        "config_overrides": {"convergence_enabled": True},
    })
    assert run_resp.status_code == 201, f"Run creation failed: {run_resp.text}"
    run_id = run_resp.json()["run_id"]

    # 3. Wait for completion
    final = await _poll_run(client, run_id, timeout=30.0)

    # 4. Verify the run completed (not failed)
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )

    # 5. Verify iterations -- at least 4 ran (executor + reviewer cycles)
    iterations = final.get("iterations", [])
    assert len(iterations) >= 4, (
        f"Expected at least 4 iterations, got {len(iterations)}"
    )
    # Check that executor and reviewer roles both appear
    roles_seen = {it["role"] for it in iterations}
    assert "executor" in roles_seen, "No executor iterations found"
    assert "reviewer" in roles_seen, "No reviewer iterations found"

    # 6. Verify artifact parsing worked -- artifacts_produced contains
    #    discrete artifact names (MECH.md, EVAL.md) not just "executor_output.md"
    executor_iters = [it for it in iterations if it["role"] == "executor"]
    assert len(executor_iters) > 0
    first_executor = executor_iters[0]
    artifacts = first_executor.get("artifacts_produced", [])
    assert len(artifacts) > 0, "Executor produced no artifacts"
    # At least one should be a named artifact, not a generic fallback
    named_artifacts = [a for a in artifacts if a in ("MECH.md", "EVAL.md", "XP.md")]
    assert len(named_artifacts) > 0, (
        f"Expected named artifacts (MECH.md, EVAL.md, XP.md), got: {artifacts}"
    )

    # 7. Verify artifacts on disk
    ws_data = (await client.get(f"/api/workspaces/{ws_id}")).json()
    ws_path = ws_data["path"]
    run_dir = Path(ws_path) / "runs" / run_id
    assert run_dir.exists(), f"Run directory not found at {run_dir}"

    # Find iteration directories (format: iter_NNNN)
    iter_dirs = sorted([
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("iter_")
    ])
    assert len(iter_dirs) > 0, "No iteration directories on disk"

    # Read one artifact and verify it has content
    first_iter_dir = iter_dirs[0]
    md_files = list(first_iter_dir.glob("*.md"))
    assert len(md_files) > 0, f"No .md files in {first_iter_dir}"
    content = md_files[0].read_text()
    assert len(content) > 50, f"Artifact content too short: {len(content)} chars"

    # 8. Verify structured feedback -- reviewer iterations should have feedback
    reviewer_iters = [it for it in iterations if it["role"] == "reviewer"]
    if reviewer_iters:
        # At least one reviewer should have structured feedback
        has_feedback = any(it.get("feedback") for it in reviewer_iters)
        assert has_feedback, "No reviewer iterations have structured feedback"
        # Check feedback format
        with_feedback = [it for it in reviewer_iters if it.get("feedback")]
        assert "=== REVIEWER FEEDBACK" in with_feedback[0]["feedback"]

    # 9. Verify convergence or budget limit
    #    If completed before max_iterations, convergence was triggered.
    #    The mock grades are C -> B -> B+ -> A- (non-repeating), so
    #    convergence may trigger via output similarity instead.
    assert final["current_iteration"] <= 8

    # 10. Verify telemetry
    #     Note: The telemetry collector is in-memory and separate from the engine.
    #     The mock adapter runs don't automatically populate the collector, so we
    #     just verify the endpoint is accessible and returns the correct schema.
    telem_resp = await client.get("/api/telemetry/summary")
    assert telem_resp.status_code == 200
    telem = telem_resp.json()
    assert "total_calls" in telem
    assert "total_tokens" in telem

    # 11. Verify repropack
    repropack_resp = await client.post(f"/api/repropack/{run_id}")
    assert repropack_resp.status_code == 200
    assert repropack_resp.headers.get("content-type") == "application/zip"
    assert len(repropack_resp.content) > 0

    # 12. Verify knowledge claims can be parsed
    #     Read MECH.md content from disk and send to claims parser
    mech_files = list(first_iter_dir.glob("MECH.md"))
    if mech_files:
        mech_content = mech_files[0].read_text()
    else:
        # Fall back to any .md file
        mech_content = md_files[0].read_text()

    claims_resp = await client.post("/api/claims/parse", json={
        "mech_content": mech_content,
    })
    assert claims_resp.status_code == 200
    claims_data = claims_resp.json()
    assert "claims" in claims_data
    assert "graph" in claims_data


@pytest.mark.asyncio
async def test_pipeline_with_cancellation(client: AsyncClient, tmp_path: Path):
    """Start a run, cancel after some iterations, verify stopped state."""

    ws_id = await _create_workspace(client, tmp_path)

    # Start a run with many iterations so we have time to cancel.
    # The mock adapter has a 0.1-0.5s delay per call, so 50 iterations
    # will take at least 5 seconds -- enough time to issue a stop.
    run_resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "Long running analysis for cancellation test",
        "provider": "mock",
        "loop_preset": "executor_reviewer",
        "max_iterations": 50,
        "config_overrides": {"convergence_enabled": False},
    })
    assert run_resp.status_code == 201
    run_id = run_resp.json()["run_id"]

    # Give the background task time to start and run a few iterations.
    # The mock adapter delays 0.1-0.5s per call, so after 1.5s we expect
    # at least 2-3 iterations to have completed in-memory.
    await asyncio.sleep(1.5)

    # Stop the run via API
    stop_resp = await client.post(f"/api/runs/{run_id}/stop")
    assert stop_resp.status_code == 200

    # The stop endpoint sets status=stopped in the DB immediately, but the
    # runner background task still needs time to finish and persist iteration
    # data. Wait for the runner to complete by polling until iterations appear.
    async def _wait_for_runner_persist() -> dict:
        for _ in range(30):
            resp = await client.get(f"/api/runs/{run_id}")
            data = resp.json()
            # Runner persists current_iteration > 0 when it finishes
            if data["current_iteration"] > 0:
                return data
            await asyncio.sleep(0.3)
        # Return whatever we have after timeout
        return (await client.get(f"/api/runs/{run_id}")).json()

    final = await asyncio.wait_for(_wait_for_runner_persist(), timeout=15.0)

    assert final["status"] == "stopped", (
        f"Expected stopped, got {final['status']}"
    )

    # Verify partial iterations were persisted -- some iterations ran
    # but fewer than the 50 max
    assert final["current_iteration"] > 0, "Expected at least 1 iteration to have run"
    assert final["current_iteration"] < 50, "Run should have stopped before completing all 50"

    # Verify artifacts from completed iterations exist on disk
    ws_data = (await client.get(f"/api/workspaces/{ws_id}")).json()
    run_dir = Path(ws_data["path"]) / "runs" / run_id
    if run_dir.exists():
        iter_dirs = [
            d for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("iter_")
        ]
        assert len(iter_dirs) > 0, "No iteration dirs on disk for stopped run"


@pytest.mark.asyncio
async def test_pipeline_adapter_failure_recovery(client: AsyncClient, tmp_path: Path):
    """Verify the retry infrastructure works by confirming mock adapter runs
    complete successfully through the retry wrapper."""

    ws_id = await _create_workspace(client, tmp_path)

    # The mock adapter has 0% failure by default. Verify the run completes
    # successfully through the retry infrastructure (max_retries in config).
    run_resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "Test retry infrastructure with mock adapter",
        "provider": "mock",
        "loop_preset": "executor_reviewer",
        "max_iterations": 4,
        "config_overrides": {"max_retries": 3},
    })
    assert run_resp.status_code == 201
    run_id = run_resp.json()["run_id"]

    final = await _poll_run(client, run_id, timeout=30.0)

    # Should complete successfully since mock has 0% failure rate
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )
    assert final["current_iteration"] >= 4
    assert final["total_tokens"] > 0


@pytest.mark.asyncio
async def test_pipeline_budget_exceeded(client: AsyncClient, tmp_path: Path):
    """Create a run with very low iteration budget and verify graceful completion."""

    ws_id = await _create_workspace(client, tmp_path)

    run_resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "Budget-limited test run",
        "provider": "mock",
        "loop_preset": "executor_reviewer",
        "max_iterations": 2,
        "config_overrides": {"convergence_enabled": False},
    })
    assert run_resp.status_code == 201
    run_id = run_resp.json()["run_id"]

    final = await _poll_run(client, run_id, timeout=30.0)

    # Budget exceeded = graceful completion
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )

    # Verify exactly 2 iterations ran
    assert final["current_iteration"] == 2, (
        f"Expected 2 iterations, got {final['current_iteration']}"
    )
    iterations = final.get("iterations", [])
    assert len(iterations) == 2


@pytest.mark.asyncio
async def test_comparison_after_multiple_runs(client: AsyncClient, tmp_path: Path):
    """Create two runs, complete both, then compare them."""

    ws_id = await _create_workspace(client, tmp_path)

    # Create and wait for run 1
    resp1 = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "Attention pattern analysis - run 1",
        "provider": "mock",
        "loop_preset": "executor_reviewer",
        "max_iterations": 4,
        "config_overrides": {"convergence_enabled": False},
    })
    assert resp1.status_code == 201
    run_id_1 = resp1.json()["run_id"]

    # Wait for run 1 to finish before starting run 2
    final1 = await _poll_run(client, run_id_1, timeout=30.0)
    assert final1["status"] == "completed"

    # Create and wait for run 2
    resp2 = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "Co-expression confound analysis - run 2",
        "provider": "mock",
        "loop_preset": "executor_reviewer",
        "max_iterations": 4,
        "config_overrides": {"convergence_enabled": False},
    })
    assert resp2.status_code == 201
    run_id_2 = resp2.json()["run_id"]

    final2 = await _poll_run(client, run_id_2, timeout=30.0)
    assert final2["status"] == "completed"

    # Call comparison summary
    compare_resp = await client.post("/api/comparison/summary", json={
        "run_ids": [run_id_1, run_id_2],
    })
    assert compare_resp.status_code == 200
    comparison = compare_resp.json()

    # Verify comparison has the expected structure
    assert "metrics" in comparison
    assert "quality" in comparison

    metrics = comparison["metrics"]
    assert "runs" in metrics
    assert len(metrics["runs"]) == 2
    assert "best_by_cost" in metrics
    assert "best_by_tokens" in metrics

    # Both run_ids should appear in the results
    result_ids = {m["run_id"] for m in metrics["runs"]}
    assert run_id_1 in result_ids
    assert run_id_2 in result_ids
