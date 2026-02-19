"""Stress tests for concurrent run execution and DB integrity."""
from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


# ── Helpers ──────────────────────────────────────────────────────────


async def _create_workspace(client: AsyncClient, name: str = "stress-ws") -> str:
    """Create a workspace and return its id."""
    resp = await client.post("/api/workspaces", json={"name": name, "path": f"/tmp/{name}"})
    assert resp.status_code == 201
    return resp.json()["id"]


async def _launch_run(
    client: AsyncClient,
    ws_id: str,
    task: str,
    max_iterations: int = 4,
) -> str:
    """Launch a run and return its run_id."""
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": task,
        "provider": "mock",
        "max_iterations": max_iterations,
        "config_overrides": {"convergence_enabled": False},
    })
    assert resp.status_code == 201
    return resp.json()["run_id"]


async def _wait_for_run(
    client: AsyncClient,
    run_id: str,
    timeout: float = 30.0,
) -> dict:
    """Poll until a run reaches a terminal status or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed", "stopped"):
            return data
        await asyncio.sleep(0.2)
    raise TimeoutError(f"Run {run_id} did not complete in {timeout}s")


# ── Test 1: 5 concurrent runs ───────────────────────────────────────


async def test_concurrent_runs_5(client: AsyncClient) -> None:
    """Launch 5 concurrent runs and verify all complete correctly."""
    ws_id = await _create_workspace(client, "stress-5")

    run_ids = []
    for i in range(5):
        rid = await _launch_run(client, ws_id, f"Stress test task {i}")
        run_ids.append(rid)

    results = await asyncio.gather(*[_wait_for_run(client, rid) for rid in run_ids])

    for result in results:
        assert result["status"] == "completed", (
            f"Run {result['run_id']} status: {result['status']}, "
            f"error: {result.get('error')}"
        )
        assert result["current_iteration"] == 4
        assert result["total_tokens"] > 0
        assert result["total_cost"] > 0


# ── Test 2: 10 concurrent runs ──────────────────────────────────────


async def test_concurrent_runs_10(client: AsyncClient) -> None:
    """Launch 10 concurrent runs to stress the SQLite layer."""
    ws_id = await _create_workspace(client, "stress-10")

    run_ids = []
    for i in range(10):
        rid = await _launch_run(client, ws_id, f"Stress 10 task {i}")
        run_ids.append(rid)

    results = await asyncio.gather(
        *[_wait_for_run(client, rid, timeout=60.0) for rid in run_ids]
    )

    completed = [r for r in results if r["status"] == "completed"]
    assert len(completed) == 10, (
        f"Expected 10 completed, got {len(completed)}. "
        f"Statuses: {[r['status'] for r in results]}"
    )
    for result in completed:
        assert result["current_iteration"] == 4
        assert result["total_tokens"] > 0


# ── Test 3: concurrent create and cancel ─────────────────────────────


async def test_concurrent_create_and_cancel(client: AsyncClient) -> None:
    """Launch 5 runs, cancel 3, verify correct outcomes."""
    ws_id = await _create_workspace(client, "stress-cancel")

    run_ids = []
    for i in range(5):
        rid = await _launch_run(client, ws_id, f"Cancel test {i}", max_iterations=20)
        run_ids.append(rid)

    # Give runs a moment to start
    await asyncio.sleep(0.3)

    # Cancel the first 3
    cancel_ids = run_ids[:3]
    keep_ids = run_ids[3:]
    for rid in cancel_ids:
        resp = await client.post(f"/api/runs/{rid}/stop")
        assert resp.status_code in (200, 400)  # 400 if already terminal

    # Wait for non-cancelled runs to finish
    keep_results = await asyncio.gather(
        *[_wait_for_run(client, rid, timeout=30.0) for rid in keep_ids]
    )
    for result in keep_results:
        assert result["status"] == "completed", (
            f"Non-cancelled run {result['run_id']} should complete, "
            f"got {result['status']}"
        )

    # Verify cancelled runs are stopped (or already completed if they finished first)
    for rid in cancel_ids:
        resp = await client.get(f"/api/runs/{rid}")
        assert resp.status_code == 200
        status = resp.json()["status"]
        assert status in ("stopped", "completed"), (
            f"Cancelled run {rid} should be stopped or completed, got {status}"
        )

    # Verify no DB corruption: all runs are fetchable
    for rid in run_ids:
        resp = await client.get(f"/api/runs/{rid}")
        assert resp.status_code == 200


# ── Test 4: DB integrity after stress ────────────────────────────────


async def test_db_not_corrupted_after_stress(client: AsyncClient) -> None:
    """Run multiple concurrent operations and verify DB consistency."""
    ws_id = await _create_workspace(client, "stress-integrity")

    run_ids = []
    for i in range(5):
        rid = await _launch_run(client, ws_id, f"Integrity task {i}")
        run_ids.append(rid)

    results = await asyncio.gather(
        *[_wait_for_run(client, rid) for rid in run_ids]
    )

    # All runs should be listable
    resp = await client.get("/api/runs", params={"workspace_id": ws_id})
    assert resp.status_code == 200
    listed = resp.json()
    listed_ids = {r["run_id"] for r in listed}
    for rid in run_ids:
        assert rid in listed_ids, f"Run {rid} not in list response"

    # Verify iteration counts match expected
    for result in results:
        if result["status"] == "completed":
            assert result["current_iteration"] == 4

    # Verify per-run token/cost totals are consistent
    for result in results:
        if result["status"] == "completed":
            resp = await client.get(f"/api/runs/{result['run_id']}")
            assert resp.status_code == 200
            run_data = resp.json()
            # Each iteration produces tokens; sum should match total
            iter_tokens = sum(it["token_usage"] for it in run_data.get("iterations", []))
            assert iter_tokens == run_data["total_tokens"], (
                f"Run {run_data['run_id']}: iteration token sum {iter_tokens} "
                f"!= total_tokens {run_data['total_tokens']}"
            )


# ── Test 5: long run completes without issues ───────────────────────


async def test_memory_bounded_long_run(client: AsyncClient) -> None:
    """Run a single long run (20 iterations) and verify completion."""
    ws_id = await _create_workspace(client, "stress-long")

    rid = await _launch_run(client, ws_id, "Long stress task", max_iterations=20)
    result = await _wait_for_run(client, rid, timeout=60.0)

    assert result["status"] == "completed"
    assert result["current_iteration"] == 20
    assert result["total_tokens"] > 0
    assert result["total_cost"] > 0


# ── Test 6: rapid create-stop cycles ─────────────────────────────────


async def test_rapid_create_stop_create(client: AsyncClient) -> None:
    """Rapidly create and stop runs in 10 cycles; verify no orphans."""
    ws_id = await _create_workspace(client, "stress-rapid")

    all_run_ids = []
    for i in range(10):
        # Create with enough iterations that it won't finish instantly
        rid = await _launch_run(client, ws_id, f"Rapid cycle {i}", max_iterations=50)
        all_run_ids.append(rid)
        # Immediately stop
        await asyncio.sleep(0.05)  # tiny delay to let the task start
        resp = await client.post(f"/api/runs/{rid}/stop")
        # May be 200 or 400 if already finished
        assert resp.status_code in (200, 400)

    # Wait briefly for any stragglers
    await asyncio.sleep(1.0)

    # Verify all runs have a terminal status
    for rid in all_run_ids:
        resp = await client.get(f"/api/runs/{rid}")
        assert resp.status_code == 200
        status = resp.json()["status"]
        assert status in ("completed", "stopped", "failed"), (
            f"Run {rid} has non-terminal status: {status}"
        )

    # Verify all runs are listable (no orphans)
    resp = await client.get("/api/runs", params={"workspace_id": ws_id})
    assert resp.status_code == 200
    listed_ids = {r["run_id"] for r in resp.json()}
    for rid in all_run_ids:
        assert rid in listed_ids, f"Run {rid} missing from list (orphaned?)"
