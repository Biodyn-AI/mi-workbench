"""Tests for API input validation and safety measures."""
from __future__ import annotations

import pytest
from pathlib import Path

from httpx import AsyncClient

from backend.api.runs import DENIED_CONFIG_KEYS, sanitize_config

pytestmark = pytest.mark.asyncio


# ── Helper ──────────────────────────────────────────────────────────

async def _create_workspace(client: AsyncClient, name: str, tmp_path: Path) -> str:
    """Create a workspace under the test's tmp_path and return its ID.

    The path must be absolute and temporary: tests that start a run write
    ``runs/<run_id>/`` into the workspace directory (a relative path used to
    create stray ``<name>-dir`` folders in the repository root).
    """
    resp = await client.post(
        "/api/workspaces", json={"name": name, "path": str(tmp_path / f"{name}-dir")}
    )
    assert resp.status_code == 201
    return resp.json()["id"]


# ── 1. Invalid workspace_id ─────────────────────────────────────────

async def test_create_run_invalid_workspace_id(client: AsyncClient) -> None:
    resp = await client.post("/api/runs", json={
        "workspace_id": "bad id with spaces!",
        "task": "test",
    })
    assert resp.status_code == 422


# ── 2. Task too long ────────────────────────────────────────────────

async def test_create_run_task_too_long(client: AsyncClient, tmp_path: Path) -> None:
    ws_id = await _create_workspace(client, "val-ws-long", tmp_path)
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "x" * 100_000,
    })
    assert resp.status_code == 422


# ── 3. Too many config keys ─────────────────────────────────────────

async def test_create_run_too_many_config_keys(client: AsyncClient, tmp_path: Path) -> None:
    ws_id = await _create_workspace(client, "val-ws-cfg", tmp_path)
    overrides = {f"key_{i}": i for i in range(100)}
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "test",
        "config_overrides": overrides,
    })
    assert resp.status_code == 422


# ── 4. max_iterations bounds ────────────────────────────────────────

async def test_create_run_max_iterations_bounds(client: AsyncClient, tmp_path: Path) -> None:
    ws_id = await _create_workspace(client, "val-ws-iter", tmp_path)
    # Too low
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "test",
        "max_iterations": 0,
    })
    assert resp.status_code == 422

    # Too high
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "test",
        "max_iterations": 10000,
    })
    assert resp.status_code == 422


# ── 5. SQL injection-like run_id ────────────────────────────────────

async def test_get_run_invalid_id(client: AsyncClient) -> None:
    resp = await client.get("/api/runs/'; DROP TABLE runs;--")
    assert resp.status_code == 422


# ── 6. Empty workspace name ─────────────────────────────────────────

async def test_create_workspace_empty_name(client: AsyncClient) -> None:
    resp = await client.post("/api/workspaces", json={
        "name": "",
        "path": "/tmp/test",
    })
    assert resp.status_code == 422


# ── 7. Empty workspace path ─────────────────────────────────────────

async def test_create_workspace_empty_path(client: AsyncClient) -> None:
    resp = await client.post("/api/workspaces", json={
        "name": "valid-name",
        "path": "",
    })
    assert resp.status_code == 422


# ── 8. Config sanitization ──────────────────────────────────────────

def test_config_sanitization() -> None:
    config = {
        "some_key": "ok",
        "follow_up_depth": 5,
        "follow_up_proposals": [{"title": "sneaky"}],
        "parent_run_id": "abc123",
        "another_key": "also_ok",
    }
    sanitized = sanitize_config(config)
    assert "some_key" in sanitized
    assert "another_key" in sanitized
    for denied in DENIED_CONFIG_KEYS:
        assert denied not in sanitized


# ── 9. Concurrent run limit ─────────────────────────────────────────

async def test_concurrent_run_limit(client: AsyncClient, tmp_path: Path) -> None:
    from backend.orchestrator.runner import MAX_CONCURRENT_RUNS, _active_runs
    import asyncio

    ws_id = await _create_workspace(client, "val-ws-conc", tmp_path)

    # Inject fake active runs to simulate the limit
    fake_events = {}
    for i in range(MAX_CONCURRENT_RUNS):
        fake_id = f"fake-run-{i:04d}"
        fake_events[fake_id] = asyncio.Event()
        _active_runs[fake_id] = fake_events[fake_id]

    try:
        resp = await client.post("/api/runs", json={
            "workspace_id": ws_id,
            "task": "one more run",
        })
        assert resp.status_code == 429
        assert "concurrent" in resp.json()["detail"].lower()
    finally:
        # Clean up fake runs
        for fake_id in fake_events:
            _active_runs.pop(fake_id, None)


# ── 10. Valid inputs pass ────────────────────────────────────────────

async def test_valid_inputs_pass(client: AsyncClient, tmp_path: Path) -> None:
    ws_id = await _create_workspace(client, "val-ws-ok", tmp_path)
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "A perfectly normal task",
        "provider": "mock",
        "max_iterations": 10,
        "config_overrides": {"key1": "val1", "key2": 42},
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["workspace_id"] == ws_id
    assert data["task"] == "A perfectly normal task"
    assert data["max_iterations"] == 10
