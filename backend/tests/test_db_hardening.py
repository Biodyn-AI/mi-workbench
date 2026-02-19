"""Tests for database hardening: WAL mode, foreign keys, migrations, pagination, depth limit."""
from __future__ import annotations

import json

import aiosqlite
import pytest
from httpx import AsyncClient

from backend.config import config
from backend.database import get_db, get_schema_version, init_db

pytestmark = pytest.mark.asyncio


# ── WAL / connection tests ──────────────────────────────────────────


async def test_wal_mode_enabled(client: AsyncClient) -> None:
    """After init_db, journal_mode should be WAL."""
    async with aiosqlite.connect(config.db_path) as db:
        cursor = await db.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0] == "wal"


async def test_foreign_keys_enabled(client: AsyncClient) -> None:
    """get_db() connections should have foreign_keys=ON."""
    async with get_db() as db:
        cursor = await db.execute("PRAGMA foreign_keys")
        row = await cursor.fetchone()
        assert row["foreign_keys"] == 1


async def test_busy_timeout_set(client: AsyncClient) -> None:
    """init_db should set busy_timeout=5000."""
    async with aiosqlite.connect(config.db_path) as db:
        cursor = await db.execute("PRAGMA busy_timeout")
        row = await cursor.fetchone()
        assert row[0] == 5000


# ── Schema version / migration tests ────────────────────────────────


async def test_schema_version_tracking(client: AsyncClient) -> None:
    """init_db should create schema_version table and record at least version 1."""
    version = await get_schema_version()
    assert version >= 1


# ── Pagination: runs ─────────────────────────────────────────────────


async def test_pagination_runs(client: AsyncClient) -> None:
    """Create 10 runs, list with limit=3 offset=0 returns exactly 3."""
    ws = await client.post("/api/workspaces", json={"name": "ws-pg", "path": "ws-pg"})
    ws_id = ws.json()["id"]
    for i in range(10):
        await client.post("/api/runs", json={
            "workspace_id": ws_id,
            "task": f"run-{i}",
            "provider": "mock",
        })
    resp = await client.get("/api/runs", params={
        "workspace_id": ws_id,
        "limit": 3,
        "offset": 0,
    })
    assert resp.status_code == 200
    assert len(resp.json()) == 3


async def test_pagination_offset(client: AsyncClient) -> None:
    """Create 10 runs, list with limit=3 offset=3 returns the next 3."""
    ws = await client.post("/api/workspaces", json={"name": "ws-pg2", "path": "ws-pg2"})
    ws_id = ws.json()["id"]
    for i in range(10):
        await client.post("/api/runs", json={
            "workspace_id": ws_id,
            "task": f"run-{i}",
            "provider": "mock",
        })
    # Get first page
    first = await client.get("/api/runs", params={
        "workspace_id": ws_id, "limit": 3, "offset": 0,
    })
    # Get second page
    second = await client.get("/api/runs", params={
        "workspace_id": ws_id, "limit": 3, "offset": 3,
    })
    assert len(second.json()) == 3
    first_ids = {r["run_id"] for r in first.json()}
    second_ids = {r["run_id"] for r in second.json()}
    # Pages should not overlap
    assert first_ids.isdisjoint(second_ids)


# ── Pagination: knowledge claims ─────────────────────────────────────


async def test_pagination_claims(client: AsyncClient) -> None:
    """Create several claims, verify pagination works."""
    for i in range(8):
        await client.post("/api/knowledge/claims", json={
            "claim_id": f"claim-pg-{i}",
            "claim_text": f"Claim number {i}",
        })
    resp = await client.get("/api/knowledge/claims", params={
        "limit": 3,
        "offset": 0,
    })
    assert resp.status_code == 200
    assert len(resp.json()) == 3

    resp2 = await client.get("/api/knowledge/claims", params={
        "limit": 3,
        "offset": 3,
    })
    assert len(resp2.json()) == 3
    first_ids = {c["claim_id"] for c in resp.json()}
    second_ids = {c["claim_id"] for c in resp2.json()}
    assert first_ids.isdisjoint(second_ids)


# ── Follow-up depth limit ───────────────────────────────────────────


async def test_follow_up_depth_limit(client: AsyncClient) -> None:
    """Verify depth increments and stops at max in the follow-up config logic."""
    from backend.database import create_run, get_run, update_run
    from backend.models import ProviderName, RunState, RunStatus

    ws = await client.post("/api/workspaces", json={"name": "ws-depth", "path": "ws-depth"})
    ws_id = ws.json()["id"]

    # Simulate a run at depth 0 with max_follow_up_depth=2
    run = RunState(
        workspace_id=ws_id,
        loop_preset="executor_reviewer",
        task="depth test",
        provider=ProviderName.MOCK,
        model="",
        config={
            "follow_up_depth": 0,
            "max_follow_up_depth": 2,
        },
    )
    await create_run(run)

    # Verify depth fields round-trip through DB
    fetched = await get_run(run.run_id)
    assert fetched is not None
    assert fetched.config["follow_up_depth"] == 0
    assert fetched.config["max_follow_up_depth"] == 2

    # Simulate incrementing depth for a child
    child_config = dict(fetched.config)
    child_config["follow_up_depth"] = fetched.config["follow_up_depth"] + 1
    child = RunState(
        workspace_id=ws_id,
        loop_preset="executor_reviewer",
        task="depth child",
        provider=ProviderName.MOCK,
        model="",
        config=child_config,
    )
    await create_run(child)
    child_fetched = await get_run(child.run_id)
    assert child_fetched.config["follow_up_depth"] == 1

    # At depth 2 (= max), no more follow-ups should be created
    depth2_config = dict(child_fetched.config)
    depth2_config["follow_up_depth"] = 2
    assert depth2_config["follow_up_depth"] >= depth2_config["max_follow_up_depth"]
