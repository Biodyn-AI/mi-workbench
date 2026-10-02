"""Additive schema migration (revision 2), new persisted fields, and the
get_db connection-leak fix."""
from __future__ import annotations

import asyncio
from datetime import datetime

import aiosqlite
import pytest
import pytest_asyncio

from backend import database
from backend.config import config
from backend.database import (
    ADDITIVE_COLUMNS,
    ADDITIVE_SCHEMA_VERSION,
    create_iteration,
    create_run,
    create_workspace,
    get_db,
    get_iterations_for_run,
    get_run,
    get_schema_version,
    init_db,
    record_iteration_progress,
    update_iteration,
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


@pytest_asyncio.fixture
async def fresh_db(tmp_path):
    config.db_path = str(tmp_path / "db.sqlite")
    await init_db()
    ws = WorkspaceConfig(name="w", path=str(tmp_path / "ws"))
    await create_workspace(ws)
    yield ws


async def _columns(table):
    async with aiosqlite.connect(config.db_path) as db:
        cur = await db.execute(f"PRAGMA table_info({table})")
        return {r[1] for r in await cur.fetchall()}


@pytest.mark.asyncio
async def test_old_database_is_migrated_in_place(tmp_path):
    config.db_path = str(tmp_path / "old.sqlite")
    # A pre-revision database: original schema and rows written with old columns.
    async with aiosqlite.connect(config.db_path) as db:
        await db.executescript(database._SCHEMA)
        await db.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (1, ?)",
            (datetime.utcnow().isoformat(),))
        await db.execute(
            "INSERT INTO workspaces (id, name, path, created_at) VALUES ('w1','w','/x', ?)",
            (datetime.utcnow().isoformat(),))
        await db.execute(
            "INSERT INTO runs (run_id, workspace_id, created_at, status) VALUES ('r1','w1',?, 'completed')",
            (datetime.utcnow().isoformat(),))
        await db.execute(
            "INSERT INTO iterations (iteration_id, run_id, iteration_number, role, status) "
            "VALUES ('i1','r1',1,'executor','completed')")
        await db.commit()
    assert "stop_reason" not in await _columns("runs")

    await init_db()
    await init_db()  # idempotent

    runs_cols, it_cols = await _columns("runs"), await _columns("iterations")
    for name, _ in ADDITIVE_COLUMNS["runs"]:
        assert name in runs_cols
    for name, _ in ADDITIVE_COLUMNS["iterations"]:
        assert name in it_cols
    assert await get_schema_version() == ADDITIVE_SCHEMA_VERSION

    run = await get_run("r1")
    assert run.stop_reason is None and run.total_input_tokens == 0
    it = run.iterations[0]
    assert it.provider == "" and it.input_tokens == 0 and it.code_execution is None
    assert it.grade is None and it.consensus_report is None


@pytest.mark.asyncio
async def test_new_fields_round_trip(fresh_db):
    run = RunState(workspace_id=fresh_db.id, loop_preset="executor_reviewer", task="t",
                   provider=ProviderName.CODEX_CLI, model="gpt-5.5")
    await create_run(run)
    it = IterationResult(
        iteration_number=1, role="executor", status=IterationStatus.COMPLETED,
        provider="codex_cli", model="gpt-5.5", reasoning_effort="low",
        cli_version="codex-cli 0.145.0", input_tokens=1500, output_tokens=300,
        cached_input_tokens=1000, duration_seconds=12.5, token_usage=1800,
        code_execution={"executed": 1, "passed": 1, "failed": 0, "backend": "sandbox_exec"},
        grade="B+", critical_count=0, high_count=2,
        consensus_report={"panel_failed": False, "attempts": [{"attempt": 1}]},
    )
    await create_iteration(it, run.run_id)
    back = (await get_iterations_for_run(run.run_id))[0]
    for field in ("provider", "model", "reasoning_effort", "cli_version", "input_tokens",
                  "output_tokens", "cached_input_tokens", "duration_seconds",
                  "code_execution", "grade", "critical_count", "high_count",
                  "consensus_report"):
        assert getattr(back, field) == getattr(it, field), field

    upd = await update_iteration(it.iteration_id, grade="A", code_execution={"executed": 2})
    assert upd.grade == "A" and upd.code_execution == {"executed": 2}

    r = await update_run(run.run_id, status=RunStatus.COMPLETED,
                         stop_reason="converged:grade_stable", total_input_tokens=1500,
                         total_output_tokens=300, total_cached_input_tokens=1000)
    assert r.stop_reason == "converged:grade_stable"
    assert (r.total_input_tokens, r.total_output_tokens, r.total_cached_input_tokens) == (1500, 300, 1000)


@pytest.mark.asyncio
async def test_record_iteration_progress_single_transaction(fresh_db):
    run = RunState(workspace_id=fresh_db.id, loop_preset="x", task="t",
                   provider=ProviderName.MOCK, model="")
    await create_run(run)
    it = IterationResult(iteration_number=1, role="executor", status=IterationStatus.COMPLETED,
                         token_usage=10)
    await record_iteration_progress(it, run.run_id, current_iteration=1, total_tokens=10,
                                    total_input_tokens=7, total_output_tokens=3)
    r = await get_run(run.run_id)
    assert r.current_iteration == 1 and r.total_tokens == 10 and r.total_input_tokens == 7
    assert [i.iteration_id for i in r.iterations] == [it.iteration_id]


@pytest.mark.asyncio
async def test_get_db_closes_connection_when_cancelled_during_setup(fresh_db, monkeypatch):
    closed = []
    orig_execute = aiosqlite.Connection.execute
    orig_close = aiosqlite.Connection.close

    async def execute(self, sql, *args, **kwargs):
        if str(sql).startswith("PRAGMA foreign_keys"):
            raise asyncio.CancelledError()
        return await orig_execute(self, sql, *args, **kwargs)

    async def close(self):
        closed.append(self)
        return await orig_close(self)

    monkeypatch.setattr(aiosqlite.Connection, "execute", execute)
    monkeypatch.setattr(aiosqlite.Connection, "close", close)
    with pytest.raises(asyncio.CancelledError):
        async with get_db():
            pass  # pragma: no cover - never reached
    assert len(closed) == 1


@pytest.mark.asyncio
async def test_open_and_close_are_serialised(fresh_db, monkeypatch):
    """Workaround for the SQLite 3.51.0-3.51.1 unix-VFS deadlock: opening and
    closing connections never overlap (queries still run concurrently)."""
    state = {"inflight": 0, "peak": 0}
    orig_connect = aiosqlite.connect
    orig_close = aiosqlite.Connection.close

    async def _track(awaitable):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        try:
            await asyncio.sleep(0.005)
            return await awaitable
        finally:
            state["inflight"] -= 1

    def connect(path, *a, **kw):
        return _track(orig_connect(path, *a, **kw))

    async def close(self):
        return await _track(orig_close(self))

    monkeypatch.setattr(aiosqlite, "connect", connect)
    monkeypatch.setattr(aiosqlite.Connection, "close", close)

    async def worker():
        async with get_db() as db:
            cur = await db.execute("SELECT COUNT(*) FROM runs")
            await cur.fetchone()
            await asyncio.sleep(0.01)

    await asyncio.gather(*(worker() for _ in range(12)))
    assert state["peak"] == 1
    assert state["inflight"] == 0
