"""SQLite write robustness under lock contention.

Reproduces the "database is locked" failures of the overhead experiment (many
concurrent runs, a slow / loaded disk): a write lock held by another connection
for longer than the busy timeout, and many concurrent writers with a
deliberately tiny busy timeout. Checks that the busy timeout is applied to
every connection, that write transactions are retried with backoff and
succeed, that a write that still fails raises ``DatabaseLockError`` without
partial writes, and that the runner never loses an iteration (deferred writes
are persisted later and recorded as warnings) or fails a run because of
transient contention.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest
import pytest_asyncio

from backend import database
from backend.adapters.mock import MockAdapter
from backend.config import config
from backend.database import (
    DatabaseLockError,
    create_run,
    create_workspace,
    get_db,
    get_run,
    init_db,
    is_lock_error,
    record_iteration_progress,
    update_run,
    update_run_if_status,
)
from backend.models import (
    IterationResult,
    IterationStatus,
    ProviderName,
    RunState,
    RunStatus,
    WorkspaceConfig,
)
from backend.orchestrator import runner
from backend.orchestrator.runner import _active_tasks, execute_run

pytestmark = pytest.mark.asyncio

_SETTINGS = ("db_path", "db_busy_timeout", "db_write_attempts", "db_retry_base_delay",
             "db_retry_max_delay", "db_retry_deadline")


@pytest_asyncio.fixture(autouse=True)
async def db(tmp_path):
    saved = {k: getattr(config, k) for k in _SETTINGS}
    config.db_path = str(tmp_path / "contention.db")
    await init_db()
    MockAdapter.reset_iteration_count()
    yield
    for k, v in saved.items():
        setattr(config, k, v)


def _retry_fast(attempts=40, base=0.005, cap=0.05, deadline=60.0, busy=0.05):
    config.db_busy_timeout = busy
    config.db_write_attempts = attempts
    config.db_retry_base_delay = base
    config.db_retry_max_delay = cap
    config.db_retry_deadline = deadline


class _LockHolder:
    """Holds SQLite's write lock from a separate (non-aiosqlite) connection."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, timeout=10, check_same_thread=False,
                                    isolation_level=None)

    def acquire(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")

    def release(self) -> None:
        if self.conn.in_transaction:
            self.conn.execute("COMMIT")

    def close(self) -> None:
        self.release()
        self.conn.close()


async def _workspace_and_run(tmp_path, **run_kw):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir(exist_ok=True)
    ws = WorkspaceConfig(name="w", path=str(ws_dir), default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    run = RunState(workspace_id=ws.id, loop_preset=run_kw.pop("preset", "executor_reviewer"),
                   task="probe", provider=ProviderName.MOCK, model="",
                   max_iterations=run_kw.pop("max_iterations", 4),
                   config={"convergence_enabled": False, **run_kw.pop("cfg", {})})
    await create_run(run)
    return ws, run, ws_dir / "runs" / run.run_id


def _iteration(n: int) -> IterationResult:
    return IterationResult(iteration_number=n, role="executor",
                           status=IterationStatus.COMPLETED, token_usage=n)


# ── connection settings ─────────────────────────────────────────────


async def test_every_connection_gets_the_configured_busy_timeout():
    config.db_busy_timeout = 12.5
    async with get_db() as db:
        row = await (await db.execute("PRAGMA busy_timeout")).fetchone()
    assert row[0] == 12500
    db = await database.open_connection()
    try:
        row = await (await db.execute("PRAGMA busy_timeout")).fetchone()
        assert row[0] == 12500
    finally:
        await database.close_connection(db)


async def test_default_busy_timeout_is_30_seconds(monkeypatch):
    from backend.config import AppConfig
    monkeypatch.delenv("MIW_DB_BUSY_TIMEOUT", raising=False)
    assert AppConfig().db_busy_timeout == 30.0
    monkeypatch.setenv("MIW_DB_BUSY_TIMEOUT", "7.5")
    assert AppConfig().db_busy_timeout == 7.5
    monkeypatch.setenv("MIW_DB_BUSY_TIMEOUT", "soon")
    with pytest.raises(ValueError):
        AppConfig()


def test_is_lock_error():
    assert is_lock_error(sqlite3.OperationalError("database is locked"))
    assert is_lock_error(sqlite3.OperationalError("database is busy"))
    assert not is_lock_error(sqlite3.OperationalError("no such table: x"))
    assert not is_lock_error(sqlite3.IntegrityError("database is locked"))
    assert not is_lock_error(ValueError("database is locked"))


# ── a write lock held longer than the busy timeout ──────────────────


async def test_writes_wait_out_a_lock_held_longer_than_the_busy_timeout(tmp_path):
    ws, run, _ = await _workspace_and_run(tmp_path)
    _retry_fast(busy=0.05)
    holder = _LockHolder(config.db_path)
    retries: list[int] = []
    real_backoff = database._backoff_delay

    def spy_backoff(attempt, base, cap):
        retries.append(attempt)
        return real_backoff(attempt, base, cap)

    try:
        holder.acquire()
        asyncio.get_running_loop().call_later(0.6, holder.release)
        t0 = time.monotonic()
        with patch.object(database, "_backoff_delay", spy_backoff):
            marked, _ = await asyncio.gather(
                update_run_if_status(run.run_id, RunStatus.PENDING, status=RunStatus.RUNNING),
                record_iteration_progress(_iteration(1), run.run_id, current_iteration=1,
                                          total_tokens=1),
            )
        waited = time.monotonic() - t0
    finally:
        holder.close()
    assert marked is True
    assert waited >= 0.5            # really blocked by the held lock ...
    assert retries                  # ... and got through by retrying
    final = await get_run(run.run_id)
    assert final.status == RunStatus.RUNNING and final.current_iteration == 1
    assert [i.iteration_number for i in final.iterations] == [1]


async def test_compare_and_set_is_reevaluated_on_retry(tmp_path):
    """A /stop that commits while the CAS is waiting for the lock wins."""
    ws, run, _ = await _workspace_and_run(tmp_path)
    _retry_fast(busy=0.05)
    holder = _LockHolder(config.db_path)
    try:
        holder.acquire()
        holder.conn.execute("UPDATE runs SET status = 'stopped' WHERE run_id = ?", (run.run_id,))
        asyncio.get_running_loop().call_later(0.3, holder.release)
        marked = await update_run_if_status(run.run_id, RunStatus.PENDING,
                                            status=RunStatus.RUNNING)
    finally:
        holder.close()
    assert marked is False
    assert (await get_run(run.run_id)).status == RunStatus.STOPPED


async def test_gives_up_with_database_lock_error_and_writes_nothing(tmp_path):
    ws, run, _ = await _workspace_and_run(tmp_path)
    _retry_fast(attempts=3, base=0.01, cap=0.02, busy=0.02)
    holder = _LockHolder(config.db_path)
    try:
        holder.acquire()
        with pytest.raises(DatabaseLockError) as info:
            await record_iteration_progress(_iteration(1), run.run_id, current_iteration=1)
    finally:
        holder.close()
    assert info.value.attempts == 3
    assert isinstance(info.value, sqlite3.OperationalError)   # old handlers still match
    final = await get_run(run.run_id)
    assert final.iterations == [] and final.current_iteration == 0   # rolled back


async def test_retry_deadline_bounds_the_total_wait(tmp_path):
    ws, run, _ = await _workspace_and_run(tmp_path)
    _retry_fast(attempts=1000, base=0.01, cap=0.01, busy=0.01, deadline=0.3)
    holder = _LockHolder(config.db_path)
    try:
        holder.acquire()
        t0 = time.monotonic()
        with pytest.raises(DatabaseLockError):
            await update_run(run.run_id, current_iteration=5)
        elapsed = time.monotonic() - t0
    finally:
        holder.close()
    assert elapsed < 3.0


async def test_non_lock_errors_are_not_retried(tmp_path):
    _retry_fast()
    calls = []

    async def bad(db):
        calls.append(1)
        await db.execute("INSERT INTO no_such_table VALUES (1)")

    with pytest.raises(sqlite3.OperationalError) as info:
        await database.write_transaction(bad)
    assert not isinstance(info.value, DatabaseLockError)
    assert calls == [1]


# ── many concurrent writers with a tiny busy timeout ────────────────


def _hammer(path: str, stop: threading.Event, hold_s: float = 0.01, gap_s: float = 0.015):
    """Repeatedly take and hold the write lock from another thread."""
    conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    try:
        while not stop.is_set():
            conn.execute("BEGIN IMMEDIATE")
            time.sleep(hold_s)
            conn.execute("COMMIT")
            time.sleep(gap_s)
    finally:
        conn.close()


async def _concurrent_writers(tmp_path, n_runs: int, iters_per_run: int):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir(exist_ok=True)
    ws = WorkspaceConfig(name="w", path=str(ws_dir))
    await create_workspace(ws)
    runs = []
    for _ in range(n_runs):
        r = RunState(workspace_id=ws.id, loop_preset="x", task="t",
                     provider=ProviderName.MOCK, model="")
        await create_run(r)
        runs.append(r)

    async def one_run(r: RunState):
        assert await update_run_if_status(r.run_id, RunStatus.PENDING, status=RunStatus.RUNNING)
        for n in range(1, iters_per_run + 1):
            await record_iteration_progress(_iteration(n), r.run_id, current_iteration=n,
                                            total_tokens=n)
        await update_run(r.run_id, status=RunStatus.COMPLETED)

    stop = threading.Event()
    t = threading.Thread(target=_hammer, args=(config.db_path, stop), daemon=True)
    t.start()
    try:
        # return_exceptions: every writer finishes (no task outlives the test)
        results = await asyncio.gather(*(one_run(r) for r in runs), return_exceptions=True)
    finally:
        stop.set()
        t.join(10)
    return runs, [r for r in results if isinstance(r, BaseException)]


async def test_contention_reproduces_database_is_locked_without_retries(tmp_path):
    """Control: with a tiny busy timeout and no retries the bug reproduces."""
    _retry_fast(attempts=1, busy=0.001)
    _, errors = await _concurrent_writers(tmp_path, n_runs=20, iters_per_run=6)
    assert errors, "expected 'database is locked' without retries"
    assert all(isinstance(e, sqlite3.OperationalError) and is_lock_error(e) for e in errors)


async def test_many_concurrent_writers_lose_nothing(tmp_path):
    _retry_fast(attempts=400, base=0.002, cap=0.03, busy=0.001, deadline=120.0)
    retries = []
    real_backoff = database._backoff_delay

    def spy_backoff(attempt, base, cap):
        retries.append(attempt)
        return real_backoff(attempt, base, cap)

    with patch.object(database, "_backoff_delay", spy_backoff):
        runs, errors = await _concurrent_writers(tmp_path, n_runs=20, iters_per_run=6)
    assert errors == []
    assert retries, "the test did not produce lock contention"
    for r in runs:
        final = await get_run(r.run_id)
        assert final.status == RunStatus.COMPLETED
        assert [i.iteration_number for i in final.iterations] == list(range(1, 7))
        assert final.current_iteration == 6 and final.total_tokens == 6


# ── runner: deferred iteration writes, no failed:exception ──────────


def _adapter():
    return MockAdapter(min_delay=0, max_delay=0)


async def test_run_completes_although_the_write_lock_is_held_at_start(tmp_path):
    """The overhead-experiment failure: update_run_if_status hit a lock held
    longer than the busy timeout and the run failed with failed:exception."""
    ws, run, run_dir = await _workspace_and_run(tmp_path, max_iterations=3)
    _retry_fast(busy=0.05)
    holder = _LockHolder(config.db_path)
    try:
        holder.acquire()
        asyncio.get_running_loop().call_later(0.5, holder.release)
        with patch.object(runner, "get_adapter", return_value=_adapter()):
            task = asyncio.create_task(execute_run(run.run_id))
            _active_tasks[run.run_id] = task
            await asyncio.wait_for(task, 60)
    finally:
        holder.close()
    final = await get_run(run.run_id)
    assert final.status == RunStatus.COMPLETED, (final.stop_reason, final.error)
    assert final.stop_reason == "max_iterations"
    assert [i.iteration_number for i in final.iterations] == [1, 2, 3]


async def test_failed_progress_writes_are_deferred_not_lost(tmp_path):
    ws, run, run_dir = await _workspace_and_run(tmp_path, max_iterations=4)
    real = database.record_iterations_progress
    calls: list[list[int]] = []
    events: list[tuple[str, dict]] = []

    async def flaky(its, run_id, telemetry=None, **fields):
        calls.append([i.iteration_number for i in its])
        if len(calls) <= 2:   # the first two incremental writes fail
            raise DatabaseLockError("database is locked (test)", attempts=8, waited_seconds=1.0)
        return await real(its, run_id, telemetry=telemetry, **fields)

    async def broadcast(run_id, event):
        events.append((run_id, event))

    old_ws = runner._ws_broadcast
    runner.set_ws_broadcast(broadcast)
    try:
        with patch.object(runner, "record_iterations_progress", flaky), \
             patch.object(runner, "get_adapter", return_value=_adapter()):
            await execute_run(run.run_id)
            await asyncio.sleep(0.05)   # let the broadcast tasks run
    finally:
        runner._ws_broadcast = old_ws

    # iteration 1 failed, then 1+2 failed together, then 1+2+3 were written
    assert calls[:3] == [[1], [1, 2], [1, 2, 3]]
    final = await get_run(run.run_id)
    assert final.status == RunStatus.COMPLETED and final.stop_reason == "max_iterations"
    assert [i.iteration_number for i in final.iterations] == [1, 2, 3, 4]
    assert final.current_iteration == 4
    assert final.total_tokens == sum(i.token_usage for i in final.iterations)
    logs = {i.iteration_number: i.logs for i in final.iterations}
    assert any("[persistence]" in line for line in logs[1])
    assert any("[persistence]" in line for line in logs[2])
    warnings = [e for _, e in events if e.get("type") == "persistence_warning"]
    assert [w["iteration"] for w in warnings] == [1, 2]
    assert warnings[1]["deferred_iterations"] == [1, 2]
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert [w["iteration"] for w in meta["persistence_warnings"]] == [1, 2]
    assert meta["total_iterations"] == 4 and len(meta["iterations"]) == 4


async def test_deferred_iterations_and_telemetry_are_flushed_at_run_end(tmp_path):
    ws, run, run_dir = await _workspace_and_run(tmp_path, max_iterations=3)
    real = database.record_iterations_progress

    async def incremental_always_fails(its, run_id, telemetry=None, **fields):
        if "current_iteration" in fields:   # every incremental write
            raise DatabaseLockError("database is locked (test)", attempts=8, waited_seconds=1.0)
        return await real(its, run_id, telemetry=telemetry, **fields)

    with patch.object(runner, "record_iterations_progress", incremental_always_fails), \
         patch.object(runner, "get_adapter", return_value=_adapter()):
        await execute_run(run.run_id)

    final = await get_run(run.run_id)
    assert final.status == RunStatus.COMPLETED
    assert [i.iteration_number for i in final.iterations] == [1, 2, 3]
    assert final.current_iteration == 3   # final run update
    from backend.telemetry.storage import get_records
    recs = await get_records(run_id=run.run_id)
    assert len(recs) == 3   # one call per iteration (executor, reviewer, executor)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert [w["iteration"] for w in meta["persistence_warnings"]] == [1, 2, 3]


async def test_concurrent_runs_with_tiny_busy_timeout_all_complete(tmp_path):
    """20 concurrent runs through execute_run while another connection keeps
    taking the write lock: every run completes and no iteration is lost."""
    _retry_fast(attempts=400, base=0.002, cap=0.03, busy=0.001, deadline=120.0)
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    ws = WorkspaceConfig(name="w", path=str(ws_dir), default_provider=ProviderName.MOCK)
    config.db_busy_timeout = 0.05
    await create_workspace(ws)
    runs = []
    for _ in range(20):
        r = RunState(workspace_id=ws.id, loop_preset="executor_reviewer", task="t",
                     provider=ProviderName.MOCK, model="", max_iterations=4,
                     config={"convergence_enabled": False})
        await create_run(r)
        runs.append(r)
    config.db_busy_timeout = 0.001
    retries = []
    real_backoff = database._backoff_delay

    def spy_backoff(attempt, base, cap):
        retries.append(attempt)
        return real_backoff(attempt, base, cap)

    stop = threading.Event()
    t = threading.Thread(target=_hammer, args=(config.db_path, stop), daemon=True)
    t.start()
    try:
        with patch.object(runner, "get_adapter", return_value=_adapter()), \
             patch.object(database, "_backoff_delay", spy_backoff):
            tasks = []
            for r in runs:
                task = asyncio.create_task(execute_run(r.run_id))
                _active_tasks[r.run_id] = task
                tasks.append(task)
            await asyncio.wait_for(asyncio.gather(*tasks), 300)
    finally:
        stop.set()
        t.join(10)
    assert retries, "the test did not produce lock contention"
    for r in runs:
        final = await get_run(r.run_id)
        assert final.status == RunStatus.COMPLETED, (final.stop_reason, final.error)
        assert [i.iteration_number for i in final.iterations] == [1, 2, 3, 4]
        assert final.current_iteration == 4
