"""Shared test fixtures for MI-Workbench backend tests."""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from typing import AsyncIterator, Iterable, Optional

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Point config at a temp directory before importing anything else
_tmpdir = tempfile.mkdtemp(prefix="miw_test_")
os.environ["MIW_DB_PATH"] = os.path.join(_tmpdir, "test.db")
os.environ["MIW_WORKSPACE_BASE"] = os.path.join(_tmpdir, "workspaces")
os.environ["MIW_PROMPTS_DIR"] = os.path.join(_tmpdir, "prompts")
os.environ["MIW_LOOPS_DIR"] = os.path.join(_tmpdir, "loops")

# Re-initialise config singleton with test paths
from backend.config import config
config.db_path = os.environ["MIW_DB_PATH"]
config.workspace_base_path = os.environ["MIW_WORKSPACE_BASE"]
config.prompts_dir = os.environ["MIW_PROMPTS_DIR"]
config.loops_dir = os.environ["MIW_LOOPS_DIR"]

from backend.database import init_db
from backend.main import app


# ── Background-task hygiene ──────────────────────────────────────────
#
# ``POST /api/runs`` (and resume/retry) start ``execute_run`` with
# ``asyncio.create_task`` and return at once. When a test finishes before that
# task does, pytest-asyncio closes the function-scoped event loop and cancels
# the task. If the cancellation lands inside ``backend.database.get_db``
# between ``aiosqlite.connect()`` and the ``try:`` (i.e. while awaiting
# ``PRAGMA foreign_keys=ON``), the connection is never closed. Its non-daemon
# ``_connection_worker_thread`` then blocks on its queue forever, and the
# pytest process never exits after the summary.
#
# The fix on the test side: before the loop closes, the ``client`` fixture asks
# every active run to stop (the same cancel event ``POST /api/runs/{id}/stop``
# uses) and waits until every task spawned during the test has finished, so
# every connection goes through its normal ``finally: await db.close()``.
# A session-end reaper is kept as a last-resort safety net: it unblocks any
# worker thread that is still orphaned and reports it in the terminal summary,
# so a leak stays visible and does not hang the process.

#: Seconds the ``client`` fixture waits for background tasks before hard-cancelling.
DRAIN_TIMEOUT_S = float(os.environ.get("MIW_TEST_DRAIN_TIMEOUT", "30"))

_AIOSQLITE_WORKER_NAME = "_connection_worker_thread"
_reaped_worker_threads: list[str] = []


async def drain_background_tasks(timeout: float = DRAIN_TIMEOUT_S) -> list[str]:
    """Stop and await all tasks spawned on the running loop (except the caller).

    Active runs are asked to stop cooperatively via ``runner.cancel_run`` and
    re-signalled on every poll, so runs that register their cancel event
    after the first poll are also caught. Tasks still pending after
    ``timeout`` seconds are hard-cancelled. Returns the names of the tasks
    that had to be hard-cancelled; an empty list means a clean drain.
    """
    from backend.orchestrator.runner import cancel_run, get_active_run_ids

    loop = asyncio.get_running_loop()
    me = asyncio.current_task()
    deadline = loop.time() + timeout
    pending: list[asyncio.Task] = []
    while True:
        pending = [t for t in asyncio.all_tasks(loop) if t is not me and not t.done()]
        if not pending:
            return []
        for run_id in get_active_run_ids():
            try:
                await cancel_run(run_id)
            except RuntimeError:  # stale event bound to an earlier, closed loop
                pass
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.wait(pending, timeout=min(0.05, remaining))
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return [t.get_name() for t in pending]


def _is_aiosqlite_worker(thread: threading.Thread) -> bool:
    target = getattr(thread, "_target", None)
    return getattr(target, "__name__", "") == _AIOSQLITE_WORKER_NAME


def alive_aiosqlite_workers() -> list[threading.Thread]:
    """Return the aiosqlite connection worker threads that are still alive."""
    return [t for t in threading.enumerate() if t.is_alive() and _is_aiosqlite_worker(t)]


def reap_orphaned_aiosqlite_threads(
    threads: Optional[Iterable[threading.Thread]] = None,
    join_timeout: float = 2.0,
) -> list[str]:
    """Unblock orphaned aiosqlite worker threads so the interpreter can exit.

    Each worker loops on ``tx.get()`` until it executes a function that returns
    aiosqlite's stop sentinel. Queueing ``(None, lambda: sentinel)`` makes it
    exit after any work already queued. This is only safe when no event loop
    still uses the connection, i.e. at session end or for a thread a test
    leaked on purpose. Returns the names of the threads that were signalled.
    """
    try:
        from aiosqlite.core import _STOP_RUNNING_SENTINEL
    except Exception:  # pragma: no cover - aiosqlite internals changed
        return []
    candidates = list(threads) if threads is not None else alive_aiosqlite_workers()
    signalled: list[threading.Thread] = []
    for thread in candidates:
        if not thread.is_alive() or not _is_aiosqlite_worker(thread):
            continue
        args = getattr(thread, "_args", ())
        if not args or not hasattr(args[0], "put_nowait"):
            continue
        args[0].put_nowait((None, lambda: _STOP_RUNNING_SENTINEL))
        signalled.append(thread)
    for thread in signalled:
        thread.join(timeout=join_timeout)
    return [t.name for t in signalled]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Last-resort reaper: never let a leaked aiosqlite thread hang pytest."""
    _reaped_worker_threads.extend(reap_orphaned_aiosqlite_threads())


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ANN001
    if _reaped_worker_threads:
        terminalreporter.write_sep(
            "=", "aiosqlite connection leak", yellow=True, bold=True
        )
        terminalreporter.write_line(
            f"{len(_reaped_worker_threads)} aiosqlite worker thread(s) were still "
            "alive at session end (an aiosqlite connection was never closed) and "
            f"were stopped by the conftest reaper: {', '.join(_reaped_worker_threads)}"
        )


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Provide an async HTTP client backed by the FastAPI app.

    On teardown, background runs started through the API are stopped and
    awaited before the event loop closes (see ``drain_background_tasks``).
    """
    # Init fresh DB for each test
    await init_db()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        await drain_background_tasks()
