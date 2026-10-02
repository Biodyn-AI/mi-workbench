"""SQLite database layer using aiosqlite."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import sqlite3
import time
import weakref
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypeVar

import aiosqlite

from backend.config import config
from backend.models import (
    GitMode,
    IterationResult,
    IterationStatus,
    ProviderName,
    RunState,
    RunStatus,
    WorkspaceConfig,
)

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    providers_enabled TEXT NOT NULL DEFAULT '[]',
    default_provider TEXT NOT NULL DEFAULT 'mock',
    default_model TEXT NOT NULL DEFAULT '',
    git_mode TEXT NOT NULL DEFAULT 'none',
    budget_max_tokens_per_run INTEGER NOT NULL DEFAULT 500000,
    budget_max_cost_per_run REAL NOT NULL DEFAULT 10.0,
    budget_max_iterations INTEGER NOT NULL DEFAULT 50,
    safety_deny_destructive_commands INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    loop_preset TEXT NOT NULL DEFAULT 'executor_reviewer',
    task TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT 'mock',
    model TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    current_iteration INTEGER NOT NULL DEFAULT 0,
    max_iterations INTEGER NOT NULL DEFAULT 50,
    created_at TEXT NOT NULL,
    started_at TEXT,
    stopped_at TEXT,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0.0,
    config TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
);

CREATE TABLE IF NOT EXISTS iterations (
    iteration_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    iteration_number INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    completed_at TEXT,
    artifacts_produced TEXT NOT NULL DEFAULT '[]',
    logs TEXT NOT NULL DEFAULT '[]',
    token_usage INTEGER NOT NULL DEFAULT 0,
    cost_estimate REAL NOT NULL DEFAULT 0.0,
    output_summary TEXT NOT NULL DEFAULT '',
    feedback TEXT,
    error TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS knowledge_claims (
    claim_id TEXT PRIMARY KEY,
    run_id TEXT,
    claim_text TEXT NOT NULL,
    evidence_pointers TEXT NOT NULL DEFAULT '[]',
    uncertainty REAL NOT NULL DEFAULT 0.5,
    strength REAL NOT NULL DEFAULT 0.0,
    falsification_tests TEXT NOT NULL DEFAULT '[]',
    source_artifact TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS knowledge_facts (
    fact_id TEXT PRIMARY KEY,
    claim_id TEXT,
    fact_text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES knowledge_claims(claim_id)
);

CREATE TABLE IF NOT EXISTS knowledge_links (
    link_id TEXT PRIMARY KEY,
    source_claim_id TEXT NOT NULL,
    target_claim_id TEXT NOT NULL,
    link_type TEXT NOT NULL DEFAULT 'supports',
    weight REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (source_claim_id) REFERENCES knowledge_claims(claim_id),
    FOREIGN KEY (target_claim_id) REFERENCES knowledge_claims(claim_id)
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL DEFAULT 0,
    applied_at TEXT NOT NULL
);
"""


# Additive (revision-2) columns, added to existing databases by init_db with
# ``ALTER TABLE ... ADD COLUMN`` when missing. Never remove or rename entries.
ADDITIVE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "runs": [
        ("stop_reason", "TEXT"),
        ("total_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("total_output_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("total_cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "iterations": [
        ("provider", "TEXT NOT NULL DEFAULT ''"),
        ("model", "TEXT NOT NULL DEFAULT ''"),
        ("reasoning_effort", "TEXT NOT NULL DEFAULT ''"),
        ("cli_version", "TEXT NOT NULL DEFAULT ''"),
        ("input_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("output_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("duration_seconds", "REAL NOT NULL DEFAULT 0.0"),
        ("code_execution", "TEXT"),        # JSON (CodeExecutionReport.summary())
        ("grade", "TEXT"),
        ("critical_count", "INTEGER"),
        ("high_count", "INTEGER"),
        ("consensus_report", "TEXT"),      # JSON
        ("agent_tools", "TEXT"),           # JSON (agent tool settings / tool-call counts)
    ],
}

#: Schema version recorded once the additive columns exist.
ADDITIVE_SCHEMA_VERSION = 2


async def _table_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {r[1] for r in rows}


async def ensure_additive_columns(db: aiosqlite.Connection) -> list[str]:
    """Add any missing :data:`ADDITIVE_COLUMNS`; returns ``table.column`` added."""
    added: list[str] = []
    for table, columns in ADDITIVE_COLUMNS.items():
        existing = await _table_columns(db, table)
        for name, decl in columns:
            if name not in existing:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                added.append(f"{table}.{name}")
    if added:
        await db.commit()
    return added


# ── Connection open/close serialisation ──────────────────────────────
#
# SQLite 3.51.0-3.51.1 (the version bundled with the project's conda env) has
# a lock-order inversion in its unix VFS: the last WAL connection to close
# takes an EXCLUSIVE lock while holding the inode mutex and then needs the
# global VFS mutex, whereas opening (findReusableFd) and closing (unixClose)
# take the global mutex first. One thread opening the database while another
# closes it can deadlock the whole process (fixed upstream in 3.51.2). Opening
# and closing connections is therefore serialised per event loop; queries are
# not affected.

_CONN_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _connection_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _CONN_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _CONN_LOCKS[loop] = lock
    return lock


def busy_timeout_seconds() -> float:
    """Per-statement SQLite busy timeout (``MIW_DB_BUSY_TIMEOUT``, default 30 s)."""
    try:
        return max(0.0, float(config.db_busy_timeout))
    except (TypeError, ValueError, AttributeError):
        return 30.0


async def open_connection(path: Optional[str] = None) -> aiosqlite.Connection:
    """``aiosqlite.connect`` serialised with every other open/close.

    Every connection gets the busy timeout :func:`busy_timeout_seconds`
    (SQLite waits that long for a lock before "database is locked")."""
    async with _connection_lock():
        return await aiosqlite.connect(path or config.db_path, timeout=busy_timeout_seconds())


async def close_connection(db: aiosqlite.Connection) -> None:
    """``db.close()`` serialised with every other open/close.

    If the caller is cancelled while waiting for the lock the connection is
    still closed (unserialised) so that it never leaks.
    """
    lock = _connection_lock()
    try:
        await lock.acquire()
    except asyncio.CancelledError:
        await db.close()
        raise
    try:
        await db.close()
    finally:
        lock.release()


# ── Lock contention: busy timeout + bounded retry ────────────────────
#
# Many runs share one database file, and each operation opens its own
# connection, so writers contend for SQLite's single write lock. Every
# connection waits up to ``busy_timeout_seconds()`` for a lock. On top of
# that, every write transaction (``write_transaction``) starts with
# ``BEGIN IMMEDIATE`` (the write lock is taken at the start, where the busy
# handler applies, instead of on a later statement) and is retried on
# "database is locked" / "database is busy" with exponential backoff and full
# jitter: at most ``config.db_write_attempts`` attempts, each sleep capped at
# ``config.db_retry_max_delay``, and no new attempt once
# ``config.db_retry_deadline`` seconds have passed. A failed attempt is rolled
# back, so retrying is safe (compare-and-set conditions are re-evaluated).
# Reads (``read_with_retry``) are retried the same way but without a
# transaction. Other errors are never retried.

T = TypeVar("T")

# SQLITE_BUSY (5) and SQLITE_LOCKED (6), including extended codes such as
# SQLITE_BUSY_SNAPSHOT / SQLITE_BUSY_RECOVERY / SQLITE_BUSY_TIMEOUT.
_LOCK_PRIMARY_CODES = (5, 6)
_LOCK_MESSAGES = ("database is locked", "database is busy", "database table is locked")


class DatabaseLockError(sqlite3.OperationalError):
    """A database operation still hit lock contention after every retry.

    Subclasses ``sqlite3.OperationalError`` so existing handlers still match;
    ``attempts`` / ``waited_seconds`` describe the retries made."""

    def __init__(self, message: str, attempts: int, waited_seconds: float):
        super().__init__(message)
        self.attempts = attempts
        self.waited_seconds = waited_seconds


def is_lock_error(exc: BaseException) -> bool:
    """True for SQLite "database is locked/busy" errors (transient contention)."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) in _LOCK_PRIMARY_CODES:
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _LOCK_MESSAGES)


def _retry_settings() -> tuple[int, float, float, float]:
    attempts = max(1, int(getattr(config, "db_write_attempts", 8) or 1))
    base = max(0.0, float(getattr(config, "db_retry_base_delay", 0.05) or 0.0))
    cap = max(0.0, float(getattr(config, "db_retry_max_delay", 2.0) or 0.0))
    deadline = max(0.0, float(getattr(config, "db_retry_deadline", 300.0) or 0.0))
    return attempts, base, cap, deadline


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Sleep before retry ``attempt + 1``: full jitter over the doubled base
    (at least half of it, so contending writers spread out but still wait)."""
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(ceiling / 2.0, ceiling) if ceiling > 0 else 0.0


async def _rollback_quietly(db: aiosqlite.Connection) -> None:
    try:
        if db.in_transaction:
            await db.rollback()
    except Exception:  # pragma: no cover - the connection is closed next anyway
        logger.debug("rollback failed", exc_info=True)


async def _with_lock_retry(
    attempt_fn: Callable[[], Awaitable[T]], what: str,
) -> T:
    """Run ``attempt_fn`` until it succeeds, retrying lock errors only."""
    attempts, base, cap, deadline = _retry_settings()
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            return await attempt_fn()
        except sqlite3.OperationalError as exc:
            if not is_lock_error(exc):
                raise
            waited = time.monotonic() - started
            if attempt >= attempts or waited >= deadline:
                logger.error("%s: database still locked after %d attempt(s) / %.1fs: %s",
                             what, attempt, waited, exc)
                raise DatabaseLockError(
                    f"{what}: {exc} (after {attempt} attempt(s), {waited:.1f}s)",
                    attempts=attempt, waited_seconds=waited) from exc
            delay = _backoff_delay(attempt, base, cap)
            logger.warning("%s: %s (attempt %d/%d); retrying in %.2fs",
                           what, exc, attempt, attempts, delay)
            await asyncio.sleep(delay)


async def write_transaction(
    fn: Callable[[aiosqlite.Connection], Awaitable[T]],
    *,
    what: str = "database write",
    path: Optional[str] = None,
    foreign_keys: bool = True,
) -> T:
    """Run ``fn(db)`` in one ``BEGIN IMMEDIATE`` ... ``COMMIT`` transaction,
    retrying the whole transaction on lock contention (see above).

    ``fn`` must only touch the database (it may run more than once). The
    connection is opened once per call (serialised open/close), uses
    ``aiosqlite.Row`` rows and, by default, ``PRAGMA foreign_keys=ON``.
    """
    db = await open_connection(path)
    try:
        db.row_factory = aiosqlite.Row
        if foreign_keys:
            await db.execute("PRAGMA foreign_keys=ON")

        async def attempt() -> T:
            try:
                await db.execute("BEGIN IMMEDIATE")
                result = await fn(db)
                await db.commit()
                return result
            except BaseException:
                await _rollback_quietly(db)
                raise

        return await _with_lock_retry(attempt, what)
    finally:
        await close_connection(db)


async def read_with_retry(
    fn: Callable[[aiosqlite.Connection], Awaitable[T]],
    *,
    what: str = "database read",
) -> T:
    """Run the read-only ``fn(db)`` on a ``get_db`` connection, retrying on
    lock contention (rare in WAL mode, e.g. during WAL recovery)."""
    async def attempt() -> T:
        async with get_db() as db:
            return await fn(db)

    return await _with_lock_retry(attempt, what)


async def execute_write(sql: str, values: Any = (), *, what: str = "database write") -> int:
    """Execute one write statement in a retried transaction; returns rowcount."""
    async def _do(db: aiosqlite.Connection) -> int:
        cursor = await db.execute(sql, values)
        return cursor.rowcount or 0

    return await write_transaction(_do, what=what)


# ── Lifecycle ─────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create tables if they do not exist, enable WAL mode, run migrations.

    Every step is retried on lock contention (another process may be using
    the database); connections get the busy timeout of ``open_connection``."""
    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _schema() -> None:
        db = await open_connection()
        try:
            await db.executescript(_SCHEMA)
            await db.commit()
            # Enable WAL mode for better concurrent read/write performance
            # (persistent in the database file).
            await db.execute("PRAGMA journal_mode=WAL")
        finally:
            await close_connection(db)

    await _with_lock_retry(_schema, "init_db: schema")

    # Run any pending schema migrations
    from backend.migrations import run_migrations

    async def _migrations() -> int:
        async with _connection_lock():
            return await run_migrations()

    applied = await _with_lock_retry(_migrations, "init_db: migrations")
    if applied:
        logger.info("Applied %d schema migration(s)", applied)

    # Additive revision-2 columns (idempotent).
    async def _additive(db: aiosqlite.Connection) -> list[str]:
        added = await ensure_additive_columns(db)
        cursor = await db.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        if (row[0] or 0) < ADDITIVE_SCHEMA_VERSION:
            await db.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (ADDITIVE_SCHEMA_VERSION, datetime.utcnow().isoformat()),
            )
        return added

    added = await write_transaction(_additive, what="init_db: additive columns",
                                    foreign_keys=False)
    if added:
        logger.info("Added columns: %s", ", ".join(added))
    # Per-call telemetry table (records written by TelemetryAdapter).
    try:
        from backend.telemetry.storage import init_telemetry_table
        await init_telemetry_table(str(db_path))
    except Exception:  # pragma: no cover - telemetry must never block startup
        logger.warning("Could not initialise the telemetry table", exc_info=True)


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Yield an aiosqlite connection with foreign keys enabled.

    Everything after ``connect`` happens inside ``try`` so that a
    cancellation while configuring the connection still closes it (an
    unclosed aiosqlite connection leaks a non-daemon worker thread).
    Opening and closing are serialised (see ``open_connection``).
    """
    db = await open_connection()
    try:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        yield db
    finally:
        await close_connection(db)


# ── Helpers ───────────────────────────────────────────────────────────

def _dt(val: Optional[datetime]) -> Optional[str]:
    return val.isoformat() if val else None


def _parse_dt(val: Optional[str]) -> Optional[datetime]:
    if val is None:
        return None
    return datetime.fromisoformat(val)


# ── Schema version helpers ────────────────────────────────────────────

async def get_schema_version() -> int:
    """Return current schema version (0 if never set)."""
    async def _read(db: aiosqlite.Connection) -> int:
        cursor = await db.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row["version"] if row else 0

    return await read_with_retry(_read, what="get_schema_version")


async def set_schema_version(version: int) -> None:
    """Record a schema version."""
    await execute_write(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, datetime.utcnow().isoformat()),
        what="set_schema_version",
    )


# ── Workspace CRUD ────────────────────────────────────────────────────

async def create_workspace(ws: WorkspaceConfig) -> WorkspaceConfig:
    await execute_write(
        """INSERT INTO workspaces
           (id, name, path, providers_enabled, default_provider, default_model,
            git_mode, budget_max_tokens_per_run, budget_max_cost_per_run,
            budget_max_iterations, safety_deny_destructive_commands, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ws.id, ws.name, ws.path,
            json.dumps([p.value for p in ws.providers_enabled]),
            ws.default_provider.value, ws.default_model,
            ws.git_mode.value, ws.budget_max_tokens_per_run,
            ws.budget_max_cost_per_run, ws.budget_max_iterations,
            int(ws.safety_deny_destructive_commands),
            _dt(ws.created_at),
        ),
        what="create_workspace",
    )
    return ws


def _row_to_workspace(row: aiosqlite.Row) -> WorkspaceConfig:
    return WorkspaceConfig(
        id=row["id"],
        name=row["name"],
        path=row["path"],
        providers_enabled=[ProviderName(p) for p in json.loads(row["providers_enabled"])],
        default_provider=ProviderName(row["default_provider"]),
        default_model=row["default_model"],
        git_mode=GitMode(row["git_mode"]),
        budget_max_tokens_per_run=row["budget_max_tokens_per_run"],
        budget_max_cost_per_run=row["budget_max_cost_per_run"],
        budget_max_iterations=row["budget_max_iterations"],
        safety_deny_destructive_commands=bool(row["safety_deny_destructive_commands"]),
        created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
    )


async def get_workspace(ws_id: str) -> Optional[WorkspaceConfig]:
    async def _read(db: aiosqlite.Connection) -> Optional[WorkspaceConfig]:
        cursor = await db.execute("SELECT * FROM workspaces WHERE id = ?", (ws_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_workspace(row)

    return await read_with_retry(_read, what="get_workspace")


async def list_workspaces(
    limit: int = 50,
    offset: int = 0,
) -> list[WorkspaceConfig]:
    async def _read(db: aiosqlite.Connection) -> list[WorkspaceConfig]:
        cursor = await db.execute(
            "SELECT * FROM workspaces ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [_row_to_workspace(r) for r in rows]

    return await read_with_retry(_read, what="list_workspaces")


async def delete_workspace(ws_id: str) -> bool:
    n = await execute_write("DELETE FROM workspaces WHERE id = ?", (ws_id,),
                            what="delete_workspace")
    return n > 0


# ── Run CRUD ──────────────────────────────────────────────────────────

async def create_run(run: RunState) -> RunState:
    await execute_write(
        """INSERT INTO runs
           (run_id, workspace_id, loop_preset, task, provider, model, status,
            current_iteration, max_iterations, created_at, started_at, stopped_at,
            total_tokens, total_cost, config, error, stop_reason,
            total_input_tokens, total_output_tokens, total_cached_input_tokens)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run.run_id, run.workspace_id, run.loop_preset, run.task,
            run.provider.value, run.model, run.status.value,
            run.current_iteration, run.max_iterations,
            _dt(run.created_at), _dt(run.started_at), _dt(run.stopped_at),
            run.total_tokens, run.total_cost,
            json.dumps(run.config), run.error, run.stop_reason,
            run.total_input_tokens, run.total_output_tokens,
            run.total_cached_input_tokens,
        ),
        what="create_run",
    )
    return run


def _col(row: aiosqlite.Row, name: str, default: Any = None) -> Any:
    """Column value, or ``default`` when the column is absent or NULL."""
    try:
        value = row[name]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _json_col(row: aiosqlite.Row, name: str) -> Any:
    raw = _col(row, name)
    if raw in (None, ""):
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _row_to_run(row: aiosqlite.Row) -> RunState:
    return RunState(
        run_id=row["run_id"],
        workspace_id=row["workspace_id"],
        loop_preset=row["loop_preset"],
        task=row["task"],
        provider=ProviderName(row["provider"]),
        model=row["model"],
        status=RunStatus(row["status"]),
        current_iteration=row["current_iteration"],
        max_iterations=row["max_iterations"],
        created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
        started_at=_parse_dt(row["started_at"]),
        stopped_at=_parse_dt(row["stopped_at"]),
        total_tokens=row["total_tokens"],
        total_cost=row["total_cost"],
        config=json.loads(row["config"]),
        error=row["error"],
        stop_reason=_col(row, "stop_reason"),
        total_input_tokens=_col(row, "total_input_tokens", 0),
        total_output_tokens=_col(row, "total_output_tokens", 0),
        total_cached_input_tokens=_col(row, "total_cached_input_tokens", 0),
    )


async def get_run(run_id: str) -> Optional[RunState]:
    async def _read(db: aiosqlite.Connection) -> Optional[RunState]:
        cursor = await db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        run = _row_to_run(row)
        run.iterations = await _load_iterations(db, run_id)
        return run

    return await read_with_retry(_read, what="get_run")


async def list_runs(
    workspace_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[RunState]:
    async def _read(db: aiosqlite.Connection) -> list[RunState]:
        if workspace_id:
            cursor = await db.execute(
                "SELECT * FROM runs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (workspace_id, limit, offset),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [_row_to_run(r) for r in rows]

    return await read_with_retry(_read, what="list_runs")


_RUN_UPDATE_FIELDS = {
    "status", "current_iteration", "started_at", "stopped_at",
    "total_tokens", "total_cost", "error", "config", "stop_reason",
    "total_input_tokens", "total_output_tokens", "total_cached_input_tokens",
}


def _run_update_sql(run_id: str, fields: dict[str, Any]) -> Optional[tuple[str, list[Any]]]:
    updates = {k: v for k, v in fields.items() if k in _RUN_UPDATE_FIELDS}
    if not updates:
        return None
    # serialise special types
    if "status" in updates and isinstance(updates["status"], RunStatus):
        updates["status"] = updates["status"].value
    if "started_at" in updates:
        updates["started_at"] = _dt(updates["started_at"])
    if "stopped_at" in updates:
        updates["stopped_at"] = _dt(updates["stopped_at"])
    if "config" in updates:
        updates["config"] = json.dumps(updates["config"])
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [run_id]
    return f"UPDATE runs SET {set_clause} WHERE run_id = ?", values


async def update_run(run_id: str, **fields: Any) -> Optional[RunState]:
    stmt = _run_update_sql(run_id, fields)
    if stmt is None:
        return await get_run(run_id)
    await execute_write(*stmt, what="update_run")
    return await get_run(run_id)


async def update_run_if_status(run_id: str, expected: RunStatus, **fields: Any) -> bool:
    """Compare-and-set: update the run only if its status is still ``expected``.

    Returns True if the row was updated. Used by the runner to mark a run
    RUNNING without overwriting a STOPPED written concurrently by /stop.
    The check and the update run in one ``BEGIN IMMEDIATE`` transaction, which
    is retried as a whole on lock contention (the condition is re-evaluated)."""
    stmt = _run_update_sql(run_id, fields)
    if stmt is None:
        return False
    sql, values = stmt
    sql += " AND status = ?"
    values = list(values) + [expected.value if isinstance(expected, RunStatus) else str(expected)]
    n = await execute_write(sql, values, what="update_run_if_status")
    return n > 0


# ── Iteration CRUD ───────────────────────────────────────────────────

def _dumps_or_none(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, default=str)


_ITERATION_INSERT_SQL = """INSERT INTO iterations
   (iteration_id, run_id, iteration_number, role, status,
    started_at, completed_at, artifacts_produced, logs,
    token_usage, cost_estimate, output_summary, feedback, error,
    provider, model, reasoning_effort, cli_version,
    input_tokens, output_tokens, cached_input_tokens, duration_seconds,
    code_execution, grade, critical_count, high_count, consensus_report,
    agent_tools)
   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def _iteration_values(it: IterationResult, run_id: str) -> tuple:
    return (
        it.iteration_id, run_id, it.iteration_number, it.role,
        it.status.value, _dt(it.started_at), _dt(it.completed_at),
        json.dumps(it.artifacts_produced), json.dumps(it.logs),
        it.token_usage, it.cost_estimate, it.output_summary,
        it.feedback, it.error,
        it.provider, it.model, it.reasoning_effort, it.cli_version,
        it.input_tokens, it.output_tokens, it.cached_input_tokens,
        it.duration_seconds,
        _dumps_or_none(it.code_execution), it.grade,
        it.critical_count, it.high_count,
        _dumps_or_none(it.consensus_report),
        _dumps_or_none(it.agent_tools),
    )


async def create_iteration(it: IterationResult, run_id: str) -> IterationResult:
    await execute_write(_ITERATION_INSERT_SQL, _iteration_values(it, run_id),
                        what="create_iteration")
    return it


# Same columns; an iteration that is already stored (same iteration_id) is
# left as is. Used for deferred / batched progress writes, which may repeat an
# iteration whose earlier write is not known to have failed.
_ITERATION_INSERT_OR_IGNORE_SQL = _ITERATION_INSERT_SQL.replace(
    "INSERT INTO iterations", "INSERT OR IGNORE INTO iterations", 1)


async def record_iterations_progress(
    its: list[IterationResult], run_id: str, telemetry: Optional[list] = None,
    **run_fields: Any,
) -> int:
    """Insert finished iterations and update the run row in ONE transaction.

    Used by the runner for incremental persistence: the iteration that just
    finished plus any earlier ones whose write failed (``run_fields`` are
    ``update_run`` fields such as ``current_iteration`` and the totals).
    Iterations already stored (same ``iteration_id``) are skipped.
    ``telemetry`` (``TelemetryRecord`` list) is inserted into the
    ``telemetry`` table in the same transaction. The transaction starts with
    ``BEGIN IMMEDIATE`` and is retried as a whole on lock contention; if it
    still fails, nothing was written (:class:`DatabaseLockError`). Returns the
    number of iteration rows inserted.
    """
    stmt = _run_update_sql(run_id, run_fields)
    rows = [_iteration_values(it, run_id) for it in its]
    tel_rows: list = []
    if telemetry:
        from backend.telemetry.storage import record_values
        tel_rows = [record_values(r) for r in telemetry]

    async def _do(db: aiosqlite.Connection) -> int:
        inserted = 0
        for values in rows:
            cursor = await db.execute(_ITERATION_INSERT_OR_IGNORE_SQL, values)
            inserted += max(0, cursor.rowcount or 0)
        if stmt is not None:
            await db.execute(*stmt)
        if tel_rows:
            from backend.telemetry.storage import INSERT_SQL
            try:
                await db.executemany(INSERT_SQL, tel_rows)
            except sqlite3.OperationalError as exc:
                if is_lock_error(exc):
                    raise
                # table missing (DB not initialised)
                logger.debug("telemetry rows not stored", exc_info=True)
        return inserted

    return await write_transaction(_do, what="record_iteration_progress")


async def record_iteration_progress(
    it: IterationResult, run_id: str, telemetry: Optional[list] = None, **run_fields: Any,
) -> None:
    """Insert one finished iteration and update the run row in ONE transaction
    (see :func:`record_iterations_progress`)."""
    await record_iterations_progress([it], run_id, telemetry=telemetry, **run_fields)


async def _load_iterations(db: aiosqlite.Connection, run_id: str) -> list[IterationResult]:
    cursor = await db.execute(
        "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration_number",
        (run_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_iteration(r) for r in rows]


def _row_to_iteration(row: aiosqlite.Row) -> IterationResult:
    return IterationResult(
        iteration_id=row["iteration_id"],
        iteration_number=row["iteration_number"],
        role=row["role"],
        status=IterationStatus(row["status"]),
        started_at=_parse_dt(row["started_at"]),
        completed_at=_parse_dt(row["completed_at"]),
        artifacts_produced=json.loads(row["artifacts_produced"]),
        logs=json.loads(row["logs"]),
        token_usage=row["token_usage"],
        cost_estimate=row["cost_estimate"],
        output_summary=row["output_summary"],
        feedback=row["feedback"],
        error=row["error"],
        code_execution=_json_col(row, "code_execution"),
        provider=_col(row, "provider", ""),
        model=_col(row, "model", ""),
        reasoning_effort=_col(row, "reasoning_effort", ""),
        cli_version=_col(row, "cli_version", ""),
        input_tokens=_col(row, "input_tokens", 0),
        output_tokens=_col(row, "output_tokens", 0),
        cached_input_tokens=_col(row, "cached_input_tokens", 0),
        duration_seconds=_col(row, "duration_seconds", 0.0),
        grade=_col(row, "grade"),
        critical_count=_col(row, "critical_count"),
        high_count=_col(row, "high_count"),
        consensus_report=_json_col(row, "consensus_report"),
        agent_tools=_json_col(row, "agent_tools"),
    )


async def get_iterations_for_run(run_id: str) -> list[IterationResult]:
    async def _read(db: aiosqlite.Connection) -> list[IterationResult]:
        return await _load_iterations(db, run_id)

    return await read_with_retry(_read, what="get_iterations_for_run")


async def update_iteration(iteration_id: str, **fields: Any) -> Optional[IterationResult]:
    allowed = {
        "status", "started_at", "completed_at", "artifacts_produced",
        "logs", "token_usage", "cost_estimate", "output_summary",
        "feedback", "error", "provider", "model", "reasoning_effort",
        "cli_version", "input_tokens", "output_tokens", "cached_input_tokens",
        "duration_seconds", "code_execution", "grade", "critical_count",
        "high_count", "consensus_report", "agent_tools",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return None
    for key in ("code_execution", "consensus_report", "agent_tools"):
        if key in updates:
            updates[key] = _dumps_or_none(updates[key])
    if "status" in updates and isinstance(updates["status"], IterationStatus):
        updates["status"] = updates["status"].value
    if "started_at" in updates:
        updates["started_at"] = _dt(updates["started_at"])
    if "completed_at" in updates:
        updates["completed_at"] = _dt(updates["completed_at"])
    if "artifacts_produced" in updates:
        updates["artifacts_produced"] = json.dumps(updates["artifacts_produced"])
    if "logs" in updates:
        updates["logs"] = json.dumps(updates["logs"])
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [iteration_id]

    async def _do(db: aiosqlite.Connection) -> Optional[IterationResult]:
        await db.execute(
            f"UPDATE iterations SET {set_clause} WHERE iteration_id = ?", values,
        )
        cursor = await db.execute(
            "SELECT * FROM iterations WHERE iteration_id = ?", (iteration_id,),
        )
        row = await cursor.fetchone()
        return None if row is None else _row_to_iteration(row)

    return await write_transaction(_do, what="update_iteration")


# ── Knowledge CRUD ───────────────────────────────────────────────────

def _row_to_knowledge_claim(row: aiosqlite.Row) -> dict:
    return {
        "claim_id": row["claim_id"],
        "run_id": row["run_id"],
        "claim_text": row["claim_text"],
        "evidence_pointers": json.loads(row["evidence_pointers"]),
        "uncertainty": row["uncertainty"],
        "strength": row["strength"],
        "falsification_tests": json.loads(row["falsification_tests"]),
        "source_artifact": row["source_artifact"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_knowledge_fact(row: aiosqlite.Row) -> dict:
    return {
        "fact_id": row["fact_id"],
        "claim_id": row["claim_id"],
        "fact_text": row["fact_text"],
        "source": row["source"],
        "confidence": row["confidence"],
        "created_at": row["created_at"],
    }


def _row_to_knowledge_link(row: aiosqlite.Row) -> dict:
    return {
        "link_id": row["link_id"],
        "source_claim_id": row["source_claim_id"],
        "target_claim_id": row["target_claim_id"],
        "link_type": row["link_type"],
        "weight": row["weight"],
    }


async def create_knowledge_claim(
    claim_id: str,
    run_id: Optional[str],
    claim_text: str,
    evidence_pointers: list[str],
    uncertainty: float,
    strength: float,
    falsification_tests: list[str],
    source_artifact: str,
) -> dict:
    now = datetime.utcnow().isoformat()

    async def _do(db: aiosqlite.Connection) -> dict:
        await db.execute(
            """INSERT INTO knowledge_claims
               (claim_id, run_id, claim_text, evidence_pointers, uncertainty,
                strength, falsification_tests, source_artifact, status,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                claim_id, run_id, claim_text,
                json.dumps(evidence_pointers), uncertainty, strength,
                json.dumps(falsification_tests), source_artifact,
                "active", now, now,
            ),
        )
        cursor = await db.execute(
            "SELECT * FROM knowledge_claims WHERE claim_id = ?", (claim_id,),
        )
        row = await cursor.fetchone()
        return _row_to_knowledge_claim(row)

    return await write_transaction(_do, what="create_knowledge_claim")


async def get_knowledge_claims(
    run_id: Optional[str] = None,
    status: str = "active",
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    async def _read(db: aiosqlite.Connection) -> list[dict]:
        if run_id:
            cursor = await db.execute(
                "SELECT * FROM knowledge_claims WHERE run_id = ? AND status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (run_id, status, limit, offset),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM knowledge_claims WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        rows = await cursor.fetchall()
        return [_row_to_knowledge_claim(r) for r in rows]

    return await read_with_retry(_read, what="get_knowledge_claims")


async def update_knowledge_claim(claim_id: str, **fields: Any) -> Optional[dict]:
    allowed = {
        "claim_text", "evidence_pointers", "uncertainty", "strength",
        "falsification_tests", "source_artifact", "status",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return None
    if "evidence_pointers" in updates:
        updates["evidence_pointers"] = json.dumps(updates["evidence_pointers"])
    if "falsification_tests" in updates:
        updates["falsification_tests"] = json.dumps(updates["falsification_tests"])
    updates["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [claim_id]

    async def _do(db: aiosqlite.Connection) -> Optional[dict]:
        await db.execute(
            f"UPDATE knowledge_claims SET {set_clause} WHERE claim_id = ?", values,
        )
        cursor = await db.execute(
            "SELECT * FROM knowledge_claims WHERE claim_id = ?", (claim_id,),
        )
        row = await cursor.fetchone()
        return None if row is None else _row_to_knowledge_claim(row)

    return await write_transaction(_do, what="update_knowledge_claim")


async def create_knowledge_fact(
    fact_id: str,
    claim_id: Optional[str],
    fact_text: str,
    source: str,
    confidence: float,
) -> dict:
    now = datetime.utcnow().isoformat()

    async def _do(db: aiosqlite.Connection) -> dict:
        await db.execute(
            """INSERT INTO knowledge_facts
               (fact_id, claim_id, fact_text, source, confidence, created_at)
               VALUES (?,?,?,?,?,?)""",
            (fact_id, claim_id, fact_text, source, confidence, now),
        )
        cursor = await db.execute(
            "SELECT * FROM knowledge_facts WHERE fact_id = ?", (fact_id,),
        )
        row = await cursor.fetchone()
        return _row_to_knowledge_fact(row)

    return await write_transaction(_do, what="create_knowledge_fact")


async def get_knowledge_facts(
    claim_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    async def _read(db: aiosqlite.Connection) -> list[dict]:
        if claim_id:
            cursor = await db.execute(
                "SELECT * FROM knowledge_facts WHERE claim_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (claim_id, limit, offset),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM knowledge_facts ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [_row_to_knowledge_fact(r) for r in rows]

    return await read_with_retry(_read, what="get_knowledge_facts")


async def create_knowledge_link(
    link_id: str,
    source_claim_id: str,
    target_claim_id: str,
    link_type: str,
    weight: float,
) -> dict:
    async def _do(db: aiosqlite.Connection) -> dict:
        await db.execute(
            """INSERT INTO knowledge_links
               (link_id, source_claim_id, target_claim_id, link_type, weight)
               VALUES (?,?,?,?,?)""",
            (link_id, source_claim_id, target_claim_id, link_type, weight),
        )
        cursor = await db.execute(
            "SELECT * FROM knowledge_links WHERE link_id = ?", (link_id,),
        )
        row = await cursor.fetchone()
        return _row_to_knowledge_link(row)

    return await write_transaction(_do, what="create_knowledge_link")


async def get_knowledge_links(claim_id: Optional[str] = None) -> list[dict]:
    async def _read(db: aiosqlite.Connection) -> list[dict]:
        if claim_id:
            cursor = await db.execute(
                "SELECT * FROM knowledge_links WHERE source_claim_id = ? OR target_claim_id = ?",
                (claim_id, claim_id),
            )
        else:
            cursor = await db.execute("SELECT * FROM knowledge_links")
        rows = await cursor.fetchall()
        return [_row_to_knowledge_link(r) for r in rows]

    return await read_with_retry(_read, what="get_knowledge_links")


async def get_knowledge_summary() -> dict:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt, AVG(uncertainty) as avg_unc, AVG(strength) as avg_str FROM knowledge_claims WHERE status = 'active'"
        )
        row = await cursor.fetchone()
        claim_count = row["cnt"]
        avg_uncertainty = row["avg_unc"]
        avg_strength = row["avg_str"]

        cursor = await db.execute("SELECT COUNT(*) as cnt FROM knowledge_facts")
        fact_count = (await cursor.fetchone())["cnt"]

        cursor = await db.execute("SELECT COUNT(*) as cnt FROM knowledge_links")
        link_count = (await cursor.fetchone())["cnt"]

        return {
            "total_claims": claim_count,
            "total_facts": fact_count,
            "total_links": link_count,
            "avg_uncertainty": round(avg_uncertainty, 3) if avg_uncertainty is not None else 0.0,
            "avg_strength": round(avg_strength, 3) if avg_strength is not None else 0.0,
        }
