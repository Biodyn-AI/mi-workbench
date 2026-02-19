"""SQLite database layer using aiosqlite."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

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


# ── Lifecycle ─────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create tables if they do not exist, enable WAL mode, run migrations."""
    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(config.db_path) as db:
        await db.executescript(_SCHEMA)
        await db.commit()
        # Enable WAL mode for better concurrent read/write performance
        await db.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5s for locks instead of failing immediately
        await db.execute("PRAGMA busy_timeout=5000")
    # Run any pending schema migrations
    from backend.migrations import run_migrations
    applied = await run_migrations()
    if applied:
        logger.info("Applied %d schema migration(s)", applied)


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Yield an aiosqlite connection with foreign keys enabled."""
    db = await aiosqlite.connect(config.db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
    finally:
        await db.close()


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
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row["version"] if row else 0


async def set_schema_version(version: int) -> None:
    """Record a schema version."""
    async with get_db() as db:
        await db.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, datetime.utcnow().isoformat()),
        )
        await db.commit()


# ── Workspace CRUD ────────────────────────────────────────────────────

async def create_workspace(ws: WorkspaceConfig) -> WorkspaceConfig:
    async with get_db() as db:
        await db.execute(
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
        )
        await db.commit()
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
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM workspaces WHERE id = ?", (ws_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_workspace(row)


async def list_workspaces(
    limit: int = 50,
    offset: int = 0,
) -> list[WorkspaceConfig]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM workspaces ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [_row_to_workspace(r) for r in rows]


async def delete_workspace(ws_id: str) -> bool:
    async with get_db() as db:
        cursor = await db.execute("DELETE FROM workspaces WHERE id = ?", (ws_id,))
        await db.commit()
        return cursor.rowcount > 0


# ── Run CRUD ──────────────────────────────────────────────────────────

async def create_run(run: RunState) -> RunState:
    async with get_db() as db:
        await db.execute(
            """INSERT INTO runs
               (run_id, workspace_id, loop_preset, task, provider, model, status,
                current_iteration, max_iterations, created_at, started_at, stopped_at,
                total_tokens, total_cost, config, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run.run_id, run.workspace_id, run.loop_preset, run.task,
                run.provider.value, run.model, run.status.value,
                run.current_iteration, run.max_iterations,
                _dt(run.created_at), _dt(run.started_at), _dt(run.stopped_at),
                run.total_tokens, run.total_cost,
                json.dumps(run.config), run.error,
            ),
        )
        await db.commit()
    return run


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
    )


async def get_run(run_id: str) -> Optional[RunState]:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        run = _row_to_run(row)
        run.iterations = await _load_iterations(db, run_id)
        return run


async def list_runs(
    workspace_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[RunState]:
    async with get_db() as db:
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


async def update_run(run_id: str, **fields: Any) -> Optional[RunState]:
    allowed = {
        "status", "current_iteration", "started_at", "stopped_at",
        "total_tokens", "total_cost", "error", "config",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return await get_run(run_id)
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
    async with get_db() as db:
        await db.execute(
            f"UPDATE runs SET {set_clause} WHERE run_id = ?", values,
        )
        await db.commit()
    return await get_run(run_id)


# ── Iteration CRUD ───────────────────────────────────────────────────

async def create_iteration(it: IterationResult, run_id: str) -> IterationResult:
    async with get_db() as db:
        await db.execute(
            """INSERT INTO iterations
               (iteration_id, run_id, iteration_number, role, status,
                started_at, completed_at, artifacts_produced, logs,
                token_usage, cost_estimate, output_summary, feedback, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                it.iteration_id, run_id, it.iteration_number, it.role,
                it.status.value, _dt(it.started_at), _dt(it.completed_at),
                json.dumps(it.artifacts_produced), json.dumps(it.logs),
                it.token_usage, it.cost_estimate, it.output_summary,
                it.feedback, it.error,
            ),
        )
        await db.commit()
    return it


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
    )


async def get_iterations_for_run(run_id: str) -> list[IterationResult]:
    async with get_db() as db:
        return await _load_iterations(db, run_id)


async def update_iteration(iteration_id: str, **fields: Any) -> Optional[IterationResult]:
    allowed = {
        "status", "started_at", "completed_at", "artifacts_produced",
        "logs", "token_usage", "cost_estimate", "output_summary",
        "feedback", "error",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return None
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
    async with get_db() as db:
        await db.execute(
            f"UPDATE iterations SET {set_clause} WHERE iteration_id = ?", values,
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM iterations WHERE iteration_id = ?", (iteration_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_iteration(row)


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
    async with get_db() as db:
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
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM knowledge_claims WHERE claim_id = ?", (claim_id,),
        )
        row = await cursor.fetchone()
        return _row_to_knowledge_claim(row)


async def get_knowledge_claims(
    run_id: Optional[str] = None,
    status: str = "active",
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    async with get_db() as db:
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
    async with get_db() as db:
        await db.execute(
            f"UPDATE knowledge_claims SET {set_clause} WHERE claim_id = ?", values,
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM knowledge_claims WHERE claim_id = ?", (claim_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_knowledge_claim(row)


async def create_knowledge_fact(
    fact_id: str,
    claim_id: Optional[str],
    fact_text: str,
    source: str,
    confidence: float,
) -> dict:
    now = datetime.utcnow().isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO knowledge_facts
               (fact_id, claim_id, fact_text, source, confidence, created_at)
               VALUES (?,?,?,?,?,?)""",
            (fact_id, claim_id, fact_text, source, confidence, now),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM knowledge_facts WHERE fact_id = ?", (fact_id,),
        )
        row = await cursor.fetchone()
        return _row_to_knowledge_fact(row)


async def get_knowledge_facts(
    claim_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    async with get_db() as db:
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


async def create_knowledge_link(
    link_id: str,
    source_claim_id: str,
    target_claim_id: str,
    link_type: str,
    weight: float,
) -> dict:
    async with get_db() as db:
        await db.execute(
            """INSERT INTO knowledge_links
               (link_id, source_claim_id, target_claim_id, link_type, weight)
               VALUES (?,?,?,?,?)""",
            (link_id, source_claim_id, target_claim_id, link_type, weight),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM knowledge_links WHERE link_id = ?", (link_id,),
        )
        row = await cursor.fetchone()
        return _row_to_knowledge_link(row)


async def get_knowledge_links(claim_id: Optional[str] = None) -> list[dict]:
    async with get_db() as db:
        if claim_id:
            cursor = await db.execute(
                "SELECT * FROM knowledge_links WHERE source_claim_id = ? OR target_claim_id = ?",
                (claim_id, claim_id),
            )
        else:
            cursor = await db.execute("SELECT * FROM knowledge_links")
        rows = await cursor.fetchall()
        return [_row_to_knowledge_link(r) for r in rows]


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
