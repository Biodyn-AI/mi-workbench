"""Simple schema migration framework for MI-Workbench."""
from __future__ import annotations

import logging

import aiosqlite

from backend.config import config

logger = logging.getLogger(__name__)

# Each migration: (version, description, sql_or_None)
# Version 1 = initial schema handled by _SCHEMA in database.py
MIGRATIONS: list[tuple[int, str, str | None]] = [
    (1, "Initial schema", None),
    # Future migrations go here, e.g.:
    # (2, "Add index on runs.workspace_id", "CREATE INDEX IF NOT EXISTS ..."),
]


async def get_schema_version_raw(db: aiosqlite.Connection) -> int:
    """Read current schema version from a connection (no get_db wrapper)."""
    try:
        cursor = await db.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


async def run_migrations() -> int:
    """Run pending migrations. Returns number of migrations applied."""
    from datetime import datetime

    from backend.database import busy_timeout_seconds

    applied = 0
    # Callers serialise this open/close with every other one (init_db holds
    # backend.database's connection lock) and retry it on lock contention.
    async with aiosqlite.connect(config.db_path, timeout=busy_timeout_seconds()) as db:
        current = await get_schema_version_raw(db)
        for version, description, sql in MIGRATIONS:
            if version <= current:
                continue
            logger.info("Applying migration v%d: %s", version, description)
            if sql:
                await db.executescript(sql)
            await db.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, datetime.utcnow().isoformat()),
            )
            await db.commit()
            applied += 1
            logger.info("Migration v%d applied successfully", version)
    return applied
