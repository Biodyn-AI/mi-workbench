"""SQLite persistence for telemetry records."""
from __future__ import annotations

from typing import Optional

import aiosqlite

from backend.config import config
from backend.telemetry.collector import TelemetryRecord


_TELEMETRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    role TEXT NOT NULL,
    adapter TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    tokens_input INTEGER NOT NULL DEFAULT 0,
    tokens_output INTEGER NOT NULL DEFAULT 0,
    tokens_total INTEGER NOT NULL DEFAULT 0,
    cost_estimate REAL NOT NULL DEFAULT 0.0,
    latency_seconds REAL NOT NULL DEFAULT 0.0,
    success INTEGER NOT NULL DEFAULT 1,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_telemetry_run_id ON telemetry(run_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry(timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_adapter ON telemetry(adapter);
CREATE INDEX IF NOT EXISTS idx_telemetry_role ON telemetry(role);
"""


async def init_telemetry_table(db_path: Optional[str] = None) -> None:
    """Create the telemetry table if it doesn't exist."""
    path = db_path or config.db_path
    async with aiosqlite.connect(path) as db:
        await db.executescript(_TELEMETRY_SCHEMA)
        await db.commit()


async def save_record(rec: TelemetryRecord, db_path: Optional[str] = None) -> None:
    """Insert a single telemetry record."""
    path = db_path or config.db_path
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """INSERT INTO telemetry
               (timestamp, run_id, iteration, role, adapter, model,
                tokens_input, tokens_output, tokens_total,
                cost_estimate, latency_seconds, success, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.timestamp, rec.run_id, rec.iteration, rec.role,
                rec.adapter, rec.model,
                rec.tokens_input, rec.tokens_output, rec.tokens_total,
                rec.cost_estimate, rec.latency_seconds,
                int(rec.success), rec.error,
            ),
        )
        await db.commit()


def _row_to_record(row: aiosqlite.Row) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=row["timestamp"],
        run_id=row["run_id"],
        iteration=row["iteration"],
        role=row["role"],
        adapter=row["adapter"],
        model=row["model"],
        tokens_input=row["tokens_input"],
        tokens_output=row["tokens_output"],
        tokens_total=row["tokens_total"],
        cost_estimate=row["cost_estimate"],
        latency_seconds=row["latency_seconds"],
        success=bool(row["success"]),
        error=row["error"],
    )


async def get_records(
    run_id: Optional[str] = None,
    adapter: Optional[str] = None,
    role: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[TelemetryRecord]:
    """Query telemetry records with optional filters."""
    path = db_path or config.db_path
    clauses = []
    params: list = []

    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if adapter:
        clauses.append("adapter = ?")
        params.append(adapter)
    if role:
        clauses.append("role = ?")
        params.append(role)
    if time_from:
        clauses.append("timestamp >= ?")
        params.append(time_from)
    if time_to:
        clauses.append("timestamp <= ?")
        params.append(time_to)

    where = " AND ".join(clauses) if clauses else "1=1"
    sql = f"SELECT * FROM telemetry WHERE {where} ORDER BY timestamp"

    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_record(r) for r in rows]


async def get_summary(
    run_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Aggregate summary from the database."""
    path = db_path or config.db_path
    where = "WHERE run_id = ?" if run_id else ""
    params = (run_id,) if run_id else ()

    async with aiosqlite.connect(path) as db:
        cursor = await db.execute(
            f"""SELECT
                COUNT(*) as total_calls,
                COALESCE(SUM(tokens_total), 0) as total_tokens,
                COALESCE(SUM(cost_estimate), 0.0) as total_cost,
                COALESCE(AVG(latency_seconds), 0.0) as avg_latency,
                COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as successes
            FROM telemetry {where}""",
            params,
        )
        row = await cursor.fetchone()
        total_calls = row[0]
        return {
            "total_calls": total_calls,
            "total_tokens": row[1],
            "total_cost": round(row[2], 6),
            "avg_latency": round(row[3], 4),
            "success_rate": round(row[4] / total_calls, 4) if total_calls > 0 else 0.0,
        }


async def get_by_role(
    run_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Per-role breakdown from the database."""
    path = db_path or config.db_path
    where = "WHERE run_id = ?" if run_id else ""
    params = (run_id,) if run_id else ()

    async with aiosqlite.connect(path) as db:
        cursor = await db.execute(
            f"""SELECT role,
                COUNT(*) as calls,
                SUM(tokens_total) as total_tokens,
                SUM(cost_estimate) as cost,
                AVG(latency_seconds) as avg_latency,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
            FROM telemetry {where}
            GROUP BY role""",
            params,
        )
        rows = await cursor.fetchall()
        result = {}
        for row in rows:
            n = row[1]
            result[row[0]] = {
                "calls": n,
                "total_tokens": row[2],
                "cost": round(row[3], 6),
                "avg_latency": round(row[4], 4),
                "success_rate": round(row[5] / n, 4) if n > 0 else 0.0,
            }
        return result


async def get_by_adapter(
    run_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Per-adapter breakdown from the database."""
    path = db_path or config.db_path
    where = "WHERE run_id = ?" if run_id else ""
    params = (run_id,) if run_id else ()

    async with aiosqlite.connect(path) as db:
        cursor = await db.execute(
            f"""SELECT adapter,
                COUNT(*) as calls,
                SUM(tokens_total) as total_tokens,
                SUM(cost_estimate) as cost,
                AVG(latency_seconds) as avg_latency,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
            FROM telemetry {where}
            GROUP BY adapter""",
            params,
        )
        rows = await cursor.fetchall()
        result = {}
        for row in rows:
            n = row[1]
            result[row[0]] = {
                "calls": n,
                "total_tokens": row[2],
                "cost": round(row[3], 6),
                "avg_latency": round(row[4], 4),
                "success_rate": round(row[5] / n, 4) if n > 0 else 0.0,
            }
        return result
