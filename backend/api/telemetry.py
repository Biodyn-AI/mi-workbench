"""Telemetry API routes for cost/latency profiling."""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from backend.telemetry.collector import TelemetryCollector

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

# Shared in-memory collector (populated during run execution)
_collector = TelemetryCollector()


def get_collector() -> TelemetryCollector:
    """Access the global telemetry collector."""
    return _collector


@router.get("/summary")
async def telemetry_summary() -> dict:
    """Aggregate summary: total tokens, cost, avg latency, success rate."""
    return _collector.get_summary()


@router.get("/by-role")
async def telemetry_by_role() -> dict:
    """Per-role breakdown."""
    return _collector.get_by_role()


@router.get("/by-adapter")
async def telemetry_by_adapter() -> dict:
    """Per-adapter breakdown."""
    return _collector.get_by_adapter()


@router.get("/timeline")
async def telemetry_timeline() -> list[dict]:
    """Cost accumulation timeline for charts."""
    return _collector.get_cost_timeline()


@router.get("/latency")
async def telemetry_latency() -> dict:
    """Latency percentiles (p50, p90, p95, p99)."""
    return _collector.get_latency_percentiles()


@router.get("/export")
async def telemetry_export(
    format: str = Query("json", pattern="^(csv|json)$"),
) -> PlainTextResponse:
    """Export raw telemetry data as CSV or JSON."""
    if format == "csv":
        return PlainTextResponse(
            content=_collector.export_csv(),
            media_type="text/csv",
        )
    return PlainTextResponse(
        content=_collector.export_json(),
        media_type="application/json",
    )


@router.get("/runs/{run_id}")
async def telemetry_for_run(run_id: str) -> dict:
    """Telemetry for a specific run."""
    records = _collector.get_records(run_id=run_id)
    if not records:
        return {"run_id": run_id, "records": [], "summary": {}}
    # Build a mini-summary for this run
    total_tokens = sum(r.tokens_total for r in records)
    total_cost = sum(r.cost_estimate for r in records)
    avg_latency = sum(r.latency_seconds for r in records) / len(records)
    successes = sum(1 for r in records if r.success)
    from dataclasses import asdict
    return {
        "run_id": run_id,
        "records": [asdict(r) for r in records],
        "summary": {
            "total_calls": len(records),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "avg_latency": round(avg_latency, 4),
            "success_rate": round(successes / len(records), 4),
        },
    }
