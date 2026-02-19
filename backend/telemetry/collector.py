"""Telemetry collection for adapter calls and run execution."""
from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TelemetryRecord:
    timestamp: str
    run_id: str
    iteration: int
    role: str
    adapter: str
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    cost_estimate: float = 0.0
    latency_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


class TelemetryCollector:
    """Collects and stores telemetry records for all adapter calls."""

    def __init__(self) -> None:
        self._records: list[TelemetryRecord] = []

    def record(self, rec: TelemetryRecord) -> None:
        """Add a telemetry record."""
        self._records.append(rec)

    async def wrap_adapter_call(self, adapter, request, run_id: str, iteration: int, role: str):
        """Wrap an adapter.run() call with telemetry collection.

        Measures latency, captures token usage and cost, records success/failure.
        Returns the AdapterRunResult.
        """
        start = time.monotonic()
        error_msg: Optional[str] = None
        success = True
        result = None

        try:
            result = await adapter.run(request)
            if not result.success:
                success = False
                error_msg = result.error
        except Exception as exc:
            success = False
            error_msg = str(exc)
            raise
        finally:
            elapsed = time.monotonic() - start
            tokens_total = result.token_usage if result else 0
            # Estimate input/output split (adapters report total; rough 70/30 split)
            tokens_input = int(tokens_total * 0.7)
            tokens_output = tokens_total - tokens_input
            cost = result.cost_estimate if result else 0.0

            rec = TelemetryRecord(
                timestamp=datetime.utcnow().isoformat(),
                run_id=run_id,
                iteration=iteration,
                role=role,
                adapter=getattr(adapter, "name", type(adapter).__name__),
                model=getattr(request, "model", ""),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_total,
                cost_estimate=cost,
                latency_seconds=round(elapsed, 4),
                success=success,
                error=error_msg,
            )
            self.record(rec)

        return result

    def get_records(self, run_id: Optional[str] = None) -> list[TelemetryRecord]:
        """Get all records, optionally filtered by run_id."""
        if run_id:
            return [r for r in self._records if r.run_id == run_id]
        return list(self._records)

    def get_summary(self) -> dict:
        """Aggregate summary: total tokens, cost, avg latency, success rate."""
        if not self._records:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "success_rate": 0.0,
            }
        total_calls = len(self._records)
        total_tokens = sum(r.tokens_total for r in self._records)
        total_cost = sum(r.cost_estimate for r in self._records)
        avg_latency = sum(r.latency_seconds for r in self._records) / total_calls
        successes = sum(1 for r in self._records if r.success)
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "avg_latency": round(avg_latency, 4),
            "success_rate": round(successes / total_calls, 4),
        }

    def get_by_role(self) -> dict:
        """Per-role breakdown: {role: {total_tokens, avg_latency, cost, calls, success_rate}}."""
        groups: dict[str, list[TelemetryRecord]] = {}
        for r in self._records:
            groups.setdefault(r.role, []).append(r)
        result = {}
        for role, recs in groups.items():
            n = len(recs)
            successes = sum(1 for r in recs if r.success)
            result[role] = {
                "calls": n,
                "total_tokens": sum(r.tokens_total for r in recs),
                "cost": round(sum(r.cost_estimate for r in recs), 6),
                "avg_latency": round(sum(r.latency_seconds for r in recs) / n, 4),
                "success_rate": round(successes / n, 4),
            }
        return result

    def get_by_adapter(self) -> dict:
        """Per-adapter breakdown."""
        groups: dict[str, list[TelemetryRecord]] = {}
        for r in self._records:
            groups.setdefault(r.adapter, []).append(r)
        result = {}
        for adapter, recs in groups.items():
            n = len(recs)
            successes = sum(1 for r in recs if r.success)
            result[adapter] = {
                "calls": n,
                "total_tokens": sum(r.tokens_total for r in recs),
                "cost": round(sum(r.cost_estimate for r in recs), 6),
                "avg_latency": round(sum(r.latency_seconds for r in recs) / n, 4),
                "success_rate": round(successes / n, 4),
            }
        return result

    def get_cost_timeline(self) -> list[dict]:
        """Cost accumulation over time: [{timestamp, cumulative_cost, cumulative_tokens}]."""
        sorted_recs = sorted(self._records, key=lambda r: r.timestamp)
        timeline = []
        cum_cost = 0.0
        cum_tokens = 0
        for r in sorted_recs:
            cum_cost += r.cost_estimate
            cum_tokens += r.tokens_total
            timeline.append({
                "timestamp": r.timestamp,
                "cumulative_cost": round(cum_cost, 6),
                "cumulative_tokens": cum_tokens,
            })
        return timeline

    def get_latency_percentiles(self) -> dict:
        """Latency p50, p90, p95, p99."""
        if not self._records:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        latencies = sorted(r.latency_seconds for r in self._records)
        n = len(latencies)

        def _percentile(pct: float) -> float:
            idx = int(pct / 100.0 * (n - 1))
            return round(latencies[idx], 4)

        return {
            "p50": _percentile(50),
            "p90": _percentile(90),
            "p95": _percentile(95),
            "p99": _percentile(99),
        }

    def export_csv(self) -> str:
        """Export all records as CSV."""
        if not self._records:
            return ""
        buf = io.StringIO()
        fieldnames = list(asdict(self._records[0]).keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for rec in self._records:
            writer.writerow(asdict(rec))
        return buf.getvalue()

    def export_json(self) -> str:
        """Export all records as JSON."""
        return json.dumps([asdict(r) for r in self._records], indent=2)
