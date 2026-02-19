"""Tests for the telemetry collector and API endpoints."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest
import pytest_asyncio

from backend.telemetry.collector import TelemetryCollector, TelemetryRecord
from backend.models import AdapterRunRequest, AdapterRunResult, PromptBundle, WorkspaceContext


# ── Helpers ──────────────────────────────────────────────────────────

def _make_record(
    run_id: str = "run1",
    iteration: int = 0,
    role: str = "executor",
    adapter: str = "mock",
    model: str = "test-model",
    tokens_total: int = 1000,
    cost: float = 0.01,
    latency: float = 1.5,
    success: bool = True,
    error: str | None = None,
) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp="2026-01-01T00:00:00",
        run_id=run_id,
        iteration=iteration,
        role=role,
        adapter=adapter,
        model=model,
        tokens_input=700,
        tokens_output=300,
        tokens_total=tokens_total,
        cost_estimate=cost,
        latency_seconds=latency,
        success=success,
        error=error,
    )


class _FakeAdapter:
    name = "fake_adapter"

    def __init__(self, result: AdapterRunResult | None = None, raise_exc: Exception | None = None):
        self._result = result or AdapterRunResult(
            success=True,
            token_usage=500,
            cost_estimate=0.005,
        )
        self._raise_exc = raise_exc

    async def run(self, request):
        if self._raise_exc:
            raise self._raise_exc
        return self._result


def _make_request() -> AdapterRunRequest:
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(system_prompt="test", user_prompt="test"),
        workspace_context=WorkspaceContext(workspace_path="/tmp"),
    )


# ── Unit Tests ───────────────────────────────────────────────────────

def test_record_telemetry():
    """Collector stores records and retrieves them."""
    c = TelemetryCollector()
    rec = _make_record()
    c.record(rec)
    assert len(c.get_records()) == 1
    assert c.get_records()[0].run_id == "run1"


def test_summary_aggregation():
    """Summary correctly aggregates tokens, cost, latency, success rate."""
    c = TelemetryCollector()
    c.record(_make_record(tokens_total=1000, cost=0.01, latency=1.0, success=True))
    c.record(_make_record(tokens_total=2000, cost=0.02, latency=3.0, success=True))
    c.record(_make_record(tokens_total=500, cost=0.005, latency=2.0, success=False))

    s = c.get_summary()
    assert s["total_calls"] == 3
    assert s["total_tokens"] == 3500
    assert abs(s["total_cost"] - 0.035) < 1e-6
    assert abs(s["avg_latency"] - 2.0) < 1e-4
    assert abs(s["success_rate"] - 2 / 3) < 1e-4


def test_by_role_breakdown():
    """Per-role breakdown groups correctly."""
    c = TelemetryCollector()
    c.record(_make_record(role="executor", tokens_total=1000, cost=0.01))
    c.record(_make_record(role="executor", tokens_total=2000, cost=0.02))
    c.record(_make_record(role="reviewer", tokens_total=500, cost=0.005))

    by_role = c.get_by_role()
    assert "executor" in by_role
    assert "reviewer" in by_role
    assert by_role["executor"]["calls"] == 2
    assert by_role["executor"]["total_tokens"] == 3000
    assert by_role["reviewer"]["calls"] == 1


def test_by_adapter_breakdown():
    """Per-adapter breakdown groups correctly."""
    c = TelemetryCollector()
    c.record(_make_record(adapter="claude_code", tokens_total=1000))
    c.record(_make_record(adapter="claude_code", tokens_total=2000))
    c.record(_make_record(adapter="mock", tokens_total=500))

    by_adapter = c.get_by_adapter()
    assert "claude_code" in by_adapter
    assert "mock" in by_adapter
    assert by_adapter["claude_code"]["calls"] == 2
    assert by_adapter["mock"]["total_tokens"] == 500


def test_cost_timeline():
    """Timeline returns cumulative cost and tokens in timestamp order."""
    c = TelemetryCollector()
    c.record(_make_record(tokens_total=100, cost=0.001))
    # Modify timestamp for second record
    rec2 = _make_record(tokens_total=200, cost=0.002)
    rec2.timestamp = "2026-01-01T00:01:00"
    c.record(rec2)

    timeline = c.get_cost_timeline()
    assert len(timeline) == 2
    assert timeline[0]["cumulative_tokens"] == 100
    assert timeline[1]["cumulative_tokens"] == 300
    assert abs(timeline[1]["cumulative_cost"] - 0.003) < 1e-6


def test_latency_percentiles():
    """Percentile calculations return sensible values."""
    c = TelemetryCollector()
    # Add 10 records with latencies 1.0 to 10.0
    for i in range(1, 11):
        rec = _make_record(latency=float(i))
        rec.timestamp = f"2026-01-01T00:{i:02d}:00"
        c.record(rec)

    p = c.get_latency_percentiles()
    assert p["p50"] > 0
    assert p["p90"] >= p["p50"]
    assert p["p99"] >= p["p90"]
    # With 10 values, p50 should be around 5
    assert 4.0 <= p["p50"] <= 6.0


@pytest.mark.asyncio
async def test_wrap_adapter_call_measures_latency():
    """wrap_adapter_call measures latency and records a TelemetryRecord."""
    c = TelemetryCollector()
    adapter = _FakeAdapter()
    request = _make_request()

    result = await c.wrap_adapter_call(adapter, request, "run-test", 1, "executor")

    assert result.success is True
    assert len(c.get_records()) == 1
    rec = c.get_records()[0]
    assert rec.run_id == "run-test"
    assert rec.iteration == 1
    assert rec.role == "executor"
    assert rec.adapter == "fake_adapter"
    assert rec.latency_seconds >= 0  # may round to 0 for fast calls
    assert rec.tokens_total == 500
    assert rec.success is True


@pytest.mark.asyncio
async def test_wrap_adapter_call_records_failure():
    """wrap_adapter_call records failure when adapter raises an exception."""
    c = TelemetryCollector()
    adapter = _FakeAdapter(raise_exc=RuntimeError("boom"))
    request = _make_request()

    with pytest.raises(RuntimeError, match="boom"):
        await c.wrap_adapter_call(adapter, request, "run-err", 2, "reviewer")

    assert len(c.get_records()) == 1
    rec = c.get_records()[0]
    assert rec.success is False
    assert rec.error == "boom"
    assert rec.run_id == "run-err"
    assert rec.role == "reviewer"


def test_export_csv():
    """Export CSV produces valid CSV with headers and data rows."""
    c = TelemetryCollector()
    c.record(_make_record(run_id="r1"))
    c.record(_make_record(run_id="r2"))

    csv_str = c.export_csv()
    lines = csv_str.strip().split("\n")
    assert len(lines) == 3  # header + 2 data rows
    assert "run_id" in lines[0]
    assert "r1" in lines[1]
    assert "r2" in lines[2]


def test_export_json():
    """Export JSON produces valid JSON array with all records."""
    c = TelemetryCollector()
    c.record(_make_record(run_id="r1"))
    c.record(_make_record(run_id="r2"))

    json_str = c.export_json()
    data = json.loads(json_str)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["run_id"] == "r1"
    assert data[1]["run_id"] == "r2"


def test_empty_summary():
    """Summary on empty collector returns zeroes."""
    c = TelemetryCollector()
    s = c.get_summary()
    assert s["total_calls"] == 0
    assert s["total_tokens"] == 0
    assert s["total_cost"] == 0.0
    assert s["avg_latency"] == 0.0
    assert s["success_rate"] == 0.0


def test_empty_latency_percentiles():
    """Latency percentiles on empty collector returns zeroes."""
    c = TelemetryCollector()
    p = c.get_latency_percentiles()
    assert p["p50"] == 0.0
    assert p["p99"] == 0.0


def test_get_records_filtered_by_run_id():
    """get_records filters by run_id when provided."""
    c = TelemetryCollector()
    c.record(_make_record(run_id="run-a"))
    c.record(_make_record(run_id="run-b"))
    c.record(_make_record(run_id="run-a"))

    filtered = c.get_records(run_id="run-a")
    assert len(filtered) == 2
    assert all(r.run_id == "run-a" for r in filtered)


@pytest.mark.asyncio
async def test_wrap_adapter_call_records_failed_result():
    """wrap_adapter_call records failure when adapter returns success=False."""
    c = TelemetryCollector()
    failed_result = AdapterRunResult(
        success=False,
        token_usage=100,
        cost_estimate=0.001,
        error="validation failed",
    )
    adapter = _FakeAdapter(result=failed_result)
    request = _make_request()

    result = await c.wrap_adapter_call(adapter, request, "run-fail", 3, "critic")

    assert result.success is False
    assert len(c.get_records()) == 1
    rec = c.get_records()[0]
    assert rec.success is False
    assert rec.error == "validation failed"
