"""Tests for structured logging, Prometheus metrics, and health check."""
from __future__ import annotations

import json
import logging

import pytest
from httpx import AsyncClient

from backend.logging_config import (
    RunContextFilter,
    StructuredFormatter,
    clear_run_context,
    set_run_context,
)

pytestmark = pytest.mark.asyncio


# ── Structured logging tests ────────────────────────────────────────


def test_structured_formatter_basic() -> None:
    """Format a log record and verify JSON output with timestamp, level, message."""
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello world",
        args=None,
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["logger"] == "test.logger"
    assert data["message"] == "hello world"
    assert "timestamp" in data
    assert data["timestamp"].endswith("Z")


def test_structured_formatter_with_run_context() -> None:
    """Set run context via contextvars and verify run_id appears in JSON."""
    formatter = StructuredFormatter()
    ctx_filter = RunContextFilter()

    set_run_context("run-abc-123", iteration=5)
    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="with context",
            args=None,
            exc_info=None,
        )
        ctx_filter.filter(record)
        output = formatter.format(record)
        data = json.loads(output)
        assert data["run_id"] == "run-abc-123"
        assert data["iteration"] == 5
    finally:
        clear_run_context()


def test_structured_formatter_with_exception() -> None:
    """Verify exception field appears when exc_info is set."""
    formatter = StructuredFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="error occurred",
        args=None,
        exc_info=exc_info,
    )
    output = formatter.format(record)
    data = json.loads(output)
    assert "exception" in data
    assert "ValueError" in data["exception"]
    assert "test error" in data["exception"]


def test_run_context_filter() -> None:
    """Verify filter injects contextvars into log records."""
    ctx_filter = RunContextFilter()
    set_run_context("run-filter-test", iteration=42)
    try:
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="filter test",
            args=None,
            exc_info=None,
        )
        result = ctx_filter.filter(record)
        assert result is True
        assert record.run_id == "run-filter-test"  # type: ignore[attr-defined]
        assert record.iteration == 42  # type: ignore[attr-defined]
    finally:
        clear_run_context()


# ── Prometheus metrics tests ─────────────────────────────────────────


async def test_metrics_endpoint_returns_prometheus_format(client: AsyncClient) -> None:
    """GET /metrics returns text with TYPE/HELP lines."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "# HELP miw_active_runs" in body
    assert "# TYPE miw_active_runs gauge" in body
    assert "# HELP miw_total_runs" in body
    assert "# TYPE miw_total_runs counter" in body
    assert "miw_iterations_total" in body
    assert "miw_adapter_errors_total" in body
    assert "miw_convergence_triggers_total" in body


async def test_metrics_active_runs_count(client: AsyncClient) -> None:
    """Verify active runs count reflects reality (0 when no runs executing)."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    # No runs are executing during tests, so active_runs should be 0
    assert "miw_active_runs 0" in body


# ── Health check tests ───────────────────────────────────────────────


async def test_health_check_reports_db_status(client: AsyncClient) -> None:
    """Health returns database: 'ok' when DB is accessible."""
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"
    assert data["version"] == "0.1.0"


async def test_health_check_reports_active_runs(client: AsyncClient) -> None:
    """Health includes active_runs count."""
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_runs" in data
    assert isinstance(data["active_runs"], int)
    assert data["active_runs"] == 0
