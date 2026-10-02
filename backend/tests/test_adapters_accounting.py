"""E4 tests: request/result accounting fields, mock adapter accounting, telemetry."""
from __future__ import annotations

import aiosqlite
import pytest

from backend.adapters.base import BaseAdapter
from backend.adapters.mock import MockAdapter
from backend.adapters.registry import get_adapter
from backend.models import (
    AdapterRunRequest,
    AdapterRunResult,
    PromptBundle,
    ProviderName,
    WorkspaceContext,
)
from backend.telemetry import storage
from backend.telemetry.collector import TelemetryCollector, TelemetryRecord


def _req(**kw) -> AdapterRunRequest:
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(system_prompt="You are a reviewer.", user_prompt="Review the analysis."),
        workspace_context=WorkspaceContext(workspace_path=""),
        **kw,
    )


# ── Model defaults / backward compatibility ──────────────────────────


class TestModelFields:
    def test_request_defaults(self):
        r = _req()
        assert r.model == "" and r.reasoning_effort == "" and r.allow_tools is True
        assert r.timeout_seconds == 300

    def test_result_defaults(self):
        r = AdapterRunResult(success=True)
        assert (r.model, r.cli_version, r.provider, r.reasoning_effort) == ("", "", "", "")
        assert (r.input_tokens, r.output_tokens, r.cached_input_tokens) == (0, 0, 0)
        assert r.raw_usage == {} and r.token_usage == 0 and r.cost_estimate == 0.0

    def test_result_raw_usage_not_shared(self):
        a, b = AdapterRunResult(success=True), AdapterRunResult(success=True)
        a.raw_usage["x"] = 1
        assert b.raw_usage == {}


# ── Mock adapter ─────────────────────────────────────────────────────


class TestMockAccounting:
    @pytest.fixture(autouse=True)
    def _reset(self):
        MockAdapter.reset_iteration_count()
        yield
        MockAdapter.reset_iteration_count()

    @pytest.mark.asyncio
    async def test_mock_new_fields(self):
        adapter = MockAdapter(min_delay=0, max_delay=0)
        r = await adapter.run(_req(reasoning_effort="low", model="ignored"))
        assert r.success is True
        assert r.model == "mock" and r.provider == "mock" and r.cli_version == "mock"
        assert r.reasoning_effort == "low"
        assert r.input_tokens + r.output_tokens == r.token_usage
        assert r.input_tokens == int(r.token_usage * 0.7)
        assert 1400 <= r.token_usage <= 1600  # reviewer target unchanged
        assert r.cost_estimate == pytest.approx(r.token_usage * 0.000003)
        assert r.raw_usage["synthetic"] is True
        assert await adapter.cli_version() == "mock"

    @pytest.mark.asyncio
    async def test_mock_content_unchanged(self):
        adapter = MockAdapter(min_delay=0, max_delay=0)
        r = await adapter.run(_req())
        assert "[CRITICAL]" in r.output
        assert r.structured_output == {"role": "reviewer", "mock": True, "iteration": 1, "grade": "C"}
        assert r.raw_log.startswith("[mock] role=reviewer iter=1 tokens=")

    @pytest.mark.asyncio
    async def test_mock_failure_has_model(self):
        r = await MockAdapter(failure_rate=1.0, min_delay=0, max_delay=0).run(_req())
        assert r.success is False and r.model == "mock" and r.cli_version == "mock"


class TestBaseAdapterVersion:
    @pytest.mark.asyncio
    async def test_default_cli_version_empty(self):
        class _A(BaseAdapter):
            name = "x"

            async def run(self, request):
                return AdapterRunResult(success=True)

            async def smoke_test(self):
                return {}

            def is_available(self):
                return True

        assert await _A().cli_version() == ""

    def test_registry_adapters_expose_cli_version(self):
        for p in ProviderName:
            assert callable(getattr(get_adapter(p), "cli_version"))


# ── Telemetry collector ──────────────────────────────────────────────


class _FakeAdapter:
    name = "fake_cli"

    def __init__(self, result: AdapterRunResult):
        self._result = result

    async def run(self, request):
        return self._result


class TestTelemetryCollector:
    @pytest.mark.asyncio
    async def test_provider_reported_split(self):
        res = AdapterRunResult(
            success=True, token_usage=1005, cost_estimate=0.02,
            input_tokens=1000, output_tokens=5, cached_input_tokens=800,
            model="claude-sonnet-4-5", cli_version="2.1.212 (Claude Code)", reasoning_effort="low",
        )
        c = TelemetryCollector()
        await c.wrap_adapter_call(_FakeAdapter(res), _req(model="sonnet"), "run-x", 1, "reviewer")
        rec = c.get_records()[0]
        assert (rec.tokens_input, rec.tokens_output, rec.tokens_total) == (1000, 5, 1005)
        assert rec.tokens_cached_input == 800
        assert rec.model == "claude-sonnet-4-5"  # answering model beats requested alias
        assert rec.cli_version == "2.1.212 (Claude Code)"
        assert rec.reasoning_effort == "low"
        assert rec.tokens_estimated is False
        assert rec.cost_estimate == 0.02

    @pytest.mark.asyncio
    async def test_legacy_total_only_is_flagged_estimate(self):
        res = AdapterRunResult(success=True, token_usage=500, cost_estimate=0.005)
        c = TelemetryCollector()
        await c.wrap_adapter_call(_FakeAdapter(res), _req(model="gpt-5.5", reasoning_effort="high"),
                                  "run-y", 2, "executor")
        rec = c.get_records()[0]
        assert (rec.tokens_input, rec.tokens_output, rec.tokens_total) == (350, 150, 500)
        assert rec.tokens_estimated is True
        assert rec.model == "gpt-5.5"  # falls back to the requested model
        # Effort comes only from the adapter result (it may not have applied the
        # requested one, e.g. Gemini "unsupported"); "" = not reported.
        assert rec.reasoning_effort == ""

    @pytest.mark.asyncio
    async def test_zero_tokens_not_flagged(self):
        c = TelemetryCollector()
        await c.wrap_adapter_call(_FakeAdapter(AdapterRunResult(success=False, error="e")), _req(), "r", 0, "x")
        rec = c.get_records()[0]
        assert rec.tokens_total == 0 and rec.tokens_estimated is False and rec.success is False

    def test_csv_export_includes_new_columns(self):
        c = TelemetryCollector()
        c.record(TelemetryRecord(timestamp="t", run_id="r", iteration=0, role="x", adapter="a", model="m",
                                 tokens_cached_input=3, cli_version="v1"))
        header = c.export_csv().splitlines()[0]
        for col in ("tokens_cached_input", "cli_version", "reasoning_effort", "tokens_estimated"):
            assert col in header


# ── Telemetry storage (schema + migration) ───────────────────────────


_OLD_SCHEMA = """
CREATE TABLE telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, run_id TEXT NOT NULL, iteration INTEGER NOT NULL,
    role TEXT NOT NULL, adapter TEXT NOT NULL, model TEXT NOT NULL DEFAULT '',
    tokens_input INTEGER NOT NULL DEFAULT 0, tokens_output INTEGER NOT NULL DEFAULT 0,
    tokens_total INTEGER NOT NULL DEFAULT 0, cost_estimate REAL NOT NULL DEFAULT 0.0,
    latency_seconds REAL NOT NULL DEFAULT 0.0, success INTEGER NOT NULL DEFAULT 1, error TEXT
);
INSERT INTO telemetry (timestamp, run_id, iteration, role, adapter, model, tokens_total)
VALUES ('2026-01-01T00:00:00', 'old-run', 1, 'executor', 'claude_code', 'm-old', 42);
"""


class TestTelemetryStorage:
    @pytest.mark.asyncio
    async def test_roundtrip_new_fields(self, tmp_path):
        db = str(tmp_path / "tel.db")
        await storage.init_telemetry_table(db)
        rec = TelemetryRecord(
            timestamp="2026-10-01T00:00:00", run_id="r1", iteration=3, role="reviewer",
            adapter="codex_cli", model="gpt-5.5", tokens_input=15350, tokens_output=5,
            tokens_total=15355, success=True, tokens_cached_input=1408,
            cli_version="codex-cli 0.145.0", reasoning_effort="low", tokens_estimated=False,
        )
        await storage.save_record(rec, db)
        got = (await storage.get_records(run_id="r1", db_path=db))[0]
        assert got == rec

    @pytest.mark.asyncio
    async def test_migrates_old_table(self, tmp_path):
        db = str(tmp_path / "old.db")
        async with aiosqlite.connect(db) as conn:
            await conn.executescript(_OLD_SCHEMA)
            await conn.commit()
        await storage.init_telemetry_table(db)
        await storage.init_telemetry_table(db)  # idempotent
        async with aiosqlite.connect(db) as conn:
            cur = await conn.execute("PRAGMA table_info(telemetry)")
            cols = {row[1] for row in await cur.fetchall()}
        assert {"tokens_cached_input", "cli_version", "reasoning_effort", "tokens_estimated"} <= cols
        old = (await storage.get_records(run_id="old-run", db_path=db))[0]
        assert old.tokens_total == 42 and old.cli_version == "" and old.tokens_estimated is False
        summary = await storage.get_summary(db_path=db)
        assert summary["total_calls"] == 1
