"""Tests for retry wrapper with exponential backoff."""
import asyncio
import time

import pytest

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult, PromptBundle, WorkspaceContext
from backend.orchestrator.retry import is_transient_error, run_with_retry


# ── Helpers ──────────────────────────────────────────────────────────


def _make_request() -> AdapterRunRequest:
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(
            system_prompt="test", user_prompt="test"
        ),
        workspace_context=WorkspaceContext(workspace_path="/tmp"),
    )


class _SequenceAdapter(BaseAdapter):
    """Adapter that returns a predefined sequence of results."""
    name = "sequence"

    def __init__(self, results: list[AdapterRunResult]):
        self._results = list(results)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        self._call_count += 1
        if self._results:
            return self._results.pop(0)
        return AdapterRunResult(success=True, output="fallback")

    async def smoke_test(self) -> dict:
        return {"status": "ok"}

    def is_available(self) -> bool:
        return True


def _ok_result(output: str = "ok") -> AdapterRunResult:
    return AdapterRunResult(success=True, output=output)


def _transient_error(msg: str = "connection timeout") -> AdapterRunResult:
    return AdapterRunResult(success=False, error=msg)


def _permanent_error(msg: str = "invalid API key") -> AdapterRunResult:
    return AdapterRunResult(success=False, error=msg)


# ── Tests ────────────────────────────────────────────────────────────


class TestIsTransientError:
    def test_success_is_not_transient(self):
        assert is_transient_error(_ok_result()) is False

    def test_timeout_is_transient(self):
        assert is_transient_error(_transient_error("request timeout")) is True

    def test_rate_limit_is_transient(self):
        assert is_transient_error(_transient_error("rate limit exceeded")) is True

    def test_503_is_transient(self):
        assert is_transient_error(_transient_error("HTTP 503 Service Unavailable")) is True

    def test_429_is_transient(self):
        assert is_transient_error(_transient_error("429 Too Many Requests")) is True

    def test_502_is_transient(self):
        assert is_transient_error(_transient_error("502 Bad Gateway")) is True

    def test_overloaded_is_transient(self):
        assert is_transient_error(_transient_error("server overloaded")) is True

    def test_connection_is_transient(self):
        assert is_transient_error(_transient_error("connection refused")) is True

    def test_permanent_error_not_transient(self):
        assert is_transient_error(_permanent_error("invalid API key")) is False

    def test_none_error_not_transient(self):
        r = AdapterRunResult(success=False, error=None)
        assert is_transient_error(r) is False


class TestRunWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_without_retry(self):
        adapter = _SequenceAdapter([_ok_result("success")])
        result = await run_with_retry(adapter, _make_request(), max_retries=3, base_delay=0.01)
        assert result.success is True
        assert result.output == "success"
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        adapter = _SequenceAdapter([
            _transient_error("connection timeout"),
            _transient_error("rate limit exceeded"),
            _ok_result("recovered"),
        ])
        result = await run_with_retry(adapter, _make_request(), max_retries=3, base_delay=0.01)
        assert result.success is True
        assert result.output == "recovered"
        assert adapter.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        adapter = _SequenceAdapter([_permanent_error("invalid API key")])
        result = await run_with_retry(adapter, _make_request(), max_retries=3, base_delay=0.01)
        assert result.success is False
        assert "invalid API key" in result.error
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failure(self):
        adapter = _SequenceAdapter([
            _transient_error("503 Service Unavailable"),
            _ok_result("back up"),
        ])
        result = await run_with_retry(adapter, _make_request(), max_retries=3, base_delay=0.01)
        assert result.success is True
        assert result.output == "back up"
        assert adapter.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        adapter = _SequenceAdapter([
            _transient_error("timeout"),
            _transient_error("timeout"),
            _transient_error("timeout"),
            _transient_error("timeout"),
        ])
        result = await run_with_retry(adapter, _make_request(), max_retries=3, base_delay=0.01)
        assert result.success is False
        # 1 initial + 3 retries = 4 calls
        assert adapter.call_count == 4

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        adapter = _SequenceAdapter([
            _transient_error("timeout"),
            _transient_error("timeout"),
            _ok_result("ok"),
        ])
        t0 = time.monotonic()
        result = await run_with_retry(
            adapter, _make_request(), max_retries=3, base_delay=0.05, max_delay=1.0,
        )
        elapsed = time.monotonic() - t0
        assert result.success is True
        # base_delay=0.05: first retry ~0.05s, second ~0.10s = ~0.15s total
        # With jitter (+-25%), minimum is 0.75 * (0.05 + 0.10) = 0.1125s
        assert elapsed >= 0.05  # at least some delay happened

    @pytest.mark.asyncio
    async def test_on_retry_callback_called(self):
        callbacks = []

        def on_retry(attempt, error, delay):
            callbacks.append((attempt, error))

        adapter = _SequenceAdapter([
            _transient_error("429 rate limited"),
            _transient_error("timeout"),
            _ok_result("ok"),
        ])
        result = await run_with_retry(
            adapter, _make_request(), max_retries=3, base_delay=0.01, on_retry=on_retry,
        )
        assert result.success is True
        assert len(callbacks) == 2
        assert callbacks[0][0] == 1  # first retry attempt
        assert callbacks[1][0] == 2  # second retry attempt
        assert "429" in callbacks[0][1]
        assert "timeout" in callbacks[1][1]

    @pytest.mark.asyncio
    async def test_zero_retries_means_single_attempt(self):
        adapter = _SequenceAdapter([_transient_error("timeout")])
        result = await run_with_retry(adapter, _make_request(), max_retries=0, base_delay=0.01)
        assert result.success is False
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_max_delay_caps_backoff(self):
        adapter = _SequenceAdapter([
            _transient_error("timeout"),
            _ok_result("ok"),
        ])
        t0 = time.monotonic()
        result = await run_with_retry(
            adapter, _make_request(), max_retries=3, base_delay=100.0, max_delay=0.05,
        )
        elapsed = time.monotonic() - t0
        assert result.success is True
        # max_delay=0.05 should cap the backoff well below 100s
        assert elapsed < 1.0
