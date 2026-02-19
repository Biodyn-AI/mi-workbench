"""Tests for adapters: mock, registry, and CLI adapter availability checks."""
import asyncio

import pytest

from backend.models import (
    AdapterRunRequest,
    AdapterRunResult,
    PromptBundle,
    ProviderName,
    WorkspaceContext,
)
from backend.adapters.mock import MockAdapter
from backend.adapters.claude_code import ClaudeCodeAdapter
from backend.adapters.codex_cli import CodexCliAdapter
from backend.adapters.gemini_cli import GeminiCliAdapter
from backend.adapters.registry import get_adapter, list_adapters
from backend.adapters.base import BaseAdapter


def _make_request(system: str = "You are an executor.",
                  user: str = "Analyze attention weights.") -> AdapterRunRequest:
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(
            system_prompt=system,
            user_prompt=user,
        ),
        workspace_context=WorkspaceContext(workspace_path="/tmp/test_ws"),
        timeout_seconds=30,
    )


# ── Mock Adapter Tests ───────────────────────────────────────────────


class TestMockAdapter:
    @pytest.fixture(autouse=True)
    def _reset_mock_iteration(self):
        MockAdapter.reset_iteration_count()

    @pytest.mark.asyncio
    async def test_mock_produces_valid_output(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        result = await adapter.run(_make_request())
        assert isinstance(result, AdapterRunResult)
        assert result.success is True
        assert len(result.output) > 0
        assert result.token_usage > 0
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_mock_smoke_test(self):
        adapter = MockAdapter()
        info = await adapter.smoke_test()
        assert info["status"] == "ok"
        assert info["adapter"] == "mock"

    def test_mock_is_always_available(self):
        adapter = MockAdapter()
        assert adapter.is_available() is True

    @pytest.mark.asyncio
    async def test_mock_tracks_invocations(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        assert adapter.invocation_count == 0
        await adapter.run(_make_request())
        await adapter.run(_make_request())
        assert adapter.invocation_count == 2
        assert len(adapter.history) == 2

    @pytest.mark.asyncio
    async def test_mock_detects_reviewer_role(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        request = _make_request(system="You are a reviewer.", user="Review the analysis.")
        result = await adapter.run(request)
        assert result.success is True
        assert "EVAL" in result.output or "Review" in result.output

    @pytest.mark.asyncio
    async def test_mock_failure_rate(self):
        adapter = MockAdapter(failure_rate=1.0, min_delay=0.01, max_delay=0.02)
        result = await adapter.run(_make_request())
        assert result.success is False
        assert result.error is not None
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_mock_artifacts_by_role(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        # Executor role -> MECH.md
        result = await adapter.run(_make_request(
            system="You are an executor.", user="Execute the analysis."
        ))
        assert "MECH.md" in result.artifacts_written

        # Reviewer role -> EVAL.md
        result = await adapter.run(_make_request(
            system="You are a reviewer.", user="Evaluate the analysis."
        ))
        assert "EVAL.md" in result.artifacts_written

    @pytest.mark.asyncio
    async def test_mock_structured_output(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        result = await adapter.run(_make_request())
        assert "role" in result.structured_output
        assert result.structured_output["mock"] is True
        assert "iteration" in result.structured_output
        assert result.structured_output["iteration"] == 1

    @pytest.mark.asyncio
    async def test_mock_grade_progression(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        grades = []
        for _ in range(4):
            result = await adapter.run(_make_request(
                system="You are a reviewer.", user="Review the analysis."
            ))
            grades.append(result.structured_output["grade"])
        assert grades == ["C", "B", "B+", "A-"]

    @pytest.mark.asyncio
    async def test_mock_critique_severity_decreases(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        # Iteration 1: should have CRITICAL
        r1 = await adapter.run(_make_request(
            system="You are a reviewer.", user="Review the analysis."
        ))
        assert "[CRITICAL]" in r1.output
        # Iteration 4: should NOT have CRITICAL or HIGH
        _ = await adapter.run(_make_request(
            system="You are a reviewer.", user="Review the analysis."
        ))
        _ = await adapter.run(_make_request(
            system="You are a reviewer.", user="Review the analysis."
        ))
        r4 = await adapter.run(_make_request(
            system="You are a reviewer.", user="Review the analysis."
        ))
        assert "[CRITICAL]" not in r4.output
        assert "[HIGH]" not in r4.output

    @pytest.mark.asyncio
    async def test_mock_executor_has_all_artifact_markers(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        result = await adapter.run(_make_request())
        assert "# MECH.md" in result.output
        assert "# EVAL.md" in result.output
        assert "# XP.md" in result.output

    @pytest.mark.asyncio
    async def test_mock_adversarial_role(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        result = await adapter.run(_make_request(
            system="You are an adversarial reviewer.", user="Challenge the claims."
        ))
        assert "Adversarial Review" in result.output
        assert result.structured_output["role"] == "adversarial"

    @pytest.mark.asyncio
    async def test_mock_idea_generator_role(self):
        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        result = await adapter.run(_make_request(
            system="You are an idea generator.", user="Propose follow-up studies."
        ))
        assert "Proposed Follow-up Studies" in result.output
        assert "XP.md" in result.artifacts_written


# ── Registry Tests ───────────────────────────────────────────────────


class TestRegistry:
    def test_get_mock_adapter(self):
        adapter = get_adapter(ProviderName.MOCK)
        assert isinstance(adapter, BaseAdapter)
        assert adapter.is_available() is True

    def test_get_claude_code_adapter(self):
        adapter = get_adapter(ProviderName.CLAUDE_CODE)
        assert isinstance(adapter, ClaudeCodeAdapter)

    def test_get_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_adapter("nonexistent")

    def test_list_adapters_returns_dict(self):
        adapters = list_adapters()
        assert isinstance(adapters, dict)
        assert "mock" in adapters
        assert adapters["mock"] is True
        assert "claude_code" in adapters
        assert "codex_cli" in adapters
        assert "gemini_cli" in adapters


# ── CLI Adapter Availability Tests ───────────────────────────────────


class TestCliAdapters:
    def test_claude_code_is_available_returns_bool(self):
        adapter = ClaudeCodeAdapter()
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_codex_cli_is_available_returns_bool(self):
        adapter = CodexCliAdapter()
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_gemini_cli_is_available_returns_bool(self):
        adapter = GeminiCliAdapter()
        result = adapter.is_available()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_unavailable_adapter_returns_error(self):
        adapter = ClaudeCodeAdapter(binary="nonexistent_binary_xyz")
        assert adapter.is_available() is False
        result = await adapter.run(_make_request())
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_unavailable_smoke_test(self):
        adapter = ClaudeCodeAdapter(binary="nonexistent_binary_xyz")
        info = await adapter.smoke_test()
        assert info["status"] == "unavailable"
