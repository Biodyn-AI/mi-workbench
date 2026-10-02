"""Tests for the upgraded Claude Code adapter."""
import json
import pytest

from backend.adapters.claude_code import ClaudeCodeAdapter, categorize_error
from backend.models import (
    AdapterRunRequest,
    AdapterRunResult,
    PromptBundle,
    WorkspaceContext,
)


def _make_request(
    system: str = "You are an executor.",
    user: str = "Analyze attention weights.",
    workspace_path: str = "/tmp/test_ws",
    files_to_read: list[str] | None = None,
    files_to_write: list[str] | None = None,
) -> AdapterRunRequest:
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(
            system_prompt=system,
            user_prompt=user,
        ),
        workspace_context=WorkspaceContext(workspace_path=workspace_path),
        files_to_read=files_to_read or [],
        files_to_write=files_to_write or [],
        timeout_seconds=30,
    )


# ── Command building tests ──────────────────────────────────────────


class TestBuildCommand:
    def test_build_command_includes_flags(self):
        adapter = ClaudeCodeAdapter(binary="claude")
        request = _make_request()
        cmd = adapter.build_command(request)
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--max-turns" in cmd
        # tools enabled (request default): max_turns, not a single turn
        assert cmd[cmd.index("--max-turns") + 1] == str(adapter.max_turns)

    def test_build_command_includes_system_prompt(self):
        adapter = ClaudeCodeAdapter(binary="claude")
        request = _make_request(system="Test system prompt")
        cmd = adapter.build_command(request)
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "Test system prompt"

    def test_build_command_includes_allowed_tools(self):
        adapter = ClaudeCodeAdapter(binary="claude")
        request = _make_request(workspace_path="/some/path")
        cmd = adapter.build_command(request)
        assert "--allowedTools" in cmd

    def test_build_command_no_tools_without_workspace(self):
        adapter = ClaudeCodeAdapter(binary="claude")
        request = _make_request(workspace_path="")
        cmd = adapter.build_command(request)
        assert "--allowedTools" not in cmd


# ── Error categorization tests ──────────────────────────────────────


class TestErrorCategorization:
    def test_rate_limit_by_code(self):
        assert categorize_error(429, "") == "rate_limit"

    def test_rate_limit_by_message(self):
        assert categorize_error(1, "rate limit exceeded") == "rate_limit"

    def test_rate_limit_by_status_in_stderr(self):
        assert categorize_error(1, "HTTP 429 Too Many Requests") == "rate_limit"

    def test_auth_error(self):
        assert categorize_error(1, "401 Unauthorized") == "auth"

    def test_auth_error_403(self):
        assert categorize_error(1, "403 Forbidden") == "auth"

    def test_timeout_error(self):
        assert categorize_error(1, "Connection timed out") == "timeout"

    def test_network_error(self):
        assert categorize_error(1, "ECONNREFUSED") == "network"

    def test_unknown_error(self):
        assert categorize_error(1, "something unexpected") == "unknown"


# ── JSON output parsing tests ───────────────────────────────────────


class TestJsonOutputParsing:
    def test_json_output_parsing(self):
        adapter = ClaudeCodeAdapter()
        raw = json.dumps({"result": "Hello world", "tokens": 100})
        text, data = adapter.parse_json_output(raw)
        assert text == "Hello world"
        assert data["tokens"] == 100

    def test_json_output_with_content_field(self):
        adapter = ClaudeCodeAdapter()
        raw = json.dumps({"content": "Some content"})
        text, data = adapter.parse_json_output(raw)
        assert text == "Some content"

    def test_json_output_with_messages(self):
        adapter = ClaudeCodeAdapter()
        raw = json.dumps({
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Response text"},
            ]
        })
        text, data = adapter.parse_json_output(raw)
        assert "Response text" in text

    def test_non_json_fallback(self):
        adapter = ClaudeCodeAdapter()
        raw = "This is plain text output"
        text, data = adapter.parse_json_output(raw)
        assert text == raw
        assert data == {}


# ── Prompt building tests ───────────────────────────────────────────


class TestBuildPrompt:
    def test_file_content_prepended(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        adapter = ClaudeCodeAdapter()
        request = _make_request(files_to_read=[str(test_file)])
        prompt = adapter.build_prompt(request)
        assert f"=== File: {test_file} ===" in prompt
        assert "print('hello')" in prompt
        assert "=== End File ===" in prompt
        # User prompt should come after
        assert prompt.index("print('hello')") < prompt.index("Analyze attention")

    def test_files_to_write_instruction(self):
        adapter = ClaudeCodeAdapter()
        request = _make_request(files_to_write=["/output/result.md"])
        prompt = adapter.build_prompt(request)
        assert "/output/result.md" in prompt
        assert "write output" in prompt.lower()

    def test_prompt_without_files(self):
        adapter = ClaudeCodeAdapter()
        request = _make_request()
        prompt = adapter.build_prompt(request)
        assert "=== File:" not in prompt
        assert "Analyze attention weights." in prompt

    def test_developer_prompt_included(self):
        adapter = ClaudeCodeAdapter()
        request = AdapterRunRequest(
            prompt_bundle=PromptBundle(
                system_prompt="system",
                developer_prompt="dev instructions",
                user_prompt="user query",
            ),
            workspace_context=WorkspaceContext(workspace_path=""),
        )
        prompt = adapter.build_prompt(request)
        assert "dev instructions" in prompt
        assert "user query" in prompt


# ── Availability tests ──────────────────────────────────────────────


class TestAdapterAvailability:
    @pytest.mark.asyncio
    async def test_unavailable_adapter_returns_error(self):
        adapter = ClaudeCodeAdapter(binary="nonexistent_binary_xyz")
        assert adapter.is_available() is False
        result = await adapter.run(_make_request())
        assert result.success is False
        assert "not found" in result.error
