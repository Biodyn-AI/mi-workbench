"""E4 tests: CLI adapter command construction, output parsing and failure handling.

No real CLI is called. ``run()`` is exercised against small fake executables
that record argv/stdin/cwd and replay captured CLI output from
``backend/tests/fixtures`` (see the README there for which samples are real).
"""
from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from backend.adapters import cli_common
from backend.adapters.claude_code import ClaudeCodeAdapter, categorize_error as claude_categorize
from backend.adapters.cli_common import categorize_error, clear_cli_version_cache, get_cli_version
from backend.adapters.codex_cli import CodexCliAdapter, parse_codex_jsonl
from backend.adapters.gemini_cli import (
    GeminiCliAdapter,
    _find_gemini_payload,
    neutralize_at_expansion,
    parse_gemini_stats,
)
from backend.models import AdapterRunRequest, PromptBundle, WorkspaceContext
from backend.orchestrator.retry import is_transient_error

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def _req(
    system: str = "You are the reviewer.",
    user: str = "Review this. Precision@k = 0.4 at @k=10.",
    developer: str = "",
    workspace: str = "",
    timeout: int = 30,
    **kw,
) -> AdapterRunRequest:
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(system_prompt=system, developer_prompt=developer, user_prompt=user),
        workspace_context=WorkspaceContext(workspace_path=workspace),
        timeout_seconds=timeout,
        **kw,
    )


@pytest.fixture(autouse=True)
def _fresh_version_cache():
    clear_cli_version_cache()
    yield
    clear_cli_version_cache()


# ── Fake CLI executable ───────────────────────────────────────────────

_FAKE_TEMPLATE = r'''#!{python}
import json, os, subprocess, sys, time
d = {dir!r}
def rd(name, default=""):
    p = os.path.join(d, name)
    return open(p).read() if os.path.exists(p) else default
if sys.argv[1:] == ["--version"]:
    sys.stdout.write(rd("version.txt", "fake-cli 1.2.3\n"))
    sys.exit(int(rd("version_exit.txt", "0") or 0))
if os.path.exists(os.path.join(d, "spawn_child")):
    # First thing (before reading stdin), and written atomically, so a test
    # can always check that the grandchild was killed.
    child = subprocess.Popen(["sleep", "30"])
    with open(os.path.join(d, "child.pid.tmp"), "w") as fh:
        fh.write(str(child.pid))
    os.replace(os.path.join(d, "child.pid.tmp"), os.path.join(d, "child.pid"))
with open(os.path.join(d, "env_codex_home.txt"), "w") as fh:
    fh.write(os.environ.get("CODEX_HOME", "<unset>"))
with open(os.path.join(d, "env_gemini_settings.txt"), "w") as fh:
    fh.write(os.environ.get("GEMINI_CLI_SYSTEM_SETTINGS_PATH", "<unset>"))
with open(os.path.join(d, "argv.json"), "w") as fh:
    json.dump(sys.argv[1:], fh)
data = sys.stdin.read()
with open(os.path.join(d, "stdin.txt"), "w") as fh:
    fh.write(data)
with open(os.path.join(d, "cwd.txt"), "w") as fh:
    fh.write(os.getcwd())
time.sleep(float(rd("sleep.txt", "0") or 0))
sys.stdout.write(rd("stdout.txt"))
sys.stderr.write(rd("stderr.txt"))
sys.exit(int(rd("exit.txt", "0") or 0))
'''


class FakeCli:
    def __init__(self, root: Path, name: str):
        self.dir = root / f"fake_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        python = sys.executable if " " not in sys.executable else "/usr/bin/env python3"
        self.path = self.dir / name
        self.path.write_text(_FAKE_TEMPLATE.format(python=python, dir=str(self.dir)))
        self.path.chmod(self.path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def set(self, stdout: str = "", stderr: str = "", exit_code: int = 0, sleep: float = 0.0,
            version: str | None = None, spawn_child: bool = False) -> "FakeCli":
        (self.dir / "stdout.txt").write_text(stdout)
        (self.dir / "stderr.txt").write_text(stderr)
        (self.dir / "exit.txt").write_text(str(exit_code))
        (self.dir / "sleep.txt").write_text(str(sleep))
        if version is not None:
            (self.dir / "version.txt").write_text(version)
        if spawn_child:
            (self.dir / "spawn_child").write_text("1")
        return self

    @property
    def argv(self) -> list[str]:
        return json.loads((self.dir / "argv.json").read_text())

    @property
    def stdin(self) -> str:
        return (self.dir / "stdin.txt").read_text()

    @property
    def cwd(self) -> str:
        return (self.dir / "cwd.txt").read_text()


@pytest.fixture
def fake(tmp_path):
    def _make(name: str) -> FakeCli:
        return FakeCli(tmp_path, name)
    return _make


# ══ Claude: command construction ═════════════════════════════════════


class TestClaudeCommand:
    def test_base_flags_unchanged(self):
        # tools off (reviewer): a single answer, --max-turns 1
        cmd = ClaudeCodeAdapter().build_command(_req(allow_tools=False))
        assert cmd[:6] == ["claude", "-p", "--output-format", "json", "--max-turns", "1"]
        # tools on: every tool call costs a turn, so the limit is max_turns
        cmd_tools = ClaudeCodeAdapter().build_command(_req())
        assert cmd_tools[cmd_tools.index("--max-turns") + 1] == "30"
        assert "--model" not in cmd and "--effort" not in cmd
        assert cmd[cmd.index("--system-prompt") + 1] == "You are the reviewer."

    def test_model_and_effort_forwarded(self):
        cmd = ClaudeCodeAdapter().build_command(_req(model="sonnet", reasoning_effort="high"))
        assert cmd[cmd.index("--model") + 1] == "sonnet"
        assert cmd[cmd.index("--effort") + 1] == "high"

    def test_adapter_defaults_and_request_override(self):
        adapter = ClaudeCodeAdapter(model="opus", reasoning_effort="low")
        cmd = adapter.build_command(_req())
        assert cmd[cmd.index("--model") + 1] == "opus"
        assert cmd[cmd.index("--effort") + 1] == "low"
        cmd = adapter.build_command(_req(model="sonnet", reasoning_effort="max"))
        assert cmd[cmd.index("--model") + 1] == "sonnet"
        assert cmd[cmd.index("--effort") + 1] == "max"

    def test_allowed_tools_kept_with_workspace(self):
        cmd = ClaudeCodeAdapter().build_command(_req(workspace="/some/ws"))
        assert cmd[cmd.index("--allowedTools") + 1] == "Read,Write,Edit,Bash,Glob,Grep"
        assert "--tools" not in cmd

    def test_no_tool_flags_without_workspace_when_tools_allowed(self):
        cmd = ClaudeCodeAdapter().build_command(_req(workspace=""))
        assert "--allowedTools" not in cmd and "--tools" not in cmd

    def test_tools_disabled_by_request(self):
        cmd = ClaudeCodeAdapter().build_command(_req(workspace="/some/ws", allow_tools=False))
        assert "--allowedTools" not in cmd
        i = cmd.index("--tools")
        assert cmd[i + 1] == ""
        assert "--strict-mcp-config" in cmd

    def test_tools_disabled_by_adapter(self):
        cmd = ClaudeCodeAdapter(allow_tools=False).build_command(_req(workspace="/some/ws"))
        assert "--allowedTools" not in cmd
        assert cmd[cmd.index("--tools") + 1] == ""

    def test_prompt_not_in_argv(self):
        cmd = ClaudeCodeAdapter().build_command(_req(user="UNIQUE_USER_TEXT"))
        assert not any("UNIQUE_USER_TEXT" in a for a in cmd)

    def test_categorize_error_reexported(self):
        assert claude_categorize is categorize_error
        assert categorize_error(1, "Failed to authenticate") == "auth"


# ══ Claude: parsing ══════════════════════════════════════════════════


class TestClaudeParsing:
    def test_success_fixture_usage_cost_model(self):
        data = json.loads(_fixture("claude_success_synthetic.json"))
        adapter = ClaudeCodeAdapter()
        assert adapter.extract_result(data) == "OK"
        u = adapter.parse_usage(data, requested_model="sonnet")
        assert u["input_tokens"] == 12 + 12000 + 1500
        assert u["cached_input_tokens"] == 12000
        assert u["output_tokens"] == 5
        assert u["cost"] == pytest.approx(0.0123)
        # The auxiliary haiku call has fewer output tokens: sonnet answered.
        assert u["model"] == "claude-sonnet-4-5"
        assert u["model_source"] == "modelUsage"
        assert u["raw_usage"]["usage"]["cache_read_input_tokens"] == 12000

    def test_auth_error_fixture_has_zero_usage_and_requested_model(self):
        data = json.loads(_fixture("claude_auth_error_sample.json"))
        u = ClaudeCodeAdapter.parse_usage(data, requested_model="sonnet")
        assert (u["input_tokens"], u["output_tokens"], u["cost"]) == (0, 0, 0.0)
        assert u["model"] == "sonnet" and u["model_source"] == "requested"

    def test_usage_falls_back_to_model_usage(self):
        data = {"result": "x", "modelUsage": {"m1": {"inputTokens": 10, "outputTokens": 4,
                                                      "cacheReadInputTokens": 6, "costUSD": 0.5}}}
        u = ClaudeCodeAdapter.parse_usage(data)
        assert (u["input_tokens"], u["output_tokens"], u["cached_input_tokens"]) == (16, 4, 6)
        assert u["cost"] == 0.5 and u["model"] == "m1"

    def test_parse_json_output_tolerates_leading_noise(self):
        raw = "some warning line\n" + json.dumps({"result": "hello", "is_error": False})
        text, data = ClaudeCodeAdapter().parse_json_output(raw)
        assert text == "hello" and data["result"] == "hello"


# ══ Claude: run() against a fake binary ══════════════════════════════


class TestClaudeRun:
    @pytest.mark.asyncio
    async def test_success(self, fake, tmp_path):
        cli = fake("claude").set(stdout=_fixture("claude_success_synthetic.json"), version="2.1.212 (Claude Code)\n")
        ws = tmp_path / "ws"
        ws.mkdir()
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(
            _req(workspace=str(ws), model="sonnet", reasoning_effort="low", developer="DEV"))
        assert r.success is True and r.error is None
        assert r.output == "OK"
        assert (r.input_tokens, r.output_tokens, r.cached_input_tokens) == (13512, 5, 12000)
        assert r.token_usage == 13517
        assert r.cost_estimate == pytest.approx(0.0123)
        assert r.model == "claude-sonnet-4-5"
        assert r.reasoning_effort == "low"
        assert r.provider == "claude_code"
        assert r.cli_version == "2.1.212 (Claude Code)"
        assert r.raw_usage["total_cost_usd"] == 0.0123
        assert r.structured_output["result"] == "OK"
        assert r.structured_output["adapter_meta"]["model_source"] == "modelUsage"
        # prompt on stdin, not argv; cwd is the workspace
        assert "DEV" in cli.stdin and "Precision@k" in cli.stdin
        assert not any("Precision@k" in a for a in cli.argv)
        assert os.path.realpath(cli.cwd) == os.path.realpath(ws)
        assert cli.argv[cli.argv.index("--model") + 1] == "sonnet"

    @pytest.mark.asyncio
    async def test_is_error_with_nonzero_exit(self, fake):
        cli = fake("claude").set(stdout=_fixture("claude_auth_error_sample.json"), exit_code=1)
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req(model="sonnet"))
        assert r.success is False
        assert r.output == ""
        assert r.exit_code == 1
        assert r.error.startswith("[auth]")
        assert "Failed to authenticate" in r.error
        assert r.model == "sonnet"
        assert "[stdout]" in r.raw_log

    @pytest.mark.asyncio
    async def test_is_error_with_zero_exit(self, fake):
        payload = {"type": "result", "subtype": "success", "is_error": True,
                   "result": "API Error: 529 overloaded", "usage": {}}
        cli = fake("claude").set(stdout=json.dumps(payload), exit_code=0)
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False and r.output == ""
        assert "overloaded" in r.error
        assert is_transient_error(r)  # retry wrapper still recognises it

    @pytest.mark.asyncio
    async def test_error_subtype_without_result(self, fake):
        payload = {"type": "result", "subtype": "error_max_turns", "is_error": False, "num_turns": 2,
                   "usage": {"input_tokens": 10, "output_tokens": 3}}
        cli = fake("claude").set(stdout=json.dumps(payload))
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False and r.output == ""
        assert "error_max_turns" in r.error
        # tokens are still accounted for failed calls
        assert (r.input_tokens, r.output_tokens) == (10, 3)

    @pytest.mark.asyncio
    async def test_empty_result_is_not_success(self, fake):
        payload = {"type": "result", "subtype": "success", "is_error": False, "result": "   "}
        cli = fake("claude").set(stdout=json.dumps(payload))
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False
        assert r.error.startswith("[empty_output]")

    @pytest.mark.asyncio
    async def test_nonzero_exit_non_json(self, fake):
        cli = fake("claude").set(stdout="", stderr="HTTP 429 rate limit exceeded", exit_code=1)
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False
        assert r.error.startswith("[rate_limit]")
        assert is_transient_error(r)

    @pytest.mark.asyncio
    async def test_nonzero_exit_no_output(self, fake):
        cli = fake("claude").set(exit_code=3)
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False
        assert r.error == "[unknown] exit code 3"

    @pytest.mark.asyncio
    async def test_non_json_stdout_success_backward_compatible(self, fake):
        cli = fake("claude").set(stdout="plain text answer\n")
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req())
        assert r.success is True and r.output == "plain text answer"
        assert r.token_usage == 0

    @pytest.mark.asyncio
    async def test_timeout_kills_process_tree(self, fake):
        cli = fake("claude").set(stdout="{}", sleep=20, spawn_child=True)
        t0 = time.monotonic()
        r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req(timeout=3))
        assert time.monotonic() - t0 < 12
        assert r.success is False
        assert r.error == "Timeout after 3s"
        assert r.exit_code == -1
        assert is_transient_error(r)
        pid_file = cli.dir / "child.pid"
        # The fake spawns its child first thing; if the file is missing the
        # check below would be vacuous, so fail instead.
        assert pid_file.exists(), "fake CLI never spawned its child; timeout fired too early"
        if True:  # the grandchild must not survive the timeout
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 5
            alive = True
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    alive = False
                    break
                time.sleep(0.1)
            assert not alive

    @pytest.mark.asyncio
    async def test_unavailable_binary(self):
        r = await ClaudeCodeAdapter(binary="nonexistent_binary_xyz_e4").run(_req(model="m"))
        assert r.success is False and "not found" in r.error
        assert r.model == "m" and r.provider == "claude_code"


# ══ Codex: command construction ══════════════════════════════════════


class TestCodexCommand:
    def test_worker_command(self):
        cmd = CodexCliAdapter().build_command(_req(workspace="/ws"))
        assert cmd == ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--json",
                       "--sandbox", "workspace-write", "--cd", "/ws", "-"]
        assert "--full-auto" not in cmd

    def test_reviewer_command(self):
        cmd = CodexCliAdapter().build_command(
            _req(workspace="/ws", model="gpt-5.5", reasoning_effort="low", allow_tools=False))
        assert cmd == ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--json",
                       "--sandbox", "read-only", "-m", "gpt-5.5",
                       "-c", "model_reasoning_effort=low",
                       "-c", 'web_search="disabled"',
                       "--disable", "shell_tool", "--disable", "unified_exec",
                       "--disable", "apps", "--disable", "plugins",
                       "--disable", "multi_agent", "--disable", "browser_use",
                       "--disable", "computer_use", "--disable", "image_generation",
                       "--disable", "in_app_browser",
                       "--ignore-user-config", "--cd", "/ws", "-"]

    def test_no_workspace_no_cd(self):
        cmd = CodexCliAdapter().build_command(_req())
        assert "--cd" not in cmd and cmd[-1] == "-"

    def test_adapter_defaults(self):
        a = CodexCliAdapter(model="gpt-5.6-sol", reasoning_effort="medium", allow_tools=False)
        cmd = a.build_command(_req())
        assert cmd[cmd.index("-m") + 1] == "gpt-5.6-sol"
        assert "model_reasoning_effort=medium" in cmd
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"

    def test_prompt_concatenation(self):
        p = CodexCliAdapter().build_prompt(_req(system="S", developer="D", user="U"))
        assert p == "S\n\nD\n\nU"


# ══ Codex: JSONL parsing ═════════════════════════════════════════════


class TestCodexParsing:
    def test_real_success_sample(self):
        p = parse_codex_jsonl(_fixture("codex_sample.jsonl"))
        assert p["text"] == "OK"
        assert p["usage"]["input_tokens"] == 15350
        assert p["usage"]["cached_input_tokens"] == 1408
        assert p["usage"]["output_tokens"] == 5
        assert p["turns_completed"] == 1 and p["turn_failed"] is None
        assert p["thread_id"]

    def test_scout_sample(self):
        p = parse_codex_jsonl(_fixture("codex_sample_scout_gpt-5.6-sol.jsonl"))
        assert p["text"] == "OK"
        assert p["usage"]["input_tokens"] == 15127 and p["usage"]["cached_input_tokens"] == 10752

    def test_real_failure_sample(self):
        p = parse_codex_jsonl(_fixture("codex_failure_sample.jsonl"))
        assert p["text"] == ""
        assert "requires a newer version of Codex" in p["turn_failed"]
        assert p["turn_failed"].startswith("status 400:")
        assert any("Model metadata" in w for w in p["warnings"])
        assert p["turns_completed"] == 0

    def test_last_agent_message_wins_and_usage_sums(self):
        lines = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Looking..."}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "FINAL"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
            {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}},
        ]
        stdout = "noise line\n" + "\n".join(json.dumps(x) for x in lines) + "\n{broken json\n"
        p = parse_codex_jsonl(stdout)
        assert p["text"] == "FINAL"
        assert p["agent_messages"] == ["Looking...", "FINAL"]
        assert p["usage"] == {"input_tokens": 15, "output_tokens": 3}


# ══ Codex: run() against a fake binary ═══════════════════════════════


class TestCodexRun:
    @pytest.mark.asyncio
    async def test_success(self, fake, tmp_path):
        cli = fake("codex").set(stdout=_fixture("codex_sample.jsonl"),
                                stderr="ERROR codex_models_manager::cache: noise\n",
                                version="codex-cli 0.145.0\n")
        r = await CodexCliAdapter(binary=str(cli.path), codex_home=str(tmp_path / "nohome")).run(
            _req(model="gpt-5.5", reasoning_effort="low", allow_tools=False, system="SYS", user="USR"))
        assert r.success is True and r.output == "OK"
        assert (r.input_tokens, r.cached_input_tokens, r.output_tokens) == (15350, 1408, 5)
        assert r.token_usage == 15355
        assert r.cost_estimate == 0.0
        assert r.model == "gpt-5.5" and r.reasoning_effort == "low"
        assert r.cli_version == "codex-cli 0.145.0"
        assert r.provider == "codex_cli"
        assert r.raw_usage["reasoning_output_tokens"] == 0
        assert r.structured_output["adapter_meta"]["sandbox"] == "read-only"
        assert r.structured_output["adapter_meta"]["model_source"] == "requested"
        assert cli.stdin == "SYS\n\nUSR"
        assert cli.argv[cli.argv.index("--sandbox") + 1] == "read-only"

    @pytest.mark.asyncio
    async def test_failure_sample(self, fake, tmp_path):
        cli = fake("codex").set(stdout=_fixture("codex_failure_sample.jsonl"), exit_code=1)
        r = await CodexCliAdapter(binary=str(cli.path), codex_home=str(tmp_path)).run(_req())
        assert r.success is False and r.output == ""
        assert r.exit_code == 1
        assert "requires a newer version of Codex" in r.error
        assert r.error.startswith("[unknown]")

    @pytest.mark.asyncio
    async def test_turn_failed_even_with_zero_exit(self, fake, tmp_path):
        stdout = json.dumps({"type": "turn.failed", "error": {"message": "stream disconnected: timeout"}})
        cli = fake("codex").set(stdout=stdout, exit_code=0)
        r = await CodexCliAdapter(binary=str(cli.path), codex_home=str(tmp_path)).run(_req())
        assert r.success is False
        assert r.error.startswith("[timeout]")

    @pytest.mark.asyncio
    async def test_no_agent_message_is_failure(self, fake, tmp_path):
        stdout = "\n".join(json.dumps(x) for x in [
            {"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 0}}])
        cli = fake("codex").set(stdout=stdout)
        r = await CodexCliAdapter(binary=str(cli.path), codex_home=str(tmp_path)).run(_req())
        assert r.success is False and r.error.startswith("[empty_output]")
        assert r.input_tokens == 7

    @pytest.mark.asyncio
    async def test_nonzero_exit_without_json_uses_stderr(self, fake, tmp_path):
        cli = fake("codex").set(stdout="", stderr="error: unexpected argument '--foo'", exit_code=2)
        r = await CodexCliAdapter(binary=str(cli.path), codex_home=str(tmp_path)).run(_req())
        assert r.success is False and "unexpected argument" in r.error

    @pytest.mark.asyncio
    async def test_model_from_config_only_when_not_requested(self, fake, tmp_path):
        home = tmp_path / "codex_home"
        home.mkdir()
        (home / "config.toml").write_text(
            'model = "gpt-cfg"\nmodel_reasoning_effort = "high"\n[projects."/x"]\ntrust_level = "trusted"\n')
        cli = fake("codex").set(stdout=_fixture("codex_sample.jsonl"))
        adapter = CodexCliAdapter(binary=str(cli.path), codex_home=str(home))
        r = await adapter.run(_req())
        assert r.model == "gpt-cfg" and r.reasoning_effort == "high"
        meta = r.structured_output["adapter_meta"]
        assert meta["model_source"] == "codex_config"
        assert meta["reasoning_effort_source"] == "codex_config"
        assert "-m" not in cli.argv  # config default is recorded, not forced
        r = await adapter.run(_req(model="gpt-5.5"))
        assert r.model == "gpt-5.5"
        assert r.structured_output["adapter_meta"]["model_source"] == "requested"

    def test_config_profile(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            'model = "base"\nprofile = "fast"\n[profiles.fast]\nmodel = "gpt-fast"\n')
        assert CodexCliAdapter(codex_home=str(tmp_path)).read_config_defaults()["model"] == "gpt-fast"

    def test_missing_config(self, tmp_path):
        assert CodexCliAdapter(codex_home=str(tmp_path / "none")).read_config_defaults() == {}


# ══ Gemini: command, prompt escaping ═════════════════════════════════


class TestGeminiCommand:
    def test_tools_allowed_command(self):
        cmd = GeminiCliAdapter().build_command(_req(model="gemini-2.5-pro"))
        assert cmd == ["gemini", "--yolo", "-m", "gemini-2.5-pro", "-o", "json", "-p", " "]

    def test_tools_disabled_command(self):
        cmd = GeminiCliAdapter().build_command(_req(allow_tools=False))
        assert "--yolo" not in cmd
        assert cmd[cmd.index("--approval-mode") + 1] == "default"
        assert cmd[-2:] == ["-p", " "]
        assert "-m" not in cmd

    def test_prompt_never_in_argv(self):
        cmd = GeminiCliAdapter().build_command(_req(user="UNIQUE_TEXT " * 1000))
        assert not any("UNIQUE_TEXT" in a for a in cmd)
        assert max(len(a) for a in cmd) < 100

    def test_at_tokens_escaped(self):
        p = GeminiCliAdapter().build_prompt(_req(user="Precision@k and @src/main.py and a@b.c"))
        assert "Precision\\@k" in p and "\\@src/main.py" in p and "a\\@b.c" in p
        # no unescaped '@' remains
        assert all(p[i - 1] == "\\" for i, ch in enumerate(p) if ch == "@")

    def test_already_escaped_stays_escaped(self):
        assert neutralize_at_expansion("x\\@y") == "x\\\\@y"

    def test_fullwidth_and_none_modes(self):
        assert neutralize_at_expansion("P@k", "fullwidth") == "P\uff20k"
        assert neutralize_at_expansion("P@k", "none") == "P@k"
        p = GeminiCliAdapter(at_escape="none").build_prompt(_req(system="", user="P@k"))
        assert p == "P@k"

    def test_invalid_escape_mode(self):
        with pytest.raises(ValueError):
            GeminiCliAdapter(at_escape="bogus")
        with pytest.raises(ValueError):
            neutralize_at_expansion("x", "bogus")

    def test_leading_slash_not_a_slash_command(self):
        p = GeminiCliAdapter().build_prompt(_req(system="", user="/memory show"))
        assert p == " /memory show"
        p = GeminiCliAdapter().build_prompt(_req(system="", user="// comment"))
        assert p == "// comment"


# ══ Gemini: JSON parsing ═════════════════════════════════════════════


class TestGeminiParsing:
    def test_stats_aggregation(self):
        data = json.loads(_fixture("gemini_success_synthetic.json"))
        u = parse_gemini_stats(data["stats"], requested_model="gemini-2.5-pro")
        assert u["input_tokens"] == 900 + 24939
        assert u["cached_input_tokens"] == 22000
        assert u["output_tokens"] == 3 + 20 + 154
        assert u["model"] == "gemini-2.5-pro" and u["model_source"] == "stats"
        assert set(u["raw_usage"]["models"]) == {"gemini-2.5-flash-lite", "gemini-2.5-pro"}

    def test_stats_missing(self):
        u = parse_gemini_stats(None, requested_model="gemini-x")
        assert (u["input_tokens"], u["output_tokens"], u["model"]) == (0, 0, "gemini-x")

    def test_input_only_stats(self):
        u = parse_gemini_stats({"models": {"m": {"tokens": {"input": 5, "cached": 3, "candidates": 1}}}})
        assert u["input_tokens"] == 8 and u["cached_input_tokens"] == 3

    def test_payload_found_after_noise(self):
        raw = "Loaded cached credentials.\n" + _fixture("gemini_success_synthetic.json")
        payload = _find_gemini_payload(raw)
        assert payload["response"].startswith("OK")

    def test_no_payload_in_plain_text(self):
        assert _find_gemini_payload("just text") is None


# ══ Gemini: run() against a fake binary ══════════════════════════════


class TestGeminiRun:
    @pytest.mark.asyncio
    async def test_success(self, fake, tmp_path):
        cli = fake("gemini").set(stdout=_fixture("gemini_success_synthetic.json"), version="0.26.0\n")
        ws = tmp_path / "ws"
        ws.mkdir()
        r = await GeminiCliAdapter(binary=str(cli.path)).run(
            _req(workspace=str(ws), model="gemini-2.5-pro", reasoning_effort="high", allow_tools=False))
        assert r.success is True
        assert r.output == "OK. Precision@k was not needed."
        assert r.model == "gemini-2.5-pro"
        assert (r.input_tokens, r.output_tokens, r.cached_input_tokens) == (25839, 177, 22000)
        assert r.token_usage == 25839 + 177
        assert r.cli_version == "0.26.0"
        # no effort flag in gemini-cli 0.26: recorded as not applied
        assert r.reasoning_effort == "unsupported"
        assert r.structured_output["adapter_meta"]["reasoning_effort_unsupported"] == "high"
        # prompt went over stdin, escaped; argv carries only the minimal -p
        assert "Precision\\@k" in cli.stdin
        assert cli.argv[-2:] == ["-p", " "]
        assert "--yolo" not in cli.argv
        assert os.path.realpath(cli.cwd) == os.path.realpath(ws)

    @pytest.mark.asyncio
    async def test_json_error_on_stderr(self, fake):
        cli = fake("gemini").set(stdout="", stderr=_fixture("gemini_error_synthetic.json"), exit_code=41)
        r = await GeminiCliAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False and r.output == ""
        assert r.exit_code == 41
        assert r.error.startswith("[auth]") and "UNAUTHENTICATED" in r.error

    @pytest.mark.asyncio
    async def test_real_auth_failure_stderr(self, fake):
        cli = fake("gemini").set(stdout="", stderr=_fixture("gemini_auth_error_stderr.txt"), exit_code=1)
        r = await GeminiCliAdapter(binary=str(cli.path)).run(_req(model="gemini-2.5-flash"))
        assert r.success is False
        assert r.error.startswith("[auth]")
        assert "GOOGLE_CLOUD_PROJECT" in r.error
        assert "    at " not in r.error  # stack frames stripped from the summary
        assert r.model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_empty_response_is_failure(self, fake):
        cli = fake("gemini").set(stdout=json.dumps({"response": "", "stats": {"models": {}}}))
        r = await GeminiCliAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False and r.error.startswith("[empty_output]")

    @pytest.mark.asyncio
    async def test_plain_text_fallback(self, fake):
        cli = fake("gemini").set(stdout="plain answer\n")
        r = await GeminiCliAdapter(binary=str(cli.path)).run(_req())
        assert r.success is True and r.output == "plain answer"

    @pytest.mark.asyncio
    async def test_error_field_with_zero_exit(self, fake):
        cli = fake("gemini").set(stdout=json.dumps({"error": {"type": "X", "message": "quota exhausted 429"}}))
        r = await GeminiCliAdapter(binary=str(cli.path)).run(_req())
        assert r.success is False and r.error.startswith("[rate_limit]")

    @pytest.mark.asyncio
    async def test_large_prompt_over_stdin(self, fake):
        cli = fake("gemini").set(stdout=json.dumps({"response": "ok"}))
        big = "x" * 300_000
        r = await GeminiCliAdapter(binary=str(cli.path)).run(_req(user=big))
        assert r.success is True
        assert big in cli.stdin


# ══ CLI version ══════════════════════════════════════════════════════


class TestCliVersion:
    @pytest.mark.asyncio
    async def test_version_cached(self, fake):
        cli = fake("codex").set(version="codex-cli 0.145.0\nextra line\n")
        adapter = CodexCliAdapter(binary=str(cli.path))
        assert await adapter.cli_version() == "codex-cli 0.145.0"
        (cli.dir / "version.txt").write_text("codex-cli 9.9.9\n")
        assert await adapter.cli_version() == "codex-cli 0.145.0"  # cached
        clear_cli_version_cache()
        assert await adapter.cli_version() == "codex-cli 9.9.9"

    @pytest.mark.asyncio
    async def test_failed_version_not_cached(self, fake):
        cli = fake("gemini").set(version="broken\n")
        (cli.dir / "version_exit.txt").write_text("1")
        assert await get_cli_version(str(cli.path)) == ""
        assert str(cli.path) not in cli_common._VERSION_CACHE
        (cli.dir / "version_exit.txt").write_text("0")
        assert await get_cli_version(str(cli.path)) == "broken"

    @pytest.mark.asyncio
    async def test_missing_binary_version(self):
        assert await get_cli_version("nonexistent_binary_xyz_e4") == ""
        assert await ClaudeCodeAdapter(binary="nonexistent_binary_xyz_e4").cli_version() == ""

    @pytest.mark.asyncio
    async def test_smoke_test_reports_version(self, fake):
        cli = fake("claude").set(version="2.1.212 (Claude Code)\n")
        info = await ClaudeCodeAdapter(binary=str(cli.path)).smoke_test()
        assert info["status"] == "ok" and info["version"] == "2.1.212 (Claude Code)"
        assert info["exit_code"] == 0
