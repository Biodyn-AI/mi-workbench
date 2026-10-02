"""Regression tests: CLI adapters, retry classification and process handling
(code-review findings adapters-1..7, engine-3 (Codex part), and the known
small issues: Codex web search off, retry prefixes, telemetry storage)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from backend.adapters.claude_code import ClaudeCodeAdapter, normalize_claude_effort
from backend.adapters.cli_common import (
    categorize_error,
    clear_cli_version_cache,
    run_cli_process,
)
from backend.adapters.codex_cli import TOOLS_OFF_ARGS, CodexCliAdapter, parse_codex_jsonl
from backend.adapters.gemini_cli import TOOLS_OFF_EXCLUDE, GeminiCliAdapter
from backend.models import AdapterRunResult
from backend.orchestrator.retry import error_category, is_transient_error, run_with_retry
from backend.tests.test_adapter_cli_commands import FakeCli, _req


@pytest.fixture(autouse=True)
def _fresh_version_cache():
    clear_cli_version_cache()
    yield
    clear_cli_version_cache()


@pytest.fixture
def fake(tmp_path):
    return lambda name: FakeCli(tmp_path, name)


# ── adapters-1 / engine-3 / known issue (1): Codex tools off ──────────


def test_codex_tools_off_command():
    cmd = CodexCliAdapter().build_command(_req(allow_tools=False, workspace="/ws"))
    i = cmd.index('web_search="disabled"')
    assert cmd[i - 1] == "-c"
    for feat in ("shell_tool", "unified_exec", "apps", "plugins", "multi_agent"):
        assert cmd[cmd.index(feat) - 1] == "--disable"
    assert "--ignore-user-config" in cmd
    assert cmd[-1] == "-" and cmd[cmd.index("--cd") + 1] == "/ws"
    # tools on: none of it
    on = CodexCliAdapter().build_command(_req(workspace="/ws"))
    assert 'web_search="disabled"' not in on and "--ignore-user-config" not in on
    # per-request option and constructor knobs
    keep = CodexCliAdapter().build_command(
        _req(allow_tools=False, adapter_options={"codex_ignore_user_config": "false"}))
    assert "--ignore-user-config" not in keep
    web_only = CodexCliAdapter(tools_off_args=["-c", 'web_search="disabled"'],
                               ignore_user_config=False).build_command(_req(allow_tools=False))
    assert "--disable" not in web_only and web_only.count('web_search="disabled"') == 1


def test_codex_counts_tool_items_and_records_meta():
    stdout = "\n".join(json.dumps(e) for e in [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
        {"type": "item.completed", "item": {"type": "web_search", "query": "x"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "y"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ])
    counts = parse_codex_jsonl(stdout)["tool_counts"]
    assert (counts["tool_calls"], counts["command_executions"], counts["web_searches"],
            counts["mcp_tool_calls"]) == (3, 1, 1, 1)


@pytest.mark.asyncio
async def test_codex_meta_and_codex_home_env(fake, tmp_path):
    home = tmp_path / "clean_home"
    home.mkdir()
    (home / "config.toml").write_text('model = "gpt-5.5"\n')
    stdout = "\n".join(json.dumps(e) for e in [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
        {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}}])
    cli = fake("codex").set(stdout=stdout)
    r = await CodexCliAdapter(binary=str(cli.path), codex_home=str(home)).run(_req())
    assert r.success
    assert (cli.dir / "env_codex_home.txt").read_text() == str(home)  # adapters-4
    meta = r.structured_output["adapter_meta"]
    assert meta["codex_home"] == str(home) and meta["model_source"] == "codex_config"
    # tools off: config ignored, so its model is not recorded as the one used
    r2 = await CodexCliAdapter(binary=str(cli.path), codex_home=str(home)).run(
        _req(allow_tools=False))
    meta2 = r2.structured_output["adapter_meta"]
    assert meta2["user_config_ignored"] is True and meta2["web_search"] == "disabled"
    assert r2.model == "" and meta2["model_source"] == "cli_default"


# ── adapters-2: Claude turn limit with tools ──────────────────────────


@pytest.mark.asyncio
async def test_claude_max_turns_error_is_categorised(fake):
    payload = {"type": "result", "subtype": "error_max_turns", "is_error": True,
               "errors": ["Reached maximum number of turns (30)"], "usage": {}}
    cli = fake("claude").set(stdout=json.dumps(payload), exit_code=1)
    r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req(workspace=str(cli.dir)))
    assert r.success is False and r.error.startswith("[max_turns]")
    assert "Reached maximum number of turns" in r.error
    assert not is_transient_error(r)
    argv = cli.argv
    assert argv[argv.index("--max-turns") + 1] == "30"
    assert ClaudeCodeAdapter().turn_limit(_req(adapter_options={"claude_max_turns": 7})) == 7


# ── adapters-3 / known issue (2): retry reads the category prefix ─────


@pytest.mark.parametrize("error,transient", [
    ("[rate_limit] Too Many Requests", True),
    ("[network] Could not connect to api.anthropic.com", True),
    ("[timeout] read timed out", True),
    ("[overloaded] API Error: 529", True),
    ("[auth] token expired (timeout while refreshing)", False),
    ("[quota] TerminalQuotaError: daily limit", False),
    ("[max_turns] Reached maximum number of turns", False),
    ("[unknown] stream disconnected before completion", True),
    ("[unknown] You have hit your usage limit", False),
    ("[empty_output] codex exited 0", False),
])
def test_is_transient_reads_prefix(error, transient):
    assert is_transient_error(AdapterRunResult(success=False, error=error)) is transient


def test_categorize_error_new_categories():
    assert categorize_error(1, "stream disconnected before completion: error sending request") == "network"
    assert categorize_error(1, '{"type":"RetryableQuotaError","message":"Resource has been exhausted"}') == "rate_limit"
    assert categorize_error(1, "TerminalQuotaError: You have exhausted your daily quota") == "quota"
    assert categorize_error(1, "Failed to authenticate") == "auth"
    assert error_category("[rate_limit] x") == "rate_limit" and error_category("plain") == ""


@pytest.mark.asyncio
async def test_codex_dropped_stream_is_retried(fake):
    stdout = json.dumps({"type": "turn.failed", "error": {
        "message": "stream disconnected before completion: error sending request for url"}})
    cli = fake("codex").set(stdout=stdout, exit_code=1)
    adapter = CodexCliAdapter(binary=str(cli.path))
    r = await run_with_retry(adapter, _req(), max_retries=2, base_delay=0.0)
    assert r.error.startswith("[network]")
    assert len(list(cli.dir.glob("argv.json"))) == 1  # the file is rewritten per call
    calls = []
    orig = adapter.run

    async def counting(req):
        calls.append(1)
        return await orig(req)

    adapter.run = counting
    await run_with_retry(adapter, _req(), max_retries=2, base_delay=0.0)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_codex_429_early_in_long_stderr_keeps_evidence(fake):
    stderr = "HTTP 429 Too Many Requests\n" + ("x" * 3000) + "\n"
    cli = fake("codex").set(stdout="not json", stderr=stderr, exit_code=1)
    r = await CodexCliAdapter(binary=str(cli.path)).run(_req())
    assert r.error.startswith("[rate_limit]") and "429" in r.error
    assert is_transient_error(r)


@pytest.mark.asyncio
async def test_gemini_retryable_quota_is_rate_limit(fake):
    payload = {"session_id": "s", "error": {"type": "RetryableQuotaError",
                                            "message": "Resource has been exhausted (e.g. check quota).",
                                            "code": 429}}
    cli = fake("gemini").set(stdout=json.dumps(payload), exit_code=1)
    r = await GeminiCliAdapter(binary=str(cli.path)).run(_req(allow_tools=False))
    assert r.error.startswith("[rate_limit]") and is_transient_error(r)


@pytest.mark.asyncio
async def test_claude_429_only_on_stderr(fake):
    payload = {"type": "result", "subtype": "success", "is_error": True,
               "result": "API Error: Too Many Requests", "usage": {}}
    cli = fake("claude").set(stdout=json.dumps(payload), stderr="status 429\n", exit_code=1)
    r = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req(allow_tools=False))
    assert r.error.startswith("[rate_limit]") and is_transient_error(r)
    payload2 = {"type": "result", "subtype": "success", "is_error": True,
                "result": "API Error", "api_error_status": 529, "usage": {}}
    cli2 = fake("claude2").set(stdout=json.dumps(payload2), exit_code=1)
    r2 = await ClaudeCodeAdapter(binary=str(cli2.path)).run(_req(allow_tools=False))
    assert r2.error.startswith("[overloaded]")


# ── adapters-5: Gemini tools off excludes every built-in tool ─────────


@pytest.mark.asyncio
async def test_gemini_tools_off_settings_file(fake):
    cli = fake("gemini").set(stdout=json.dumps({"response": "ok", "stats": {}}))
    r = await GeminiCliAdapter(binary=str(cli.path)).run(_req(allow_tools=False))
    assert r.success
    path = (cli.dir / "env_gemini_settings.txt").read_text()
    settings = json.loads(Path(path).read_text())
    assert "google_web_search" in settings["tools"]["exclude"]
    assert set(settings["tools"]["exclude"]) == set(TOOLS_OFF_EXCLUDE)
    assert settings["mcp"]["allowed"] == []
    meta = r.structured_output["adapter_meta"]
    assert meta["tools_excluded"] == list(TOOLS_OFF_EXCLUDE) and meta["gemini_system_settings"]["sha256"]
    r2 = await GeminiCliAdapter(binary=str(cli.path)).run(_req())
    assert (cli.dir / "env_gemini_settings.txt").read_text() == "<unset>"


# ── adapters-6: recorded effort is the applied effort ─────────────────


@pytest.mark.asyncio
async def test_claude_effort_validation_and_warning(fake):
    assert normalize_claude_effort("MED") == "medium" and normalize_claude_effort("minimal") is None
    r = await ClaudeCodeAdapter(binary="claude-not-called").run(_req(reasoning_effort="minimal"))
    assert r.success is False and r.error.startswith("[config]")
    payload = {"type": "result", "subtype": "success", "result": "ok", "usage": {}}
    cli = fake("claude").set(stdout=json.dumps(payload),
                             stderr="Unknown --effort value 'high' — ignoring it and using the default effort.\n")
    r2 = await ClaudeCodeAdapter(binary=str(cli.path)).run(_req(reasoning_effort="high",
                                                                allow_tools=False))
    assert r2.success and r2.reasoning_effort == "default"
    assert r2.structured_output["adapter_meta"]["requested_reasoning_effort"] == "high"


def test_engine_describe_calls_does_not_fall_back_when_results_exist():
    from backend.models import IterationResult, ProviderName, RunState
    from backend.orchestrator.engine import LoopEngine
    from backend.adapters.mock import MockAdapter
    eng = LoopEngine(adapter=MockAdapter())
    it = IterationResult(iteration_number=1, role="reviewer")
    run = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK, model="")
    eng._describe_calls(run, it, [AdapterRunResult(success=True, reasoning_effort="unsupported")],
                        {"reasoning_effort": "high"})
    assert it.reasoning_effort == "unsupported"
    it2 = IterationResult(iteration_number=1, role="reviewer")
    eng._describe_calls(run, it2, [], {"reasoning_effort": "high"})
    assert it2.reasoning_effort == "high"


# ── adapters-7: process tree handling ─────────────────────────────────


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.asyncio
async def test_output_returned_when_descendant_holds_pipes(tmp_path):
    pidf = tmp_path / "bg.pid"
    import shlex
    script = f'sleep 23.5 & echo $! > {shlex.quote(str(pidf))}; echo \'{{"result":"ok"}}\''
    t0 = time.monotonic()
    res = await run_cli_process(["sh", "-c", script], "", timeout=15)
    assert time.monotonic() - t0 < 12
    assert res.timed_out is False and res.returncode == 0
    assert '{"result":"ok"}' in res.stdout
    pid = int(pidf.read_text())
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert not _alive(pid)


@pytest.mark.asyncio
async def test_timeout_terminates_group_and_keeps_partial_output(tmp_path):
    script = "echo partial; sleep 30"
    res = await run_cli_process(["sh", "-c", script], "", timeout=1)
    assert res.timed_out and "partial" in res.stdout


@pytest.mark.asyncio
async def test_sigterm_lets_cli_clean_up_its_own_process_group(tmp_path):
    """A CLI that puts a child in its own process group and kills it on
    SIGTERM (as codex does for its tool processes) leaves no survivor."""
    pidf = tmp_path / "child.pid"
    code = (
        "import os, signal, subprocess, sys, time\n"
        "c = subprocess.Popen(['sleep', '30'], process_group=0)\n"
        f"open({str(pidf)!r}, 'w').write(str(c.pid))\n"
        "def term(*a):\n"
        "    os.killpg(c.pid, signal.SIGKILL); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, term)\n"
        "time.sleep(60)\n"
    )
    res = await run_cli_process([sys.executable, "-c", code], "", timeout=1.5)
    assert res.timed_out
    pid = int(pidf.read_text())
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert not _alive(pid)


# ── known issue (4): telemetry storage uses the serialised open/close ──


@pytest.mark.asyncio
async def test_telemetry_storage_uses_serialised_connections(tmp_path, monkeypatch):
    from backend import database
    from backend.telemetry import storage
    from backend.telemetry.collector import TelemetryRecord

    opened = []
    real_open = database.open_connection

    async def spy(path=None):
        opened.append(path)
        return await real_open(path)

    monkeypatch.setattr(database, "open_connection", spy)
    db_path = str(tmp_path / "t.db")
    await storage.init_telemetry_table(db_path)
    await storage.save_record(TelemetryRecord(timestamp="t", run_id="r", iteration=1, role="x",
                                              adapter="mock", model="m"), db_path)
    assert opened == [db_path, db_path]
    src = Path(storage.__file__).read_text()
    assert "aiosqlite.connect(" not in src
