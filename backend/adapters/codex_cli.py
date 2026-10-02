"""Codex CLI adapter. Shells out to the `codex` binary.

Command (codex-cli 0.145.x; ``--full-auto`` is deprecated there)::

    codex exec --skip-git-repo-check --ephemeral --json
               --sandbox <read-only|workspace-write>
               [-m <model>] [-c model_reasoning_effort=<effort>]
               [<tools-off arguments>] [--ignore-user-config]
               [--cd <workspace>] -

The prompt (system, developer and user prompt concatenated) is sent on stdin.
The sandbox is ``read-only`` when tools are disabled (reviewer calls, and
executor calls when verified code execution is on) and ``workspace-write``
otherwise. ``--sandbox`` only governs model-generated shell commands, so with
tools disabled the adapter also passes (``TOOLS_OFF_ARGS``)::

    -c web_search="disabled"
    --disable shell_tool --disable unified_exec      # no shell at all
    --disable apps --disable plugins --disable multi_agent
    --disable browser_use --disable computer_use --disable image_generation
    --disable in_app_browser

and ``--ignore-user-config`` (``$CODEX_HOME/config.toml`` is not loaded, so
the user's MCP servers, plugins, feature flags, notify hooks and approval
policy do not apply; auth still comes from ``CODEX_HOME``). Both are
configurable (constructor ``tools_off_args`` / ``ignore_user_config``; per
request ``adapter_options={"codex_ignore_user_config": bool}``); the
benchmark harness passes exactly ``-c web_search="disabled"`` and keeps the
user config, its frozen pilot condition. ``$CODEX_HOME/AGENTS.md`` is still
injected by the CLI; use ``codex_home`` (a separate, logged-in home without
AGENTS.md) to avoid it. ``codex_home`` is passed to the subprocess as
``CODEX_HOME``. Flag names were verified against codex-cli 0.145.0
(``codex features list`` with ``--disable``, and a full argv parse).

``--json`` prints a JSONL event stream:

    {"type":"thread.started","thread_id":...}
    {"type":"turn.started"}
    {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
    {"type":"turn.completed","usage":{"input_tokens":..,"cached_input_tokens":..,
                                      "output_tokens":..,"reasoning_output_tokens":..}}

and on failure ``{"type":"error","message":...}`` / ``{"type":"turn.failed",
"error":{"message":...}}``. Tool items (``command_execution``, ``web_search``,
``mcp_tool_call``, ...) are counted into ``structured_output["tool_counts"]``.
The events do not name the model, so the result records the requested model,
or (only when none was requested and the user config is loaded) the ``model``
key of ``$CODEX_HOME/config.toml``; ``adapter_meta.model_source`` says which
(``cli_default`` when the config is ignored and no model was requested).
"""
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

try:  # Python >= 3.11
    import tomllib as _toml
except ImportError:  # pragma: no cover - optional
    _toml = None

from backend.adapters.base import BaseAdapter
from backend.adapters.cli_common import (
    as_int,
    build_raw_log,
    categorize_error,
    detail_with_evidence,
    elide_command,
    format_error,
    get_cli_version,
    run_cli_process,
    smoke_test_cli,
    timeout_result,
)
from backend.models import AdapterRunRequest, AdapterRunResult

#: Features disabled when tools are off (each adds a tool to the model).
TOOLS_OFF_FEATURES: tuple[str, ...] = (
    "shell_tool", "unified_exec", "apps", "plugins", "multi_agent",
    "browser_use", "computer_use", "image_generation", "in_app_browser",
)
#: Arguments added for allow_tools=False (before the trailing "-").
TOOLS_OFF_ARGS: tuple[str, ...] = (
    "-c", 'web_search="disabled"',
    *[a for f in TOOLS_OFF_FEATURES for a in ("--disable", f)],
)
#: The benchmark's frozen tools-off condition (web search off only).
WEB_SEARCH_OFF_ARGS: tuple[str, ...] = ("-c", 'web_search="disabled"')

_NON_TOOL_ITEMS = {"agent_message", "assistant_message", "reasoning", "error"}


def _unwrap_error_message(message: Any) -> str:
    """Codex embeds the provider's JSON error as a string; surface its text."""
    if not isinstance(message, str):
        return json.dumps(message) if message else ""
    text = message.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict):
            err = payload.get("error")
            inner = err.get("message") if isinstance(err, dict) else err
            status = payload.get("status")
            if inner:
                return f"status {status}: {inner}" if status else str(inner)
    return text


def parse_codex_jsonl(stdout: str) -> dict:
    """Parse ``codex exec --json`` output into text, usage and errors.

    Non-JSON lines are ignored. The answer is the last ``agent_message``
    item; usage is summed over ``turn.completed`` events.
    """
    events: list[dict] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)

    agent_messages: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    usage_totals: dict[str, int] = {}
    tool_counts = {"tool_calls": 0, "command_executions": 0, "web_searches": 0,
                   "mcp_tool_calls": 0, "item_types": {}}
    turns_completed = 0
    turn_failed: Optional[str] = None
    thread_id = ""

    for ev in events:
        etype = ev.get("type", "")
        if etype == "thread.started":
            thread_id = str(ev.get("thread_id", "") or "")
        elif etype == "item.completed":
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            itype = item.get("type") or item.get("item_type")
            if itype in ("agent_message", "assistant_message"):
                text = item.get("text")
                if isinstance(text, str):
                    agent_messages.append(text)
            elif itype == "error":
                # Non-fatal notices (e.g. missing model metadata).
                warnings.append(str(item.get("message", "")))
            if itype and itype not in _NON_TOOL_ITEMS:
                tool_counts["tool_calls"] += 1
                tool_counts["item_types"][itype] = tool_counts["item_types"].get(itype, 0) + 1
                if itype == "command_execution":
                    tool_counts["command_executions"] += 1
                elif itype == "web_search":
                    tool_counts["web_searches"] += 1
                elif itype == "mcp_tool_call":
                    tool_counts["mcp_tool_calls"] += 1
        elif etype == "turn.completed":
            turns_completed += 1
            usage = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage_totals[key] = usage_totals.get(key, 0) + int(value)
        elif etype == "turn.failed":
            err = ev.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            turn_failed = _unwrap_error_message(msg) or "turn failed"
        elif etype == "error":
            errors.append(_unwrap_error_message(ev.get("message", "")))

    return {
        "text": agent_messages[-1] if agent_messages else "",
        "agent_messages": agent_messages,
        "usage": usage_totals,
        "turns_completed": turns_completed,
        "turn_failed": turn_failed,
        "errors": [e for e in errors if e],
        "warnings": [w for w in warnings if w],
        "thread_id": thread_id,
        "event_types": [ev.get("type", "") for ev in events],
        "tool_counts": tool_counts,
    }


class CodexCliAdapter(BaseAdapter):
    name: str = "codex_cli"

    def __init__(
        self,
        binary: str = "codex",
        model: str = "",
        reasoning_effort: str = "",
        allow_tools: bool = True,
        codex_home: Optional[str] = None,
        tools_off_args: Optional[Sequence[str]] = None,
        ignore_user_config: bool = True,
    ):
        self.binary = binary
        self.default_model = model
        self.default_reasoning_effort = reasoning_effort
        self.allow_tools = allow_tools
        self.codex_home = codex_home
        # Arguments added when tools are off (default TOOLS_OFF_ARGS).
        self.tools_off_args = list(TOOLS_OFF_ARGS if tools_off_args is None else tools_off_args)
        # Skip $CODEX_HOME/config.toml when tools are off (no user MCP servers,
        # plugins, feature flags, hooks).
        self.ignore_user_config = bool(ignore_user_config)

    # ── Settings resolution ─────────────────────────────────────────

    def resolve_model(self, request: AdapterRunRequest) -> str:
        return (getattr(request, "model", "") or self.default_model or "").strip()

    def resolve_reasoning_effort(self, request: AdapterRunRequest) -> str:
        return (getattr(request, "reasoning_effort", "") or self.default_reasoning_effort or "").strip()

    def tools_enabled(self, request: AdapterRunRequest) -> bool:
        return bool(self.allow_tools and getattr(request, "allow_tools", True))

    def sandbox_mode(self, request: AdapterRunRequest) -> str:
        return "workspace-write" if self.tools_enabled(request) else "read-only"

    def user_config_ignored(self, request: AdapterRunRequest) -> bool:
        """True if ``--ignore-user-config`` is passed (tools off only)."""
        if self.tools_enabled(request):
            return False
        opts = getattr(request, "adapter_options", None) or {}
        if "codex_ignore_user_config" in opts:
            from backend.utils.config_values import parse_bool
            return parse_bool(opts["codex_ignore_user_config"], "codex_ignore_user_config")
        return self.ignore_user_config

    def home_dir(self) -> Path:
        """The CODEX_HOME the subprocess uses (``codex_home``, else the
        inherited ``$CODEX_HOME``, else ``~/.codex``)."""
        return Path(self.codex_home or os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"))

    def config_path(self) -> Path:
        return self.home_dir() / "config.toml"

    def subprocess_env(self) -> Optional[dict[str, str]]:
        """Environment for the CLI: ``CODEX_HOME`` set when ``codex_home`` is."""
        if not self.codex_home:
            return None
        return {**os.environ, "CODEX_HOME": str(self.codex_home)}

    def read_config_defaults(self) -> dict[str, str]:
        """``model`` / ``model_reasoning_effort`` from the codex config file.

        Honours a top-level ``profile = "<name>"`` selecting ``[profiles.<name>]``.
        Returns {} if the file or a TOML parser is unavailable.
        """
        if _toml is None:
            return {}
        try:
            with open(self.config_path(), "rb") as fh:
                cfg = _toml.load(fh)
        except (OSError, ValueError):
            return {}
        out: dict[str, str] = {}
        for key in ("model", "model_reasoning_effort"):
            if isinstance(cfg.get(key), str):
                out[key] = cfg[key]
        profile = cfg.get("profile")
        profiles = cfg.get("profiles") if isinstance(cfg.get("profiles"), dict) else {}
        if isinstance(profile, str) and isinstance(profiles.get(profile), dict):
            for key in ("model", "model_reasoning_effort"):
                if isinstance(profiles[profile].get(key), str):
                    out[key] = profiles[profile][key]
        return out

    # ── Availability / version ──────────────────────────────────────

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def cli_version(self) -> str:
        return await get_cli_version(self.binary)

    async def smoke_test(self) -> dict:
        if not self.is_available():
            return {"status": "unavailable", "adapter": "codex_cli",
                    "error": f"'{self.binary}' not found on PATH"}
        return await smoke_test_cli(self.binary, "codex_cli")

    # ── Command / prompt construction ───────────────────────────────

    def build_prompt(self, request: AdapterRunRequest) -> str:
        prompt_parts = []
        if request.prompt_bundle.system_prompt:
            prompt_parts.append(request.prompt_bundle.system_prompt)
        if request.prompt_bundle.developer_prompt:
            prompt_parts.append(request.prompt_bundle.developer_prompt)
        prompt_parts.append(request.prompt_bundle.user_prompt)
        return "\n\n".join(prompt_parts)

    def build_command(self, request: AdapterRunRequest) -> list[str]:
        cmd = [
            self.binary, "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--json",
            "--sandbox", self.sandbox_mode(request),
        ]
        model = self.resolve_model(request)
        if model:
            cmd.extend(["-m", model])
        effort = self.resolve_reasoning_effort(request)
        if effort:
            cmd.extend(["-c", f"model_reasoning_effort={effort}"])
        if not self.tools_enabled(request):
            cmd.extend(self.tools_off_args)
            if self.user_config_ignored(request):
                cmd.append("--ignore-user-config")
        workspace = request.workspace_context.workspace_path
        if workspace:
            cmd.extend(["--cd", workspace])
        # Codex exec reads the prompt from stdin when "-" is given
        cmd.append("-")
        return cmd

    # ── Run ─────────────────────────────────────────────────────────

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        requested_model = self.resolve_model(request)
        requested_effort = self.resolve_reasoning_effort(request)
        base_fields: dict[str, Any] = dict(
            provider=self.name, model=requested_model, reasoning_effort=requested_effort,
        )
        if not self.is_available():
            return AdapterRunResult(
                success=False, error=f"'{self.binary}' not found on PATH",
                exit_code=-1, **base_fields,
            )

        # The JSON events never name the model: record the requested one, or
        # the config default (flagged via model_source) when none was requested.
        model, model_source = requested_model, "requested"
        effort, effort_source = requested_effort, "requested"
        config_ignored = self.user_config_ignored(request)
        if not requested_model or not requested_effort:
            # With --ignore-user-config the CLI does not read config.toml, so
            # its values must not be recorded as the ones used.
            cfg = {} if config_ignored else self.read_config_defaults()
            if not requested_model:
                model = cfg.get("model", "")
                model_source = "codex_config" if model else (
                    "cli_default" if config_ignored else "unknown")
            if not requested_effort:
                effort = cfg.get("model_reasoning_effort", "")
                effort_source = "codex_config" if effort else (
                    "cli_default" if config_ignored else "unknown")
        base_fields.update(model=model, reasoning_effort=effort)
        base_fields["cli_version"] = await self.cli_version()

        cmd = self.build_command(request)
        meta = {
            "command": elide_command(cmd),
            "requested_model": requested_model,
            "model_source": model_source,
            "requested_reasoning_effort": requested_effort,
            "reasoning_effort_source": effort_source,
            "sandbox": self.sandbox_mode(request),
            "tools_enabled": self.tools_enabled(request),
            "tools_off_args": [] if self.tools_enabled(request) else list(self.tools_off_args),
            "web_search": ("disabled" if (not self.tools_enabled(request)
                                          and 'web_search="disabled"' in self.tools_off_args)
                           else "cli_default"),
            "user_config_ignored": config_ignored,
            "codex_home": str(self.home_dir()),
        }

        proc = await run_cli_process(
            cmd, self.build_prompt(request), cwd=None, timeout=request.timeout_seconds,
            env=self.subprocess_env(),
        )
        if proc.spawn_error:
            return AdapterRunResult(
                success=False, exit_code=-1, duration_seconds=proc.duration,
                error=format_error("unknown", f"failed to start {self.binary}: {proc.spawn_error}"),
                structured_output={"adapter_meta": meta}, **base_fields,
            )
        if proc.timed_out:
            return timeout_result(request.timeout_seconds, proc.duration,
                                  structured_output={"adapter_meta": meta}, **base_fields)

        parsed = parse_codex_jsonl(proc.stdout)
        text = parsed["text"]
        usage = parsed["usage"]
        input_tokens = as_int(usage.get("input_tokens"))
        output_tokens = as_int(usage.get("output_tokens"))
        cached = as_int(usage.get("cached_input_tokens"))

        error_msg: Optional[str] = None
        json_error = parsed["turn_failed"] or (parsed["errors"][-1] if parsed["errors"] else "")
        if proc.returncode != 0 or parsed["turn_failed"]:
            if json_error:
                detail = json_error
                category = categorize_error(proc.returncode, json_error)
            else:
                category = categorize_error(proc.returncode, proc.stderr)
                # The detail always contains the line the category came from.
                detail = (detail_with_evidence(proc.stderr, category)
                          or f"exit code {proc.returncode}")
            error_msg = format_error(category, detail)
        elif not text.strip():
            detail = json_error or (
                "codex exited 0 but produced no agent_message "
                f"(events: {', '.join(parsed['event_types']) or 'none'})"
            )
            error_msg = format_error("empty_output", detail)

        success = error_msg is None
        structured = {
            "adapter_meta": meta,
            "thread_id": parsed["thread_id"],
            "agent_messages": parsed["agent_messages"],
            "usage": usage,
            "turns_completed": parsed["turns_completed"],
            "errors": parsed["errors"],
            "warnings": parsed["warnings"],
            "event_types": parsed["event_types"],
            "tool_counts": parsed["tool_counts"],
        }
        raw_usage = dict(usage)
        raw_usage["turns_completed"] = parsed["turns_completed"]
        return AdapterRunResult(
            success=success,
            output=text if success else "",
            exit_code=proc.returncode or 0,
            duration_seconds=proc.duration,
            error=error_msg,
            raw_log=build_raw_log(proc.stderr, proc.stdout, include_stdout=not success),
            structured_output=structured,
            token_usage=input_tokens + output_tokens,
            cost_estimate=0.0,  # codex reports no cost
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached,
            raw_usage=raw_usage,
            **base_fields,
        )
