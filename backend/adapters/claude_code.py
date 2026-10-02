"""Claude Code CLI adapter. Shells out to the `claude` binary.

Command (claude 2.1.x)::

    claude -p --output-format json --max-turns <1 | N>
           [--model <model>] [--effort <low|medium|high|xhigh|max>]
           [--system-prompt <SP>]
           [--allowedTools Read,Write,Edit,Bash,Glob,Grep]   # tools on + workspace set
           [--tools "" --strict-mcp-config]                  # tools off (reviewers)

``--max-turns 1`` is used only with tools off (a single answer). With tools on
every tool call needs another agentic turn, so the limit is ``max_turns``
(constructor, default 30; per request ``adapter_options["claude_max_turns"]``);
a run that hits it returns subtype ``error_max_turns``, reported as a
``[max_turns]`` error (not retried).

Reasoning effort is normalised like the CLI does (``med`` -> ``medium``,
``ultracode`` -> ``xhigh``) and must be one of low/medium/high/xhigh/max; any
other value fails the call with a ``[config]`` error before the CLI starts
(the CLI would otherwise ignore it with a warning and use its default). The
effort the CLI actually applied (``default`` after an "Unknown --effort
value" warning, or the lowered value after an organisation-limit warning on
stderr) is recorded in ``reasoning_effort``; the requested one in
``adapter_meta.requested_reasoning_effort``.

The prompt (files_to_read, write instructions, developer and user prompt) is
sent on stdin. The single JSON result object is parsed for ``result``,
``is_error``/``subtype``/``errors``/``api_error_status``, ``usage``
(input/output/cache tokens), ``total_cost_usd`` and ``modelUsage`` (its keys
name the model(s) that answered).
"""
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from backend.adapters.base import BaseAdapter
from backend.adapters.cli_common import (
    as_float,
    as_int,
    build_raw_log,
    categorize_error,  # re-exported: tests and callers import it from here
    detail_with_evidence,
    elide_command,
    format_error,
    get_cli_version,
    run_cli_process,
    smoke_test_cli,
    timeout_result,
)
from backend.models import AdapterRunRequest, AdapterRunResult

__all__ = ["ClaudeCodeAdapter", "categorize_error", "normalize_claude_effort"]

#: Effort levels accepted by claude 2.1.x ``--effort`` (plus CLI aliases).
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {"med": "medium", "ultracode": "xhigh"}
DEFAULT_TOOL_MAX_TURNS = 30

_UNKNOWN_EFFORT_RE = re.compile(r"Unknown --effort value", re.IGNORECASE)
_EFFORT_LOWERED_RE = re.compile(r"exceeds your organization's limit.*?using '([a-z]+)'",
                                re.IGNORECASE | re.S)


def normalize_claude_effort(effort: str) -> Optional[str]:
    """CLI-normalised effort, or None if claude would not accept it."""
    e = (effort or "").strip().lower()
    e = _EFFORT_ALIASES.get(e, e)
    return e if e in CLAUDE_EFFORTS else None


class ClaudeCodeAdapter(BaseAdapter):
    name: str = "claude_code"
    WORKSPACE_TOOLS = "Read,Write,Edit,Bash,Glob,Grep"

    def __init__(
        self,
        binary: str = "claude",
        model: str = "",
        reasoning_effort: str = "",
        allow_tools: bool = True,
        max_turns: int = DEFAULT_TOOL_MAX_TURNS,
    ):
        self.binary = binary
        # Adapter-level defaults, used when the request leaves them empty.
        self.default_model = model
        self.default_reasoning_effort = reasoning_effort
        # Adapter-level tool switch: tools are enabled only if both the adapter
        # and the request allow them.
        self.allow_tools = allow_tools
        # Turn limit when tools are on (each tool call costs a turn).
        self.max_turns = max(1, int(max_turns))

    # ── Settings resolution ─────────────────────────────────────────

    def resolve_model(self, request: AdapterRunRequest) -> str:
        return (getattr(request, "model", "") or self.default_model or "").strip()

    def resolve_reasoning_effort(self, request: AdapterRunRequest) -> str:
        return (getattr(request, "reasoning_effort", "") or self.default_reasoning_effort or "").strip()

    def tools_enabled(self, request: AdapterRunRequest) -> bool:
        return bool(self.allow_tools and getattr(request, "allow_tools", True))

    def turn_limit(self, request: AdapterRunRequest) -> int:
        """``--max-turns``: 1 with tools off, else ``max_turns`` (or the
        request's ``adapter_options["claude_max_turns"]``)."""
        if not self.tools_enabled(request):
            return 1
        opts = getattr(request, "adapter_options", None) or {}
        try:
            return max(1, int(opts.get("claude_max_turns", self.max_turns)))
        except (TypeError, ValueError):
            return self.max_turns

    # ── Availability / version ──────────────────────────────────────

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def cli_version(self) -> str:
        return await get_cli_version(self.binary)

    async def smoke_test(self) -> dict:
        if not self.is_available():
            return {"status": "unavailable", "adapter": "claude_code",
                    "error": f"'{self.binary}' not found on PATH"}
        return await smoke_test_cli(self.binary, "claude_code")

    # ── Command / prompt construction ───────────────────────────────

    def build_command(
        self,
        request: AdapterRunRequest,
    ) -> list[str]:
        """Build the CLI command for claude invocation."""
        cmd = [
            self.binary,
            "-p",  # pipe/print mode
            "--output-format", "json",
            "--max-turns", str(self.turn_limit(request)),
        ]

        model = self.resolve_model(request)
        if model:
            cmd.extend(["--model", model])

        effort = self.resolve_reasoning_effort(request)
        if effort:
            cmd.extend(["--effort", normalize_claude_effort(effort) or effort])

        # System prompt via flag
        if request.prompt_bundle.system_prompt:
            cmd.extend(["--system-prompt", request.prompt_bundle.system_prompt])

        if self.tools_enabled(request):
            # Allowed tools for workspace-aware execution
            if request.workspace_context.workspace_path:
                cmd.extend(["--allowedTools", self.WORKSPACE_TOOLS])
        else:
            # `--tools ""` disables every built-in tool (claude --help);
            # `--strict-mcp-config` without `--mcp-config` loads no MCP servers.
            cmd.extend(["--tools", "", "--strict-mcp-config"])

        return cmd

    def build_prompt(self, request: AdapterRunRequest) -> str:
        """Build the full prompt text, prepending file contents if specified."""
        prompt_parts: list[str] = []

        # Prepend file contents with clear markers
        if request.files_to_read:
            for file_path in request.files_to_read:
                try:
                    content = Path(file_path).read_text(errors="replace")
                    prompt_parts.append(
                        f"=== File: {file_path} ===\n{content}\n=== End File ==="
                    )
                except Exception:
                    prompt_parts.append(f"=== File: {file_path} === [could not read]")

        # Add write instructions if files_to_write specified
        if request.files_to_write:
            write_instruction = (
                "You must write output to the following files:\n"
                + "\n".join(f"- {p}" for p in request.files_to_write)
            )
            prompt_parts.append(write_instruction)

        # Developer prompt goes before user prompt
        if request.prompt_bundle.developer_prompt:
            prompt_parts.append(request.prompt_bundle.developer_prompt)

        # User prompt is always included
        prompt_parts.append(request.prompt_bundle.user_prompt)

        return "\n\n".join(prompt_parts)

    # ── Output parsing ──────────────────────────────────────────────

    @staticmethod
    def _load_json_dict(raw_output: str) -> Optional[dict]:
        """Parse the JSON result object; tolerate stray non-JSON lines."""
        try:
            data = json.loads(raw_output)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass
        for line in reversed((raw_output or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    @staticmethod
    def extract_result(data: dict) -> str:
        """Answer text from a parsed result object ("" if none)."""
        text: Any = data.get("result", data.get("content", data.get("text", "")))
        if not isinstance(text, str):
            text = json.dumps(text) if text else ""
        if not text and "messages" in data:
            # Extract text from messages array
            for msg in data.get("messages", []):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    text += content if isinstance(content, str) else ""
        return text

    def parse_json_output(self, raw_output: str) -> tuple[str, dict]:
        """Try to parse JSON output from claude CLI.

        Returns (text_output, structured_data).
        Falls back to raw text if JSON parsing fails.
        """
        data = self._load_json_dict(raw_output)
        if data is None:
            return raw_output, {}
        text = self.extract_result(data)
        return text or raw_output, data

    @staticmethod
    def parse_usage(data: dict, requested_model: str = "") -> dict:
        """Token/cost/model accounting from a claude result object.

        input_tokens = input + cache_read + cache_creation (all prompt tokens);
        cached_input_tokens = cache_read_input_tokens. The answering model is
        the ``modelUsage`` key with the most output tokens (claude may also
        call a small auxiliary model); falls back to the requested model.
        """
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        model_usage = data.get("modelUsage") if isinstance(data.get("modelUsage"), dict) else {}

        if usage:
            fresh = as_int(usage.get("input_tokens"))
            cache_read = as_int(usage.get("cache_read_input_tokens"))
            cache_create = as_int(usage.get("cache_creation_input_tokens"))
            output = as_int(usage.get("output_tokens"))
        else:
            fresh = sum(as_int(m.get("inputTokens")) for m in model_usage.values() if isinstance(m, dict))
            cache_read = sum(as_int(m.get("cacheReadInputTokens")) for m in model_usage.values() if isinstance(m, dict))
            cache_create = sum(as_int(m.get("cacheCreationInputTokens")) for m in model_usage.values() if isinstance(m, dict))
            output = sum(as_int(m.get("outputTokens")) for m in model_usage.values() if isinstance(m, dict))

        cost = as_float(data.get("total_cost_usd", data.get("cost_usd")))
        if not cost and model_usage:
            cost = sum(as_float(m.get("costUSD")) for m in model_usage.values() if isinstance(m, dict))

        model = ""
        model_source = "requested" if requested_model else "unknown"
        candidates = [(k, v) for k, v in model_usage.items() if isinstance(v, dict)]
        if candidates:
            candidates.sort(
                key=lambda kv: (as_int(kv[1].get("outputTokens")), as_float(kv[1].get("costUSD"))),
                reverse=True,
            )
            model = candidates[0][0]
            model_source = "modelUsage"
        elif model_usage:  # non-dict entries: the key still names the model
            model = next(iter(model_usage))
            model_source = "modelUsage"
        if not model:
            model = requested_model

        input_tokens = fresh + cache_read + cache_create
        return {
            "input_tokens": input_tokens,
            "output_tokens": output,
            "cached_input_tokens": cache_read,
            "cost": cost,
            "model": model,
            "model_source": model_source,
            "raw_usage": {
                "usage": usage,
                "modelUsage": model_usage,
                "total_cost_usd": data.get("total_cost_usd"),
            },
        }

    # ── Run ─────────────────────────────────────────────────────────

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        requested_model = self.resolve_model(request)
        requested_effort = self.resolve_reasoning_effort(request)
        effort = normalize_claude_effort(requested_effort) if requested_effort else ""
        base_fields: dict[str, Any] = dict(
            provider=self.name, model=requested_model, reasoning_effort=effort or "",
        )
        if requested_effort and effort is None:
            return AdapterRunResult(
                success=False, exit_code=-1,
                error=format_error("config", f"unsupported reasoning effort "
                                   f"{requested_effort!r} for claude (use one of "
                                   f"{', '.join(CLAUDE_EFFORTS)})"),
                structured_output={"adapter_meta": {
                    "requested_reasoning_effort": requested_effort}},
                **{**base_fields, "reasoning_effort": ""},
            )
        if not self.is_available():
            return AdapterRunResult(
                success=False, error=f"'{self.binary}' not found on PATH",
                exit_code=-1, **base_fields,
            )

        version = await self.cli_version()
        base_fields["cli_version"] = version

        full_prompt = self.build_prompt(request)
        cmd = self.build_command(request)
        meta = {
            "command": elide_command(cmd),
            "requested_model": requested_model,
            "requested_reasoning_effort": requested_effort,
            "reasoning_effort": effort,
            "tools_enabled": self.tools_enabled(request),
            "max_turns": self.turn_limit(request),
        }

        # Set working directory to workspace path if provided
        cwd = request.workspace_context.workspace_path or None
        proc = await run_cli_process(cmd, full_prompt, cwd=cwd, timeout=request.timeout_seconds)

        if proc.spawn_error:
            return AdapterRunResult(
                success=False, exit_code=-1, duration_seconds=proc.duration,
                error=format_error("unknown", f"failed to start {self.binary}: {proc.spawn_error}"),
                structured_output={"adapter_meta": meta}, **base_fields,
            )
        if proc.timed_out:
            return timeout_result(request.timeout_seconds, proc.duration,
                                  structured_output={"adapter_meta": meta}, **base_fields)

        data = self._load_json_dict(proc.stdout)
        usage: dict = {}
        if data is not None:
            text = self.extract_result(data)
            usage = self.parse_usage(data, requested_model)
            subtype = str(data.get("subtype", "") or "")
            is_error = bool(data.get("is_error")) or subtype.startswith("error")
            meta["model_source"] = usage["model_source"]
            structured = dict(data)
        else:
            # Non-JSON stdout (e.g. an old CLI without --output-format json).
            text = proc.stdout.strip()
            subtype = ""
            is_error = False
            meta["model_source"] = "requested" if requested_model else "unknown"
            structured = {}
        structured["adapter_meta"] = meta

        # Effort the CLI actually applied (it ignores unknown values and may
        # lower values above an organisation limit, warning on stderr).
        applied_effort = effort or ""
        if effort:
            lowered = _EFFORT_LOWERED_RE.search(proc.stderr or "")
            if _UNKNOWN_EFFORT_RE.search(proc.stderr or ""):
                applied_effort = "default"
            elif lowered:
                applied_effort = lowered.group(1).lower()
            if applied_effort != effort:
                meta["effort_not_applied"] = True
        base_fields["reasoning_effort"] = applied_effort

        error_msg: Optional[str] = None
        cli_errors = data.get("errors") if data else None
        cli_errors_text = "; ".join(str(e) for e in cli_errors) if isinstance(cli_errors, list) else ""
        api_status = as_int(data.get("api_error_status")) if data else 0
        status_category = ("rate_limit" if api_status == 429 else
                           "overloaded" if api_status == 529 or 500 <= api_status < 600 else "")
        if subtype == "error_max_turns":
            detail = (cli_errors_text or text
                      or f"claude stopped at the turn limit ({meta['max_turns']})")
            error_msg = format_error("max_turns", f"{detail} (subtype=error_max_turns)")
        elif proc.returncode != 0:
            evidence = proc.stderr + "\n" + (text if is_error else "") + "\n" + cli_errors_text
            category = status_category or categorize_error(proc.returncode, evidence)
            detail = ((text if is_error else "") or cli_errors_text
                      or detail_with_evidence(proc.stderr, category)
                      or f"exit code {proc.returncode}")
            error_msg = format_error(category, detail)
        elif is_error:
            detail = (text or cli_errors_text
                      or f"claude reported an error (subtype={subtype or 'unknown'})")
            category = status_category or categorize_error(
                proc.returncode, text + "\n" + cli_errors_text + "\n" + proc.stderr)
            if category != "unknown" and category not in detail.lower():
                evidence_line = detail_with_evidence(proc.stderr, category)
                if evidence_line and evidence_line not in detail:
                    detail = f"{detail} | {evidence_line}"
            error_msg = format_error(category, detail)
        elif not text.strip():
            stop = data.get("stop_reason") if data else None
            error_msg = format_error(
                "empty_output",
                f"claude exited 0 but returned no result text (subtype={subtype or 'n/a'}, stop_reason={stop})",
            )

        success = error_msg is None
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return AdapterRunResult(
            success=success,
            output=text if success else "",
            exit_code=proc.returncode or 0,
            duration_seconds=proc.duration,
            error=error_msg,
            raw_log=build_raw_log(proc.stderr, proc.stdout, include_stdout=not success),
            structured_output=structured,
            token_usage=input_tokens + output_tokens,
            cost_estimate=usage.get("cost", 0.0),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=usage.get("cached_input_tokens", 0),
            raw_usage=usage.get("raw_usage", {}),
            **{**base_fields, "model": usage.get("model") or requested_model},
        )
