"""Gemini CLI adapter. Shells out to the `gemini` binary.

Command (gemini-cli 0.26.x)::

    gemini [--yolo | --approval-mode default] [-m <model>] -o json -p " "

The prompt is sent on **stdin**, never as argv (no OS argument-size limit).
In non-interactive mode the CLI reads piped stdin and appends the ``-p``
text to it (``gemini --help``: "-p Prompt. Appended to input on stdin";
``dist/src/gemini.js``: ``input = `${stdinData}\\n\\n${input}```), so ``-p``
carries only a single space.

'@' file expansion: the combined stdin+prompt input is still passed through
``handleAtCommand`` (``dist/src/nonInteractiveCli.js``), which treats every
unescaped ``@token`` as a workspace path, falls back to a recursive glob
``**/*token*`` and splices matching file contents into the prompt; it also
re-joins the text around such tokens. Reading from stdin therefore does NOT
avoid the expansion (this is how "Precision@k" once corrupted a prompt). The
parser skips an ``@`` preceded by a backslash
(``dist/src/ui/hooks/atCommandProcessor.js``: "Find next unescaped '@'"), and
when no unescaped ``@`` remains the query is forwarded unchanged. The adapter
therefore escapes every ``@`` as ``\\@`` by default (``at_escape="backslash"``;
the model sees ``\\@``). ``at_escape="fullwidth"`` substitutes U+FF20 instead,
``"none"`` disables the protection. A leading ``/`` (slash-command syntax) is
prefixed with a space for the same reason.

Tools: with tools allowed the adapter passes ``--yolo`` (previous behaviour).
With tools disabled it passes ``--approval-mode default`` (which in
non-interactive mode only excludes the shell, edit, write-file and web-fetch
tools; ``google_web_search``, ``read_file``, ``list_directory``, ``glob`` and
``search_file_content`` would stay allowed by the default policy) AND points
the child at a system-settings file (env ``GEMINI_CLI_SYSTEM_SETTINGS_PATH``)
with ``tools.exclude`` listing every built-in tool (``TOOLS_OFF_EXCLUDE``)
and ``mcp.allowed = []``, which empties the tool registry (verified on
0.26.0). System and user settings merge, so the user's auth settings still
apply. The excluded tools and the settings file's hash are recorded in
``adapter_meta``.

``-o json`` prints ``{"session_id", "response", "stats": {"models": {<model>:
{"api": {...}, "tokens": {"input", "prompt", "candidates", "total", "cached",
"thoughts", "tool"}}}, "tools", "files"}}`` on success and
``{"session_id", "error": {"type", "message", "code"}}`` on failure (the
error is categorised from type + message + code, so ``RetryableQuotaError``
is a retryable rate limit and ``TerminalQuotaError`` a permanent quota error).
There is no reasoning-effort flag in 0.26; a requested effort is not
forwarded, ``reasoning_effort`` is recorded as ``"unsupported"`` and the
requested value as ``adapter_meta.reasoning_effort_unsupported``.
"""
import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Optional

from backend.adapters.base import BaseAdapter
from backend.adapters.cli_common import (
    as_int,
    build_raw_log,
    categorize_error,
    elide_command,
    evidence_line,
    has_evidence,
    format_error,
    get_cli_version,
    run_cli_process,
    smoke_test_cli,
    stderr_tail,
    timeout_result,
)
from backend.models import AdapterRunRequest, AdapterRunResult

AT_ESCAPE_MODES = ("backslash", "fullwidth", "none")
FULLWIDTH_AT = "＠"

#: Built-in tools excluded when tools are off (gemini-cli 0.26 tool names).
TOOLS_OFF_EXCLUDE: tuple[str, ...] = (
    "google_web_search", "web_fetch", "read_file", "read_many_files", "list_directory",
    "glob", "search_file_content", "save_memory", "write_todos", "activate_skill",
    "delegate_to_agent", "run_shell_command", "replace", "write_file",
)
#: Reported as the effort of a call when one was requested (no CLI flag).
EFFORT_UNSUPPORTED = "unsupported"


def tools_off_settings() -> dict:
    return {"tools": {"exclude": list(TOOLS_OFF_EXCLUDE)}, "mcp": {"allowed": []}}


def tools_off_settings_file() -> tuple[str, str]:
    """Path and sha256 of the tools-off system-settings file (content-addressed,
    written once per content into the temp directory)."""
    text = json.dumps(tools_off_settings(), indent=2, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = os.path.join(tempfile.gettempdir(), f"miw_gemini_tools_off_{digest[:16]}.json")
    if not os.path.isfile(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    return path, digest


def neutralize_at_expansion(text: str, mode: str = "backslash") -> str:
    """Protect ``@tokens`` from gemini-cli's @file expansion (see module doc)."""
    if mode == "none":
        return text
    if mode == "fullwidth":
        return text.replace("@", FULLWIDTH_AT)
    if mode != "backslash":
        raise ValueError(f"at_escape must be one of {AT_ESCAPE_MODES}, got {mode!r}")
    # The CLI only checks the immediately preceding character, so an '@' that
    # is already escaped stays escaped after adding another backslash.
    return text.replace("@", "\\@")


def _iter_json_objects(text: str):
    """Yield JSON objects found in ``text`` (whole text first, then per '{' line start)."""
    stripped = (text or "").strip()
    if not stripped:
        return
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            yield obj
            return
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    lines = stripped.splitlines(keepends=True)
    offsets = []
    pos = 0
    for line in lines:
        if line.lstrip().startswith("{"):
            offsets.append(pos + (len(line) - len(line.lstrip())))
        pos += len(line)
    for off in offsets:
        try:
            obj, _ = decoder.raw_decode(stripped, off)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _find_gemini_payload(text: str) -> Optional[dict]:
    """Last JSON object carrying ``response``/``error``/``stats``."""
    found = None
    for obj in _iter_json_objects(text):
        if any(k in obj for k in ("response", "error", "stats")):
            found = obj
    return found


def parse_gemini_stats(stats: Any, requested_model: str = "") -> dict:
    """Token accounting from ``stats.models``, summed over every model used.

    input_tokens = sum(prompt) (includes cached), cached_input_tokens =
    sum(cached), output_tokens = sum(candidates + thoughts). The answering
    model is the one with the most candidate tokens.
    """
    models = stats.get("models") if isinstance(stats, dict) and isinstance(stats.get("models"), dict) else {}
    input_tokens = output_tokens = cached = 0
    ranking = []
    for name, entry in models.items():
        tokens = entry.get("tokens") if isinstance(entry, dict) and isinstance(entry.get("tokens"), dict) else {}
        prompt = as_int(tokens.get("prompt"))
        c = as_int(tokens.get("cached"))
        if not prompt and ("input" in tokens):
            prompt = as_int(tokens.get("input")) + c
        cand = as_int(tokens.get("candidates"))
        thoughts = as_int(tokens.get("thoughts"))
        input_tokens += prompt
        cached += c
        output_tokens += cand + thoughts
        ranking.append((cand, name == requested_model, as_int(tokens.get("total")), name))
    model, source = requested_model, ("requested" if requested_model else "unknown")
    if ranking:
        ranking.sort(reverse=True)
        model, source = ranking[0][3], "stats"
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached,
        "model": model,
        "model_source": source,
        "raw_usage": {"models": models},
    }


def _summarize_stderr(stderr: str) -> str:
    """Most informative error line from a node stack trace, else the tail."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("at ")]
    for ln in reversed(lines):
        if "error" in ln.lower():
            return ln
    return stderr_tail(stderr)


class GeminiCliAdapter(BaseAdapter):
    name: str = "gemini_cli"
    # Minimal -p text; the real prompt is on stdin and the CLI appends this.
    PROMPT_ARG = " "

    def __init__(
        self,
        binary: str = "gemini",
        model: str = "",
        reasoning_effort: str = "",
        allow_tools: bool = True,
        at_escape: str = "backslash",
    ):
        if at_escape not in AT_ESCAPE_MODES:
            raise ValueError(f"at_escape must be one of {AT_ESCAPE_MODES}, got {at_escape!r}")
        self.binary = binary
        self.default_model = model
        self.default_reasoning_effort = reasoning_effort
        self.allow_tools = allow_tools
        self.at_escape = at_escape

    # ── Settings resolution ─────────────────────────────────────────

    def resolve_model(self, request: AdapterRunRequest) -> str:
        return (getattr(request, "model", "") or self.default_model or "").strip()

    def resolve_reasoning_effort(self, request: AdapterRunRequest) -> str:
        return (getattr(request, "reasoning_effort", "") or self.default_reasoning_effort or "").strip()

    def tools_enabled(self, request: AdapterRunRequest) -> bool:
        return bool(self.allow_tools and getattr(request, "allow_tools", True))

    # ── Availability / version ──────────────────────────────────────

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def cli_version(self) -> str:
        return await get_cli_version(self.binary)

    async def smoke_test(self) -> dict:
        if not self.is_available():
            return {"status": "unavailable", "adapter": "gemini_cli",
                    "error": f"'{self.binary}' not found on PATH"}
        return await smoke_test_cli(self.binary, "gemini_cli")

    # ── Command / prompt construction ───────────────────────────────

    def build_prompt(self, request: AdapterRunRequest) -> str:
        """System + developer + user prompt, protected against @/slash handling."""
        prompt_parts = []
        if request.prompt_bundle.system_prompt:
            prompt_parts.append(request.prompt_bundle.system_prompt)
        if request.prompt_bundle.developer_prompt:
            prompt_parts.append(request.prompt_bundle.developer_prompt)
        prompt_parts.append(request.prompt_bundle.user_prompt)
        text = neutralize_at_expansion("\n\n".join(prompt_parts), self.at_escape)
        if text.startswith("/") and not text.startswith(("//", "/*")):
            text = " " + text  # would otherwise be parsed as a slash command
        return text

    def build_command(self, request: AdapterRunRequest) -> list[str]:
        cmd = [self.binary]
        if self.tools_enabled(request):
            cmd.append("--yolo")
        else:
            cmd.extend(["--approval-mode", "default"])
        model = self.resolve_model(request)
        if model:
            cmd.extend(["-m", model])
        cmd.extend(["-o", "json", "-p", self.PROMPT_ARG])
        return cmd

    # ── Run ─────────────────────────────────────────────────────────

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        requested_model = self.resolve_model(request)
        requested_effort = self.resolve_reasoning_effort(request)
        base_fields: dict[str, Any] = dict(provider=self.name, model=requested_model)
        if not self.is_available():
            return AdapterRunResult(
                success=False, error=f"'{self.binary}' not found on PATH",
                exit_code=-1, **base_fields,
            )
        base_fields["cli_version"] = await self.cli_version()

        cmd = self.build_command(request)
        meta: dict[str, Any] = {
            "command": elide_command(cmd),
            "requested_model": requested_model,
            "tools_enabled": self.tools_enabled(request),
            "at_escape": self.at_escape,
        }
        if requested_effort:
            meta["reasoning_effort_unsupported"] = requested_effort
            base_fields["reasoning_effort"] = EFFORT_UNSUPPORTED
        env = None
        if not self.tools_enabled(request):
            settings_path, settings_sha = tools_off_settings_file()
            env = {**os.environ, "GEMINI_CLI_SYSTEM_SETTINGS_PATH": settings_path}
            meta["tools_excluded"] = list(TOOLS_OFF_EXCLUDE)
            meta["gemini_system_settings"] = {"path": settings_path, "sha256": settings_sha}

        proc = await run_cli_process(
            cmd, self.build_prompt(request),
            cwd=request.workspace_context.workspace_path or None,
            timeout=request.timeout_seconds,
            env=env,
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

        payload = _find_gemini_payload(proc.stdout)
        if payload is None and proc.returncode != 0:
            payload = _find_gemini_payload(proc.stderr)  # JSON errors may go to stderr

        usage: dict = {}
        json_error = ""
        json_error_evidence = ""
        if payload is not None:
            response = payload.get("response")
            text = response if isinstance(response, str) else ""
            err = payload.get("error")
            if isinstance(err, dict):
                json_error = str(err.get("message") or err.get("type") or "error")
                json_error_evidence = " ".join(
                    str(err.get(k) or "") for k in ("type", "message", "code", "status"))
                if err.get("type") and str(err["type"]) not in json_error:
                    json_error = f"{err['type']}: {json_error}"
            elif err:
                json_error = str(err)
                json_error_evidence = json_error
            usage = parse_gemini_stats(payload.get("stats"), requested_model)
            meta["model_source"] = usage["model_source"]
            structured = dict(payload)
            stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
            tools = stats.get("tools") if isinstance(stats.get("tools"), dict) else {}
            structured["tool_counts"] = {"tool_calls": as_int(tools.get("totalCalls"))}
        else:
            text = proc.stdout.strip() if proc.returncode == 0 else ""
            meta["model_source"] = "requested" if requested_model else "unknown"
            structured = {}
        structured["adapter_meta"] = meta

        error_msg: Optional[str] = None
        if proc.returncode != 0 or json_error:
            detail = json_error or _summarize_stderr(proc.stderr) or f"exit code {proc.returncode}"
            if json_error:
                category = categorize_error(proc.returncode, json_error_evidence or json_error)
            else:
                category = categorize_error(proc.returncode, proc.stderr)
                if category != "unknown" and not has_evidence(detail, category):
                    line = evidence_line(proc.stderr, category)
                    if line:
                        detail = f"{line} | {detail}"
            error_msg = format_error(category, detail)
        elif not text.strip():
            error_msg = format_error("empty_output", "gemini exited 0 but returned no response text")

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
            cost_estimate=0.0,  # gemini-cli reports no cost
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=usage.get("cached_input_tokens", 0),
            raw_usage=usage.get("raw_usage", {}),
            **{**base_fields, "model": usage.get("model") or requested_model},
        )
