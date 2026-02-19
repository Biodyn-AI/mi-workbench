"""Claude Code CLI adapter. Shells out to the `claude` binary."""
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult


def categorize_error(exit_code: int, stderr: str) -> str:
    """Categorize an error from Claude CLI output for retry logic.

    Returns one of: "rate_limit", "auth", "timeout", "network", "unknown".
    """
    stderr_lower = stderr.lower()
    if exit_code == 429 or "429" in stderr or "rate limit" in stderr_lower:
        return "rate_limit"
    if "401" in stderr or "403" in stderr or "auth" in stderr_lower or "unauthorized" in stderr_lower:
        return "auth"
    if "timeout" in stderr_lower or "timed out" in stderr_lower:
        return "timeout"
    if "connect" in stderr_lower or "network" in stderr_lower or "ECONNREFUSED" in stderr:
        return "network"
    return "unknown"


class ClaudeCodeAdapter(BaseAdapter):
    name: str = "claude_code"

    def __init__(self, binary: str = "claude"):
        self.binary = binary

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def smoke_test(self) -> dict:
        if not self.is_available():
            return {"status": "unavailable", "adapter": "claude_code",
                    "error": f"'{self.binary}' not found on PATH"}
        try:
            proc = await asyncio.create_subprocess_exec(
                self.binary, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "adapter": "claude_code",
                "version": stdout.decode().strip(),
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"status": "timeout", "adapter": "claude_code"}
        except Exception as exc:
            return {"status": "error", "adapter": "claude_code", "error": str(exc)}

    def build_command(
        self,
        request: AdapterRunRequest,
    ) -> list[str]:
        """Build the CLI command for claude invocation."""
        cmd = [
            self.binary,
            "-p",  # pipe/print mode
            "--output-format", "json",
            "--max-turns", "1",
        ]

        # System prompt via flag
        if request.prompt_bundle.system_prompt:
            cmd.extend(["--system-prompt", request.prompt_bundle.system_prompt])

        # Allowed tools for workspace-aware execution
        if request.workspace_context.workspace_path:
            cmd.extend(["--allowedTools", "Read,Write,Edit,Bash,Glob,Grep"])

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

    def parse_json_output(self, raw_output: str) -> tuple[str, dict]:
        """Try to parse JSON output from claude CLI.

        Returns (text_output, structured_data).
        Falls back to raw text if JSON parsing fails.
        """
        try:
            data = json.loads(raw_output)
            # Claude CLI JSON output typically has a "result" or "content" field
            text = ""
            if isinstance(data, dict):
                text = data.get("result", data.get("content", data.get("text", "")))
                if not text and "messages" in data:
                    # Extract text from messages array
                    for msg in data.get("messages", []):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            text += msg.get("content", "")
                return text or raw_output, data
            return raw_output, {}
        except (json.JSONDecodeError, TypeError):
            return raw_output, {}

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        if not self.is_available():
            return AdapterRunResult(
                success=False, error=f"'{self.binary}' not found on PATH",
                exit_code=-1,
            )

        start = time.monotonic()

        # Build the full prompt text
        full_prompt = self.build_prompt(request)

        # Write prompt to temp file to avoid shell escaping issues
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        try:
            tmp.write(full_prompt)
            tmp.flush()
            tmp.close()

            cmd = self.build_command(request)

            # Set working directory to workspace path if provided
            cwd = request.workspace_context.workspace_path or None

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=open(tmp.name, "r"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=request.timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return AdapterRunResult(
                    success=False,
                    error=f"Timeout after {request.timeout_seconds}s",
                    exit_code=-1,
                    duration_seconds=time.monotonic() - start,
                )

            raw_output = stdout.decode(errors="replace")
            err_output = stderr.decode(errors="replace")
            duration = time.monotonic() - start

            # Parse JSON output if available
            text_output, structured = self.parse_json_output(raw_output)

            # Categorize errors for retry logic
            error_msg: Optional[str] = None
            if proc.returncode != 0:
                error_category = categorize_error(proc.returncode, err_output)
                error_msg = f"[{error_category}] {err_output}" if err_output else f"[{error_category}] exit code {proc.returncode}"

            return AdapterRunResult(
                success=proc.returncode == 0,
                output=text_output,
                exit_code=proc.returncode or 0,
                duration_seconds=duration,
                error=error_msg,
                raw_log=err_output,
                structured_output=structured,
            )

        finally:
            Path(tmp.name).unlink(missing_ok=True)
