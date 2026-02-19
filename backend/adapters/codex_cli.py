"""Codex CLI adapter. Shells out to the `codex` binary."""
import asyncio
import shutil
import tempfile
import time
from pathlib import Path

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult


class CodexCliAdapter(BaseAdapter):
    name: str = "codex_cli"

    def __init__(self, binary: str = "codex"):
        self.binary = binary

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def smoke_test(self) -> dict:
        if not self.is_available():
            return {"status": "unavailable", "adapter": "codex_cli",
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
                "adapter": "codex_cli",
                "version": stdout.decode().strip(),
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"status": "timeout", "adapter": "codex_cli"}
        except Exception as exc:
            return {"status": "error", "adapter": "codex_cli", "error": str(exc)}

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        if not self.is_available():
            return AdapterRunResult(
                success=False, error=f"'{self.binary}' not found on PATH",
                exit_code=-1,
            )

        start = time.monotonic()

        prompt_parts = []
        if request.prompt_bundle.system_prompt:
            prompt_parts.append(request.prompt_bundle.system_prompt)
        if request.prompt_bundle.developer_prompt:
            prompt_parts.append(request.prompt_bundle.developer_prompt)
        prompt_parts.append(request.prompt_bundle.user_prompt)
        full_prompt = "\n\n".join(prompt_parts)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        try:
            tmp.write(full_prompt)
            tmp.flush()
            tmp.close()

            cmd = [self.binary, "exec", "--full-auto", "--skip-git-repo-check"]

            cwd = request.workspace_context.workspace_path or None
            if cwd:
                cmd.extend(["--cd", cwd])

            # Codex exec reads prompt from stdin when "-" is given
            cmd.append("-")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=open(tmp.name, "r"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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

            output = stdout.decode(errors="replace")
            err_output = stderr.decode(errors="replace")
            duration = time.monotonic() - start

            return AdapterRunResult(
                success=proc.returncode == 0,
                output=output,
                exit_code=proc.returncode or 0,
                duration_seconds=duration,
                error=err_output if proc.returncode != 0 else None,
                raw_log=err_output,
            )

        finally:
            Path(tmp.name).unlink(missing_ok=True)
