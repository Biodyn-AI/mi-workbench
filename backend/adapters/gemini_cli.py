"""Gemini CLI adapter. Shells out to the `gemini` binary."""
import asyncio
import shutil
import time

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult


class GeminiCliAdapter(BaseAdapter):
    name: str = "gemini_cli"

    def __init__(self, binary: str = "gemini"):
        self.binary = binary

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def smoke_test(self) -> dict:
        if not self.is_available():
            return {"status": "unavailable", "adapter": "gemini_cli",
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
                "adapter": "gemini_cli",
                "version": stdout.decode().strip(),
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"status": "timeout", "adapter": "gemini_cli"}
        except Exception as exc:
            return {"status": "error", "adapter": "gemini_cli", "error": str(exc)}

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

        # Use -p for non-interactive prompt mode, --yolo for auto-approve
        cmd = [self.binary, "--yolo", "-p", full_prompt]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=request.workspace_context.workspace_path or None,
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
