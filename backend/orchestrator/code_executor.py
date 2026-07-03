"""Sandboxed execution of analysis code emitted by the executor agent.

Executor artifacts may contain fenced ``python`` code blocks describing an
analysis. When code execution is enabled, each block is run in an isolated
subprocess and its captured result is folded back into the feedback stream so
that downstream reviewers critique *executed* results rather than an untested
plan.

Isolation properties (best-effort, standard-library only):
  * separate ``python -I`` process (ignores user site/env), never in-process
    ``exec``/``eval``;
  * a fresh temporary working directory, deleted afterwards;
  * a hard wall-clock timeout (SIGKILL on expiry);
  * captured stdout/stderr truncated to a byte cap;
  * ``PYTHONDONTWRITEBYTECODE`` and a minimal environment.

Execution is opt-in (``code_execution_enabled`` in run config, default off).
It verifies deterministic, self-contained snippets; it is not a general secure
sandbox for adversarial code and must not be pointed at untrusted input without
an OS-level sandbox around it.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field

# Fenced ```python (or ```py) ... ``` blocks.
_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_DEFAULT_TIMEOUT = 20  # seconds per block
_OUTPUT_CAP = 4000     # bytes of stdout/stderr retained per block


@dataclass
class CodeBlockResult:
    index: int
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0


@dataclass
class CodeExecutionReport:
    blocks: list[CodeBlockResult] = field(default_factory=list)

    @property
    def executed(self) -> int:
        return len(self.blocks)

    @property
    def passed(self) -> int:
        return sum(1 for b in self.blocks if b.success)

    @property
    def failed(self) -> int:
        return sum(1 for b in self.blocks if not b.success)

    def to_feedback(self) -> str:
        """Render a concise execution report for injection into the loop."""
        if not self.blocks:
            return ""
        lines = [
            "=== CODE EXECUTION REPORT ===",
            f"Executed {self.executed} block(s): {self.passed} passed, "
            f"{self.failed} failed.",
        ]
        for b in self.blocks:
            status = (
                "TIMEOUT" if b.timed_out
                else ("OK" if b.success else f"FAILED (exit {b.exit_code})")
            )
            lines.append(f"\n[Block {b.index}] {status} "
                         f"({b.duration_seconds:.2f}s)")
            if b.stdout.strip():
                lines.append("stdout:\n" + b.stdout.strip())
            if b.stderr.strip():
                lines.append("stderr:\n" + b.stderr.strip())
        lines.append("=== END CODE EXECUTION REPORT ===")
        return "\n".join(lines)


def extract_code_blocks(text: str) -> list[str]:
    """Return the source of every fenced python block in ``text``."""
    if not text:
        return []
    return [m.group(1) for m in _CODE_BLOCK_RE.finditer(text)]


class CodeExecutor:
    """Runs extracted python blocks in isolated subprocesses."""

    def __init__(
        self,
        timeout_seconds: int = _DEFAULT_TIMEOUT,
        output_cap: int = _OUTPUT_CAP,
        python_executable: str | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.output_cap = output_cap
        self.python_executable = python_executable or sys.executable

    async def run_block(self, index: int, code: str) -> CodeBlockResult:
        loop = asyncio.get_event_loop()
        start = loop.time()
        with tempfile.TemporaryDirectory(prefix="miw_exec_") as workdir:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                # No network is granted by this layer; callers needing a hard
                # guarantee should wrap the process in an OS sandbox.
            }
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.python_executable, "-I", "-c", code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir,
                    env=env,
                )
            except Exception as exc:  # pragma: no cover - spawn failure
                return CodeBlockResult(
                    index=index, success=False, exit_code=-1,
                    stderr=f"spawn failed: {exc}",
                    duration_seconds=loop.time() - start,
                )

            timed_out = False
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                except ProcessLookupError:  # pragma: no cover
                    pass
                await proc.wait()
                stdout_b, stderr_b = b"", b"process killed after timeout"

            exit_code = proc.returncode if proc.returncode is not None else -1
            duration = loop.time() - start
            stdout = stdout_b.decode("utf-8", "replace")[: self.output_cap]
            stderr = stderr_b.decode("utf-8", "replace")[: self.output_cap]
            return CodeBlockResult(
                index=index,
                success=(not timed_out and exit_code == 0),
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                duration_seconds=duration,
            )

    async def run_artifact(self, text: str) -> CodeExecutionReport:
        """Execute every python block found in an artifact, in order."""
        blocks = extract_code_blocks(text)
        report = CodeExecutionReport()
        for i, code in enumerate(blocks, start=1):
            report.blocks.append(await self.run_block(i, code))
        return report
