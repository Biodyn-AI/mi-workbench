"""Shared helpers for the CLI-backed adapters (claude, codex, gemini).

Pure standard library. Provides:

- ``categorize_error``: coarse error category for retry logic / reporting
  (the category becomes the ``[category]`` prefix of ``AdapterRunResult.error``,
  which ``backend.orchestrator.retry`` reads).
- ``run_cli_process``: spawn a CLI with the prompt on stdin, enforce a timeout,
  and terminate the CLI's process group (SIGTERM, then SIGKILL after a grace
  period) on timeout, on cancellation and after the CLI exits.
- ``get_cli_version``: ``<binary> --version``, cached per binary.
- ``failure_result`` / ``elide_command``: uniform failure results and a
  loggable copy of the argv (long arguments such as system prompts elided).
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Optional

from backend.models import AdapterRunResult


# Evidence patterns per category, checked in this order. Quota messages that
# mean "do not retry" (usage limits, terminal quota) are "quota", not
# "rate_limit".
_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("quota", ("terminalquotaerror", "usage limit", "quota exceeded for the day",
               "insufficient_quota", "billing")),
    ("rate_limit", ("429", "rate limit", "rate_limit", "too many requests",
                    "resource_exhausted", "resource has been exhausted",
                    "retryablequotaerror")),
    ("auth", ("401", "403", "unauthorized", "authenticat", "auth ", "auth:",
              "not logged in", "login required", "invalid api key", "oauth")),
    ("overloaded", ("overloaded", "529", "503", "502", "service unavailable",
                    "bad gateway", "unavailable")),
    ("timeout", ("timeout", "timed out", "deadline exceeded")),
    ("network", ("connect", "network", "econnrefused", "econnreset", "disconnected",
                 "stream error", "socket hang up", "enotfound", "eai_again",
                 "error sending request")),
)


def categorize_error(exit_code: int, stderr: str) -> str:
    """Categorize an error from CLI output for retry logic.

    Returns one of: "quota", "rate_limit", "auth", "overloaded", "timeout",
    "network", "unknown". (Exit code 429 cannot occur: POSIX exit codes are
    truncated to 0-255, so the status must appear in the text.)
    """
    text = (stderr or "").lower()
    if exit_code == 429:  # only reachable for synthetic codes
        return "rate_limit"
    for category, patterns in _CATEGORY_PATTERNS:
        if any(p in text for p in patterns):
            return category
    return "unknown"


def evidence_line(text: str, category: str) -> str:
    """First line of ``text`` that matches ``category``'s patterns ("" if none)."""
    patterns = dict(_CATEGORY_PATTERNS).get(category, ())
    for line in (text or "").splitlines():
        if any(p in line.lower() for p in patterns):
            return line.strip()[:500]
    return ""


def has_evidence(text: str, category: str) -> bool:
    patterns = dict(_CATEGORY_PATTERNS).get(category, ())
    return any(p in (text or "").lower() for p in patterns)


def detail_with_evidence(stderr: str, category: str, limit: int = 2000) -> str:
    """Error detail (the tail of stderr) that also contains the line the
    ``category`` was derived from, so category and detail always agree."""
    tail = stderr_tail(stderr, limit)
    if category == "unknown" or not stderr:
        return tail
    patterns = dict(_CATEGORY_PATTERNS).get(category, ())
    if any(p in tail.lower() for p in patterns):
        return tail
    for line in (stderr or "").splitlines():
        if any(p in line.lower() for p in patterns):
            return line.strip()[:500] + " | " + tail
    return tail


@dataclass
class CliProcessResult:
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    spawn_error: Optional[str] = None


#: Seconds between SIGTERM and SIGKILL to the CLI's process group, so a CLI
#: (e.g. codex) can shut down the process groups it created for its own tools.
TERM_GRACE_SECONDS = 3.0
#: Seconds to keep reading stdout/stderr after the CLI itself exited (a
#: descendant may still hold the pipes).
PIPE_DRAIN_SECONDS = 2.0


def _signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to process group ``pgid``; False if the group is gone."""
    if pgid <= 0 or os.name != "posix":
        return False
    try:
        if pgid == os.getpgrp():
            return False
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


async def _wait_exit(proc: asyncio.subprocess.Process, timeout: float) -> bool:
    """Wait until the child itself has exited (not until its pipes close)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)
    delay = 0.005
    while proc.returncode is None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.05)
    return True


async def _terminate_group(proc: asyncio.subprocess.Process, grace: float = TERM_GRACE_SECONDS) -> None:
    """SIGTERM the child's process group, wait up to ``grace`` s, then SIGKILL it.

    Always signals the group (also when the direct child has already exited,
    since descendants may survive it); a missing group is ignored."""
    pgid = proc.pid
    alive = _signal_group(pgid, signal.SIGTERM)
    if proc.returncode is None and not alive:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    if alive or proc.returncode is None:
        await _wait_exit(proc, grace)
        # Give remaining group members the same grace, then kill them.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace
        while loop.time() < deadline and _signal_group(pgid, 0):
            await asyncio.sleep(0.05)
    _signal_group(pgid, signal.SIGKILL)
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Immediate SIGKILL of the child's process group (also after the child
    exited); kept for callers that cannot await."""
    if not _signal_group(proc.pid, signal.SIGKILL) and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def _read_all(reader: Optional[asyncio.StreamReader], buf: bytearray) -> None:
    if reader is None:
        return
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            return
        buf += chunk


async def _feed_stdin(proc: asyncio.subprocess.Process, data: bytes) -> None:
    if proc.stdin is None:
        return
    try:
        proc.stdin.write(data)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:  # pragma: no cover - already closed
            pass


async def run_cli_process(
    cmd: list[str],
    stdin_text: str,
    cwd: Optional[str] = None,
    timeout: float = 300,
    env: Optional[dict[str, str]] = None,
) -> CliProcessResult:
    """Run ``cmd`` with ``stdin_text`` piped to stdin.

    The child gets its own session (and process group). stdout/stderr are
    read incrementally and the timeout applies to the CLI process itself,
    not to pipe EOF, so:

    * on timeout or cancellation the group gets SIGTERM, then SIGKILL after
      ``TERM_GRACE_SECONDS`` (the CLI can clean up its own children first);
      a timed-out result keeps the partial stdout/stderr;
    * when the CLI exits normally, its output is returned even if a leftover
      descendant still holds the pipes (they are read for at most
      ``PIPE_DRAIN_SECONDS`` more), and the group is then terminated so
      leftovers do not survive.

    A descendant that left the process group (``setsid``) is not reached.
    """
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None,
            env=env,
            start_new_session=True,
        )
    except (FileNotFoundError, PermissionError, NotADirectoryError, OSError) as exc:
        return CliProcessResult(
            returncode=-1, stdout="", stderr="",
            duration=time.monotonic() - start,
            spawn_error=f"{type(exc).__name__}: {exc}",
        )

    out_buf, err_buf = bytearray(), bytearray()
    tasks = [
        asyncio.ensure_future(_feed_stdin(proc, stdin_text.encode("utf-8"))),
        asyncio.ensure_future(_read_all(proc.stdout, out_buf)),
        asyncio.ensure_future(_read_all(proc.stderr, err_buf)),
    ]

    async def finish_readers(grace: float) -> None:
        _done, pending = await asyncio.wait(tasks, timeout=grace)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    timed_out = False
    try:
        exited = await _wait_exit(proc, timeout)
        if not exited:
            timed_out = True
            await _terminate_group(proc)
            await _wait_exit(proc, 5)
            await finish_readers(PIPE_DRAIN_SECONDS)
        else:
            # The CLI exited: read what is left, then make sure nothing of its
            # group survives (a descendant holding the pipes is cut off).
            _done, pending = await asyncio.wait(tasks, timeout=PIPE_DRAIN_SECONDS)
            await _terminate_group(proc, grace=0.5 if pending else 0.0)
            await finish_readers(PIPE_DRAIN_SECONDS)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(_terminate_group(proc))
        except Exception:
            _kill_process_tree(proc)
        for t in tasks:
            t.cancel()
        raise

    return CliProcessResult(
        returncode=(proc.returncode if proc.returncode is not None else -1) if not timed_out else -1,
        stdout=out_buf.decode("utf-8", errors="replace"),
        stderr=err_buf.decode("utf-8", errors="replace"),
        duration=time.monotonic() - start,
        timed_out=timed_out,
    )


# ── CLI version (cached per binary) ─────────────────────────────────────

_VERSION_CACHE: dict[str, str] = {}


async def get_cli_version(binary: str, timeout: float = 15) -> str:
    """Return the first line of ``<binary> --version`` ("" if unavailable).

    Successful lookups are cached per binary for the life of the process;
    failures are not cached so a later call can retry.
    """
    if binary in _VERSION_CACHE:
        return _VERSION_CACHE[binary]
    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "--version",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return ""
    if proc.returncode != 0:
        return ""
    lines = [ln.strip() for ln in stdout.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    version = lines[0] if lines else ""
    if version:
        _VERSION_CACHE[binary] = version
    return version


def clear_cli_version_cache() -> None:
    """Forget cached versions (tests, or after a CLI upgrade)."""
    _VERSION_CACHE.clear()


async def smoke_test_cli(binary: str, adapter_name: str, timeout: float = 15) -> dict:
    """Uncached ``<binary> --version`` health check (original adapter behaviour).

    A successful probe also refreshes the version cache.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "--version",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return {"status": "timeout", "adapter": adapter_name}
        version = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode == 0 and version:
            _VERSION_CACHE[binary] = version.splitlines()[0].strip()
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "adapter": adapter_name,
            "version": version,
            "exit_code": proc.returncode,
        }
    except Exception as exc:
        return {"status": "error", "adapter": adapter_name, "error": str(exc)}


# ── Result helpers ──────────────────────────────────────────────────────

MAX_ERROR_DETAIL = 2000
MAX_RAW_STDOUT_IN_LOG = 20000


def elide_command(cmd: list[str], max_len: int = 200) -> list[str]:
    """Copy of argv safe to log: long arguments (system prompts) are elided."""
    out = []
    for arg in cmd:
        if len(arg) > max_len:
            out.append(f"<{len(arg)} chars>")
        else:
            out.append(arg)
    return out


def truncate(text: str, limit: int = MAX_ERROR_DETAIL) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def stderr_tail(stderr: str, limit: int = MAX_ERROR_DETAIL) -> str:
    """Last ``limit`` characters of stderr (errors are usually at the end)."""
    stderr = (stderr or "").strip()
    if len(stderr) <= limit:
        return stderr
    return "..." + stderr[-limit:]


def format_error(category: str, detail: str) -> str:
    return f"[{category}] {truncate(detail)}" if detail else f"[{category}]"


def build_raw_log(stderr: str, stdout: str, include_stdout: bool) -> str:
    """raw_log is stderr; on failure the (truncated) stdout is appended."""
    log = stderr or ""
    if include_stdout and stdout:
        log = (log + "\n" if log else "") + "[stdout]\n" + truncate(stdout, MAX_RAW_STDOUT_IN_LOG)
    return log


def timeout_result(timeout: float, duration: float, **fields: Any) -> AdapterRunResult:
    return AdapterRunResult(
        success=False,
        error=f"Timeout after {timeout}s",
        exit_code=-1,
        duration_seconds=duration,
        **fields,
    )


def as_int(value: Any) -> int:
    """Best-effort int conversion for provider usage numbers."""
    try:
        if value is None or isinstance(value, bool):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        if value is None or isinstance(value, bool):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
