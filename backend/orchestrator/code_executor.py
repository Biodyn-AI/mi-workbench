"""Execution of analysis code emitted by the executor agent.

Executor artifacts may contain fenced ``python`` (or ``py``) code blocks.
When code execution is enabled (run config ``code_execution_enabled``,
default off), the blocks are run in a separate process, never in-process
via ``exec``/``eval``, and the captured result is appended to the feedback
stream so that reviewers critique *executed* results rather than an untested
plan.

Block modes
-----------
``block_mode="single"`` (default)
    All fenced blocks of one artifact are concatenated in order into one
    script, run in one process with one working directory and one timeout,
    so later blocks can use names and files created by earlier ones
    (analysis code usually spans blocks).  A failure is attributed to the
    fenced block that contains the failing line.
``block_mode="separate"``
    Each block runs in its own process and its own fresh working directory,
    with no shared state (the behaviour before revision 2).

The report always states which mode was used.

What every backend does
-----------------------
* Runs ``<python> -I -B -u -c <code>``.  ``-I`` (isolated mode) only makes
  the interpreter ignore ``PYTHON*`` environment variables and the user
  site-packages directory, and keeps the script directory off ``sys.path``.
  It is NOT a security boundary.  Because ``-I`` ignores ``PYTHON*``
  variables, ``-B`` (no ``.pyc`` files) and ``-u`` (unbuffered output) are
  passed as flags.  ``PYTHONDONTWRITEBYTECODE``/``PYTHONUNBUFFERED`` are
  still set in the environment for child interpreters started without
  ``-I``.
* Creates a fresh temporary root per execution unit and deletes it
  afterwards: ``<root>/work`` is the working directory, ``<root>/home`` is
  ``HOME`` and ``<root>/tmp`` is ``TMPDIR``.
* Passes a minimal environment, with nothing inherited from the backend
  (API keys and other secrets in the backend environment are not passed):
  ``PATH`` (the interpreter's ``bin`` directory, then ``/usr/bin:/bin:/usr/sbin:/sbin``),
  ``HOME``, ``TMPDIR``, ``PYTHONDONTWRITEBYTECODE=1``, ``PYTHONUNBUFFERED=1``,
  ``MPLBACKEND=Agg`` and ``OMP_NUM_THREADS``/``OPENBLAS_NUM_THREADS``/``MKL_NUM_THREADS=1``,
  plus any operator-supplied ``extra_env``.
* Connects stdin to ``/dev/null``.
* Reads stdout and stderr incrementally and keeps at most
  ``output_limit_bytes`` bytes of each stream: stdout keeps its first and
  last halves, stderr a short head and mostly its tail (so the final
  exception of a traceback is always visible), with a
  ``...[truncated: N bytes omitted; kept first H and last T of TOTAL bytes]...``
  marker in between; the rest is counted and discarded.  Failure
  classification (``limit_hit``) and the attribution of an error to a fenced
  block (``error_block``) use a separate bounded tail of the raw stderr
  (16 KB), not the truncated text.  The backend's memory use is therefore
  bounded however much the code prints.
* Enforces a wall-clock timeout (``timeout_seconds``) per execution unit.
* Lists the files the code wrote in ``work``: name and size, plus the
  contents of small text files, up to a cap.  It never follows symlinks and
  never opens hard links or non-regular files.  Executed outputs such as
  ``results.json`` can therefore be shown to reviewers.
* Reports the exit code, the terminating signal, ``timed_out``,
  ``killed_by_limit``/``limit_hit``, byte counts and truncation flags, the
  wall time, the backend, the block mode, and the limits that were and were
  not enforced.

Backends
--------
``subprocess`` (default): resource-limited isolation, NOT a security boundary
    Guarantees:
    * The child starts a new session and process group
      (``start_new_session=True``).
    * On timeout, on cancellation, and again after the main process exits,
      SIGKILL is sent to the whole process group, so children and
      grandchildren that stay in the group do not survive.
    * POSIX rlimits are applied before the first line of user code runs.
      A launcher in the child calls ``setrlimit`` and then ``execv``s the
      interpreter; ``preexec_fn`` is avoided because it is unsafe in the
      threaded backend.  If the kernel rejects a limit (or the ``resource``
      module is missing) the launcher exits with status 125 without running
      the code, and the unit is reported as failed with
      ``error="resource limits could not be applied"`` (fail closed).  The
      kernel may silently clamp a limit to a stricter value (e.g. macOS
      clamps ``RLIMIT_NPROC`` to ``kern.maxprocperuid``).  The rlimits are:
      ``RLIMIT_CPU`` (SIGXCPU at ``cpu_seconds``, SIGKILL one second later);
      ``RLIMIT_FSIZE`` (``max_file_mb``; writes beyond it fail with EFBIG);
      ``RLIMIT_NOFILE`` (``max_open_files``); ``RLIMIT_CORE=0``;
      ``RLIMIT_AS`` (``memory_mb``, Linux only; it caps *virtual* address
      space, which some runtimes reserve generously, so raise it or set 0
      if imports fail); and ``RLIMIT_NPROC``
      (``max_processes``), applied only when set and then as headroom over
      the user's current process count, because NPROC counts every process
      of the user.  Soft and hard limits are both lowered, so the code
      cannot raise them.  They also bind orphaned descendants.

    Does NOT guarantee:
    * Any network restriction.
    * Any filesystem confinement.  The code runs with the backend user's
      UID and can read and write any file that user can, using absolute
      paths.  ``read_only_paths`` is informational only.
    * A memory limit on macOS: the kernel does not enforce
      ``RLIMIT_AS``/``RLIMIT_DATA``.
    * Cleanup of a descendant that calls ``setsid()``.  Such a process
      leaves the process group and survives the group kill, although
      ``RLIMIT_CPU`` still bounds its CPU time.  The executor does not wait
      for it: if it holds stdout/stderr open, reading stops after a 2 s
      grace period and the report notes it.
    * Protection against privilege escalation or kernel exploits.

``sandbox_exec`` (macOS only): OS-level confinement of network access, writes and reads
    Runs the same launcher and command, with the same rlimits and the same
    process-group handling, under ``/usr/bin/sandbox-exec`` with the Seatbelt
    profile in ``sandbox/seatbelt_profile.sb``.  The kernel enforces the
    profile, and all descendants inherit it.

    Guarantees (verified by the live tests in
    ``backend/tests/test_code_executor_sandbox.py``):
    * All network operations are denied (``deny network*``), including
      localhost and Unix-domain sockets, unless ``allow_network=True``.
    * File writes are denied everywhere except the per-unit temporary root
      and the device nodes ``/dev/null``, ``/dev/zero``, ``/dev/tty``,
      ``/dev/dtracehelper`` (the DTrace helper device dyld opens) and
      ``/dev/fd/<n>``.
    * File reads are denied except in: the root directory listing only
      (``/``); ``/System``, ``/usr``, ``/bin``, ``/sbin``,
      ``/Library/Frameworks``, ``/Library/Fonts``, ``/private/etc``,
      ``/private/var/db/timezone`` and all of ``/dev``; the package-manager
      prefixes ``/opt/homebrew`` and ``/opt/local`` (readable in full,
      including their ``etc/`` and ``var/``; on Apple-silicon Homebrew these
      are user-owned trees); the interpreter's prefixes and ``sys.path``; the
      temporary root; and ``read_only_paths``.  ``sys.path`` can include the
      source directories of editable installs, and those then become
      readable too.  Neither the interpreter paths nor ``read_only_paths``
      may be ``/``, the home directory or one of its ancestors, a volume root
      (``/Volumes/<name>``) or a shared temp root: such ``read_only_paths``
      entries are rejected with ``ValueError`` (for every backend), such
      interpreter paths are dropped.
      File metadata (``stat``) stays readable everywhere, so the code can
      learn that a path exists but cannot read a file or list a directory
      outside the allow-list.
    * Mach service lookups are denied except for a few that libc needs.
      The code therefore cannot ask LaunchServices (``open``), the
      pasteboard or AppleEvents to act outside the sandbox on its behalf.
    * Process inspection and signals are limited to processes in the same
      sandbox (``process-info*``, ``signal``), and the ``kern.procargs*``
      sysctl is denied, so the code cannot read the arguments or environment
      of the backend (which may hold API keys) or signal it.

    Does NOT guarantee:
    * A memory limit (as for ``subprocess`` on macOS).
    * Secrecy of all metadata.  Path existence (``stat``) and other
      non-content metadata, such as the list of running processes, may
      remain visible.
    * Cleanup of ``setsid()`` escapees.  They survive the group kill but
      stay sandboxed and CPU-limited.
    * Protection against kernel or sandbox escapes.
    * Stability across macOS releases.  Apple marks ``sandbox-exec``
      DEPRECATED (``man sandbox-exec``) and the profile language is
      undocumented.  It works on the tested macOS 26, but future releases
      may change or remove it.

``docker``: container isolation
    Runs ``docker run --rm --pull never --network none --read-only --tmpfs
    /tmp -v <work>:/work -w /work [-v <ro>:<ro>:ro ...] --memory <m>m
    --memory-swap <m>m --cpus 1 --pids-limit <n> --cap-drop ALL
    --security-opt no-new-privileges --ulimit core/cpu/fsize/nofile
    --user <uid>:<gid> <image> python -I -B -u -c <code>``.

    Guarantees:
    * No network (``--network none``) unless ``allow_network``.
    * A read-only root filesystem.  Only ``/work`` (the unit's work
      directory) and ``/tmp`` (tmpfs) are writable, and ``read_only_paths``
      are mounted read-only.
    * A hard memory cap (cgroup, swap disabled).
    * CPU, PID, file-size and open-file caps.
    * All capabilities dropped and a non-root user.
    * The container is killed (``docker kill``) on timeout and removed.

    Does NOT guarantee:
    * Isolation from the host kernel.  A container shares the kernel of
      its Linux host; on macOS, Docker Desktop adds a VM boundary.
    * Anything about the image contents.  The image must exist locally,
      because images are never pulled.

Configuration keys
------------------
``CodeExecutor.from_run_config(config)`` reads these run-config keys (all
optional; defaults in brackets):

=====================================  ==========================================
``code_execution_backend``             ``subprocess`` | ``sandbox_exec`` | ``docker`` [``subprocess``]
``code_execution_python``              interpreter path or name on PATH [``sys.executable``]
``code_execution_timeout``             wall-clock seconds per execution unit [20]
``code_execution_cpu_seconds``         RLIMIT_CPU seconds; 0 disables [2 x timeout]
``code_execution_memory_mb``           RLIMIT_AS (Linux) / ``--memory`` (docker); 0 disables [4096]
``code_execution_read_only_paths``     list (or os.pathsep-separated string) of readable dirs [[]]
``code_execution_allow_network``       bool; only enforceable by sandbox_exec/docker [False]
``code_execution_max_file_mb``         RLIMIT_FSIZE MB; 0 disables [64]
``code_execution_max_processes``       NPROC headroom / docker ``--pids-limit`` [unset; docker 128]
``code_execution_max_open_files``      RLIMIT_NOFILE; 0 disables [256]
``code_execution_output_limit_bytes``  bytes kept per stream [4000]
``code_execution_block_mode``          ``single`` | ``separate`` [``single``]
``code_execution_docker_image``        image for the docker backend [``python:3.11-slim``]
``code_execution_capture_text_bytes``  bytes of text kept per written file [4000]
``code_execution_capture_total_bytes`` bytes of text kept over all written files [16000]
``code_execution_capture_max_files``   files listed from the work directory [50]
``code_execution_persist_dir``         directory that receives the complete outputs [unset]
``code_execution_persist_max_mb``      MB copied per execution unit [256]
=====================================  ==========================================

File capture reads ``results.json`` in the work directory first (the
verified-execution prompt asks for every reported number to be written
there), so other text files written by the code cannot exhaust the total
text budget before it; the per-file cap still applies to it.

With ``code_execution_persist_dir`` set, every execution unit also keeps its
complete outputs in ``<persist_dir>/<timestamp>_<id>/``: ``code.py``, the full
``stdout.txt`` / ``stderr.txt`` (not only the head and tail shown in the
report) and ``work/`` with every regular file the code wrote (symlinks, hard
links and special files are never followed or copied; at most
``persist_max_mb`` per unit). The path is reported as ``persisted_to``.

If a backend is unavailable (for example ``sandbox_exec`` off macOS, or the
Docker daemon is down) the executor fails closed.  The execution unit is
reported as failed, with the reason, and never silently falls back to a
weaker backend.
"""
from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import signal
import sys
import tempfile
import textwrap
import uuid
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from backend.orchestrator.sandbox import capture as _capture
from backend.orchestrator.sandbox import docker as _docker
from backend.orchestrator.sandbox import interpreter as _interp
from backend.orchestrator.sandbox import limits as _limits
from backend.orchestrator.sandbox import seatbelt as _seatbelt
from backend.orchestrator.sandbox.capture import CapturedFile

__all__ = [
    "BACKENDS", "BLOCK_MODES", "BACKEND_GUARANTEES", "CapturedFile",
    "CodeBlockResult", "CodeExecutionReport", "CodeExecutor",
    "combine_blocks", "extract_code_blocks",
]

# Fenced ```python (or ```py) ... ``` blocks.
_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_DEFAULT_TIMEOUT = 20      # wall-clock seconds per execution unit
_OUTPUT_CAP = 4000         # bytes of stdout/stderr retained per stream
_DEFAULT_MEMORY_MB = 4096
_DEFAULT_MAX_FILE_MB = 64
_DEFAULT_MAX_OPEN_FILES = 256
_MIN_OPEN_FILES = 16       # the interpreter itself needs a few descriptors
_DRAIN_GRACE_SECONDS = 2.0 # wait for pipes to close after the group is killed
_STDERR_HEAD_BYTES = 512   # stderr keeps a short head and mostly its tail
# Files read before any other written file (top level of the work dir):
# results.json is where the executor prompt asks for every reported number.
_CAPTURE_PRIORITY = ("results.json",)

BACKENDS = ("subprocess", "sandbox_exec", "docker")
BLOCK_MODES = ("single", "separate")
_BLOCK_MODE_ALIASES = {
    "single": "single", "single_script": "single", "script": "single",
    "combined": "single", "separate": "separate", "separate_blocks": "separate",
    "per_block": "separate", "blocks": "separate",
}
_PYTHON_FLAGS = ("-I", "-B", "-u")
_SYSTEM_PATH = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")

BACKEND_GUARANTEES: dict[str, dict[str, Any]] = {
    "subprocess": {
        "security_boundary": False,
        "summary": "resource-limited isolation (NOT a security boundary)",
        "network": "not restricted",
        "file_writes": "not confined (backend user's permissions)",
        "file_reads": "not confined (backend user's permissions)",
        "processes": "new session; whole process group SIGKILLed on timeout/exit",
    },
    "sandbox_exec": {
        "security_boundary": True,
        "summary": "macOS Seatbelt confinement (sandbox-exec, deprecated by Apple)",
        "network": "denied by the kernel (deny network*)",
        "file_writes": "denied outside the per-run temporary directory",
        "file_reads": "limited to system/interpreter paths, the temporary "
                      "directory and read_only_paths",
        "processes": "new session; whole process group SIGKILLed on timeout/exit; "
                     "no signals to / inspection of processes outside the sandbox",
    },
    "docker": {
        "security_boundary": True,
        "summary": "container isolation (docker run --network none --read-only)",
        "network": "none (--network none)",
        "file_writes": "read-only root; only /work and tmpfs /tmp writable",
        "file_reads": "container image + /work + read_only_paths (mounted ro)",
        "processes": "--pids-limit; container killed on timeout and removed",
    },
}


# ── Results ─────────────────────────────────────────────────────────────


@dataclass
class CodeBlockResult:
    """Result of one execution unit.

    In ``separate`` mode a unit is one fenced block; in ``single`` mode it
    is the combined script and ``block_indices`` lists the blocks it holds.
    """
    index: int
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0
    block_indices: list[int] = field(default_factory=list)
    signal: Optional[int] = None
    killed_by_limit: bool = False
    limit_hit: Optional[str] = None  # wall_timeout|cpu|memory|file_size|open_files|processes|sigkill
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error_block: Optional[int] = None
    files: list[CapturedFile] = field(default_factory=list)
    backend: str = "subprocess"
    error: Optional[str] = None      # executor-level failure (e.g. backend unavailable)
    notes: list[str] = field(default_factory=list)
    persisted_to: Optional[str] = None  # complete outputs (code_execution_persist_dir)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["files"] = [f.to_dict() for f in self.files]
        return d


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _block_range(indices: Sequence[int]) -> str:
    if not indices:
        return ""
    if len(indices) == 1:
        return f"block {indices[0]}"
    if list(indices) == list(range(indices[0], indices[-1] + 1)):
        return f"blocks {indices[0]}-{indices[-1]}"
    return "blocks " + ", ".join(str(i) for i in indices)


@dataclass
class CodeExecutionReport:
    blocks: list[CodeBlockResult] = field(default_factory=list)
    mode: str = "separate"
    backend: str = "subprocess"
    code_blocks: int = 0
    limits: dict[str, Any] = field(default_factory=dict)
    guarantees: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def executed(self) -> int:
        """Number of execution units run (1 per artifact in single mode)."""
        return len(self.blocks)

    @property
    def passed(self) -> int:
        return sum(1 for b in self.blocks if b.success)

    @property
    def failed(self) -> int:
        return sum(1 for b in self.blocks if not b.success)

    @property
    def n_code_blocks(self) -> int:
        return self.code_blocks or sum(len(b.block_indices) or 1 for b in self.blocks)

    @property
    def mode_description(self) -> str:
        n = self.n_code_blocks
        if self.mode == "single":
            return (f"single script ({n} fenced python block(s) concatenated in "
                    f"order and run in one process with shared state)")
        return (f"separate blocks ({n} fenced python block(s), each run in its "
                f"own process and directory, no shared state)")

    def summary(self) -> dict[str, Any]:
        """Compact dict for ``IterationResult.code_execution``."""
        return {
            "executed": self.executed,
            "passed": self.passed,
            "failed": self.failed,
            "mode": self.mode,
            "backend": self.backend,
            "code_blocks": self.n_code_blocks,
            "timed_out": sum(1 for b in self.blocks if b.timed_out),
            "killed_by_limit": sum(1 for b in self.blocks if b.killed_by_limit),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "limits": self.limits,
            "guarantees": self.guarantees,
            "notes": list(self.notes),
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def _limits_line(self) -> str:
        parts = []
        if "wall_timeout_seconds" in self.limits:
            parts.append(f"wall {self.limits['wall_timeout_seconds']:g}s")
        for key, text in (self.limits.get("enforced") or {}).items():
            parts.append(f"{key}: {text}")
        for key in (self.limits.get("not_enforced") or {}):
            parts.append(f"{key}: not enforced")
        return "; ".join(parts)

    def to_feedback(self) -> str:
        """Render a concise execution report for injection into the loop."""
        if not self.blocks:
            return ""
        lines = ["=== CODE EXECUTION REPORT ===",
                 f"Mode: {self.mode_description}."]
        g = self.guarantees or BACKEND_GUARANTEES.get(self.backend, {})
        backend_line = f"Backend: {self.backend}"
        if g:
            backend_line += (f" - {g.get('summary', '')}; network: {g.get('network', '?')}; "
                             f"writes: {g.get('file_writes', '?')}")
        lines.append(backend_line + ".")
        limits_line = self._limits_line()
        if limits_line:
            lines.append(f"Limits: {limits_line}.")
        unit = "script" if self.mode == "single" else "block"
        lines.append(f"Executed {self.executed} {unit}(s): {self.passed} passed, "
                     f"{self.failed} failed.")
        for note in self.notes:
            lines.append(f"Note: {note}")
        for b in self.blocks:
            if b.error:
                status = f"NOT RUN ({b.error})"
            elif b.timed_out:
                status = "TIMEOUT (process group killed)"
            elif b.killed_by_limit:
                status = f"KILLED BY LIMIT ({b.limit_hit}, signal {b.signal})"
            elif b.success:
                status = "OK"
            else:
                status = f"FAILED (exit {b.exit_code})"
                if b.limit_hit:
                    status += f" [limit: {b.limit_hit}]"
            if self.mode == "single":
                label = f"Script ({_block_range(b.block_indices)})" if b.block_indices else "Script"
            else:
                label = f"Block {b.index}"
            where = ""
            if b.error_block is not None and self.mode == "single" and not b.success:
                where = f" - error in block {b.error_block}"
            lines.append(f"\n[{label}] {status}{where} ({b.duration_seconds:.2f}s)")
            if b.stdout.strip():
                lines.append("stdout:\n" + b.stdout.strip())
            if b.stderr.strip():
                lines.append("stderr:\n" + b.stderr.strip())
            for note in b.notes:
                lines.append(f"note: {note}")
            if b.files:
                listing = ", ".join(
                    f"{f.path} ({_fmt_size(f.size_bytes)}"
                    + (f", {f.skipped}" if f.skipped else "") + ")"
                    for f in b.files)
                lines.append(f"files written: {listing}")
                for f in b.files:
                    if f.content is not None and f.content.strip():
                        tail = " [truncated]" if f.truncated else ""
                        lines.append(f"--- {f.path}{tail} ---\n{f.content.rstrip()}")
        lines.append("=== END CODE EXECUTION REPORT ===")
        return "\n".join(lines)


# ── Extraction ──────────────────────────────────────────────────────────


def extract_code_blocks(text: str) -> list[str]:
    """Return the source of every fenced python block in ``text``."""
    if not text:
        return []
    return [m.group(1) for m in _CODE_BLOCK_RE.finditer(text)]


_BLOCK_HEADER = "# ---- MI-Workbench: fenced python block {i} of {n} ----"


def combine_blocks(blocks: Sequence[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate blocks into one script.

    Returns ``(script, spans)`` where each span is
    ``(block_index, first_line, last_line)`` (1-based script line numbers),
    used to attribute a traceback line to its fenced block.
    """
    out: list[str] = []
    spans: list[tuple[int, int, int]] = []
    n = len(blocks)
    for i, code in enumerate(blocks, start=1):
        out.append(_BLOCK_HEADER.format(i=i, n=n))
        first = len(out) + 1
        body = textwrap.dedent(code).rstrip("\n")
        out.extend(body.split("\n") if body else [""])
        spans.append((i, first, len(out)))
    return "\n".join(out) + "\n", spans


_TRACE_LINE_RE = re.compile(r'File "<string>", line (\d+)')


def _error_block(stderr: str, spans: Sequence[tuple[int, int, int]]) -> Optional[int]:
    hits = _TRACE_LINE_RE.findall(stderr or "")
    if not hits:
        return None
    line = int(hits[-1])
    for idx, first, last in spans:
        if first - 1 <= line <= last:  # header line counts as the block's
            return idx
    return None


# ── Helpers ─────────────────────────────────────────────────────────────


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return bool(value)


def _opt_number(name: str, value: Any, *, integer: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        num = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if num < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return num


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [p for p in str(value).split(os.pathsep) if p]
    return [str(p) for p in value if p]


def _force_remove(path: str) -> bool:
    """Remove a directory tree that ``shutil.rmtree`` could not (``rm -rf``
    in a subprocess does not hold one descriptor per level)."""
    import shutil
    import subprocess
    shutil.rmtree(path, ignore_errors=True)
    if not os.path.exists(path):
        return True
    try:
        subprocess.run(["rm", "-rf", "--", path], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return not os.path.exists(path)


def _kill_process_group(pid: Optional[int]) -> bool:
    """SIGKILL the process group led by ``pid`` (never our own group)."""
    if pid is None or os.name != "posix":
        return False
    try:
        if pid == os.getpgrp():
            return False
        os.killpg(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


async def _wait_for_exit(proc: "asyncio.subprocess.Process", timeout: float) -> bool:
    """Wait until the child itself has exited; True if it did within ``timeout``.

    ``Process.wait()`` is not used: on Python <= 3.11 it only returns once
    every pipe is closed, so a grandchild that inherited stdout would keep it
    blocked until the timeout even though the child had exited.  The return
    code is set as soon as the child is reaped, so it is polled instead.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    delay = 0.005
    while proc.returncode is None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.05)
    return True


_ERRNO_LIMITS = (
    (errno.EFBIG, "file_size"),
    (errno.EMFILE, "open_files"),
)


def _classify(exit_code: int, timed_out: bool, stderr: str, backend: str,
              nproc_applied: bool) -> tuple[Optional[int], bool, Optional[str]]:
    """Return ``(signal, killed_by_limit, limit_hit)``."""
    sig: Optional[int] = None
    if exit_code < 0:
        sig = -exit_code
    elif backend == "docker" and 128 < exit_code < 160:
        sig = exit_code - 128
    if timed_out:
        return sig, False, "wall_timeout"
    if sig == getattr(signal, "SIGXCPU", -1):
        return sig, True, "cpu"
    if sig == getattr(signal, "SIGXFSZ", -1):
        return sig, True, "file_size"
    if sig == signal.SIGKILL:
        # RLIMIT_CPU hard limit, kernel OOM killer or docker memory cgroup.
        return sig, True, "sigkill"
    if exit_code != 0:
        tail = "\n".join((stderr or "").strip().splitlines()[-5:])
        if "MemoryError" in tail:
            return sig, False, "memory"
        for code, label in _ERRNO_LIMITS:
            if f"[Errno {code}]" in tail:
                return sig, False, label
        if nproc_applied and f"[Errno {errno.EAGAIN}]" in tail:
            return sig, False, "processes"
    return sig, False, None


# ── Executor ────────────────────────────────────────────────────────────


class CodeExecutor:
    """Run extracted python blocks in a confined child process.

    The first three positional parameters keep the pre-revision signature
    ``CodeExecutor(timeout_seconds, output_cap, python_executable)``.
    Everything else is keyword-only; see the module docstring for the
    guarantees of each backend.

    Limit parameters: ``None`` means the default (``cpu_seconds=None``
    means 2 x ``timeout_seconds``); ``0`` disables that limit.
    """

    def __init__(
        self,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        output_cap: Optional[int] = None,
        python_executable: Optional[str] = None,
        *,
        python: Optional[str] = None,
        backend: str = "subprocess",
        cpu_seconds: Optional[float] = None,
        memory_mb: Optional[int] = _DEFAULT_MEMORY_MB,
        max_file_mb: Optional[float] = _DEFAULT_MAX_FILE_MB,
        max_processes: Optional[int] = None,
        max_open_files: Optional[int] = _DEFAULT_MAX_OPEN_FILES,
        output_limit_bytes: Optional[int] = None,
        read_only_paths: Optional[Sequence[str]] = None,
        allow_network: bool = False,
        block_mode: str = "single",
        docker_image: str = _docker.DEFAULT_IMAGE,
        docker_python: str = "python",
        docker_bin: str = "docker",
        docker_user: Optional[str] = None,
        workdir_parent: Optional[str] = None,
        extra_env: Optional[Mapping[str, str]] = None,
        capture_files: bool = True,
        capture_max_files: int = 50,
        capture_text_bytes: int = 4000,
        capture_total_bytes: int = 16000,
        persist_dir: Optional[str] = None,
        persist_max_mb: Optional[float] = 256,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        mode = _BLOCK_MODE_ALIASES.get(str(block_mode).strip().lower())
        if mode is None:
            raise ValueError(f"block_mode must be one of {BLOCK_MODES}, got {block_mode!r}")
        timeout = _opt_number("timeout_seconds", timeout_seconds)
        if not timeout:
            raise ValueError("timeout_seconds must be > 0")
        self.timeout_seconds = timeout
        self.backend = backend
        self.block_mode = mode

        cpu = _opt_number("cpu_seconds", cpu_seconds)
        self.cpu_seconds = (max(1.0, 2.0 * timeout) if cpu is None else cpu) or None
        self.memory_mb = int(_opt_number("memory_mb", memory_mb, integer=True) or 0) or None
        self.max_file_mb = _opt_number("max_file_mb", max_file_mb) or None
        self.max_processes = int(_opt_number("max_processes", max_processes, integer=True) or 0) or None
        nofile = int(_opt_number("max_open_files", max_open_files, integer=True) or 0) or None
        if nofile is not None and nofile < _MIN_OPEN_FILES:
            raise ValueError(f"max_open_files must be 0 (disabled) or >= {_MIN_OPEN_FILES}")
        self.max_open_files = nofile

        cap = output_limit_bytes if output_limit_bytes is not None else output_cap
        cap = _OUTPUT_CAP if cap is None else int(_opt_number("output_limit_bytes", cap, integer=True))
        self.output_limit_bytes = cap
        self.output_cap = cap  # backward-compatible attribute name

        self.python = python or python_executable or sys.executable
        self.python_executable = self.python  # backward-compatible attribute name
        self.read_only_paths = _as_path_list(read_only_paths)
        # Fail closed on allow-list entries that would defeat confinement.
        _interp.check_read_paths(
            os.path.realpath(os.path.expanduser(str(p))) for p in self.read_only_paths)
        self.allow_network = _to_bool(allow_network)
        self.docker_image = docker_image or _docker.DEFAULT_IMAGE
        self.docker_python = docker_python or "python"
        self.docker_bin = docker_bin or "docker"
        if docker_user is None and hasattr(os, "getuid"):
            docker_user = f"{os.getuid()}:{os.getgid()}"
        self.docker_user = docker_user
        self.workdir_parent = workdir_parent
        self.extra_env = dict(extra_env or {})
        self.capture_files = bool(capture_files)
        self.capture_max_files = int(_opt_number("capture_max_files", capture_max_files,
                                                 integer=True) or 0)
        self.capture_text_bytes = int(_opt_number("capture_text_bytes", capture_text_bytes,
                                                  integer=True) or 0)
        self.capture_total_bytes = int(_opt_number("capture_total_bytes", capture_total_bytes,
                                                   integer=True) or 0)
        self.persist_dir = (os.path.realpath(os.path.expanduser(str(persist_dir)))
                            if persist_dir else None)
        pmb = _opt_number("persist_max_mb", persist_max_mb)
        self.persist_max_bytes = int(pmb * _limits.MB) if pmb else 256 * _limits.MB
        self._availability: Optional[tuple[bool, str]] = None

    # ── configuration ──────────────────────────────────────────────────

    RUN_CONFIG_KEYS: dict[str, str] = {
        "code_execution_backend": "backend",
        "code_execution_python": "python",
        "code_execution_timeout": "timeout_seconds",
        "code_execution_cpu_seconds": "cpu_seconds",
        "code_execution_memory_mb": "memory_mb",
        "code_execution_read_only_paths": "read_only_paths",
        "code_execution_allow_network": "allow_network",
        "code_execution_max_file_mb": "max_file_mb",
        "code_execution_max_processes": "max_processes",
        "code_execution_max_open_files": "max_open_files",
        "code_execution_output_limit_bytes": "output_limit_bytes",
        "code_execution_block_mode": "block_mode",
        "code_execution_docker_image": "docker_image",
        "code_execution_capture_text_bytes": "capture_text_bytes",
        "code_execution_capture_total_bytes": "capture_total_bytes",
        "code_execution_capture_max_files": "capture_max_files",
        "code_execution_persist_dir": "persist_dir",
        "code_execution_persist_max_mb": "persist_max_mb",
    }

    @classmethod
    def from_run_config(cls, config: Optional[Mapping[str, Any]], **overrides: Any) -> "CodeExecutor":
        """Build an executor from run-config keys (see module docstring).

        Keys that are absent or ``None`` keep the constructor defaults;
        ``overrides`` (constructor keyword arguments) win over the config.
        Raises ``ValueError`` for invalid values.
        """
        kwargs: dict[str, Any] = {}
        for key, param in cls.RUN_CONFIG_KEYS.items():
            if config and config.get(key) is not None:
                kwargs[param] = config[key]
        kwargs.update(overrides)
        return cls(**kwargs)

    def config_dict(self) -> dict[str, Any]:
        """Effective configuration (JSON-serialisable)."""
        return {
            "backend": self.backend,
            "python": self.python if self.backend != "docker" else self.docker_python,
            "timeout_seconds": self.timeout_seconds,
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
            "max_file_mb": self.max_file_mb,
            "max_processes": self.max_processes,
            "max_open_files": self.max_open_files,
            "output_limit_bytes": self.output_limit_bytes,
            "read_only_paths": list(self.read_only_paths),
            "allow_network": self.allow_network,
            "block_mode": self.block_mode,
            "docker_image": self.docker_image if self.backend == "docker" else None,
            "capture_files": self.capture_files,
            "capture_max_files": self.capture_max_files,
            "capture_text_bytes": self.capture_text_bytes,
            "capture_total_bytes": self.capture_total_bytes,
            "persist_dir": self.persist_dir,
            "persist_max_bytes": self.persist_max_bytes if self.persist_dir else None,
        }

    def guarantees(self) -> dict[str, Any]:
        """What this configuration does and does not guarantee."""
        g = dict(BACKEND_GUARANTEES[self.backend])
        if self.allow_network and self.backend != "subprocess":
            g["network"] = "allowed by configuration (allow_network=True)"
        if self.backend == "subprocess" and not self.allow_network:
            g["network"] = "not restricted (allow_network=False cannot be enforced by the subprocess backend)"
        return g

    def _limit_plan(self) -> _limits.LimitPlan:
        if self.backend == "docker":
            plan = _limits.LimitPlan()
            if self.cpu_seconds:
                plan.enforced["cpu_seconds"] = f"--ulimit cpu={int(-(-self.cpu_seconds // 1))} and --cpus 1"
            if self.memory_mb:
                plan.enforced["memory_mb"] = f"--memory {self.memory_mb}m (cgroup, swap disabled)"
            if self.max_file_mb:
                plan.enforced["max_file_mb"] = f"--ulimit fsize ({self.max_file_mb:g} MB)"
            if self.max_open_files:
                plan.enforced["max_open_files"] = f"--ulimit nofile={self.max_open_files}"
            plan.enforced["max_processes"] = (
                f"--pids-limit {self.max_processes or _docker.DEFAULT_PIDS_LIMIT}")
            return plan
        return _limits.plan_limits(
            cpu_seconds=self.cpu_seconds, memory_mb=self.memory_mb,
            max_file_mb=self.max_file_mb, max_processes=self.max_processes,
            max_open_files=self.max_open_files,
        )

    def describe_limits(self, plan: Optional[_limits.LimitPlan] = None) -> dict[str, Any]:
        plan = plan or self._limit_plan()
        return {
            "wall_timeout_seconds": self.timeout_seconds,
            "output_limit_bytes": self.output_limit_bytes,
            "enforced": dict(plan.enforced),
            "not_enforced": dict(plan.not_enforced),
        }

    # ── availability ───────────────────────────────────────────────────

    def _check_backend_sync(self) -> tuple[bool, str]:
        if self.backend == "docker":
            return _docker.docker_available(self.docker_bin, self.docker_image)
        if os.name != "posix":  # pragma: no cover - Windows
            return False, "code execution requires a POSIX host"
        if self.backend == "sandbox_exec":
            ok, reason = _seatbelt.sandbox_exec_available()
            if not ok:
                return ok, reason
        if _interp.resolve_python(self.python) is None:
            return False, f"interpreter not found or not executable: {self.python}"
        return True, "ok"

    async def check_available(self) -> tuple[bool, str]:
        """(available, reason) for the configured backend (cached)."""
        if self._availability is None:
            self._availability = await asyncio.to_thread(self._check_backend_sync)
        return self._availability

    # ── command construction ───────────────────────────────────────────

    def _env(self, python: str, home: str, tmp: str) -> dict[str, str]:
        path_dirs = [os.path.dirname(python)] + [d for d in _SYSTEM_PATH
                                                 if d != os.path.dirname(python)]
        env = {
            "PATH": os.pathsep.join(path_dirs),
            "HOME": home,
            "TMPDIR": tmp,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        env.update(self.extra_env)
        return env

    def _local_command(self, python: str, code: str, plan: _limits.LimitPlan) -> list[str]:
        target = [python, *_PYTHON_FLAGS, "-c", code]
        if not _limits.HAS_RESOURCE:  # pragma: no cover - Windows
            return target
        return [python, "-I", "-c", _limits.LAUNCHER_SOURCE,
                json.dumps(plan.launcher_spec()), *target]

    async def _sandbox_read_paths(self, python: str) -> tuple[list[str], list[str]]:
        interp_paths = await asyncio.to_thread(_interp.interpreter_read_paths, python)
        ro, notes = _interp.normalize_read_paths(self.read_only_paths)
        paths: list[str] = []
        for p in interp_paths + ro:
            if p not in paths:
                paths.append(p)
        return paths, notes

    def build_docker_command(self, code: str, work_dir: str, container_name: str) -> list[str]:
        ro, _ = _interp.normalize_read_paths(self.read_only_paths)
        return _docker.build_docker_command(
            image=self.docker_image, work_dir=work_dir, code=code,
            container_name=container_name, read_only_paths=ro,
            memory_mb=self.memory_mb, cpus=1.0, pids_limit=self.max_processes,
            cpu_seconds=self.cpu_seconds, max_file_mb=self.max_file_mb,
            max_open_files=self.max_open_files, allow_network=self.allow_network,
            env={"HOME": "/tmp", "TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **self.extra_env},
            python=self.docker_python, python_flags=_PYTHON_FLAGS,
            user=self.docker_user, docker_bin=self.docker_bin,
        )

    # ── execution ──────────────────────────────────────────────────────

    async def _docker_kill(self, name: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_bin, "kill", name,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=15)
        except Exception:  # pragma: no cover - best effort
            pass

    def _open_persist(self, code: str, notes: list[str]):
        """Create the unit's persist directory and stream sinks (or nothing)."""
        if not self.persist_dir:
            return None, (None, None)
        try:
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            unit_dir = os.path.join(self.persist_dir, f"{stamp}_{uuid.uuid4().hex[:8]}")
            os.makedirs(unit_dir, exist_ok=False)
            with open(os.path.join(unit_dir, "code.py"), "w", encoding="utf-8") as fh:
                fh.write(code)
            sinks = (open(os.path.join(unit_dir, "stdout.txt"), "wb"),
                     open(os.path.join(unit_dir, "stderr.txt"), "wb"))
            return unit_dir, sinks
        except OSError as exc:
            notes.append(f"complete outputs not persisted: {exc}")
            return None, (None, None)

    async def _finish_persist(self, unit_dir: str, sinks, streams, work: str,
                              notes: list[str]) -> None:
        for fh in sinks:
            try:
                if fh is not None:
                    fh.close()
            except OSError:  # pragma: no cover
                pass
        for name, stream in zip(("stdout", "stderr"), streams):
            if stream.sink_truncated:
                notes.append(f"persisted {name} cut at {stream.sink_written} bytes")
            if stream.sink_error:
                notes.append(f"persisted {name} incomplete: {stream.sink_error}")
        try:
            n, nbytes, pnotes = await asyncio.to_thread(
                _capture.persist_files, work, os.path.join(unit_dir, "work"),
                max_total_bytes=self.persist_max_bytes)
            notes += pnotes
        except Exception as exc:  # pragma: no cover - never fail the unit over a copy
            notes.append(f"written files not persisted: {exc}")

    async def run_block(
        self,
        index: int,
        code: str,
        *,
        block_indices: Optional[Sequence[int]] = None,
        spans: Optional[Sequence[tuple[int, int, int]]] = None,
    ) -> CodeBlockResult:
        """Run ``code`` as one execution unit and return its result."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        indices = list(block_indices) if block_indices is not None else [index]
        base = dict(index=index, block_indices=indices, backend=self.backend)

        ok, reason = await self.check_available()
        if not ok:
            return CodeBlockResult(success=False, exit_code=-1,
                                   stderr=f"backend unavailable: {reason}",
                                   error="backend unavailable",
                                   duration_seconds=loop.time() - start, **base)
        plan = self._limit_plan()
        nproc_applied = any(r.name == "RLIMIT_NPROC" for r in plan.rlimits)
        notes: list[str] = []
        container: Optional[str] = None

        with tempfile.TemporaryDirectory(prefix="miw_exec_", dir=self.workdir_parent,
                                         ignore_cleanup_errors=True) as tmp:
            root = os.path.realpath(tmp)
            work = os.path.join(root, "work")
            home = os.path.join(root, "home")
            tmpdir = os.path.join(root, "tmp")
            for d in (work, home, tmpdir):
                os.mkdir(d, 0o700)

            try:
                if self.backend == "docker":
                    container = f"miw_exec_{uuid.uuid4().hex[:12]}"
                    argv = self.build_docker_command(code, work, container)
                    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
                    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"):
                        if key in os.environ:
                            env[key] = os.environ[key]
                    env["HOME"] = os.environ.get("HOME", home)
                else:
                    python = _interp.resolve_python(self.python)
                    assert python is not None  # checked in check_available
                    argv = self._local_command(python, code, plan)
                    env = self._env(python, home, tmpdir)
                    if self.backend == "sandbox_exec":
                        reads, ro_notes = await self._sandbox_read_paths(python)
                        notes += ro_notes
                        argv = _seatbelt.build_sandbox_command(
                            work_root=root, read_paths=reads,
                            allow_network=self.allow_network, command=argv)
            except (RuntimeError, ValueError, OSError) as exc:
                return CodeBlockResult(success=False, exit_code=-1,
                                       stderr=f"could not prepare execution: {exc}",
                                       error="preparation failed",
                                       duration_seconds=loop.time() - start, **base)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=work,
                    env=env,
                    start_new_session=(os.name == "posix"),
                )
            except Exception as exc:  # spawn failure
                return CodeBlockResult(success=False, exit_code=-1,
                                       stderr=f"spawn failed: {exc}", error="spawn failed",
                                       duration_seconds=loop.time() - start, **base)

            unit_dir, sinks = self._open_persist(code, notes)
            out = _capture.CappedStream(self.output_limit_bytes,
                                        head=self.output_limit_bytes // 2,
                                        sink=sinks[0], sink_limit=self.persist_max_bytes)
            err = _capture.CappedStream(
                self.output_limit_bytes,
                head=min(_STDERR_HEAD_BYTES, self.output_limit_bytes // 4),
                sink=sinks[1], sink_limit=self.persist_max_bytes)
            readers = [asyncio.ensure_future(out.drain(proc.stdout)),
                       asyncio.ensure_future(err.drain(proc.stderr))]
            timed_out = False
            try:
                timed_out = not await _wait_for_exit(proc, self.timeout_seconds)
            finally:
                # Always runs, including on cancellation: kill the container
                # (docker) and every process left in the child's group.
                if container and (timed_out or proc.returncode is None):
                    await self._docker_kill(container)
                _kill_process_group(proc.pid)
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                await _wait_for_exit(proc, 10)
                done, pending = await asyncio.wait(readers, timeout=_DRAIN_GRACE_SECONDS)
                if pending:
                    for task in pending:
                        task.cancel()
                    transport = getattr(proc, "_transport", None)
                    if transport is not None:
                        try:
                            transport.close()
                        except Exception:  # pragma: no cover
                            pass
                    notes.append("a descendant process outside the process group "
                                 "(e.g. after setsid()) kept stdout/stderr open; "
                                 "output may be incomplete and that process may "
                                 "still be running")

            exit_code = proc.returncode if proc.returncode is not None else -1
            if unit_dir is not None:
                await self._finish_persist(unit_dir, sinks, (out, err), work, notes)
            files: list[CapturedFile] = []
            if self.capture_files:
                try:
                    files, fnotes = await asyncio.to_thread(
                        _capture.capture_files, work,
                        max_files=self.capture_max_files,
                        max_text_bytes=self.capture_text_bytes,
                        max_total_text_bytes=self.capture_total_bytes,
                        priority=_CAPTURE_PRIORITY)
                    notes += fnotes
                except Exception as exc:  # pragma: no cover - defensive
                    notes.append(f"file capture failed: {exc}")

        if os.path.exists(root):
            # TemporaryDirectory's cleanup (shutil.rmtree) can fail, e.g. on a
            # very deep tree; never leave the tree behind silently.
            removed = await asyncio.to_thread(_force_remove, root)
            notes.append("work directory cleanup needed a forced removal"
                         + ("" if removed else f" and failed: {root} may remain"))

        stdout, stdout_trunc = out.result()
        stderr, stderr_trunc = err.result()
        # Launcher notes are written before the user code starts, so only the
        # leading lines of stderr are accepted (user code cannot add notes).
        launcher_failed = False
        for line in err.head_text().splitlines():
            if not line.startswith(_limits.LAUNCHER_NOTE_PREFIX):
                break
            notes.append(line.strip())
            launcher_failed = True
        launcher_failed = launcher_failed and exit_code == _limits.LAUNCHER_FAIL_EXIT
        diag = err.diag_text()
        if timed_out:
            msg = f"process group killed after {self.timeout_seconds:g}s wall-clock timeout"
            stderr = stderr + ("\n" if stderr and not stderr.endswith("\n") else "") + msg
            diag = diag + ("\n" if diag and not diag.endswith("\n") else "") + msg
        sig, killed, limit_hit = _classify(exit_code, timed_out, diag, self.backend, nproc_applied)
        if self.max_file_mb:
            cap_bytes = max(1, int(float(self.max_file_mb) * _limits.MB))
            at_cap = [f.path for f in files if f.skipped is None or f.skipped in
                      ("binary", "text capture budget exhausted")
                      if f.size_bytes >= cap_bytes]
            if at_cap:
                notes.append(f"file(s) reached the {self.max_file_mb:g} MB file-size "
                             f"limit and are probably truncated: {', '.join(at_cap)}")
                if limit_hit is None:
                    limit_hit = "file_size"
        success = (not timed_out and exit_code == 0)
        error_block = None
        if not success and not launcher_failed:
            error_block = _error_block(diag, spans) if spans else (indices[0] if len(indices) == 1 else None)
        return CodeBlockResult(
            success=success,
            exit_code=exit_code,
            error=("resource limits could not be applied (code not run)"
                   if launcher_failed else None),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=loop.time() - start,
            signal=sig,
            killed_by_limit=killed,
            limit_hit=limit_hit,
            stdout_bytes=out.total,
            stderr_bytes=err.total,
            stdout_truncated=stdout_trunc,
            stderr_truncated=stderr_trunc,
            error_block=error_block,
            files=files,
            notes=notes,
            persisted_to=unit_dir,
            **base,
        )

    async def run_artifact(self, text: str, *, block_mode: Optional[str] = None) -> CodeExecutionReport:
        """Execute the python blocks of an artifact.

        ``block_mode`` overrides the executor's mode for this call
        (``"single"``: one combined script; ``"separate"``: one process per
        block, in order).
        """
        mode = self.block_mode
        if block_mode is not None:
            mode = _BLOCK_MODE_ALIASES.get(str(block_mode).strip().lower(), "")
            if not mode:
                raise ValueError(f"block_mode must be one of {BLOCK_MODES}, got {block_mode!r}")
        blocks = extract_code_blocks(text)
        report = CodeExecutionReport(
            mode=mode, backend=self.backend, code_blocks=len(blocks),
            limits=self.describe_limits(), guarantees=self.guarantees(),
        )
        if not blocks:
            return report
        if self.backend == "subprocess" and self.read_only_paths:
            report.notes.append("read_only_paths is not enforced by the subprocess "
                                "backend (no filesystem confinement)")
        if mode == "single":
            script, spans = combine_blocks(blocks)
            report.blocks.append(await self.run_block(
                1, script, block_indices=list(range(1, len(blocks) + 1)), spans=spans))
        else:
            for i, code in enumerate(blocks, start=1):
                report.blocks.append(await self.run_block(i, textwrap.dedent(code)))
        return report
