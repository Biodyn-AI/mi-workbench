"""POSIX resource limits for code-execution child processes (stdlib only).

The limits are applied by a tiny *launcher* that runs in the child: the
launcher (executed by the target interpreter itself) calls ``setrlimit`` and
then ``execv``s the real ``python -I -c <code>`` command.  rlimits survive
``execv``, so the analysis code starts with the limits already in force and
cannot raise them (soft and hard limits are both lowered).  This avoids
``subprocess``'s ``preexec_fn``, which CPython documents as unsafe in a
multi-threaded parent (the backend runs threads for aiosqlite/uvicorn).

Platform notes (what the kernel actually enforces):

* ``RLIMIT_CPU`` - enforced on Linux and macOS: SIGXCPU at the soft limit,
  SIGKILL at the hard limit (soft + 1 s).
* ``RLIMIT_FSIZE`` - enforced on Linux and macOS.  CPython ignores SIGXFSZ,
  so an oversized write raises ``OSError(EFBIG, "File too large")``.
  Pipes (stdout/stderr) are not affected.
* ``RLIMIT_NOFILE`` - enforced on Linux and macOS (``EMFILE``).
* ``RLIMIT_AS`` - enforced on Linux only.  The macOS kernel rejects any value
  below the address space a process has already reserved (hundreds of GB,
  because of the dyld shared region) and does not enforce the limit for
  ``mmap``; memory is therefore NOT limited on macOS by this layer (use the
  docker backend for a hard memory cap).
* ``RLIMIT_NPROC`` - counts *all* processes of the real user ID (on Linux:
  all threads), not just descendants of the child.  A small absolute value
  would make every ``fork`` fail immediately on a desktop session, so it is
  applied only when ``max_processes`` is set, and then as *headroom*:
  ``limit = current processes (threads on Linux) of the user + max_processes``.
  The count is taken just before launch, so it is approximate when other
  processes of the same user start or exit concurrently.
* ``RLIMIT_CORE`` is always set to 0 (no core dumps from killed children).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# POSIX only; the module is used by the launcher inside the child, not here.
HAS_RESOURCE = importlib.util.find_spec("resource") is not None

MB = 1024 * 1024

# Source of the launcher executed as ``<python> -I -c LAUNCHER_SOURCE <json> <argv...>``.
# It must stay compatible with any CPython >= 3.6 because it runs in the
# *target* interpreter (which may differ from the backend's).
# It fails CLOSED: if any planned limit cannot be applied it writes a note
# and exits with LAUNCHER_FAIL_EXIT without running the user code.
LAUNCHER_SOURCE = r"""
import json, os, sys
try:
    import resource
except ImportError:
    resource = None
spec = json.loads(sys.argv[1])
failed = False
if spec and resource is None:
    sys.stderr.write("[miw-sandbox] could not apply limits: no resource module\n")
    failed = True
elif resource is not None:
    for name, soft, hard in spec:
        res = getattr(resource, name, None)
        if res is None:
            sys.stderr.write("[miw-sandbox] could not apply %s: not supported\n" % name)
            failed = True
            continue
        try:
            cur_soft, cur_hard = resource.getrlimit(res)
            if cur_hard != resource.RLIM_INFINITY:
                hard = min(hard, cur_hard)
            soft = min(soft, hard)
            resource.setrlimit(res, (soft, hard))
        except (ValueError, OSError) as exc:
            sys.stderr.write("[miw-sandbox] could not apply %s: %s\n" % (name, exc))
            failed = True
if failed:
    sys.stderr.flush()
    os._exit(125)
argv = sys.argv[2:]
os.execv(argv[0], argv)
"""

LAUNCHER_NOTE_PREFIX = "[miw-sandbox] could not apply "
#: Exit status of the launcher when a limit could not be applied (code not run).
LAUNCHER_FAIL_EXIT = 125


@dataclass(frozen=True)
class RlimitEntry:
    """One ``setrlimit`` call: resource name (``RLIMIT_*``), soft, hard."""
    name: str
    soft: int
    hard: int

    def as_json(self) -> list:
        return [self.name, int(self.soft), int(self.hard)]


@dataclass
class LimitPlan:
    """Which limits will be applied and which cannot be on this platform."""
    rlimits: list[RlimitEntry] = field(default_factory=list)
    enforced: dict[str, str] = field(default_factory=dict)
    not_enforced: dict[str, str] = field(default_factory=dict)

    def launcher_spec(self) -> list[list]:
        return [r.as_json() for r in self.rlimits]


def _is_darwin(platform: str) -> bool:
    return platform == "darwin"


def _is_linux(platform: str) -> bool:
    return platform.startswith("linux")


def count_user_processes(platform: str = sys.platform) -> Optional[int]:
    """Number of processes (threads on Linux) of the current real user.

    This is the quantity ``RLIMIT_NPROC`` is checked against.  Returns
    ``None`` when it cannot be determined; the NPROC limit is then skipped.
    """
    try:
        if _is_darwin(platform):
            return _count_darwin_processes()
        if _is_linux(platform):
            return _count_linux_threads()
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _count_darwin_processes() -> Optional[int]:
    path = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
    lib = ctypes.CDLL(path, use_errno=True)
    lib.proc_listpids.restype = ctypes.c_int
    lib.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_int]
    PROC_RUID_ONLY = 5
    uid = os.getuid()
    needed = lib.proc_listpids(PROC_RUID_ONLY, uid, None, 0)
    if needed <= 0:
        return None
    n = needed // ctypes.sizeof(ctypes.c_int) + 256
    buf = (ctypes.c_int * n)()
    got = lib.proc_listpids(PROC_RUID_ONLY, uid, buf, ctypes.sizeof(buf))
    if got <= 0:
        return None
    return sum(1 for i in range(got // ctypes.sizeof(ctypes.c_int)) if buf[i] > 0)


def _count_linux_threads() -> Optional[int]:
    uid = os.getuid()
    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status", "r") as fh:
                real_uid = None
                threads = 1
                for line in fh:
                    if line.startswith("Uid:"):
                        real_uid = int(line.split()[1])
                    elif line.startswith("Threads:"):
                        threads = int(line.split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if real_uid == uid:
            total += threads
    return total or None


def plan_limits(
    *,
    cpu_seconds: Optional[float],
    memory_mb: Optional[int],
    max_file_mb: Optional[float],
    max_processes: Optional[int],
    max_open_files: Optional[int],
    platform: str = sys.platform,
    user_process_count: Optional[int] = None,
) -> LimitPlan:
    """Translate executor limits into rlimits for ``platform``.

    ``user_process_count`` is only consulted when ``max_processes`` is set;
    pass it explicitly in tests, otherwise it is measured.
    """
    plan = LimitPlan()
    if not HAS_RESOURCE:  # pragma: no cover - Windows
        for key in ("cpu_seconds", "memory_mb", "max_file_mb",
                    "max_processes", "max_open_files"):
            plan.not_enforced[key] = "no POSIX resource limits on this platform"
        return plan

    plan.rlimits.append(RlimitEntry("RLIMIT_CORE", 0, 0))

    if cpu_seconds:
        soft = max(1, int(math.ceil(cpu_seconds)))
        plan.rlimits.append(RlimitEntry("RLIMIT_CPU", soft, soft + 1))
        plan.enforced["cpu_seconds"] = (
            f"RLIMIT_CPU={soft}s (SIGXCPU; SIGKILL at {soft + 1}s)")

    if memory_mb:
        nbytes = int(memory_mb) * MB
        if _is_linux(platform):
            plan.rlimits.append(RlimitEntry("RLIMIT_AS", nbytes, nbytes))
            plan.enforced["memory_mb"] = f"RLIMIT_AS={int(memory_mb)} MB (virtual address space)"
        elif _is_darwin(platform):
            plan.not_enforced["memory_mb"] = (
                "not enforced: macOS rejects RLIMIT_AS/RLIMIT_DATA below the "
                "already-reserved address space and does not enforce them for "
                "mmap (use the docker backend for a hard memory cap)")
        else:
            plan.rlimits.append(RlimitEntry("RLIMIT_AS", nbytes, nbytes))
            plan.enforced["memory_mb"] = (
                f"RLIMIT_AS={int(memory_mb)} MB (best effort on {platform})")

    if max_file_mb:
        nbytes = max(1, int(float(max_file_mb) * MB))
        plan.rlimits.append(RlimitEntry("RLIMIT_FSIZE", nbytes, nbytes))
        plan.enforced["max_file_mb"] = f"RLIMIT_FSIZE={nbytes} bytes per file"

    if max_open_files:
        n = int(max_open_files)
        plan.rlimits.append(RlimitEntry("RLIMIT_NOFILE", n, n))
        plan.enforced["max_open_files"] = f"RLIMIT_NOFILE={n}"

    if max_processes:
        count = user_process_count
        if count is None:
            count = count_user_processes(platform)
        if count is None:
            plan.not_enforced["max_processes"] = (
                "not applied: could not count the user's processes, and an "
                "absolute RLIMIT_NPROC would count every process of the user")
        else:
            limit = int(count) + int(max_processes)
            plan.rlimits.append(RlimitEntry("RLIMIT_NPROC", limit, limit))
            unit = "threads" if _is_linux(platform) else "processes"
            plan.enforced["max_processes"] = (
                f"RLIMIT_NPROC={limit} (= {count} existing {unit} of the user "
                f"+ {int(max_processes)} headroom; approximate)")
    else:
        plan.not_enforced["max_processes"] = (
            "not applied (max_processes unset): RLIMIT_NPROC counts all "
            "processes of the user, not only the child's descendants")
    return plan
