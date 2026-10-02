"""Interpreter resolution and the read paths an interpreter needs.

The ``sandbox_exec`` backend denies file reads outside an allow-list, so it
must know where the configured interpreter keeps its standard library and
site-packages.  The interpreter is asked once (outside any sandbox; it is an
operator-configured, trusted binary, not agent code) and the answer cached.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

_PROBE = (
    "import json, os, sys\n"
    "print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix, "
    "'base_prefix': getattr(sys, 'base_prefix', sys.prefix), "
    "'exec_prefix': sys.exec_prefix, "
    "'base_exec_prefix': getattr(sys, 'base_exec_prefix', sys.exec_prefix), "
    "'path': [p for p in sys.path if p]}))\n"
)

_cache: dict[tuple[str, float], list[str]] = {}
_lock = threading.Lock()


def resolve_python(python: Optional[str]) -> Optional[str]:
    """Absolute path of ``python`` (a path or a name looked up on PATH)."""
    if not python:
        return None
    if os.sep in python or (os.altsep and os.altsep in python):
        path = os.path.abspath(os.path.expanduser(python))
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    found = shutil.which(python)
    return os.path.abspath(found) if found else None


_BROAD_PATHS = (os.sep, "/Users", "/Volumes", "/home", "/private", "/private/var",
                "/private/tmp", "/tmp", "/var", "/private/var/folders", "/var/folders",
                "/mnt", "/media", "/srv", "/root")


def is_too_broad(path: str) -> bool:
    """Paths that would defeat read confinement if allow-listed wholesale:
    ``/``, user/volume/temp roots, a mounted volume's root
    (``/Volumes/<name>``, ``/media/<name>``), the per-user temp parents
    under ``/private/var/folders``, and the home directory or any ancestor of
    it."""
    home = os.path.realpath(os.path.expanduser("~"))
    p = os.path.normpath(path) if path else os.sep
    p = p.rstrip(os.sep) or os.sep
    if p in _BROAD_PATHS:
        return True
    parts = [x for x in p.split(os.sep) if x]
    if parts and parts[0] in ("Volumes", "media", "mnt") and len(parts) <= 2:
        return True  # a volume's root (all of an external drive)
    if p.startswith(("/private/var/folders/", "/var/folders/")):
        depth = len(parts) - (3 if parts[0] == "private" else 2)
        if depth <= 2:  # /private/var/folders/<xx>/<id> (the T/ dir is deeper)
            return True
    # The home directory itself or any ancestor of it.
    return home == p or home.startswith(p + os.sep)


# Backward-compatible private name.
_too_broad = is_too_broad


def check_read_paths(paths) -> None:
    """Raise ``ValueError`` naming the first entry that is too broad."""
    for p in paths:
        if is_too_broad(p):
            raise ValueError(
                f"read-only path {p!r} is too broad to allow-list (/, the home directory "
                "or one of its ancestors, a volume root or a shared temp root); give "
                "the specific data directory instead")


def normalize_read_paths(paths) -> tuple[list[str], list[str]]:
    """Absolute, symlink-resolved, de-duplicated existing paths + notes."""
    out: list[str] = []
    notes: list[str] = []
    for raw in paths or []:
        if not raw:
            continue
        real = os.path.realpath(os.path.expanduser(str(raw)))
        if not os.path.exists(real):
            notes.append(f"read-only path does not exist and was ignored: {raw}")
            continue
        if real not in out:
            out.append(real)
    return out, notes


def interpreter_read_paths(python: str, timeout: float = 30.0) -> list[str]:
    """Directories the interpreter reads at start-up and import time.

    Includes its prefixes and every existing ``sys.path`` entry (resolved),
    minus entries so broad that they would expose the user's files (``/``,
    the home directory or its ancestors).  Raises ``RuntimeError`` if the
    interpreter cannot be probed.
    """
    real = os.path.realpath(python)
    try:
        mtime = os.path.getmtime(real)
    except OSError as exc:
        raise RuntimeError(f"interpreter not found: {python}") from exc
    key = (real, mtime)
    with _lock:
        if key in _cache:
            return list(_cache[key])
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    try:
        proc = subprocess.run(
            [python, "-I", "-c", _PROBE], capture_output=True, text=True,
            timeout=timeout, env=env, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not probe interpreter {python}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not probe interpreter {python}: exit {proc.returncode}: "
            f"{proc.stderr.strip()[:500]}")
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    candidates = [
        info["prefix"], info["base_prefix"], info["exec_prefix"],
        info["base_exec_prefix"], str(Path(real).parent.parent),
        str(Path(os.path.realpath(info["executable"])).parent.parent),
        *info["path"],
    ]
    resolved, _ = normalize_read_paths(candidates)
    result = [p for p in resolved if not _too_broad(p)]
    with _lock:
        _cache[key] = list(result)
    return result
