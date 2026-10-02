"""Docker backend: command construction and daemon/image checks.

Images are never pulled (``--pull never``); the image must exist locally.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from typing import Mapping, Optional, Sequence

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_PIDS_LIMIT = 128
CONTAINER_WORKDIR = "/work"
DEFAULT_TMPFS_MB = 256


def docker_binary(docker_bin: str = "docker") -> Optional[str]:
    return shutil.which(docker_bin)


def docker_available(docker_bin: str = "docker", image: Optional[str] = None,
                     timeout: float = 10.0) -> tuple[bool, str]:
    """(available, reason): CLI present, daemon reachable, image present."""
    exe = docker_binary(docker_bin)
    if not exe:
        return False, f"docker CLI '{docker_bin}' not found on PATH"
    try:
        info = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"],
                              capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker daemon not reachable: {exc}"
    if info.returncode != 0 or not info.stdout.strip():
        msg = (info.stderr or info.stdout).strip().splitlines()
        return False, "docker daemon not reachable: " + (msg[-1] if msg else "unknown error")
    if image:
        try:
            insp = subprocess.run([exe, "image", "inspect", "--format", "{{.Id}}", image],
                                  capture_output=True, text=True, timeout=timeout,
                                  stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"docker image check failed: {exc}"
        if insp.returncode != 0:
            return False, f"docker image '{image}' is not present locally (images are never pulled)"
    return True, "ok"


def _check_mount_path(path: str) -> str:
    if ":" in path or "," in path or "\x00" in path:
        raise ValueError(f"path cannot be bind-mounted with -v (contains ':' or ','): {path!r}")
    if not os.path.isabs(path):
        raise ValueError(f"bind-mount path must be absolute: {path!r}")
    return path


def build_docker_command(
    *,
    image: str,
    work_dir: str,
    code: str,
    container_name: str,
    read_only_paths: Sequence[str] = (),
    memory_mb: Optional[int] = None,
    cpus: float = 1.0,
    pids_limit: Optional[int] = None,
    cpu_seconds: Optional[float] = None,
    max_file_mb: Optional[float] = None,
    max_open_files: Optional[int] = None,
    allow_network: bool = False,
    env: Optional[Mapping[str, str]] = None,
    python: str = "python",
    python_flags: Sequence[str] = ("-I", "-B", "-u"),
    user: Optional[str] = None,
    docker_bin: str = "docker",
    tmpfs_mb: int = DEFAULT_TMPFS_MB,
) -> list[str]:
    """Build the ``docker run`` argv.

    Shape (spec order)::

        docker run --rm --network none --read-only --tmpfs /tmp
          -v <work_dir>:/work -w /work -v <ro>:<ro>:ro
          --memory <m>m --cpus 1 --pids-limit <n> <image> python -I -c <code>

    plus hardening flags (``--pull never``, ``--cap-drop ALL``,
    ``--security-opt no-new-privileges``, ``--memory-swap`` = ``--memory``,
    ``--ulimit`` cpu/fsize/nofile/core, ``--user``, ``--name``).
    """
    work_dir = _check_mount_path(work_dir)
    argv = [docker_bin, "run", "--rm", "--pull", "never",
            "--name", container_name,
            "--network", "bridge" if allow_network else "none",
            "--read-only",
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size={int(tmpfs_mb)}m",
            "-v", f"{work_dir}:{CONTAINER_WORKDIR}",
            "-w", CONTAINER_WORKDIR]
    for ro in read_only_paths:
        ro = _check_mount_path(ro)
        argv += ["-v", f"{ro}:{ro}:ro"]
    if memory_mb:
        argv += ["--memory", f"{int(memory_mb)}m", "--memory-swap", f"{int(memory_mb)}m"]
    argv += ["--cpus", f"{float(cpus):g}"]
    argv += ["--pids-limit", str(int(pids_limit or DEFAULT_PIDS_LIMIT))]
    argv += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
    argv += ["--ulimit", "core=0:0"]
    if cpu_seconds:
        soft = max(1, int(math.ceil(cpu_seconds)))
        argv += ["--ulimit", f"cpu={soft}:{soft + 1}"]
    if max_file_mb:
        nbytes = max(1, int(float(max_file_mb) * 1024 * 1024))
        argv += ["--ulimit", f"fsize={nbytes}:{nbytes}"]
    if max_open_files:
        argv += ["--ulimit", f"nofile={int(max_open_files)}:{int(max_open_files)}"]
    if user:
        argv += ["--user", user]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    argv += [image, python, *python_flags, "-c", code]
    return argv
