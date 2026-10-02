"""macOS ``sandbox-exec`` (Seatbelt) command construction.

The profile template lives next to this module (``seatbelt_profile.sb``).
Paths are passed as ``-D`` parameters, never spliced into profile text.
Apple marks ``sandbox-exec`` as deprecated (``man sandbox-exec``); it is
still shipped and functional on current macOS, but the profile language is
undocumented and may change between releases, so the behaviour is verified
by live tests (``backend/tests/test_code_executor_sandbox.py``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
TEMPLATE_PATH = Path(__file__).with_name("seatbelt_profile.sb")

_NETWORK_MARK = ";;@@NETWORK_POLICY@@"
_NETWORK_MACH_MARK = ";;@@NETWORK_MACH_SERVICES@@"
_READ_MARK = ";;@@READ_ALLOWLIST@@"

_DENY_NETWORK = "(deny network*)"
_ALLOW_NETWORK = ";; allow_network=True: network access is permitted.\n(allow network*)"
# Extra Mach services needed for DNS resolution / TLS trust when networking
# is explicitly enabled.
_NETWORK_MACH_SERVICES = (
    "com.apple.dnssd.service",
    "com.apple.trustd",
    "com.apple.trustd.agent",
    "com.apple.SystemConfiguration.configd",
    "com.apple.SystemConfiguration.DNSConfiguration",
    "com.apple.networkd",
)


def sandbox_exec_available() -> tuple[bool, str]:
    """(available, reason).  Only macOS ships ``/usr/bin/sandbox-exec``."""
    if sys.platform != "darwin":
        return False, f"sandbox_exec backend requires macOS (platform is {sys.platform})"
    if not os.access(SANDBOX_EXEC, os.X_OK):
        return False, f"{SANDBOX_EXEC} not found or not executable"
    if not TEMPLATE_PATH.is_file():
        return False, f"Seatbelt profile template missing: {TEMPLATE_PATH}"
    return True, "ok"


def render_profile(n_read_paths: int, allow_network: bool) -> str:
    """Return the profile text for ``n_read_paths`` READ_<i> parameters."""
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    for mark in (_NETWORK_MARK, _NETWORK_MACH_MARK, _READ_MARK):
        if mark not in text:  # pragma: no cover - template corrupted
            raise RuntimeError(f"Seatbelt template lacks marker {mark}")
    text = text.replace(_NETWORK_MARK, _ALLOW_NETWORK if allow_network else _DENY_NETWORK)
    if allow_network:
        mach = "(allow mach-lookup\n" + "\n".join(
            f'  (global-name "{name}")' for name in _NETWORK_MACH_SERVICES) + ")"
    else:
        mach = ";; network disabled: no DNS/TLS Mach services"
    text = text.replace(_NETWORK_MACH_MARK, mach)
    if n_read_paths:
        rules = "(allow file-read*\n" + "\n".join(
            f'  (subpath (param "READ_{i}"))' for i in range(n_read_paths)) + ")"
    else:
        rules = ";; no additional read-only paths"
    text = text.replace(_READ_MARK, rules)
    return text


def _check_param_value(value: str) -> str:
    if "\x00" in value:
        raise ValueError("sandbox parameter contains a NUL byte")
    return value


def build_sandbox_command(
    *,
    work_root: str,
    read_paths: Iterable[str],
    allow_network: bool,
    command: Sequence[str],
) -> list[str]:
    """``sandbox-exec -p <profile> -D WORK_ROOT=... -D READ_i=... <command>``.

    All paths must already be absolute and symlink-resolved (Seatbelt
    matches on the resolved path, e.g. ``/private/var`` not ``/var``).
    """
    reads = [_check_param_value(p) for p in read_paths]
    profile = render_profile(len(reads), allow_network)
    argv = [SANDBOX_EXEC, "-p", profile,
            "-D", f"WORK_ROOT={_check_param_value(work_root)}"]
    for i, path in enumerate(reads):
        argv += ["-D", f"READ_{i}={path}"]
    argv += list(command)
    return argv
