#!/usr/bin/env python3
"""Codex CLI shim for the benchmark: adds extra ``-c`` overrides to ``codex exec``.

LEGACY: no longer used. The backend ``CodexCliAdapter`` now adds the tools-off
arguments itself (``common.make_adapter`` passes the benchmark's exact
``-c web_search="disabled"``), so the harness calls ``codex`` directly. The
shim is kept so older commands keep working; inserting the same override
twice is harmless (codex applies the last identical value).

The backend ``CodexCliAdapter`` builds the command; this shim (used as the
adapter's ``binary``) only inserts the arguments from the environment variable
``MIW_BENCH_CODEX_EXTRA_ARGS`` (a JSON list, e.g.
``["-c", "web_search=\\"disabled\\""]``) right after ``exec`` and then
``exec``s the real CLI (``MIW_BENCH_CODEX_REAL`` or ``codex`` on PATH), so the
process, its session/process group, stdin and stdout are the CLI's own.
Every other invocation (e.g. ``--version``) is passed through unchanged.
"""
import json
import os
import shutil
import sys


def main() -> None:
    real = os.environ.get("MIW_BENCH_CODEX_REAL") or shutil.which("codex")
    if not real:
        sys.stderr.write("codex_wrapper: real codex binary not found on PATH\n")
        sys.exit(127)
    extra = json.loads(os.environ.get("MIW_BENCH_CODEX_EXTRA_ARGS", "[]") or "[]")
    args = sys.argv[1:]
    if args and args[0] == "exec" and extra:
        args = ["exec", *[str(a) for a in extra], *args[1:]]
    os.execv(real, [real, *args])


if __name__ == "__main__":
    main()
