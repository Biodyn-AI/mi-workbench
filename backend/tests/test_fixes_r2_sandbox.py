"""Regression tests: verified-execution sandbox (code-review findings
sandbox-1..5 and the known small issue: keep the tail of stderr)."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from backend.orchestrator.code_executor import CodeExecutor
from backend.orchestrator.sandbox import capture as cap
from backend.orchestrator.sandbox import interpreter as interp
from backend.orchestrator.sandbox import limits as lim


def run(coro):
    return asyncio.run(coro)


# ── sandbox-1: broad read_only_paths are rejected ─────────────────────


@pytest.mark.parametrize("path", ["/", "~", "/Users", "/private/tmp", "/tmp"])
def test_broad_read_only_paths_rejected(path):
    for backend in ("subprocess", "sandbox_exec", "docker"):
        with pytest.raises(ValueError, match="too broad"):
            CodeExecutor(backend=backend, read_only_paths=[path])


def test_home_ancestor_and_volume_root_rejected():
    home = os.path.realpath(os.path.expanduser("~"))
    assert interp.is_too_broad(os.path.dirname(home))
    assert interp.is_too_broad("/Volumes/SomeDrive")
    assert interp.is_too_broad("/private/var/folders/ab/cd")
    assert not interp.is_too_broad("/private/var/folders/ab/cd/T/data")
    assert not interp.is_too_broad("/Volumes/SomeDrive/project/data")


def test_specific_data_dir_accepted(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    ex = CodeExecutor(backend="subprocess", read_only_paths=[str(kit)])
    assert ex.read_only_paths == [str(kit)]


# ── sandbox-2 / sandbox-3: tail of stderr, classification on the tail ─


NOISE = "import sys\nfor i in range(300):\n    sys.stderr.write('warning line %d padding padding\\n' % i)\n"


def test_long_stderr_keeps_the_final_exception():
    art = (f"```python\n{NOISE}```\n"
           "```python\nraise ValueError('REAL-ERROR-AT-END')\n```\n")
    report = run(CodeExecutor(timeout_seconds=20).run_artifact(art))
    b = report.blocks[0]
    assert b.stderr_bytes > 9000 and b.stderr_truncated
    assert "ValueError: REAL-ERROR-AT-END" in b.stderr
    assert "REAL-ERROR-AT-END" in report.to_feedback()
    assert b.error_block == 2


def test_memory_error_after_long_stderr_classified():
    code = NOISE + "raise MemoryError()\n"
    res = run(CodeExecutor(timeout_seconds=20).run_block(1, code))
    assert res.limit_hit == "memory"


def test_emfile_after_long_stderr_classified():
    code = NOISE + "fs = [open('/dev/null') for _ in range(10000)]\n"
    res = run(CodeExecutor(timeout_seconds=20, max_open_files=64).run_block(1, code))
    assert res.limit_hit == "open_files"


def test_efbig_after_long_stderr_classified_without_file_capture():
    code = NOISE + "open('big.bin', 'wb').write(b'x' * (3 * 1024 * 1024))\n"
    res = run(CodeExecutor(timeout_seconds=20, max_file_mb=1, capture_files=False).run_block(1, code))
    assert res.limit_hit == "file_size"


# ── sandbox-4: deep trees do not exhaust descriptors ──────────────────


def test_capture_files_bounded_depth_under_low_nofile(tmp_path):
    work = tmp_path / "work"
    deep = work
    for _ in range(300):
        deep = deep / "d"
    os.makedirs(deep)
    (deep / "x.txt").write_text("hi")
    probe = textwrap.dedent(f"""
        import json, resource, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        from backend.orchestrator.sandbox.capture import capture_files
        files, notes = capture_files({str(work)!r})
        print(json.dumps(notes))
    """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    notes = json.loads(out.stdout.strip().splitlines()[-1])
    assert any("did not descend below depth" in n for n in notes)


def test_directories_count_towards_max_entries(tmp_path):
    for i in range(30):
        (tmp_path / f"dir{i}").mkdir()
    files, notes = cap.capture_files(str(tmp_path), max_entries=10)
    assert any("stopped after 10 entries" in n for n in notes)


# ── sandbox-5: launcher fails closed; notes cannot be spoofed ─────────


def test_launcher_fails_closed_on_unapplied_limit():
    spec = json.dumps([["RLIMIT_BOGUS", 1, 1]])
    out = subprocess.run([sys.executable, "-I", "-c", lim.LAUNCHER_SOURCE, spec,
                          sys.executable, "-c", "print('ran')"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == lim.LAUNCHER_FAIL_EXIT
    assert "ran" not in out.stdout
    assert out.stderr.startswith(lim.LAUNCHER_NOTE_PREFIX)


def test_unapplied_limit_reported_as_not_run(monkeypatch):
    ex = CodeExecutor(timeout_seconds=10)
    real = ex._limit_plan

    def plan():
        p = real()
        p.rlimits.append(lim.RlimitEntry("RLIMIT_BOGUS", 1, 1))
        return p

    monkeypatch.setattr(ex, "_limit_plan", plan)
    res = run(ex.run_block(1, "print('ran')"))
    assert res.success is False and "ran" not in res.stdout
    assert res.error == "resource limits could not be applied (code not run)"
    assert any("RLIMIT_BOGUS" in n for n in res.notes)


def test_user_code_cannot_add_launcher_notes():
    code = ("import sys\nsys.stderr.write('progress\\n')\n"
            "sys.stderr.write('[miw-sandbox] could not apply RLIMIT_CPU: spoofed\\n')\n")
    res = run(CodeExecutor(timeout_seconds=10).run_block(1, code))
    assert not any("spoofed" in n for n in res.notes)
