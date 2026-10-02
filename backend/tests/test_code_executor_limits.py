"""Live tests of the subprocess backend's guarantees.

Covers process-group kill (grandchildren), rlimits (CPU, file size, open
files, processes, memory where enforceable), byte-accurate output
truncation, the scrubbed environment, closed stdin and safe file capture.
Everything here runs on the default ``subprocess`` backend, and the same
mechanisms are reused by ``sandbox_exec``.
"""
import asyncio
import json
import os
import re
import signal
import sys
import textwrap
import time

import pytest

from backend.orchestrator.code_executor import CodeExecutor
from backend.orchestrator.sandbox import capture as cap
from backend.orchestrator.sandbox import limits as lim

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups/rlimits")
pytestmark = posix_only


def run(coro):
    return asyncio.run(coro)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - pid reused by another user
        return False
    return True


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


# ── Process-group kill ──────────────────────────────────────────────────


def test_timeout_kills_grandchildren():
    code = textwrap.dedent("""
        import subprocess, sys, time
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print("GRANDCHILD", p.pid, flush=True)
        time.sleep(120)
    """)
    res = run(CodeExecutor(timeout_seconds=6).run_block(1, code))
    assert res.timed_out and not res.success
    m = re.search(r"GRANDCHILD (\d+)", res.stdout)
    assert m, res.stdout + res.stderr
    pid = int(m.group(1))
    try:
        assert _wait_gone(pid), "grandchild survived the timeout"
    finally:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
    assert res.duration_seconds < 6 + 8


def test_normal_exit_kills_leftover_background_children():
    code = textwrap.dedent("""
        import subprocess, sys
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print("BACKGROUND", p.pid, flush=True)
    """)
    res = run(CodeExecutor(timeout_seconds=20).run_block(1, code))
    assert res.success, res.stderr
    pid = int(re.search(r"BACKGROUND (\d+)", res.stdout).group(1))
    try:
        assert _wait_gone(pid), "background child survived after the script exited"
    finally:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
    assert res.duration_seconds < 15


def test_setsid_escapee_is_reported_not_hung():
    """Documented limitation: a descendant that calls setsid() leaves the
    process group.  The executor must not hang on its inherited pipe; it
    reports the situation.  (RLIMIT_CPU still bounds the escapee.)"""
    code = textwrap.dedent("""
        import subprocess, sys
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             start_new_session=True)
        print("ESCAPED", p.pid, flush=True)
    """)
    start = time.monotonic()
    res = run(CodeExecutor(timeout_seconds=20).run_block(1, code))
    elapsed = time.monotonic() - start
    m = re.search(r"ESCAPED (\d+)", res.stdout)
    assert m, res.stdout + res.stderr
    pid = int(m.group(1))
    try:
        assert res.success
        assert elapsed < 15
        assert any("outside the process group" in n for n in res.notes)
    finally:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


# ── rlimits ─────────────────────────────────────────────────────────────


def test_cpu_limit_kills_busy_loop():
    ex = CodeExecutor(timeout_seconds=30, cpu_seconds=1)
    res = run(ex.run_block(1, "while True:\n    pass"))
    assert not res.success and not res.timed_out
    assert res.killed_by_limit and res.limit_hit == "cpu"
    assert res.signal == signal.SIGXCPU
    assert res.duration_seconds < 15
    assert "KILLED BY LIMIT (cpu" in _feedback(ex, res)


def _feedback(ex, res):
    from backend.orchestrator.code_executor import CodeExecutionReport
    return CodeExecutionReport(blocks=[res], mode="single", backend=ex.backend,
                               limits=ex.describe_limits()).to_feedback()


def test_cpu_limit_applies_to_orphaned_descendants_rlimit_inherited():
    code = "import resource; print(resource.getrlimit(resource.RLIMIT_CPU))"
    res = run(CodeExecutor(timeout_seconds=10, cpu_seconds=7).run_block(1, code))
    assert res.success and "(7, 8)" in res.stdout


def test_limits_cannot_be_raised_by_code():
    code = textwrap.dedent("""
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (10**6, 10**6))
            print("RAISED")
        except (ValueError, OSError):
            print("DENIED")
    """)
    res = run(CodeExecutor(timeout_seconds=10, cpu_seconds=5).run_block(1, code))
    assert res.success and "DENIED" in res.stdout


def test_file_size_limit():
    code = textwrap.dedent("""
        with open("big.bin", "wb") as fh:
            fh.write(b"x" * (2 * 1024 * 1024))
    """)
    res = run(CodeExecutor(timeout_seconds=10, max_file_mb=1).run_block(1, code))
    assert not res.success
    assert res.limit_hit == "file_size"
    assert "File too large" in res.stderr
    sizes = {f.path: f.size_bytes for f in res.files}
    assert sizes.get("big.bin") == 1024 * 1024


def test_file_size_limit_detected_even_when_error_is_swallowed():
    # Without a context manager CPython reports the EFBIG only at interpreter
    # shutdown (and may swallow it); the executor still flags the capped file.
    code = "open('big.bin', 'wb').write(b'x' * (2 * 1024 * 1024))"
    res = run(CodeExecutor(timeout_seconds=10, max_file_mb=1).run_block(1, code))
    assert res.limit_hit == "file_size"
    assert any("file-size limit" in n and "big.bin" in n for n in res.notes)


def test_open_files_limit():
    code = "fhs = [open('/dev/null') for _ in range(200)]"
    res = run(CodeExecutor(timeout_seconds=10, max_open_files=32).run_block(1, code))
    assert not res.success and res.limit_hit == "open_files"
    assert "Too many open files" in res.stderr


def test_process_limit_headroom():
    code = textwrap.dedent("""
        import os, time
        pids = []
        try:
            for _ in range(300):
                pid = os.fork()
                if pid == 0:
                    time.sleep(30)
                    os._exit(0)
                pids.append(pid)
            print("FORKED_ALL", len(pids))
        except BlockingIOError:
            print("FORK_FAILED_AFTER", len(pids), flush=True)
            raise
    """)
    ex = CodeExecutor(timeout_seconds=30, max_processes=5)
    res = run(ex.run_block(1, code))
    if lim.count_user_processes() is None:  # pragma: no cover
        pytest.skip("cannot count user processes on this platform")
    assert "FORK_FAILED_AFTER" in res.stdout, res.stdout + res.stderr
    n = int(re.search(r"FORK_FAILED_AFTER (\d+)", res.stdout).group(1))
    assert n < 300
    assert res.limit_hit == "processes"
    assert "max_processes" in ex.describe_limits()["enforced"]


def test_nproc_not_applied_by_default():
    plan = CodeExecutor()._limit_plan()
    assert "RLIMIT_NPROC" not in {r.name for r in plan.rlimits}
    assert "max_processes" in plan.not_enforced


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS enforced on Linux only")
def test_memory_limit_linux():  # pragma: no cover - not run on macOS
    code = "b = bytearray(1024 * 1024 * 1024)"
    res = run(CodeExecutor(timeout_seconds=20, memory_mb=256).run_block(1, code))
    assert not res.success and res.limit_hit == "memory"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific statement")
def test_memory_limit_reported_as_not_enforced_on_macos():
    ex = CodeExecutor(memory_mb=512)
    limits = ex.describe_limits()
    assert "memory_mb" in limits["not_enforced"]
    assert "RLIMIT_AS" not in {r.name for r in ex._limit_plan().rlimits}
    report = run(ex.run_artifact("```python\nprint(1)\n```"))
    assert "memory_mb: not enforced" in report.to_feedback()


def test_plan_limits_platform_table():
    linux = lim.plan_limits(cpu_seconds=10, memory_mb=100, max_file_mb=2,
                            max_processes=4, max_open_files=64, platform="linux",
                            user_process_count=50)
    names = {r.name: (r.soft, r.hard) for r in linux.rlimits}
    assert names["RLIMIT_CPU"] == (10, 11)
    assert names["RLIMIT_AS"] == (100 * lim.MB, 100 * lim.MB)
    assert names["RLIMIT_FSIZE"] == (2 * lim.MB, 2 * lim.MB)
    assert names["RLIMIT_NOFILE"] == (64, 64)
    assert names["RLIMIT_NPROC"] == (54, 54)
    assert names["RLIMIT_CORE"] == (0, 0)
    darwin = lim.plan_limits(cpu_seconds=10, memory_mb=100, max_file_mb=2,
                             max_processes=None, max_open_files=64, platform="darwin")
    assert "RLIMIT_AS" not in {r.name for r in darwin.rlimits}
    assert "memory_mb" in darwin.not_enforced


# ── Output ──────────────────────────────────────────────────────────────


def test_output_truncated_in_bytes_with_marker():
    code = ("import sys; sys.stdout.write('é' * 1000 + 'END'); "
            "sys.stderr.write('x' * 5000 + 'LAST')")
    res = run(CodeExecutor(timeout_seconds=10, output_limit_bytes=101).run_block(1, code))
    assert res.success
    assert res.stdout_bytes == 2003 and res.stdout_truncated
    # stdout keeps its first and last halves (cut on character boundaries)
    head, rest = res.stdout.split("\n...[truncated:", 1)
    marker, tail = rest.split("]...\n", 1)
    assert head == "é" * 25  # 50 bytes, no U+FFFD
    assert tail.endswith("END") and "\ufffd" not in tail
    assert len(head.encode()) + len(tail.encode()) <= 101
    assert "of 2003 bytes" in marker and "omitted" in marker
    # stderr keeps a short head and mostly its TAIL (the end of a traceback)
    assert res.stderr_bytes == 5004 and res.stderr_truncated
    assert res.stderr.endswith("LAST")
    assert "of 5004 bytes" in res.stderr


def test_output_flood_is_bounded():
    code = "import sys\nfor _ in range(320):\n    sys.stdout.write('y' * 65536)"
    res = run(CodeExecutor(timeout_seconds=30, output_limit_bytes=1000).run_block(1, code))
    assert res.success
    assert res.stdout_bytes == 320 * 65536
    assert len(res.stdout) < 1100


def test_untruncated_output_has_no_marker():
    res = run(CodeExecutor(timeout_seconds=10).run_block(1, "print('short')"))
    assert res.stdout == "short\n" and not res.stdout_truncated


def test_truncate_helpers():
    data = "aéb".encode()  # 61 c3 a9 62
    assert cap.utf8_prefix(data, 2) == b"a"
    assert cap.utf8_prefix(data, 3) == "aé".encode()
    text, trunc = cap.truncate_output(data, len(data), 10)
    assert text == "aéb" and not trunc
    stream = cap.CappedStream(3)
    stream.feed(b"abcdef")
    text, trunc = stream.result()
    assert trunc and text.startswith("abc") and "kept 3 of 6 bytes" in text


# ── Environment, stdin, working directory ───────────────────────────────


ALLOWED_ENV = {
    "PATH", "HOME", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
    "MPLBACKEND", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    # Added by CPython itself (PEP 538 locale coercion) or by macOS.
    "LC_CTYPE", "__CF_USER_TEXT_ENCODING",
}


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("MIW_TEST_SECRET_TOKEN", "s3cr3t-value")
    monkeypatch.setenv("PYTHONPATH", "/should/not/matter")
    code = "import os, json, sys; print(json.dumps({'env': dict(os.environ), 'cwd': os.getcwd(), 'path': sys.path}))"
    res = run(CodeExecutor(timeout_seconds=10).run_block(1, code))
    assert res.success, res.stderr
    info = json.loads(res.stdout)
    env = info["env"]
    assert "MIW_TEST_SECRET_TOKEN" not in env
    assert "s3cr3t-value" not in res.stdout
    assert set(env) <= ALLOWED_ENV, set(env) - ALLOWED_ENV
    assert env["MPLBACKEND"] == "Agg" and env["OMP_NUM_THREADS"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["PYTHONUNBUFFERED"] == "1"
    root = os.path.dirname(info["cwd"])
    assert os.path.basename(info["cwd"]) == "work"
    assert env["HOME"] == os.path.join(root, "home")
    assert env["TMPDIR"] == os.path.join(root, "tmp")
    assert "/should/not/matter" not in info["path"]
    assert not os.path.exists(root), "temporary root not deleted"


def test_extra_env_is_passed():
    ex = CodeExecutor(timeout_seconds=10, extra_env={"MIW_DATA_KIT": "/data/kit"})
    res = run(ex.run_block(1, "import os; print(os.environ['MIW_DATA_KIT'])"))
    assert res.success and res.stdout.strip() == "/data/kit"


def test_stdin_is_devnull():
    code = textwrap.dedent("""
        import os, sys
        data = sys.stdin.read()
        print(repr(data), os.path.samestat(os.fstat(0), os.stat(os.devnull)))
    """)
    res = run(CodeExecutor(timeout_seconds=10).run_block(1, code))
    assert res.success and res.stdout.strip() == "'' True"


def test_interpreter_flags_isolated_and_no_bytecode():
    code = "import sys; print(sys.flags.isolated, sys.flags.dont_write_bytecode, sys.flags.ignore_environment)"
    res = run(CodeExecutor(timeout_seconds=10).run_block(1, code))
    assert res.success and res.stdout.split() == ["1", "1", "1"]


def test_configurable_interpreter(tmp_path):
    res = run(CodeExecutor(timeout_seconds=10, python=sys.executable).run_block(
        1, "import sys; print(sys.executable)"))
    assert res.success
    assert os.path.realpath(res.stdout.strip()) == os.path.realpath(sys.executable)


# ── File capture ────────────────────────────────────────────────────────


def test_written_files_are_captured():
    code = textwrap.dedent("""
        import json, os
        json.dump({"auroc": 0.71, "n": 3}, open("results.json", "w"))
        os.makedirs("figs", exist_ok=True)
        open("figs/plot.png", "wb").write(b"\\x89PNG\\r\\n\\x1a\\n" + bytes(200))
        open("long.txt", "w").write("z" * 10000)
    """)
    ex = CodeExecutor(timeout_seconds=10, capture_text_bytes=1000)
    res = run(ex.run_block(1, code))
    assert res.success, res.stderr
    files = {f.path: f for f in res.files}
    assert json.loads(files["results.json"].content) == {"auroc": 0.71, "n": 3}
    assert files["figs/plot.png"].skipped == "binary" and files["figs/plot.png"].content is None
    assert files["figs/plot.png"].size_bytes == 208
    assert files["long.txt"].truncated and len(files["long.txt"].content) == 1000


def test_symlink_out_of_workdir_is_not_followed(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    code = f"import os; os.symlink({str(secret)!r}, 'leak.txt'); os.symlink({str(tmp_path)!r}, 'leakdir')"
    res = run(CodeExecutor(timeout_seconds=10).run_block(1, code))
    if not res.success and "Operation not supported" in res.stderr:  # pragma: no cover
        pytest.skip("filesystem without symlinks")
    assert res.success, res.stderr
    files = {f.path: f for f in res.files}
    assert files["leak.txt"].skipped.startswith("symlink") and files["leak.txt"].content is None
    assert files["leakdir"].skipped.startswith("symlink")
    assert "TOP-SECRET-CONTENT" not in json.dumps([f.to_dict() for f in res.files])


def test_hard_links_are_not_read():
    code = textwrap.dedent("""
        import os
        open("a.txt", "w").write("hard-linked content")
        try:
            os.link("a.txt", "b.txt")
        except OSError:
            print("LINK_UNSUPPORTED")
    """)
    res = run(CodeExecutor(timeout_seconds=10).run_block(1, code))
    if "LINK_UNSUPPORTED" in res.stdout:  # e.g. exFAT volumes
        pytest.skip("filesystem without hard links")
    files = {f.path: f for f in res.files}
    assert files["a.txt"].skipped == "hard link (not read)"
    assert files["b.txt"].content is None


def test_capture_skips_fifo_without_blocking(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "ok.txt").write_text("fine")
    try:
        os.mkfifo(work / "pipe")
    except OSError:  # pragma: no cover
        pytest.skip("filesystem without FIFOs")
    files, notes = cap.capture_files(str(work))
    by = {f.path: f for f in files}
    assert by["pipe"].skipped == "not a regular file"
    assert by["ok.txt"].content == "fine"


def test_capture_limits_file_count(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i:02d}.txt").write_text(str(i))
    files, notes = cap.capture_files(str(tmp_path), max_files=3)
    assert len(files) == 3 and any("more than 3 files" in n for n in notes)


def test_capture_ignores_appledouble(tmp_path):
    (tmp_path / "r.json").write_text("{}")
    (tmp_path / "._r.json").write_bytes(b"\x00\x05\x16\x07")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    files, _ = cap.capture_files(str(tmp_path))
    assert [f.path for f in files] == ["r.json"]


def test_capture_disabled():
    ex = CodeExecutor(timeout_seconds=10, capture_files=False)
    res = run(ex.run_block(1, "open('x.txt', 'w').write('1')"))
    assert res.success and res.files == []
