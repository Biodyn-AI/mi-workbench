"""Live tests of the OS-level confinement backends.

``sandbox_exec`` tests run only on macOS with ``/usr/bin/sandbox-exec``.
They check that the kernel enforces the profile: network denied, writes
confined, reads allow-listed, no signals to or inspection of outside
processes, Mach lookups restricted, and numpy works inside.  Each negative
result is checked against the unconfined ``subprocess`` backend, which shows
that the operation is possible without the sandbox.

``docker`` tests check command construction always.  The live test runs
only when the Docker daemon is up and the image is present locally, because
images are never pulled.
"""
import asyncio
import errno
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest

from backend.orchestrator.code_executor import CodeExecutor
from backend.orchestrator.sandbox import docker as dk
from backend.orchestrator.sandbox import interpreter as interp
from backend.orchestrator.sandbox import seatbelt as sb


def run(coro):
    return asyncio.run(coro)


_SB_OK, _SB_REASON = sb.sandbox_exec_available()
needs_sandbox = pytest.mark.skipif(not _SB_OK, reason=f"sandbox-exec unavailable: {_SB_REASON}")

ANACONDA_PY = "/Users/ihorkendiukhov/anaconda3/bin/python"


def _python_with_numpy():
    candidates = [os.environ.get("MIW_TEST_NUMPY_PYTHON"), ANACONDA_PY, sys.executable]
    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        try:
            ok = subprocess.run([cand, "-I", "-c", "import numpy"], capture_output=True,
                                timeout=60, stdin=subprocess.DEVNULL).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            return cand
    return None


def sandboxed(**kw):
    kw.setdefault("timeout_seconds", 20)
    return CodeExecutor(backend="sandbox_exec", **kw)


def unconfined(**kw):
    kw.setdefault("timeout_seconds", 20)
    return CodeExecutor(backend="subprocess", **kw)


# ── Profile rendering (any OS) ──────────────────────────────────────────


def test_profile_template_rendering():
    text = sb.render_profile(2, allow_network=False)
    assert "(deny network*)" in text and "(allow network*)" not in text
    assert '(subpath (param "READ_0"))' in text and '(subpath (param "READ_1"))' in text
    assert "(deny file-write*)" in text and '(subpath (param "WORK_ROOT"))' in text
    for marker in ("NETWORK_POLICY@@", "READ_ALLOWLIST@@", "NETWORK_MACH_SERVICES@@"):
        assert marker not in text
    assert '(deny sysctl-read (sysctl-name-prefix "kern.procargs"))' in text
    allowed = sb.render_profile(0, allow_network=True)
    assert "(allow network*)" in allowed and "com.apple.dnssd.service" in allowed
    assert "READ_0" not in allowed


def test_sandbox_command_passes_paths_as_parameters():
    weird = '/tmp/dir with "quotes") (allow default'
    argv = sb.build_sandbox_command(work_root="/private/tmp/w", read_paths=[weird],
                                    allow_network=False, command=["python", "-c", "1"])
    assert argv[0] == sb.SANDBOX_EXEC and argv[1] == "-p"
    profile = argv[2]
    assert weird not in profile  # never interpolated into profile text
    assert argv[3:7] == ["-D", "WORK_ROOT=/private/tmp/w", "-D", f"READ_0={weird}"]
    assert argv[-3:] == ["python", "-c", "1"]


# ── sandbox_exec: network ───────────────────────────────────────────────


NET_CODE = textwrap.dedent("""
    import errno, socket
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=5).close()
        print("NET_OK")
    except OSError as exc:
        print("NET_ERR", exc.errno, type(exc).__name__)
""")


@needs_sandbox
def test_sandbox_denies_internet_connect():
    res = run(sandboxed().run_block(1, NET_CODE))
    assert res.success, res.stderr
    # EPERM proves the sandbox refused it (not merely "no route to host").
    assert f"NET_ERR {errno.EPERM} PermissionError" in res.stdout, res.stdout


def _local_listener():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    return srv, srv.getsockname()[1]


def _connect_code(port):
    return textwrap.dedent(f"""
        import socket
        try:
            socket.create_connection(("127.0.0.1", {port}), timeout=5).close()
            print("CONNECTED")
        except OSError as exc:
            print("DENIED", exc.errno)
    """)


@needs_sandbox
def test_sandbox_denies_localhost_but_allow_network_permits_it():
    srv, port = _local_listener()
    try:
        denied = run(sandboxed().run_block(1, _connect_code(port)))
        assert f"DENIED {errno.EPERM}" in denied.stdout, denied.stdout + denied.stderr
        allowed = run(sandboxed(allow_network=True).run_block(1, _connect_code(port)))
        assert "CONNECTED" in allowed.stdout, allowed.stdout + allowed.stderr
        # Contrast: the subprocess backend does NOT confine the network.
        free = run(unconfined().run_block(1, _connect_code(port)))
        assert "CONNECTED" in free.stdout
    finally:
        srv.close()


@needs_sandbox
def test_sandbox_denies_unix_domain_sockets():
    # AF_UNIX paths are limited to ~104 bytes, so use a short directory.
    sock_dir = tempfile.mkdtemp(prefix="miw_s_", dir="/tmp")
    path = os.path.join(sock_dir, "s")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(path)
        srv.listen(4)
        code = textwrap.dedent(f"""
            import socket
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect({path!r})
                print("CONNECTED")
            except OSError as exc:
                print("DENIED", exc.errno)
        """)
        res = run(sandboxed().run_block(1, code))
        assert "DENIED" in res.stdout and "CONNECTED" not in res.stdout, res.stdout + res.stderr
        free = run(unconfined().run_block(1, code))
        assert "CONNECTED" in free.stdout
    finally:
        srv.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


# ── sandbox_exec: filesystem ────────────────────────────────────────────


@needs_sandbox
def test_sandbox_denies_writes_outside_temp_dir():
    outside = tempfile.mkdtemp(prefix="miw_outside_")
    victim = os.path.join(outside, "existing.txt")
    with open(victim, "w") as fh:
        fh.write("keep me")
    target = os.path.join(outside, "created.txt")
    code = textwrap.dedent(f"""
        import os
        for label, fn in [
            ("create", lambda: open({target!r}, "w").write("x")),
            ("append", lambda: open({victim!r}, "a").write("x")),
            ("delete", lambda: os.remove({victim!r})),
            ("mkdir", lambda: os.mkdir(os.path.join({outside!r}, "d"))),
        ]:
            try:
                fn()
                print(label, "ALLOWED")
            except PermissionError:
                print(label, "DENIED")
        open("inside.txt", "w").write("written inside")
        print("INSIDE_OK")
    """)
    try:
        res = run(sandboxed().run_block(1, code))
        assert res.success, res.stderr
        for label in ("create", "append", "delete", "mkdir"):
            assert f"{label} DENIED" in res.stdout, res.stdout
        assert "INSIDE_OK" in res.stdout
        assert not os.path.exists(target)
        assert open(victim).read() == "keep me"
        assert not os.path.exists(os.path.join(outside, "d"))
        assert {f.path: f.content for f in res.files}["inside.txt"] == "written inside"
    finally:
        shutil.rmtree(outside, ignore_errors=True)


@needs_sandbox
def test_sandbox_reads_read_only_paths_only():
    kit = tempfile.mkdtemp(prefix="miw_kit_")
    other = tempfile.mkdtemp(prefix="miw_private_")
    with open(os.path.join(kit, "data.tsv"), "w") as fh:
        fh.write("gene\tscore\nTP53\t0.9\n")
    with open(os.path.join(other, "secret.txt"), "w") as fh:
        fh.write("NOT-FOR-THE-SANDBOX")
    home = os.path.expanduser("~")
    code = textwrap.dedent(f"""
        import os
        print("KIT", open(os.path.join({kit!r}, "data.tsv")).read().split()[2])
        print("KIT_LIST", sorted(os.listdir({kit!r})))
        for label, fn in [
            ("other_read", lambda: open(os.path.join({other!r}, "secret.txt")).read()),
            ("other_list", lambda: os.listdir({other!r})),
            ("home_list", lambda: os.listdir({home!r})),
        ]:
            try:
                fn()
                print(label, "ALLOWED")
            except PermissionError:
                print(label, "DENIED")
        try:
            open(os.path.join({kit!r}, "data.tsv"), "a").write("tamper")
            print("kit_write ALLOWED")
        except PermissionError:
            print("kit_write DENIED")
    """)
    try:
        res = run(sandboxed(read_only_paths=[kit]).run_block(1, code))
        assert res.success, res.stderr
        assert "KIT TP53" in res.stdout
        assert re.search(r"KIT_LIST \[.*'data.tsv'.*\]", res.stdout)
        for label in ("other_read", "other_list", "home_list", "kit_write"):
            assert f"{label} DENIED" in res.stdout, res.stdout
        assert "NOT-FOR-THE-SANDBOX" not in res.stdout
        # Contrast: the subprocess backend can read the private file.
        free = run(unconfined().run_block(
            1, f"print(open({os.path.join(other, 'secret.txt')!r}).read())"))
        assert "NOT-FOR-THE-SANDBOX" in free.stdout
    finally:
        for d in (kit, other):
            shutil.rmtree(d, ignore_errors=True)


@needs_sandbox
def test_sandbox_numpy_import_with_configured_interpreter():
    py = _python_with_numpy()
    if py is None:
        pytest.skip("no interpreter with numpy available")
    code = textwrap.dedent("""
        import json, numpy as np
        x = np.arange(10, dtype=float)
        json.dump({"mean": float(x.mean()), "numpy": np.__version__}, open("results.json", "w"))
        print("NUMPY_OK", x.sum())
    """)
    res = run(sandboxed(python=py).run_block(1, code))
    assert res.success, res.stderr
    assert "NUMPY_OK 45.0" in res.stdout
    results = json.loads({f.path: f.content for f in res.files}["results.json"])
    assert results["mean"] == 4.5


# ── sandbox_exec: processes and IPC ─────────────────────────────────────


@needs_sandbox
def test_sandbox_cannot_signal_outside_processes():
    code = f"import os\ntry:\n    os.kill({os.getpid()}, 0)\n    print('SIGNAL_OK')\nexcept PermissionError:\n    print('SIGNAL_DENIED')"
    res = run(sandboxed().run_block(1, code))
    assert "SIGNAL_DENIED" in res.stdout, res.stdout + res.stderr
    free = run(unconfined().run_block(1, code))
    assert "SIGNAL_OK" in free.stdout


_PROCARGS_CODE = textwrap.dedent("""
    import ctypes, ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, {pid})  # CTL_KERN, KERN_PROCARGS2, pid
    size = ctypes.c_size_t(1 << 20)
    buf = ctypes.create_string_buffer(size.value)
    rc = libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0)
    data = buf.raw[: size.value] if rc == 0 else b""
    print("ENV_VISIBLE" if b"PATH=" in data or b"HOME=" in data else "ENV_HIDDEN")
""")


@needs_sandbox
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sysctl")
def test_sandbox_cannot_read_backend_environment():
    code = _PROCARGS_CODE.format(pid=os.getpid())
    free = run(unconfined().run_block(1, code))
    if "ENV_VISIBLE" not in free.stdout:
        pytest.skip("KERN_PROCARGS2 does not expose the environment here")
    res = run(sandboxed().run_block(1, code))
    assert res.success, res.stderr
    assert "ENV_HIDDEN" in res.stdout


_MACH_CODE = textwrap.dedent("""
    import ctypes, ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    bp = ctypes.c_uint.in_dll(libc, "bootstrap_port")
    port = ctypes.c_uint(0)
    kr = libc.bootstrap_look_up(bp, b"com.apple.coreservices.launchservicesd", ctypes.byref(port))
    print("LOOKUP", kr)
""")


@needs_sandbox
def test_sandbox_blocks_launchservices_lookup():
    free = run(unconfined().run_block(1, _MACH_CODE))
    if "LOOKUP 0" not in free.stdout:
        pytest.skip("launchservicesd lookup not available outside the sandbox")
    res = run(sandboxed().run_block(1, _MACH_CODE))
    assert res.success, res.stderr
    assert "LOOKUP 0" not in res.stdout and "LOOKUP" in res.stdout


@needs_sandbox
def test_sandbox_applies_rlimits_and_group_kill():
    busy = run(sandboxed(cpu_seconds=1, timeout_seconds=30).run_block(1, "while True: pass"))
    assert busy.killed_by_limit and busy.limit_hit == "cpu"
    code = textwrap.dedent("""
        import subprocess, sys, time
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print("GRANDCHILD", p.pid, flush=True)
        time.sleep(120)
    """)
    res = run(sandboxed(timeout_seconds=6).run_block(1, code))
    assert res.timed_out
    pid = int(re.search(r"GRANDCHILD (\d+)", res.stdout).group(1))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:  # pragma: no cover
        os.kill(pid, signal.SIGKILL)
        pytest.fail("grandchild survived under sandbox_exec")


@needs_sandbox
def test_sandbox_report_states_backend_and_guarantees():
    report = run(sandboxed().run_artifact("```python\nprint('hi')\n```"))
    fb = report.to_feedback()
    assert report.backend == "sandbox_exec" and report.passed == 1
    assert "Backend: sandbox_exec" in fb and "deny network*" in fb


@needs_sandbox
def test_sandbox_missing_read_only_path_is_noted(tmp_path):
    missing = str(tmp_path / "does-not-exist")
    res = run(sandboxed(read_only_paths=[missing]).run_block(1, "print(1)"))
    assert res.success
    assert any("does not exist" in n for n in res.notes)


def test_interpreter_read_paths_exclude_home_and_root():
    paths = interp.interpreter_read_paths(sys.executable)
    home = os.path.realpath(os.path.expanduser("~"))
    assert paths, "no interpreter paths found"
    assert "/" not in paths and home not in paths
    assert any(os.path.realpath(sys.prefix) == p or os.path.realpath(sys.prefix).startswith(p)
               for p in paths)


# ── docker ──────────────────────────────────────────────────────────────


def test_docker_command_construction(tmp_path):
    work = str(tmp_path / "work")
    kit = str(tmp_path / "kit")
    os.makedirs(work)
    os.makedirs(kit)
    argv = dk.build_docker_command(
        image="python:3.11-slim", work_dir=work, code="print(1)",
        container_name="miw_exec_test", read_only_paths=[kit], memory_mb=512,
        pids_limit=32, cpu_seconds=10, max_file_mb=8, max_open_files=128,
        env={"HOME": "/tmp"}, user="501:20",
    )
    joined = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--pull never" in joined
    assert "--network none" in joined
    assert "--read-only" in argv
    assert any(a.startswith("/tmp:rw") for a in argv) and "--tmpfs" in argv
    i = argv.index(f"{work}:/work")
    assert argv[i - 1] == "-v"
    assert argv[argv.index("-w") + 1] == "/work"
    j = argv.index(f"{kit}:{kit}:ro")
    assert argv[j - 1] == "-v"
    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--memory-swap") + 1] == "512m"
    assert argv[argv.index("--cpus") + 1] == "1"
    assert argv[argv.index("--pids-limit") + 1] == "32"
    assert "cpu=10:11" in argv and "fsize=8388608:8388608" in argv
    assert "nofile=128:128" in argv and "core=0:0" in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in argv
    assert argv[argv.index("--user") + 1] == "501:20"
    assert argv[argv.index("--name") + 1] == "miw_exec_test"
    assert "HOME=/tmp" in argv
    # image, then the interpreter command, last
    assert argv[-7:] == ["python:3.11-slim", "python", "-I", "-B", "-u", "-c", "print(1)"]


def test_docker_command_network_and_defaults(tmp_path):
    argv = dk.build_docker_command(image="img:1", work_dir=str(tmp_path), code="1",
                                   container_name="n", allow_network=True)
    assert argv[argv.index("--network") + 1] == "bridge"
    assert argv[argv.index("--pids-limit") + 1] == str(dk.DEFAULT_PIDS_LIMIT)
    assert "--memory" not in argv
    with pytest.raises(ValueError):
        dk.build_docker_command(image="i", work_dir="/a:b", code="1", container_name="n")
    with pytest.raises(ValueError):
        dk.build_docker_command(image="i", work_dir=str(tmp_path), code="1",
                                container_name="n", read_only_paths=["relative/path"])


def test_executor_builds_docker_command_from_config(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    ex = CodeExecutor.from_run_config({
        "code_execution_backend": "docker",
        "code_execution_memory_mb": 1024,
        "code_execution_max_processes": 64,
        "code_execution_read_only_paths": [str(kit)],
        "code_execution_docker_image": "python:3.12-slim",
        "code_execution_timeout": 15,
    })
    argv = ex.build_docker_command("print(2)", str(tmp_path), "c1")
    assert "python:3.12-slim" in argv
    assert argv[argv.index("--memory") + 1] == "1024m"
    assert argv[argv.index("--pids-limit") + 1] == "64"
    assert f"{os.path.realpath(kit)}:{os.path.realpath(kit)}:ro" in argv
    assert "cpu=30:31" in argv  # cpu_seconds defaults to 2 x timeout
    assert argv[-1] == "print(2)"
    assert ex.describe_limits()["enforced"]["memory_mb"].startswith("--memory 1024m")


_DOCKER_OK, _DOCKER_REASON = dk.docker_available(image=dk.DEFAULT_IMAGE, timeout=5)


@pytest.mark.skipif(not _DOCKER_OK, reason=f"docker unavailable: {_DOCKER_REASON}")
def test_docker_live_network_none_and_readonly():  # pragma: no cover - daemon not running in CI
    code = textwrap.dedent("""
        import socket
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=3); print("NET_OK")
        except OSError:
            print("NET_BLOCKED")
        try:
            open("/etc/miw_test", "w"); print("ROOT_WRITABLE")
        except OSError:
            print("ROOT_READONLY")
        open("results.json", "w").write('{"ok": true}')
    """)
    res = run(CodeExecutor(backend="docker", timeout_seconds=60).run_block(1, code))
    assert res.success, res.stderr
    assert "NET_BLOCKED" in res.stdout and "ROOT_READONLY" in res.stdout
    assert {f.path: f.content for f in res.files}["results.json"] == '{"ok": true}'
