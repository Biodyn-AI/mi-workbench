"""Tests for sandboxed code execution and its engine integration.

Backend-specific guarantees are tested in ``test_code_executor_limits.py``
(subprocess backend: rlimits, process-group kill, env, stdin, capture) and
``test_code_executor_sandbox.py`` (macOS sandbox-exec, docker).
"""
import asyncio
import os
import sys

import pytest

from backend.adapters.mock import MockAdapter
from backend.orchestrator import code_executor as ce
from backend.orchestrator.code_executor import (
    BACKEND_GUARANTEES,
    CodeBlockResult,
    CodeExecutionReport,
    CodeExecutor,
    combine_blocks,
    extract_code_blocks,
)
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset
from backend.models import ProviderName, RunState


def run(coro):
    return asyncio.run(coro)


def test_extract_code_blocks():
    text = (
        "Some prose.\n\n```python\nprint(1)\n```\n\n"
        "more prose\n```py\nx = 2\n```\n```bash\necho no\n```\n"
    )
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert "print(1)" in blocks[0]
    assert "x = 2" in blocks[1]


def test_run_block_success_captures_stdout():
    ex = CodeExecutor(timeout_seconds=10)
    res = asyncio.run(ex.run_block(1, "print('hello'); print(6 * 7)"))
    assert res.success and res.exit_code == 0
    assert "hello" in res.stdout and "42" in res.stdout


def test_run_block_failure_reports_nonzero_exit():
    ex = CodeExecutor(timeout_seconds=10)
    res = asyncio.run(ex.run_block(1, "raise ValueError('boom')"))
    assert not res.success and res.exit_code != 0
    assert "ValueError" in res.stderr


def test_run_block_timeout_is_killed():
    ex = CodeExecutor(timeout_seconds=1)
    res = asyncio.run(ex.run_block(1, "while True:\n    pass"))
    assert res.timed_out and not res.success
    assert res.limit_hit == "wall_timeout" and not res.killed_by_limit
    assert "wall-clock timeout" in res.stderr


def test_run_artifact_report_and_feedback_separate_mode():
    # Pre-revision behaviour: one process per block, no shared state.
    artifact = (
        "# MECH.md\n\n```python\nprint('AUROC', round(0.74, 2))\n```\n"
        "```python\nassert 1 == 2\n```\n"
    )
    ex = CodeExecutor(timeout_seconds=10, block_mode="separate")
    report = asyncio.run(ex.run_artifact(artifact))
    assert report.mode == "separate"
    assert report.executed == 2 and report.passed == 1 and report.failed == 1
    fb = report.to_feedback()
    assert "CODE EXECUTION REPORT" in fb and "AUROC" in fb
    assert "separate blocks" in fb
    assert "[Block 2] FAILED" in fb


# ── Block modes ─────────────────────────────────────────────────────────


SHARED_STATE_ARTIFACT = (
    "Load data.\n```python\nx = 21\n```\n"
    "Then analyse.\n```python\nprint('answer', x * 2)\n```\n"
)


def test_default_block_mode_is_single_script():
    ex = CodeExecutor(timeout_seconds=10)
    assert ex.block_mode == "single"
    report = run(ex.run_artifact(SHARED_STATE_ARTIFACT))
    assert report.mode == "single"
    assert report.code_blocks == 2 and report.n_code_blocks == 2
    assert report.executed == 1 and report.passed == 1
    unit = report.blocks[0]
    assert unit.block_indices == [1, 2]
    assert "answer 42" in unit.stdout
    fb = report.to_feedback()
    assert "Mode: single script (2 fenced python block(s)" in fb
    assert "[Script (blocks 1-2)] OK" in fb


def test_separate_mode_has_no_shared_state():
    ex = CodeExecutor(timeout_seconds=10)
    report = run(ex.run_artifact(SHARED_STATE_ARTIFACT, block_mode="separate"))
    assert report.mode == "separate"
    assert report.executed == 2 and report.passed == 1 and report.failed == 1
    assert "NameError" in report.blocks[1].stderr
    assert report.blocks[1].error_block == 2


def test_single_mode_attributes_error_to_block():
    artifact = (
        "```python\na = 1\nb = 2\n```\n"
        "```python\nc = a + b\nraise RuntimeError('in block two')\n```\n"
        "```python\nprint('never')\n```\n"
    )
    report = run(CodeExecutor(timeout_seconds=10).run_artifact(artifact))
    unit = report.blocks[0]
    assert not unit.success
    assert unit.error_block == 2
    assert "never" not in unit.stdout
    assert "error in block 2" in report.to_feedback()


def test_single_mode_syntax_error_attributed():
    artifact = "```python\nx = 1\n```\n```python\nprint(x)\n```\n```python\ndef f(:\n```\n"
    report = run(CodeExecutor(timeout_seconds=10).run_artifact(artifact))
    unit = report.blocks[0]
    assert not unit.success and "SyntaxError" in unit.stderr
    assert unit.error_block == 3


def test_combine_blocks_spans_and_dedent():
    script, spans = combine_blocks(["x = 1\ny = 2\n", "    print(x)\n    print(y)\n"])
    lines = script.split("\n")
    assert spans == [(1, 2, 3), (2, 5, 6)]
    assert lines[1] == "x = 1" and lines[4] == "print(x)"  # dedented
    assert lines[0].startswith("# ---- MI-Workbench: fenced python block 1 of 2")


def test_indented_block_runs_after_dedent():
    artifact = "1. Step one:\n   ```python\n   z = 5\n   print(z + 1)\n   ```\n"
    report = run(CodeExecutor(timeout_seconds=10).run_artifact(artifact))
    assert report.passed == 1 and "6" in report.blocks[0].stdout


def test_invalid_block_mode_rejected():
    with pytest.raises(ValueError):
        CodeExecutor(block_mode="parallel")
    with pytest.raises(ValueError):
        run(CodeExecutor().run_artifact(SHARED_STATE_ARTIFACT, block_mode="nope"))
    assert CodeExecutor(block_mode="single_script").block_mode == "single"
    assert CodeExecutor(block_mode="per_block").block_mode == "separate"


def test_no_blocks_yields_empty_report():
    report = run(CodeExecutor().run_artifact("no code here"))
    assert report.executed == 0 and report.to_feedback() == ""


# ── Configuration ───────────────────────────────────────────────────────


def test_backward_compatible_positional_signature():
    ex = CodeExecutor(5, 123, sys.executable)
    assert ex.timeout_seconds == 5
    assert ex.output_cap == 123 and ex.output_limit_bytes == 123
    assert ex.python_executable == sys.executable and ex.python == sys.executable
    assert ex.backend == "subprocess"


def test_defaults_are_safe():
    ex = CodeExecutor()
    assert ex.backend == "subprocess"
    assert ex.timeout_seconds == 20
    assert ex.cpu_seconds == 40  # 2 x timeout
    assert ex.memory_mb == 4096
    assert ex.max_file_mb == 64
    assert ex.max_open_files == 256
    assert ex.max_processes is None
    assert ex.output_limit_bytes == 4000
    assert ex.allow_network is False
    assert ex.read_only_paths == []
    assert ex.python == sys.executable
    assert ex.block_mode == "single"


def test_zero_disables_limits():
    ex = CodeExecutor(cpu_seconds=0, memory_mb=0, max_file_mb=0, max_open_files=0)
    assert ex.cpu_seconds is None and ex.memory_mb is None
    assert ex.max_file_mb is None and ex.max_open_files is None
    names = {r.name for r in ex._limit_plan().rlimits}
    assert names == {"RLIMIT_CORE"}


@pytest.mark.parametrize("kwargs", [
    {"backend": "chroot"},
    {"timeout_seconds": 0},
    {"cpu_seconds": -1},
    {"memory_mb": "lots"},
    {"max_open_files": 3},
    {"allow_network": "maybe"},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        CodeExecutor(**kwargs)


def test_from_run_config_reads_all_keys(tmp_path):
    ro1, ro2 = tmp_path / "a", tmp_path / "b"
    ro1.mkdir()
    ro2.mkdir()
    config = {
        "code_execution_enabled": True,
        "code_execution_backend": "sandbox_exec",
        "code_execution_python": "/usr/bin/python3",
        "code_execution_timeout": "30",
        "code_execution_cpu_seconds": 12,
        "code_execution_memory_mb": 1024,
        "code_execution_read_only_paths": [str(ro1), str(ro2)],
        "code_execution_allow_network": "false",
        "code_execution_max_file_mb": 8,
        "code_execution_max_processes": 16,
        "code_execution_max_open_files": 64,
        "code_execution_output_limit_bytes": 2048,
        "code_execution_block_mode": "separate",
        "code_execution_docker_image": "python:3.12-slim",
    }
    ex = CodeExecutor.from_run_config(config)
    assert ex.backend == "sandbox_exec"
    assert ex.python == "/usr/bin/python3"
    assert ex.timeout_seconds == 30.0
    assert ex.cpu_seconds == 12
    assert ex.memory_mb == 1024
    assert ex.read_only_paths == [str(ro1), str(ro2)]
    assert ex.allow_network is False
    assert ex.max_file_mb == 8
    assert ex.max_processes == 16
    assert ex.max_open_files == 64
    assert ex.output_limit_bytes == 2048
    assert ex.block_mode == "separate"
    assert ex.docker_image == "python:3.12-slim"
    cfg = ex.config_dict()
    assert cfg["backend"] == "sandbox_exec" and cfg["block_mode"] == "separate"


def test_from_run_config_defaults_and_overrides(tmp_path):
    ex = CodeExecutor.from_run_config({})
    assert ex.backend == "subprocess" and ex.timeout_seconds == 20
    ex = CodeExecutor.from_run_config(None, timeout_seconds=3)
    assert ex.timeout_seconds == 3 and ex.cpu_seconds == 6
    # None values keep defaults; a PATH-like string is split on os.pathsep.
    ex = CodeExecutor.from_run_config({
        "code_execution_timeout": None,
        "code_execution_read_only_paths": os.pathsep.join(["/x", "/y"]),
        "code_execution_allow_network": "true",
    })
    assert ex.timeout_seconds == 20
    assert ex.read_only_paths == ["/x", "/y"]
    assert ex.allow_network is True
    with pytest.raises(ValueError):
        CodeExecutor.from_run_config({"code_execution_backend": "vm"})


def test_run_config_keys_documented_in_module_docstring():
    doc = ce.__doc__
    for key in CodeExecutor.RUN_CONFIG_KEYS:
        assert f"``{key}``" in doc, key
    for backend in ce.BACKENDS:
        assert f"``{backend}``" in doc
    assert "NOT a security boundary" in doc
    assert "DEPRECATED" in doc


def test_guarantees_are_explicit_per_backend():
    assert BACKEND_GUARANTEES["subprocess"]["security_boundary"] is False
    assert BACKEND_GUARANTEES["sandbox_exec"]["security_boundary"] is True
    assert BACKEND_GUARANTEES["docker"]["security_boundary"] is True
    g = CodeExecutor().guarantees()
    assert "cannot be enforced" in g["network"]
    g = CodeExecutor(backend="sandbox_exec", allow_network=True).guarantees()
    assert "allowed by configuration" in g["network"]


# ── Fail closed ─────────────────────────────────────────────────────────


def test_unavailable_docker_fails_closed():
    ex = CodeExecutor(backend="docker", docker_bin="definitely-missing-docker-xyz")
    report = run(ex.run_artifact("```python\nprint(1)\n```"))
    assert report.executed == 1 and report.failed == 1
    unit = report.blocks[0]
    assert unit.error == "backend unavailable" and "not found" in unit.stderr
    assert "NOT RUN (backend unavailable)" in report.to_feedback()


def test_unavailable_sandbox_fails_closed(monkeypatch):
    monkeypatch.setattr(ce._seatbelt, "sandbox_exec_available",
                        lambda: (False, "sandbox_exec backend requires macOS"))
    ex = CodeExecutor(backend="sandbox_exec")
    res = run(ex.run_block(1, "print('should not run')"))
    assert not res.success and res.error == "backend unavailable"
    assert "should not run" not in res.stdout
    assert "requires macOS" in res.stderr


def test_missing_interpreter_fails_closed():
    ex = CodeExecutor(python="/nonexistent/bin/python9")
    res = run(ex.run_block(1, "print(1)"))
    assert not res.success and res.error == "backend unavailable"
    assert "interpreter not found" in res.stderr


# ── Report serialisation ────────────────────────────────────────────────


def test_report_to_dict_and_summary():
    report = run(CodeExecutor(timeout_seconds=10).run_artifact(
        "```python\nopen('results.json', 'w').write('{\"auroc\": 0.7}')\nprint('ok')\n```"))
    s = report.summary()
    assert s == {"executed": 1, "passed": 1, "failed": 0, "mode": "single",
                 "backend": "subprocess", "code_blocks": 1, "timed_out": 0,
                 "killed_by_limit": 0}
    d = report.to_dict()
    import json
    json.dumps(d)  # serialisable
    files = d["blocks"][0]["files"]
    assert any(f["path"] == "results.json" and f["content"] == '{"auroc": 0.7}' for f in files)
    assert d["limits"]["wall_timeout_seconds"] == 10
    fb = report.to_feedback()
    assert "files written: results.json" in fb and '{"auroc": 0.7}' in fb
    assert "Backend: subprocess" in fb and "NOT a security boundary" in fb


def test_legacy_report_construction_still_works():
    report = CodeExecutionReport(blocks=[CodeBlockResult(index=1, success=True, exit_code=0,
                                                         stdout="x")])
    assert report.executed == 1 and report.passed == 1
    assert "[Block 1] OK" in report.to_feedback()


# ── Engine integration ──────────────────────────────────────────────────


def test_engine_executes_code_when_enabled():
    # The default executor artifact contains a runnable analysis snippet that
    # prints "incremental_auroc"; the engine should run it when enabled.
    async def go(enabled):
        MockAdapter.reset_iteration_count()
        MockAdapter.fixed_iteration = 1
        logs = []
        eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0),
                         artifact_writer=lambda i, n, c: logs.append((n, c)))
        loop = get_preset("executor_reviewer", max_iterations=2)
        run_state = RunState(workspace_id="w", loop_preset="executor_reviewer",
                             task="probe", provider=ProviderName.MOCK, model="",
                             max_iterations=2,
                             config={"convergence_enabled": False,
                                     "code_execution_enabled": enabled})
        final = await eng.run_loop(run_state, loop)
        MockAdapter.fixed_iteration = None
        return final, logs

    final, logs = asyncio.run(go(True))
    exec_iters = [it for it in final.iterations
                  if it.code_execution and it.code_execution["executed"] > 0]
    assert exec_iters, "no iteration recorded code execution"
    assert exec_iters[0].code_execution["passed"] == 1
    assert any(name == "RUN_LOG.md" and "incremental_auroc" in content
               for name, content in logs)

    # Disabled by default: no code execution recorded.
    final_off, _ = asyncio.run(go(False))
    assert all(not it.code_execution for it in final_off.iterations)
