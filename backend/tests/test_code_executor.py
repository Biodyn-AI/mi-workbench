"""Tests for sandboxed code execution and its engine integration."""
import asyncio

import pytest

from backend.adapters.mock import MockAdapter
from backend.orchestrator.code_executor import CodeExecutor, extract_code_blocks
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset
from backend.models import ProviderName, RunState


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


def test_run_artifact_report_and_feedback():
    artifact = (
        "# MECH.md\n\n```python\nprint('AUROC', round(0.74, 2))\n```\n"
        "```python\nassert 1 == 2\n```\n"
    )
    report = asyncio.run(CodeExecutor(timeout_seconds=10).run_artifact(artifact))
    assert report.executed == 2 and report.passed == 1 and report.failed == 1
    fb = report.to_feedback()
    assert "CODE EXECUTION REPORT" in fb and "AUROC" in fb


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
        run = RunState(workspace_id="w", loop_preset="executor_reviewer",
                       task="probe", provider=ProviderName.MOCK, model="",
                       max_iterations=2,
                       config={"convergence_enabled": False,
                               "code_execution_enabled": enabled})
        final = await eng.run_loop(run, loop)
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
