"""Runner: registry cleanup on every exit path, incremental iteration
persistence, per-iteration accounting in the DB and in run_meta.json."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
import pytest_asyncio

from backend.adapters.mock import MockAdapter
from backend.config import config
from backend.database import create_run, create_workspace, get_run, init_db
from backend.models import ProviderName, RunState, RunStatus, WorkspaceConfig
from backend.orchestrator import runner
from backend.orchestrator.runner import (
    MAX_CONCURRENT_RUNS,
    _active_runs,
    _active_tasks,
    build_run_meta,
    cancel_run,
    execute_run,
)


@pytest_asyncio.fixture(autouse=True)
async def db(tmp_path):
    config.db_path = str(tmp_path / "runner.db")
    await init_db()
    MockAdapter.reset_iteration_count()
    yield


async def _make_run(tmp_path, preset="executor_reviewer", max_iterations=4, cfg=None,
                    model="", with_workspace=True):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir(exist_ok=True)
    ws = WorkspaceConfig(name="w", path=str(ws_dir) if with_workspace else str(tmp_path / "nope"),
                         default_provider=ProviderName.MOCK)
    await create_workspace(ws)
    run = RunState(workspace_id=ws.id, loop_preset=preset, task="probe",
                   provider=ProviderName.MOCK, model=model, max_iterations=max_iterations,
                   config={"convergence_enabled": False, **(cfg or {})})
    await create_run(run)
    return run, ws_dir / "runs" / run.run_id


@pytest.mark.asyncio
async def test_registries_cleaned_on_unknown_preset(tmp_path):
    run, _ = await _make_run(tmp_path, preset="custom:/nope/missing.yaml")
    task = asyncio.create_task(execute_run(run.run_id))
    _active_tasks[run.run_id] = task
    await task
    assert run.run_id not in _active_runs and run.run_id not in _active_tasks
    final = await get_run(run.run_id)
    assert final.status == RunStatus.FAILED and final.stop_reason == "failed:unknown_preset"


@pytest.mark.asyncio
async def test_registries_cleaned_when_run_missing():
    task = asyncio.create_task(execute_run("doesnotexist"))
    _active_tasks["doesnotexist"] = task
    await task
    assert "doesnotexist" not in _active_tasks and "doesnotexist" not in _active_runs


@pytest.mark.asyncio
async def test_registries_cleaned_when_concurrency_limit_hit(tmp_path):
    run, _ = await _make_run(tmp_path)
    fakes = {f"fake-{i}": asyncio.Event() for i in range(MAX_CONCURRENT_RUNS)}
    _active_runs.update(fakes)
    try:
        task = asyncio.create_task(execute_run(run.run_id))
        _active_tasks[run.run_id] = task
        await task
    finally:
        for k in fakes:
            _active_runs.pop(k, None)
    assert run.run_id not in _active_tasks
    final = await get_run(run.run_id)
    assert final.status == RunStatus.FAILED
    assert final.stop_reason == "failed:concurrency_limit"


@pytest.mark.asyncio
async def test_registries_cleaned_when_task_cancelled_during_setup(tmp_path):
    run, _ = await _make_run(tmp_path)
    started = asyncio.Event()

    async def slow_get_workspace(ws_id):
        started.set()
        await asyncio.sleep(30)

    with patch.object(runner, "get_workspace", slow_get_workspace):
        task = asyncio.create_task(execute_run(run.run_id))
        _active_tasks[run.run_id] = task
        await asyncio.wait_for(started.wait(), 5)
        assert run.run_id in _active_runs
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert run.run_id not in _active_runs and run.run_id not in _active_tasks


@pytest.mark.asyncio
async def test_iterations_persisted_incrementally(tmp_path):
    run, _ = await _make_run(tmp_path, max_iterations=100)
    task = asyncio.create_task(execute_run(run.run_id))
    _active_tasks[run.run_id] = task
    # Wait until at least two iterations are in the DB while the run is live.
    for _ in range(200):
        live = await get_run(run.run_id)
        if live and len(live.iterations) >= 2:
            break
        await asyncio.sleep(0.05)
    assert len(live.iterations) >= 2
    assert live.current_iteration >= 2 and live.total_tokens > 0
    # RUNNING is persisted before the loop starts (startup recovery needs it)
    assert live.status == RunStatus.RUNNING
    await cancel_run(run.run_id)
    await asyncio.wait_for(task, 10)
    final = await get_run(run.run_id)
    assert final.status == RunStatus.STOPPED and final.stop_reason == "cancelled"
    assert len(final.iterations) == final.current_iteration
    assert len({i.iteration_id for i in final.iterations}) == len(final.iterations)


@pytest.mark.asyncio
async def test_accounting_in_db_and_run_meta(tmp_path):
    run, run_dir = await _make_run(tmp_path, max_iterations=3,
                                   cfg={"reasoning_effort": "low",
                                        "consensus_similarity_method": "tfidf"},
                                   model="gpt-x")
    await execute_run(run.run_id)
    final = await get_run(run.run_id)
    assert final.stop_reason == "max_iterations"
    for it in final.iterations:
        assert it.provider == "mock"
        assert it.model == "mock"  # the answering model reported by the adapter
        assert it.reasoning_effort == "low"
        assert it.cli_version == "mock"
        assert it.input_tokens > 0 and it.output_tokens > 0
        assert it.token_usage == it.input_tokens + it.output_tokens
        assert it.duration_seconds > 0
    assert final.total_tokens == sum(i.token_usage for i in final.iterations)
    assert final.total_input_tokens == sum(i.input_tokens for i in final.iterations)
    reviewer_it = [i for i in final.iterations if i.role == "reviewer"][0]
    assert reviewer_it.grade and reviewer_it.critical_count is not None

    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["stop_reason"] == "max_iterations"
    assert meta["model"] == "gpt-x" and meta["reasoning_effort"] == "low"
    assert meta["total_input_tokens"] == final.total_input_tokens
    assert meta["cli_versions"] == ["mock"] and meta["models_used"] == ["mock"]
    assert meta["loop_source"] == "python"
    assert meta["config"]["consensus_similarity_method"] == "tfidf"
    assert "convergence_enabled" in meta["config"]
    assert [m["iteration"] for m in meta["iterations"]] == [1, 2, 3]
    first = meta["iterations"][0]
    for key in ("provider", "model", "reasoning_effort", "cli_version", "input_tokens",
                "output_tokens", "cached_input_tokens", "duration_seconds", "cost_estimate"):
        assert key in first
    assert meta["convergence"]["rule"]["rule"] == "any"


@pytest.mark.asyncio
async def test_failed_run_records_reason_in_meta(tmp_path):
    run, run_dir = await _make_run(tmp_path, max_iterations=3)
    adapter = MockAdapter(failure_rate=1.0, min_delay=0, max_delay=0)
    with patch.object(runner, "get_adapter", return_value=adapter):
        await execute_run(run.run_id)
    final = await get_run(run.run_id)
    assert final.status == RunStatus.FAILED and final.stop_reason == "failed:adapter_failed"
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["stop_reason"] == "failed:adapter_failed" and meta["error"]
    assert len(final.iterations) == 1 and final.iterations[0].status.value == "failed"


def test_build_run_meta_without_loop():
    st = RunState(workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
                  model="m", config={"reasoning_effort": "high", "secret_thing": "x"})
    meta = build_run_meta(st)
    assert meta.reasoning_effort == "high"
    assert "secret_thing" not in meta.config
    assert meta.iterations == [] and meta.loop_source == ""
