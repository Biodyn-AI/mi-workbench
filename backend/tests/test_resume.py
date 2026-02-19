"""Tests for run resume logic — engine, state machine, and runner integration."""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.models import (
    IterationResult,
    IterationStatus,
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    RunState,
    RunStatus,
)
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.state_machine import InvalidTransitionError, RunStateMachine
from backend.adapters.mock import MockAdapter


# ── Helpers ──────────────────────────────────────────────────────────


def _make_run_state(**kwargs) -> RunState:
    defaults = dict(
        workspace_id="ws1",
        loop_preset="executor_reviewer",
        task="test task",
        provider=ProviderName.MOCK,
        model="mock",
    )
    defaults.update(kwargs)
    return RunState(**defaults)


def _simple_loop(max_iterations: int = 10) -> LoopDefinition:
    return LoopDefinition(
        name="simple",
        nodes=[
            LoopNode(id="exec", role="executor", prompt_ref="exec/test"),
            LoopNode(id="review", role="reviewer", prompt_ref="review/test"),
        ],
        edges=[
            LoopEdge(source="exec", target="review", condition="always"),
            LoopEdge(source="review", target="exec", condition="always"),
        ],
        max_iterations=max_iterations,
    )


def _make_completed_iterations(count: int) -> list[IterationResult]:
    """Create a list of completed iteration results."""
    iterations = []
    roles = ["executor", "reviewer"]
    for i in range(1, count + 1):
        iterations.append(
            IterationResult(
                iteration_number=i,
                role=roles[(i - 1) % 2],
                status=IterationStatus.COMPLETED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                output_summary=f"Output from iteration {i}",
                token_usage=1500,
                cost_estimate=0.005,
            )
        )
    return iterations


# ── State Machine Resume Tests ───────────────────────────────────────


class TestStateMachineResume:
    def test_resume_from_paused(self):
        """PAUSED -> RUNNING via resume()."""
        rs = _make_run_state(status=RunStatus.PAUSED)
        rs.started_at = datetime.utcnow()
        sm = RunStateMachine(rs)
        state = sm.resume()
        assert state.status == RunStatus.RUNNING

    def test_resume_from_completed_raises(self):
        """Cannot resume a COMPLETED run."""
        rs = _make_run_state(status=RunStatus.COMPLETED)
        sm = RunStateMachine(rs)
        with pytest.raises(InvalidTransitionError):
            sm.resume()

    def test_resume_from_failed_raises(self):
        """Cannot resume a FAILED run."""
        rs = _make_run_state(status=RunStatus.FAILED)
        sm = RunStateMachine(rs)
        with pytest.raises(InvalidTransitionError):
            sm.resume()

    def test_resume_preserves_started_at(self):
        """resume() should not overwrite the original started_at timestamp."""
        original_start = datetime(2025, 1, 1, 12, 0, 0)
        rs = _make_run_state(status=RunStatus.PAUSED)
        rs.started_at = original_start
        sm = RunStateMachine(rs)
        sm.resume()
        assert rs.started_at == original_start

    def test_resume_does_not_reset_iteration(self):
        """resume() should not reset the iteration counter."""
        rs = _make_run_state(status=RunStatus.PAUSED, current_iteration=5)
        sm = RunStateMachine(rs)
        sm.resume()
        assert rs.current_iteration == 5


# ── Engine Resume Tests ──────────────────────────────────────────────


class TestEngineResume:
    @pytest.fixture(autouse=True)
    def reset_mock(self):
        MockAdapter.reset_iteration_count()
        yield

    @pytest.fixture
    def mock_adapter(self):
        return MockAdapter(min_delay=0.01, max_delay=0.02)

    @pytest.mark.asyncio
    async def test_resume_continues_from_last_iteration(self, mock_adapter):
        """Engine should start at resume_from iteration, not iteration 0."""
        prior_iterations = _make_completed_iterations(3)
        run_state = _make_run_state(
            status=RunStatus.PAUSED,
            current_iteration=3,
            max_iterations=5,
        )
        run_state.started_at = datetime.utcnow()
        run_state.iterations = prior_iterations
        run_state.total_tokens = 4500
        run_state.total_cost = 0.015

        engine = LoopEngine(adapter=mock_adapter)
        result = await engine.run_loop(
            run_state, _simple_loop(max_iterations=5), resume_from=3,
        )

        assert result.status in (RunStatus.COMPLETED, RunStatus.STOPPED)
        # Should have done 2 more iterations (4 and 5) to reach max_iterations=5
        assert result.current_iteration == 5
        # Total iterations: 3 prior + 2 new = 5
        assert len(result.iterations) == 5

    @pytest.mark.asyncio
    async def test_resume_preserves_previous_output(self, mock_adapter):
        """The resumed engine should pass prior iteration output as context."""
        prior_iterations = _make_completed_iterations(2)
        prior_iterations[-1].output_summary = "Important prior context from iteration 2"

        run_state = _make_run_state(
            status=RunStatus.PAUSED,
            current_iteration=2,
            max_iterations=3,
        )
        run_state.started_at = datetime.utcnow()
        run_state.iterations = prior_iterations

        # Track what prompts the adapter receives
        prompts_received = []
        original_run = mock_adapter.run

        async def _spy_run(request):
            prompts_received.append(request.prompt_bundle.user_prompt)
            return await original_run(request)

        mock_adapter.run = _spy_run

        engine = LoopEngine(adapter=mock_adapter)
        await engine.run_loop(
            run_state, _simple_loop(max_iterations=3), resume_from=2,
        )

        # The first prompt after resume should contain prior output
        assert len(prompts_received) >= 1
        assert "Important prior context from iteration 2" in prompts_received[0]

    @pytest.mark.asyncio
    async def test_resume_convergence_state_restored(self, mock_adapter):
        """Convergence detector should be seeded with historical data on resume."""
        prior_iterations = _make_completed_iterations(4)

        run_state = _make_run_state(
            status=RunStatus.PAUSED,
            current_iteration=4,
            max_iterations=10,
        )
        run_state.started_at = datetime.utcnow()
        run_state.iterations = prior_iterations

        engine = LoopEngine(adapter=mock_adapter)

        # After constructing and before running, check that resume seeds convergence
        # We run with resume_from=4, which should replay 4 iterations into the detector
        # Use a cancel event to stop quickly after checking
        cancel = asyncio.Event()

        events_seen = []
        def on_event(etype, data):
            events_seen.append(etype)
            # Cancel after first iteration completes so we can check state
            if etype == "iteration_completed":
                cancel.set()

        engine._on_event = on_event
        await engine.run_loop(
            run_state, _simple_loop(max_iterations=10),
            cancel_event=cancel, resume_from=4,
        )

        # Convergence detector should have data from historical iterations
        metrics = engine._convergence.get_metrics()
        # iteration_count should be >= 4 (historical) + at least 1 new
        assert metrics["iteration_count"] >= 4

    @pytest.mark.asyncio
    async def test_resume_from_zero_is_normal_start(self, mock_adapter):
        """resume_from=0 should behave exactly like a fresh start."""
        run_state = _make_run_state(max_iterations=2)
        engine = LoopEngine(adapter=mock_adapter)
        result = await engine.run_loop(
            run_state, _simple_loop(max_iterations=2), resume_from=0,
        )

        assert result.status in (RunStatus.COMPLETED, RunStatus.STOPPED)
        assert result.current_iteration == 2
        assert len(result.iterations) == 2

    @pytest.mark.asyncio
    async def test_resume_accumulates_tokens_and_cost(self, mock_adapter):
        """Resumed runs should accumulate tokens/cost on top of prior totals."""
        prior_iterations = _make_completed_iterations(2)
        run_state = _make_run_state(
            status=RunStatus.PAUSED,
            current_iteration=2,
            max_iterations=3,
        )
        run_state.started_at = datetime.utcnow()
        run_state.iterations = prior_iterations
        run_state.total_tokens = 3000
        run_state.total_cost = 0.01

        engine = LoopEngine(adapter=mock_adapter)
        result = await engine.run_loop(
            run_state, _simple_loop(max_iterations=3), resume_from=2,
        )

        # Should have added tokens from the new iteration
        assert result.total_tokens > 3000
        assert result.total_cost > 0.01


# ── Runner Integration Test ─────────────────────────────────────────


class TestExecuteRunResume:
    @pytest.mark.asyncio
    async def test_execute_run_detects_paused_and_resumes(self):
        """Integration test: create a paused run, call execute_run, verify resume."""
        from backend.orchestrator import runner

        prior_iterations = _make_completed_iterations(3)
        paused_run = _make_run_state(
            status=RunStatus.PAUSED,
            current_iteration=3,
            max_iterations=5,
        )
        paused_run.started_at = datetime.utcnow()
        paused_run.iterations = prior_iterations
        paused_run.total_tokens = 4500
        paused_run.total_cost = 0.015

        # Mock the DB calls and adapter
        mock_adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        MockAdapter.reset_iteration_count()

        captured_resume_from = []
        original_run_loop = LoopEngine.run_loop

        async def _spy_run_loop(self_engine, run_state, loop_def, cancel_event=None,
                                workspace_path="", resume_from=0):
            captured_resume_from.append(resume_from)
            return await original_run_loop(
                self_engine, run_state, loop_def,
                cancel_event=cancel_event,
                workspace_path=workspace_path,
                resume_from=resume_from,
            )

        with patch.object(runner, "get_run", new_callable=AsyncMock, return_value=paused_run), \
             patch.object(runner, "get_workspace", new_callable=AsyncMock, return_value=None), \
             patch.object(runner, "get_adapter", return_value=mock_adapter), \
             patch.object(runner, "update_run", new_callable=AsyncMock, return_value=None), \
             patch.object(runner, "create_iteration", new_callable=AsyncMock, return_value=None), \
             patch.object(LoopEngine, "run_loop", _spy_run_loop):
            await runner.execute_run(paused_run.run_id)

        # Verify that run_loop was called with resume_from=3
        assert len(captured_resume_from) == 1
        assert captured_resume_from[0] == 3
