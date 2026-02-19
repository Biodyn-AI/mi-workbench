"""Tests for the orchestrator: state machine, presets, DSL, and engine."""
import asyncio

import pytest

from backend.models import (
    IterationStatus,
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    RunState,
    RunStatus,
)
from backend.orchestrator.state_machine import InvalidTransitionError, RunStateMachine
from backend.orchestrator.presets import executor_reviewer_preset, get_preset, research_followups_preset
from backend.orchestrator.dsl import compile_loop, parse_loop_yaml, validate_loop
from backend.orchestrator.engine import LoopEngine
from backend.adapters.mock import MockAdapter


# ── State Machine Tests ──────────────────────────────────────────────


def _make_run_state(**kwargs) -> RunState:
    defaults = dict(workspace_id="ws1", loop_preset="executor_reviewer",
                    task="test", provider=ProviderName.MOCK, model="mock")
    defaults.update(kwargs)
    return RunState(**defaults)


class TestStateMachine:
    def test_start_from_pending(self):
        sm = RunStateMachine(_make_run_state())
        state = sm.start()
        assert state.status == RunStatus.RUNNING
        assert state.started_at is not None

    def test_cannot_start_from_running(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.RUNNING))
        with pytest.raises(InvalidTransitionError):
            sm.start()

    def test_complete_from_running(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.RUNNING))
        state = sm.complete()
        assert state.status == RunStatus.COMPLETED

    def test_cannot_resume_completed(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.COMPLETED))
        with pytest.raises(InvalidTransitionError):
            sm.resume()

    def test_pause_and_resume(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.RUNNING))
        state = sm.pause()
        assert state.status == RunStatus.PAUSED
        state = sm.resume()
        assert state.status == RunStatus.RUNNING

    def test_fail_from_running(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.RUNNING))
        state = sm.fail("something broke")
        assert state.status == RunStatus.FAILED
        assert state.error == "something broke"

    def test_stop_from_running(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.RUNNING))
        state = sm.stop()
        assert state.status == RunStatus.STOPPED
        assert state.stopped_at is not None

    def test_cannot_stop_completed(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.COMPLETED))
        with pytest.raises(InvalidTransitionError):
            sm.stop()

    def test_next_iteration_increments(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.RUNNING))
        assert sm.next_iteration() == 1
        assert sm.next_iteration() == 2

    def test_next_iteration_requires_running(self):
        sm = RunStateMachine(_make_run_state(status=RunStatus.PAUSED))
        with pytest.raises(InvalidTransitionError):
            sm.next_iteration()

    def test_state_change_callback(self):
        events = []
        def on_change(old, new, state):
            events.append((old, new))

        sm = RunStateMachine(_make_run_state(), on_state_change=on_change)
        sm.start()
        sm.pause()
        sm.resume()
        sm.complete()
        assert events == [
            (RunStatus.PENDING, RunStatus.RUNNING),
            (RunStatus.RUNNING, RunStatus.PAUSED),
            (RunStatus.PAUSED, RunStatus.RUNNING),
            (RunStatus.RUNNING, RunStatus.COMPLETED),
        ]


# ── Presets Tests ────────────────────────────────────────────────────


class TestPresets:
    def test_executor_reviewer_preset_structure(self):
        loop = executor_reviewer_preset()
        assert loop.name == "executor_reviewer"
        assert len(loop.nodes) == 3
        assert len(loop.edges) == 4
        node_ids = {n.id for n in loop.nodes}
        assert node_ids == {"executor", "reviewer", "adversarial"}

    def test_research_followups_preset(self):
        loop = research_followups_preset()
        assert loop.name == "research_followups"
        assert len(loop.nodes) == 2

    def test_get_preset_by_name(self):
        loop = get_preset("executor_reviewer")
        assert loop.name == "executor_reviewer"

    def test_get_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_executor_reviewer_adversarial_edge_condition(self):
        loop = executor_reviewer_preset(adversarial_every_k=5)
        adversarial_edges = [e for e in loop.edges if e.target == "adversarial"]
        assert len(adversarial_edges) == 1
        assert adversarial_edges[0].condition == "every_k:5"


# ── DSL Tests ────────────────────────────────────────────────────────


class TestDSL:
    def test_parse_yaml_basic(self):
        yaml_str = """
name: test_loop
description: A test loop
nodes:
  - id: step1
    role: executor
    prompt_ref: exec/test
  - id: step2
    role: reviewer
    prompt_ref: review/test
edges:
  - source: step1
    target: step2
    condition: always
  - source: step2
    target: step1
    condition: always
max_iterations: 10
"""
        loop = parse_loop_yaml(yaml_str)
        assert loop.name == "test_loop"
        assert len(loop.nodes) == 2
        assert len(loop.edges) == 2
        assert loop.max_iterations == 10

    def test_validate_empty_nodes(self):
        loop = LoopDefinition(name="empty", nodes=[], edges=[])
        errors = validate_loop(loop)
        assert any("at least one node" in e for e in errors)

    def test_validate_invalid_edge_reference(self):
        loop = LoopDefinition(
            name="bad",
            nodes=[LoopNode(id="a", role="exec")],
            edges=[LoopEdge(source="a", target="b", condition="always")],
        )
        errors = validate_loop(loop)
        assert any("'b' not found" in e for e in errors)

    def test_validate_invalid_condition(self):
        loop = LoopDefinition(
            name="bad_cond",
            nodes=[
                LoopNode(id="a", role="exec"),
                LoopNode(id="b", role="review"),
            ],
            edges=[LoopEdge(source="a", target="b", condition="maybe")],
        )
        errors = validate_loop(loop)
        assert any("Invalid condition" in e for e in errors)

    def test_compile_simple_loop(self):
        loop = LoopDefinition(
            name="simple",
            nodes=[
                LoopNode(id="a", role="executor", prompt_ref="exec/test"),
                LoopNode(id="b", role="reviewer", prompt_ref="review/test"),
            ],
            edges=[
                LoopEdge(source="a", target="b", condition="always"),
                LoopEdge(source="b", target="a", condition="always"),
            ],
        )
        plan = compile_loop(loop)
        assert len(plan.steps) >= 2
        assert plan.steps[0].node_id == "a"
        assert plan.steps[1].node_id == "b"

    def test_compile_with_preset(self):
        loop = executor_reviewer_preset()
        plan = compile_loop(loop)
        assert plan.cycle_length > 0
        roles = [s.role for s in plan.steps]
        assert "executor" in roles
        assert "reviewer" in roles

    def test_compile_invalid_raises(self):
        loop = LoopDefinition(name="invalid", nodes=[], edges=[])
        with pytest.raises(ValueError, match="Invalid loop"):
            compile_loop(loop)


# ── Engine Tests ─────────────────────────────────────────────────────


class TestEngine:
    @pytest.fixture
    def mock_adapter(self):
        return MockAdapter(min_delay=0.01, max_delay=0.02)

    @pytest.fixture
    def simple_loop(self):
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
            max_iterations=3,
        )

    @pytest.mark.asyncio
    async def test_engine_runs_iterations(self, mock_adapter, simple_loop):
        run_state = _make_run_state(max_iterations=3)
        engine = LoopEngine(adapter=mock_adapter)
        result = await engine.run_loop(run_state, simple_loop)
        assert result.status in (RunStatus.COMPLETED, RunStatus.STOPPED)
        assert result.current_iteration == 3
        assert len(result.iterations) == 3

    @pytest.mark.asyncio
    async def test_engine_emits_events(self, mock_adapter, simple_loop):
        events = []
        def on_event(etype, data):
            events.append(etype)

        run_state = _make_run_state(max_iterations=2)
        engine = LoopEngine(adapter=mock_adapter, on_event=on_event)
        await engine.run_loop(run_state, simple_loop)
        assert "loop_started" in events
        assert "iteration_started" in events
        assert "iteration_completed" in events

    @pytest.mark.asyncio
    async def test_engine_cancellation(self, mock_adapter, simple_loop):
        cancel = asyncio.Event()
        cancel.set()  # immediately cancelled

        run_state = _make_run_state(max_iterations=10)
        engine = LoopEngine(adapter=mock_adapter)
        result = await engine.run_loop(run_state, simple_loop, cancel_event=cancel)
        assert result.status == RunStatus.STOPPED

    @pytest.mark.asyncio
    async def test_engine_budget_exceeded(self, mock_adapter, simple_loop):
        run_state = _make_run_state(max_iterations=100)
        run_state.total_tokens = 499_999
        run_state.config["budget_max_tokens"] = 500_000
        engine = LoopEngine(adapter=mock_adapter)
        result = await engine.run_loop(run_state, simple_loop)
        assert result.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_engine_handles_adapter_failure(self):
        adapter = MockAdapter(failure_rate=1.0, min_delay=0.01, max_delay=0.02)
        loop = LoopDefinition(
            name="fail_loop",
            nodes=[LoopNode(id="exec", role="executor", prompt_ref="exec/test")],
            edges=[LoopEdge(source="exec", target="exec", condition="always")],
            max_iterations=5,
        )
        run_state = _make_run_state(max_iterations=5)
        engine = LoopEngine(adapter=adapter)
        result = await engine.run_loop(run_state, loop)
        assert result.status == RunStatus.FAILED

    @pytest.mark.asyncio
    async def test_engine_accumulates_cost(self, mock_adapter, simple_loop):
        run_state = _make_run_state(max_iterations=2)
        engine = LoopEngine(adapter=mock_adapter)
        result = await engine.run_loop(run_state, simple_loop)
        assert result.total_tokens > 0
        assert result.total_cost > 0
