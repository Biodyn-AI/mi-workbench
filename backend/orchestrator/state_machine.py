"""Run state machine — manages RunStatus transitions and emits events."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from backend.models import RunState, RunStatus


# Valid transitions: from_status -> set of allowed to_statuses
_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.RUNNING: {RunStatus.PAUSED, RunStatus.STOPPED, RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.STOPPED, RunStatus.FAILED},
    RunStatus.STOPPED: set(),
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
}


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""
    pass


class RunStateMachine:
    """Manages state transitions for a single Run."""

    def __init__(self, run_state: RunState,
                 on_state_change: Optional[Callable[[RunStatus, RunStatus, RunState], Any]] = None):
        self.state = run_state
        self._on_state_change = on_state_change

    def _transition(self, to_status: RunStatus) -> RunState:
        current = self.state.status
        allowed = _TRANSITIONS.get(current, set())
        if to_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {current.value} to {to_status.value}"
            )
        old_status = current
        self.state.status = to_status
        if self._on_state_change:
            self._on_state_change(old_status, to_status, self.state)
        return self.state

    def start(self) -> RunState:
        """Transition PENDING -> RUNNING."""
        self.state.started_at = datetime.utcnow()
        return self._transition(RunStatus.RUNNING)

    def next_iteration(self) -> int:
        """Increment iteration counter. Must be RUNNING."""
        if self.state.status != RunStatus.RUNNING:
            raise InvalidTransitionError(
                f"Cannot advance iteration in state {self.state.status.value}"
            )
        self.state.current_iteration += 1
        return self.state.current_iteration

    def pause(self) -> RunState:
        """Transition RUNNING -> PAUSED."""
        return self._transition(RunStatus.PAUSED)

    def stop(self) -> RunState:
        """Transition RUNNING|PAUSED -> STOPPED."""
        self.state.stopped_at = datetime.utcnow()
        return self._transition(RunStatus.STOPPED)

    def resume(self) -> RunState:
        """Transition PAUSED -> RUNNING."""
        return self._transition(RunStatus.RUNNING)

    def complete(self) -> RunState:
        """Transition RUNNING -> COMPLETED."""
        self.state.stopped_at = datetime.utcnow()
        return self._transition(RunStatus.COMPLETED)

    def fail(self, error: str = "") -> RunState:
        """Transition to FAILED from PENDING, RUNNING, or PAUSED."""
        self.state.error = error
        self.state.stopped_at = datetime.utcnow()
        return self._transition(RunStatus.FAILED)
