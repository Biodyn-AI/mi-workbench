"""Tests for the convergence detection module."""
import asyncio

import pytest

from backend.models import (
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    ReviewCritique,
    ReviewResult,
    RunState,
    RunStatus,
    SeverityLevel,
)
from backend.orchestrator.convergence import ConvergenceDetector


def _make_review(grade: str = "", critiques: list[ReviewCritique] | None = None) -> ReviewResult:
    return ReviewResult(
        overall_grade=grade,
        critiques=critiques or [],
    )


def _make_run_state(**kwargs) -> RunState:
    defaults = dict(
        workspace_id="ws1",
        loop_preset="executor_reviewer",
        task="test",
        provider=ProviderName.MOCK,
        model="mock",
    )
    defaults.update(kwargs)
    return RunState(**defaults)


class TestConvergenceDetector:
    def test_no_convergence_before_min_iterations(self):
        """should_stop returns False before min_iterations even with stable grades."""
        cd = ConvergenceDetector(window_size=2, min_iterations=4)
        for i in range(1, 4):
            cd.record_iteration(
                iteration_num=i,
                role="reviewer",
                output=f"review {i}",
                review=_make_review(grade="A"),
            )
        converged, reason = cd.should_stop()
        assert not converged
        assert reason == ""

    def test_grade_stabilization(self):
        """Same grade for window_size consecutive reviews triggers convergence."""
        cd = ConvergenceDetector(window_size=3, min_iterations=4)
        # Record 5 iterations: grades vary then stabilize
        grades = ["B", "A", "A", "A", "A"]
        for i, g in enumerate(grades, start=1):
            cd.record_iteration(
                iteration_num=i,
                role="reviewer",
                output=f"review {i}",
                review=_make_review(grade=g),
            )
        converged, reason = cd.should_stop()
        assert converged
        assert "Grade stabilized at A" in reason

    def test_critique_count_convergence(self):
        """CRITICAL/HIGH critiques dropping to 0 for window triggers convergence."""
        cd = ConvergenceDetector(window_size=3, min_iterations=4)

        # First iteration has critiques
        cd.record_iteration(
            iteration_num=1,
            role="reviewer",
            output="review 1",
            review=_make_review(grade="C", critiques=[
                ReviewCritique(severity=SeverityLevel.CRITICAL, category="stats",
                               description="Missing control"),
                ReviewCritique(severity=SeverityLevel.HIGH, category="method",
                               description="No baseline"),
            ]),
        )
        # Next 3 have zero CRITICAL/HIGH (but may have LOW), with varying grades
        varying_grades = ["B", "B+", "A-"]
        for i, g in zip(range(2, 5), varying_grades):
            cd.record_iteration(
                iteration_num=i,
                role="reviewer",
                output=f"review {i}",
                review=_make_review(grade=g, critiques=[
                    ReviewCritique(severity=SeverityLevel.LOW, category="style",
                                   description="Minor typo"),
                ]),
            )

        converged, reason = cd.should_stop()
        assert converged
        assert "Zero CRITICAL/HIGH" in reason

    def test_output_similarity_convergence(self):
        """Near-identical executor outputs trigger convergence."""
        cd = ConvergenceDetector(window_size=3, similarity_threshold=0.9, min_iterations=4)

        base_output = "The attention mechanism captures co-expression patterns in layer 15."

        for i in range(1, 6):
            cd.record_iteration(
                iteration_num=i,
                role="executor",
                output=base_output,
            )

        converged, reason = cd.should_stop()
        assert converged
        assert "similarity" in reason.lower()

    def test_mixed_signals_any_triggers(self):
        """Only ONE convergence signal is needed to trigger should_stop."""
        cd = ConvergenceDetector(window_size=2, min_iterations=3)

        # Give different grades (no grade convergence) and varying outputs (no similarity)
        # but zero critiques for window_size
        cd.record_iteration(
            iteration_num=1,
            role="reviewer",
            output="unique review alpha",
            review=_make_review(grade="C", critiques=[]),
        )
        cd.record_iteration(
            iteration_num=2,
            role="reviewer",
            output="unique review beta",
            review=_make_review(grade="B", critiques=[]),
        )
        cd.record_iteration(
            iteration_num=3,
            role="reviewer",
            output="unique review gamma",
            review=_make_review(grade="A", critiques=[]),
        )

        converged, reason = cd.should_stop()
        assert converged
        # Should be critique-count signal (all different grades, but 0 critiques)
        assert "CRITICAL/HIGH" in reason

    def test_convergence_disabled_in_engine(self):
        """When convergence_enabled=False in config, engine skips convergence check."""
        from backend.adapters.mock import MockAdapter
        from backend.orchestrator.engine import LoopEngine

        adapter = MockAdapter(min_delay=0.01, max_delay=0.02)
        events = []

        def on_event(etype, data):
            events.append(etype)

        loop = LoopDefinition(
            name="simple",
            nodes=[
                LoopNode(id="exec", role="executor", prompt_ref="exec/test"),
                LoopNode(id="review", role="reviewer", prompt_ref="review/test"),
            ],
            edges=[
                LoopEdge(source="exec", target="review", condition="always"),
                LoopEdge(source="review", target="exec", condition="always"),
            ],
            max_iterations=6,
        )
        run_state = _make_run_state(max_iterations=6, config={"convergence_enabled": False})
        engine = LoopEngine(adapter=adapter, on_event=on_event)

        result = asyncio.get_event_loop().run_until_complete(
            engine.run_loop(run_state, loop)
        )
        # Should NOT have emitted convergence_detected
        assert "convergence_detected" not in events
        # Should have reached budget/iteration limit instead
        assert result.current_iteration == 6

    def test_get_metrics_returns_expected_keys(self):
        """Metrics dict has all expected keys."""
        cd = ConvergenceDetector()
        cd.record_iteration(
            iteration_num=1,
            role="reviewer",
            output="test",
            review=_make_review(grade="B"),
        )
        metrics = cd.get_metrics()
        assert "iteration_count" in metrics
        assert "grade_history" in metrics
        assert "critique_counts" in metrics
        assert "similarity_scores" in metrics
        assert "window_size" in metrics
        assert "similarity_threshold" in metrics
        assert "min_iterations" in metrics
        assert metrics["grade_history"] == ["B"]

    def test_different_roles_tracked_separately(self):
        """Reviewer grades vs executor output are tracked independently."""
        cd = ConvergenceDetector(window_size=2, min_iterations=3)

        # Executor outputs are very different
        cd.record_iteration(iteration_num=1, role="executor", output="first draft about topic A")
        cd.record_iteration(iteration_num=2, role="executor", output="completely different topic B")
        cd.record_iteration(iteration_num=3, role="executor", output="yet another direction C")

        # Reviewer grades are stable
        for i in range(1, 4):
            cd.record_iteration(
                iteration_num=i,
                role="reviewer",
                output=f"review output {i}",
                review=_make_review(grade="A"),
            )

        converged, reason = cd.should_stop()
        assert converged
        # Grade stabilization triggers but output similarity does not
        assert "Grade stabilized" in reason

        # Check metrics: executor outputs not tracked in similarity since they differ
        metrics = cd.get_metrics()
        assert len(metrics["grade_history"]) == 3
        # Similarity scores come from executor outputs only
        assert len(metrics["similarity_scores"]) == 2  # 2 comparisons from 3 executor outputs

    def test_reviewer_output_not_tracked_for_similarity(self):
        """Reviewer output should NOT contribute to output similarity tracking."""
        cd = ConvergenceDetector(window_size=2, similarity_threshold=0.9, min_iterations=3)

        # Only reviewer iterations with identical output but varying grades
        same_review = "This is a fine analysis with no issues."
        grades = ["C", "B", "B+", "A-"]
        for i, g in zip(range(1, 5), grades):
            cd.record_iteration(
                iteration_num=i,
                role="reviewer",
                output=same_review,
                review=_make_review(grade=g, critiques=[
                    ReviewCritique(severity=SeverityLevel.HIGH, category="stats",
                                   description="Missing p-value"),
                ]),
            )

        converged, reason = cd.should_stop()
        # Should NOT converge on similarity because reviewer outputs are excluded
        # and critiques are not zero
        assert not converged

    def test_jaccard_edge_cases(self):
        """Verify similarity computation handles edge cases."""
        cd = ConvergenceDetector(window_size=2, similarity_threshold=0.5, min_iterations=3)

        # Empty strings
        cd.record_iteration(iteration_num=1, role="executor", output="")
        cd.record_iteration(iteration_num=2, role="executor", output="")
        cd.record_iteration(iteration_num=3, role="executor", output="")

        metrics = cd.get_metrics()
        # Empty-to-empty similarity = 1.0
        assert all(s == 1.0 for s in metrics["similarity_scores"])
