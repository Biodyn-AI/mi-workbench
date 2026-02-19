"""Convergence detection for the loop engine.

Monitors grade stabilization, critique count trends, and output similarity
to automatically stop iteration when quality has stabilized.
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.models import ReviewResult, SeverityLevel

logger = logging.getLogger(__name__)

# Numeric mapping for letter grades
_GRADE_SCORES: dict[str, float] = {
    "A+": 4.3, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
    "PASS": 4.0, "FAIL": 0.0,
}


def _grade_to_score(grade: str) -> float:
    """Convert a letter grade to a numeric score."""
    return _GRADE_SCORES.get(grade.upper(), 2.0)


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class ConvergenceDetector:
    """Detects when the review loop has converged and further iterations
    are unlikely to improve output quality."""

    def __init__(
        self,
        window_size: int = 3,
        similarity_threshold: float = 0.9,
        min_iterations: int = 4,
    ):
        self._window_size = window_size
        self._similarity_threshold = similarity_threshold
        self._min_iterations = min_iterations

        # Tracked state
        self._grade_history: list[str] = []
        self._critique_counts: list[int] = []  # CRITICAL + HIGH per review
        self._executor_outputs: list[str] = []
        self._similarity_scores: list[float] = []
        self._iteration_count = 0

    def record_iteration(
        self,
        iteration_num: int,
        role: str,
        output: str,
        review: Optional[ReviewResult] = None,
    ) -> None:
        """Record data from one iteration for convergence tracking."""
        self._iteration_count = max(self._iteration_count, iteration_num)

        if review is not None:
            # Track grade
            if review.overall_grade:
                self._grade_history.append(review.overall_grade.upper())

            # Track CRITICAL + HIGH critique count
            severe = sum(
                1 for c in review.critiques
                if c.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)
            )
            self._critique_counts.append(severe)

        if role not in ("reviewer", "adversarial_reviewer"):
            # Track executor output similarity
            if self._executor_outputs:
                sim = _jaccard_similarity(self._executor_outputs[-1], output)
                self._similarity_scores.append(sim)
            self._executor_outputs.append(output)

    def should_stop(self) -> tuple[bool, str]:
        """Check whether convergence has been detected.

        Returns (converged, reason). Convergence requires at least
        min_iterations to have passed, then ANY single signal suffices.
        """
        if self._iteration_count < self._min_iterations:
            return False, ""

        # Signal 1: Grade stabilization
        if len(self._grade_history) >= self._window_size:
            window = self._grade_history[-self._window_size:]
            if len(set(window)) == 1:
                return True, f"Grade stabilized at {window[0]} for {self._window_size} consecutive reviews"

        # Signal 2: CRITICAL+HIGH critiques drop to 0
        if len(self._critique_counts) >= self._window_size:
            window = self._critique_counts[-self._window_size:]
            if all(c == 0 for c in window):
                return True, f"Zero CRITICAL/HIGH critiques for {self._window_size} consecutive reviews"

        # Signal 3: Executor output similarity
        if len(self._similarity_scores) >= self._window_size:
            window = self._similarity_scores[-self._window_size:]
            if all(s >= self._similarity_threshold for s in window):
                avg = sum(window) / len(window)
                return True, (
                    f"Executor output similarity >= {self._similarity_threshold} "
                    f"for {self._window_size} iterations (avg={avg:.3f})"
                )

        return False, ""

    def get_metrics(self) -> dict:
        """Return current convergence metrics for API/WebSocket reporting."""
        return {
            "iteration_count": self._iteration_count,
            "grade_history": list(self._grade_history),
            "critique_counts": list(self._critique_counts),
            "similarity_scores": list(self._similarity_scores),
            "window_size": self._window_size,
            "similarity_threshold": self._similarity_threshold,
            "min_iterations": self._min_iterations,
        }
