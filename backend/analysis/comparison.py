"""Multi-run comparison logic for MI-Workbench."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from backend.models import RunState, RunStatus


class RunComparator:
    """Compare multiple runs on metrics, outputs, and convergence."""

    def compare_metrics(self, runs: list[RunState]) -> dict:
        """Compare basic metrics across runs."""
        run_metrics = []
        for r in runs:
            duration_s = 0.0
            if r.started_at and r.stopped_at:
                duration_s = (r.stopped_at - r.started_at).total_seconds()
            run_metrics.append({
                "run_id": r.run_id,
                "iterations": r.current_iteration,
                "tokens": r.total_tokens,
                "cost": r.total_cost,
                "duration_s": duration_s,
                "status": r.status.value,
            })

        result: dict = {"runs": run_metrics}

        if run_metrics:
            result["best_by_cost"] = min(run_metrics, key=lambda m: m["cost"])["run_id"]
            result["best_by_tokens"] = min(run_metrics, key=lambda m: m["tokens"])["run_id"]
            # Fewest iterations among completed runs; fall back to all if none completed
            completed = [m for m in run_metrics if m["status"] == RunStatus.COMPLETED.value]
            pool = completed if completed else run_metrics
            result["best_by_iterations"] = min(pool, key=lambda m: m["iterations"])["run_id"]

        return result

    def compare_outputs(
        self, runs: list[RunState], artifact_base_paths: list[str]
    ) -> dict:
        """Compare final artifacts across runs using word-level Jaccard similarity."""
        # Collect artifacts per run
        all_artifacts: dict[str, dict[str, str]] = {}
        for run, base_path in zip(runs, artifact_base_paths):
            all_artifacts[run.run_id] = self._read_final_artifacts(run.run_id, base_path)

        # Find artifact names present in any run
        all_names: set[str] = set()
        for arts in all_artifacts.values():
            all_names.update(arts.keys())

        run_ids = [r.run_id for r in runs]
        comparisons: dict = {}
        for name in sorted(all_names):
            n = len(run_ids)
            matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    text_a = all_artifacts.get(run_ids[i], {}).get(name, "")
                    text_b = all_artifacts.get(run_ids[j], {}).get(name, "")
                    matrix[i][j] = self._jaccard_similarity(text_a, text_b)
            comparisons[name] = {
                "pairwise_similarity": matrix,
                "run_ids": run_ids,
            }

        return {"artifact_comparisons": comparisons}

    def compare_quality(self, runs: list[RunState]) -> dict:
        """Compare final reviewer grades and critique counts."""
        run_quality = []
        for r in runs:
            feedback = self._get_last_reviewer_feedback(r)
            grade = self._extract_grade_from_feedback(feedback) if feedback else ""
            counts = self._count_critiques_by_severity(feedback) if feedback else {}
            total = sum(counts.values())
            run_quality.append({
                "run_id": r.run_id,
                "final_grade": grade,
                "critical_count": counts.get("critical", 0),
                "high_count": counts.get("high", 0),
                "total_critiques": total,
            })

        result: dict = {"runs": run_quality}
        graded = [q for q in run_quality if q["final_grade"]]
        if graded:
            grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
            result["best_by_grade"] = min(
                graded,
                key=lambda q: grade_order.get(q["final_grade"].upper(), 99),
            )["run_id"]

        return result

    def generate_summary(
        self, runs: list[RunState], artifact_base_paths: list[str]
    ) -> dict:
        """Generate a combined comparison summary."""
        return {
            "metrics": self.compare_metrics(runs),
            "outputs": self.compare_outputs(runs, artifact_base_paths),
            "quality": self.compare_quality(runs),
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Word-level Jaccard similarity between two texts."""
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        if not words_a and not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _read_final_artifacts(run_id: str, base_path: str) -> dict[str, str]:
        """Read MECH.md, EVAL.md, XP.md etc. from the latest iteration directory."""
        base = Path(base_path)
        if not base.exists():
            return {}
        # Find highest-numbered iteration directory
        iter_dirs = sorted(
            [d for d in base.iterdir() if d.is_dir() and d.name.startswith("iter_")],
            key=lambda d: d.name,
        )
        if not iter_dirs:
            # Fall back to base path itself
            target = base
        else:
            target = iter_dirs[-1]

        artifacts: dict[str, str] = {}
        for suffix in ("MECH.md", "EVAL.md", "XP.md", "METHOD.md"):
            fpath = target / suffix
            if fpath.exists():
                artifacts[suffix] = fpath.read_text(errors="replace")
        return artifacts

    @staticmethod
    def _extract_grade_from_feedback(feedback: str) -> str:
        """Parse a letter grade (A-F) from reviewer feedback text."""
        # Match patterns like "Grade: A", "Overall grade: B", "grade=C"
        m = re.search(r"[Gg]rade[\s:=]+([A-Fa-f])\b", feedback)
        if m:
            return m.group(1).upper()
        return ""

    @staticmethod
    def _count_critiques_by_severity(feedback: str) -> dict[str, int]:
        """Count critique lines tagged by severity in reviewer feedback."""
        counts: dict[str, int] = {}
        for level in ("critical", "high", "medium", "low", "info"):
            # Match patterns like "[CRITICAL]", "(high)", "severity: medium"
            pattern = rf"(?:\[{level}\]|\({level}\)|severity:\s*{level})"
            matches = re.findall(pattern, feedback, re.IGNORECASE)
            if matches:
                counts[level] = len(matches)
        return counts

    @staticmethod
    def _get_last_reviewer_feedback(run: RunState) -> Optional[str]:
        """Get feedback from the last reviewer iteration in a run."""
        for it in reversed(run.iterations):
            if it.role == "reviewer" and it.feedback:
                return it.feedback
        return None
