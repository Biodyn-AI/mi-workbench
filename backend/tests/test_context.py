"""Tests for workspace context reader."""
import tempfile
from pathlib import Path

import pytest

from backend.orchestrator.context import MAX_ARTIFACT_CHARS, WorkspaceContextReader


def _setup_workspace(tmp: Path, run_id: str, iterations: dict[int, dict[str, str]]) -> str:
    """Create a workspace directory with iteration artifacts.

    iterations: {iter_num: {filename: content}}
    """
    for iter_num, files in iterations.items():
        iter_dir = tmp / "runs" / run_id / f"iter_{iter_num:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (iter_dir / name).write_text(content)
    return str(tmp)


class TestReadLatestArtifacts:
    def test_read_latest_artifacts_from_iterations(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run1", {
            1: {"MECH.md": "mech v1", "EVAL.md": "eval v1"},
            2: {"MECH.md": "mech v2"},
            3: {"XP.md": "xp v3"},
        })
        reader = WorkspaceContextReader(ws, "run1")
        artifacts = reader.read_latest_artifacts()
        # MECH.md should be from iter 2 (latest), EVAL.md from iter 1, XP.md from iter 3
        assert artifacts["MECH.md"] == "mech v2"
        assert artifacts["EVAL.md"] == "eval v1"
        assert artifacts["XP.md"] == "xp v3"

    def test_read_empty_workspace(self, tmp_path):
        reader = WorkspaceContextReader(str(tmp_path), "nonexistent_run")
        artifacts = reader.read_latest_artifacts()
        assert artifacts == {}

    def test_read_multiple_artifact_types(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run2", {
            1: {
                "MECH.md": "mech",
                "EVAL.md": "eval",
                "XP.md": "xp",
                "METHOD.md": "method",
            },
        })
        reader = WorkspaceContextReader(ws, "run2")
        artifacts = reader.read_latest_artifacts()
        assert len(artifacts) == 4
        assert "MECH.md" in artifacts
        assert "EVAL.md" in artifacts
        assert "XP.md" in artifacts
        assert "METHOD.md" in artifacts


class TestReadRunArtifacts:
    def test_read_specific_iteration(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run3", {
            1: {"MECH.md": "v1"},
            2: {"MECH.md": "v2"},
        })
        reader = WorkspaceContextReader(ws, "run3")
        artifacts = reader.read_run_artifacts(iteration=1)
        assert artifacts["MECH.md"] == "v1"

    def test_read_latest_iteration_by_default(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run4", {
            1: {"MECH.md": "v1"},
            2: {"MECH.md": "v2"},
        })
        reader = WorkspaceContextReader(ws, "run4")
        artifacts = reader.read_run_artifacts()
        assert artifacts["MECH.md"] == "v2"

    def test_read_nonexistent_iteration(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run5", {
            1: {"MECH.md": "v1"},
        })
        reader = WorkspaceContextReader(ws, "run5")
        artifacts = reader.read_run_artifacts(iteration=99)
        assert artifacts == {}


class TestBuildContextSection:
    def test_build_context_section_format(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run6", {
            1: {"MECH.md": "mech content", "EVAL.md": "eval content"},
        })
        reader = WorkspaceContextReader(ws, "run6")
        section = reader.build_context_section()
        assert "=== Current Artifact State ===" in section
        assert "=== End Current State ===" in section
        assert "### MECH.md (from iteration 1)" in section
        assert "### EVAL.md (from iteration 1)" in section
        assert "mech content" in section
        assert "eval content" in section

    def test_context_truncates_long_artifacts(self, tmp_path):
        long_content = "x" * (MAX_ARTIFACT_CHARS + 500)
        ws = _setup_workspace(tmp_path, "run7", {
            1: {"MECH.md": long_content},
        })
        reader = WorkspaceContextReader(ws, "run7")
        section = reader.build_context_section()
        assert "...[truncated]" in section
        # The content in the section should be at most MAX_ARTIFACT_CHARS + truncation prefix
        lines = section.split("\n")
        # Find MECH content line - should not contain the full long_content
        assert len(section) < len(long_content)

    def test_empty_workspace_returns_empty_string(self, tmp_path):
        reader = WorkspaceContextReader(str(tmp_path), "no_run")
        section = reader.build_context_section()
        assert section == ""

    def test_context_picks_latest_across_iterations(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run8", {
            1: {"MECH.md": "old mech"},
            3: {"MECH.md": "new mech"},
        })
        reader = WorkspaceContextReader(ws, "run8")
        section = reader.build_context_section()
        assert "### MECH.md (from iteration 3)" in section
        assert "new mech" in section
        assert "old mech" not in section


class TestReviewerFeedback:
    def test_reviewer_feedback_extraction(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run9", {
            1: {"executor_output.md": "exec stuff"},
            2: {"reviewer_output.md": "needs more analysis"},
        })
        reader = WorkspaceContextReader(ws, "run9")
        feedback = reader.get_reviewer_feedback_from_last_iteration()
        assert feedback == "needs more analysis"

    def test_reviewer_feedback_none_when_missing(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run10", {
            1: {"executor_output.md": "exec stuff"},
        })
        reader = WorkspaceContextReader(ws, "run10")
        feedback = reader.get_reviewer_feedback_from_last_iteration()
        assert feedback is None

    def test_reviewer_feedback_picks_latest(self, tmp_path):
        ws = _setup_workspace(tmp_path, "run11", {
            1: {"reviewer_output.md": "old feedback"},
            3: {"reviewer_output.md": "latest feedback"},
        })
        reader = WorkspaceContextReader(ws, "run11")
        feedback = reader.get_reviewer_feedback_from_last_iteration()
        assert feedback == "latest feedback"
