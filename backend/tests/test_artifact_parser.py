"""Tests for the artifact parser."""
import pytest

from backend.artifacts.parser import ArtifactParser


@pytest.fixture
def parser():
    return ArtifactParser()


class TestArtifactParser:
    def test_parse_explicit_mech_marker(self, parser):
        output = "# MECH.md\n\nSome mechanistic analysis content here."
        result = parser.parse_output(output)
        assert "MECH.md" in result
        assert "mechanistic analysis" in result["MECH.md"]

    def test_parse_multiple_artifacts(self, parser):
        output = (
            "# MECH.md\n\nMech content here.\n\n"
            "# EVAL.md\n\nEvaluation content here.\n\n"
            "# XP.md\n\nExperiment plan content."
        )
        result = parser.parse_output(output)
        assert len(result) == 3
        assert "MECH.md" in result
        assert "EVAL.md" in result
        assert "XP.md" in result
        assert "Mech content" in result["MECH.md"]
        assert "Evaluation content" in result["EVAL.md"]
        assert "Experiment plan" in result["XP.md"]

    def test_parse_section_headers(self, parser):
        output = (
            "# Mechanistic Analysis\n\nSome mech content.\n\n"
            "# Evaluation\n\nSome eval content."
        )
        result = parser.parse_output(output)
        assert "MECH.md" in result
        assert "EVAL.md" in result

    def test_assign_by_role_executor(self, parser):
        output = "This is some unstructured output blob."
        result = parser.parse_output(output, role="executor")
        assert "MECH.md" in result
        assert "unstructured output" in result["MECH.md"]

    def test_assign_by_role_reviewer(self, parser):
        output = "This is some review output."
        result = parser.parse_output(output, role="reviewer")
        assert "EVAL.md" in result

    def test_empty_output(self, parser):
        result = parser.parse_output("")
        assert result == {}
        result2 = parser.parse_output("   \n  ")
        assert result2 == {}

    def test_mixed_markers_and_sections(self, parser):
        """Explicit markers take priority over section headers."""
        output = (
            "# MECH.md\n\nExplicit mech content.\n\n"
            "# Evaluation\n\nThis should be part of MECH since explicit won."
        )
        result = parser.parse_output(output)
        # Explicit markers are found first and used
        assert "MECH.md" in result

    def test_preserves_content_integrity(self, parser):
        content = (
            "# MECH.md\n\n"
            "## Summary\n"
            "Attention weight analysis across 18 transformer layers.\n\n"
            "## Key Findings\n"
            "- Best layer: L15\n"
            "- Correlation baseline: 0.703\n"
            "- Confound decomposition: 24% retained"
        )
        result = parser.parse_output(content)
        assert "MECH.md" in result
        parsed = result["MECH.md"]
        assert "## Summary" in parsed
        assert "## Key Findings" in parsed
        assert "L15" in parsed
        assert "0.703" in parsed

    def test_yaml_frontmatter_detection(self, parser):
        output = (
            "---\n"
            "artifact_type: EVAL\n"
            "version: 1.0\n"
            "---\n"
            "This is evaluation content from YAML frontmatter."
        )
        result = parser.parse_output(output)
        assert "EVAL.md" in result
        assert "evaluation content" in result["EVAL.md"]

    def test_role_fallback_unknown_role(self, parser):
        output = "Generic output from a custom role."
        result = parser.parse_output(output, role="custom_analyzer")
        assert "custom_analyzer_output.md" in result

    def test_no_role_fallback(self, parser):
        output = "Generic output with no role hint."
        result = parser.parse_output(output)
        assert "output.md" in result

    def test_case_insensitive_markers(self, parser):
        output = "# mech.md\n\nLower-case marker content."
        result = parser.parse_output(output)
        assert "MECH.md" in result

    def test_h2_explicit_markers(self, parser):
        output = "## EVAL.md\n\nH2-level marker content."
        result = parser.parse_output(output)
        assert "EVAL.md" in result
