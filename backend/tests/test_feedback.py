"""Tests for the feedback formatter."""
import pytest

from backend.models import ReviewCritique, ReviewResult, SeverityLevel
from backend.orchestrator.feedback import FeedbackFormatter


@pytest.fixture
def formatter():
    return FeedbackFormatter()


class TestFeedbackFormatter:
    def test_parse_bracket_format(self, formatter):
        output = (
            "Some preamble text.\n\n"
            "[CRITICAL] The causal claim in section 2 is unsupported.\n"
            "[HIGH] Missing negative control for co-expression.\n"
            "[LOW] Typo in figure caption.\n"
        )
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 3
        severities = [c.severity for c in review.critiques]
        assert SeverityLevel.CRITICAL in severities
        assert SeverityLevel.HIGH in severities
        assert SeverityLevel.LOW in severities

    def test_parse_markdown_format(self, formatter):
        output = (
            "## Statistical Concerns\n"
            "- **Severity: MEDIUM** — Attention-perturbation correlation is weak.\n"
            "- **Severity: LOW** — Per-TF bootstrap CIs show limited significance.\n"
        )
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 2
        assert any(c.severity == SeverityLevel.MEDIUM for c in review.critiques)
        assert any(c.severity == SeverityLevel.LOW for c in review.critiques)

    def test_parse_structured_format(self, formatter):
        output = (
            "- Severity: HIGH, Category: statistics, Description: P-values not corrected\n"
            "- Severity: CRITICAL, Category: methodology, Description: No negative control\n"
        )
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 2
        categories = [c.category for c in review.critiques]
        assert "statistics" in categories
        assert "methodology" in categories

    def test_parse_bulleted_concerns(self, formatter):
        output = (
            "# Concerns\n"
            "- The sample size is too small for this claim\n"
            "- Random seed not documented\n"
            "\n# Next Steps\n"
            "Continue analysis."
        )
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 2
        descriptions = [c.description for c in review.critiques]
        assert any("sample size" in d for d in descriptions)
        assert any("Random seed" in d for d in descriptions)

    def test_format_for_executor_has_sections(self, formatter):
        review = ReviewResult(
            overall_grade="B",
            critiques=[
                ReviewCritique(severity=SeverityLevel.CRITICAL, category="stats",
                               description="Fix the causal claim"),
                ReviewCritique(severity=SeverityLevel.LOW, category="style",
                               description="Typo in caption"),
            ],
            reproducibility_gaps=["Random seed not documented"],
            suspected_confounders=["Cell type composition"],
        )
        formatted = formatter.format_for_executor(review)
        assert "=== REVIEWER FEEDBACK (Grade: B) ===" in formatted
        assert "REQUIRED FIXES" in formatted
        assert "SUGGESTED IMPROVEMENTS" in formatted
        assert "REPRODUCIBILITY GAPS" in formatted
        assert "SUSPECTED CONFOUNDERS" in formatted
        assert "=== END FEEDBACK ===" in formatted

    def test_format_required_fixes_first(self, formatter):
        review = ReviewResult(
            critiques=[
                ReviewCritique(severity=SeverityLevel.LOW, category="style",
                               description="Minor typo"),
                ReviewCritique(severity=SeverityLevel.CRITICAL, category="stats",
                               description="Must fix causal claim"),
            ],
        )
        formatted = formatter.format_for_executor(review)
        # REQUIRED FIXES section should come before SUGGESTED IMPROVEMENTS
        req_pos = formatted.find("REQUIRED FIXES")
        sug_pos = formatted.find("SUGGESTED IMPROVEMENTS")
        assert req_pos < sug_pos
        # Critical item should appear under REQUIRED FIXES (with category tag)
        assert "[CRITICAL | stats] Must fix causal claim" in formatted

    def test_extract_action_items(self, formatter):
        review = ReviewResult(
            critiques=[
                ReviewCritique(severity=SeverityLevel.HIGH, category="stats",
                               description="Add negative control",
                               required_fix="Add a negative control experiment"),
                ReviewCritique(severity=SeverityLevel.MEDIUM, category="style",
                               description="Consider bootstrapping CIs"),
            ],
        )
        items = formatter.extract_action_items(review)
        assert len(items) == 2
        # required_fix takes priority over description
        assert items[0] == "Add a negative control experiment"
        assert items[1] == "Consider bootstrapping CIs"

    def test_severity_keyword_detection(self, formatter):
        output = (
            "Some general output.\n"
            "[INFO] This is informational.\n"
            "[CRITICAL] This must fix a blocking issue.\n"
        )
        review = formatter.parse_review(output)
        info_critiques = [c for c in review.critiques if c.severity == SeverityLevel.INFO]
        crit_critiques = [c for c in review.critiques if c.severity == SeverityLevel.CRITICAL]
        assert len(info_critiques) >= 1
        assert len(crit_critiques) >= 1

    def test_grade_extraction(self, formatter):
        output = "Overall Grade: B+\n\nSome review content."
        review = formatter.parse_review(output)
        assert review.overall_grade == "B+"

    def test_reproducibility_gaps_extraction(self, formatter):
        output = (
            "## Reproducibility Gaps\n"
            "- Random seed not documented\n"
            "- Split not specified\n"
            "\n## Other\nSomething else."
        )
        review = formatter.parse_review(output)
        assert len(review.reproducibility_gaps) == 2
        assert "Random seed not documented" in review.reproducibility_gaps

    def test_suspected_confounders_extraction(self, formatter):
        output = (
            "## Suspected Confounders\n"
            "- Cell type composition may drive correlation\n"
            "- Batch effects\n"
        )
        review = formatter.parse_review(output)
        assert len(review.suspected_confounders) == 2

    def test_empty_output(self, formatter):
        review = formatter.parse_review("")
        assert review.critiques == []
        assert review.overall_grade == ""

    def test_format_no_critiques(self, formatter):
        review = ReviewResult()
        formatted = formatter.format_for_executor(review)
        assert "=== REVIEWER FEEDBACK ===" in formatted
        assert "=== END FEEDBACK ===" in formatted

    def test_infer_artifact_ref_from_description(self, formatter):
        output = "[HIGH] The protocol in PROTOCOL.md is missing negative controls.\n"
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 1
        proto_critique = [c for c in review.critiques if c.artifact_ref == "PROTOCOL.md"]
        assert len(proto_critique) >= 1

    def test_infer_section_ref_from_description(self, formatter):
        output = "[HIGH] Section 2.1 Controls in PROTOCOL.md lacks degree-matched nulls.\n"
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 1
        c = review.critiques[0]
        assert c.artifact_ref == "PROTOCOL.md"
        assert c.section_ref is not None
        assert "2.1" in c.section_ref

    def test_infer_refs_mech_md(self, formatter):
        output = "[CRITICAL] Effect size not reported in MECH.md Section 3 Results.\n"
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 1
        c = review.critiques[0]
        assert c.artifact_ref == "MECH.md"
        assert c.section_ref is not None
        assert "3" in c.section_ref

    def test_format_includes_location_tags(self, formatter):
        review = ReviewResult(
            overall_grade="C",
            critiques=[
                ReviewCritique(
                    severity=SeverityLevel.HIGH,
                    category="reproducibility",
                    description="Missing dataset URLs",
                    artifact_ref="PROTOCOL.md",
                    section_ref="Section 2.1",
                ),
                ReviewCritique(
                    severity=SeverityLevel.CRITICAL,
                    category="statistics",
                    description="No multiple testing correction",
                ),
            ],
        )
        formatted = formatter.format_for_executor(review)
        assert "PROTOCOL.md § Section 2.1" in formatted
        assert "reproducibility" in formatted
        assert "statistics" in formatted

    def test_no_refs_when_absent(self, formatter):
        output = "[MEDIUM] The analysis could be improved.\n"
        review = formatter.parse_review(output)
        assert len(review.critiques) >= 1
        c = review.critiques[0]
        assert c.artifact_ref is None
        assert c.section_ref is None
