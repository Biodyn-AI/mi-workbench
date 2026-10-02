"""E1: single-reviewer parsing (FeedbackFormatter.parse_review) - JSON/YAML
contract support and regressions for the legacy regex bugs."""
from __future__ import annotations

import pytest

from backend.adapters import mock as mock_mod
from backend.models import ReviewCritique, ReviewResult, SeverityLevel
from backend.orchestrator.feedback import FeedbackFormatter, _parse_severity


@pytest.fixture
def fmt():
    return FeedbackFormatter()


class TestRegexRegressions:
    def test_highly_significant_is_not_a_high_critique(self, fmt):
        review = fmt.parse_review("The results are highly significant (p < 0.001).\n")
        assert review.critiques == []
        assert not any("ly significant" in c.description for c in review.critiques)

    def test_lower_layers_is_not_a_low_critique(self, fmt):
        review = fmt.parse_review("Attention in lower layers is diffuse.\n")
        assert review.critiques == []

    def test_other_embedded_severity_words(self, fmt):
        text = ("Follow-up Experiments\n"
                "The information content is high-dimensional.\n"
                "Mediumship aside, criticality analysis was run.\n"
                "High-dimensional data is hard.\n")
        assert fmt.parse_review(text).critiques == []

    def test_adversarial_grade_compromised_is_not_grade_c(self, fmt):
        review = fmt.parse_review("adversarial_grade: compromised\nplausibility_grade: implausible\n")
        assert review.overall_grade == ""

    def test_grade_word_boundaries(self, fmt):
        assert fmt.parse_review("Overall Grade: B+\n").overall_grade == "B+"
        assert fmt.parse_review("overall_grade: B-\n").overall_grade == "B-"
        assert fmt.parse_review("**Overall Grade:** C\n").overall_grade == "C"
        assert fmt.parse_review("Grade: pass").overall_grade == "PASS"
        assert fmt.parse_review("Rating: Excellent work").overall_grade == ""
        assert fmt.parse_review("Score: 0.85").overall_grade == ""

    def test_brackets_still_parse_anywhere_on_line(self, fmt):
        review = fmt.parse_review("1. [HIGH] Missing null model.\nNote - [low] typo\n")
        assert [(c.severity, c.description) for c in review.critiques] == [
            (SeverityLevel.HIGH, "Missing null model."),
            (SeverityLevel.LOW, "typo"),
        ]

    def test_line_initial_severity_with_separator(self, fmt):
        review = fmt.parse_review(
            "HIGH: missing controls\n- **Critical** — leakage\n2. low - typo\nInfo: see appendix\n"
        )
        assert [(c.severity, c.description) for c in review.critiques] == [
            (SeverityLevel.HIGH, "missing controls"),
            (SeverityLevel.CRITICAL, "leakage"),
            (SeverityLevel.LOW, "typo"),
            (SeverityLevel.INFO, "see appendix"),
        ]
        assert review.parse_method == "regex"

    def test_bulleted_concern_with_bracket_counts_once(self, fmt):
        review = fmt.parse_review("## Issues\n- [HIGH] foo bar\n- Random seed missing\n")
        assert [(c.severity, c.description) for c in review.critiques] == [
            (SeverityLevel.HIGH, "foo bar"),
            (SeverityLevel.MEDIUM, "Random seed missing"),
        ]

    def test_structured_line_with_bracket_counts_once(self, fmt):
        review = fmt.parse_review("- Severity: HIGH, Category: stats, Description: [LOW] odd\n")
        assert len(review.critiques) == 1
        assert review.critiques[0].category == "stats"

    def test_keyword_severity_uses_word_boundaries(self):
        assert _parse_severity("The workflow is unclear") == SeverityLevel.MEDIUM
        assert _parse_severity("results are highly variable") == SeverityLevel.MEDIUM
        assert _parse_severity("a minor wording issue") == SeverityLevel.LOW
        assert _parse_severity("this is a blocking problem") == SeverityLevel.CRITICAL
        assert _parse_severity("high dropout rate") == SeverityLevel.HIGH


class TestContractInSingleReviewer:
    JSON = ('Notes.\n```json\n{"critiques": [\n'
            ' {"severity": "high", "category": "statistics", "description": '
            '"No CI for the AUROC difference in EVAL.md.", "required_fix": "Bootstrap CIs."},\n'
            ' {"severity": "low", "category": "style", "description": "Typo.", "required_fix": "Fix."}\n'
            '], "overall_assessment": "Needs work."}\n```')

    def test_json_contract(self, fmt):
        review = fmt.parse_review(self.JSON)
        assert review.parse_method == "json"
        assert [(c.severity, c.category) for c in review.critiques] == [
            (SeverityLevel.HIGH, "statistics"), (SeverityLevel.LOW, "style")]
        assert review.critiques[0].required_fix == "Bootstrap CIs."
        assert review.critiques[0].artifact_ref == "EVAL.md"  # refs still inferred
        assert review.overall_assessment == "Needs work."
        # No explicit grade: derived from severities (1 HIGH -> C)
        assert review.overall_grade == "C"

    def test_prose_grade_outside_contract_is_ignored(self, fmt):
        # Text before the block is "ignored by the pipeline": the grade comes
        # from the contract severities (1 HIGH -> C), never from prose.
        review = fmt.parse_review("Overall Grade: B\n" + self.JSON)
        assert review.overall_grade == "C"
        review = fmt.parse_review("Overall Grade: A\n" + self.JSON)
        assert review.overall_grade == "C"

    def test_json_empty_list_is_clean(self, fmt):
        review = fmt.parse_review('```json\n{"critiques": []}\n```')
        assert review.critiques == [] and review.overall_grade == "A"
        assert review.parse_method == "json"

    def test_yaml_contract(self, fmt):
        review = fmt.parse_review("- severity: critical\n  description: Leakage.\n  fix: Hold out.\n")
        assert review.parse_method == "yaml"
        assert review.critiques[0].severity == SeverityLevel.CRITICAL
        assert review.critiques[0].required_fix == "Hold out."
        assert review.overall_grade == "D"

    def test_legacy_grade_not_derived(self, fmt):
        review = fmt.parse_review("[HIGH] Missing control.\n")
        assert review.overall_grade == ""
        assert review.parse_method == "regex"

    def test_nothing_parsed(self, fmt):
        review = fmt.parse_review("Looks fine.")
        assert review.critiques == [] and review.parse_method == "none"
        assert review.overall_grade == ""


class TestMockReviewerOutputsUnchanged:
    @pytest.mark.parametrize("iteration", range(1, 7))
    @pytest.mark.parametrize("role", ["reviewer", "adversarial", "bio_plausibility"])
    def test_mock_reviewer_parse(self, fmt, role, iteration):
        out = mock_mod._OUTPUT_BUILDERS[role](iteration)
        review = fmt.parse_review(out)
        expected = [line for line in out.splitlines() if line.startswith("[")]
        assert [f"[{c.severity.value.upper()}] {c.description}" for c in review.critiques] == expected
        assert review.overall_grade == mock_mod._grade_for_iteration(iteration).upper()
        assert review.parse_method == "regex"

    def test_executor_output_yields_no_spurious_critiques(self, fmt):
        # The old optional-bracket regex read "Follow-up" as a LOW critique.
        assert fmt.parse_review(mock_mod._OUTPUT_BUILDERS["executor"](1)).critiques == []
        assert fmt.parse_review(mock_mod._OUTPUT_BUILDERS["idea_generator"](1)).critiques == []


class TestPanelStatusInFeedback:
    def test_no_meta_no_status_line(self, fmt):
        assert "PANEL STATUS" not in fmt.format_for_executor(ReviewResult(overall_grade="A"))

    def test_partial_panel_status(self, fmt):
        rr = ReviewResult(
            overall_grade="C",
            critiques=[ReviewCritique(severity=SeverityLevel.HIGH, category="g", description="x")],
            consensus_meta={"failed_lenses": ["bio_plausibility_checker"], "panel_failed": False},
        )
        text = fmt.format_for_executor(rr)
        assert "PANEL STATUS: PARTIAL" in text and "bio_plausibility_checker" in text
