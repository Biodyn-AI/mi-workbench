"""Tests for the Reviewer Consensus Mode."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models import (
    AdapterRunResult,
    ReviewCritique,
    ReviewResult,
    SeverityLevel,
    WorkspaceContext,
)
from backend.orchestrator.consensus import ConsensusReviewer, _parse_severity, _escalate_severity


@pytest.fixture
def consensus():
    return ConsensusReviewer()


# ── Parsing Tests ───────────────────────────────────────────────────


class TestParseCritiques:
    def test_parse_critique_from_markdown(self, consensus):
        output = (
            "Some preamble text.\n"
            "**Severity: HIGH** -- Missing effect size for AUROC comparison.\n"
            "**Severity: MEDIUM** -- No confidence intervals reported.\n"
            "Some trailing text."
        )
        critiques = consensus._parse_review_output(output)
        assert len(critiques) == 2
        assert critiques[0].severity == SeverityLevel.HIGH
        assert "effect size" in critiques[0].description.lower()
        assert critiques[1].severity == SeverityLevel.MEDIUM
        assert "confidence intervals" in critiques[1].description.lower()

    def test_parse_critique_from_severity_format(self, consensus):
        output = (
            "Severity: critical, Category: statistics, Description: "
            "P-value reported without multiple testing correction.\n"
            "Severity: low, Category: reproducibility, Description: "
            "Random seed not documented."
        )
        critiques = consensus._parse_review_output(output)
        assert len(critiques) == 2
        assert critiques[0].severity == SeverityLevel.CRITICAL
        assert critiques[0].category == "statistics"
        assert "multiple testing" in critiques[0].description.lower()
        assert critiques[1].severity == SeverityLevel.LOW
        assert critiques[1].category == "reproducibility"

    def test_parse_critique_from_bracket_format(self, consensus):
        output = (
            "[CRITICAL] Core claim about causal regulation is unsupported.\n"
            "[HIGH] Confound analysis missing degree-preserving null.\n"
            "[INFO] Consider adding a supplementary table.\n"
        )
        critiques = consensus._parse_review_output(output)
        assert len(critiques) == 3
        assert critiques[0].severity == SeverityLevel.CRITICAL
        assert critiques[1].severity == SeverityLevel.HIGH
        assert critiques[2].severity == SeverityLevel.INFO

    def test_parse_empty_output(self, consensus):
        assert consensus._parse_review_output("") == []
        assert consensus._parse_review_output("No issues found.") == []


# ── Merge Tests ─────────────────────────────────────────────────────


class TestMergeCritiques:
    def _make_result(self, output: str) -> AdapterRunResult:
        return AdapterRunResult(success=True, output=output)

    def test_merge_deduplicates_similar(self, consensus):
        r1 = self._make_result(
            "[HIGH] Missing effect size for the AUROC comparison between layers."
        )
        r2 = self._make_result(
            "[HIGH] The AUROC comparison between layers is missing effect size."
        )
        r3 = self._make_result(
            "[LOW] Consider using a different color scheme for figures."
        )
        result = consensus.merge_critiques([r1, r2, r3])
        # The two similar HIGH critiques should be merged, the LOW is separate
        assert len(result.critiques) == 2
        # The deduplicated one should have 2 reviewer tags
        high_critiques = [c for c in result.critiques if c.severity == SeverityLevel.CRITICAL
                          or c.severity == SeverityLevel.HIGH]
        assert len(high_critiques) >= 1

    def test_merge_escalates_severity(self, consensus):
        # Two reviewers flag the same issue as MEDIUM -> should escalate to HIGH
        r1 = self._make_result(
            "[MEDIUM] No null model comparison for the attention-derived network AUROC."
        )
        r2 = self._make_result(
            "[MEDIUM] Missing null model comparison for the attention-derived network AUROC."
        )
        r3 = self._make_result("")  # third reviewer has nothing
        result = consensus.merge_critiques([r1, r2, r3])
        # Should be escalated because 2 reviewers flagged it
        escalated = [c for c in result.critiques
                     if "null model" in c.description.lower()]
        assert len(escalated) == 1
        assert escalated[0].severity == SeverityLevel.HIGH

    def test_merge_ranks_by_severity(self, consensus):
        r1 = self._make_result(
            "[LOW] Minor formatting issue in table headers."
        )
        r2 = self._make_result(
            "[CRITICAL] Core statistical test is inappropriate for the data distribution."
        )
        r3 = self._make_result(
            "[MEDIUM] Expression level validation is missing."
        )
        result = consensus.merge_critiques([r1, r2, r3])
        assert len(result.critiques) == 3
        # Verify sorted by severity descending
        severities = [c.severity for c in result.critiques]
        assert severities[0] == SeverityLevel.CRITICAL
        assert severities[1] == SeverityLevel.MEDIUM
        assert severities[2] == SeverityLevel.LOW

    def test_merge_empty_results(self, consensus):
        # Updated (E1): empty output and output with no recognisable critique
        # format are lens failures, not clean reviews. Previously this panel
        # silently merged to grade "A"; it must now be reported INCOMPLETE.
        r1 = self._make_result("")
        r2 = self._make_result("All looks good, no issues.")
        r3 = self._make_result("")
        result = consensus.merge_critiques([r1, r2, r3])
        assert len(result.critiques) == 0
        assert result.overall_grade == "INCOMPLETE"
        assert result.consensus_meta["panel_failed"] is True
        assert result.consensus_meta["lens_status"] == {
            "reviewer": "empty",
            "adversarial_reviewer": "unparsed",
            "bio_plausibility_checker": "empty",
        }

    def test_merge_explicit_clean_reviews_grade_a(self, consensus):
        # A clean review under the JSON contract is an explicit empty list.
        clean = self._make_result('```json\n{"critiques": [], "overall_assessment": "Sound."}\n```')
        result = consensus.merge_critiques([clean, clean, clean])
        assert result.critiques == []
        assert result.overall_grade == "A"
        assert result.consensus_meta["panel_failed"] is False
        assert result.consensus_meta["failed_lenses"] == []


# ── Similarity Tests ────────────────────────────────────────────────


class TestSimilarity:
    def test_similarity_score_identical(self, consensus):
        text = "Missing effect size for AUROC comparison"
        score = consensus._similarity_score(text, text)
        assert score == 1.0

    def test_similarity_score_different(self, consensus):
        a = "Missing effect size for AUROC comparison"
        b = "Color scheme of the scatter plot needs improvement"
        score = consensus._similarity_score(a, b)
        assert score < 0.3

    def test_similarity_score_partial_overlap(self, consensus):
        a = "No null model comparison for the attention network analysis"
        b = "Attention network analysis lacks a proper null model"
        score = consensus._similarity_score(a, b)
        assert score > 0.4

    def test_similarity_score_empty(self, consensus):
        assert consensus._similarity_score("", "something") == 0.0
        assert consensus._similarity_score("", "") == 0.0


# ── Consensus Runner Tests ──────────────────────────────────────────


class TestConsensusRunner:
    @pytest.mark.asyncio
    async def test_consensus_runs_parallel(self):
        """Mock adapter should be called 3 times (once per reviewer role)."""
        mock_adapter = AsyncMock()
        mock_adapter.run.return_value = AdapterRunResult(
            success=True,
            output="[HIGH] Test critique from mock reviewer.",
        )

        # Lens calls only: the lexical merge makes no adjudicator call (the
        # default llm method adds one; see test_merge_defaults.py).
        cr = ConsensusReviewer(similarity_method="jaccard")
        ws_ctx = WorkspaceContext(workspace_path="/tmp/test")
        result = await cr.run_consensus("artifact content", mock_adapter, ws_ctx)

        assert mock_adapter.run.call_count == 3
        assert isinstance(result, ReviewResult)
        assert len(result.critiques) > 0


# ── Helper Tests ────────────────────────────────────────────────────


class TestHelpers:
    def test_parse_severity_valid(self):
        assert _parse_severity("critical") == SeverityLevel.CRITICAL
        assert _parse_severity("HIGH") == SeverityLevel.HIGH
        assert _parse_severity("Medium") == SeverityLevel.MEDIUM
        assert _parse_severity("low") == SeverityLevel.LOW
        assert _parse_severity("info") == SeverityLevel.INFO

    def test_parse_severity_unknown_defaults_medium(self):
        assert _parse_severity("unknown") == SeverityLevel.MEDIUM

    def test_escalate_severity(self):
        assert _escalate_severity(SeverityLevel.INFO) == SeverityLevel.LOW
        assert _escalate_severity(SeverityLevel.LOW) == SeverityLevel.MEDIUM
        assert _escalate_severity(SeverityLevel.MEDIUM) == SeverityLevel.HIGH
        assert _escalate_severity(SeverityLevel.HIGH) == SeverityLevel.CRITICAL
        assert _escalate_severity(SeverityLevel.CRITICAL) == SeverityLevel.CRITICAL

    def test_grade_computation(self, consensus):
        # Updated (E1): "no critiques -> A" only holds for a lens that actually
        # returned a (clean) review; an empty output is a failed lens and the
        # panel is INCOMPLETE rather than silently graded A.
        clean = self._make_result('```json\n{"critiques": []}\n```')
        assert consensus.merge_critiques([clean]).overall_grade == "A"
        r = self._make_result("")
        assert consensus.merge_critiques([r]).overall_grade == "INCOMPLETE"

    def test_thinking_traces_stripped_before_parsing(self, consensus):
        """Traces like 'I will read...' should not affect critique parsing."""
        output_with_traces = (
            "I will read the PROTOCOL.md file to assess its state.\n"
            "I will list the contents of the knowledge directory.\n"
            "\n"
            "[HIGH] Missing negative control for co-expression.\n"
            "[MEDIUM] Effect size not reported.\n"
        )
        r1 = self._make_result(output_with_traces)
        r2 = self._make_result("")
        r3 = self._make_result("")
        result = consensus.merge_critiques([r1, r2, r3])
        # Traces should not create extra critiques
        assert len(result.critiques) == 2
        descriptions = " ".join(c.description for c in result.critiques)
        assert "negative control" in descriptions.lower()
        assert "effect size" in descriptions.lower()

    def _make_result(self, output: str) -> AdapterRunResult:
        return AdapterRunResult(success=True, output=output)
