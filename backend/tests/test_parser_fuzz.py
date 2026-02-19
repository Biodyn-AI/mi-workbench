"""Property-based fuzz tests for all parsers using Hypothesis.

Ensures parsers never crash on arbitrary input.
"""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from backend.artifacts.parser import ArtifactParser
from backend.orchestrator.feedback import FeedbackFormatter
from backend.orchestrator.followups import FollowUpExecutor
from backend.orchestrator.convergence import ConvergenceDetector
from backend.knowledge.parser import MechParser
from backend.models import ReviewResult


# ── Shared strategies ─────────────────────────────────────────────────

# Strategy for markdown-like text
markdown_text = st.recursive(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=100,
    ),
    lambda base: st.one_of(
        base,
        st.tuples(st.just("# "), base).map(lambda t: t[0] + t[1]),
        st.tuples(st.just("## "), base).map(lambda t: t[0] + t[1]),
        st.tuples(st.just("- "), base).map(lambda t: t[0] + t[1]),
        st.tuples(st.just("[CRITICAL] "), base).map(lambda t: t[0] + t[1]),
        st.tuples(st.just("```\n"), base, st.just("\n```")).map(
            lambda t: t[0] + t[1] + t[2]
        ),
    ),
    max_leaves=10,
)


# ── ArtifactParser ────────────────────────────────────────────────────

parser = ArtifactParser()


@given(output=st.text(min_size=0, max_size=10000))
@settings(max_examples=500)
def test_artifact_parser_never_crashes(output):
    """ArtifactParser.parse_output must never raise on any input."""
    result = parser.parse_output(output, role="executor")
    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


@given(
    output=st.text(min_size=0, max_size=10000),
    role=st.sampled_from(
        ["executor", "reviewer", "adversarial", "idea_generator", "knowledge_extractor", ""]
    ),
)
@settings(max_examples=300)
def test_artifact_parser_role_variants(output, role):
    """ArtifactParser handles all role variants without crashing."""
    result = parser.parse_output(output, role=role)
    assert isinstance(result, dict)


@given(output=markdown_text)
@settings(max_examples=200)
def test_artifact_parser_markdown_input(output):
    """ArtifactParser handles markdown-structured input."""
    result = parser.parse_output(output, role="executor")
    assert isinstance(result, dict)


# ── FeedbackFormatter ─────────────────────────────────────────────────

formatter = FeedbackFormatter()


@given(output=st.text(min_size=0, max_size=10000))
@settings(max_examples=500)
def test_feedback_parser_never_crashes(output):
    """FeedbackFormatter.parse_review must never raise on any input."""
    result = formatter.parse_review(output)
    assert hasattr(result, "critiques")
    assert hasattr(result, "overall_grade")
    assert isinstance(result.critiques, list)


@given(output=st.text(min_size=0, max_size=5000))
@settings(max_examples=200)
def test_feedback_format_roundtrip(output):
    """parse_review -> format_for_executor should never crash."""
    review = formatter.parse_review(output)
    formatted = formatter.format_for_executor(review)
    assert isinstance(formatted, str)
    assert "=== REVIEWER FEEDBACK" in formatted


@given(output=markdown_text)
@settings(max_examples=200)
def test_feedback_parser_markdown_input(output):
    """FeedbackFormatter handles markdown-structured input."""
    result = formatter.parse_review(output)
    assert isinstance(result.critiques, list)


# ── FollowUpExecutor ─────────────────────────────────────────────────

followup_executor = FollowUpExecutor()


@given(output=st.text(min_size=0, max_size=10000))
@settings(max_examples=500)
def test_followup_parser_never_crashes(output):
    """FollowUpExecutor.parse_proposals must never raise on any input."""
    result = followup_executor.parse_proposals(output)
    assert isinstance(result, list)
    for proposal in result:
        assert hasattr(proposal, "title")
        assert hasattr(proposal, "description")


@given(output=markdown_text)
@settings(max_examples=200)
def test_followup_parser_markdown_input(output):
    """FollowUpExecutor handles markdown-structured input."""
    result = followup_executor.parse_proposals(output)
    assert isinstance(result, list)


# ── ConvergenceDetector ──────────────────────────────────────────────

@given(
    outputs=st.lists(st.text(min_size=0, max_size=1000), min_size=0, max_size=20),
    grades=st.lists(st.text(min_size=0, max_size=5), min_size=0, max_size=20),
)
@settings(max_examples=200)
def test_convergence_never_crashes(outputs, grades):
    """ConvergenceDetector must handle any sequence of inputs."""
    cd = ConvergenceDetector(window_size=3, min_iterations=2)
    for i, output in enumerate(outputs):
        review = None
        if i < len(grades) and grades[i]:
            review = ReviewResult(overall_grade=grades[i])
        cd.record_iteration(
            iteration_num=i + 1,
            role="executor" if i % 2 == 0 else "reviewer",
            output=output,
            review=review,
        )
    converged, reason = cd.should_stop()
    assert isinstance(converged, bool)
    assert isinstance(reason, str)
    metrics = cd.get_metrics()
    assert isinstance(metrics, dict)


# ── MechParser (ClaimGraph) ──────────────────────────────────────────

mech_parser = MechParser()


@given(content=st.text(min_size=0, max_size=5000))
@settings(max_examples=300)
def test_mech_parser_never_crashes(content):
    """MechParser.parse_mech_md must never raise on any input."""
    result = mech_parser.parse_mech_md(content)
    assert isinstance(result, list)


@given(content=markdown_text)
@settings(max_examples=200)
def test_mech_parser_markdown_input(content):
    """MechParser handles markdown-structured input."""
    result = mech_parser.parse_mech_md(content)
    assert isinstance(result, list)


@given(content=st.text(min_size=0, max_size=5000))
@settings(max_examples=200)
def test_mech_parser_eval_never_crashes(content):
    """MechParser.parse_eval_md must never raise on any input."""
    result = mech_parser.parse_eval_md(content)
    assert isinstance(result, list)
