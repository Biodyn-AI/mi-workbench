"""Regression tests: critique parsing, grading and executor feedback
(code-review findings consensus-1, -2, -4, -8, -9 and engine-2)."""
from __future__ import annotations

import json

import pytest

from backend.models import ReviewCritique, RunStatus, SeverityLevel
from backend.orchestrator.consensus import ConsensusReviewer, parse_lens_output
from backend.orchestrator.feedback import (
    FeedbackFormatter,
    extract_json_critiques,
    normalize_severity,
    parse_yaml_critiques,
)
from backend.orchestrator.presets import get_preset
from backend.models import AdapterRunResult
from backend.tests.fixes_r2_helpers import ScriptedReviewAdapter, json_block, run_engine

TRUNCATED = 'Review.\n```json\n{"critiques": [{"severity": "critical", "category": "leak'
BARE_TRUNCATED = '{"critiques": ['
TEMPLATE = ('```json\n{"critiques": [{"severity": "critical|high|medium|low|info", '
            '"category": "<short>", "description": "<the specific problem>", '
            '"required_fix": "<concrete fix>"}], "overall_assessment": "<one sentence>"}\n```')
UNKNOWN_KEY = '```json\n{"critiques": [{"severity": "critical", "notes": "Leakage in split."}]}\n```'
EMPTY_OK = '```json\n{"critiques": []}\n```'


# ── consensus-1: unusable contract blocks are never a clean review ─────


@pytest.mark.parametrize("text,reason", [
    (TRUNCATED, "truncated_empty"),
    (BARE_TRUNCATED, "truncated_empty"),
    (TEMPLATE, "all_items_unreadable"),
    (UNKNOWN_KEY, "all_items_unreadable"),
])
def test_invalid_contract_is_unparsed_not_clean(text, reason):
    p = extract_json_critiques(text)
    assert p is not None and p.invalid_reason == reason and p.method == "none"
    lens = parse_lens_output(text)
    assert lens.method == "none" and lens.critiques == []
    assert lens.detail == f"invalid_contract:{reason}"
    review = FeedbackFormatter().parse_review(text)
    assert review.parse_method == "none"
    assert review.overall_grade == ""  # never a derived "A"
    assert review.parse_detail == f"invalid_contract:{reason}"


def test_complete_empty_list_is_the_only_clean_empty_answer():
    p = parse_lens_output(EMPTY_OK)
    assert p.method == "json" and p.critiques == [] and not p.invalid_reason
    assert FeedbackFormatter().parse_review(EMPTY_OK).overall_grade == "A"
    # "no issues" entries are a clean answer too
    p2 = parse_lens_output('{"critiques": ["No issues found."]}')
    assert p2.method == "json" and p2.critiques == [] and p2.n_no_issue == 1


def test_invalid_block_falls_back_to_legacy_lines_in_prose():
    text = "[HIGH] Effect sizes are missing.\n" + TEMPLATE
    p = parse_lens_output(text)
    assert p.method == "regex" and [c.severity for c in p.critiques] == [SeverityLevel.HIGH]


def test_partial_truncation_counts_lens_but_is_reported():
    text = ('```json\n{"critiques": [{"severity": "high", "description": "Leak in split."},'
            ' {"severity": "crit')
    p = parse_lens_output(text)
    assert p.method == "json" and len(p.critiques) == 1 and p.detail == "truncated"
    rev = ConsensusReviewer(panel=[("reviewer", "r")])
    merged = rev.merge_critiques([AdapterRunResult(success=True, output=text)], ["reviewer"])
    meta = merged.consensus_meta
    assert meta["lens_status"] == {"reviewer": "ok"}
    assert meta["truncated_lenses"] == ["reviewer"]


@pytest.mark.parametrize("text", [TRUNCATED, TEMPLATE, UNKNOWN_KEY])
def test_merger_marks_invalid_contract_lens_failed(text):
    rev = ConsensusReviewer()
    results = [AdapterRunResult(success=True, output=text) for _ in range(3)]
    merged = rev.merge_critiques(results)
    meta = merged.consensus_meta
    assert set(meta["lens_status"].values()) == {"unparsed"}
    assert meta["panel_failed"] is True
    assert merged.overall_grade == "INCOMPLETE"


@pytest.mark.parametrize("text", [TRUNCATED, TEMPLATE, UNKNOWN_KEY])
def test_engine_panel_never_stops_on_invalid_contract(text):
    adapter = ScriptedReviewAdapter(lambda role, req, n: text)
    final, _ = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=6), 6,
                          {"grade_at_least": "B", "convergence_enabled": False})
    assert final.stop_reason != "stop_condition:grade_at_least"
    # every lens is unparsed -> panel retried once, then the step fails
    assert final.stop_reason == "failed:consensus_panel_failed"
    assert final.status == RunStatus.FAILED


@pytest.mark.parametrize("text", [TRUNCATED, TEMPLATE, UNKNOWN_KEY])
def test_engine_single_reviewer_never_stops_on_invalid_contract(text):
    adapter = ScriptedReviewAdapter(lambda role, req, n: text)
    final, eng = run_engine(adapter, get_preset("executor_reviewer", max_iterations=6), 6,
                            {"grade_at_least": "B", "convergence_enabled": False})
    assert final.stop_reason == "max_iterations"
    reviews = [i for i in final.iterations if i.role == "reviewer"]
    assert reviews and all(i.grade is None for i in reviews)
    assert eng._convergence.get_metrics()["grade_history"] == []
    # the executor still receives the reviewer's full text
    last_exec = adapter.executor_requests()[-1].prompt_bundle.user_prompt
    assert "could not be parsed" in last_exec


# ── consensus-2: single-reviewer grade only from the contract ─────────


TWO_CRITICAL = json_block([
    {"severity": "critical", "category": "s", "description": "Leakage between folds."},
    {"severity": "critical", "category": "s", "description": "Memorised reference network."},
])


@pytest.mark.parametrize("prose", [
    "attention score: a mean over all heads",
    "highest score: B cells",
    "score: e.g. AUROC",
    "Rating: a bit optimistic",
    "confidence score = a weighted sum",
])
def test_prose_grade_like_text_is_not_a_grade(prose):
    review = FeedbackFormatter().parse_review(prose + "\n\n" + TWO_CRITICAL)
    assert review.overall_grade == "F"


def test_grade_phrase_inside_description_is_ignored():
    text = json_block([{"severity": "critical", "description": "The reported attention score: "
                        "a single AUROC without a null."}])
    assert FeedbackFormatter().parse_review(text).overall_grade == "D"


def test_payload_grade_can_only_make_it_worse():
    worse = '```json\n{"critiques": [{"severity": "low", "description": "Typo."}], "overall_grade": "D"}\n```'
    better = ('```json\n{"critiques": [{"severity": "critical", "description": "Leak."}], '
              '"overall_grade": "A"}\n```')
    assert FeedbackFormatter().parse_review(worse).overall_grade == "D"
    assert FeedbackFormatter().parse_review(better).overall_grade == "D"


def test_legacy_explicit_grade_still_read():
    f = FeedbackFormatter()
    assert f.parse_review("Overall Grade: B+\n[HIGH] Missing control").overall_grade == "B+"
    assert f.parse_review("Grade: pass").overall_grade == "PASS"
    assert f.parse_review("Grade: e").overall_grade == ""  # E is not a grade


def test_engine_single_reviewer_with_critical_never_stops_on_grade():
    prose = "The attention score: a mean over all heads.\n\n"
    adapter = ScriptedReviewAdapter(lambda role, req, n: prose + TWO_CRITICAL)
    for thr in ("A", "B"):
        final, _ = run_engine(adapter, get_preset("executor_reviewer", max_iterations=6), 6,
                              {"grade_at_least": thr, "convergence_enabled": False})
        assert final.stop_reason == "max_iterations"
        assert all(i.grade == "F" for i in final.iterations if i.role == "reviewer")


# ── consensus-4: multiple contract blocks are visible ─────────────────


def test_multi_block_reported_and_dropped_earlier_counted():
    text = (json_block([{"severity": "critical", "description": "Leakage."}])
            + "\n" + json_block([]))
    p = extract_json_critiques(text)
    assert p.critiques == [] and p.method == "json"  # last block wins (contract)
    assert p.n_contract_blocks == 2 and p.dropped_earlier == 1
    draft_final = (json_block([{"severity": "high", "description": "No CI."}]) + "\n"
                   + json_block([{"severity": "high", "description": "No CI."}]))
    p2 = extract_json_critiques(draft_final)
    assert p2.n_contract_blocks == 2 and p2.dropped_earlier == 0 and len(p2.critiques) == 1
    rev = ConsensusReviewer(panel=[("reviewer", "r")])
    merged = rev.merge_critiques([AdapterRunResult(success=True, output=text)], ["reviewer"])
    assert merged.consensus_meta["multi_block_lenses"] == {"reviewer": 2}
    assert merged.consensus_meta["dropped_earlier"] == {"reviewer": 1}


# ── consensus-8: alias priority ───────────────────────────────────────


def test_contract_keys_win_over_aliases():
    text = json.dumps({"critiques": [
        {"title": "Missing CIs", "description": "Table 2 AUROC has no CI.",
         "type": "bug", "category": "statistics", "level": "low", "severity": "critical"},
        {"summary": "Missing CIs", "description": "Table 3 AUROC has no CI either."},
    ]})
    cs = extract_json_critiques(text).critiques
    assert [c.description for c in cs] == ["Table 2 AUROC has no CI.",
                                           "Table 3 AUROC has no CI either."]
    assert cs[0].category == "statistics" and cs[0].severity == SeverityLevel.CRITICAL


def test_yaml_title_then_description_is_one_item():
    cs = parse_yaml_critiques("- title: Missing CIs\n  description: The AUROC has no CI.\n"
                              "  required_fix: Bootstrap.\n  severity: high\n")
    assert len(cs) == 1
    assert cs[0].description == "The AUROC has no CI." and cs[0].required_fix == "Bootstrap."


def test_details_alias_is_a_description():
    cs = extract_json_critiques('{"critiques": [{"severity": "high", "details": "Leak."}]}').critiques
    assert [c.description for c in cs] == ["Leak."]


# ── consensus-9: echoed severity placeholder ──────────────────────────


def test_echoed_severity_placeholder_defaults_to_medium():
    assert normalize_severity("critical|high|medium|low|info") is None
    assert normalize_severity("critical/high/medium/low/info") is None
    assert normalize_severity("high|medium") is None
    assert normalize_severity("low (cosmetic)") == SeverityLevel.LOW
    text = json.dumps({"critiques": [{"severity": "critical|high|medium|low|info",
                                      "description": "No CIs for AUROC."}]})
    p = parse_lens_output(text)
    assert [(c.severity, c.description) for c in p.critiques] == [
        (SeverityLevel.MEDIUM, "No CIs for AUROC.")]
    assert FeedbackFormatter().parse_review(text).overall_grade == "B"


# ── engine-2: required_fix reaches the executor ───────────────────────


def test_format_for_executor_includes_fix_and_experiment():
    review = FeedbackFormatter().parse_review(json_block([{
        "severity": "high", "category": "statistics",
        "description": "AUROC 0.74 has no permutation null.",
        "required_fix": "Add a degree-preserving permutation null (1000 draws).",
        "suggested_experiment": "Shuffle edges   keeping degrees.",
    }]))
    text = FeedbackFormatter().format_for_executor(review)
    assert "1. [HIGH | statistics] AUROC 0.74 has no permutation null." in text
    assert "   Fix: Add a degree-preserving permutation null (1000 draws)." in text
    assert "   Suggested experiment: Shuffle edges keeping degrees." in text


def test_merged_members_text_reaches_the_executor():
    c = ReviewCritique(severity=SeverityLevel.HIGH, category="stats", description="[a, b] Long rep.",
                       required_fix="Fix A.",
                       merged_members=[{"lens": "b", "severity": "medium",
                                        "description": "Short other point.",
                                        "required_fix": "Fix B."}])
    from backend.models import ReviewResult
    text = FeedbackFormatter().format_for_executor(ReviewResult(critiques=[c]))
    assert "Also raised (b, MEDIUM): Short other point." in text and "Fix: Fix B." in text


FIX_REVIEW = json_block([{"severity": "high", "category": "statistics",
                          "description": "AUROC 0.74 has no permutation null.",
                          "required_fix": "Add a degree-preserving permutation null."}])


def test_engine_single_reviewer_fix_in_next_executor_prompt():
    adapter = ScriptedReviewAdapter(lambda role, req, n: FIX_REVIEW)
    final, _ = run_engine(adapter, get_preset("executor_reviewer", max_iterations=3), 3,
                          {"convergence_enabled": False})
    second = adapter.executor_requests()[1].prompt_bundle.user_prompt
    assert "Fix: Add a degree-preserving permutation null." in second


def test_engine_consensus_fix_in_next_executor_prompt_and_eval(tmp_path):
    written: dict[tuple[int, str], str] = {}

    def writer(n, name, content):
        written[(n, name)] = content

    adapter = ScriptedReviewAdapter(lambda role, req, n: FIX_REVIEW)
    final, _ = run_engine(adapter, get_preset("reviewer_consensus", max_iterations=3), 3,
                          {"convergence_enabled": False}, artifact_writer=writer)
    second = adapter.executor_requests()[1].prompt_bundle.user_prompt
    assert "Fix: Add a degree-preserving permutation null." in second
    assert "Fix: Add a degree-preserving permutation null." in written[(2, "EVAL.md")]
