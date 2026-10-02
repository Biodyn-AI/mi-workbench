"""E1/E5: panel robustness (timeouts, retries, concurrency, failed lenses) and
the consensus report fields."""
from __future__ import annotations

import asyncio
import json

import pytest

from backend.adapters.mock import MockAdapter
from backend.models import AdapterRunResult, ProviderName, RunState, SeverityLevel, WorkspaceContext
from backend.orchestrator.consensus import INCOMPLETE_GRADE, ConsensusReviewer
from backend.orchestrator.feedback import FeedbackFormatter

PANEL = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]
CTX = WorkspaceContext(workspace_path="")


def _json(*critiques, assessment="ok"):
    return "```json\n" + json.dumps({"critiques": list(critiques),
                                      "overall_assessment": assessment}) + "\n```"


SHARED = {"severity": "high", "category": "statistics",
          "description": "Effect sizes are not reported alongside p-values for the main comparison",
          "required_fix": "Report Cohen's d with 95% CIs."}


class ScriptedAdapter:
    """Per-role scripted behaviour: a list consumed one item per call.

    Items: an AdapterRunResult, a string (successful output), an Exception
    instance (raised), or ("sleep", seconds, then_item).
    """

    def __init__(self, script: dict, delay: float = 0.0):
        self.script = {k: list(v) for k, v in script.items()}
        self.delay = delay
        self.calls: list = []
        self.in_flight = 0
        self.peak = 0

    async def run(self, request):
        role = request.prompt_bundle.variables.get("role", "")
        self.calls.append((role, request))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            items = self.script.get(role) or self.script.get("*") or [""]
            item = items.pop(0) if len(items) > 1 else items[0]
            if isinstance(item, tuple) and item[0] == "sleep":
                await asyncio.sleep(item[1])
                item = item[2]
            if isinstance(item, Exception):
                raise item
            if isinstance(item, AdapterRunResult):
                return item
            return AdapterRunResult(success=True, output=item, token_usage=10)
        finally:
            self.in_flight -= 1

    def roles_called(self):
        return [r for r, _ in self.calls]


def _run(cr, adapter):
    return asyncio.run(cr.run_panel("ARTIFACT", adapter, CTX))


def _fast(**kw):
    kw.setdefault("retry_base_delay", 0.0)
    kw.setdefault("panel", PANEL)
    return ConsensusReviewer(**kw)


class TestFailedLenses:
    def test_all_lenses_failed_is_incomplete_not_grade_a(self):
        adapter = ScriptedAdapter({"*": [AdapterRunResult(success=False, error="auth failed")]})
        cr = _fast()
        merged, raw = _run(cr, adapter)
        assert merged.overall_grade == INCOMPLETE_GRADE
        assert merged.critiques == []
        report = cr.compute_consensus_report(merged)
        assert report["panel_failed"] is True
        assert report["panel_partial"] is False
        assert report["grade"] == INCOMPLETE_GRADE
        assert report["failed_lenses"] == [r for r, _ in PANEL]
        assert set(report["lens_status"].values()) == {"failed"}
        assert report["lens_errors"]["reviewer"] == "auth failed"
        assert report["raw_critique_counts"] == {r: 0 for r, _ in PANEL}
        assert report["parse_methods"] == {r: "none" for r, _ in PANEL}
        assert len(raw) == 3 and not any(r.success for r in raw)
        # Permanent errors are not retried.
        assert len(adapter.calls) == 3
        text = FeedbackFormatter().format_for_executor(merged)
        assert "Grade: INCOMPLETE" in text and "PANEL STATUS: INCOMPLETE" in text

    def test_exception_in_one_lens_is_recorded_not_raised(self):
        adapter = ScriptedAdapter({
            "reviewer": [_json(SHARED)],
            "adversarial_reviewer": [RuntimeError("CLI crashed")],
            "bio_plausibility_checker": [_json({"severity": "medium",
                                                "description": "TRRUST is not K562-specific."})],
        })
        cr = _fast()
        merged, raw = _run(cr, adapter)
        meta = merged.consensus_meta
        assert meta["panel_partial"] is True and meta["panel_failed"] is False
        assert meta["failed_lenses"] == ["adversarial_reviewer"]
        assert meta["lens_status"]["adversarial_reviewer"] == "failed"
        assert "CLI crashed" in meta["lens_errors"]["adversarial_reviewer"]
        assert raw[1].success is False
        assert merged.overall_grade == "C"  # one HIGH from the surviving lenses
        text = FeedbackFormatter().format_for_executor(merged)
        assert "PANEL STATUS: PARTIAL" in text and "adversarial_reviewer" in text

    def test_empty_and_unparsed_outputs(self):
        adapter = ScriptedAdapter({
            "reviewer": [""],
            "adversarial_reviewer": ["Everything looks fine to me."],
            "bio_plausibility_checker": [_json(SHARED)],
        })
        merged, _ = _run(_fast(), adapter)
        meta = merged.consensus_meta
        assert meta["lens_status"] == {"reviewer": "empty", "adversarial_reviewer": "unparsed",
                                       "bio_plausibility_checker": "ok"}
        assert meta["failed_lenses"] == ["reviewer", "adversarial_reviewer"]
        assert meta["panel_partial"] is True
        # unparsed_is_failure=False: prose without critiques counts as a clean lens
        merged2, _ = _run(_fast(unparsed_is_failure=False),
                          ScriptedAdapter({"*": ["Everything looks fine to me."]}))
        assert merged2.consensus_meta["lens_status"]["reviewer"] == "ok"
        assert merged2.overall_grade == "A"

    def test_min_ok_lenses(self):
        adapter = ScriptedAdapter({
            "reviewer": [_json(SHARED)],
            "*": [AdapterRunResult(success=False, error="nope")],
        })
        merged, _ = _run(_fast(min_ok_lenses=2), adapter)
        assert merged.overall_grade == INCOMPLETE_GRADE
        assert merged.consensus_meta["panel_failed"] is True
        assert len(merged.critiques) == 1  # surviving critiques are still reported

    def test_escalation_ignores_failed_lens(self):
        adapter = ScriptedAdapter({
            "reviewer": [_json(SHARED)],
            "adversarial_reviewer": [AdapterRunResult(success=False, output=_json(SHARED),
                                                       error="exit 1")],
            "bio_plausibility_checker": [_json()],
        })
        merged, _ = _run(_fast(), adapter)
        assert [c.severity for c in merged.critiques] == [SeverityLevel.HIGH]


class TestTimeoutsAndRetries:
    def test_default_lens_timeout(self):
        assert ConsensusReviewer().lens_timeout == 900.0
        assert ConsensusReviewer(adapter_timeout=120).lens_timeout == 120.0
        assert ConsensusReviewer(adapter_timeout=120, lens_timeout=30).lens_timeout == 30.0
        with pytest.raises(ValueError):
            ConsensusReviewer(lens_timeout=0)

    def test_request_carries_timeout_and_extra_fields(self):
        adapter = ScriptedAdapter({"*": [_json()]})
        cr = _fast(lens_timeout=12.5, request_kwargs={"files_to_read": ["MECH.md"]})
        _run(cr, adapter)
        for _, req in adapter.calls:
            assert req.timeout_seconds == 13
            assert req.files_to_read == ["MECH.md"]

    def test_hanging_lens_times_out_and_is_recorded(self):
        adapter = ScriptedAdapter({
            "reviewer": [("sleep", 5.0, _json(SHARED))],
            "*": [_json()],
        })
        cr = _fast(lens_timeout=0.05, timeout_grace=0.05, max_retries=1)
        merged, raw = _run(cr, adapter)
        meta = merged.consensus_meta
        assert meta["lens_status"]["reviewer"] == "failed"
        assert "Timeout" in meta["lens_errors"]["reviewer"]
        assert meta["attempts"]["reviewer"] == 2  # timeout is transient -> retried once
        assert meta["panel_partial"] is True
        assert merged.overall_grade == "A"  # the two surviving lenses were clean

    def test_transient_error_is_retried(self):
        adapter = ScriptedAdapter({
            "reviewer": [AdapterRunResult(success=False, error="429 rate limit"), _json(SHARED)],
            "*": [_json()],
        })
        merged, raw = _run(_fast(max_retries=2), adapter)
        meta = merged.consensus_meta
        assert meta["lens_status"]["reviewer"] == "ok"
        assert meta["attempts"] == {"reviewer": 2, "adversarial_reviewer": 1,
                                    "bio_plausibility_checker": 1}
        assert raw[0].success is True
        assert adapter.roles_called().count("reviewer") == 2

    def test_retries_exhausted(self):
        adapter = ScriptedAdapter({"*": [AdapterRunResult(success=False, error="connection reset")]})
        merged, _ = _run(_fast(max_retries=2), adapter)
        assert merged.consensus_meta["attempts"]["reviewer"] == 3
        assert merged.overall_grade == INCOMPLETE_GRADE

    def test_concurrency_limit(self):
        unlimited = ScriptedAdapter({"*": [_json()]}, delay=0.05)
        _run(_fast(), unlimited)
        assert unlimited.peak == 3
        limited = ScriptedAdapter({"*": [_json()]}, delay=0.05)
        _run(_fast(max_concurrency=1), limited)
        assert limited.peak == 1


class TestReportFields:
    def test_report_has_merge_provenance(self):
        adapter = ScriptedAdapter({
            "reviewer": [_json(SHARED, {"severity": "low", "description": "Typo in Table 2."})],
            "adversarial_reviewer": [_json(dict(SHARED, severity="medium"))],
            "bio_plausibility_checker": [_json({"severity": "medium",
                                                "description": "TRRUST is not K562-specific."})],
        })
        cr = _fast(similarity_method="jaccard", similarity_threshold=0.5)  # legacy merge
        merged, _ = _run(cr, adapter)
        report = cr.compute_consensus_report(merged)
        assert report["similarity_method"] == "jaccard"
        assert report["similarity_threshold"] == 0.5
        assert report["similarity_fallback"] is None
        assert report["effective_similarity_method"] == "jaccard"
        assert report["n_groups_multi_lens"] == 1
        assert report["multi_reviewer"] == 1
        assert report["total_merged"] == 3
        assert report["raw_critique_counts"] == {"reviewer": 2, "adversarial_reviewer": 1,
                                                 "bio_plausibility_checker": 1}
        assert report["parse_methods"] == {r: "json" for r, _ in PANEL}
        # merge_groups[k] <-> merged.critiques[k]; members are [lens_key, index]
        assert report["merge_groups"][0] == [["reviewer", 0], ["adversarial_reviewer", 0]]
        assert sorted(map(tuple, (m for g in report["merge_groups"] for m in g))) == sorted([
            ("reviewer", 0), ("reviewer", 1), ("adversarial_reviewer", 0),
            ("bio_plausibility_checker", 0)])
        assert merged.critiques[0].severity == SeverityLevel.CRITICAL  # HIGH escalated
        json.dumps(report)  # event payload must be JSON-serialisable
        # An engine-style second reviewer (no similarity config) reads the
        # provenance from the merged result, not from its own defaults.
        second = ConsensusReviewer(panel=PANEL, consensus_threshold=2)
        assert second.compute_consensus_report(merged)["merge_groups"] == report["merge_groups"]

    def test_report_on_plain_review_result_has_defaults(self):
        from backend.models import ReviewCritique, ReviewResult
        rr = ReviewResult(overall_grade="C", critiques=[ReviewCritique(
            severity=SeverityLevel.HIGH, category="g", description="[reviewer, adversarial_reviewer] x")])
        report = ConsensusReviewer(similarity_method="tfidf").compute_consensus_report(rr)
        assert report["panel_failed"] is False and report["failed_lenses"] == []
        assert report["similarity_method"] == "tfidf"
        assert report["n_groups_multi_lens"] == 1 and report["multi_reviewer"] == 1

    def test_tfidf_method_merges_paraphrases_in_panel(self):
        adapter = ScriptedAdapter({
            "reviewer": [_json(SHARED)],
            "adversarial_reviewer": [_json({"severity": "medium", "description":
                                            "The main comparison reports p-values without any effect size."})],
            "bio_plausibility_checker": [_json()],
        })
        jac, _ = _run(_fast(similarity_method="jaccard", similarity_threshold=0.5), adapter)
        adapter2 = ScriptedAdapter(adapter.script)
        tf, _ = _run(_fast(similarity_method="tfidf", similarity_threshold=0.3), adapter2)
        assert len(jac.critiques) == 2 and len(tf.critiques) == 1
        assert tf.consensus_meta["similarity_method"] == "tfidf"
        assert tf.consensus_meta["similarity_threshold"] == 0.3
        adapter3 = ScriptedAdapter(adapter.script)
        tf_default, _ = _run(_fast(similarity_method="tfidf"), adapter3)
        assert len(tf_default.critiques) == 1
        assert tf_default.consensus_meta["similarity_threshold"] == 0.1  # calibrated

    def test_llm_method_uses_panel_adapter_once(self):
        adapter = ScriptedAdapter({
            "reviewer": [_json(SHARED)],
            "adversarial_reviewer": [_json({"severity": "medium", "description": "No effect size."})],
            "bio_plausibility_checker": [_json({"severity": "low", "description": "Pathway unclear."})],
            "consensus_adjudicator": ['```json\n{"groups": [[0, 1], [2]]}\n```'],
        })
        cr = _fast(similarity_method="llm")
        merged, raw = _run(cr, adapter)
        assert adapter.roles_called().count("consensus_adjudicator") == 1
        assert len(raw) == 3  # aligned with the panel
        assert len(merged.critiques) == 2
        info = merged.consensus_meta["merge_info"]["adjudicator"]
        assert info["called"] and info["success"]
        assert cr.compute_consensus_report(merged)["similarity_threshold"] is None

    def test_duplicate_role_names_get_unique_keys(self):
        panel = [("reviewer", "reviewer/mi_reviewer")] * 2
        adapter = ScriptedAdapter({"*": [_json(SHARED)]})
        merged, _ = _run(_fast(panel=panel), adapter)
        meta = merged.consensus_meta
        assert meta["lenses"] == ["reviewer#1", "reviewer#2"]
        # Same lens name twice is ONE distinct lens: no escalation.
        assert [c.severity for c in merged.critiques] == [SeverityLevel.HIGH]
        assert meta["merge_groups"] == [[["reviewer#1", 0], ["reviewer#2", 0]]]


class TestFromConfig:
    def test_from_config_keys(self):
        cr = ConsensusReviewer.from_config({
            "consensus_similarity_method": "tfidf",
            "consensus_threshold": 3,
            "adapter_timeout": 600,
            "max_retries": 1,
            "consensus_max_concurrency": 2,
            "consensus_role_weights": {"reviewer": 2.0},
        }, panel=PANEL)
        assert cr.similarity_method == "tfidf" and cr.similarity_threshold == 0.1
        assert cr.consensus_threshold == 3
        assert cr.lens_timeout == 600.0
        assert cr.max_retries == 1
        assert cr.max_concurrency == 2
        assert cr.role_weights == {"reviewer": 2.0}
        cr2 = ConsensusReviewer.from_config({"consensus_lens_timeout": 45, "adapter_timeout": 600,
                                              "consensus_similarity_threshold": 0.6})
        assert cr2.lens_timeout == 45.0
        # The default method is llm (the threshold is kept but unused for llm).
        assert cr2.similarity_method == "llm" and cr2.similarity_threshold == 0.6
        default = ConsensusReviewer.from_config(None)
        assert default.similarity_method == "llm" and default.similarity_threshold is None
        assert default.llm_fallback == "tfidf"
        assert ConsensusReviewer.from_config({"consensus_llm_fallback": "none"}).llm_fallback is None
        cr3 = ConsensusReviewer.from_config({"consensus_similarity_method": "jaccard"})
        assert cr3.similarity_threshold == 0.1


class TestMockPanelUnchanged:
    def test_mock_panel_report_matches_legacy_numbers(self):
        async def go():
            MockAdapter.fixed_iteration = 1
            try:
                cr = ConsensusReviewer(panel=PANEL)
                merged, raw = await cr.run_panel("EXEC", MockAdapter(min_delay=0, max_delay=0), CTX)
                return cr.compute_consensus_report(merged)
            finally:
                MockAdapter.fixed_iteration = None

        report = asyncio.run(go())
        assert report["total_merged"] == 7
        assert report["multi_reviewer"] == 1
        assert report["unresolved_critical"] == 3
        assert report["grade"] == "F"
        assert report["severity_histogram"] == {"critical": 3, "high": 2, "medium": 2}
        assert report["panel_failed"] is False


def test_engine_consensus_event_reports_panel_failure():
    """Through the live engine: reviewers fail, executor succeeds -> the
    consensus_merged report flags the failure instead of grade A."""
    from backend.orchestrator.engine import LoopEngine
    from backend.orchestrator.presets import get_preset

    class ReviewersDown:
        async def run(self, request):
            role = request.prompt_bundle.variables.get("role", "")
            if role in {r for r, _ in PANEL}:
                return AdapterRunResult(success=False, error="auth failed", exit_code=1)
            return AdapterRunResult(success=True, output="## MECH.md\nClaim.\n", token_usage=5)

        async def smoke_test(self):
            return {"status": "ok"}

        def is_available(self):
            return True

    events = []

    async def go():
        eng = LoopEngine(adapter=ReviewersDown(), on_event=lambda t, d: events.append((t, d)))
        loop = get_preset("reviewer_consensus", max_iterations=2)
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus", task="t",
                       provider=ProviderName.MOCK, model="", max_iterations=2,
                       config={"convergence_enabled": False, "max_retries": 0})
        return await eng.run_loop(run, loop)

    asyncio.run(go())
    reports = [d["report"] for t, d in events if t == "consensus_merged"]
    assert reports, "no consensus step ran"
    assert reports[0]["panel_failed"] is True
    assert reports[0]["grade"] == INCOMPLETE_GRADE
