"""Calibrated merge defaults (experiments/benchmark/results/merge_eval.json).

* The default consensus similarity method is ``llm`` (LLM adjudication with the
  panel adapter); jaccard / tfidf default thresholds are the calibrated 0.1.
* A failed adjudication (call error, unparseable answer, invalid partition
  after its retry) or a missing adjudicator falls back to ``tfidf`` at 0.1,
  recorded in ``merge_info["fallback"]`` and ``similarity_fallback``;
  ``consensus_llm_fallback: none`` restores the legacy "no merging".
* The mock adapter answers adjudication requests with a valid deterministic
  partition (exact duplicates), without changing the mock's grade
  progression, RNG stream or call counters, so mock runs stay deterministic
  and merge exactly what the legacy lexical merge merged.
"""
from __future__ import annotations

import asyncio
import json
import random

import pytest

from backend.adapters.mock import MockAdapter, mock_adjudication_groups
from backend.models import (
    AdapterRunRequest,
    AdapterRunResult,
    PromptBundle,
    ProviderName,
    RunState,
    RunStatus,
    WorkspaceContext,
)
from backend.orchestrator.consensus import (
    ConsensusReviewer,
    SimilarityBackendError,
    merge_parsed_critiques,
)
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.presets import get_preset
from backend.orchestrator.similarity import (
    CALIBRATION_SOURCE,
    DEFAULT_LLM_FALLBACK,
    DEFAULT_SIMILARITY_METHOD,
    DEFAULT_THRESHOLDS,
    build_adjudication_prompt,
)

PANEL = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]
LENSES = tuple(r for r, _ in PANEL)
CTX = WorkspaceContext(workspace_path="")

# Cross-lens paraphrases (tfidf 0.1 merges them, jaccard 0.5 does not) plus
# one unrelated critique.
ES_RIG = "Effect sizes are not reported alongside p-values for the main comparison"
ES_ADV = "The main comparison reports p-values without any effect size"
BIO = "Tissue specificity of the TRRUST edges is not established for K562 cells"


def _json(*descs, severity="medium"):
    return "```json\n" + json.dumps({"critiques": [
        {"severity": severity, "category": "g", "description": d} for d in descs]}) + "\n```"


LENS_OUTPUT = {"reviewer": _json(ES_RIG), "adversarial_reviewer": _json(ES_ADV),
               "bio_plausibility_checker": _json(BIO)}


class PanelAdapter:
    """Lens roles answer from LENS_OUTPUT; the adjudicator from ``adjudicator``
    (a list consumed per call; strings are outputs, AdapterRunResults returned)."""

    name = "scripted"

    def __init__(self, adjudicator):
        self.adjudicator = list(adjudicator)
        self.roles: list[str] = []

    async def run(self, request):
        role = request.prompt_bundle.variables.get("role", "")
        self.roles.append(role)
        if role == "consensus_adjudicator":
            item = self.adjudicator.pop(0) if len(self.adjudicator) > 1 else self.adjudicator[0]
            if isinstance(item, AdapterRunResult):
                return item
            return AdapterRunResult(success=True, output=item, token_usage=30,
                                    input_tokens=20, output_tokens=10)
        return AdapterRunResult(success=True, output=LENS_OUTPUT[role], token_usage=10)


def _panel(adjudicator, **kw):
    adapter = PanelAdapter(adjudicator)
    cr = ConsensusReviewer(panel=PANEL, retry_base_delay=0.0, **kw)
    merged, _ = asyncio.run(cr.run_panel("ARTIFACT", adapter, CTX))
    return cr, merged, adapter


def _tfidf_groups():
    pairs = [("reviewer", ES_RIG), ("adversarial_reviewer", ES_ADV),
             ("bio_plausibility_checker", BIO)]
    out = merge_parsed_critiques(
        [(lens, {"severity": "medium", "description": d}) for lens, d in pairs],
        method="tfidf", threshold=0.1)
    return out


# ── defaults ────────────────────────────────────────────────────────


def test_shipped_defaults():
    assert DEFAULT_SIMILARITY_METHOD == "llm"
    assert DEFAULT_LLM_FALLBACK == "tfidf"
    assert DEFAULT_THRESHOLDS["jaccard"] == 0.1 and DEFAULT_THRESHOLDS["tfidf"] == 0.1
    assert CALIBRATION_SOURCE.endswith("results/merge_eval.json")
    cr = ConsensusReviewer.from_config({})
    assert (cr.similarity_method, cr.similarity_threshold, cr.llm_fallback) == ("llm", None, "tfidf")


def test_calibration_file_supports_the_defaults():
    """The shipped defaults are the ones merge_eval.json recommends / selected."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    ev = json.loads((root / CALIBRATION_SOURCE).read_text())
    assert ev["recommended_default"]["method"] == DEFAULT_SIMILARITY_METHOD
    for method in ("jaccard", "tfidf"):
        folds = {f["threshold"] for f in ev["methods"][method]["folds"]}
        assert DEFAULT_THRESHOLDS[method] == max(folds)      # upper end of the selected range
        assert min(folds) >= 0.075
    f1 = {m: ev["methods"][m]["held_out"]["f1"] for m in ("llm", "tfidf", "jaccard")}
    assert f1["llm"] > f1["tfidf"] > f1["jaccard"]           # fallback = next best


def test_successful_adjudication_is_used():
    cr, merged, adapter = _panel(['```json\n{"groups": [[0, 1], [2]]}\n```'])
    assert adapter.roles.count("consensus_adjudicator") == 1
    report = cr.compute_consensus_report(merged)
    assert report["similarity_method"] == "llm" and report["similarity_fallback"] is None
    assert report["effective_similarity_method"] == "llm"
    assert report["total_merged"] == 2 and report["multi_reviewer"] == 1


# ── fallback to tfidf (calibrated threshold) ────────────────────────


@pytest.mark.parametrize("answers,reason", [
    (["```json\n{\"groups\": [[0, 1], [1, 2], [5]]}\n```"] * 2, "invalid partition"),
    (["I cannot do that."], "no parseable"),
    ([AdapterRunResult(success=False, error="[timeout] adjudicator timed out",
                       input_tokens=5, output_tokens=0, token_usage=5)], "timed out"),
])
def test_failed_adjudication_falls_back_to_tfidf(answers, reason):
    cr, merged, adapter = _panel(answers)
    meta = merged.consensus_meta
    fb = meta["merge_info"]["fallback"]
    assert fb["method"] == "tfidf" and fb["threshold"] == 0.1
    assert fb["reason"].startswith("adjudication failed") and reason in fb["reason"]
    adj = meta["merge_info"]["adjudicator"]
    assert adj["called"] is True and adj["success"] is False
    report = cr.compute_consensus_report(merged)
    assert report["similarity_method"] == "llm"
    assert report["similarity_fallback"] == fb
    assert report["effective_similarity_method"] == "tfidf"
    # Same partition as tfidf at 0.1 (the paraphrases merge and escalate) ...
    ref = _tfidf_groups()
    assert len(merged.critiques) == len(ref.result.critiques) == 2
    assert [c.severity for c in merged.critiques] == [c.severity for c in ref.result.critiques]
    assert report["multi_reviewer"] == 1
    # ... not "no merging".
    assert len(merged.critiques) < 3
    expected_calls = 2 if reason == "invalid partition" else 1   # one retry on a bad partition
    assert adapter.roles.count("consensus_adjudicator") == expected_calls


def test_fallback_none_restores_no_merging():
    cr, merged, _ = _panel(["nonsense"], llm_fallback="none")
    assert "fallback" not in merged.consensus_meta["merge_info"]
    assert merged.consensus_meta["merge_info"]["adjudicator"]["success"] is False
    assert len(merged.critiques) == 3          # singletons: no merging, no escalation
    assert cr.compute_consensus_report(merged)["similarity_fallback"] is None


def test_run_config_key_selects_the_fallback():
    cr = ConsensusReviewer.from_config({"consensus_llm_fallback": "jaccard"}, panel=PANEL)
    adapter = PanelAdapter(["nonsense"])
    merged, _ = asyncio.run(cr.run_panel("A", adapter, CTX))
    fb = merged.consensus_meta["merge_info"]["fallback"]
    assert fb["method"] == "jaccard" and fb["threshold"] == 0.1
    with pytest.raises(ValueError):
        ConsensusReviewer.from_config({"consensus_llm_fallback": "llm"})


def test_sync_merge_without_adjudicator_uses_the_fallback():
    results = [AdapterRunResult(success=True, output=LENS_OUTPUT[r]) for r in LENSES]
    merged = ConsensusReviewer(panel=PANEL).merge_critiques(results, list(LENSES))
    fb = merged.consensus_meta["merge_info"]["fallback"]
    assert fb == {"method": "tfidf", "threshold": 0.1, "reason": "no adjudicator available"}
    assert len(merged.critiques) == 2


def test_offline_merge_function_keeps_its_contract():
    pairs = [("reviewer", {"severity": "medium", "description": ES_RIG}),
             ("adversarial_reviewer", {"severity": "medium", "description": ES_ADV})]
    with pytest.raises(SimilarityBackendError):
        merge_parsed_critiques(pairs, method="llm")            # no adjudicator, no fallback
    out = merge_parsed_critiques(pairs, method="llm", fallback_method="tfidf")
    assert out.info["fallback"]["method"] == "tfidf" and out.groups == [[0, 1]]
    out2 = merge_parsed_critiques(pairs, method="llm", adjudicator=lambda p: "garbage")
    assert sorted(out2.groups) == [[0], [1]] and "fallback" not in out2.info   # legacy: no merging


# ── mock adapter: deterministic adjudication ────────────────────────


def _adj_request(texts):
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(system_prompt="sys", user_prompt=build_adjudication_prompt(texts),
                                   variables={"role": "consensus_adjudicator"}),
        workspace_context=CTX)


def test_mock_adjudicator_groups_exact_duplicates_deterministically():
    texts = ["Effect sizes missing", "No CI", "effect   sizes MISSING", "Other", "no ci"]
    assert mock_adjudication_groups(build_adjudication_prompt(texts)) == [[0, 2], [1, 4], [3]]
    MockAdapter.reset_iteration_count()
    adapter = MockAdapter(min_delay=0, max_delay=0)
    state = random.getstate()
    res1 = asyncio.run(adapter.run(_adj_request(texts)))
    res2 = asyncio.run(adapter.run(_adj_request(texts)))
    assert random.getstate() == state               # no RNG draws
    assert MockAdapter._iteration_count == 0        # grade progression untouched
    assert adapter.invocation_count == 0 and adapter.history == []
    assert adapter.adjudication_count == 2
    assert res1.output == res2.output and res1.success
    assert json.loads(res1.output.strip("`").removeprefix("json")) == {
        "groups": [[0, 2], [1, 4], [3]]}
    assert res1.token_usage == res1.input_tokens + res1.output_tokens


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_mock_panel_default_merge_equals_legacy_merge(tier):
    async def panel(**kw):
        MockAdapter.fixed_iteration = tier
        try:
            cr = ConsensusReviewer(panel=PANEL, **kw)
            merged, _ = await cr.run_panel("EXEC", MockAdapter(min_delay=0, max_delay=0), CTX)
            return merged
        finally:
            MockAdapter.fixed_iteration = None

    default = asyncio.run(panel())
    legacy = asyncio.run(panel(similarity_method="jaccard", similarity_threshold=0.5))
    assert default.consensus_meta["similarity_method"] == "llm"
    assert default.consensus_meta["similarity_fallback"] is None
    assert [(c.description, c.severity) for c in default.critiques] == \
        [(c.description, c.severity) for c in legacy.critiques]
    assert default.overall_grade == legacy.overall_grade


def _engine_run(config=None, max_iterations=6):
    async def go():
        MockAdapter.reset_iteration_count()
        events = []
        eng = LoopEngine(adapter=MockAdapter(min_delay=0, max_delay=0),
                         on_event=lambda t, d: events.append((t, d)))
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus", task="t",
                       provider=ProviderName.MOCK, model="", max_iterations=max_iterations,
                       config={"convergence_enabled": False, **(config or {})})
        final = await eng.run_loop(run, get_preset("reviewer_consensus",
                                                   max_iterations=max_iterations))
        return final, events

    return asyncio.run(go())


def _trajectory(final):
    return [(i.iteration_number, i.role, i.grade, i.critical_count, i.high_count)
            for i in sorted(final.iterations, key=lambda i: i.iteration_number)]


def test_mock_engine_runs_with_the_default_are_deterministic_and_match_legacy():
    a, events = _engine_run()
    b, _ = _engine_run()
    legacy, _ = _engine_run({"consensus_similarity_method": "jaccard",
                             "consensus_similarity_threshold": 0.5})
    assert a.status == RunStatus.COMPLETED and a.stop_reason == "max_iterations"
    assert _trajectory(a) == _trajectory(b) == _trajectory(legacy)
    reports = [d["report"] for t, d in events if t == "consensus_merged"]
    assert reports and all(r["similarity_method"] == "llm" for r in reports)
    assert all(r["similarity_fallback"] is None for r in reports)
    # one adjudicator call per panel, accounted in the iteration's usage
    for it in a.iterations:
        if it.role == "consensus_merger":
            assert it.agent_tools["calls"] == 4      # 3 lenses + 1 adjudicator
            assert it.token_usage == it.input_tokens + it.output_tokens


class FailingAdjudicatorMock(MockAdapter):
    async def run(self, request):
        if request.prompt_bundle.variables.get("role") == "consensus_adjudicator":
            return AdapterRunResult(success=False, error="[network] connection reset",
                                    input_tokens=7, output_tokens=0, token_usage=7)
        return await super().run(request)


def test_engine_records_the_fallback_and_keeps_running(tmp_path):
    async def go():
        MockAdapter.reset_iteration_count()
        written = {}
        eng = LoopEngine(adapter=FailingAdjudicatorMock(min_delay=0, max_delay=0),
                         artifact_writer=lambda n, name, c: written.__setitem__((n, name), c),
                         max_retries=0)
        run = RunState(workspace_id="w", loop_preset="reviewer_consensus", task="t",
                       provider=ProviderName.MOCK, model="", max_iterations=2,
                       config={"convergence_enabled": False})
        final = await eng.run_loop(run, get_preset("reviewer_consensus", max_iterations=2))
        return final, written

    final, written = asyncio.run(go())
    assert final.stop_reason == "max_iterations"
    panel = [i for i in final.iterations if i.role == "consensus_merger"][0]
    assert panel.consensus_report["similarity_fallback"]["method"] == "tfidf"
    assert "connection reset" in panel.consensus_report["similarity_fallback"]["reason"]
    payload = json.loads(written[(panel.iteration_number, "CONSENSUS.json")])
    assert payload["consensus_meta"]["merge_info"]["fallback"]["threshold"] == 0.1
    assert payload["report"]["effective_similarity_method"] == "tfidf"
