"""Regression tests: critique merging (code-review findings consensus-5, -6,
-7, -10, -11)."""
from __future__ import annotations

import asyncio

import pytest

from backend.models import AdapterRunResult, ReviewCritique, SeverityLevel
from backend.orchestrator.consensus import (
    ConsensusReviewer,
    amerge_parsed_critiques,
    merge_parsed_critiques,
    split_same_lens_groups,
)
from backend.orchestrator.similarity import LLMAdjudicator, detect_one_based
from backend.tests.fixes_r2_helpers import json_block


def _c(text, sev=SeverityLevel.HIGH, fix=None):
    return ReviewCritique(severity=sev, category="general", description=text, required_fix=fix)


FOUR = [
    ("reviewer", _c("Train/test leakage between the folds of the probing split")),
    ("adversarial_reviewer", _c("The reference network was memorised from the training data")),
    ("bio_plausibility_checker", _c("Typo in the legend", SeverityLevel.LOW)),
    ("reviewer", _c("Random seeds are not reported", SeverityLevel.LOW)),
]


# ── consensus-5: strict LLM adjudication ──────────────────────────────


def _adj(*answers):
    seq = list(answers)
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return seq.pop(0) if seq else seq_last[0]
    seq_last = [answers[-1]]
    return fn, calls


def test_one_based_answer_is_shifted_not_misgrouped():
    fn, calls = _adj('{"groups": [[1, 2], [3], [4]]}')
    res = LLMAdjudicator(fn).group_sync([c.description for _, c in FOUR])
    assert res.success and res.groups == [[0, 1], [2], [3]]
    assert any("1-based" in i for i in res.issues) and res.repaired
    assert len(calls) == 1
    assert detect_one_based([[1, 2], [3], [4]], 4) and not detect_one_based([[0, 1], [2], [3]], 4)


def test_invalid_partition_is_retried_then_fails_to_singletons():
    fn, calls = _adj('{"groups": [[0, 1], [1, 7]]}', '{"groups": [[0, 0], [9]]}')
    res = LLMAdjudicator(fn).group_sync(["a", "b", "c", "d"])
    assert len(calls) == 2 and "not a valid partition" in calls[1]
    assert res.success is False and res.groups == [[0], [1], [2], [3]]
    assert res.error.startswith("invalid partition")


def test_invalid_then_valid_answer_is_accepted():
    fn, calls = _adj('{"groups": [[0, 5]]}', '{"groups": [[0, 1], [2], [3]]}')
    res = LLMAdjudicator(fn).group_sync(["a", "b", "c", "d"])
    assert res.success and res.groups == [[0, 1], [2], [3]] and len(calls) == 2


def test_missing_indices_only_is_a_recorded_repair():
    fn, _ = _adj('{"groups": [[0, 1]]}')
    res = LLMAdjudicator(fn).group_sync(["a", "b", "c"])
    assert res.success and res.repaired and res.groups == [[0, 1], [2]]


class _AdjAdapter:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def run(self, request):
        out = self.outputs.pop(0)
        return AdapterRunResult(success=True, output=out, token_usage=150, input_tokens=100,
                                output_tokens=50, provider="p", model="adj-model",
                                cli_version="7.7.7")


def test_adjudicator_usage_summed_over_retry():
    adapter = _AdjAdapter(['{"groups": [[0, 9]]}', '{"groups": [[0], [1]]}'])
    res = asyncio.run(LLMAdjudicator(adapter).group(["a", "b"]))
    assert res.success and res.n_calls == 2
    assert (res.input_tokens, res.output_tokens, res.token_usage) == (200, 100, 300)
    assert res.model == "adj-model" and res.cli_version == "7.7.7"


def test_same_lens_rule_applies_to_llm_and_supplied_groups():
    pairs = [("reviewer", _c("A one")), ("adversarial_reviewer", _c("B two")),
             ("bio_plausibility_checker", _c("C three")), ("reviewer", _c("D four"))]
    fn, _ = _adj('{"groups": [[0, 3], [1], [2]]}')
    out = merge_parsed_critiques(pairs, method="llm", adjudicator=fn, same_lens_merge=False)
    assert sorted(out.groups) == [[0], [1], [2], [3]]
    assert out.info["same_lens_splits"] == 1
    out2 = asyncio.run(amerge_parsed_critiques(pairs, method="llm", adjudicator=fn,
                                               same_lens_merge=False))
    assert sorted(out2.groups) == [[0], [1], [2], [3]]
    out3 = merge_parsed_critiques(pairs, method="llm", groups=[[0, 1, 3], [2]])
    assert sorted(out3.groups) == [[0, 1], [2], [3]]  # default False splits too
    out4 = merge_parsed_critiques(pairs, method="llm", groups=[[0, 3], [1], [2]],
                                  same_lens_merge=True)
    assert sorted(out4.groups) == [[0, 3], [1], [2]]
    assert split_same_lens_groups([[0, 1, 2]], ["a", "a", "b"]) == ([[0, 2], [1]], 1)


# ── consensus-6: same-lens merging is off by default; nothing is dropped ─


CI2 = "Table 2 reports AUROC without confidence intervals; add bootstrap CIs for layer-4 AUROC"
CI3 = "Table 3 reports the head ablation effect without confidence intervals; add bootstrap CIs"


def test_default_keeps_two_distinct_critiques_of_one_lens():
    pairs = [("reviewer", _c(CI2, fix="Bootstrap layer 4.")),
             ("reviewer", _c(CI3, fix="Bootstrap ablation."))]
    for method in ("jaccard", "tfidf"):
        out = merge_parsed_critiques(pairs, method=method)
        assert len(out.result.critiques) == 2, method
    assert ConsensusReviewer().same_lens_merge is False
    assert ConsensusReviewer.from_config({}).same_lens_merge is False
    # three same-lens HIGH critiques keep their grade (D)
    three = [("reviewer", _c(f"Item {i} lacks confidence intervals for metric {i}")) for i in range(3)]
    assert merge_parsed_critiques(three).result.overall_grade == "D"


def test_merged_group_keeps_every_member_text():
    pairs = [("reviewer", _c("Effect sizes are not reported for the main comparison", fix="Fix A")),
             ("adversarial_reviewer",
              _c("Effect sizes are not reported for the main comparison in Table 2 at all",
                 fix="Fix B"))]
    out = merge_parsed_critiques(pairs, method="jaccard")
    assert len(out.result.critiques) == 1
    c = out.result.critiques[0]
    assert len(c.merged_members) == 1
    assert c.merged_members[0]["description"].startswith("Effect sizes are not reported")
    assert c.merged_members[0]["required_fix"] == "Fix A"


# ── consensus-7: lens identity for repeated roles ─────────────────────


LEAK = json_block([{"severity": "high", "description": "Train/test leakage between folds."}])
PANEL3 = [("reviewer", "reviewer/mi_reviewer")] * 3


def test_repeated_roles_role_identity_default():
    rev = ConsensusReviewer(panel=PANEL3)
    merged = rev.merge_critiques([AdapterRunResult(success=True, output=LEAK)] * 3)
    assert len(merged.critiques) == 1
    assert merged.critiques[0].severity == SeverityLevel.HIGH  # one lens: no escalation
    meta = merged.consensus_meta
    assert meta["lens_identity"] == "role"
    assert meta["n_groups_multi_lens"] == 0 and meta["n_groups_multi_call"] == 1


def test_repeated_roles_call_identity_escalates():
    rev = ConsensusReviewer(panel=PANEL3, lens_identity="call")
    merged = rev.merge_critiques([AdapterRunResult(success=True, output=LEAK)] * 3)
    assert len(merged.critiques) == 1
    assert merged.critiques[0].severity == SeverityLevel.CRITICAL
    assert merged.critiques[0].raised_by == ["reviewer#1", "reviewer#2", "reviewer#3"]
    assert ConsensusReviewer.from_config({"consensus_lens_identity": "call"}).lens_identity == "call"
    with pytest.raises(ValueError):
        ConsensusReviewer(lens_identity="bogus")


def test_same_lens_rule_is_per_call_for_repeated_roles():
    # same_lens_merge=False must still merge copies of one role across calls
    rev = ConsensusReviewer(panel=PANEL3, same_lens_merge=False)
    merged = rev.merge_critiques([AdapterRunResult(success=True, output=LEAK)] * 3)
    assert len(merged.critiques) == 1


# ── consensus-10: single / multi reviewer counts ──────────────────────


def test_reviewer_counts_do_not_depend_on_threshold():
    shared = "Effect sizes are not reported for the main comparison"
    results = [AdapterRunResult(success=True, output=json_block([
                   {"severity": "high", "description": shared},
                   {"severity": "low", "description": "Random seeds are not documented anywhere"}])),
               AdapterRunResult(success=True, output=json_block([
                   {"severity": "high", "description": shared}]))]
    names = ["reviewer", "adversarial_reviewer"]
    for k, escalated in ((2, 1), (3, 0)):
        rev = ConsensusReviewer(consensus_threshold=k, panel=[(n, n) for n in names])
        rep = rev.compute_consensus_report(rev.merge_critiques(results, names))
        assert (rep["single_reviewer"], rep["multi_reviewer"]) == (1, 1), k
        assert rep["escalated_by_agreement"] == escalated, k
    with pytest.raises(ValueError):
        ConsensusReviewer(consensus_threshold=1)
    with pytest.raises(ValueError):
        merge_parsed_critiques([("a", _c("x"))], consensus_threshold=1)


# ── consensus-11: strict boolean config parsing ───────────────────────


@pytest.mark.parametrize("value,expected", [
    ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ("true", True), ("On", True), (False, False), (1, True),
])
def test_bool_config_keys_parsed_strictly(value, expected):
    r = ConsensusReviewer.from_config({"consensus_same_lens_merge": value,
                                       "consensus_unparsed_is_failure": value})
    assert r.same_lens_merge is expected and r.unparsed_is_failure is expected


def test_invalid_bool_string_is_an_error():
    with pytest.raises(ValueError):
        ConsensusReviewer.from_config({"consensus_same_lens_merge": "flase"})
