"""E5: pluggable critique similarity and order-independent grouping."""
from __future__ import annotations

import asyncio
import itertools
import random

import pytest

from backend.models import AdapterRunResult, ReviewCritique, SeverityLevel
from backend.orchestrator import similarity as sim
from backend.orchestrator.consensus import (
    ConsensusReviewer,
    amerge_parsed_critiques,
    merge_parsed_critiques,
)
from backend.orchestrator.similarity import (
    DEFAULT_THRESHOLDS,
    UNCALIBRATED_THRESHOLDS,
    LLMAdjudicator,
    SimilarityBackendError,
    TfidfSimilarity,
    average_linkage_groups,
    group_texts,
    jaccard_similarity,
    light_stem,
    normalize_tokens,
    parse_adjudication_output,
    validate_groups,
)

PARAPHRASES = [
    ("Effect sizes are not reported alongside p-values for the main comparison",
     "The main comparison reports p-values without any effect size."),
    ("Layer 15 choice is arbitrary and may reflect p-hacking.",
     "The choice of layer-15 is arbitrary and suggests p-hacking."),
]
UNRELATED = ("Random seeds are not documented for the bootstrap.",
             "TRRUST is not specific to K562 cells.")


def _legacy_jaccard(a: str, b: str) -> float:
    """Verbatim copy of the original ConsensusReviewer._similarity_score."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ── Jaccard (legacy) ──────────────────────────────────────────────────


class TestJaccard:
    @pytest.mark.parametrize("a,b", PARAPHRASES + [UNRELATED, ("", "x"), ("", ""), ("A b", "a B")])
    def test_bit_for_bit_legacy(self, a, b):
        assert jaccard_similarity(a, b) == _legacy_jaccard(a, b)
        assert ConsensusReviewer()._similarity_score(a, b) == _legacy_jaccard(a, b)

    def test_default_thresholds_are_calibrated(self):
        # Calibrated on the benchmark merge evaluation (fold-selected 0.075-0.1).
        assert DEFAULT_THRESHOLDS["jaccard"] == 0.1
        assert DEFAULT_THRESHOLDS["tfidf"] == 0.1
        # The uncalibrated pre-calibration values stay available.
        assert UNCALIBRATED_THRESHOLDS == {"jaccard": 0.5, "tfidf": 0.3, "embedding": 0.7,
                                           "llm": None}
        cr = ConsensusReviewer()
        assert cr.similarity_method == "llm" and cr.similarity_threshold is None
        assert cr.llm_fallback == "tfidf"
        assert ConsensusReviewer(similarity_method="jaccard").similarity_threshold == 0.1


# ── TF-IDF ────────────────────────────────────────────────────────────


class TestTfidf:
    def test_normalisation(self):
        assert light_stem("reports") == light_stem("reported") == light_stem("reporting") == "report"
        assert light_stem("sizes") == light_stem("size")
        assert light_stem("controlled") == "control"
        assert light_stem("call") == "call"
        assert light_stem("15") == "15"
        toks = normalize_tokens("The choice of layer-15 is arbitrary!")
        assert toks == ["choic", "layer", "15", "arbitrary"]

    @pytest.mark.parametrize("a,b", PARAPHRASES)
    def test_paraphrases_score_higher_than_jaccard(self, a, b):
        s = TfidfSimilarity().pair(a, b)
        assert s > jaccard_similarity(a, b)
        assert s >= DEFAULT_THRESHOLDS["tfidf"]

    def test_unrelated_low(self):
        assert TfidfSimilarity().pair(*UNRELATED) < 0.1

    def test_matrix_symmetric_unit_diagonal(self):
        texts = [p for pair in PARAPHRASES for p in pair] + list(UNRELATED)
        m = sim.similarity_matrix(texts, "tfidf")
        for i in range(len(texts)):
            assert m[i][i] == 1.0
            for j in range(len(texts)):
                assert m[i][j] == m[j][i]
                assert 0.0 <= m[i][j] <= 1.0

    def test_idf_fitted_on_current_panel(self):
        a, b = PARAPHRASES[0]
        alone = TfidfSimilarity().pair(a, b)
        with_context = sim.similarity_matrix([a, b, "effect effect effect sizes"], "tfidf")[0][1]
        assert alone != with_context

    def test_tfidf_groups_paraphrases_only(self):
        texts = [PARAPHRASES[0][0], UNRELATED[0], PARAPHRASES[0][1], UNRELATED[1]]
        groups = group_texts(texts, "tfidf", lenses=["r", "a", "b", "r"])
        assert [0, 2] in groups
        assert [1] in groups and [3] in groups


# ── Average-linkage clustering ────────────────────────────────────────


class TestAverageLinkage:
    def test_threshold_and_transitivity(self):
        # 0-1 and 1-2 similar, 0-2 dissimilar: average linkage decides on the mean.
        m = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.8], [0.1, 0.8, 1.0]]
        # After merging {0,1} (0.9), link to 2 is (0.1+0.8)/2 = 0.45.
        assert average_linkage_groups(m, 0.5) == [[0, 1], [2]]
        assert average_linkage_groups(m, 0.4) == [[0, 1, 2]]
        assert average_linkage_groups(m, 0.95) == [[0], [1], [2]]

    # Chain: J(A,B) = J(B,C) = 0.5 >= tau, J(A,C) = 0.2 < tau.
    CHAIN = [
        "effect sizes missing for layer comparison",
        "effect sizes missing for attention baseline",
        "missing for attention baseline null model",
        "random seeds for bootstrap resampling are not documented",
        "bootstrap resampling random seeds are undocumented",
    ]
    CHAIN_LENSES = ["reviewer", "adversarial_reviewer", "bio_plausibility_checker",
                    "reviewer", "adversarial_reviewer"]

    @staticmethod
    def _legacy_greedy(texts, tau=0.5):
        used = [False] * len(texts)
        groups = []
        for i in range(len(texts)):
            if used[i]:
                continue
            g = [i]
            used[i] = True
            for j in range(i + 1, len(texts)):
                if not used[j] and _legacy_jaccard(texts[i], texts[j]) >= tau:
                    g.append(j)
                    used[j] = True
            groups.append(g)
        return groups

    def test_legacy_greedy_was_order_dependent(self):
        seen = set()
        for perm in itertools.permutations(range(len(self.CHAIN))):
            t = [self.CHAIN[i] for i in perm]
            seen.add(repr(sorted(sorted(t[i] for i in g) for g in self._legacy_greedy(t))))
        assert len(seen) > 1  # the regression this work package removes

    @pytest.mark.parametrize("method", ["jaccard", "tfidf"])
    @pytest.mark.parametrize("thresholds", ["uncalibrated", "default"])
    def test_grouping_is_order_independent(self, method, thresholds):
        tau = (UNCALIBRATED_THRESHOLDS if thresholds == "uncalibrated"
               else DEFAULT_THRESHOLDS)[method]
        ref = None
        for perm in itertools.permutations(range(len(self.CHAIN))):
            t = [self.CHAIN[i] for i in perm]
            lz = [self.CHAIN_LENSES[i] for i in perm]
            groups = group_texts(t, method, threshold=tau, lenses=lz)
            canon = sorted(sorted(t[i] for i in g) for g in groups)
            if ref is None:
                ref = canon
            assert canon == ref, (method, perm)
        if method == "jaccard" and tau == 0.5:
            # Average linkage never chains A and C together (mean link 0.35 < 0.5).
            assert not any(self.CHAIN[0] in g and self.CHAIN[2] in g for g in ref)

    def test_order_independence_randomised_matrix(self):
        rng = random.Random(7)
        n = 9
        base = [[0.0] * n for _ in range(n)]
        for i in range(n):
            base[i][i] = 1.0
            for j in range(i + 1, n):
                v = rng.choice([0.2, 0.4, 0.5, 0.6, 0.8])  # many exact ties
                base[i][j] = base[j][i] = v
        keys = [f"item{i}" for i in range(n)]
        ref = sorted(sorted(keys[i] for i in g) for g in average_linkage_groups(base, 0.5, keys=keys))
        for _ in range(40):
            perm = list(range(n))
            rng.shuffle(perm)
            m = [[base[perm[a]][perm[b]] for b in range(n)] for a in range(n)]
            k = [keys[p] for p in perm]
            got = sorted(sorted(k[i] for i in g) for g in average_linkage_groups(m, 0.5, keys=k))
            assert got == ref

    def test_same_lens_cannot_link(self):
        m = [[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]]
        lenses = ["r", "r", "a"]
        assert average_linkage_groups(m, 0.5, lenses=lenses, same_lens_merge=True) == [[0, 1, 2]]
        # default: same-lens merging is not allowed
        groups = average_linkage_groups(m, 0.5, lenses=lenses)
        assert groups == average_linkage_groups(m, 0.5, lenses=lenses, same_lens_merge=False)
        assert sorted(len(g) for g in groups) == [1, 2]
        for g in groups:
            assert len({lenses[i] for i in g}) == len(g)

    def test_empty_and_single(self):
        assert average_linkage_groups([], 0.5) == []
        assert average_linkage_groups([[1.0]], 0.5) == [[0]]


# ── Embedding ─────────────────────────────────────────────────────────


class _FakeEncoder:
    """Bag-of-keywords 'embedding' so the dense path is testable offline."""

    VOCAB = ["effect", "seed", "trrust", "layer"]

    def encode(self, texts):
        return [[float(w in t.lower()) + 0.01 for w in self.VOCAB] for t in texts]


class TestEmbedding:
    def test_injected_encoder_groups(self):
        texts = ["effect size missing", "no effect size given", "seed undocumented"]
        groups = group_texts(texts, "embedding", embedding_encoder=_FakeEncoder())
        assert groups == [[0, 1], [2]]

    def test_requires_model_or_encoder(self):
        with pytest.raises(SimilarityBackendError, match="embedding_model"):
            sim.EmbeddingSimilarity()

    @pytest.mark.skipif(sim.embedding_backend_available(),
                        reason="sentence-transformers installed")
    def test_clear_error_when_backend_missing(self):
        with pytest.raises(SimilarityBackendError, match="sentence-transformers"):
            ConsensusReviewer(similarity_method="embedding",
                              embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        with pytest.raises(SimilarityBackendError, match="sentence-transformers"):
            sim.similarity_matrix(["a", "b"], "embedding",
                                  embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    def test_reviewer_accepts_injected_encoder_without_backend(self):
        cr = ConsensusReviewer(similarity_method="embedding", embedding_encoder=_FakeEncoder())
        assert cr.similarity_threshold == DEFAULT_THRESHOLDS["embedding"]

    @pytest.mark.skipif(not sim.embedding_backend_available(),
                        reason="sentence-transformers not installed in this environment")
    def test_real_backend_if_available(self):  # pragma: no cover - env dependent
        try:
            m = sim.similarity_matrix(list(PARAPHRASES[0]), "embedding",
                                      embedding_model=sim.DEFAULT_EMBEDDING_MODEL)
        except SimilarityBackendError as exc:
            pytest.skip(f"model not available locally: {exc}")
        assert m[0][1] > 0.5


# ── LLM adjudication ──────────────────────────────────────────────────


class _FakeAdjudicatorAdapter:
    def __init__(self, output, success=True):
        self.output = output
        self.success = success
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return AdapterRunResult(success=self.success, output=self.output,
                                error=None if self.success else "boom", token_usage=42)


class TestLLMAdjudicator:
    TEXTS = ["effect size missing", "no effect size", "seed undocumented", "TRRUST generic"]

    def test_adapter_groups_one_call(self):
        fake = _FakeAdjudicatorAdapter('Sure.\n```json\n{"groups": [[0, 1], [2], [3]]}\n```')
        res = asyncio.run(LLMAdjudicator(fake).group(self.TEXTS))
        assert res.success and res.called
        assert res.groups == [[0, 1], [2], [3]]
        assert len(fake.requests) == 1
        prompt = fake.requests[0].prompt_bundle.user_prompt
        assert "[0] effect size missing" in prompt and "[3] TRRUST generic" in prompt
        assert fake.requests[0].prompt_bundle.variables["role"] == "consensus_adjudicator"
        assert res.token_usage == 42

    def test_callable_backend_sync_and_async(self):
        res = LLMAdjudicator(lambda prompt: '{"groups": [[0, 1, 2, 3]]}').group_sync(self.TEXTS)
        assert res.groups == [[0, 1, 2, 3]]

        async def afn(prompt):
            return {"groups": [[3, 2], [1], [0]]}

        res = asyncio.run(LLMAdjudicator(afn).group(self.TEXTS))
        assert res.groups == [[0], [1], [2, 3]]

    def test_invalid_groups_are_repaired_and_reported(self):
        groups, issues = validate_groups([[0, 1], [1, 7], ["2"], [True]], 4)
        assert groups == [[0, 1], [2], [3]]
        assert any("duplicate index 1" in i for i in issues)
        assert any("out-of-range index 7" in i for i in issues)
        assert any("missing indices [3]" in i for i in issues)

    def test_lenient_group_output(self):
        groups, issues = parse_adjudication_output("```json\n{'groups': [[0, 1], [2,],],}\n", 3)
        assert groups == [[0, 1], [2]] and issues == []

    def test_unparseable_or_failed_call_falls_back_to_singletons(self):
        res = asyncio.run(LLMAdjudicator(_FakeAdjudicatorAdapter("no json here")).group(self.TEXTS))
        assert not res.success and res.groups == [[0], [1], [2], [3]]
        res = asyncio.run(LLMAdjudicator(_FakeAdjudicatorAdapter("x", success=False)).group(self.TEXTS))
        assert not res.success and "boom" in res.error

    def test_no_call_for_fewer_than_two(self):
        fake = _FakeAdjudicatorAdapter('{"groups": [[0]]}')
        res = asyncio.run(LLMAdjudicator(fake).group(["only one"]))
        assert res.groups == [[0]] and not res.called and fake.requests == []

    def test_group_sync_refuses_inside_loop(self):
        async def inner():
            with pytest.raises(RuntimeError):
                LLMAdjudicator(lambda p: "{}").group_sync(["a", "b"])
        asyncio.run(inner())


# ── Pure merge function + escalation ─────────────────────────────────


def _c(desc, sev=SeverityLevel.MEDIUM):
    return ReviewCritique(severity=sev, category="general", description=desc)


class TestMergeParsedCritiques:
    def test_groups_aligned_with_ranked_critiques(self):
        pairs = [
            ("reviewer", _c(PARAPHRASES[0][0], SeverityLevel.MEDIUM)),
            ("adversarial_reviewer", _c(PARAPHRASES[0][1], SeverityLevel.HIGH)),
            ("bio_plausibility_checker", _c(UNRELATED[1], SeverityLevel.LOW)),
        ]
        out = merge_parsed_critiques(pairs, method="tfidf")
        assert out.method == "tfidf" and out.threshold == DEFAULT_THRESHOLDS["tfidf"]
        assert out.groups == [[0, 1], [2]]
        top = out.result.critiques[0]
        assert top.severity == SeverityLevel.CRITICAL  # HIGH escalated by 2 lenses
        assert top.raised_by == ["adversarial_reviewer", "reviewer"]
        assert out.info["n_groups_multi_lens"] == 1
        # Jaccard at its legacy (uncalibrated) 0.5 does not merge this paraphrase pair.
        assert merge_parsed_critiques(pairs, method="jaccard",
                                      threshold=0.5).groups == [[1], [0], [2]]

    def test_escalation_counts_distinct_lenses_only(self):
        same = _c("Effect sizes are not reported for the main comparison", SeverityLevel.HIGH)
        pairs = [("reviewer", same), ("reviewer", same.model_copy())]
        out = merge_parsed_critiques(pairs, method="jaccard", same_lens_merge=True)
        assert out.groups == [[0, 1]]
        assert out.result.critiques[0].severity == SeverityLevel.HIGH  # not escalated
        assert out.result.critiques[0].raised_by == ["reviewer"]
        assert out.info["n_groups_multi_lens"] == 0
        pairs.append(("adversarial_reviewer", same.model_copy()))
        out = merge_parsed_critiques(pairs, method="jaccard")
        assert out.result.critiques[0].severity == SeverityLevel.CRITICAL
        # consensus_threshold=3 needs three distinct lenses: two lenses do not escalate.
        out3 = merge_parsed_critiques(pairs, method="jaccard", consensus_threshold=3)
        assert out3.result.critiques[0].severity == SeverityLevel.HIGH

    def test_same_lens_merge_flag(self):
        same = _c("Effect sizes are not reported for the main comparison")
        pairs = [("reviewer", same), ("reviewer", same.model_copy())]
        out = merge_parsed_critiques(pairs, method="jaccard", same_lens_merge=False)
        assert sorted(out.groups) == [[0], [1]]
        # False is the default
        assert sorted(merge_parsed_critiques(pairs, method="jaccard").groups) == [[0], [1]]

    def test_supplied_groups_override_and_are_validated(self):
        pairs = [("a", _c("x")), ("b", _c("y")), ("c", _c("z"))]
        out = merge_parsed_critiques(pairs, method="llm", groups=[[0, 2], [2]])
        assert sorted(out.groups) == [[0, 2], [1]]
        assert out.info["groups_supplied"] and out.info["group_issues"]

    def test_llm_method_with_callable_and_async_adapter(self):
        pairs = [("a", _c("x one")), ("b", _c("x two")), ("c", _c("unrelated"))]
        out = merge_parsed_critiques(pairs, method="llm",
                                     adjudicator=lambda p: '{"groups": [[0, 1], [2]]}')
        assert out.groups == [[0, 1], [2]] and out.info["adjudicator"]["success"]
        fake = _FakeAdjudicatorAdapter('```json\n{"groups": [[1, 0], [2]]}\n```')
        out = asyncio.run(amerge_parsed_critiques(pairs, method="llm", adjudicator=fake))
        assert out.groups == [[0, 1], [2]]
        assert len(fake.requests) == 1

    def test_llm_without_adjudicator_errors(self):
        with pytest.raises(SimilarityBackendError):
            merge_parsed_critiques([("a", _c("x")), ("b", _c("y"))], method="llm")

    def test_accepts_contract_dicts(self):
        pairs = [("a", {"severity": "high", "description": "Missing CI"}),
                 ("b", {"severity": "low", "description": "Missing CI"})]
        out = merge_parsed_critiques(pairs)
        assert len(out.result.critiques) == 1
        assert out.result.critiques[0].severity == SeverityLevel.CRITICAL

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            merge_parsed_critiques([], method="cosine")
        with pytest.raises(ValueError):
            ConsensusReviewer(similarity_method="cosine")

    def test_ranking_deterministic_under_permutation(self):
        pairs = [
            ("reviewer", _c("Alpha issue with effect sizes", SeverityLevel.HIGH)),
            ("adversarial_reviewer", _c("Beta issue with seeds", SeverityLevel.HIGH)),
            ("bio_plausibility_checker", _c("Gamma TRRUST context", SeverityLevel.HIGH)),
            ("reviewer", _c("Delta minor typo", SeverityLevel.LOW)),
        ]
        ref = [c.description for c in merge_parsed_critiques(pairs, method="tfidf").result.critiques]
        for perm in itertools.permutations(pairs):
            got = [c.description for c in merge_parsed_critiques(list(perm), method="tfidf").result.critiques]
            assert got == ref
