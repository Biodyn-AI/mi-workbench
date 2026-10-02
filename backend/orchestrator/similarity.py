"""Pluggable critique-similarity strategies and order-independent grouping.

The consensus step has to decide which critiques raised by different reviewer
lenses describe the *same underlying issue*. This module keeps that decision
separate from the consensus bookkeeping so methods can be swapped and
evaluated offline on cached reviewer outputs.

Similarity methods (all share :class:`SimilarityStrategy`'s ``matrix`` API,
except ``llm`` which returns groups directly):

``jaccard``
    Legacy whitespace-token Jaccard (``str.lower().split()``), kept
    bit-for-bit identical to the original ``ConsensusReviewer`` scorer.
    (Its default threshold is now the calibrated 0.1, not the legacy 0.5.)
``tfidf``
    Normalised lexical cosine: lowercase, punctuation stripped, English
    stopwords removed, light suffix stemming, TF-IDF weights fitted over the
    critiques of the current panel only.
``embedding``
    Cosine similarity of sentence embeddings. Optional: needs the
    ``sentence-transformers`` package and a configured model name (for example
    ``sentence-transformers/all-MiniLM-L6-v2``), or an injected encoder. This
    module never downloads anything itself; if the package is missing a
    :class:`SimilarityBackendError` with an actionable message is raised.
``llm`` (default, :data:`DEFAULT_SIMILARITY_METHOD`)
    An adjudicator (adapter or callable) receives the numbered critique list
    and returns groups of indices that share an underlying issue, as strict
    JSON. One call per consensus step. See :class:`LLMAdjudicator`. When the
    adjudication fails, the consensus step falls back to
    :data:`DEFAULT_LLM_FALLBACK` (``tfidf``) at its calibrated threshold.

Default thresholds (:data:`DEFAULT_THRESHOLDS`) and the default method were
chosen on the planted-flaw benchmark (:data:`CALIBRATION_SOURCE`);
:data:`UNCALIBRATED_THRESHOLDS` keeps the earlier, uncalibrated values.

Grouping for the matrix-based methods is order-independent average-linkage
(UPGMA) agglomerative clustering at threshold ``tau``: repeatedly merge the two
clusters with the highest mean pairwise similarity while that mean is
``>= tau``. Ties are broken on a canonical key of the cluster contents, and
means are computed with ``math.fsum``, so the resulting partition does not
depend on the order in which critiques were parsed.

Everything here is pure Python (no numpy required).
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from backend.orchestrator.feedback import extract_json_payloads

# ── Public constants ─────────────────────────────────────────────────

SIMILARITY_METHODS: tuple[str, ...] = ("jaccard", "tfidf", "embedding", "llm")

#: Where the shipped defaults come from: the planted-flaw benchmark's merge
#: evaluation (``experiments/benchmark/merge_eval.py``; 162 C5 panels, 4,556
#: cross-lens pairs labelled by the judge, leave-one-scenario-out CV over 12
#: flawed scenarios). Held-out pairwise F1: llm 0.824, tfidf 0.814,
#: jaccard 0.804 (embedding not evaluated: sentence-transformers unavailable).
CALIBRATION_SOURCE = "experiments/benchmark/results/merge_eval.json"

#: Default consensus similarity method: LLM adjudication, the pre-specified
#: winner of the merge evaluation (best held-out F1; one extra adjudicator
#: call per consensus step).
DEFAULT_SIMILARITY_METHOD = "llm"

#: Method used when the LLM adjudicator fails (call error, unparseable answer,
#: or an invalid partition after its retry) or no adjudicator is available:
#: TF-IDF at its calibrated threshold (the next-best method), never "no merging".
DEFAULT_LLM_FALLBACK = "tfidf"

# Method-specific default thresholds (tau). jaccard and tfidf: 0.1, the upper
# end of the thresholds selected in the leave-one-scenario-out folds of the
# merge evaluation (every fold chose 0.075 or 0.1 for both methods).
# embedding: 0.7, uncalibrated (not evaluated). llm: no threshold.
DEFAULT_THRESHOLDS: dict[str, Optional[float]] = {
    "jaccard": 0.1,
    "tfidf": 0.1,
    "embedding": 0.7,
    "llm": None,  # the adjudicator decides; no threshold
}

#: The thresholds shipped before the calibration (jaccard 0.5, tfidf 0.3,
#: embedding 0.7). They are uncalibrated: on the benchmark, jaccard at 0.5
#: recalled 1.2% of the duplicate pairs (tfidf at 0.3: 61.7%). Kept for
#: analyses that were pre-specified with them (the benchmark harness pins
#: jaccard 0.5 for its severity outcomes) and for reproducing earlier results.
UNCALIBRATED_THRESHOLDS: dict[str, Optional[float]] = {
    "jaccard": 0.5,
    "tfidf": 0.3,
    "embedding": 0.7,
    "llm": None,
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SimilarityBackendError(RuntimeError):
    """Raised when a similarity backend is misconfigured or unavailable."""


def resolve_threshold(method: str, threshold: Optional[float]) -> Optional[float]:
    """Return ``threshold`` or the method-specific default when it is None."""
    method = validate_method(method)
    if threshold is None:
        return DEFAULT_THRESHOLDS[method]
    return float(threshold)


def validate_method(method: str) -> str:
    name = (method or "").strip().lower()
    if name not in SIMILARITY_METHODS:
        raise ValueError(
            f"Unknown similarity method {method!r}; expected one of "
            f"{', '.join(SIMILARITY_METHODS)}"
        )
    return name


# ── Legacy Jaccard ───────────────────────────────────────────────────


def jaccard_similarity(a: str, b: str) -> float:
    """Legacy word-overlap Jaccard (bit-for-bit the original scorer).

    Tokenisation is ``str.lower().split()``: punctuation stays attached to
    words and there is no stopword removal or stemming.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ── Text normalisation for TF-IDF ────────────────────────────────────

# A compact English stopword list (function words only; no domain terms).
ENGLISH_STOPWORDS: frozenset[str] = frozenset("""
a about above after again against all almost also although always am among an and
any are aren around as at be because been before being below between both but by
can cannot could couldn did didn do does doesn doing done down during each either
else etc even ever every few for from further had hadn has hasn have haven having
he her here hers herself him himself his how however i if in into is isn it its
itself just least less let like made make makes many may me might more most much
must my myself neither no nor not now of off often on once one only onto or other
others otherwise our ours ourselves out over own per perhaps quite rather really
same shall she should shouldn since so some such than that the their theirs them
themselves then there these they this those though through thus to too toward
towards under until up upon us very via was wasn we were weren what whatever when
where whether which while who whom whose why will with within without would wouldn
yet you your yours yourself yourselves
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _measure(stem: str) -> int:
    """Porter measure m: the number of vowel-consonant sequences in ``stem``."""
    pattern = []
    for i, ch in enumerate(stem):
        is_vowel = ch in "aeiou" or (ch == "y" and i > 0 and pattern and pattern[-1] == "c")
        tag = "v" if is_vowel else "c"
        if not pattern or pattern[-1] != tag:
            pattern.append(tag)
    return "".join(pattern).count("vc")


def light_stem(word: str) -> str:
    """Light, deterministic suffix stemmer (plurals, -ed, -ing, -ly, final -e).

    Deliberately conservative: it only conflates inflectional variants
    ("reports"/"reported"/"reporting" -> "report", "sizes"/"size" -> "siz",
    "controlled"/"control" -> "control"). Numeric tokens and words of three
    characters or fewer are returned unchanged.
    """
    w = word
    if len(w) <= 3 or w.isdigit():
        return w
    # plurals
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("es") and w[:-2].endswith(("x", "z", "ch", "sh", "ss")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    # verbal / adverbial suffixes (stem must keep a vowel and some length)
    for suffix, min_stem in (("ing", 3), ("ed", 3), ("ly", 4)):
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if len(stem) >= min_stem and any(ch in "aeiouy" for ch in stem):
                w = stem
                # undouble a final consonant: "stopp" -> "stop" (not l/s/z)
                if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouylsz":
                    w = w[:-1]
            break
    # "controll" -> "control" only for longer stems ("call" stays "call")
    if w.endswith("ll") and _measure(w[:-1]) > 1:
        w = w[:-1]
    # final -e ("base"/"based" -> "bas", "size"/"sizes" -> "siz")
    if len(w) > 3 and w.endswith("e") and _measure(w[:-1]) >= 1:
        w = w[:-1]
    return w


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords/1-char tokens, stem."""
    tokens = _TOKEN_RE.findall((text or "").lower())
    out: list[str] = []
    for tok in tokens:
        if len(tok) < 2 or tok in ENGLISH_STOPWORDS:
            continue
        out.append(light_stem(tok))
    return out


# ── Vector helpers (pure Python) ─────────────────────────────────────


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = math.fsum(a[t] * b[t] for t in sorted(a) if t in b)
    na = math.sqrt(math.fsum(v * v for v in a.values()))
    nb = math.sqrt(math.fsum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _cosine_dense(a: Sequence[float], b: Sequence[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _symmetric_matrix(n: int, pair_fn: Callable[[int, int], float]) -> list[list[float]]:
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            s = float(pair_fn(i, j))
            sim[i][j] = s
            sim[j][i] = s
    return sim


# ── Strategies ───────────────────────────────────────────────────────


class SimilarityStrategy:
    """Common interface: ``matrix(texts) -> n x n symmetric similarity``."""

    name: str = "base"

    def matrix(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def pair(self, a: str, b: str) -> float:
        """Similarity of a single pair (fits on just these two texts)."""
        return self.matrix([a, b])[0][1]


class JaccardSimilarity(SimilarityStrategy):
    name = "jaccard"

    def matrix(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        return _symmetric_matrix(len(texts), lambda i, j: jaccard_similarity(texts[i], texts[j]))

    def pair(self, a: str, b: str) -> float:
        return jaccard_similarity(a, b)


class TfidfSimilarity(SimilarityStrategy):
    """TF-IDF cosine fitted over the texts passed to :meth:`matrix`.

    ``tf`` is ``1 + ln(count)`` (sublinear); ``idf`` is the smoothed
    ``ln((1 + N) / (1 + df)) + 1``. Vectors are compared with cosine.
    """

    name = "tfidf"

    def __init__(self, sublinear_tf: bool = True):
        self.sublinear_tf = sublinear_tf

    def vectors(self, texts: Sequence[str]) -> list[dict[str, float]]:
        docs = [Counter(normalize_tokens(t)) for t in texts]
        n = len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        idf = {t: math.log((1.0 + n) / (1.0 + c)) + 1.0 for t, c in df.items()}
        vecs: list[dict[str, float]] = []
        for d in docs:
            vec: dict[str, float] = {}
            for t, c in d.items():
                tf = (1.0 + math.log(c)) if self.sublinear_tf else float(c)
                vec[t] = tf * idf[t]
            vecs.append(vec)
        return vecs

    def matrix(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self.vectors(list(texts))
        return _symmetric_matrix(len(vecs), lambda i, j: _cosine_sparse(vecs[i], vecs[j]))


_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


def embedding_backend_available() -> bool:
    """True when ``sentence-transformers`` is importable (no import side effects)."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def _load_sentence_transformer(model_name: str) -> Any:
    if model_name in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # ImportError or a broken install
        raise SimilarityBackendError(
            "similarity_method='embedding' requires the optional "
            "'sentence-transformers' package (pip install sentence-transformers) "
            f"and a locally available model ({model_name!r}); import failed: {exc}. "
            "Use similarity_method='tfidf' or 'jaccard' instead, or inject an "
            "encoder via embedding_encoder=..."
        ) from exc
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise SimilarityBackendError(
            f"Could not load sentence-transformers model {model_name!r}: {exc}. "
            "Pre-fetch the model into the local cache or configure another "
            "embedding_model."
        ) from exc
    _EMBEDDING_MODEL_CACHE[model_name] = model
    return model


class EmbeddingSimilarity(SimilarityStrategy):
    """Cosine similarity of sentence embeddings.

    Either pass ``encoder`` (any object with ``encode(list[str])`` returning a
    sequence of vectors, or a plain callable) or ``model_name`` for a
    sentence-transformers model that is already available locally.
    """

    name = "embedding"

    def __init__(self, model_name: Optional[str] = None, encoder: Any = None):
        if encoder is None and not model_name:
            raise SimilarityBackendError(
                "similarity_method='embedding' needs embedding_model (e.g. "
                f"{DEFAULT_EMBEDDING_MODEL!r}) or an injected embedding_encoder"
            )
        self.model_name = model_name
        self._encoder = encoder

    def check_available(self) -> None:
        if self._encoder is None and not embedding_backend_available():
            raise SimilarityBackendError(
                "similarity_method='embedding' requires the optional "
                "'sentence-transformers' package, which is not installed in this "
                "environment (pip install sentence-transformers). Use "
                "similarity_method='tfidf' or 'jaccard' instead."
            )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        enc = self._encoder
        if enc is None:
            enc = _load_sentence_transformer(self.model_name or DEFAULT_EMBEDDING_MODEL)
        if hasattr(enc, "encode"):
            raw = enc.encode(texts)
        elif callable(enc):
            raw = enc(texts)
        else:
            raise SimilarityBackendError("embedding encoder must have .encode() or be callable")
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        vecs = [list(map(float, v.tolist() if hasattr(v, "tolist") else v)) for v in raw]
        if len(vecs) != len(texts):
            raise SimilarityBackendError(
                f"embedding encoder returned {len(vecs)} vectors for {len(texts)} texts"
            )
        return vecs

    def matrix(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        vecs = self._encode(texts)
        return _symmetric_matrix(len(vecs), lambda i, j: _cosine_dense(vecs[i], vecs[j]))


def make_strategy(
    method: str,
    embedding_model: Optional[str] = None,
    embedding_encoder: Any = None,
) -> SimilarityStrategy:
    """Instantiate the matrix-based strategy for ``method``."""
    method = validate_method(method)
    if method == "jaccard":
        return JaccardSimilarity()
    if method == "tfidf":
        return TfidfSimilarity()
    if method == "embedding":
        return EmbeddingSimilarity(model_name=embedding_model, encoder=embedding_encoder)
    raise ValueError("method 'llm' has no similarity matrix; use LLMAdjudicator")


def similarity_matrix(
    texts: Sequence[str],
    method: str = "jaccard",
    embedding_model: Optional[str] = None,
    embedding_encoder: Any = None,
) -> list[list[float]]:
    """Pairwise similarity matrix for ``texts`` under a matrix-based method."""
    return make_strategy(method, embedding_model, embedding_encoder).matrix(list(texts))


# ── Order-independent average-linkage clustering ─────────────────────


def average_linkage_groups(
    sim: Sequence[Sequence[float]],
    threshold: float,
    keys: Optional[Sequence[Any]] = None,
    lenses: Optional[Sequence[str]] = None,
    same_lens_merge: bool = False,
) -> list[list[int]]:
    """Average-linkage (UPGMA) agglomerative clustering on a similarity matrix.

    Repeatedly merges the pair of clusters with the highest mean pairwise
    similarity, as long as that mean is ``>= threshold``. Means are computed
    with ``math.fsum`` (exactly rounded, so independent of summation order)
    and ties are broken on the sorted ``keys`` of the two clusters' members.
    With content-based keys (``group_texts`` uses ``(text, lens)``) the
    resulting partition is independent of the order of the items; with the
    default keys (the item indices) only ties can depend on order.

    If ``same_lens_merge`` is False (the default; it needs ``lenses`` to have
    any effect), two clusters that share a lens are never merged, so each
    group holds at most one critique per lens.

    Returns groups as ascending index lists, sorted by their first index.
    """
    n = len(sim)
    if n == 0:
        return []
    key_list = [str(i).zfill(8) for i in range(n)] if keys is None else [repr(k) for k in keys]
    if len(key_list) != n:
        raise ValueError("keys must have one entry per item")
    if lenses is not None and len(lenses) != n:
        raise ValueError("lenses must have one entry per item")

    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    ckey: dict[int, tuple] = {i: (key_list[i],) for i in range(n)}
    clens: dict[int, frozenset] = {
        i: frozenset([lenses[i]]) if lenses is not None else frozenset() for i in range(n)
    }
    link: dict[tuple[int, int], float] = {}
    for x in range(n):
        for y in range(x + 1, n):
            link[(x, y)] = float(sim[x][y])

    def get_link(a: int, b: int) -> float:
        return link[(a, b)] if a < b else link[(b, a)]

    next_id = n
    while len(clusters) > 1:
        best: Optional[tuple] = None
        best_pair: Optional[tuple[int, int]] = None
        live = sorted(clusters)
        for ai, a in enumerate(live):
            for b in live[ai + 1:]:
                s = get_link(a, b)
                if s < threshold:
                    continue
                if not same_lens_merge and (clens[a] & clens[b]):
                    continue
                ka, kb = ckey[a], ckey[b]
                lo, hi = (ka, kb) if ka <= kb else (kb, ka)
                cand = (-s, lo, hi)
                if best is None or cand < best:
                    best = cand
                    best_pair = (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        members = sorted(clusters.pop(a) + clusters.pop(b))
        new = next_id
        next_id += 1
        for other, other_members in clusters.items():
            # new id is larger than every live id, so (other, new) is ordered
            link[(other, new)] = math.fsum(
                sim[i][j] for i in other_members for j in members
            ) / (len(other_members) * len(members))
        clusters[new] = members
        ckey[new] = tuple(sorted(ckey.pop(a) + ckey.pop(b)))
        clens[new] = clens.pop(a) | clens.pop(b)

    groups = [sorted(m) for m in clusters.values()]
    groups.sort(key=lambda g: g[0])
    return groups


def group_texts(
    texts: Sequence[str],
    method: str = "jaccard",
    threshold: Optional[float] = None,
    lenses: Optional[Sequence[str]] = None,
    embedding_model: Optional[str] = None,
    embedding_encoder: Any = None,
    same_lens_merge: bool = False,
    return_matrix: bool = False,
):
    """Group ``texts`` with a matrix-based method (jaccard/tfidf/embedding).

    Returns a list of index groups (or ``(groups, matrix)`` when
    ``return_matrix``). Keys for order-independent tie-breaking are
    ``(text, lens)``.
    """
    method = validate_method(method)
    if method == "llm":
        raise ValueError("group_texts does not support 'llm'; use LLMAdjudicator.group")
    tau = resolve_threshold(method, threshold)
    texts = list(texts)
    matrix = similarity_matrix(texts, method, embedding_model, embedding_encoder)
    keys = [
        (t, lenses[i] if lenses is not None else "")
        for i, t in enumerate(texts)
    ]
    groups = average_linkage_groups(
        matrix, tau, keys=keys, lenses=lenses, same_lens_merge=same_lens_merge
    )
    if return_matrix:
        return groups, matrix
    return groups


# ── LLM adjudication ─────────────────────────────────────────────────

ADJUDICATOR_SYSTEM_PROMPT = """\
You are a meticulous meta-reviewer. Several independent reviewers critiqued the
same research artifact. You receive their critiques as a numbered list.

Your only job is to decide which critiques identify the SAME underlying problem
in the artifact (the same flaw, even if worded differently, at a different
severity, or proposing a different fix). Critiques that concern different
problems must stay in different groups, even if they share vocabulary or touch
the same section. Do not judge whether a critique is correct.

Output ONLY one fenced JSON block, nothing else:

```json
{"groups": [[0, 3], [1], [2, 4]]}
```

Rules: every index from 0 to N-1 must appear in exactly one group; a critique
with no duplicate is a single-element group; use integers only; valid JSON
(double quotes, no trailing commas, no comments).
"""


def build_adjudication_prompt(texts: Sequence[str]) -> str:
    """User prompt listing the critiques as ``[i] text`` lines."""
    lines = [f"N = {len(texts)} critiques:", ""]
    for i, t in enumerate(texts):
        lines.append(f"[{i}] {' '.join(str(t).split())}")
    lines.append("")
    lines.append('Return {"groups": [...]} covering every index exactly once.')
    return "\n".join(lines)


# Issue prefixes produced by validate_groups. "missing" alone means the
# answer left out some indices (treated as singletons, i.e. "no duplicate");
# the others mean the answer is not a partition of 0..N-1.
_STRUCTURAL_ISSUES = ("groups is not a list", "non-list group", "non-integer index",
                      "out-of-range index", "duplicate index")


def structural_group_issues(issues: Sequence[str]) -> list[str]:
    """The issues that make an adjudicator answer an invalid partition
    (everything except indices merely left out)."""
    return [i for i in issues if i.startswith(_STRUCTURAL_ISSUES)]


def detect_one_based(raw_groups: Any, n: int) -> bool:
    """True if ``raw_groups`` looks like a partition of 1..n (a 1-based answer):
    integers only, 0 absent, n present, every index in 1..n exactly once."""
    if not isinstance(raw_groups, (list, tuple)) or n < 1:
        return False
    flat: list[int] = []
    for g in raw_groups:
        if isinstance(g, (int, str)) and not isinstance(g, bool):
            g = [g]
        if not isinstance(g, (list, tuple)):
            return False
        for x in g:
            if isinstance(x, bool):
                return False
            try:
                flat.append(int(str(x).strip().strip("[]#")))
            except (TypeError, ValueError):
                return False
    return bool(flat) and 0 not in flat and n in flat and sorted(flat) == list(range(1, n + 1))


def shift_groups(raw_groups: Sequence[Any], delta: int) -> list[list[int]]:
    out: list[list[int]] = []
    for g in raw_groups:
        g = [g] if isinstance(g, (int, str)) else g
        out.append([int(str(x).strip().strip("[]#")) + delta for x in g])
    return out


def validate_groups(raw_groups: Any, n: int) -> tuple[list[list[int]], list[str]]:
    """Validate adjudicator groups: every index in ``range(n)`` exactly once.

    Invalid entries are repaired conservatively and reported: out-of-range or
    non-integer indices are dropped, repeated indices keep their first
    occurrence, and missing indices become singleton groups. Callers that
    need a trustworthy partition must check :func:`structural_group_issues`
    (``LLMAdjudicator`` rejects such answers instead of using the repair).
    """
    issues: list[str] = []
    seen: set[int] = set()
    groups: list[list[int]] = []
    if not isinstance(raw_groups, (list, tuple)):
        issues.append("groups is not a list")
        raw_groups = []
    for g in raw_groups:
        if isinstance(g, (int, str)) and not isinstance(g, bool):
            g = [g]
        if isinstance(g, dict):
            g = g.get("indices") or g.get("members") or g.get("group") or []
        if not isinstance(g, (list, tuple)):
            issues.append(f"non-list group {g!r} ignored")
            continue
        cur: list[int] = []
        for x in g:
            try:
                if isinstance(x, bool):
                    raise ValueError
                idx = int(str(x).strip().strip("[]#"))
            except (TypeError, ValueError):
                issues.append(f"non-integer index {x!r} dropped")
                continue
            if idx < 0 or idx >= n:
                issues.append(f"out-of-range index {idx} dropped")
                continue
            if idx in seen:
                issues.append(f"duplicate index {idx} dropped")
                continue
            seen.add(idx)
            cur.append(idx)
        if cur:
            groups.append(sorted(cur))
    missing = [i for i in range(n) if i not in seen]
    if missing:
        issues.append(f"missing indices {missing} added as singletons")
        groups.extend([i] for i in missing)
    groups.sort(key=lambda g: g[0])
    return groups, issues


def parse_adjudication_output(output: Any, n: int) -> tuple[list[list[int]], list[str]]:
    """Parse an adjudicator response (text or already-decoded object)."""
    if isinstance(output, dict):
        return validate_groups(output.get("groups"), n)
    if isinstance(output, (list, tuple)):
        return validate_groups(list(output), n)
    text = str(output or "")
    for payload, _detail in reversed(extract_json_payloads(text)):
        if isinstance(payload, dict) and "groups" in payload:
            return validate_groups(payload["groups"], n)
        if isinstance(payload, list) and all(isinstance(g, (list, tuple)) for g in payload):
            return validate_groups(payload, n)
    raise ValueError("adjudicator output contains no parseable {\"groups\": ...} JSON")


@dataclass
class AdjudicationResult:
    """Outcome of one adjudication (up to two calls: one retry on an invalid
    partition). Usage fields are summed over every call made, including
    failed ones; provider / model / CLI version come from the adapter."""

    groups: list[list[int]]
    issues: list[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    raw_output: str = ""
    token_usage: int = 0
    cost_estimate: float = 0.0
    called: bool = False
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    provider: str = ""
    model: str = ""
    reasoning_effort: str = ""
    cli_version: str = ""
    # True when the accepted answer needed a benign repair (indices left out,
    # or a consistent 1-based answer shifted to 0-based); see ``issues``.
    repaired: bool = False


class _AdjudicatorCallError(RuntimeError):
    def __init__(self, message: str, result: Any = None):
        super().__init__(message)
        self.result = result


class LLMAdjudicator:
    """Groups critiques by asking a model which ones share an underlying issue.

    ``backend`` is either an adapter (anything with ``async run(request)``
    returning an ``AdapterRunResult``) or a callable ``fn(prompt: str)`` that
    returns text, a decoded ``{"groups": ...}`` object, or an awaitable of
    either. No call is made when fewer than two critiques are given.

    The answer must be a partition of ``0..N-1``. A consistently 1-based
    answer (a partition of ``1..N``) is shifted and recorded in ``issues``;
    indices merely left out become singletons (recorded, ``repaired=True``).
    Any other structural defect (out-of-range, duplicated or non-integer
    indices) is NOT repaired into a grouping: the call is retried once with
    the defects listed, and if the answer is still invalid (or the call
    fails / is unparseable) the result falls back to singleton groups (no
    merging, hence no escalation) with ``success=False`` and the error
    recorded.
    """

    def __init__(
        self,
        backend: Any,
        system_prompt: str = ADJUDICATOR_SYSTEM_PROMPT,
        timeout_seconds: float = 900.0,
        workspace_context: Any = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds
        self.workspace_context = workspace_context

    async def _call(self, prompt: str) -> tuple[Any, Any]:
        """One adjudicator call: ``(output, adapter_result_or_None)``.

        Raises ``_AdjudicatorCallError`` (carrying the adapter result, so its
        usage is still accounted) when the adapter reports a failure."""
        backend = self.backend
        run = getattr(backend, "run", None)
        if run is not None and inspect.iscoroutinefunction(run):
            from backend.models import AdapterRunRequest, PromptBundle, WorkspaceContext

            ctx = self.workspace_context or WorkspaceContext(workspace_path="")
            request = AdapterRunRequest(
                prompt_bundle=PromptBundle(
                    system_prompt=self.system_prompt,
                    user_prompt=prompt,
                    variables={"role": "consensus_adjudicator"},
                ),
                workspace_context=ctx,
                timeout_seconds=int(math.ceil(self.timeout_seconds)),
            )
            result = await asyncio.wait_for(run(request), timeout=self.timeout_seconds + 30)
            if not getattr(result, "success", False):
                raise _AdjudicatorCallError(
                    f"adjudicator call failed: {getattr(result, 'error', None)}", result)
            return getattr(result, "output", ""), result
        if callable(backend):
            out = backend(self.system_prompt + "\n\n" + prompt)
            if inspect.isawaitable(out):
                out = await out
            return out, None
        raise TypeError("adjudicator must be an adapter with async run() or a callable")

    @staticmethod
    def _add_usage(res: AdjudicationResult, adapter_result: Any) -> None:
        if adapter_result is None:
            return
        g = lambda k: getattr(adapter_result, k, None)  # noqa: E731
        res.token_usage += int(g("token_usage") or 0)
        res.cost_estimate += float(g("cost_estimate") or 0.0)
        res.input_tokens += int(g("input_tokens") or 0)
        res.output_tokens += int(g("output_tokens") or 0)
        res.cached_input_tokens += int(g("cached_input_tokens") or 0)
        for k in ("provider", "model", "reasoning_effort", "cli_version"):
            v = str(g(k) or "")
            if v and not getattr(res, k):
                setattr(res, k, v)

    @staticmethod
    def _interpret(out: Any, n: int) -> tuple[Optional[list[list[int]]], list[str], bool, str]:
        """Return ``(groups | None, issues, repaired, error)`` for one answer."""
        raw_groups: Any = None
        if isinstance(out, dict):
            raw_groups = out.get("groups")
        elif isinstance(out, (list, tuple)):
            raw_groups = list(out)
        else:
            for payload, _detail in reversed(extract_json_payloads(str(out or ""))):
                if isinstance(payload, dict) and "groups" in payload:
                    raw_groups = payload["groups"]
                    break
                if isinstance(payload, list) and all(isinstance(g, (list, tuple)) for g in payload):
                    raw_groups = payload
                    break
            else:
                return None, [], False, "adjudicator output contains no parseable {\"groups\": ...} JSON"
        if detect_one_based(raw_groups, n):
            groups, issues = validate_groups(shift_groups(raw_groups, -1), n)
            return groups, ["shifted 1-based indices to 0-based"] + issues, True, ""
        groups, issues = validate_groups(raw_groups, n)
        structural = structural_group_issues(issues)
        if structural:
            return None, issues, False, "invalid partition: " + "; ".join(structural)
        return groups, issues, bool(issues), ""

    async def group(self, texts: Sequence[str]) -> AdjudicationResult:
        texts = list(texts)
        n = len(texts)
        if n < 2:
            return AdjudicationResult(groups=[[i] for i in range(n)])
        prompt = build_adjudication_prompt(texts)
        res = AdjudicationResult(groups=[[i] for i in range(n)], called=True)
        all_issues: list[str] = []
        for attempt in range(2):
            res.n_calls += 1
            try:
                out, adapter_result = await self._call(prompt)
            except _AdjudicatorCallError as exc:
                self._add_usage(res, exc.result)
                res.success, res.error = False, str(exc)
                return res
            except Exception as exc:
                res.success, res.error = False, f"{type(exc).__name__}: {exc}"
                return res
            self._add_usage(res, adapter_result)
            res.raw_output = out if isinstance(out, str) else repr(out)
            groups, issues, repaired, error = self._interpret(out, n)
            if groups is not None:
                res.groups, res.repaired, res.success, res.error = groups, repaired, True, None
                res.issues = all_issues + issues
                return res
            all_issues.extend(f"attempt {attempt + 1}: {i}" for i in issues)
            res.issues = list(all_issues)
            res.success, res.error = False, error
            if not issues:  # unparseable: no point listing defects
                return res
            prompt = (
                build_adjudication_prompt(texts)
                + "\n\nYour previous answer was not a valid partition of 0.."
                + f"{n - 1}: " + "; ".join(issues)
                + ". Use 0-based indices; every index from 0 to "
                + f"{n - 1} exactly once."
            )
        res.groups = [[i] for i in range(n)]
        return res

    def group_sync(self, texts: Sequence[str]) -> AdjudicationResult:
        """Synchronous wrapper (only when no event loop is running)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.group(texts))
        raise RuntimeError(
            "LLMAdjudicator.group_sync called inside a running event loop; "
            "await LLMAdjudicator.group(...) instead"
        )
