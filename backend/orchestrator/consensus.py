"""Reviewer Consensus Mode - run multiple reviewers and merge critiques.

Pipeline for one consensus step:

1. ``run_panel`` calls every reviewer lens concurrently (optional concurrency
   limit), each with a per-lens timeout and transient-error retries
   (``backend.orchestrator.retry.run_with_retry``). A lens that raises, times
   out, returns ``success=False``, returns empty output, or returns output with
   no recognisable critique format (including a JSON contract block that
   cannot count as a review: all entries unreadable, or truncated before any
   critique was complete) is recorded as failed; it never silently counts as
   a clean review. ``run_panel(reuse=...)`` re-runs only the failed lenses of
   an earlier attempt.
2. Each lens output is parsed once: JSON critique contract, then YAML-like
   blocks, then the legacy bracket / structured / markdown line formats.
3. Critiques are grouped by a pluggable similarity method (``jaccard`` |
   ``tfidf`` | ``embedding`` | ``llm``, see ``backend.orchestrator.similarity``)
   with order-independent average-linkage clustering (or LLM adjudication).
   The default is ``llm`` (LLM adjudication with the panel adapter, the
   winner of the benchmark merge evaluation); if the adjudication fails (call
   error, unparseable answer, invalid partition after its retry) or no
   adjudicator is available, the step falls back to ``tfidf`` at its
   calibrated threshold (``llm_fallback``; recorded as ``fallback`` in
   ``merge_info`` and as ``similarity_fallback`` in the report).
   By default two critiques of the same reviewer call never merge
   (``same_lens_merge=False``), for every method.
4. Each group becomes one merged critique: highest severity in the group,
   escalated one level when at least ``consensus_threshold`` (>= 2) DISTINCT
   lenses raised it; ranked by (severity, role weight, number of lenses,
   text). The longest description represents the group and the other
   members' text and fixes are kept in ``merged_members``.
5. The grade is computed from the merged critiques, or ``"INCOMPLETE"`` when
   the panel failed (no usable lens). A partial panel's grade is flagged
   (``panel_partial``, ``grade_valid=False``). Per-lens status, parse methods, raw
   counts, similarity settings and group membership are recorded in
   ``ReviewResult.consensus_meta`` and surfaced by
   :meth:`ConsensusReviewer.compute_consensus_report`.

:func:`merge_parsed_critiques` / :func:`amerge_parsed_critiques` expose step
3-4 as pure functions over already-parsed ``(lens, critique)`` pairs so that
methods can be evaluated offline on cached reviewer outputs.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from backend.models import (
    AdapterRunRequest,
    AdapterRunResult,
    PromptBundle,
    ReviewCritique,
    ReviewResult,
    SeverityLevel,
    WorkspaceContext,
)
from backend.adapters.base import BaseAdapter
from backend.orchestrator.feedback import (
    ParsedCritiques,
    critique_from_mapping,
    extract_json_critiques,
    parse_yaml_critiques,
)
from backend.orchestrator.retry import run_with_retry
from backend.utils.config_values import parse_bool
from backend.orchestrator.similarity import (
    DEFAULT_LLM_FALLBACK,
    DEFAULT_SIMILARITY_METHOD,
    EmbeddingSimilarity,
    LLMAdjudicator,
    SimilarityBackendError,
    group_texts,
    jaccard_similarity,
    resolve_threshold,
    validate_groups,
    validate_method,
)
from backend.utils.output_cleaner import strip_thinking_traces

__all__ = [
    "ConsensusReviewer",
    "MergeOutcome",
    "LensReport",
    "merge_parsed_critiques",
    "amerge_parsed_critiques",
    "parse_lens_output",
    "INCOMPLETE_GRADE",
    "SimilarityBackendError",
]

# Prompt refs for the three reviewer roles
_REVIEWER_ROLES = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]

# Severity ordering for comparison and escalation
_SEVERITY_ORDER = {
    SeverityLevel.INFO: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}

# Threshold above which two critiques are considered duplicates (legacy
# Jaccard, uncalibrated; the calibrated defaults are similarity.DEFAULT_THRESHOLDS)
_SIMILARITY_THRESHOLD = 0.5

# Values of ``llm_fallback`` / ``consensus_llm_fallback`` meaning "no fallback"
# (the legacy behaviour: a failed adjudication merges nothing).
_NO_FALLBACK = ("", "none", "off", "false", "no")

# Default number of agreeing reviewers required to escalate a shared critique
_CONSENSUS_THRESHOLD = 2

# Panel call robustness defaults
DEFAULT_LENS_TIMEOUT = 900.0
DEFAULT_MAX_RETRIES = 3

# Grade reported when the panel produced no usable review
INCOMPLETE_GRADE = "INCOMPLETE"

# Per-lens status values
LENS_OK = "ok"
LENS_FAILED = "failed"      # exception, timeout, or success=False
LENS_EMPTY = "empty"        # success=True but no output
LENS_UNPARSED = "unparsed"  # output present but no recognisable critique format
                            # (or a contract block that cannot count as a review)

# Reviewer identity used for escalation / raised_by when a panel repeats a role.
LENS_IDENTITIES = ("role", "call")

# ── Legacy line formats (mock adapter and older prompts) ─────────────

# Pattern 1: [SEVERITY] description
_BRACKET_PATTERN = re.compile(
    r'\[(?P<sev>CRITICAL|HIGH|MEDIUM|LOW|INFO)\]\s*(?P<desc>.+)',
    re.IGNORECASE,
)
# Pattern 2: Severity: X, Category: Y, Description: Z
_STRUCTURED_PATTERN = re.compile(
    r'[Ss]everity:\s*(?P<sev>\w+)\s*[,;]\s*'
    r'[Cc]ategory:\s*(?P<cat>[^,;]+)\s*[,;]\s*'
    r'[Dd]escription:\s*(?P<desc>.+)',
    re.IGNORECASE,
)
# Pattern 3: **Severity: X** -- description (markdown bold)
_MARKDOWN_PATTERN = re.compile(
    r'\*\*[Ss]everity:\s*(?P<sev>\w+)\*\*\s*[-—]+\s*(?P<desc>.+)',
    re.IGNORECASE,
)


def _parse_legacy_lines(output: str) -> list[ReviewCritique]:
    """Legacy bracket / structured / markdown formats, one critique per span.

    Patterns are tried most-specific first (structured, markdown, bracket);
    a match that overlaps text already claimed by another match is skipped,
    so a line like ``- [HIGH] Severity: high, Category: x, Description: y``
    yields one critique, not two. Results are in document order.
    """
    found: list[tuple[int, ReviewCritique]] = []
    claimed: list[tuple[int, int]] = []

    def take(start: int, end: int, critique: ReviewCritique) -> None:
        if any(start < e and s < end for s, e in claimed):
            return
        claimed.append((start, end))
        found.append((start, critique))

    for match in _STRUCTURED_PATTERN.finditer(output):
        take(match.start(), match.end(), ReviewCritique(
            severity=_parse_severity(match.group("sev")),
            category=match.group("cat").strip(),
            description=match.group("desc").strip(),
        ))
    for match in _MARKDOWN_PATTERN.finditer(output):
        take(match.start(), match.end(), ReviewCritique(
            severity=_parse_severity(match.group("sev")),
            category="general",
            description=match.group("desc").strip(),
        ))
    for match in _BRACKET_PATTERN.finditer(output):
        take(match.start(), match.end(), ReviewCritique(
            severity=_parse_severity(match.group("sev")),
            category="general",
            description=match.group("desc").strip(),
        ))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


def parse_lens_output(output: str) -> ParsedCritiques:
    """Parse one lens output: JSON contract -> YAML-like -> legacy lines.

    The first method that yields critiques wins (an explicit, complete empty
    JSON ``critiques`` list counts as a valid, clean answer), so no critique
    is counted twice. ``method`` is one of ``json``, ``yaml``, ``regex``,
    ``none``.

    A contract block that cannot count as a review (every entry unreadable,
    e.g. an echoed template; or truncated before any critique was complete)
    is never a clean answer: the YAML / legacy parsers are tried on the text,
    and if they find nothing the result has ``method="none"`` (an unparsed
    lens) with ``detail="invalid_contract:<reason>"`` and the block's counts.
    """
    if not output or not output.strip():
        return ParsedCritiques(method="none")
    parsed = extract_json_critiques(output)
    if parsed is not None and parsed.is_valid_contract:
        return parsed
    yaml_critiques = parse_yaml_critiques(output)
    if yaml_critiques:
        return ParsedCritiques(critiques=yaml_critiques, method="yaml")
    legacy = _parse_legacy_lines(output)
    if legacy:
        return ParsedCritiques(critiques=legacy, method="regex")
    if parsed is not None:
        parsed.method = "none"
        parsed.detail = f"invalid_contract:{parsed.invalid_reason}"
        return parsed
    return ParsedCritiques(method="none")


# ── Pure merging over parsed critiques ───────────────────────────────


@dataclass
class MergeOutcome:
    """Result of merging parsed ``(lens, critique)`` pairs.

    ``groups[k]`` lists the input-pair indices merged into
    ``result.critiques[k]`` (ranked order).
    """

    result: ReviewResult
    groups: list[list[int]]
    method: str
    threshold: Optional[float]
    info: dict[str, Any] = field(default_factory=dict)


def _coerce_pairs(pairs: Sequence[tuple[str, Any]]) -> list[tuple[str, ReviewCritique]]:
    out: list[tuple[str, ReviewCritique]] = []
    for lens, c in pairs:
        if not isinstance(c, ReviewCritique):
            c2 = critique_from_mapping(c)
            if c2 is None:
                raise ValueError(f"cannot interpret critique {c!r}")
            c = c2
        out.append((str(lens), c))
    return out


def _sev_rank(s: SeverityLevel) -> int:
    return _SEVERITY_ORDER.get(s, 0)


def _build_merged(
    pairs: list[tuple[str, ReviewCritique]],
    groups: list[list[int]],
    consensus_threshold: int,
    role_weights: Optional[dict[str, float]],
) -> tuple[list[ReviewCritique], list[list[int]]]:
    entries = []
    for g in groups:
        members = [pairs[i] for i in g]
        best = max((c.severity for _, c in members), key=_sev_rank)
        names = sorted({lens for lens, _ in members})
        # Escalation counts DISTINCT lenses only.
        if len(names) >= consensus_threshold:
            best = _escalate_severity(best)
        # Longest description represents the group (text breaks length ties).
        rep_pos = max(range(len(members)),
                      key=lambda k: (len(members[k][1].description), members[k][1].description))
        rep = members[rep_pos][1]
        tag = f"[{', '.join(names)}]"
        # Keep every other member's text (its description may name a distinct
        # point the representative does not), in input order, without exact
        # repeats of the representative.
        others: list[dict[str, Any]] = []
        seen_desc = {" ".join(rep.description.split()).lower()}
        for k, (lens, c) in enumerate(members):
            if k == rep_pos:
                continue
            norm = " ".join(c.description.split()).lower()
            if norm in seen_desc:
                continue
            seen_desc.add(norm)
            others.append({
                "lens": lens,
                "severity": c.severity.value,
                "description": c.description,
                "required_fix": c.required_fix,
            })
        critique = ReviewCritique(
            severity=best,
            category=rep.category,
            description=f"{tag} {rep.description}",
            required_fix=rep.required_fix,
            suggested_experiment=rep.suggested_experiment,
            artifact_ref=rep.artifact_ref,
            section_ref=rep.section_ref,
            raised_by=names,
            merged_members=others,
        )
        weight = (
            max((role_weights.get(n, 1.0) for n in names), default=1.0)
            if role_weights else 1.0
        )
        canon = tuple(sorted((c.description, lens) for lens, c in members))
        key = (-_sev_rank(best), -weight, -len(names), critique.description, canon)
        entries.append((key, critique, sorted(g)))
    # Sort: severity desc, role-weight desc, reviewer count desc, then text
    # (and group content) so ties never depend on parse/gather ordering.
    entries.sort(key=lambda e: e[0])
    return [e[1] for e in entries], [e[2] for e in entries]


def _matrix_groups(
    pairs: list[tuple[str, ReviewCritique]],
    method: str,
    threshold: Optional[float],
    embedding_model: Optional[str],
    embedding_encoder: Any,
    same_lens_merge: bool,
    block_keys: Optional[Sequence[str]] = None,
) -> list[list[int]]:
    texts = [c.description for _, c in pairs]
    lenses = list(block_keys) if block_keys is not None else [lens for lens, _ in pairs]
    return group_texts(
        texts, method=method, threshold=threshold, lenses=lenses,
        embedding_model=embedding_model, embedding_encoder=embedding_encoder,
        same_lens_merge=same_lens_merge,
    )


def split_same_lens_groups(
    groups: Sequence[Sequence[int]], lenses: Sequence[str],
) -> tuple[list[list[int]], int]:
    """Split groups so that each holds at most one critique per lens.

    Members are placed, in ascending index order, into the first subgroup
    that has no member of their lens yet (a new subgroup otherwise), so a
    group of distinct lenses is unchanged. Returns ``(groups, n_split)``,
    where ``n_split`` counts the original groups that had to be split.
    """
    out: list[list[int]] = []
    n_split = 0
    for g in groups:
        subs: list[list[int]] = []
        sub_lenses: list[set[str]] = []
        for i in sorted(g):
            for sub, used in zip(subs, sub_lenses):
                if lenses[i] not in used:
                    sub.append(i)
                    used.add(lenses[i])
                    break
            else:
                subs.append([i])
                sub_lenses.append({lenses[i]})
        if len(subs) > 1:
            n_split += 1
        out.extend(subs)
    out.sort(key=lambda g: g[0])
    return out, n_split


def _check_consensus_threshold(consensus_threshold: Any) -> int:
    k = int(consensus_threshold)
    if k < 2:
        raise ValueError(
            "consensus_threshold must be >= 2 (escalation requires agreement of at "
            f"least two distinct lenses; k=1 would escalate every critique), got {k}"
        )
    return k


def _finish(pairs, groups, method, tau, consensus_threshold, role_weights, info) -> MergeOutcome:
    critiques, ranked = _build_merged(pairs, groups, consensus_threshold, role_weights)
    info = dict(info)
    info["n_groups_multi_lens"] = sum(
        1 for g in ranked if len({pairs[i][0] for i in g}) >= 2
    )
    return MergeOutcome(
        result=ReviewResult(overall_grade=_compute_grade(critiques), critiques=critiques),
        groups=ranked, method=method, threshold=tau, info=info,
    )


def _check_fallback(fallback_method: Optional[str]) -> Optional[str]:
    """Normalise an LLM fallback method: a matrix method, or None (no fallback)."""
    if fallback_method is None:
        return None
    name = str(fallback_method).strip().lower()
    if name in _NO_FALLBACK:
        return None
    name = validate_method(name)
    if name == "llm":
        raise ValueError("the LLM fallback must be a matrix method (jaccard, tfidf, "
                         "embedding) or 'none'")
    return name


def _fallback_groups(pairs, fallback_method, fallback_threshold, embedding_model,
                     embedding_encoder, same_lens_merge, block_keys, reason, info):
    """Matrix-method groups used when LLM adjudication is unavailable or failed."""
    tau = resolve_threshold(fallback_method, fallback_threshold)
    grp = _matrix_groups(pairs, fallback_method, tau, embedding_model, embedding_encoder,
                         same_lens_merge, block_keys)
    info["fallback"] = {"method": fallback_method, "threshold": tau, "reason": reason}
    return grp


def _prepare(pairs, method, threshold, groups, consensus_threshold, block_keys):
    pairs = _coerce_pairs(pairs)
    method = validate_method(method)
    tau = resolve_threshold(method, threshold)
    k = _check_consensus_threshold(consensus_threshold)
    if block_keys is not None:
        block_keys = [str(b) for b in block_keys]
        if len(block_keys) != len(pairs):
            raise ValueError("block_keys must have one entry per critique pair")
    info: dict[str, Any] = {}
    fixed = None
    if groups is not None:
        fixed, issues = validate_groups(groups, len(pairs))
        info["groups_supplied"] = True
        if issues:
            info["group_issues"] = issues
    return pairs, method, tau, info, fixed, k, block_keys


def _apply_same_lens_rule(pairs, groups, same_lens_merge, block_keys, info):
    """Enforce ``same_lens_merge=False`` on externally decided groups
    (supplied groups, LLM adjudication), which the matrix methods enforce
    inside the clustering itself."""
    if same_lens_merge:
        return groups
    lenses = block_keys if block_keys is not None else [lens for lens, _ in pairs]
    groups, n_split = split_same_lens_groups(groups, lenses)
    if n_split:
        info["same_lens_splits"] = n_split
    return groups


def merge_parsed_critiques(
    pairs: Sequence[tuple[str, Any]],
    *,
    method: str = "jaccard",
    threshold: Optional[float] = None,
    consensus_threshold: int = _CONSENSUS_THRESHOLD,
    role_weights: Optional[dict[str, float]] = None,
    embedding_model: Optional[str] = None,
    embedding_encoder: Any = None,
    adjudicator: Any = None,
    groups: Optional[Sequence[Sequence[int]]] = None,
    same_lens_merge: bool = False,
    block_keys: Optional[Sequence[str]] = None,
    fallback_method: Optional[str] = None,
    fallback_threshold: Optional[float] = None,
) -> MergeOutcome:
    """Merge already-parsed ``(lens, critique)`` pairs (no reviewer calls).

    ``critique`` may be a :class:`ReviewCritique` or a contract-style dict.
    ``method``/``threshold`` select the similarity strategy (threshold
    ``None`` = method default). These offline functions default to
    ``jaccard`` because they need no adjudicator; the platform's consensus
    step (:class:`ConsensusReviewer`) defaults to ``llm``. For ``llm``, either pass precomputed
    ``groups`` (e.g. a cached adjudicator answer) or an ``adjudicator``; the
    synchronous form runs it with ``asyncio.run`` and therefore cannot be used
    inside a running event loop (use :func:`amerge_parsed_critiques`).
    ``groups`` overrides the similarity method for any method.

    ``same_lens_merge`` (default False): whether two critiques of the same
    lens may end up in one group. The reviewer prompts already require one
    critique per distinct problem, so by default merging is cross-lens only;
    the rule is applied for every method (for ``llm`` and supplied groups by
    splitting groups after the fact). ``block_keys`` optionally gives the
    per-pair identity used for this rule (default: the pair's lens), e.g. one
    key per reviewer call when several calls share a lens name.
    ``consensus_threshold`` must be >= 2.

    ``fallback_method`` (default None = no fallback, the offline-evaluation
    behaviour): for ``llm``, the matrix method used when no adjudicator is
    given or the adjudication fails (``fallback_threshold`` None = that
    method's default); recorded as ``info["fallback"]``. Without a fallback a
    missing adjudicator raises :class:`SimilarityBackendError` and a failed
    adjudication merges nothing.

    Returns a :class:`MergeOutcome` whose ``groups`` are aligned with the
    ranked merged critiques.
    """
    pairs, method, tau, info, fixed, k, block_keys = _prepare(
        pairs, method, threshold, groups, consensus_threshold, block_keys)
    fallback_method = _check_fallback(fallback_method)
    if fixed is not None:
        grp = _apply_same_lens_rule(pairs, fixed, same_lens_merge, block_keys, info)
    elif not pairs:
        grp = []
    elif method == "llm":
        if adjudicator is None and fallback_method is None:
            raise SimilarityBackendError(
                "similarity_method='llm' needs an adjudicator (adapter or callable) "
                "or precomputed groups"
            )
        if adjudicator is None:
            grp = _fallback_groups(pairs, fallback_method, fallback_threshold,
                                   embedding_model, embedding_encoder, same_lens_merge,
                                   block_keys, "no adjudicator available", info)
        else:
            adj = (adjudicator if isinstance(adjudicator, LLMAdjudicator)
                   else LLMAdjudicator(adjudicator))
            res = adj.group_sync([c.description for _, c in pairs])
            info["adjudicator"] = _adjudication_info(res)
            grp = _llm_or_fallback(pairs, res, fallback_method, fallback_threshold,
                                   embedding_model, embedding_encoder, same_lens_merge,
                                   block_keys, info)
    else:
        grp = _matrix_groups(pairs, method, tau, embedding_model, embedding_encoder,
                             same_lens_merge, block_keys)
    return _finish(pairs, grp, method, tau, k, role_weights, info)


async def amerge_parsed_critiques(
    pairs: Sequence[tuple[str, Any]],
    *,
    method: str = "jaccard",
    threshold: Optional[float] = None,
    consensus_threshold: int = _CONSENSUS_THRESHOLD,
    role_weights: Optional[dict[str, float]] = None,
    embedding_model: Optional[str] = None,
    embedding_encoder: Any = None,
    adjudicator: Any = None,
    groups: Optional[Sequence[Sequence[int]]] = None,
    same_lens_merge: bool = False,
    workspace_context: Optional[WorkspaceContext] = None,
    adjudicator_timeout: float = DEFAULT_LENS_TIMEOUT,
    block_keys: Optional[Sequence[str]] = None,
    fallback_method: Optional[str] = None,
    fallback_threshold: Optional[float] = None,
) -> MergeOutcome:
    """Async :func:`merge_parsed_critiques` (supports adapter adjudicators)."""
    pairs, method, tau, info, fixed, k, block_keys = _prepare(
        pairs, method, threshold, groups, consensus_threshold, block_keys)
    fallback_method = _check_fallback(fallback_method)
    if fixed is not None:
        grp = _apply_same_lens_rule(pairs, fixed, same_lens_merge, block_keys, info)
    elif not pairs:
        grp = []
    elif method == "llm":
        if adjudicator is None and fallback_method is None:
            raise SimilarityBackendError(
                "similarity_method='llm' needs an adjudicator (adapter or callable) "
                "or precomputed groups"
            )
        if adjudicator is None:
            grp = _fallback_groups(pairs, fallback_method, fallback_threshold,
                                   embedding_model, embedding_encoder, same_lens_merge,
                                   block_keys, "no adjudicator available", info)
        else:
            adj = (
                adjudicator if isinstance(adjudicator, LLMAdjudicator)
                else LLMAdjudicator(adjudicator, timeout_seconds=adjudicator_timeout,
                                    workspace_context=workspace_context)
            )
            res = await adj.group([c.description for _, c in pairs])
            info["adjudicator"] = _adjudication_info(res)
            grp = _llm_or_fallback(pairs, res, fallback_method, fallback_threshold,
                                   embedding_model, embedding_encoder, same_lens_merge,
                                   block_keys, info)
    else:
        grp = _matrix_groups(pairs, method, tau, embedding_model, embedding_encoder,
                             same_lens_merge, block_keys)
    return _finish(pairs, grp, method, tau, k, role_weights, info)


def _llm_or_fallback(pairs, res, fallback_method, fallback_threshold, embedding_model,
                     embedding_encoder, same_lens_merge, block_keys, info):
    """Groups of a finished adjudication, or the fallback method's groups when
    it failed (and a fallback is configured)."""
    if res.success or fallback_method is None:
        return _apply_same_lens_rule(pairs, res.groups, same_lens_merge, block_keys, info)
    reason = f"adjudication failed: {res.error or 'unknown error'}"
    return _fallback_groups(pairs, fallback_method, fallback_threshold, embedding_model,
                            embedding_encoder, same_lens_merge, block_keys, reason, info)


def _adjudication_info(res) -> dict[str, Any]:
    return {
        "called": res.called,
        "success": res.success,
        "error": res.error,
        "issues": list(res.issues),
        "repaired": bool(getattr(res, "repaired", False)),
        "n_calls": int(getattr(res, "n_calls", 0) or 0),
        "token_usage": res.token_usage,
        "cost_estimate": res.cost_estimate,
        "input_tokens": int(getattr(res, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(res, "output_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(res, "cached_input_tokens", 0) or 0),
        "provider": getattr(res, "provider", "") or "",
        "model": getattr(res, "model", "") or "",
        "reasoning_effort": getattr(res, "reasoning_effort", "") or "",
        "cli_version": getattr(res, "cli_version", "") or "",
    }


# ── Per-lens bookkeeping ─────────────────────────────────────────────


@dataclass
class LensReport:
    """Outcome of one reviewer lens call."""

    name: str                      # role name (used for tags and escalation)
    key: str                       # unique within the panel (name, or name#k)
    status: str = LENS_OK
    error: Optional[str] = None
    parse_method: str = "none"
    parse_detail: str = ""
    critiques: list[ReviewCritique] = field(default_factory=list)
    overall_assessment: str = ""
    attempts: int = 1
    duration_seconds: float = 0.0
    # JSON-contract bookkeeping (see feedback.ParsedCritiques)
    n_raw_items: int = 0
    n_dropped: int = 0
    n_contract_blocks: int = 0
    dropped_earlier: int = 0
    reused: bool = False           # result carried over from an earlier panel attempt


class _GuardedAdapter:
    """Adapter proxy: hard timeout and exception capture for one lens.

    Timeouts and exceptions become ``AdapterRunResult(success=False)`` so one
    failing lens cannot abort the panel; "timeout" errors are transient for
    ``run_with_retry``.
    """

    def __init__(self, inner: Any, hard_timeout: Optional[float]):
        self.inner = inner
        self.hard_timeout = hard_timeout
        self.calls = 0

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        self.calls += 1
        try:
            if self.hard_timeout:
                return await asyncio.wait_for(self.inner.run(request), timeout=self.hard_timeout)
            return await self.inner.run(request)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return AdapterRunResult(
                success=False, exit_code=-1,
                error=f"Timeout: reviewer lens exceeded {self.hard_timeout:.0f}s",
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a lens failure
            return AdapterRunResult(
                success=False, exit_code=-1,
                error=f"Adapter exception: {type(exc).__name__}: {exc}",
            )


class ConsensusReviewer:
    """Runs a configurable panel of reviewer roles in parallel and merges results.

    The panel defaults to the three MI reviewer lenses (rigour, adversarial,
    biological plausibility) but any list of ``(role_name, prompt_ref)`` pairs
    may be supplied, so a loop can instantiate a panel of any size. ``merge``
    groups critiques that describe the same issue (``similarity_method``),
    escalates severity when at least ``consensus_threshold`` distinct lenses
    raise the same issue, and ranks the result deterministically.

    Configuration (keyword arguments; all optional):

    - ``similarity_method``: ``"llm"`` (default: LLM adjudication, the winner
      of the benchmark merge evaluation) | ``"tfidf"`` | ``"jaccard"`` |
      ``"embedding"``.
    - ``similarity_threshold``: tau; ``None`` = method default (calibrated:
      jaccard 0.1, tfidf 0.1; embedding 0.7 uncalibrated; unused for llm).
    - ``embedding_model`` / ``embedding_encoder``: for ``embedding``.
    - ``adjudicator``: adapter or callable for ``llm`` (``run_panel`` falls
      back to the panel adapter).
    - ``llm_fallback``: matrix method used at its default (calibrated)
      threshold when the LLM adjudication fails or no adjudicator is
      available (default ``"tfidf"``; ``"none"`` = merge nothing, the legacy
      behaviour). Recorded as ``similarity_fallback`` in the report.
    - ``lens_timeout``: per-lens timeout in seconds; defaults to
      ``adapter_timeout`` if given, else 900 s.
    - ``max_retries`` / ``retry_base_delay`` / ``retry_max_delay``: transient
      error retries per lens (``run_with_retry``).
    - ``max_concurrency``: limit on simultaneous lens calls (None = all).
    - ``same_lens_merge``: allow two critiques of one lens (one reviewer
      call) in a group (default False: merging is cross-lens only, because
      every lens prompt already asks for one critique per distinct problem).
    - ``lens_identity``: ``"role"`` (default) or ``"call"``. Identity used for
      escalation, ``raised_by`` and the multi-lens counts when a panel repeats
      a role: ``"role"`` treats repeated roles as ONE lens (no escalation
      among copies of one role); ``"call"`` treats every panel member as a
      distinct reviewer (self-ensembles escalate like a panel). The
      same-lens merge rule always applies per call.
    - ``unparsed_is_failure``: treat non-empty output with no recognisable
      critique format as a failed lens (default True).
    - ``min_ok_lenses``: fewer usable lenses than this => panel failed and
      grade ``"INCOMPLETE"`` (default 1). A panel with at least this many but
      not all lenses usable is *partial*: its grade is computed from the
      usable lenses and flagged (``panel_partial``, ``grade_valid=False``);
      the engine retries the failed lenses and, by default, never lets a
      partial panel's grade stop the run.
    - ``request_kwargs``: extra ``AdapterRunRequest`` fields for lens calls
      (e.g. ``{"model": ..., "allow_tools": False}``).
    - ``timeout_grace``: seconds added to ``lens_timeout`` for the hard asyncio
      guard (default ``max(5, 0.05 * lens_timeout)``).
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        consensus_threshold: int = _CONSENSUS_THRESHOLD,
        panel: Optional[list[tuple[str, str]]] = None,
        role_weights: Optional[dict[str, float]] = None,
        *,
        similarity_method: str = DEFAULT_SIMILARITY_METHOD,
        embedding_model: Optional[str] = None,
        embedding_encoder: Any = None,
        adjudicator: Any = None,
        lens_timeout: Optional[float] = None,
        adapter_timeout: Optional[float] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        max_concurrency: Optional[int] = None,
        same_lens_merge: Any = False,
        unparsed_is_failure: Any = True,
        min_ok_lenses: int = 1,
        request_kwargs: Optional[dict[str, Any]] = None,
        timeout_grace: Optional[float] = None,
        lens_identity: str = "role",
        llm_fallback: Optional[str] = DEFAULT_LLM_FALLBACK,
    ):
        self.similarity_method = validate_method(similarity_method)
        self.similarity_threshold = resolve_threshold(self.similarity_method, similarity_threshold)
        self.llm_fallback = _check_fallback(llm_fallback)
        self.consensus_threshold = _check_consensus_threshold(consensus_threshold)
        self.panel = panel or list(_REVIEWER_ROLES)
        # Optional per-role influence on ranking; default uniform (weight 1.0).
        self.role_weights = role_weights or {}
        self.embedding_model = embedding_model
        self.embedding_encoder = embedding_encoder
        self.adjudicator = adjudicator
        if lens_timeout is not None:
            self.lens_timeout = float(lens_timeout)
        elif adapter_timeout is not None:
            self.lens_timeout = float(adapter_timeout)
        else:
            self.lens_timeout = DEFAULT_LENS_TIMEOUT
        if self.lens_timeout <= 0:
            raise ValueError("lens_timeout must be positive")
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.retry_max_delay = max(0.0, float(retry_max_delay))
        self.max_concurrency = int(max_concurrency) if max_concurrency else None
        # Strict parsing: the string "false" must not mean True.
        self.same_lens_merge = parse_bool(same_lens_merge, "consensus_same_lens_merge")
        self.unparsed_is_failure = parse_bool(unparsed_is_failure, "consensus_unparsed_is_failure")
        self.min_ok_lenses = max(1, int(min_ok_lenses))
        identity = str(lens_identity or "role").strip().lower()
        if identity not in LENS_IDENTITIES:
            raise ValueError(f"lens_identity must be one of {LENS_IDENTITIES}, got {lens_identity!r}")
        self.lens_identity = identity
        # Fields set by run_panel itself cannot be overridden here.
        self.request_kwargs = {
            k: v for k, v in (request_kwargs or {}).items()
            if k not in ("prompt_bundle", "workspace_context", "timeout_seconds")
        }
        # Extra seconds before the asyncio guard fires, so the adapter's own
        # timeout (which kills the CLI subprocess) normally triggers first.
        self.timeout_grace = (
            max(5.0, 0.05 * self.lens_timeout) if timeout_grace is None
            else max(0.0, float(timeout_grace))
        )
        if self.similarity_method == "embedding" or (
                self.similarity_method == "llm" and self.llm_fallback == "embedding"):
            # Fail before any reviewer call if the backend cannot work.
            EmbeddingSimilarity(embedding_model, embedding_encoder).check_available()

    @classmethod
    def from_config(
        cls,
        config: Optional[dict[str, Any]],
        panel: Optional[list[tuple[str, str]]] = None,
        **overrides: Any,
    ) -> "ConsensusReviewer":
        """Build from run-config keys.

        Keys: ``consensus_similarity_method`` (default "llm"),
        ``consensus_similarity_threshold`` (None = method default),
        ``consensus_llm_fallback`` ("tfidf"; "none" = no fallback),
        ``consensus_threshold`` (2), ``consensus_role_weights``,
        ``consensus_embedding_model``, ``consensus_lens_timeout`` (falls back
        to ``adapter_timeout``, then 900), ``consensus_max_retries`` (falls
        back to ``max_retries``, then 3), ``consensus_retry_base_delay`` (1.0),
        ``consensus_max_concurrency``, ``consensus_same_lens_merge`` (False),
        ``consensus_unparsed_is_failure`` (True), ``consensus_min_ok_lenses``
        (1), ``consensus_lens_identity`` ("role"). Boolean keys accept
        true/false/1/0/yes/no/on/off (any other string is an error).
        ``overrides`` win over config values.
        """
        cfg = config or {}
        kwargs: dict[str, Any] = dict(
            similarity_threshold=cfg.get("consensus_similarity_threshold"),
            consensus_threshold=cfg.get("consensus_threshold", _CONSENSUS_THRESHOLD),
            panel=panel,
            role_weights=cfg.get("consensus_role_weights") or None,
            similarity_method=cfg.get("consensus_similarity_method") or DEFAULT_SIMILARITY_METHOD,
            embedding_model=cfg.get("consensus_embedding_model"),
            lens_timeout=cfg.get("consensus_lens_timeout"),
            adapter_timeout=cfg.get("adapter_timeout"),
            max_retries=cfg.get("consensus_max_retries", cfg.get("max_retries", DEFAULT_MAX_RETRIES)),
            retry_base_delay=cfg.get("consensus_retry_base_delay", 1.0),
            max_concurrency=cfg.get("consensus_max_concurrency"),
            same_lens_merge=cfg.get("consensus_same_lens_merge", False),
            unparsed_is_failure=cfg.get("consensus_unparsed_is_failure", True),
            min_ok_lenses=cfg.get("consensus_min_ok_lenses", 1),
            lens_identity=cfg.get("consensus_lens_identity") or "role",
            llm_fallback=cfg.get("consensus_llm_fallback", DEFAULT_LLM_FALLBACK),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    async def run_consensus(
        self,
        artifacts_content: str,
        adapter: BaseAdapter,
        workspace_context: WorkspaceContext,
    ) -> ReviewResult:
        """Run the reviewer panel in parallel and merge into one ranked list."""
        result, _ = await self.run_panel(artifacts_content, adapter, workspace_context)
        return result

    async def run_panel(
        self,
        artifacts_content: str,
        adapter: BaseAdapter,
        workspace_context: WorkspaceContext,
        system_prompt_builder: Optional[Callable[[str, str], str]] = None,
        reuse: Optional[Sequence[Optional[AdapterRunResult]]] = None,
    ) -> tuple[ReviewResult, list[AdapterRunResult]]:
        """Run the panel concurrently, returning the merged result and the raw
        per-reviewer results (for token/cost accounting and artifact writing).

        Each lens call gets ``timeout_seconds = lens_timeout`` (enforced by the
        adapter, with a hard asyncio guard slightly above it), transient-error
        retries, and runs under the optional concurrency limit. The raw result
        list is aligned with ``self.panel``; failed lenses appear as
        ``success=False`` results.

        ``reuse`` (aligned with the panel) carries results of an earlier
        attempt: a lens whose entry is not None is NOT called again and its
        earlier result is merged as is (used to retry only the failed lenses
        of a partial panel; the caller accounts only for fresh calls).
        """
        sem = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency else None
        hard_timeout = self.lens_timeout + self.timeout_grace

        async def call_lens(role_name: str, prompt_ref: str):
            if system_prompt_builder is not None:
                system_prompt = system_prompt_builder(role_name, prompt_ref)
            else:
                system_prompt = (
                    f"You are the {role_name}. Review the following artifacts."
                )
            request = AdapterRunRequest(
                prompt_bundle=PromptBundle(
                    system_prompt=system_prompt,
                    user_prompt=artifacts_content,
                    variables={"role": role_name},
                ),
                workspace_context=workspace_context,
                timeout_seconds=max(1, int(math.ceil(self.lens_timeout))),
                **self.request_kwargs,
            )
            guard = _GuardedAdapter(adapter, hard_timeout)
            started = time.monotonic()

            async def attempt() -> AdapterRunResult:
                return await run_with_retry(
                    guard, request,
                    max_retries=self.max_retries,
                    base_delay=self.retry_base_delay,
                    max_delay=self.retry_max_delay,
                )

            if sem is not None:
                async with sem:
                    result = await attempt()
            else:
                result = await attempt()
            if result is None:  # defensive: retry helper returned nothing
                result = AdapterRunResult(success=False, error="no result", exit_code=-1)
            return result, {
                "attempts": guard.calls,
                "duration_seconds": round(time.monotonic() - started, 3),
            }

        async def reused(i: int):
            return reuse[i], {"attempts": 0, "duration_seconds": 0.0, "reused": True}

        coros = []
        for i, (r, p) in enumerate(self.panel):
            if reuse is not None and i < len(reuse) and reuse[i] is not None:
                coros.append(reused(i))
            else:
                coros.append(call_lens(r, p))
        outs = await asyncio.gather(*coros)
        results: list[AdapterRunResult] = [o[0] for o in outs]
        call_info = [o[1] for o in outs]
        role_names = [r[0] for r in self.panel]
        adjudicator = self.adjudicator if self.adjudicator is not None else adapter
        merged = await self.amerge_critiques(
            results, role_names=role_names, adjudicator=adjudicator,
            workspace_context=workspace_context, call_info=call_info,
        )
        return merged, results

    # ── merging ──────────────────────────────────────────────────────

    def analyze_lenses(
        self,
        results: Sequence[Optional[AdapterRunResult]],
        role_names: Optional[Sequence[str]] = None,
        call_info: Optional[Sequence[dict[str, Any]]] = None,
    ) -> list[LensReport]:
        """Classify and parse every lens result (status, parse method, critiques)."""
        if role_names is None:
            role_names = [r[0] for r in self.panel]
        names = [
            role_names[i] if i < len(role_names) else f"reviewer_{i}"
            for i in range(len(results))
        ]
        totals: dict[str, int] = {}
        for n in names:
            totals[n] = totals.get(n, 0) + 1
        seen: dict[str, int] = {}
        reports: list[LensReport] = []
        for i, result in enumerate(results):
            name = names[i]
            seen[name] = seen.get(name, 0) + 1
            key = name if totals[name] == 1 else f"{name}#{seen[name]}"
            rep = LensReport(name=name, key=key)
            if call_info is not None and i < len(call_info):
                rep.attempts = int(call_info[i].get("attempts", 1))
                rep.duration_seconds = float(call_info[i].get("duration_seconds", 0.0))
                rep.reused = bool(call_info[i].get("reused", False))
            if result is None or not getattr(result, "success", False):
                rep.status = LENS_FAILED
                rep.error = (getattr(result, "error", None) if result is not None else None) \
                    or "reviewer call failed"
            elif not (result.output or "").strip():
                rep.status = LENS_EMPTY
                rep.error = "empty output"
            else:
                parsed = parse_lens_output(strip_thinking_traces(result.output))
                rep.parse_method = parsed.method
                rep.parse_detail = parsed.detail
                rep.critiques = list(parsed.critiques)
                rep.overall_assessment = parsed.overall_assessment
                rep.n_raw_items = parsed.n_raw_items
                rep.n_dropped = parsed.n_dropped
                rep.n_contract_blocks = parsed.n_contract_blocks
                rep.dropped_earlier = parsed.dropped_earlier
                if parsed.method == "none" and self.unparsed_is_failure:
                    rep.status = LENS_UNPARSED
                    if parsed.invalid_reason:
                        rep.error = ("critique block cannot count as a review "
                                     f"({parsed.invalid_reason})")
                    else:
                        rep.error = "no recognisable critique format in output"
            reports.append(rep)
        return reports

    def _pairs(self, reports: list[LensReport]):
        """(lens, critique) pairs of the usable lenses, the per-call index, and
        the per-call keys used for the same-lens merge rule.

        The pair lens is the role name (``lens_identity="role"``) or the
        unique panel key (``"call"``); it drives escalation, ``raised_by``
        and the multi-lens counts. The same-lens rule always uses the call
        key, so copies of one role can still merge across calls."""
        pairs: list[tuple[str, ReviewCritique]] = []
        index: list[tuple[str, int]] = []
        block: list[str] = []
        for rep in reports:
            if rep.status != LENS_OK:
                continue
            lens = rep.key if self.lens_identity == "call" else rep.name
            for j, c in enumerate(rep.critiques):
                pairs.append((lens, c))
                index.append((rep.key, j))
                block.append(rep.key)
        return pairs, index, block

    def _merge_kwargs(self) -> dict[str, Any]:
        return dict(
            method=self.similarity_method,
            threshold=self.similarity_threshold,
            consensus_threshold=self.consensus_threshold,
            role_weights=self.role_weights or None,
            embedding_model=self.embedding_model,
            embedding_encoder=self.embedding_encoder,
            same_lens_merge=self.same_lens_merge,
            fallback_method=self.llm_fallback,
        )

    def merge_critiques(
        self,
        results: list[AdapterRunResult],
        role_names: Optional[list[str]] = None,
    ) -> ReviewResult:
        """Merge multiple review results into consensus (synchronous).

        - Parse each lens once (JSON -> YAML -> legacy lines); failed / empty
          / unparseable lenses are recorded and contribute nothing
        - Group critiques that describe the same issue (``similarity_method``)
        - Escalate severity if at least ``consensus_threshold`` distinct lenses
          flag it
        - Rank by: severity (desc), role weight, reviewer count (desc), text
        - Tag each critique with which reviewer(s) raised it

        For ``similarity_method='llm'`` prefer :meth:`amerge_critiques`; this
        synchronous form runs the adjudicator with ``asyncio.run`` (without an
        adjudicator it uses ``llm_fallback``).
        """
        reports = self.analyze_lenses(results, role_names)
        pairs, index, block = self._pairs(reports)
        outcome = merge_parsed_critiques(
            pairs, adjudicator=self.adjudicator, block_keys=block, **self._merge_kwargs()
        )
        return self._finalize(reports, pairs, index, outcome)

    async def amerge_critiques(
        self,
        results: list[AdapterRunResult],
        role_names: Optional[list[str]] = None,
        adjudicator: Any = None,
        workspace_context: Optional[WorkspaceContext] = None,
        call_info: Optional[Sequence[dict[str, Any]]] = None,
    ) -> ReviewResult:
        """Async :meth:`merge_critiques` (one adjudicator call for ``llm``)."""
        reports = self.analyze_lenses(results, role_names, call_info)
        pairs, index, block = self._pairs(reports)
        adj = adjudicator if adjudicator is not None else self.adjudicator
        outcome = await amerge_parsed_critiques(
            pairs, adjudicator=adj, workspace_context=workspace_context,
            adjudicator_timeout=self.lens_timeout, block_keys=block, **self._merge_kwargs()
        )
        return self._finalize(reports, pairs, index, outcome)

    def _finalize(
        self,
        reports: list[LensReport],
        pairs: list[tuple[str, ReviewCritique]],
        index: list[tuple[str, int]],
        outcome: MergeOutcome,
    ) -> ReviewResult:
        merged = outcome.result
        ok = [r for r in reports if r.status == LENS_OK]
        failed = [r.key for r in reports if r.status != LENS_OK]
        panel_failed = len(ok) < self.min_ok_lenses
        panel_partial = bool(failed) and not panel_failed
        grade = INCOMPLETE_GRADE if panel_failed else merged.overall_grade
        multi_call = sum(1 for g in outcome.groups if len({index[i][0] for i in g}) >= 2)
        meta: dict[str, Any] = {
            "panel_size": len(reports),
            "lenses": [r.key for r in reports],
            "lens_status": {r.key: r.status for r in reports},
            "lens_errors": {r.key: r.error for r in reports if r.error},
            "failed_lenses": failed,
            "ok_lenses": len(ok),
            "panel_failed": panel_failed,
            "panel_partial": panel_partial,
            # A grade is a full-panel grade only when every lens was usable.
            "panel_complete": not failed,
            "grade_valid": not failed,
            "raw_critique_counts": {r.key: len(r.critiques) for r in reports},
            "parse_methods": {r.key: r.parse_method for r in reports},
            "parse_details": {r.key: r.parse_detail for r in reports if r.parse_detail},
            # Usable lenses whose contract block was truncated: a trailing
            # critique may be missing (partial loss, lens still counted).
            "truncated_lenses": [r.key for r in reports
                                 if r.status == LENS_OK and r.parse_detail == "truncated"],
            "dropped_items": {r.key: r.n_dropped for r in reports if r.n_dropped},
            # More than one contract block (contract violation; last one counted)
            "multi_block_lenses": {r.key: r.n_contract_blocks for r in reports
                                   if r.n_contract_blocks > 1},
            "dropped_earlier": {r.key: r.dropped_earlier for r in reports if r.dropped_earlier},
            "reused_lenses": [r.key for r in reports if r.reused],
            "overall_assessments": {
                r.key: r.overall_assessment for r in reports if r.overall_assessment
            },
            "attempts": {r.key: r.attempts for r in reports},
            "durations_seconds": {r.key: r.duration_seconds for r in reports},
            "similarity_method": outcome.method,
            "similarity_threshold": outcome.threshold,
            # Set when LLM adjudication failed (or had no adjudicator) and the
            # groups come from the fallback method: {method, threshold, reason}.
            "similarity_fallback": outcome.info.get("fallback"),
            "effective_similarity_method": (outcome.info.get("fallback") or {}).get(
                "method", outcome.method),
            "same_lens_merge": self.same_lens_merge,
            "lens_identity": self.lens_identity,
            "n_parsed_critiques": len(pairs),
            # Counted with the lens identity (role or call; see lens_identity).
            "n_groups_multi_lens": outcome.info.get("n_groups_multi_lens", 0),
            # Counted over distinct panel calls (merge_groups keys).
            "n_groups_multi_call": multi_call,
            # merge_groups[k] = [[call_key, critique_index_within_call], ...]
            # for merged critique k (ranked order). Keys are per-call panel
            # keys (role, or role#k when a role repeats).
            "merge_groups": [[list(index[i]) for i in g] for g in outcome.groups],
            "merge_info": {k: v for k, v in outcome.info.items() if k != "n_groups_multi_lens"},
            "grade": grade,
        }
        return ReviewResult(
            overall_grade=grade,
            critiques=merged.critiques,
            consensus_meta=meta,
        )

    def compute_consensus_report(self, merged: ReviewResult) -> dict:
        """Summarise agreement across the panel for the merged critique list.

        Returns counts used for reporting and for the (optional) advancement
        gate: how many merged critiques were raised by a single reviewer
        (``single_reviewer``, exactly one lens) versus by multiple reviewers
        (``multi_reviewer``, two or more lenses), how many reached the
        escalation threshold (``escalated_by_agreement``, at least
        ``consensus_threshold`` lenses), the per-severity histogram, and the
        number of unresolved CRITICAL items;
        plus panel health (``panel_failed``, ``panel_partial``,
        ``failed_lenses``, ``lens_status``, ``raw_critique_counts``,
        ``parse_methods``) and merge provenance (``similarity_method``,
        ``similarity_threshold``, ``similarity_fallback`` /
        ``effective_similarity_method`` when a failed LLM adjudication fell
        back to a matrix method, ``n_groups_multi_lens``, ``merge_groups``).
        """
        single, multi, escalated = 0, 0, 0
        histogram: dict[str, int] = {}
        for c in merged.critiques:
            n = len(c.raised_by) if c.raised_by else _count_reviewers_in_tag(c.description)
            if n >= 2:
                multi += 1
            else:
                single += 1
            if n >= self.consensus_threshold:
                escalated += 1
            histogram[c.severity.value] = histogram.get(c.severity.value, 0) + 1
        meta = merged.consensus_meta or {}
        failed_lenses = list(meta.get("failed_lenses", []))
        return {
            "panel_size": len(self.panel),
            "consensus_threshold": self.consensus_threshold,
            "total_merged": len(merged.critiques),
            "single_reviewer": single,
            "multi_reviewer": multi,
            "escalated_by_agreement": escalated,
            "severity_histogram": histogram,
            "unresolved_critical": sum(
                1 for c in merged.critiques if c.severity == SeverityLevel.CRITICAL
            ),
            "grade": merged.overall_grade,
            "panel_failed": bool(meta.get("panel_failed", False)),
            "panel_partial": bool(meta.get("panel_partial", False)),
            "panel_complete": bool(meta.get("panel_complete", not failed_lenses)),
            "grade_valid": bool(meta.get("grade_valid", not failed_lenses)),
            "failed_lenses": failed_lenses,
            "lens_status": dict(meta.get("lens_status", {})),
            "lens_errors": dict(meta.get("lens_errors", {})),
            "raw_critique_counts": dict(meta.get("raw_critique_counts", {})),
            "parse_methods": dict(meta.get("parse_methods", {})),
            "similarity_method": meta.get("similarity_method", self.similarity_method),
            "similarity_threshold": meta.get("similarity_threshold", self.similarity_threshold),
            "similarity_fallback": meta.get("similarity_fallback"),
            "effective_similarity_method": meta.get(
                "effective_similarity_method", meta.get("similarity_method", self.similarity_method)),
            "n_groups_multi_lens": meta.get("n_groups_multi_lens", multi),
            "n_groups_multi_call": meta.get("n_groups_multi_call", multi),
            "lens_identity": meta.get("lens_identity", self.lens_identity),
            "truncated_lenses": list(meta.get("truncated_lenses", [])),
            "multi_block_lenses": dict(meta.get("multi_block_lenses", {})),
            "merge_groups": list(meta.get("merge_groups", [])),
        }

    def _parse_review_output(self, output: str) -> list[ReviewCritique]:
        """Parse reviewer output into structured critiques.

        Tries, in order (first that yields critiques wins, each critique once):
        - the fenced JSON critique contract (lenient decoding)
        - YAML-like ``- severity: X`` / ``description: Y`` blocks
        - [CRITICAL] description text / [HIGH] description text
        - Severity: high, Category: stats, Description: ...
        - **Severity: MEDIUM** -- description
        """
        return list(parse_lens_output(output or "").critiques)

    def parse_review_output_detailed(self, output: str) -> ParsedCritiques:
        """Like :meth:`_parse_review_output` but also returns the parse method."""
        return parse_lens_output(output or "")

    def _critique_weight(self, description: str) -> float:
        """Ranking weight for a merged critique from its reviewer tag.

        The weight is the maximum ``role_weights`` value among the reviewers that
        raised the critique (default 1.0 for any role not listed). With no
        configured weights every critique weighs 1.0 and ranking is unchanged.
        """
        if not self.role_weights:
            return 1.0
        match = re.match(r'\[([^\]]+)\]', description)
        if not match:
            return 1.0
        reviewers = [r.strip() for r in match.group(1).split(',')]
        return max((self.role_weights.get(r, 1.0) for r in reviewers), default=1.0)

    def _similarity_score(self, a: str, b: str) -> float:
        """Legacy word-overlap (Jaccard) similarity (see similarity.jaccard_similarity)."""
        return jaccard_similarity(a, b)


def _parse_severity(raw: str) -> SeverityLevel:
    """Parse a severity string into a SeverityLevel enum."""
    mapping = {
        "critical": SeverityLevel.CRITICAL,
        "high": SeverityLevel.HIGH,
        "medium": SeverityLevel.MEDIUM,
        "low": SeverityLevel.LOW,
        "info": SeverityLevel.INFO,
    }
    return mapping.get(raw.strip().lower(), SeverityLevel.MEDIUM)


def _escalate_severity(severity: SeverityLevel) -> SeverityLevel:
    """Bump severity up one level (capped at CRITICAL)."""
    order = [SeverityLevel.INFO, SeverityLevel.LOW, SeverityLevel.MEDIUM,
             SeverityLevel.HIGH, SeverityLevel.CRITICAL]
    idx = order.index(severity)
    return order[min(idx + 1, len(order) - 1)]


def _count_reviewers_in_tag(description: str) -> int:
    """Count reviewer names in the [reviewer1, reviewer2] tag prefix."""
    match = re.match(r'\[([^\]]+)\]', description)
    if not match:
        return 1
    return len(match.group(1).split(','))


def _compute_grade(critiques: list[ReviewCritique]) -> str:
    """Compute an overall grade from the merged critique list."""
    if not critiques:
        return "A"
    max_sev = max(_SEVERITY_ORDER.get(c.severity, 0) for c in critiques)
    critical_count = sum(1 for c in critiques if c.severity == SeverityLevel.CRITICAL)
    high_count = sum(1 for c in critiques if c.severity == SeverityLevel.HIGH)

    if critical_count >= 2:
        return "F"
    if critical_count >= 1:
        return "D"
    if high_count >= 3:
        return "D"
    if high_count >= 1:
        return "C"
    if max_sev >= _SEVERITY_ORDER[SeverityLevel.MEDIUM]:
        return "B"
    return "A"
