"""Reviewer Consensus Mode - run multiple reviewers and merge critiques."""
from __future__ import annotations

import asyncio
import re
from typing import Optional

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
from backend.utils.output_cleaner import strip_thinking_traces

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

# Threshold above which two critiques are considered duplicates
_SIMILARITY_THRESHOLD = 0.5

# Default number of agreeing reviewers required to escalate a shared critique
_CONSENSUS_THRESHOLD = 2


class ConsensusReviewer:
    """Runs a configurable panel of reviewer roles in parallel and merges results.

    The panel defaults to the three MI reviewer lenses (rigour, adversarial,
    biological plausibility) but any list of ``(role_name, prompt_ref)`` pairs
    may be supplied, so a loop can instantiate a panel of any size. ``merge``
    deduplicates near-identical critiques (word-level Jaccard), escalates
    severity when at least ``consensus_threshold`` distinct reviewers raise the
    same issue, and ranks the result deterministically.
    """

    def __init__(
        self,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        consensus_threshold: int = _CONSENSUS_THRESHOLD,
        panel: Optional[list[tuple[str, str]]] = None,
        role_weights: Optional[dict[str, float]] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.consensus_threshold = max(1, consensus_threshold)
        self.panel = panel or list(_REVIEWER_ROLES)
        # Optional per-role influence on ranking; default uniform (weight 1.0).
        self.role_weights = role_weights or {}

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
        system_prompt_builder: Optional[callable] = None,
    ) -> tuple[ReviewResult, list[AdapterRunResult]]:
        """Run the panel concurrently, returning the merged result and the raw
        per-reviewer results (for token/cost accounting and artifact writing)."""
        tasks = []
        for role_name, prompt_ref in self.panel:
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
            )
            tasks.append(adapter.run(request))

        results: list[AdapterRunResult] = await asyncio.gather(*tasks)
        role_names = [r[0] for r in self.panel]
        return self.merge_critiques(results, role_names=role_names), results

    def merge_critiques(
        self,
        results: list[AdapterRunResult],
        role_names: Optional[list[str]] = None,
    ) -> ReviewResult:
        """Merge multiple review results into consensus.

        - Deduplicate similar critiques (fuzzy match on description)
        - Escalate severity if at least ``consensus_threshold`` reviewers flag it
        - Rank by: severity (desc), reviewer count (desc), then text (stable tie-break)
        - Tag each critique with which reviewer(s) raised it
        """
        # Parse all critiques from each reviewer
        all_tagged: list[tuple[ReviewCritique, str]] = []
        if role_names is None:
            role_names = [r[0] for r in self.panel]
        for i, result in enumerate(results):
            role_name = role_names[i] if i < len(role_names) else f"reviewer_{i}"
            critiques = self._parse_review_output(strip_thinking_traces(result.output))
            for c in critiques:
                all_tagged.append((c, role_name))

        # Deduplicate: group similar critiques together
        groups: list[list[tuple[ReviewCritique, str]]] = []
        used = [False] * len(all_tagged)

        for i, (ci, ri) in enumerate(all_tagged):
            if used[i]:
                continue
            group = [(ci, ri)]
            used[i] = True
            for j in range(i + 1, len(all_tagged)):
                if used[j]:
                    continue
                cj, rj = all_tagged[j]
                if self._similarity_score(ci.description, cj.description) >= self.similarity_threshold:
                    group.append((cj, rj))
                    used[j] = True
            groups.append(group)

        # Build merged critiques
        merged: list[ReviewCritique] = []
        for group in groups:
            # Pick the highest severity in the group
            best_severity = max(
                (c.severity for c, _ in group),
                key=lambda s: _SEVERITY_ORDER.get(s, 0),
            )
            # Escalate if enough distinct reviewers flagged the same issue
            reviewer_names = sorted(set(r for _, r in group))
            if len(reviewer_names) >= self.consensus_threshold:
                best_severity = _escalate_severity(best_severity)

            # Use the longest description as the representative
            representative = max(group, key=lambda x: len(x[0].description))
            critique = representative[0]

            tag = f"[{', '.join(reviewer_names)}]"
            merged.append(ReviewCritique(
                severity=best_severity,
                category=critique.category,
                description=f"{tag} {critique.description}",
                required_fix=critique.required_fix,
                suggested_experiment=critique.suggested_experiment,
            ))

        # Sort: severity desc, role-weight desc, reviewer count desc, then
        # description text as a deterministic tie-break so ties never depend on
        # parse/gather ordering. With no role_weights the weight term is uniform.
        merged.sort(
            key=lambda c: (
                -_SEVERITY_ORDER.get(c.severity, 0),
                -self._critique_weight(c.description),
                -_count_reviewers_in_tag(c.description),
                c.description,
            )
        )

        return ReviewResult(
            overall_grade=_compute_grade(merged),
            critiques=merged,
        )

    def compute_consensus_report(self, merged: ReviewResult) -> dict:
        """Summarise agreement across the panel for the merged critique list.

        Returns counts used for reporting and for the (optional) advancement
        gate: how many merged critiques were raised by a single reviewer versus
        by multiple reviewers, how many were severity-escalated by agreement,
        the per-severity histogram, and the number of unresolved CRITICAL items.
        """
        single, agreed = 0, 0
        histogram: dict[str, int] = {}
        for c in merged.critiques:
            n = _count_reviewers_in_tag(c.description)
            if n >= self.consensus_threshold:
                agreed += 1
            else:
                single += 1
            histogram[c.severity.value] = histogram.get(c.severity.value, 0) + 1
        return {
            "panel_size": len(self.panel),
            "consensus_threshold": self.consensus_threshold,
            "total_merged": len(merged.critiques),
            "single_reviewer": single,
            "multi_reviewer": agreed,
            "severity_histogram": histogram,
            "unresolved_critical": sum(
                1 for c in merged.critiques if c.severity == SeverityLevel.CRITICAL
            ),
            "grade": merged.overall_grade,
        }

    def _parse_review_output(self, output: str) -> list[ReviewCritique]:
        """Parse reviewer output into structured critiques.

        Handles patterns like:
        - [CRITICAL] description text
        - [HIGH] description text
        - Severity: high, Category: stats, Description: ...
        - **Severity: MEDIUM** -- description
        Also handles markdown-formatted critiques.
        """
        critiques: list[ReviewCritique] = []
        if not output:
            return critiques

        # Pattern 1: [SEVERITY] description
        bracket_pattern = re.compile(
            r'\[(?P<sev>CRITICAL|HIGH|MEDIUM|LOW|INFO)\]\s*(?P<desc>.+)',
            re.IGNORECASE,
        )
        for match in bracket_pattern.finditer(output):
            sev = _parse_severity(match.group("sev"))
            desc = match.group("desc").strip()
            critiques.append(ReviewCritique(
                severity=sev,
                category="general",
                description=desc,
            ))

        # Pattern 2: Severity: X, Category: Y, Description: Z
        structured_pattern = re.compile(
            r'[Ss]everity:\s*(?P<sev>\w+)\s*[,;]\s*'
            r'[Cc]ategory:\s*(?P<cat>[^,;]+)\s*[,;]\s*'
            r'[Dd]escription:\s*(?P<desc>.+)',
            re.IGNORECASE,
        )
        for match in structured_pattern.finditer(output):
            sev = _parse_severity(match.group("sev"))
            cat = match.group("cat").strip()
            desc = match.group("desc").strip()
            critiques.append(ReviewCritique(
                severity=sev,
                category=cat,
                description=desc,
            ))

        # Pattern 3: **Severity: X** -- description (markdown bold)
        markdown_pattern = re.compile(
            r'\*\*[Ss]everity:\s*(?P<sev>\w+)\*\*\s*[-\u2014]+\s*(?P<desc>.+)',
            re.IGNORECASE,
        )
        for match in markdown_pattern.finditer(output):
            sev = _parse_severity(match.group("sev"))
            desc = match.group("desc").strip()
            critiques.append(ReviewCritique(
                severity=sev,
                category="general",
                description=desc,
            ))

        return critiques

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
        """Simple word-overlap (Jaccard) similarity for deduplication."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


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
