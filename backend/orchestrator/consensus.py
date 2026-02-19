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


class ConsensusReviewer:
    """Runs multiple reviewer roles in parallel and merges results."""

    def __init__(self, similarity_threshold: float = _SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    async def run_consensus(
        self,
        artifacts_content: str,
        adapter: BaseAdapter,
        workspace_context: WorkspaceContext,
    ) -> ReviewResult:
        """Run reviewer, adversarial reviewer, and bio plausibility checker in parallel.
        Merge their outputs into a single ranked critique list."""
        tasks = []
        for role_name, prompt_ref in _REVIEWER_ROLES:
            request = AdapterRunRequest(
                prompt_bundle=PromptBundle(
                    system_prompt=f"You are the {role_name}. Review the following artifacts.",
                    user_prompt=artifacts_content,
                ),
                workspace_context=workspace_context,
            )
            tasks.append(adapter.run(request))

        results: list[AdapterRunResult] = await asyncio.gather(*tasks)
        return self.merge_critiques(results)

    def merge_critiques(self, results: list[AdapterRunResult]) -> ReviewResult:
        """Merge multiple review results into consensus.

        - Deduplicate similar critiques (fuzzy match on description)
        - Escalate severity if multiple reviewers flag same issue
        - Rank by: severity (desc), then number of reviewers who flagged it
        - Tag each critique with which reviewer(s) raised it
        """
        # Parse all critiques from each reviewer
        all_tagged: list[tuple[ReviewCritique, str]] = []
        role_names = [r[0] for r in _REVIEWER_ROLES]
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
            # Escalate if multiple reviewers flagged the same issue
            reviewer_names = sorted(set(r for _, r in group))
            if len(reviewer_names) >= 2:
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

        # Sort: severity descending, then number of reviewers descending
        merged.sort(
            key=lambda c: (
                -_SEVERITY_ORDER.get(c.severity, 0),
                -_count_reviewers_in_tag(c.description),
            )
        )

        return ReviewResult(
            overall_grade=_compute_grade(merged),
            critiques=merged,
        )

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
