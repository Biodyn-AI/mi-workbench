"""Parse reviewer output into structured feedback for executors."""
from __future__ import annotations

import re
import logging
from typing import Optional

from backend.models import ReviewCritique, ReviewResult, SeverityLevel

logger = logging.getLogger(__name__)

# Known artifact names for reference inference
_KNOWN_ARTIFACTS = ["MECH.md", "PROTOCOL.md", "EVAL.md", "XP.md", "METHODS_COMPARISON.md"]
_ARTIFACT_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _KNOWN_ARTIFACTS) + r")\b",
    re.IGNORECASE,
)
# Section reference patterns: "Section 2.1", "§3", "section 4.2 Controls"
_SECTION_RE = re.compile(
    r"(?:(?:Section|§)\s*[\d]+(?:\.[\d]+)*(?:\s+[A-Z][a-zA-Z &]+)?)",
    re.IGNORECASE,
)

# Keywords for severity detection
_SEVERITY_KEYWORDS: dict[SeverityLevel, list[str]] = {
    SeverityLevel.CRITICAL: ["must fix", "blocking", "invalid", "fatal", "critical"],
    SeverityLevel.HIGH: ["should fix", "significant", "major", "high", "important"],
    SeverityLevel.MEDIUM: ["consider", "moderate", "suggest", "medium"],
    SeverityLevel.LOW: ["minor", "nit", "optional", "low", "trivial", "nitpick"],
}

# Bracket format: [CRITICAL] description
_BRACKET_RE = re.compile(
    r"\[?(CRITICAL|HIGH|MEDIUM|LOW|INFO)\]?\s*[-—:.]?\s*(.+)",
    re.IGNORECASE,
)

# Markdown format: **Severity: X** — description
_MARKDOWN_SEVERITY_RE = re.compile(
    r"\*\*Severity:\s*(CRITICAL|HIGH|MEDIUM|LOW|INFO)\*\*\s*[-—:.]?\s*(.+)",
    re.IGNORECASE,
)

# Structured format: - Severity: X, Category: Y, Description: Z
_STRUCTURED_RE = re.compile(
    r"[-*]\s*Severity:\s*(CRITICAL|HIGH|MEDIUM|LOW|INFO)\s*[,;]\s*"
    r"Category:\s*([^,;]+?)\s*[,;]\s*Description:\s*(.+)",
    re.IGNORECASE,
)

# Grade patterns
_GRADE_RE = re.compile(
    r"(?:Overall\s+)?(?:Grade|Score|Rating)\s*[:=]\s*([A-F][+-]?|pass|fail)",
    re.IGNORECASE,
)


def _parse_severity(text: str) -> SeverityLevel:
    """Infer severity from a string by keyword matching."""
    lower = text.lower()
    # Check explicit level names first
    for level in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW]:
        if level.value in lower:
            return level
    # Then check keyword lists
    for level, keywords in _SEVERITY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return level
    return SeverityLevel.MEDIUM


def _str_to_severity(s: str) -> SeverityLevel:
    """Convert a severity string to enum."""
    try:
        return SeverityLevel(s.lower())
    except ValueError:
        return SeverityLevel.MEDIUM


class FeedbackFormatter:
    """Parse reviewer output and format as actionable feedback for executors."""

    def parse_review(self, output: str) -> ReviewResult:
        """Parse reviewer output into structured ReviewResult.

        Detects critiques in multiple formats:
        - "[SEVERITY] description" (bracket format)
        - "**Severity: X** -- description" (markdown format)
        - "- Severity: X, Category: Y, Description: Z" (structured format)
        - Bulleted lists under "Concerns" / "Issues" / "Required Fixes" headers
        """
        result = ReviewResult()

        # Extract grade
        grade_match = _GRADE_RE.search(output)
        if grade_match:
            result.overall_grade = grade_match.group(1).upper()

        # Extract reproducibility gaps
        result.reproducibility_gaps = self._extract_section_items(
            output, r"(?:Reproducibility|Repro)\s*(?:Gaps?|Check|Issues?)"
        )

        # Extract suspected confounders
        result.suspected_confounders = self._extract_section_items(
            output, r"(?:Suspected\s+)?Confounders?"
        )

        # Parse critiques in order of specificity
        critiques: list[ReviewCritique] = []

        # 1. Structured format (most specific)
        for match in _STRUCTURED_RE.finditer(output):
            severity = _str_to_severity(match.group(1))
            category = match.group(2).strip()
            description = match.group(3).strip()
            critiques.append(ReviewCritique(
                severity=severity,
                category=category,
                description=description,
            ))

        # 2. Markdown severity format
        for match in _MARKDOWN_SEVERITY_RE.finditer(output):
            severity = _str_to_severity(match.group(1))
            description = match.group(2).strip()
            # Avoid duplicates from structured matches
            if not any(c.description == description for c in critiques):
                critiques.append(ReviewCritique(
                    severity=severity,
                    category="general",
                    description=description,
                ))

        # 3. Bracket format
        for match in _BRACKET_RE.finditer(output):
            severity = _str_to_severity(match.group(1))
            description = match.group(2).strip()
            if not any(c.description == description for c in critiques):
                critiques.append(ReviewCritique(
                    severity=severity,
                    category="general",
                    description=description,
                ))

        # 4. Bulleted items under concern/issue headers
        bulleted = self._extract_bulleted_concerns(output)
        for item in bulleted:
            if not any(c.description == item for c in critiques):
                severity = _parse_severity(item)
                critiques.append(ReviewCritique(
                    severity=severity,
                    category="general",
                    description=item,
                ))

        # Infer artifact/section references from critique text
        result.critiques = [self._infer_refs(c) for c in critiques]
        return result

    def _extract_section_items(self, output: str, header_pattern: str) -> list[str]:
        """Extract bulleted items under a section header."""
        pattern = re.compile(
            r"#{1,3}\s+" + header_pattern + r"\s*\n((?:[-*]\s+.+\n?)+)",
            re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(output)
        if not match:
            return []
        items = []
        for line in match.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith(("-", "*")):
                items.append(re.sub(r"^[-*]\s+", "", line).strip())
        return items

    def _extract_bulleted_concerns(self, output: str) -> list[str]:
        """Extract items under Concerns/Issues/Required Fixes headers."""
        header_re = re.compile(
            r"#{1,3}\s+(?:Concerns?|Issues?|Required\s+Fixes?|Problems?|Weaknesses?)\s*\n",
            re.IGNORECASE,
        )
        results: list[str] = []
        for match in header_re.finditer(output):
            pos = match.end()
            # Collect bulleted lines after the header
            remaining = output[pos:]
            for line in remaining.split("\n"):
                stripped = line.strip()
                if stripped.startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    item = re.sub(r"^[-*\d.]+\s+", "", stripped).strip()
                    if item:
                        results.append(item)
                elif stripped.startswith("#") or (stripped == "" and results):
                    break
        return results

    @staticmethod
    def _infer_refs(critique: ReviewCritique) -> ReviewCritique:
        """Infer artifact_ref and section_ref from critique description text."""
        text = critique.description
        if critique.required_fix:
            text = text + " " + critique.required_fix

        if not critique.artifact_ref:
            match = _ARTIFACT_RE.search(text)
            if match:
                # Normalize to uppercase filename
                name = match.group(1)
                for known in _KNOWN_ARTIFACTS:
                    if name.lower() == known.lower():
                        critique.artifact_ref = known
                        break

        if not critique.section_ref:
            match = _SECTION_RE.search(text)
            if match:
                critique.section_ref = match.group(0).strip()

        return critique

    def format_for_executor(self, review: ReviewResult) -> str:
        """Format structured review as actionable feedback for the executor."""
        lines: list[str] = []
        grade_str = f" (Grade: {review.overall_grade})" if review.overall_grade else ""
        lines.append(f"=== REVIEWER FEEDBACK{grade_str} ===")
        lines.append("")

        # Split critiques into required (critical/high) and suggested (medium/low/info)
        required = [c for c in review.critiques if c.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)]
        suggested = [c for c in review.critiques if c.severity not in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)]

        counter = 1
        if required:
            lines.append("REQUIRED FIXES (must address):")
            for c in required:
                tag = c.severity.value.upper()
                loc = self._format_location(c)
                lines.append(f"{counter}. [{tag}{loc}] {c.description}")
                counter += 1
            lines.append("")

        if suggested:
            lines.append("SUGGESTED IMPROVEMENTS:")
            for c in suggested:
                tag = c.severity.value.upper()
                loc = self._format_location(c)
                lines.append(f"{counter}. [{tag}{loc}] {c.description}")
                counter += 1
            lines.append("")

        if review.reproducibility_gaps:
            lines.append("REPRODUCIBILITY GAPS:")
            for gap in review.reproducibility_gaps:
                lines.append(f"- {gap}")
            lines.append("")

        if review.suspected_confounders:
            lines.append("SUSPECTED CONFOUNDERS:")
            for conf in review.suspected_confounders:
                lines.append(f"- {conf}")
            lines.append("")

        lines.append("=== END FEEDBACK ===")
        return "\n".join(lines)

    @staticmethod
    def _format_location(critique: ReviewCritique) -> str:
        """Build a location tag like ' | PROTOCOL.md § Section 2.1' for a critique."""
        parts: list[str] = []
        if critique.category and critique.category != "general":
            parts.append(critique.category)
        if critique.artifact_ref:
            ref = critique.artifact_ref
            if critique.section_ref:
                ref += f" § {critique.section_ref}"
            parts.append(ref)
        if parts:
            return " | " + " | ".join(parts)
        return ""

    def extract_action_items(self, review: ReviewResult) -> list[str]:
        """Extract a simple checklist of action items from the review."""
        items: list[str] = []
        for c in review.critiques:
            if c.required_fix:
                items.append(c.required_fix)
            else:
                items.append(c.description)
        return items
