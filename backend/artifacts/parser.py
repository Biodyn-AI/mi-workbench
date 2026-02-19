"""Parse adapter output into separate artifact files."""
from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ARTIFACT_TYPES = ["MECH", "EVAL", "XP", "METHOD", "KNOWLEDGE", "RUN_SUMMARY", "REVIEW"]

# Map section titles to artifact filenames
_SECTION_TITLE_MAP: dict[str, str] = {
    "mechanistic analysis": "MECH.md",
    "mechanistic": "MECH.md",
    "mech": "MECH.md",
    "evaluation": "EVAL.md",
    "evaluation report": "EVAL.md",
    "review": "EVAL.md",
    "experiment plan": "XP.md",
    "experiment": "XP.md",
    "follow-up": "XP.md",
    "follow-ups": "XP.md",
    "proposed follow-ups": "XP.md",
    "methodology": "METHOD.md",
    "methods": "METHOD.md",
    "knowledge": "KNOWLEDGE.md",
    "knowledge base": "KNOWLEDGE.md",
    "run summary": "RUN_SUMMARY.md",
    "summary": "RUN_SUMMARY.md",
    "adversarial review": "REVIEW.md",
    "critique": "REVIEW.md",
}

ROLE_TO_ARTIFACT: dict[str, str] = {
    "executor": "MECH.md",
    "reviewer": "EVAL.md",
    "adversarial_reviewer": "REVIEW.md",
    "adversarial": "REVIEW.md",
    "idea_generator": "XP.md",
    "knowledge_extractor": "KNOWLEDGE.md",
}

# Regex for explicit markers like "# MECH.md" or "## EVAL.md"
_EXPLICIT_MARKER_RE = re.compile(
    r"^(#{1,3})\s+(" + "|".join(ARTIFACT_TYPES) + r")\.md\b",
    re.MULTILINE | re.IGNORECASE,
)


class ArtifactParser:
    """Parse raw adapter output into separate artifact files."""

    def parse_output(self, output: str, role: str = "") -> dict[str, str]:
        """Parse adapter output into {filename: content} dict.

        Detection strategies (in order):
        1. Explicit markers: "# MECH.md" or "## MECH.md" at start of section
        2. YAML frontmatter with artifact_type field
        3. Section headers matching artifact names
        4. Role-based defaults
        """
        if not output or not output.strip():
            return {}

        # Strategy 1: explicit markers
        result = self._detect_explicit_markers(output)
        if result:
            return result

        # Strategy 2: YAML frontmatter
        result = self._detect_yaml_frontmatter(output)
        if result:
            return result

        # Strategy 3: section headers
        result = self._detect_section_headers(output)
        if result:
            return result

        # Strategy 4: role-based default
        return self._assign_by_role(output, role)

    def _detect_explicit_markers(self, output: str) -> dict[str, str]:
        """Find sections starting with '# ARTIFACT_TYPE.md'."""
        matches = list(_EXPLICIT_MARKER_RE.finditer(output))
        if not matches:
            return {}

        result: dict[str, str] = {}
        for i, match in enumerate(matches):
            artifact_name = match.group(2).upper() + ".md"
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
            content = output[start:end].strip()
            if content:
                result[artifact_name] = content

        return result

    def _detect_yaml_frontmatter(self, output: str) -> dict[str, str]:
        """Detect YAML frontmatter with artifact_type field."""
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", output, re.DOTALL)
        if not fm_match:
            return {}

        frontmatter = fm_match.group(1)
        type_match = re.search(r"artifact_type:\s*(\S+)", frontmatter)
        if not type_match:
            return {}

        atype = type_match.group(1).strip().upper()
        if atype in ARTIFACT_TYPES:
            body = output[fm_match.end():].strip()
            if body:
                return {f"{atype}.md": body}

        return {}

    def _detect_section_headers(self, output: str) -> dict[str, str]:
        """Map common section titles to artifact types."""
        header_re = re.compile(r"^(#{1,3})\s+(.+?)(?:\s*[-—]\s*.+)?$", re.MULTILINE)
        matches = list(header_re.finditer(output))
        if not matches:
            return {}

        # Try to find mappable sections
        sections: list[tuple[str, int]] = []  # (artifact_name, start_pos)
        for match in matches:
            title = match.group(2).strip().lower()
            # Strip trailing " — ..." or " - ..." that may follow
            title = re.sub(r"\s*[-—].*$", "", title).strip()
            if title in _SECTION_TITLE_MAP:
                sections.append((_SECTION_TITLE_MAP[title], match.start()))

        if not sections:
            return {}

        result: dict[str, str] = {}
        for i, (artifact_name, start) in enumerate(sections):
            end = sections[i + 1][1] if i + 1 < len(sections) else len(output)
            content = output[start:end].strip()
            if content:
                # If same artifact name already seen, append
                if artifact_name in result:
                    result[artifact_name] += "\n\n" + content
                else:
                    result[artifact_name] = content

        return result

    def _assign_by_role(self, output: str, role: str) -> dict[str, str]:
        """Default assignment based on role name."""
        if not role:
            return {"output.md": output.strip()}

        role_lower = role.lower()
        artifact_name = ROLE_TO_ARTIFACT.get(role_lower, f"{role_lower}_output.md")
        return {artifact_name: output.strip()}
