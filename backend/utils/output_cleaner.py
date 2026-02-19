"""Strip LLM thinking traces from adapter output.

Models often emit narration like "I will read the file..." or "Let me search
for..." before producing the actual artifact content.  This module removes
those traces so downstream consumers (artifact parser, feedback formatter,
consensus merger) receive clean content.
"""
from __future__ import annotations

import re

# Patterns that match common LLM thinking/narration lines.
# Each pattern is anchored to the start of a line (after optional whitespace).
_TRACE_PATTERNS: list[re.Pattern[str]] = [
    # "I will ...", "I need to ...", "I should ...", "I am going to ..."
    re.compile(
        r"^\s*I\s+(?:will|need to|should|am going to)\s+.+$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Contractions: "I'll ...", "I'm going to ..."
    re.compile(
        r"^\s*I(?:'ll|'m going to)\s+.+$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Let me ..."
    re.compile(
        r"^\s*Let me\s+.+$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Now I will ..." / "First, I will ..." / "Next, I'll ..."
    re.compile(
        r"^\s*(?:Now|First|Next|Then|Finally)[,;]?\s+I\s+(?:will|need to|should|am going to)\s+.+$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:Now|First|Next|Then|Finally)[,;]?\s+I(?:'ll|'m going to)\s+.+$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "The following is ..." (meta-narration about what follows)
    re.compile(
        r"^\s*The following is\s+.+$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Lines that look like pure tool-call narration (very short, no substance)
_TOOL_NARRATION_RE = re.compile(
    r"^\s*I(?:\s+will|'ll)\s+"
    r"(?:read|list|search|check|verify|look|open|examine|inspect|overwrite|update|create|write)\s+"
    r"(?:the|this|for|a|an)\s+.+$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_thinking_traces(text: str) -> str:
    """Remove LLM thinking/narration lines from raw output.

    Preserves:
    - Markdown headers (lines starting with #)
    - Code blocks (fenced with ```)
    - Substantive analysis text
    - Blank lines that separate real content sections

    Returns the cleaned text with collapsed excessive blank lines.
    """
    if not text:
        return text

    lines = text.split("\n")
    cleaned: list[str] = []
    in_code_block = False

    for line in lines:
        # Track code block state - never strip inside code blocks
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned.append(line)
            continue

        if in_code_block:
            cleaned.append(line)
            continue

        # Check if this line matches any trace pattern
        is_trace = False
        for pattern in _TRACE_PATTERNS:
            if pattern.match(line):
                is_trace = True
                break

        if not is_trace and _TOOL_NARRATION_RE.match(line):
            is_trace = True

        if not is_trace:
            cleaned.append(line)

    result = "\n".join(cleaned)

    # Collapse runs of 3+ blank lines into 2
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    # Strip leading blank lines
    result = result.lstrip("\n")

    return result
