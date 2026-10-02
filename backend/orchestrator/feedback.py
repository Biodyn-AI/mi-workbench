"""Parse reviewer output into structured feedback for executors.

Reviewer-lens prompts end with a required machine-readable block (the critique
contract)::

    ```json
    {"critiques": [{"severity": "critical|high|medium|low|info",
                    "category": "<short>",
                    "description": "<the specific problem>",
                    "required_fix": "<concrete fix>"}],
     "overall_assessment": "<one sentence>"}
    ```

Parsing order (first method that yields critiques wins, and every critique is
parsed exactly once):

1. ``json``  - the fenced JSON block (tolerates trailing commas, comments,
   single quotes / Python literals, a missing closing fence, truncation, and
   as a last resort salvages individual critique objects);
2. ``yaml``  - YAML-like ``- severity: X`` / ``description: Y`` blocks;
3. ``regex`` - the legacy bracket / markdown / structured line formats.

The module-level helpers (:func:`extract_json_critiques`,
:func:`parse_yaml_critiques`, :func:`extract_json_payloads`,
:func:`lenient_json_loads`) are shared with the consensus merger.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.models import ReviewCritique, ReviewResult, SeverityLevel
from backend.orchestrator.convergence import grade_score

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

# Keywords for severity detection (matched on word boundaries)
_SEVERITY_KEYWORDS: dict[SeverityLevel, list[str]] = {
    SeverityLevel.CRITICAL: ["must fix", "blocking", "invalid", "fatal", "critical"],
    SeverityLevel.HIGH: ["should fix", "significant", "major", "high", "important"],
    SeverityLevel.MEDIUM: ["consider", "moderate", "suggest", "medium"],
    SeverityLevel.LOW: ["minor", "nit", "optional", "low", "trivial", "nitpick"],
}

_SEV_ALT = r"(CRITICAL|HIGH|MEDIUM|LOW|INFO)"

# Bracket format: [CRITICAL] description (brackets are required).
_BRACKET_RE = re.compile(
    r"\[[ \t]*" + _SEV_ALT + r"[ \t]*\][ \t]*[-—–:.]?[ \t]*(.+)",
    re.IGNORECASE,
)

# Line-initial bare severity token followed by an explicit separator:
#   "HIGH: text", "- **Critical** — text", "2. low - text".
# The token must start the line (after an optional bullet/number and bold
# markers), end on a word boundary, and be followed by ':' / an em or en dash
# / ' -', so prose such as "highly significant" or "lower layers" never
# matches.
_LINE_SEVERITY_RE = re.compile(
    r"^[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?(?:\*\*|__)?" + _SEV_ALT
    + r"\b(?:\*\*|__)?[ \t]*(?::|—|–|[ \t]-)(?:\*\*|__)?[ \t]*(\S.*)$",
    re.IGNORECASE | re.MULTILINE,
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

# Explicit grade in legacy (non-contract) reviews, e.g. "Overall Grade: B+".
# Only the keyword is case-insensitive: the grade token must be an UPPERCASE
# letter A-D or F (optionally +/-) or pass/fail, so prose such as
# "attention score: a mean over heads" or "score: e.g. AUROC" is not a grade.
# The keyword must not be glued to a preceding identifier ("adversarial_grade"),
# the token must end on a boundary ("compromised" is not grade "C"), and it
# must not be followed by a lowercase word ("score: B cells" is not grade B).
_GRADE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?i:(?:Overall[ \t_]+)?(?:Grade|Score|Rating))(?:\*\*|__)?[ \t]*[:=]"
    r"[ \t]*(?:\*\*|__)?[ \t]*([A-DF][+-]?|(?i:pass|fail))(?![A-Za-z0-9_])(?![ \t]+[a-z])",
)


def _parse_severity(text: str) -> SeverityLevel:
    """Infer severity from free text by keyword matching on word boundaries.

    Word boundaries matter: "workflow" must not read as LOW and "highly" must
    not read as HIGH.
    """
    lower = text.lower()
    # Check explicit level names first
    for level in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW]:
        if re.search(r"\b" + re.escape(level.value) + r"\b", lower):
            return level
    # Then check keyword lists
    for level, keywords in _SEVERITY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                return level
    return SeverityLevel.MEDIUM


def _str_to_severity(s: str) -> SeverityLevel:
    """Convert a severity string to enum."""
    try:
        return SeverityLevel(s.lower())
    except ValueError:
        return SeverityLevel.MEDIUM


# ── Severity normalisation (JSON / YAML contract) ─────────────────────

_SEVERITY_SYNONYMS: dict[str, SeverityLevel] = {
    "critical": SeverityLevel.CRITICAL, "crit": SeverityLevel.CRITICAL,
    "blocker": SeverityLevel.CRITICAL, "blocking": SeverityLevel.CRITICAL,
    "fatal": SeverityLevel.CRITICAL, "severe": SeverityLevel.CRITICAL,
    "high": SeverityLevel.HIGH, "major": SeverityLevel.HIGH,
    "medium": SeverityLevel.MEDIUM, "med": SeverityLevel.MEDIUM,
    "moderate": SeverityLevel.MEDIUM,
    "low": SeverityLevel.LOW, "minor": SeverityLevel.LOW, "nit": SeverityLevel.LOW,
    "trivial": SeverityLevel.LOW,
    "info": SeverityLevel.INFO, "informational": SeverityLevel.INFO,
    "information": SeverityLevel.INFO, "note": SeverityLevel.INFO,
    "observation": SeverityLevel.INFO,
}


def normalize_severity(raw: Any) -> Optional[SeverityLevel]:
    """Map a severity value (any case, synonyms allowed) to a level, or None.

    A value that lists several different levels as alternatives (an echoed
    template placeholder such as ``"critical|high|medium|low|info"``, or
    ``"high/medium"``) is ambiguous and returns None, so callers fall back to
    their default (MEDIUM) instead of silently taking the first, most severe
    alternative. A single level with a qualifier (``"low (cosmetic)"``) still
    maps to that level.
    """
    if raw is None:
        return None
    if isinstance(raw, SeverityLevel):
        return raw
    text = str(raw).strip().strip("*_`\"'[]().:;,").strip().lower()
    if not text:
        return None
    if text in _SEVERITY_SYNONYMS:
        return _SEVERITY_SYNONYMS[text]
    tokens = [t for t in re.split(r"[\s/|,;:()\-]+", text) if t]
    levels = {_SEVERITY_SYNONYMS[t] for t in tokens if t in _SEVERITY_SYNONYMS}
    if len(levels) >= 3 or (len(levels) >= 2 and re.search(r"[|/,]", text)):
        return None
    first = tokens[0] if tokens else ""
    return _SEVERITY_SYNONYMS.get(first)


_NO_ISSUE_RE = re.compile(
    r"^\s*(?:no\s+(?:issues?|problems?|concerns?|vulnerabilit(?:y|ies)|findings?)"
    r"(?:\s+(?:detected|found|identified))?|none(?:\s+found)?|n/?a|"
    r"nothing\s+(?:to\s+report|found))\s*[.!]?\s*$",
    re.IGNORECASE,
)

# A description that is just a template placeholder ("<one or two sentences ...>"),
# i.e. the model echoed the contract example.
_PLACEHOLDER_RE = re.compile(r"^\s*<[^<>]*>\s*$")

# Contract field -> accepted keys, in PRIORITY order: the contract key comes
# first and wins whenever it is present with a value, so a model that adds
# e.g. a short "title" before the real "description" keeps its description.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "severity": ("severity", "level", "priority", "sev"),
    "category": ("category", "type", "area", "dimension", "threat", "topic", "lens"),
    "description": ("description", "issue", "problem", "finding", "critique",
                    "concern", "text", "details", "detail", "explanation",
                    "rationale", "summary", "title"),
    "required_fix": ("required_fix", "fix", "requiredfix", "recommendation",
                     "suggested_fix", "remedy", "action", "required_action"),
    "suggested_experiment": ("suggested_experiment", "experiment",
                             "suggested_experiments"),
}
_ALIAS_TO_FIELD: dict[str, str] = {
    alias: fld for fld, aliases in _FIELD_ALIASES.items() for alias in aliases
}
_ALIAS_PRIORITY: dict[str, int] = {
    alias: i for aliases in _FIELD_ALIASES.values() for i, alias in enumerate(aliases)
}


def _norm_key(key: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(_as_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_as_text(v)}" for k, v in value.items())
    return " ".join(str(value).split())


#: Reasons an item of a critique list yields no critique.
DROP_NO_ISSUE = "no_issue"            # an explicit "no issues found" entry (benign)
DROP_PLACEHOLDER = "placeholder"      # echoed template text such as "<the problem>"
DROP_NO_DESCRIPTION = "no_description"  # no recognised description key / empty
DROP_NOT_MAPPING = "not_mapping"      # neither an object nor a string


def _resolve_fields(item: dict) -> dict[str, Any]:
    """Map an item's keys to contract fields by alias PRIORITY (not dict order)."""
    norm: dict[str, Any] = {}
    for key, value in item.items():
        nk = _norm_key(key)
        if nk not in norm:
            norm[nk] = value
    fields: dict[str, Any] = {}
    for fld, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in norm and _as_text(norm[alias]):
                fields[fld] = norm[alias]
                break
    return fields


def classify_critique_item(item: Any) -> tuple[Optional[ReviewCritique], str]:
    """Build a critique from a decoded JSON/YAML item and say why if none.

    Returns ``(critique, "ok")`` or ``(None, reason)`` with reason one of
    :data:`DROP_NO_ISSUE` (a benign "no issues found" entry),
    :data:`DROP_PLACEHOLDER`, :data:`DROP_NO_DESCRIPTION` or
    :data:`DROP_NOT_MAPPING`.
    """
    if isinstance(item, str):
        desc = _as_text(item)
        if not desc:
            return None, DROP_NO_DESCRIPTION
        if _NO_ISSUE_RE.match(desc):
            return None, DROP_NO_ISSUE
        if _PLACEHOLDER_RE.match(desc):
            return None, DROP_PLACEHOLDER
        return ReviewCritique(severity=SeverityLevel.MEDIUM, category="general",
                              description=desc), "ok"
    if not isinstance(item, dict):
        return None, DROP_NOT_MAPPING
    fields = _resolve_fields(item)
    desc = _as_text(fields.get("description"))
    if not desc:
        return None, DROP_NO_DESCRIPTION
    if _NO_ISSUE_RE.match(desc):
        return None, DROP_NO_ISSUE
    if _PLACEHOLDER_RE.match(desc):
        return None, DROP_PLACEHOLDER
    severity = normalize_severity(fields.get("severity")) or SeverityLevel.MEDIUM
    category = _as_text(fields.get("category")) or "general"
    fix = _as_text(fields.get("required_fix")) or None
    exp = _as_text(fields.get("suggested_experiment")) or None
    return ReviewCritique(
        severity=severity,
        category=category,
        description=desc,
        required_fix=fix,
        suggested_experiment=exp,
    ), "ok"


def critique_from_mapping(item: Any) -> Optional[ReviewCritique]:
    """Build a :class:`ReviewCritique` from a decoded JSON/YAML item.

    Accepts the contract keys plus common aliases (``issue``, ``finding``,
    ``details``, ``fix``, ...); when several keys map to one field the
    contract key wins (alias priority, not key order). Missing or ambiguous
    severities default to MEDIUM. Items with no description, a "no issue
    detected" entry or an echoed ``<...>`` template placeholder are skipped
    (see :func:`classify_critique_item` for the reason).
    """
    return classify_critique_item(item)[0]


# ── Lenient JSON ──────────────────────────────────────────────────────


def _scan_outside_strings(s: str):
    """Yield (index, char, in_string) while tracking JSON/Python string state."""
    in_str: Optional[str] = None
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            yield i, ch, True
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            yield i, ch, True
            continue
        yield i, ch, False


def _strip_json_comments(s: str) -> str:
    out: list[str] = []
    i, n = 0, len(s)
    in_str: Optional[str] = None
    esc = False
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and s[i + 1] == "*":
            end = s.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch == "#" and (not out or out[-1] in "\n\t ,{["):
            # Python/YAML-style comment outside strings
            while i < n and s[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _remove_trailing_commas(s: str) -> str:
    chars = list(s)
    for i, ch, in_string in _scan_outside_strings(s):
        if in_string or ch != ",":
            continue
        j = i + 1
        while j < len(s) and s[j] in " \t\r\n":
            j += 1
        if j >= len(s) or s[j] in "}]":
            chars[i] = ""
    return "".join(chars)


def _pythonize_literals(s: str) -> str:
    """Replace bare JSON literals (true/false/null) outside strings."""
    out: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            chunk = "".join(buf)
            chunk = re.sub(r"\btrue\b", "True", chunk)
            chunk = re.sub(r"\bfalse\b", "False", chunk)
            chunk = re.sub(r"\bnull\b", "None", chunk)
            out.append(chunk)
            buf.clear()

    for _, ch, in_string in _scan_outside_strings(s):
        if in_string:
            flush()
            out.append(ch)
        else:
            buf.append(ch)
    flush()
    return "".join(out)


def _top_level_value(s: str) -> tuple[str, bool, list[str], Optional[str], list[tuple[int, list[str]]]]:
    """Cut ``s`` to its first top-level JSON value.

    Returns (segment, complete, open_stack, open_string_quote, safe_points)
    where ``safe_points`` are (end_index, stack) positions right after a
    nested value closed (used to repair truncated output).
    """
    start = None
    for idx, ch in enumerate(s):
        if ch in "{[":
            start = idx
            break
    if start is None:
        return "", False, [], None, []
    seg = s[start:]
    stack: list[str] = []
    in_str: Optional[str] = None
    esc = False
    safe: list[tuple[int, list[str]]] = []
    for i, ch in enumerate(seg):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return seg[: i + 1], True, [], None, safe
            safe.append((i + 1, list(stack)))
    return seg, False, stack, in_str, safe


def _close(stack: list[str]) -> str:
    return "".join("}" if ch == "{" else "]" for ch in reversed(stack))


def _try_parse(s: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(s)
    except (ValueError, TypeError, RecursionError):
        pass
    try:
        return True, ast.literal_eval(_pythonize_literals(s))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def lenient_json_loads(text: str) -> tuple[Any, str]:
    """Decode JSON leniently; returns ``(value, detail)`` or raises ValueError.

    ``detail`` is ``"strict"`` (valid JSON), ``"repaired"`` (after removing
    comments / trailing commas / fixing structural curly quotes, or via a
    Python-literal fallback for single quotes), or ``"truncated"`` (unclosed
    strings or brackets were closed, dropping an incomplete trailing element
    if needed).
    """
    s = (text or "").strip().lstrip("﻿")
    if not s:
        raise ValueError("empty JSON text")
    try:
        return json.loads(s), "strict"
    except (ValueError, TypeError, RecursionError):
        pass
    head, head_complete, _, _, _ = _top_level_value(s)
    if head_complete:
        try:
            return json.loads(head), "strict"
        except (ValueError, TypeError, RecursionError):
            pass
    if '"' not in s and ("“" in s or "”" in s):
        s = s.replace("“", '"').replace("”", '"')
    cleaned = _remove_trailing_commas(_strip_json_comments(s))
    segment, complete, stack, open_quote, safe = _top_level_value(cleaned)
    if not segment:
        raise ValueError("no JSON object or array found")
    if complete:
        ok, value = _try_parse(segment)
        if ok:
            return value, "repaired"
        ok, value = _try_parse(_remove_trailing_commas(segment))
        if ok:
            return value, "repaired"
        raise ValueError("malformed JSON")
    # Truncated: close at the end, then fall back to the latest safe points.
    candidates = []
    tail = segment + (open_quote or "")
    tail = re.sub(r"[\s,:]+$", "", tail)
    candidates.append(_remove_trailing_commas(tail + _close(stack)))
    for end, st in reversed(safe[-5:]):
        cut = re.sub(r"[\s,]+$", "", segment[:end])
        candidates.append(_remove_trailing_commas(cut + _close(st)))
    for cand in candidates:
        ok, value = _try_parse(cand)
        if ok:
            return value, "truncated"
    raise ValueError("malformed or truncated JSON")


_FENCE_LINE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([A-Za-z0-9_+\-.]*)[^\n]*$")
_JSONISH_LANGS = {"", "json", "json5", "jsonc", "javascript", "js"}


def _fenced_blocks(text: str) -> list[tuple[int, int, str, str]]:
    """Return fenced code blocks as (start, end, lang, content).

    An unclosed fence runs to the end of the text (missing closing fence).
    A one-line ```json {...}``` block is also recognised.
    """
    blocks: list[tuple[int, int, str, str]] = []
    lines = text.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    i = 0
    while i < len(lines):
        line = lines[i]
        inline = re.match(r"^[ \t]*```[ \t]*([A-Za-z0-9_+\-.]*)[ \t]*(\{.*\}|\[.*\])[ \t]*```[ \t]*$", line)
        if inline:
            blocks.append((offsets[i], offsets[i] + len(line), inline.group(1).lower(), inline.group(2)))
            i += 1
            continue
        m = _FENCE_LINE_RE.match(line)
        if not m:
            i += 1
            continue
        fence, lang = m.group(1), m.group(2).lower()
        rest_of_line = line[m.end(2):].strip() if m.group(2) else line.strip().lstrip("`~").strip()
        body_lines: list[str] = []
        if rest_of_line.startswith(("{", "[")):
            body_lines.append(rest_of_line)
        j = i + 1
        closed = False
        while j < len(lines):
            stripped = lines[j].strip()
            if stripped.startswith(fence[0] * 3) and stripped.strip(fence[0]) == "":
                closed = True
                break
            if stripped.endswith("```") and stripped.rstrip("`").rstrip().endswith(("}", "]")):
                body_lines.append(lines[j].rstrip().rstrip("`"))
                closed = True
                break
            body_lines.append(lines[j])
            j += 1
        end = offsets[j] + len(lines[j]) if j < len(lines) else len(text)
        blocks.append((offsets[i], end, lang, "\n".join(body_lines)))
        i = j + 1 if closed else len(lines)
    return blocks


_BARE_KEY_RE = re.compile(r"""["'](critiques|groups)["']\s*:""", re.IGNORECASE)
_BARE_LIST_RE = re.compile(r"""\[\s*\{\s*["'](?:severity|description|issue)["']""", re.IGNORECASE)


def _enclosing_open_brace(text: str, pos: int) -> Optional[int]:
    depth = 0
    for k in range(pos - 1, -1, -1):
        ch = text[k]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth == 0:
                return k
            depth -= 1
    return None


def extract_json_payloads(text: str) -> list[tuple[Any, str]]:
    """Find and leniently decode JSON payloads in free text (document order).

    Sources: fenced code blocks (json/untagged/js, or any block whose body
    starts with ``{``/``[``), then bare objects that contain a ``critiques``
    or ``groups`` key, then a bare list of critique objects. Returns
    ``(value, detail)`` pairs; undecodable candidates are skipped.
    """
    if not text:
        return []
    found: list[tuple[int, Any, str]] = []
    spans: list[tuple[int, int]] = []
    for start, end, lang, body in _fenced_blocks(text):
        stripped = body.strip()
        if not stripped:
            continue
        if lang not in _JSONISH_LANGS and not stripped.startswith(("{", "[")):
            continue
        if "{" not in stripped and "[" not in stripped:
            continue
        spans.append((start, end))
        try:
            value, detail = lenient_json_loads(stripped)
        except ValueError:
            continue
        found.append((start, value, detail))

    def inside_fence(p: int) -> bool:
        return any(a <= p < b for a, b in spans)

    seen_starts: set[int] = set()
    for m in _BARE_KEY_RE.finditer(text):
        if inside_fence(m.start()):
            continue
        brace = _enclosing_open_brace(text, m.start())
        if brace is None or brace in seen_starts or inside_fence(brace):
            continue
        seen_starts.add(brace)
        try:
            value, detail = lenient_json_loads(text[brace:])
        except ValueError:
            continue
        found.append((brace, value, detail))
    if not found:
        for m in _BARE_LIST_RE.finditer(text):
            if inside_fence(m.start()) or m.start() in seen_starts:
                continue
            seen_starts.add(m.start())
            try:
                value, detail = lenient_json_loads(text[m.start():])
            except ValueError:
                continue
            found.append((m.start(), value, detail))
    found.sort(key=lambda t: t[0])
    return [(v, d) for _, v, d in found]


def _looks_like_critique(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    keys = {_ALIAS_TO_FIELD.get(_norm_key(k)) for k in item}
    return "description" in keys and ("severity" in keys or "required_fix" in keys)


def _find_critique_list(payload: Any) -> Optional[list]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _norm_key(key) == "critiques":
                if isinstance(value, list):
                    return value
                if value in (None, "", {}):
                    return []
        for value in payload.values():
            if isinstance(value, dict):
                inner = _find_critique_list(value)
                if inner is not None:
                    return inner
        return None
    if isinstance(payload, list) and payload and all(_looks_like_critique(x) for x in payload):
        return payload
    return None


def _payload_str(payload: Any, *keys: str) -> str:
    if not isinstance(payload, dict):
        return ""
    wanted = set(keys)
    for key, value in payload.items():
        if _norm_key(key) in wanted and isinstance(value, (str, int, float)):
            return _as_text(value)
    for value in payload.values():
        if isinstance(value, dict):
            inner = _payload_str(value, *keys)
            if inner:
                return inner
    return ""


_OBJ_RE = re.compile(r"\{[^{}]*\}", re.S)
_FIELD_VALUE_RE = r"""["']?{key}["']?\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')"""


# Value up to the quote that is followed by the next key or the object end;
# tolerates unescaped quotes inside the value ("The "76%" metric ...").
_FIELD_VALUE_LOOSE_RE = (
    r"""["']?{key}["']?\s*:\s*(["'])(.*?)\1\s*(?=,\s*["']?[A-Za-z_][A-Za-z_ -]*["']?\s*:|\}})"""
)


def _salvage_field(chunk: str, names: tuple[str, ...]) -> str:
    for name in names:
        loose = re.search(
            _FIELD_VALUE_LOOSE_RE.format(key=re.escape(name)), chunk, re.IGNORECASE | re.S
        )
        if loose:
            return _as_text(loose.group(2).replace('\\"', '"').replace("\\'", "'"))
        m = re.search(_FIELD_VALUE_RE.format(key=re.escape(name)), chunk, re.IGNORECASE | re.S)
        if not m:
            continue
        if m.group(1) is not None:
            try:
                return _as_text(json.loads('"' + m.group(1) + '"', strict=False))
            except ValueError:
                return _as_text(m.group(1).replace('\\"', '"'))
        return _as_text(m.group(2).replace("\\'", "'"))
    return ""


def _salvage_critiques(text: str) -> list[ReviewCritique]:
    """Last-resort per-object field extraction from badly broken JSON."""
    out: list[ReviewCritique] = []
    for m in _OBJ_RE.finditer(text):
        chunk = m.group(0)
        desc = _salvage_field(chunk, _FIELD_ALIASES["description"])
        if not desc:
            continue
        item = {
            "severity": _salvage_field(chunk, _FIELD_ALIASES["severity"]),
            "category": _salvage_field(chunk, _FIELD_ALIASES["category"]),
            "description": desc,
            "required_fix": _salvage_field(chunk, _FIELD_ALIASES["required_fix"]),
        }
        c = critique_from_mapping(item)
        if c is not None:
            out.append(c)
    return out


@dataclass
class ParsedCritiques:
    """Result of parsing one reviewer output.

    For the JSON contract, ``n_raw_items`` is the number of entries in the
    winning ``critiques`` list and ``n_dropped`` the entries that could not
    be read as a critique (no description, an echoed ``<...>`` placeholder,
    a non-object); benign "no issues found" entries are counted in
    ``n_no_issue`` instead. ``n_contract_blocks`` counts the payloads that
    carried a ``critiques`` list (the contract allows exactly one; the last
    one wins) and ``dropped_earlier`` the distinct critiques that appeared
    only in earlier blocks and were therefore not counted.

    ``invalid_reason`` is set (and ``method`` is ``"none"``) when a contract
    block was found but cannot be trusted as a review: every entry was
    unreadable (``"all_items_unreadable"``) or the block was cut off before
    any critique was complete (``"truncated_empty"``). Such an output is NOT
    a clean "no problems found" answer; only a complete ``"critiques": []``
    is.
    """

    critiques: list[ReviewCritique] = field(default_factory=list)
    method: str = "none"          # json | yaml | regex | none
    detail: str = ""              # json: strict | repaired | truncated | salvaged
    overall_assessment: str = ""
    grade: str = ""               # explicit grade inside the JSON payload, if any
    n_raw_items: int = 0
    n_dropped: int = 0
    n_no_issue: int = 0
    n_contract_blocks: int = 0
    dropped_earlier: int = 0
    invalid_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.detail == "truncated"

    @property
    def is_valid_contract(self) -> bool:
        """True unless this is a contract block that cannot count as a review."""
        return not self.invalid_reason


def _clean_assessment(text: str) -> str:
    return "" if _PLACEHOLDER_RE.match(text or "") else text


_CONTRACT_HINT_RE = re.compile(r"""["']critiques["']\s*:""", re.IGNORECASE)


def _critique_key(c: ReviewCritique) -> tuple[str, str]:
    return (c.severity.value, " ".join(c.description.lower().split()))


def extract_json_critiques(text: str) -> Optional[ParsedCritiques]:
    """Parse the JSON critique contract; None if no contract block is present.

    The last payload that carries a ``critiques`` list wins (the contract
    block is the reply's final section; earlier blocks are usually drafts).
    An explicit, complete empty list is a valid "no problems found" answer
    and returns an empty ``ParsedCritiques`` with ``method="json"``.

    A block whose entries were all unreadable (e.g. an echoed template, or
    descriptions under an unknown key), or a truncated block from which no
    critique could be recovered, is returned with ``invalid_reason`` set and
    ``method="none"``: callers must treat it as unparsed, never as a clean
    review. The raw item / drop / block counts are always reported.
    """
    if not text:
        return None
    payloads = extract_json_payloads(text)
    contract: list[tuple[Any, str, list]] = []
    for payload, detail in payloads:
        items = _find_critique_list(payload)
        if items is not None:
            contract.append((payload, detail, items))
    if contract:
        payload, detail, items = contract[-1]
        critiques: list[ReviewCritique] = []
        n_dropped = n_no_issue = 0
        for x in items:
            c, reason = classify_critique_item(x)
            if c is not None:
                critiques.append(c)
            elif reason == DROP_NO_ISSUE:
                n_no_issue += 1
            else:
                n_dropped += 1
        dropped_earlier = 0
        if len(contract) > 1:
            last_keys = {_critique_key(c) for c in critiques}
            earlier: set[tuple[str, str]] = set()
            for _p, _d, its in contract[:-1]:
                for x in its:
                    c = critique_from_mapping(x)
                    if c is not None:
                        earlier.add(_critique_key(c))
            dropped_earlier = len(earlier - last_keys)
        invalid = ""
        if not critiques:
            if detail == "truncated" and n_no_issue == 0:
                invalid = "truncated_empty"
            elif n_dropped > 0:
                invalid = "all_items_unreadable"
        grade = _payload_str(payload, "overall_grade", "grade").upper()
        if not re.fullmatch(r"[A-DF][+-]?|PASS|FAIL", grade or ""):
            grade = ""
        return ParsedCritiques(
            critiques=critiques,
            method="none" if invalid else "json",
            detail=detail,
            overall_assessment=_clean_assessment(
                _payload_str(payload, "overall_assessment", "assessment", "summary")
            ),
            grade="" if invalid else grade,
            n_raw_items=len(items),
            n_dropped=n_dropped,
            n_no_issue=n_no_issue,
            n_contract_blocks=len(contract),
            dropped_earlier=dropped_earlier,
            invalid_reason=invalid,
        )
    if _CONTRACT_HINT_RE.search(text):
        salvaged = _salvage_critiques(text)
        if salvaged:
            return ParsedCritiques(critiques=salvaged, method="json", detail="salvaged",
                                   n_raw_items=len(salvaged), n_contract_blocks=1)
    return None


# ── YAML-like blocks ──────────────────────────────────────────────────

_KV_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<bullet>(?:[-*+]|\d+[.)])[ \t]+)?(?:\*\*|__)?"
    r"(?P<key>[A-Za-z][A-Za-z _\-]{0,40}?)(?:\*\*|__)?[ \t]*:(?:\*\*|__)?[ \t]*(?P<val>.*)$"
)


def _clean_scalar(val: str) -> str:
    v = val.strip()
    v = re.sub(r"^(?:\*\*|__)|(?:\*\*|__)$", "", v).strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v


def parse_yaml_critiques(text: str) -> list[ReviewCritique]:
    """Parse YAML-like critique blocks, e.g.::

        - severity: high
          category: statistics
          description: Effect sizes are missing.
          required_fix: Report Cohen's d with CIs.

    An item starts at a bulleted ``key: value`` line (or when the same key
    repeats; two different aliases of one field, e.g. ``title:`` then
    ``description:``, stay in one item and the higher-priority key wins).
    Continuation lines and ``|`` / ``>`` block scalars are folded into the
    previous value. Only items with a recognised single-word severity and a
    description are returned, so prose such as "Severity: unclear because..."
    or the one-line legacy "Severity: X, Category: Y, ..." format is ignored
    here (the legacy parser handles it).
    """
    if not text:
        return []
    items: list[dict[str, str]] = []
    cur: Optional[dict[str, str]] = None
    cur_raw: set[str] = set()          # normalised raw keys seen in this item
    cur_prio: dict[str, int] = {}      # field -> priority of the alias that set it
    last_key: Optional[str] = None
    item_indent = 0
    # "dict": list-of-mappings style (only the first key is bulleted);
    # "bullets": every field is its own bullet. None = not yet known.
    style: Optional[str] = None

    def flush():
        nonlocal cur, last_key, style
        if cur:
            items.append(cur)
        cur = None
        cur_raw.clear()
        cur_prio.clear()
        last_key = None
        style = None

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            flush()
            continue
        m = _KV_LINE_RE.match(line)
        raw_key = _norm_key(m.group("key")) if m else ""
        fld = _ALIAS_TO_FIELD.get(raw_key) if m else None
        indent = len(line[: len(line) - len(line.lstrip())].expandtabs(4))
        if m and fld:
            val = m.group("val")
            bullet = bool(m.group("bullet"))
            if cur is not None and raw_key not in cur_raw and style is None:
                style = "bullets" if (bullet and indent <= item_indent) else "dict"
            prio = _ALIAS_PRIORITY.get(raw_key, 99)
            starts = (
                cur is None
                or raw_key in cur_raw
                or (bullet and style == "dict" and indent <= item_indent)
                # one-field-per-bullet style: the same field again (not a
                # better alias of it) begins the next item
                or (bullet and style == "bullets" and fld in cur
                    and cur_prio.get(fld, 99) <= prio)
            )
            if starts:
                flush()
                cur = {}
                item_indent = indent
            assert cur is not None
            cur_raw.add(raw_key)
            if fld in cur and cur_prio.get(fld, 99) <= prio:
                # A lower-priority alias of a field already set (e.g. "title"
                # after "description"): keep the existing value.
                last_key = None
                continue
            cur[fld] = "" if val.strip() in ("|", ">", "|-", ">-", "|+", ">+") else _clean_scalar(val)
            cur_prio[fld] = prio
            last_key = fld
            continue
        if cur is None:
            continue
        if not stripped:
            last_key = None
            continue
        if last_key and indent > item_indent and not re.match(r"^(?:[-*+]|\d+[.)])[ \t]+\S", stripped):
            cur[last_key] = (cur[last_key] + " " + stripped).strip()
            continue
        if m and indent >= item_indent and (m.group("bullet") or indent > item_indent):
            # an unknown key inside the item (e.g. "attack_plan:"): ignore it
            last_key = None
            continue
        flush()
    flush()

    out: list[ReviewCritique] = []
    for item in items:
        sev = normalize_severity(item.get("severity")) if item.get("severity") else None
        if sev is None:
            continue
        raw_sev = _clean_scalar(item.get("severity", ""))
        if re.search(r"\b(?:category|description)\s*:", raw_sev, re.IGNORECASE):
            continue  # one-line legacy "Severity: X, Category: Y, ..." is not YAML
        c = critique_from_mapping({**item, "severity": sev.value})
        if c is not None:
            out.append(c)
    return out


def _derived_grade(critiques: list[ReviewCritique]) -> str:
    # Imported lazily: consensus imports this module.
    from backend.orchestrator.consensus import _compute_grade

    return _compute_grade(critiques)


class FeedbackFormatter:
    """Parse reviewer output and format as actionable feedback for executors."""

    def parse_review(self, output: str) -> ReviewResult:
        """Parse reviewer output into structured ReviewResult.

        Detects critiques, first match wins and each critique is parsed once:
        1. the fenced JSON critique contract (lenient decoding);
        2. YAML-like ``- severity: X`` / ``description: Y`` blocks;
        3. legacy line formats, de-duplicated by text span:
           "- Severity: X, Category: Y, Description: Z" (structured),
           "**Severity: X** -- description" (markdown),
           "[SEVERITY] description" (bracket, brackets required),
           "SEVERITY: description" (line-initial token + separator), and
           bulleted lists under "Concerns" / "Issues" / "Required Fixes" headers.

        Grading (the same rule as the consensus panel for contract replies):

        * JSON / YAML contract: the grade is derived from the critique
          severities (``consensus._compute_grade``). A grade field inside the
          JSON payload can only make it worse, never better; free text
          outside the block (which the prompts declare "ignored by the
          pipeline") is never searched for a grade.
        * legacy line formats: an explicit ``Grade: X`` line (uppercase
          letter A-D/F or pass/fail) is used when present.
        * a contract block that cannot count as a review (all entries
          unreadable, or truncated before any critique was complete) yields
          ``parse_method="none"`` and NO grade, so it can never trigger a
          grade-based stop; its reason is in ``parse_detail``.
        """
        result = ReviewResult()
        output = output or ""

        # Extract reproducibility gaps
        result.reproducibility_gaps = self._extract_section_items(
            output, r"(?:Reproducibility|Repro)\s*(?:Gaps?|Check|Issues?)"
        )

        # Extract suspected confounders
        result.suspected_confounders = self._extract_section_items(
            output, r"(?:Suspected\s+)?Confounders?"
        )

        grade = ""
        parsed = extract_json_critiques(output)
        if parsed is not None and parsed.is_valid_contract:
            critiques = parsed.critiques
            method = "json"
            result.overall_assessment = parsed.overall_assessment
            result.parse_detail = parsed.detail
            grade = _derived_grade(critiques)
            if parsed.grade and (grade_score(parsed.grade) or 0.0) < (grade_score(grade) or 0.0):
                grade = parsed.grade
        else:
            critiques = parse_yaml_critiques(output)
            method = "yaml" if critiques else ""
            if critiques:
                grade = _derived_grade(critiques)
            else:
                critiques = self._parse_legacy_critiques(output)
                method = "regex" if critiques else "none"
                if critiques or parsed is None:
                    # Legacy reply (or plain prose with no contract block):
                    # an explicit "Grade: X" line is the grade.
                    grade_match = _GRADE_RE.search(output)
                    if grade_match:
                        grade = grade_match.group(1).upper()
            if parsed is not None:
                # A contract block was present but could not be trusted.
                result.parse_detail = f"invalid_contract:{parsed.invalid_reason}"

        result.overall_grade = grade
        result.parse_method = method
        # Infer artifact/section references from critique text
        result.critiques = [self._infer_refs(c) for c in critiques]
        return result

    def _parse_legacy_critiques(self, output: str) -> list[ReviewCritique]:
        """Legacy line formats; every text span yields at most one critique."""
        found: list[tuple[int, ReviewCritique]] = []
        claimed: list[tuple[int, int]] = []

        def overlaps(a: int, b: int) -> bool:
            return any(a < e and s < b for s, e in claimed)

        def add(start: int, end: int, critique: ReviewCritique) -> None:
            if overlaps(start, end):
                return
            if any(c.description == critique.description for _, c in found):
                claimed.append((start, end))
                return
            claimed.append((start, end))
            found.append((start, critique))

        # 1. Structured format (most specific)
        for match in _STRUCTURED_RE.finditer(output):
            add(match.start(), match.end(), ReviewCritique(
                severity=_str_to_severity(match.group(1)),
                category=match.group(2).strip(),
                description=match.group(3).strip(),
            ))

        # 2. Markdown severity format
        for match in _MARKDOWN_SEVERITY_RE.finditer(output):
            add(match.start(), match.end(), ReviewCritique(
                severity=_str_to_severity(match.group(1)),
                category="general",
                description=match.group(2).strip(),
            ))

        # 3. Bracket format (brackets required)
        for match in _BRACKET_RE.finditer(output):
            add(match.start(), match.end(), ReviewCritique(
                severity=_str_to_severity(match.group(1)),
                category="general",
                description=match.group(2).strip(),
            ))

        # 4. Line-initial severity token with an explicit separator
        for match in _LINE_SEVERITY_RE.finditer(output):
            add(match.start(), match.end(), ReviewCritique(
                severity=_str_to_severity(match.group(1)),
                category="general",
                description=match.group(2).strip(),
            ))

        # 5. Bulleted items under concern/issue headers (skip lines already parsed)
        for start, end, item in self._extract_bulleted_concerns_spans(output):
            add(start, end, ReviewCritique(
                severity=_parse_severity(item),
                category="general",
                description=item,
            ))

        found.sort(key=lambda t: t[0])
        return [c for _, c in found]

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
        return [item for _, _, item in self._extract_bulleted_concerns_spans(output)]

    def _extract_bulleted_concerns_spans(self, output: str) -> list[tuple[int, int, str]]:
        """Like :meth:`_extract_bulleted_concerns` but with (start, end) line offsets."""
        header_re = re.compile(
            r"#{1,3}\s+(?:Concerns?|Issues?|Required\s+Fixes?|Problems?|Weaknesses?)\s*\n",
            re.IGNORECASE,
        )
        results: list[tuple[int, int, str]] = []
        for match in header_re.finditer(output):
            pos = match.end()
            offset = pos
            # Collect bulleted lines after the header
            for line in output[pos:].split("\n"):
                line_start, line_end = offset, offset + len(line)
                offset = line_end + 1
                stripped = line.strip()
                if stripped.startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    item = re.sub(r"^[-*\d.]+\s+", "", stripped).strip()
                    if item:
                        results.append((line_start, line_end, item))
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

        meta = review.consensus_meta or {}
        failed = list(meta.get("failed_lenses") or [])
        if meta.get("panel_failed"):
            lines.append(
                "PANEL STATUS: INCOMPLETE - no reviewer lens returned a usable review "
                f"(failed: {', '.join(failed) or 'all'}). No critiques are available; "
                "this is not an endorsement of the artifacts."
            )
            lines.append("")
        elif failed:
            lines.append(
                f"PANEL STATUS: PARTIAL - lens(es) {', '.join(failed)} failed; "
                "critiques below come from the remaining lenses only."
            )
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
                lines.extend(self._detail_lines(c))
                counter += 1
            lines.append("")

        if suggested:
            lines.append("SUGGESTED IMPROVEMENTS:")
            for c in suggested:
                tag = c.severity.value.upper()
                loc = self._format_location(c)
                lines.append(f"{counter}. [{tag}{loc}] {c.description}")
                lines.extend(self._detail_lines(c))
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
    def _detail_lines(critique: ReviewCritique) -> list[str]:
        """Indented lines under a critique: the reviewer's required fix, the
        suggested experiment, and the text of any other critique merged into
        it (consensus), each on one line."""
        def one_line(text: Any) -> str:
            return " ".join(str(text or "").split())

        out: list[str] = []
        fix = one_line(critique.required_fix)
        if fix:
            out.append(f"   Fix: {fix}")
        exp = one_line(critique.suggested_experiment)
        if exp:
            out.append(f"   Suggested experiment: {exp}")
        for m in critique.merged_members or []:
            desc = one_line(m.get("description"))
            if not desc:
                continue
            lens = one_line(m.get("lens"))
            sev = one_line(m.get("severity")).upper()
            who = ", ".join(p for p in (lens, sev) if p)
            out.append(f"   Also raised{f' ({who})' if who else ''}: {desc}")
            mfix = one_line(m.get("required_fix"))
            if mfix and mfix != fix:
                out.append(f"     Fix: {mfix}")
        return out

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
