#!/usr/bin/env python
"""Trace every number reported in each executor write-up to executed output.

For every executor iteration of every X4 run (see ``run_case_study.py``):

1. Two write-up channels are analysed. ``agent_text``: the executor's full
   cleaned output (``executor_output.md``) with all fenced code blocks removed
   (code is executed, not reported). ``generated_reports``: narrative ``*.md``
   files written BY THE EXECUTED CODE (agents often emit only code that writes
   MECH.md / EVAL.md with formatted results; the reviewers see these files in
   the execution report). A generated-report number is ``computed`` unless the
   executed code contains it as a literal; such a literal counts as traceable
   only if it also equals a value in stdout or a non-narrative output file
   (``literal_matches_output``), otherwise it is ``hard_coded`` (untraceable).
2. Every numeric literal in the write-up is extracted and classified:

   ``task_given``   the literal appears in the task text (kit sizes, seeds, limits);
   ``list_marker``  ordered-list / heading numbering;
   ``index``        layer / head / iteration / section / critique / table indices
                    ("layer 3", "heads 1-12", "Fix 4", "l=3, h=7", ...);
   ``year``         a year in a citation-like context;
   ``small_integer``an integer with |value| <= 12 (layer/head counts and indices,
                    counts trivially matched by any output); traced, reported
                    separately;
   ``parameter``    an analysis setting (seed, number of permutations / bootstrap
                    resamples, thresholds such as ">= 50 cells", "top 100",
                    "95% CI", chance level);
   ``threshold``    a conventional significance level next to p / q / FDR / alpha;
   ``result``       everything else: the numbers whose provenance is checked.

   Identifier-attached digits (``L3``, ``layer_3``, ``V2-104M``, ``10x``,
   ``auroc_l3``), version strings (``1.26.4``) and digits inside words never
   count as numbers.
3. A number is *traceable* when it equals, after rounding to the precision
   shown in the write-up, a number printed to stdout or contained in a text
   file written by the executed code (``results.json`` and others), in the same
   iteration ("current") or in an earlier executor
   iteration of the same run ("earlier"). Matching tolerance is half a unit
   in the last shown digit (scaled for scientific notation); percentages also
   match the corresponding fraction and vice versa; the sign may differ
   (recorded as ``sign_mismatch``); thousands separators are ignored.
4. For the plan-only control nothing is executed, so every reported result
   number is untraceable by definition.

Sources are the COMPLETE outputs persisted per execution unit
(``code_execution_persist_dir``; ``CODE_EXECUTION.json`` ``blocks[].persisted_to``:
``stdout.txt`` and every text file under ``work/``); without them the capped
report in ``CODE_EXECUTION.json`` is used. Caveats recorded per iteration:
whether the outputs used were complete or capped/truncated (an untraceable
number may then sit in the omitted part) and how
many traceable results also appear as a literal in the executed code (a
hard-coded number printed back is "traceable" without being computed).

Output: ``results/trace_summary.json`` (per iteration and per run counts and
every untraceable result number with its context), ``results/trace_details/
<key>.json`` (every extracted number) and ``results/trace_per_iteration.csv``.

Usage::

    python experiments/case_study/agent_runs/trace_numbers.py \
        [--data-dir experiments_data/case_study_agent_runs] [--out-dir .../results]

Pure standard library (runs in the platform and in the analysis environment).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as _dt
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_DATA_DIR = REPO / "experiments_data" / "case_study_agent_runs"
DEFAULT_OUT_DIR = HERE / "results"

# ── Number extraction ─────────────────────────────────────────────────────

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_MINUS_CHARS = "-−"           # hyphen-minus, U+2212 (en/em dashes are never signs)

_NUM_RE = re.compile(
    r"""
    (?<![\w.])                                  # not part of an identifier / version
    (?P<sign>[-+−]?)
    (?P<mant>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)
    (?P<exp>
        [eE][-+−]?\d+
      | \s?[x×]\s?10\s?\^\s?\(?[-+−]?\d+\)?
      | \s?[x×]\s?10[⁻⁺]?[⁰¹²³⁴-⁹]+
    )?
    (?P<pct>\s?%)?
    (?![\w])                                    # not followed by letters (104M, 10x, 5th)
    (?!\.\d)                                    # not a version / IP prefix (1.26.4)
    """,
    re.VERBOSE,
)


@dataclass
class Num:
    text: str
    value: float
    decimals: int
    exponent: int
    percent: bool
    start: int
    end: int

    @property
    def tolerance(self) -> float:
        return 0.5 * 10.0 ** (self.exponent - self.decimals)


def _parse_exp(raw: str) -> int:
    if not raw:
        return 0
    raw = raw.translate(_SUPERSCRIPT).replace("−", "-")
    m = re.search(r"[eE]([-+]?\d+)$", raw.strip())
    if m:
        return int(m.group(1))
    m = re.search(r"10\s?\^?\s?\(?([-+]?\d+)\)?$", raw.strip())
    if m:
        return int(m.group(1))
    return 0


def extract_numbers(text: str) -> list[Num]:
    """All numeric literals of ``text`` (see module docstring for the rules)."""
    out: list[Num] = []
    for m in _NUM_RE.finditer(text):
        sign = m.group("sign")
        start = m.start()
        if sign:
            prev = text[start - 1] if start > 0 else ""
            # "0.16-0.32", "3-5", "10%-20%", "(a)-(b)": a range separator, not a sign.
            if prev and (prev.isalnum() or prev in "%)]_"):
                sign = ""
                start += 1
        mant = m.group("mant").replace(",", "")
        decimals = len(mant.split(".", 1)[1]) if "." in mant else 0
        exponent = _parse_exp(m.group("exp") or "")
        try:
            value = float(mant) * (10.0 ** exponent)
        except (ValueError, OverflowError):
            continue
        if sign in _MINUS_CHARS and sign:
            value = -value
        out.append(Num(text=text[start:m.end()].strip(), value=value, decimals=decimals,
                       exponent=exponent, percent=bool(m.group("pct")),
                       start=start, end=m.end()))
    return out


def output_values(text: str) -> list[float]:
    """Numbers of an execution output (stdout or captured file), any context.

    Identifier-attached digits are still skipped; JSON numbers are found by
    the same scan (``"auroc": 0.6734`` -> 0.6734)."""
    vals = [n.value for n in extract_numbers(text)]
    # Percent signs in outputs: also keep the fraction.
    vals += [n.value / 100.0 for n in extract_numbers(text) if n.percent]
    return vals


# ── Write-up preparation ──────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
_FENCE_OPEN_RE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*\Z", re.DOTALL | re.MULTILINE)
_CODE_PY_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def split_writeup(output: str) -> tuple[str, list[str]]:
    """(write-up without fenced blocks, python code blocks).

    Removed blocks are replaced by blank lines of equal count so that line
    numbers in the write-up still match the original output."""
    code = [m.group(1) for m in _CODE_PY_RE.finditer(output)]

    def blank(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")

    text = _FENCE_RE.sub(blank, output)
    text = _FENCE_OPEN_RE.sub(blank, text)      # an unterminated final fence
    return text, code


# ── Classification ────────────────────────────────────────────────────────

_INDEX_LEFT = re.compile(
    r"(?:\b(?:layers?|heads?|blocks?|iterations?|iter|steps?|sections?|fix(?:es)?|"
    r"critiques?|tables?|figures?|figs?|items?|hypothes[ie]s|rounds?|parts?|phases?|"
    r"stages?|criteri(?:on|a)|questions?|lens(?:es)?|cycles?|columns?|rows?|versions?|"
    r"python|numpy|scipy|pandas|gpt|points?|issues?|comments?|requirements?|"
    r"deliverables?|recommendations?|concerns?|findings?|claims?)\.?[-\s]*(?:#|no\.?|=|:)?\s*"
    r"|§\s*|\b[lh]\s*=\s*)$",
    re.IGNORECASE,
)
_RANGE_GAP = re.compile(r"^\s*(?:-|–|—|to|through|thru|,|and|or|&|/|\.\.)\s*$",
                        re.IGNORECASE)
_HEADING_PREFIX = re.compile(r"[ \t>]*#{1,6}[ \t]*(?:\*\*)?")
_LIST_PREFIX = re.compile(r"[ \t>]*(?:[-*+][ \t]+)?(?:\*\*)?")


def _is_list_marker(text: str, n: "Num") -> bool:
    """Ordered-list numbers ("1. ", "2) ", "**3.**") and heading numbering
    ("### 2.1 Methods", "## 3. Results")."""
    prefix = text[text.rfind("\n", 0, n.start) + 1:n.start]
    right = text[n.end:n.end + 3]
    if _HEADING_PREFIX.fullmatch(prefix):
        return right[:1] in (" ", "\t", ".", ")", ":", "")
    is_int = n.decimals == 0 and n.exponent == 0 and not n.percent
    return bool(is_int and _LIST_PREFIX.fullmatch(prefix) and right[:1] in (".", ")")
                and right[1:2] in (" ", "\t", "\n", "", "*"))
_PARAM_LEFT = re.compile(
    r"(?:\b(?:seeds?|random[_ ]state|n[_ ]?perm\w*|n[_ ]?boot\w*|permutations?|bootstraps?|"
    r"resamples?|replicates?|alpha|thresholds?|cut-?offs?|at\s+least|at\s+most|minimum|"
    r"min|maximum|max|top|top-?k|k|window|bins?|folds?|tolerance|tol|lambda|"
    r"n_estimators|n_components|chance(?:\s+level)?(?:\s+of)?|random\s+(?:level|baseline))"
    r"\.?\s*(?:of|=|:|≥|>=|≤|<=|>|<)?\s*|[αλ]\s*=?\s*|(?:≥|>=)\s*)$",
    re.IGNORECASE,
)
_PARAM_RIGHT = re.compile(
    r"^(?:\s*-?\s*fold\b|\s*(?:permutations?|perms?|bootstraps?|bootstrap\s+\w+|resamples?|"
    r"draws|replicates|random\s+seeds?|seeds?|bins?|restarts?|shuffles?|rewirings?)\b|"
    r"\s*%?\s*(?:ci|c\.i\.|confidence|credible|bootstrap\s+ci|interval)\b|\s*\(chance\)|"
    r"\s*=?\s*chance\b|\s*\(random\))",
    re.IGNORECASE,
)
_SIG_LEVELS = {0.05, 0.01, 0.001, 0.1, 0.005, 0.0001, 0.025, 0.2}
_THRESH_LEFT = re.compile(
    r"(?:\b(?:p|q|fdr|bh|padj|p_adj|p-adj\w*|adjusted\s+p|q-value|p-value|alpha|"
    r"significance(?:\s+level)?)|α)\s*(?:<|≤|<=|=|>|at|of|level)?\s*$",
    re.IGNORECASE,
)
_THRESH_RIGHT = re.compile(r"^\s*%?\s*(?:fdr|significance|level|threshold)\b", re.IGNORECASE)
_YEAR_LEFT = re.compile(r"(?:\(|et\s+al\.?,?|\bin|,)\s*$", re.IGNORECASE)


def _canon(n: Num) -> str:
    """Canonical literal (no thousands separators / percent sign / sign)."""
    t = n.text.lstrip("+-−").replace(",", "").replace("%", "").strip()
    return t


def task_literals(task: str) -> set[str]:
    """Numeric literals of the task text (excluding list markers)."""
    lits: set[str] = set()
    for n in extract_numbers(task):
        if _is_list_marker(task, n):
            continue
        lits.add(_canon(n))
    return lits


def classify(text: str, nums: list[Num], task_lits: set[str]) -> list[str]:
    cats: list[str] = []
    prev_cat: Optional[str] = None
    prev_end = 0
    for n in nums:
        # Markdown emphasis / inline-code marks around a number ("at least `100`")
        # do not change its context.
        left = re.sub(r"[`*]+$", "", text[max(0, n.start - 48):n.start])
        right = re.sub(r"^[`*]+", "", text[n.end:n.end + 32])
        gap = text[prev_end:n.start]
        is_int = n.decimals == 0 and n.exponent == 0 and not n.percent
        a = abs(n.value)
        cat = "result"
        if _canon(n) in task_lits and not (is_int and a <= 12):
            cat = "task_given"
        elif _is_list_marker(text, n):
            cat = "list_marker"
        elif _INDEX_LEFT.search(left) or (prev_cat == "index" and _RANGE_GAP.match(gap)):
            cat = "index"
        elif is_int and 1990 <= a <= 2035 and (_YEAR_LEFT.search(left)
                                                 or right[:1] in (")", ";")):
            cat = "year"
        elif (any(math.isclose(a, s, rel_tol=0, abs_tol=1e-12) for s in _SIG_LEVELS)
              and (_THRESH_LEFT.search(left) or _THRESH_RIGHT.match(right))) or \
                (n.percent and a in (1.0, 5.0, 10.0) and
                 (_THRESH_RIGHT.match(right) or _THRESH_LEFT.search(left))):
            cat = "threshold"
        elif _PARAM_LEFT.search(left) or _PARAM_RIGHT.match(right) or \
                (n.percent and a in (90.0, 95.0, 99.0) and re.match(r"^\s*(?:ci|confidence|bootstrap|interval|-)",
                                                               right, re.IGNORECASE)):
            cat = "parameter"
        elif is_int and a <= 12:
            cat = "small_integer"
        cats.append(cat)
        prev_cat, prev_end = cat, n.end
    return cats


# ── Matching ──────────────────────────────────────────────────────────────


class ValueIndex:
    """Sorted numeric values of one source for tolerance lookups."""

    def __init__(self, values: Iterable[float]):
        vals = [v for v in values if math.isfinite(v)]
        self.signed = sorted(vals)
        self.absolute = sorted(abs(v) for v in vals)

    def __len__(self) -> int:
        return len(self.signed)

    @staticmethod
    def _hit(arr: list[float], x: float, tol: float) -> bool:
        i = bisect.bisect_left(arr, x - tol)
        return i < len(arr) and arr[i] <= x + tol

    def match(self, x: float, tol: float) -> Optional[str]:
        tol = tol + 1e-12 * max(1.0, abs(x))
        if self._hit(self.signed, x, tol):
            return "exact"
        if self._hit(self.absolute, abs(x), tol):
            return "sign_mismatch"
        return None


def match_number(n: Num, idx: ValueIndex) -> Optional[str]:
    if not len(idx):
        return None
    tol = n.tolerance
    got = idx.match(n.value, tol)
    if got:
        return got
    if n.percent:                      # 64% vs 0.6412
        got = idx.match(n.value / 100.0, tol / 100.0)
        if got:
            return got + "+percent_as_fraction"
    elif n.exponent == 0 and abs(n.value) <= 1.0:   # 0.64 vs "64.1%" / 64.12
        got = idx.match(n.value * 100.0, tol * 100.0)
        if got:
            return got + "+fraction_as_percent"
    return None


# ── Run discovery ─────────────────────────────────────────────────────────


@dataclass
class ExecSource:
    iteration: int
    executed: int = 0
    passed: int = 0
    failed: int = 0
    timed_out: int = 0
    stdout_truncated: bool = False
    files: list[str] = field(default_factory=list)
    files_truncated: list[str] = field(default_factory=list)
    files_unread: list[str] = field(default_factory=list)
    texts: list[tuple[str, str]] = field(default_factory=list)   # (label, text): all outputs
    data_texts: list[tuple[str, str]] = field(default_factory=list)  # stdout + non-.md files
    narratives: list[tuple[str, str]] = field(default_factory=list)  # (path, text) of *.md files
    complete: bool = False          # complete outputs (persisted_to) were available
    persist_notes: list[str] = field(default_factory=list)

    def add(self, label: str, path: str, text: str) -> None:
        self.texts.append((label, text))
        if path != "stdout" and path.lower().endswith(".md"):
            self.narratives.append((path, text))
        else:
            self.data_texts.append((label, text))


MAX_PERSISTED_FILE_BYTES = 64 * 1024 * 1024


def load_exec_source(iter_dir: Path, iteration: int) -> Optional[ExecSource]:
    """Executed output of one executor iteration: the complete persisted
    outputs (``blocks[].persisted_to``: full stdout and every text file the code
    wrote) when available, else the capped report in CODE_EXECUTION.json."""
    f = iter_dir / "CODE_EXECUTION.json"
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    src = ExecSource(iteration=iteration, executed=int(data.get("executed") or 0),
                     passed=int(data.get("passed") or 0), failed=int(data.get("failed") or 0),
                     timed_out=int(data.get("timed_out") or 0))
    blocks = data.get("blocks") or []
    persisted = [b.get("persisted_to") for b in blocks]
    if blocks and all(p and Path(p).is_dir() for p in persisted):
        src.complete = True
        for b in blocks:
            unit = Path(b["persisted_to"])
            notes = [n for n in (b.get("notes") or []) if n.startswith(("persist", "complete outputs",
                                                                           "written files"))]
            src.persist_notes += notes
            if (unit / "stdout.txt").is_file():
                src.add(f"iter{iteration}:stdout(full)", "stdout",
                        (unit / "stdout.txt").read_text(errors="replace"))
            work = unit / "work"
            for f in sorted(work.rglob("*")) if work.is_dir() else []:
                if not f.is_file() or f.name.startswith("._"):
                    continue
                rel = f.relative_to(work).as_posix()
                src.files.append(rel)
                raw = f.read_bytes()[:MAX_PERSISTED_FILE_BYTES]
                if b"\x00" in raw[:8192]:
                    continue                      # binary (e.g. .npy): not a number source
                src.add(f"iter{iteration}:file(full):{rel}", rel, raw.decode("utf-8", "replace"))
        return src
    for b in blocks:
        if b.get("stdout"):
            src.add(f"iter{iteration}:stdout", "stdout", b["stdout"])
        src.stdout_truncated |= bool(b.get("stdout_truncated"))
        for fi in b.get("files") or []:
            path = fi.get("path", "?")
            src.files.append(path)
            if fi.get("content"):
                src.add(f"iter{iteration}:file:{path}", path, fi["content"])
            if fi.get("truncated"):
                src.files_truncated.append(path)
            if fi.get("content") is None and fi.get("skipped") not in (None, "binary"):
                src.files_unread.append(f"{path} ({fi.get('skipped')})")
    return src


def executor_iterations(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for d in sorted(run_dir.glob("iter_*")):
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and d.is_dir() and (d / "executor_output.md").is_file():
            out.append((int(m.group(1)), d))
    return out


def discover_runs(data_dir: Path) -> list[dict[str, Any]]:
    runs = []
    ws_root = data_dir / "workspaces"
    if not ws_root.is_dir():
        return runs
    for ws in sorted(p for p in ws_root.iterdir() if p.is_dir() and not p.name.startswith("._")):
        for rd in sorted((ws / "runs").glob("*")) if (ws / "runs").is_dir() else []:
            if not rd.is_dir() or rd.name.startswith("._"):
                continue
            spec = {}
            if (rd / "x4_run_spec.json").is_file():
                try:
                    spec = json.loads((rd / "x4_run_spec.json").read_text())
                except json.JSONDecodeError:
                    spec = {}
            execution = bool((spec.get("config") or {}).get("code_execution_enabled", False))
            runs.append({"key": ws.name, "run_id": rd.name, "run_dir": rd, "spec": spec,
                         "execution": execution})
    return runs


# ── Tracing ───────────────────────────────────────────────────────────────

PRIMARY = "result"
EXCLUDED = ("task_given", "list_marker", "index", "year")
SECONDARY = ("small_integer", "parameter", "threshold")


def trace_generated_reports(src: Optional[ExecSource], earlier: list[ExecSource],
                            code_lits: set[str], task_lits: set[str]) -> dict[str, Any]:
    """Channel 2: narrative reports (``*.md``) WRITTEN BY THE EXECUTED CODE.

    A number in such a report was produced by the executed code unless the
    code contains it as a literal (hard-coded text typed by the agent). Status
    per result number: ``computed`` (not a literal of the executed code),
    ``literal_matches_output`` (a literal, but it also equals a value in
    stdout / a non-narrative output file of this or an earlier iteration) or
    ``hard_coded`` (a literal found nowhere else in the outputs: untraceable).
    """
    counts = {c: {"total": 0, "traceable": 0} for c in (PRIMARY,) + SECONDARY}
    counts[PRIMARY].update({"computed": 0, "literal_matches_output": 0, "hard_coded": 0})
    excluded = {c: 0 for c in EXCLUDED}
    records, hard = [], []
    files = [p for p, _ in (src.narratives if src else [])]
    if src is None or not src.narratives:
        return {"files": files, "numbers_extracted": 0, "excluded": excluded,
                "counts": counts, "hard_coded_results": hard, "records": records}
    data_idx = ValueIndex(v for s in [src] + earlier for _, t in s.data_texts
                          for v in output_values(t))
    n_total = 0
    for path, text in src.narratives:
        nums = extract_numbers(text)
        n_total += len(nums)
        for n, cat in zip(nums, classify(text, nums, task_lits)):
            ctx = text[max(0, n.start - 60):n.end + 40].replace("\n", " ").strip()
            rec = {"channel": "generated_report", "file": path, "text": n.text,
                   "value": n.value, "category": cat, "context": ctx}
            records.append(rec)
            if cat in EXCLUDED:
                excluded[cat] += 1
                continue
            literal = _canon(n) in code_lits
            if not literal:
                status = "computed"
            elif match_number(n, data_idx):
                status = "literal_matches_output"
            else:
                status = "hard_coded"
            rec["status"] = status
            counts[cat]["total"] += 1
            if status != "hard_coded":
                counts[cat]["traceable"] += 1
            if cat == PRIMARY:
                counts[PRIMARY][status] += 1
                if status == "hard_coded":
                    hard.append({"file": path, "text": n.text, "context": ctx})
    return {"files": files, "numbers_extracted": n_total, "excluded": excluded,
            "counts": counts, "hard_coded_results": hard, "records": records}


def trace_run(run: dict[str, Any], task_text: Optional[str] = None) -> tuple[dict, dict]:
    rd: Path = run["run_dir"]
    task = task_text
    if task is None:
        tf = rd / "x4_task.md"
        task = tf.read_text(encoding="utf-8") if tf.is_file() else ""
    lits = task_literals(task)
    meta = {}
    if (rd / "run_meta.json").is_file():
        try:
            meta = json.loads((rd / "run_meta.json").read_text())
        except json.JSONDecodeError:
            meta = {}
    earlier_sources: list[ExecSource] = []
    earlier_code_lits: set[str] = set()
    iters_summary, iters_detail = [], []
    totals = {c: {"total": 0, "traceable": 0} for c in (PRIMARY,) + SECONDARY}
    totals[PRIMARY].update({"traceable_current": 0, "traceable_earlier": 0, "untraceable": 0,
                            "traceable_but_literal_in_code": 0})
    excluded_totals = {c: 0 for c in EXCLUDED}
    gen_totals: dict[str, int] = {}
    for round_no, (n_iter, idir) in enumerate(executor_iterations(rd), start=1):
        output = (idir / "executor_output.md").read_text(encoding="utf-8", errors="replace")
        writeup, code_blocks = split_writeup(output)
        code_lits = {_canon(n) for c in code_blocks for n in extract_numbers(c)}
        src = load_exec_source(idir, n_iter) if run["execution"] else None
        cur_idx = ValueIndex(v for _, t in (src.texts if src else []) for v in output_values(t))
        early_idx = ValueIndex(v for s in earlier_sources for _, t in s.texts
                               for v in output_values(t))
        nums = extract_numbers(writeup)
        cats = classify(writeup, nums, lits)
        line_starts = [0] + [i + 1 for i, ch in enumerate(writeup) if ch == "\n"]
        records, untraceable = [], []
        counts = {c: {"total": 0, "traceable": 0} for c in (PRIMARY,) + SECONDARY}
        counts[PRIMARY].update({"traceable_current": 0, "traceable_earlier": 0,
                                "untraceable": 0, "traceable_but_literal_in_code": 0})
        excluded = {c: 0 for c in EXCLUDED}
        for n, cat in zip(nums, cats):
            line = bisect.bisect_right(line_starts, n.start)
            ctx = writeup[max(0, n.start - 60):n.end + 40].replace("\n", " ").strip()
            rec = {"text": n.text, "value": n.value, "category": cat, "line": line,
                   "context": ctx}
            if cat in EXCLUDED:
                excluded[cat] += 1
                records.append(rec)
                continue
            where, how = None, None
            how = match_number(n, cur_idx)
            if how:
                where = "current"
            else:
                how = match_number(n, early_idx)
                if how:
                    where = "earlier"
            rec.update({"traced": where, "match": how})
            counts[cat]["total"] += 1
            if where:
                counts[cat]["traceable"] += 1
            if cat == PRIMARY:
                if where == "current":
                    counts[PRIMARY]["traceable_current"] += 1
                elif where == "earlier":
                    counts[PRIMARY]["traceable_earlier"] += 1
                else:
                    counts[PRIMARY]["untraceable"] += 1
                    untraceable.append({"text": n.text, "line": line, "context": ctx})
                lit_in_code = _canon(n) in code_lits or _canon(n) in earlier_code_lits
                rec["literal_in_code"] = lit_in_code
                if where and lit_in_code:
                    counts[PRIMARY]["traceable_but_literal_in_code"] += 1
            records.append(rec)
        gen = trace_generated_reports(src, earlier_sources, code_lits | earlier_code_lits, lits)
        records.extend(gen.pop("records"))
        res = counts[PRIMARY]
        frac = (res["traceable"] / res["total"]) if res["total"] else None
        g = gen["counts"][PRIMARY]
        comb_total = res["total"] + g["total"]
        comb_trace = res["traceable"] + g["traceable"]
        exec_info = None
        if src is not None:
            exec_info = {k: getattr(src, k) for k in ("executed", "passed", "failed", "timed_out",
                                                       "complete", "stdout_truncated", "files",
                                                       "files_truncated", "files_unread",
                                                       "persist_notes")}
        summary = {
            "iteration": n_iter, "executor_round": round_no,
            "n_python_blocks": len(code_blocks),
            "execution": exec_info,
            "execution_available": src is not None,
            "output_possibly_truncated": bool(src and (
                bool(src.persist_notes) if src.complete
                else (src.stdout_truncated or src.files_truncated or src.files_unread))),
            "numbers_extracted": len(nums),
            "excluded": excluded,
            "counts": counts,
            "result_traceable_fraction": frac,
            "untraceable_results": untraceable,
            "generated_reports": gen,
            "combined_result": {"total": comb_total, "traceable": comb_trace,
                                "untraceable": comb_total - comb_trace,
                                "traceable_fraction": (comb_trace / comb_total) if comb_total else None},
        }
        if not run["execution"]:
            summary["note"] = ("plan-only control: nothing was executed, so every reported "
                               "result number is untraceable by definition")
        iters_summary.append(summary)
        iters_detail.append({"iteration": n_iter, "numbers": records})
        for c in (PRIMARY,) + SECONDARY:
            for k, v in counts[c].items():
                totals[c][k] = totals[c].get(k, 0) + v
        for c in EXCLUDED:
            excluded_totals[c] += excluded[c]
        for k, v in gen["counts"][PRIMARY].items():
            gen_totals[k] = gen_totals.get(k, 0) + v
        if src is not None:
            earlier_sources.append(src)
        earlier_code_lits |= code_lits
    res = totals[PRIMARY]
    ct = res["total"] + gen_totals.get("total", 0)
    ctr = res["traceable"] + gen_totals.get("traceable", 0)
    run_summary = {
        "key": run["key"], "run_id": run["run_id"], "execution": run["execution"],
        "model": (run["spec"].get("spec") or {}).get("model") or meta.get("model"),
        "reasoning_effort": (run["spec"].get("spec") or {}).get("effort")
                            or meta.get("reasoning_effort"),
        "status": meta.get("status"), "stop_reason": meta.get("stop_reason"),
        "executor_iterations": len(iters_summary),
        "totals": totals, "excluded_totals": excluded_totals,
        "result_traceable_fraction": (res["traceable"] / res["total"]) if res["total"] else None,
        "result_untraceable_per_iteration": [it["counts"][PRIMARY]["untraceable"]
                                             for it in iters_summary],
        "generated_reports_totals": gen_totals,
        "combined_result": {"total": ct, "traceable": ctr, "untraceable": ct - ctr,
                            "traceable_fraction": (ctr / ct) if ct else None},
        "iterations": iters_summary,
    }
    return run_summary, {"key": run["key"], "run_id": run["run_id"], "iterations": iters_detail}


METHOD = {
    "writeup": "executor_output.md minus fenced code blocks",
    "categories": {
        "result": "primary: checked for provenance",
        "small_integer": "integers |v| <= 12; traced, reported separately",
        "parameter": "analysis settings (seeds, permutations, thresholds, CI level, chance)",
        "threshold": "conventional significance levels next to p/q/FDR/alpha",
        "task_given": "literal appears in the task text (excluded)",
        "list_marker": "ordered-list / heading numbering (excluded)",
        "index": "layer/head/iteration/section/critique indices (excluded)",
        "year": "citation-like years (excluded)",
    },
    "sources": "complete stdout and every text file written by the code (persisted per "
               "execution unit via code_execution_persist_dir; fallback: the capped report in "
               "CODE_EXECUTION.json) of the same iteration (current) or of earlier executor "
               "iterations of the run (earlier); stderr and code are not sources",
    "match": "equal after rounding to the shown precision (tolerance half a unit in the "
             "last shown digit); percent <-> fraction; sign-agnostic fallback flagged",
    "plan_only": "no execution: every result number is untraceable by definition",
    "channels": {
        "agent_text": "numbers the agent typed in its response outside code blocks; traced "
                      "against the executed outputs",
        "generated_reports": "numbers in *.md reports written by the executed code; "
                             "computed unless the executed code contains the literal; a literal "
                             "that matches no stdout / non-narrative output value is hard_coded "
                             "(untraceable)",
        "combined_result": "agent_text + generated_reports result numbers",
    },
}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Trace reported numbers to executed output")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--runs", default="", help="comma-separated run keys (default: all)")
    args = ap.parse_args(argv)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    keys = {k.strip() for k in args.runs.split(",") if k.strip()}
    runs = [r for r in discover_runs(data_dir) if not keys or r["key"] in keys]
    summaries, details = {}, {}
    for r in runs:
        s, d = trace_run(r)
        summaries[r["key"]] = s
        details[r["key"]] = d
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace_details").mkdir(exist_ok=True)
    payload = {"generated_at": _dt.datetime.now().isoformat(), "data_dir": str(data_dir),
               "method": METHOD, "runs": summaries}
    (out_dir / "trace_summary.json").write_text(json.dumps(payload, indent=2))
    for k, d in details.items():
        (out_dir / "trace_details" / f"{k}.json").write_text(json.dumps(d, indent=1))
    with open(out_dir / "trace_per_iteration.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "execution", "iteration", "executor_round", "exec_passed",
                    "output_possibly_truncated", "numbers_extracted", "result_total",
                    "result_traceable", "result_traceable_current", "result_traceable_earlier",
                    "result_untraceable", "result_traceable_fraction",
                    "traceable_but_literal_in_code", "small_integer_total",
                    "parameter_total", "threshold_total", "genrep_files",
                    "genrep_result_total", "genrep_computed", "genrep_literal_matches_output",
                    "genrep_hard_coded", "combined_result_total", "combined_result_traceable",
                    "combined_traceable_fraction"])
        for k, s in summaries.items():
            for it in s["iterations"]:
                c = it["counts"]
                w.writerow([k, s["execution"], it["iteration"], it["executor_round"],
                            (it["execution"] or {}).get("passed"),
                            it["output_possibly_truncated"], it["numbers_extracted"],
                            c[PRIMARY]["total"], c[PRIMARY]["traceable"],
                            c[PRIMARY]["traceable_current"], c[PRIMARY]["traceable_earlier"],
                            c[PRIMARY]["untraceable"],
                            ("" if it["result_traceable_fraction"] is None
                             else round(it["result_traceable_fraction"], 4)),
                            c[PRIMARY]["traceable_but_literal_in_code"],
                            c["small_integer"]["total"], c["parameter"]["total"],
                            c["threshold"]["total"], len(it["generated_reports"]["files"]),
                            it["generated_reports"]["counts"][PRIMARY]["total"],
                            it["generated_reports"]["counts"][PRIMARY]["computed"],
                            it["generated_reports"]["counts"][PRIMARY]["literal_matches_output"],
                            it["generated_reports"]["counts"][PRIMARY]["hard_coded"],
                            it["combined_result"]["total"], it["combined_result"]["traceable"],
                            ("" if it["combined_result"]["traceable_fraction"] is None
                             else round(it["combined_result"]["traceable_fraction"], 4))])
    for k, s in summaries.items():
        t = s["totals"][PRIMARY]
        g, cb = s["generated_reports_totals"], s["combined_result"]
        print(f"{k}: executor iterations {s['executor_iterations']}; agent-text result numbers "
              f"{t['total']}, traceable {t['traceable']} (current {t['traceable_current']}, "
              f"earlier {t['traceable_earlier']}), untraceable {t['untraceable']}; "
              f"code-generated reports: {g.get('total', 0)} results, computed "
              f"{g.get('computed', 0)}, literal+output {g.get('literal_matches_output', 0)}, "
              f"hard-coded {g.get('hard_coded', 0)}; combined untraceable {cb['untraceable']}"
              f"/{cb['total']}")
    print(f"wrote {out_dir / 'trace_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
