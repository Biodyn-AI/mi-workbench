"""Shared definitions for the stopping-rule calibration study (X3).

Contract: the "Stopping-rule calibration (real loops)" section of
``experiments/benchmark/ANALYSIS_PLAN.md``. Items are the 12 FLAWED benchmark
artifacts; each trajectory is one run of the platform engine
(``reviewer_consensus`` preset, seeded with the original write-up as the
executor's iteration-0 output) over a fixed horizon:

    P0 E1 P1 E2 P2 E3 P3 E4 P4 E5 P5      (engine iterations 1..11)

``P_j`` is the three-lens consensus panel reviewing executor state ``E_j``
(``E0`` = the original flawed write-up). Engine iteration ``n`` is panel
``P_{(n-1)/2}`` when ``n`` is odd and executor revision ``E_{n/2}`` when even.

The module is pure standard library plus the platform's ``backend`` package
and the benchmark helpers (``experiments/benchmark/common.py``, loaded under
the module name ``miw_bench_common`` so it never clashes with another
``common`` module).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

STOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STOP_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCH_DIR = REPO_ROOT / "experiments" / "benchmark"
DEFAULT_DATA_DIR = REPO_ROOT / "experiments_data" / "stopping_runs"
RESULTS_DIR = STOP_DIR / "results"
PROMPTS_DIR = STOP_DIR / "prompts"

#: Executor revisions per trajectory (ANALYSIS_PLAN: fixed horizon of 5 cycles).
HORIZON = 5
LOOP_PRESET = "reviewer_consensus"
PANEL_ROLE = "consensus_merger"
EXECUTOR_ROLE = "executor"
LENS_ROLES = ("reviewer", "adversarial_reviewer", "bio_plausibility_checker")

TRAJECTORY_SCHEMA = "miw-stopping-trajectory/1"
UNIT_SCHEMA = "miw-stopping-unit/1"
CALL_SCHEMA = "miw-stopping-call/1"
JUDGMENT_SCHEMA = "miw-stopping-judgment/1"
RESULTS_SCHEMA = "miw-stopping-results/1"

#: Model configurations. ``sol-medium`` is the primary configuration.
CONFIGS: dict[str, dict[str, str]] = {
    "sol-medium": {"model": "gpt-5.6-sol", "effort": "medium"},
    "luna-low": {"model": "gpt-5.6-luna", "effort": "low"},
}
PRIMARY_CONFIG = "sol-medium"

#: Executor instructions (the run's task text). The engine shows it to the
#: executor in every revision request ("Original task: ...") and to the
#: reviewers as context only ("TASK GIVEN TO THE EXECUTOR (context only)").
EXECUTOR_TASK = """\
You are revising a written analysis report (a mechanistic-interpretability analysis \
of a single-cell foundation model) in response to reviewer feedback.

What you can and cannot do:
- You CANNOT run new analyses, execute code, or access any data, model, or file. The \
only information available to you is the write-up itself and the reviewers' feedback.
- You MAY correct or withdraw claims; rephrase, narrow or weaken conclusions; add \
caveats and limitations; fix incorrect technical, methodological or statistical \
statements; and specify analyses that must be run before a claim can stand. Mark each \
such analysis explicitly as "[PENDING ANALYSIS]" and say what would have to be done \
and what outcome would support or refute the claim.
- You MUST NOT report any new numerical result (statistic, p-value, effect size, \
count, percentage, metric) that is not already in the original write-up, and you must \
not state or imply that any new analysis, control, correction or test was performed. \
If a reviewer asks for an analysis, describe it as pending; never state or guess its \
outcome. Numbers that are already in the write-up may be kept only where they remain \
valid; otherwise remove or qualify them.

Output format: respond with the COMPLETE revised write-up in Markdown, from the title \
to the last section, every time. Do not output a list of changes, a diff, separate \
files or artifacts, or any commentary before or after the write-up."""


# ── small utilities ─────────────────────────────────────────────────────


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def append_line(path: Path, line: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
        fh.flush()


def model_dirname(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split(":", 1)[-1])


def parse_csv(value: Optional[str]) -> list[str]:
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


# ── trajectory geometry ─────────────────────────────────────────────────


def total_iterations(horizon: int = HORIZON) -> int:
    """Engine iterations of a seeded trajectory: H+1 panels and H revisions."""
    return 2 * int(horizon) + 1


def expected_roles(horizon: int = HORIZON) -> list[str]:
    return [PANEL_ROLE if n % 2 == 1 else EXECUTOR_ROLE
            for n in range(1, total_iterations(horizon) + 1)]


def step_label(iteration: int) -> str:
    """``P_j`` for odd engine iterations, ``E_k`` for even ones."""
    n = int(iteration)
    if n <= 0:
        return "E0"
    return f"P{(n - 1) // 2}" if n % 2 == 1 else f"E{n // 2}"


def state_index_at(iteration: int) -> int:
    """Index k of the executor state current after engine iteration ``n``
    (a panel ``P_k`` reviews ``E_k``; revision ``E_k`` produces it)."""
    return max(0, int(iteration)) // 2


def run_id_for(config: str, item: str) -> str:
    return f"{config}__{item}"


def unit_dir(data_dir: Path, config: str, item: str) -> Path:
    return Path(data_dir) / config / item


def judgment_path(data_dir: Path, judge_model: str, config: str, item: str, k: int) -> Path:
    return (Path(data_dir) / "judgments" / model_dirname(judge_model) / config / item
            / f"E{int(k)}.json")


# ── items (benchmark helpers) ───────────────────────────────────────────

_BENCH_COMMON = None


def bench_common():
    """``experiments/benchmark/common.py`` loaded as ``miw_bench_common``."""
    global _BENCH_COMMON
    if _BENCH_COMMON is None:
        name = "miw_bench_common"
        if name in sys.modules:
            _BENCH_COMMON = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, BENCH_DIR / "common.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            _BENCH_COMMON = mod
    return _BENCH_COMMON


def flawed_items(bench_dir: Optional[Path] = None) -> dict:
    """The flawed benchmark items (``Item`` objects), keyed by item id."""
    bc = bench_common()
    items = bc.load_items(Path(bench_dir) if bench_dir else bc.BENCH_DIR)
    return {k: v for k, v in items.items() if v.is_flawed}


# ── text helpers used for the trajectory record ─────────────────────────

_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.,])[-−]?\d+(?:[.,]\d+)*(?:\s?%)?")


def numeric_tokens(text: str) -> set[str]:
    """Numbers in a text, normalised (thousands separators kept as written,
    unicode minus mapped to '-', spaces before % removed). Single digits
    0-9 are ignored (list numbering, layer/section labels)."""
    out: set[str] = set()
    for m in _NUM_RE.finditer(text or ""):
        tok = m.group(0).replace("−", "-").replace(" ", "")
        core = tok.lstrip("-").rstrip("%")
        if len(core) == 1 and core.isdigit():
            continue
        out.add(tok)
    return out


def new_numbers(current: str, original: str) -> list[str]:
    """Numbers in ``current`` that do not occur in ``original`` (a cheap,
    deterministic cross-check of the judge's fabrication labels; numbers may
    legitimately be new, e.g. 144 x 0.05 = 7.2 derived from the original)."""
    orig = numeric_tokens(original)
    orig_bare = {t.rstrip("%").lstrip("-") for t in orig}
    out = []
    for tok in sorted(numeric_tokens(current)):
        if tok in orig or tok.rstrip("%").lstrip("-") in orig_bare:
            continue
        out.append(tok)
    return out


def pending_markers(text: str) -> int:
    return len(re.findall(r"\[PENDING ANALYSIS\]", text or "", flags=re.IGNORECASE))
