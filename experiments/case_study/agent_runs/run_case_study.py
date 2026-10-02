#!/usr/bin/env python
"""X4: executed case study with verified execution on the real Geneformer kit.

Four runs of the REAL platform engine (``backend.orchestrator.runner.execute_run``
driving ``LoopEngine``) with the Codex CLI adapter, preset ``reviewer_consensus``
(executor <-> three-lens consensus panel), fixed horizon of 5 executor
iterations (E1 P1 ... E5 P5 = 10 iterations), convergence disabled:

=================  ================  ======  =========================================
key                model             effort  code execution
=================  ================  ======  =========================================
``sol_A``          gpt-5.6-sol       medium  sandbox_exec (replicate A)
``sol_B``          gpt-5.6-sol       medium  sandbox_exec (replicate B)
``gpt55``          gpt-5.5           medium  sandbox_exec
``sol_planonly``   gpt-5.6-sol       medium  OFF (control; task text without the
                                             execution section)
=================  ================  ======  =========================================

Everything is persisted under ``experiments_data/case_study_agent_runs/``
(gitignored): a dedicated SQLite DB (runs, iterations, telemetry), one
workspace per run (``workspaces/<key>/runs/<run_id>/iter_NNNN/...`` with
``CODE_EXECUTION.json``, ``RUN_LOG.md``, lens outputs, ``CONSENSUS.json``,
``run_meta.json``), the exact prompt of every adapter call
(``x4_prompts/``), the engine events (``x4_events.jsonl``) and one
reproducibility package per run (``repropacks/<key>.zip``, built with
``backend.repropack`` and extended with the iteration directories).

Resumable: a completed run is skipped; a run that was interrupted (process
killed) or failed (adapter / panel failure after the engine's own retries)
is cleaned up (failed tail iterations removed from the DB, their directories
moved to ``x4_failed_attempts/``) and resumed from its last completed
iteration, so the fixed E/P horizon is kept. Every repair is logged in
``x4_repairs.jsonl``.

Commands (run from ``automation/mi-workbench``; see README.md)::

    python experiments/case_study/agent_runs/run_case_study.py print-task --variant executed
    python experiments/case_study/agent_runs/run_case_study.py smoke-sandbox
    python experiments/case_study/agent_runs/run_case_study.py run [--runs sol_A,sol_B,gpt55,sol_planonly]
    python experiments/case_study/agent_runs/run_case_study.py status
    python experiments/case_study/agent_runs/run_case_study.py repack [--runs ...]
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # automation/mi-workbench
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

KIT_DIR = REPO / "experiments_data" / "case_study_kit"
DATA_DIR = REPO / "experiments_data" / "case_study_agent_runs"
RESULTS_DIR = HERE / "results"
ANALYSIS_PYTHON = "/Users/ihorkendiukhov/anaconda3/bin/python"
TMPDIR_DEFAULT = "/Volumes/Crucial X6/tmp_miw"

QUESTION = ("Does Geneformer V2-104M attention encode TF->target regulation beyond "
            "co-expression, expression-rank proximity and hub (degree) structure?")

# Fixed horizon: 5 executor iterations, each followed by one consensus panel.
N_EXECUTOR_ITERATIONS = 5
MAX_ITERATIONS = 2 * N_EXECUTOR_ITERATIONS

EXEC_TIMEOUT_S = 900
EXEC_CPU_S = 1800
OUTPUT_LIMIT_BYTES = 24000
CAPTURE_TEXT_BYTES = 24000
CAPTURE_TOTAL_BYTES = 48000

logger = logging.getLogger("x4")


# ── Run specifications ────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RunSpec:
    key: str
    model: str
    effort: str
    execution: bool
    note: str

    @property
    def run_id(self) -> str:
        return f"x4_{self.key}"

    @property
    def workspace_id(self) -> str:
        return f"x4ws_{self.key}"


RUN_SPECS: dict[str, RunSpec] = {
    s.key: s for s in (
        RunSpec("sol_A", "gpt-5.6-sol", "medium", True, "verified execution, replicate A"),
        RunSpec("sol_B", "gpt-5.6-sol", "medium", True, "verified execution, replicate B"),
        RunSpec("gpt55", "gpt-5.5", "medium", True, "verified execution, second model"),
        RunSpec("sol_planonly", "gpt-5.6-sol", "medium", False,
                "CONTROL: code execution disabled (plan-only), task without execution section"),
    )
}
DEFAULT_ORDER = ["sol_A", "sol_B", "gpt55", "sol_planonly"]


# ── Task text ─────────────────────────────────────────────────────────────

_TASK_HEAD = """\
RESEARCH QUESTION
{question}

Answer this question with a quantitative analysis of the precomputed analysis kit described
below. The model itself is not available; everything the analysis needs is in the kit.

KIT DIRECTORY (read-only)
KIT = "{kit}"

HOW THE KIT WAS BUILT
- Model: Geneformer V2-104M (Hugging Face ctheodoris/Geneformer, snapshot
  fcd26c45fc30fba1989e586bdc46bc366dda8655), BertForMaskedLM, 12 layers x 12 heads, eager
  attention, eval mode (no dropout).
- Cells: 1000 cells from a Tabula Sapiens immune subset, restricted to the 10x 3' v3 assay (a
  single UMI chemistry), sampled stratified by cell_type with a cap of 32 cells per type (types with
  fewer cells contribute all of them; leftover slots go to randomly chosen larger types; seed
  20261001). Counts are the raw integer UMI counts.
- Tokenisation (Geneformer V2 rank-value encoding): per cell, counts / n_counts x 10000 (n_counts =
  total raw counts over all 60,606 genes), divided by the gene's Geneformer non-zero median; genes
  in the token dictionary with a non-zero value are sorted in descending order (stable sort) and
  truncated to 4094 genes; <cls> is prepended and <eos> appended (at most 4096 tokens).
- Gene rank r = token position in the sequence (1 = most highly ranked gene; <cls> is position 0).
- Gene set G (1322 genes): the 1200 genes most frequently present in the 1000 tokenised sequences
  (ties broken by mean normalised expression, then Ensembl ID), plus every TRRUST TF present in at
  least 30% of cells; ordered by presence frequency (descending). Index i in every matrix = row i
  of genes.tsv (0-based gene_index).
- Attention: for every cell and layer, the attention tensor (heads x tokens x tokens) was sliced to
  the rows and columns of the G genes present in that cell; per-head and head-averaged sums were
  accumulated over cells and divided by the number of cells in which both genes are co-present.
  "Co-present in cell c" means that both genes appear in cell c's (truncated) token sequence.
- Symbol -> Ensembl mapping for TRRUST and DoRothEA used the Geneformer gene-name dictionary
  first, then a unique match on the dataset's feature names.

FILES (every matrix is indexed by gene_index from genes.tsv)
- genes.tsv (1322 rows): gene_index, ensembl_id, symbol, token_id, is_tf (gene is a TRRUST
  source), is_dorothea_tf, selection (top_frequency or trrust_tf_ge30pct), n_cells_present,
  presence_frac, mean_norm_expr (mean counts/n_counts x 1e4 over all cells, zeros included),
  mean_log1p_norm_expr, mean_median_scaled_expr, mean_rank_when_present (mean token position over
  the cells where the gene is present), geneformer_median, trrust_out_degree_in_G,
  trrust_in_degree_in_G (non-self TRRUST edges with both ends in G).
- cells.tsv (1000 rows): cell_index, obs_name, h5ad_row, cell_type, donor_id, assay, tissue,
  n_counts_raw_allgenes, obs_total_counts, n_tokenizable_nonzero (before truncation),
  seq_len_with_special, truncated, n_G_present.
- tokenized_cells.npz (ragged arrays): input_ids (all token sequences concatenated, special tokens
  included) with offsets (cell c = input_ids[offsets[c]:offsets[c+1]]); g_index, g_position and
  g_offsets (for cell c, the G genes present and their token positions:
  g_index[g_offsets[c]:g_offsets[c+1]] and g_position[g_offsets[c]:g_offsets[c+1]]).
- attention_layer_mean.npy, float32 [12, 1322, 1322]: A[l, i, j] = mean attention weight FROM
  query gene i TO key gene j in layer l (l = 0..11 is the array index of the encoder layer, first
  layer first), averaged over the 12 heads and over the cells in which genes i and j are
  co-present. Rows are NOT renormalised: each full attention row sums to 1 over all tokens of that
  cell (including <cls>, <eos> and genes outside G), so a row of A restricted to G is not a
  probability distribution over G. NaN where copresence_counts == 0. The diagonal is
  self-attention.
- attention_heads_f32.npy, float32 [12, 12, 1322, 1322] (about 1 GB): A[l, h, i, j], the same
  definition for head h of layer l (no head averaging). Use this file for per-head statistics.
- attention_heads.npy, float16 [12, 12, 1322, 1322]: the same per-head means stored in float16. A
  large fraction of the values fall below the float16 normal range (6.1e-5), where relative
  precision is reduced; prefer the float32 file.
- copresence_counts.npy, int32 [1322, 1322]: C[i, j] = number of cells in which genes i and j are
  co-present (symmetric; the diagonal is the number of cells in which gene i is present).
- rank_distance_mean.npy, float32 [1322, 1322]: mean over co-present cells of |r_i - r_j|
  (token-position distance). NaN where C == 0.
- coexpr_pearson.npy, float32 [1322, 1322]: Pearson correlation across the 1000 cells of
  log1p(counts / n_counts x 1e4) (zeros included).
- coexpr_spearman.npy, float32 [1322, 1322]: Spearman correlation of the same vectors (average
  ranks for ties).
- trrust_edges.tsv (433 rows): unique TRRUST (TF -> target) pairs with both genes in G; columns
  tf, target, tf_ensembl, target_ensembl, tf_index, target_index, mode (';'-joined unique modes:
  Activation, Repression, Unknown), pmids, n_records, is_self (autoregulation).
- dorothea_abc_edges.tsv (2118 rows): unique DoRothEA (TF -> target) pairs with confidence A, B or
  C and both genes in G (best confidence kept); columns tf, target, tf_ensembl, target_ensembl,
  tf_index, target_index, confidence, is_self.
- attention_per_cell_log.tsv (1000 rows): cell_index, seq_len, n_G_present, max_rowsum_dev (a
  row-sum sanity check of the attention tensor), sec.
- prepare_summary.json: sampling strata, symbol-mapping statistics, gene-set construction counts.
- kit_manifest.json: provenance, md5 sums, package versions, sanity checks and runtimes of the kit
  build.
- README.md: the kit's own description (the information above).
- _checkpoint/: raw accumulators used while the kit was built; not needed.

ORIENTATION
TRRUST and DoRothEA edges are directed (TF -> target) and A is directed (query -> key). A
TF -> target pair (t, g) can be read as A[l, t, g] (TF as query), A[l, g, t] (target as query) or
a symmetrised combination. State which reading(s) you use and why.

WHAT THE ANALYSIS MUST DELIVER
1. Define the attention edge score(s), the candidate pair universe (which TF-gene pairs count as
   negatives) and the evaluation metric(s), and justify these choices.
2. Baselines and null models that address each confounder named in the question: co-expression
   (coexpr_*.npy), expression-rank proximity (rank_distance_mean.npy; co-presence and expression
   level are related nuisance variables) and hub / degree structure (TF out-degree, target
   in-degree). Test whether attention carries TF->target information beyond these confounders, not
   only whether it beats random pairs.
3. Effect sizes with uncertainty (for example bootstrap confidence intervals, stating the
   resampling unit), not p-values alone.
4. Multiple-testing control across the layers, heads and score variants you test (state the family
   and the procedure).
5. Null and negative results reported with the same care as positive ones, and a verdict on the
   research question that separates direct evidence, inference and hypothesis.
"""

_TASK_EXECUTION = """
COMPUTING ENVIRONMENT AND REPORTING RULE
- Your analysis code is executed by the orchestrator in a sandbox: no network access; it can read
  only the kit directory and write only to its current working directory; numpy, scipy, pandas,
  scikit-learn and statsmodels are available; BLAS/OpenMP use a single thread; each execution has
  a wall-clock limit of {timeout} s (CPU-time limit {cpu} s).
  np.load(path, mmap_mode="r") reads single heads of attention_heads_f32.npy without loading the
  whole array.
- Only numbers produced by your executed code may be reported. Every number in the write-up must be
  printed by the code or written to results.json. Do not report numbers from memory, from the
  literature or from estimation, and do not report numbers from a run that failed.
"""


def build_task_text(execution: bool, kit_dir: Path | str = KIT_DIR) -> str:
    """The task given to the agents (no results of any kind)."""
    text = _TASK_HEAD.format(question=QUESTION, kit=str(kit_dir))
    if execution:
        text += _TASK_EXECUTION.format(timeout=EXEC_TIMEOUT_S, cpu=EXEC_CPU_S)
    return text


def task_variant(spec: RunSpec) -> str:
    return "executed" if spec.execution else "plan_only"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Run configuration ─────────────────────────────────────────────────────


def run_config(spec: RunSpec, kit_dir: Path | str = KIT_DIR,
               analysis_python: str = ANALYSIS_PYTHON,
               persist_root: Path | str | None = None) -> dict[str, Any]:
    """Run-config overrides (``RunState.config``) for one run.

    ``persist_root``: complete executed outputs go to ``<persist_root>/<key>``
    (default ``DATA_DIR/exec_outputs``)."""
    cfg: dict[str, Any] = {
        "reasoning_effort": spec.effort,
        # Fixed horizon: no convergence stop, no grade stop, budgets that never bind.
        "convergence_enabled": False,
        "max_iterations": MAX_ITERATIONS,
        "budget_max_tokens": 50_000_000,
        "budget_max_cost": 1_000_000.0,
        "adapter_timeout": 2400,
        "max_retries": 3,
        # Reviewers (lenses and the LLM adjudicator) strictly tools-off.
        "reviewer_allow_tools": False,
        # Consensus: LLM adjudication of duplicate critiques (one extra call per
        # panel, after the 3 concurrent lens calls), cross-lens merging only.
        "consensus_similarity_method": "llm",
        "consensus_lens_timeout": 2400,
        "consensus_partial_policy": "no_stop",
    }
    if spec.execution:
        cfg.update({
            "code_execution_enabled": True,
            "code_execution_backend": "sandbox_exec",
            "code_execution_python": analysis_python,
            "code_execution_read_only_paths": [str(kit_dir)],
            "code_execution_timeout": EXEC_TIMEOUT_S,
            "code_execution_cpu_seconds": EXEC_CPU_S,
            "code_execution_output_limit_bytes": OUTPUT_LIMIT_BYTES,
            "code_execution_capture_text_bytes": CAPTURE_TEXT_BYTES,
            "code_execution_capture_total_bytes": CAPTURE_TOTAL_BYTES,
            "code_execution_block_mode": "single",
            # Complete outputs (full stdout/stderr, code, every written file) of
            # every execution unit, so each reported number can be traced.
            "code_execution_persist_dir": str(Path(persist_root or (DATA_DIR / "exec_outputs"))
                                              / spec.key),
            "code_execution_persist_max_mb": 512,
            # executor_allow_tools left unset: the verified-execution default is
            # OFF, so every number has to come from the sandboxed executor.
        })
    else:
        cfg.update({
            "code_execution_enabled": False,
            # Without code execution the engine default would give the executor
            # agent tools; the control must not be able to compute anything.
            "executor_allow_tools": False,
        })
    return cfg


# ── Paths ─────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Paths:
    data: Path

    @property
    def db(self) -> Path:
        return self.data / "miw_case_study.db"

    def workspace(self, spec: RunSpec) -> Path:
        return self.data / "workspaces" / spec.key

    def run_dir(self, spec: RunSpec) -> Path:
        return self.workspace(spec) / "runs" / spec.run_id

    def exec_outputs(self, spec: RunSpec) -> Path:
        return self.data / "exec_outputs" / spec.key

    @property
    def logs(self) -> Path:
        return self.data / "logs"

    @property
    def repropacks(self) -> Path:
        return self.data / "repropacks"

    @property
    def smoke(self) -> Path:
        return self.data / "smoke"

    @property
    def lock(self) -> Path:
        return self.data / "run.lock"


def setup_backend(db_path: Path) -> None:
    """Point the platform at the case-study DB (works whatever was imported before)."""
    os.environ["MIW_DB_PATH"] = str(db_path)
    from backend.config import config as _cfg
    _cfg.db_path = str(db_path)


# ── Adapter proxy: exact prompt of every call ────────────────────────────

_ITER_RE = re.compile(r"iter_(\d+)")


class RecordingAdapter:
    """Transparent adapter proxy that saves the exact prompt of every call
    (system + developer + user prompt, as the adapter receives it) and one
    JSONL index line per call. It changes nothing in the request or result."""

    def __init__(self, inner: Any, prompt_dir: Path):
        self.inner = inner
        self.prompt_dir = Path(prompt_dir)

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", type(self.inner).__name__)

    def __getattr__(self, item):
        return getattr(self.inner, item)

    async def run(self, request):
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        pb = request.prompt_bundle
        role = str((pb.variables or {}).get("role", "") or "unknown")
        m = _ITER_RE.search(request.workspace_context.iteration_dir or "")
        iteration = int(m.group(1)) if m else 0
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
        fname = f"iter_{iteration:04d}__{re.sub(r'[^A-Za-z0-9_.-]', '_', role)}__{stamp}.prompt.md"
        body = (f"=== SYSTEM PROMPT ===\n{pb.system_prompt}\n\n"
                + (f"=== DEVELOPER PROMPT ===\n{pb.developer_prompt}\n\n" if pb.developer_prompt else "")
                + f"=== USER PROMPT ===\n{pb.user_prompt}\n")
        (self.prompt_dir / fname).write_text(body, encoding="utf-8")
        t0 = time.time()
        result = await self.inner.run(request)
        entry = {
            "file": fname, "iteration": iteration, "role": role,
            "model": request.model, "reasoning_effort": request.reasoning_effort,
            "allow_tools": request.allow_tools, "timeout_seconds": request.timeout_seconds,
            "prompt_sha256": sha256_text(body), "prompt_chars": len(body),
            "user_prompt_has_execution_report": "=== CODE EXECUTION REPORT ===" in pb.user_prompt,
            "started_at": _dt.datetime.fromtimestamp(t0).isoformat(),
            "wall_seconds": round(time.time() - t0, 3),
            "success": getattr(result, "success", None),
            "error": getattr(result, "error", None),
            "input_tokens": getattr(result, "input_tokens", None),
            "output_tokens": getattr(result, "output_tokens", None),
            "cli_version": getattr(result, "cli_version", None),
        }
        with open(self.prompt_dir / "index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return result


# ── Provenance helpers ────────────────────────────────────────────────────


def _md5(path: Path) -> Optional[str]:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _cmd(argv: list[str], cwd: Optional[Path] = None, timeout: int = 60) -> str:
    try:
        out = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                             timeout=timeout, stdin=subprocess.DEVNULL)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def agents_md_on_path(workspace: Path) -> list[str]:
    """AGENTS.md files Codex would pick up between the git root and ``workspace``."""
    found: list[str] = []
    root = _cmd(["git", "rev-parse", "--show-toplevel"], cwd=REPO)
    try:
        root_p = Path(root).resolve()
        cur = workspace.resolve()
        chain = [cur] + list(cur.parents)
        for d in chain:
            for name in ("AGENTS.md", "AGENTS.override.md"):
                if (d / name).is_file():
                    found.append(str(d / name))
            if d == root_p:
                break
    except Exception:  # noqa: BLE001
        pass
    return found


def provenance(spec: RunSpec, paths: Paths, task: str, cfg: dict[str, Any]) -> dict[str, Any]:
    codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    return {
        "spec": dataclasses.asdict(spec),
        "run_id": spec.run_id,
        "task_variant": task_variant(spec),
        "task_sha256": sha256_text(task),
        "config": cfg,
        "loop_preset": "reviewer_consensus",
        "max_iterations": MAX_ITERATIONS,
        "codex_cli_version": _cmd(["codex", "--version"]),
        "codex_home": str(codex_home),
        "codex_global_agents_md_md5": _md5(codex_home / "AGENTS.md"),
        "project_agents_md_on_path": agents_md_on_path(paths.workspace(spec)),
        "kit_dir": str(KIT_DIR),
        "kit_manifest_md5": _md5(KIT_DIR / "kit_manifest.json"),
        "kit_readme_md5": _md5(KIT_DIR / "README.md"),
        "analysis_python": ANALYSIS_PYTHON if spec.execution else None,
        "analysis_python_version": (_cmd([ANALYSIS_PYTHON, "-c",
                                          "import sys,numpy,scipy,pandas,sklearn,statsmodels;"
                                          "print(sys.version.split()[0],numpy.__version__,"
                                          "scipy.__version__,pandas.__version__,"
                                          "sklearn.__version__,statsmodels.__version__)"])
                                    if spec.execution else None),
        "platform_python": sys.version,
        "git_head": _cmd(["git", "rev-parse", "HEAD"], cwd=REPO),
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO),
        "git_dirty_files": len(_cmd(["git", "status", "--porcelain"], cwd=REPO).splitlines()),
        "script_md5": _md5(Path(__file__)),
        "recorded_at": _dt.datetime.now().isoformat(),
    }


# ── DB maintenance: resume / repair ──────────────────────────────────────

NON_REPAIRABLE = ("failed:invalid_config", "failed:unknown_preset", "failed:start")


def _move_iter_dirs_above(run_dir: Path, last_good: int, tag: str) -> list[str]:
    moved: list[str] = []
    if not run_dir.is_dir():
        return moved
    dest_root = run_dir / "x4_failed_attempts" / tag
    for d in sorted(run_dir.glob("iter_*")):
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if not m or not d.is_dir() or int(m.group(1)) <= last_good:
            continue
        dest_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest_root / d.name))
        moved.append(d.name)
    return moved


async def prepare_resume(spec: RunSpec, paths: Paths, *, repair_failed: bool) -> dict[str, Any]:
    """Bring an existing run row into a resumable state.

    Returns ``{"action": ...}`` with ``action`` in ``new`` (no row),
    ``completed`` (nothing to do), ``resume`` (interrupted or repaired) or
    ``failed`` (failed and not repaired)."""
    from backend.database import get_db, get_run
    from backend.models import IterationStatus, RunStatus

    run = await get_run(spec.run_id)
    if run is None:
        return {"action": "new"}
    if run.status == RunStatus.COMPLETED:
        return {"action": "completed", "stop_reason": run.stop_reason}
    if run.status == RunStatus.FAILED:
        if not repair_failed or (run.stop_reason or "").startswith(NON_REPAIRABLE):
            return {"action": "failed", "stop_reason": run.stop_reason, "error": run.error}
    its = sorted(run.iterations, key=lambda i: i.iteration_number)
    keep = list(its)
    while keep and keep[-1].status != IterationStatus.COMPLETED:
        keep.pop()
    last_good = keep[-1].iteration_number if keep else 0
    dropped = [it.iteration_number for it in its if it.iteration_number > last_good]
    new_status = RunStatus.PAUSED if last_good > 0 else RunStatus.PENDING
    async with get_db() as db:
        await db.execute("DELETE FROM iterations WHERE run_id=? AND iteration_number>?",
                         (spec.run_id, last_good))
        await db.execute(
            "UPDATE runs SET status=?, current_iteration=?, stop_reason=NULL, error=NULL, "
            "stopped_at=NULL WHERE run_id=?",
            (new_status.value, last_good, spec.run_id))
        await db.commit()
    tag = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    moved = _move_iter_dirs_above(paths.run_dir(spec), last_good, tag)
    record = {
        "time": _dt.datetime.now().isoformat(),
        "previous_status": run.status.value,
        "previous_stop_reason": run.stop_reason,
        "previous_error": (run.error or "")[:2000],
        "previous_current_iteration": run.current_iteration,
        "resumed_from_iteration": last_good,
        "dropped_iteration_rows": dropped,
        "moved_iteration_dirs": moved,
        "moved_to": f"x4_failed_attempts/{tag}" if moved else None,
        "new_status": new_status.value,
    }
    rd = paths.run_dir(spec)
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "x4_repairs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return {"action": "resume", **record}


# ── One run ───────────────────────────────────────────────────────────────

AdapterFactory = Callable[[Any], Any]


async def ensure_rows(spec: RunSpec, paths: Paths, task: str, cfg: dict[str, Any]) -> None:
    from backend.database import create_run, create_workspace, get_run, get_workspace
    from backend.models import ProviderName, RunState, WorkspaceConfig

    ws_path = paths.workspace(spec)
    ws_path.mkdir(parents=True, exist_ok=True)
    if await get_workspace(spec.workspace_id) is None:
        await create_workspace(WorkspaceConfig(
            id=spec.workspace_id, name=f"x4-{spec.key}", path=str(ws_path),
            providers_enabled=[ProviderName.CODEX_CLI],
            default_provider=ProviderName.CODEX_CLI, default_model=spec.model,
            budget_max_tokens_per_run=50_000_000, budget_max_cost_per_run=1_000_000.0,
            budget_max_iterations=MAX_ITERATIONS,
        ))
    if await get_run(spec.run_id) is None:
        await create_run(RunState(
            run_id=spec.run_id, workspace_id=spec.workspace_id,
            loop_preset="reviewer_consensus", task=task,
            provider=ProviderName.CODEX_CLI, model=spec.model,
            max_iterations=MAX_ITERATIONS, config=dict(cfg),
        ))


def _check_frozen(spec: RunSpec, paths: Paths, prov: dict[str, Any]) -> None:
    """Refuse to resume a run whose task text or config changed."""
    f = paths.run_dir(spec) / "x4_run_spec.json"
    if not f.is_file():
        return
    old = json.loads(f.read_text())
    if old.get("task_sha256") != prov["task_sha256"] or old.get("config") != prov["config"]:
        raise RuntimeError(
            f"{spec.key}: task text or run config changed since the run was created "
            f"({f}); move the run aside or restore the original settings")


async def run_one(spec: RunSpec, paths: Paths, *, adapter_factory: Optional[AdapterFactory] = None,
                  max_repairs: int = 4, repair_waits: tuple[float, ...] = (60, 300, 900, 1800),
                  kit_dir: Path | str = KIT_DIR, analysis_python: str = ANALYSIS_PYTHON,
                  build_pack: bool = True) -> dict[str, Any]:
    """Run (or resume) one case-study run to completion; return its final status."""
    from backend.database import get_run, init_db
    from backend.orchestrator import runner

    await init_db()
    task = build_task_text(spec.execution, kit_dir)
    cfg = run_config(spec, kit_dir, analysis_python, persist_root=paths.data / "exec_outputs")
    prov = provenance(spec, paths, task, cfg)
    _check_frozen(spec, paths, prov)

    run_dir = paths.run_dir(spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_file = run_dir / "x4_run_spec.json"
    if not spec_file.is_file():
        spec_file.write_text(json.dumps(prov, indent=2))
        (run_dir / "x4_task.md").write_text(task, encoding="utf-8")
    else:
        with open(run_dir / "x4_invocations.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": prov["recorded_at"], "git_head": prov["git_head"],
                                 "codex_cli_version": prov["codex_cli_version"],
                                 "script_md5": prov["script_md5"]}) + "\n")

    events_path = run_dir / "x4_events.jsonl"

    async def broadcast(run_id: str, payload: dict) -> None:
        if run_id != spec.run_id:
            return
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": _dt.datetime.now().isoformat(), **payload},
                                default=str) + "\n")

    real_get_adapter = runner.get_adapter
    prompt_dir = run_dir / "x4_prompts"

    def patched_get_adapter(provider):
        inner = adapter_factory(provider) if adapter_factory else real_get_adapter(provider)
        return RecordingAdapter(inner, prompt_dir)

    repairs = 0
    status: dict[str, Any] = {}
    runner.set_ws_broadcast(broadcast)
    runner.get_adapter = patched_get_adapter
    try:
        while True:
            prep = await prepare_resume(spec, paths, repair_failed=True)
            logger.info("%s: %s", spec.key, prep.get("action"))
            if prep["action"] == "completed":
                break
            if prep["action"] == "failed":
                logger.error("%s: failed and not repairable: %s", spec.key, prep)
                break
            if prep["action"] == "new":
                await ensure_rows(spec, paths, task, cfg)
            await runner.execute_run(spec.run_id)
            run = await get_run(spec.run_id)
            logger.info("%s: status=%s stop_reason=%s iterations=%s", spec.key,
                        run.status.value, run.stop_reason, run.current_iteration)
            if run.status.value == "completed":
                break
            if run.status.value != "failed" or (run.stop_reason or "").startswith(NON_REPAIRABLE):
                break
            if repairs >= max_repairs:
                logger.error("%s: giving up after %d repairs (%s)", spec.key, repairs,
                             run.stop_reason)
                break
            wait = repair_waits[min(repairs, len(repair_waits) - 1)] if repair_waits else 0
            repairs += 1
            logger.warning("%s: failed (%s: %s); repair %d/%d in %.0fs", spec.key,
                           run.stop_reason, (run.error or "")[:300], repairs, max_repairs, wait)
            if wait:
                await asyncio.sleep(wait)
    finally:
        runner.get_adapter = real_get_adapter
        runner.set_ws_broadcast(None)

    run = await get_run(spec.run_id)
    status = summarize_run(spec, paths, run)
    if build_pack and run is not None and run.status.value == "completed":
        status["repropack"] = str(build_repropack(spec, paths))
    return status


def summarize_run(spec: RunSpec, paths: Paths, run) -> dict[str, Any]:
    if run is None:
        return {"key": spec.key, "run_id": spec.run_id, "status": "absent"}
    its = sorted(run.iterations, key=lambda i: i.iteration_number)
    return {
        "key": spec.key, "run_id": spec.run_id, "model": spec.model, "effort": spec.effort,
        "execution": spec.execution, "status": run.status.value,
        "stop_reason": run.stop_reason, "current_iteration": run.current_iteration,
        "total_input_tokens": run.total_input_tokens,
        "total_output_tokens": run.total_output_tokens, "error": run.error,
        "run_dir": str(paths.run_dir(spec)),
        "iterations": [{
            "n": it.iteration_number, "role": it.role, "status": it.status.value,
            "grade": it.grade, "duration_s": it.duration_seconds,
            "code_execution": it.code_execution,
            "consensus_grade_valid": ((it.consensus_report or {}).get("grade_valid")
                                      if it.consensus_report else None),
        } for it in its],
    }


# ── Reproducibility package ───────────────────────────────────────────────


def build_repropack(spec: RunSpec, paths: Paths) -> Path:
    """``backend.repropack`` package, extended with the iteration directories,
    prompts, events, task text and the case-study scripts (the generator only
    collects top-level run-directory files)."""
    from backend.repropack.generator import ReproPackGenerator

    run_dir = paths.run_dir(spec)
    paths.repropacks.mkdir(parents=True, exist_ok=True)
    out = paths.repropacks / f"{spec.key}.zip"
    data = ReproPackGenerator().generate(str(run_dir), str(paths.workspace(spec)))
    tmp = out.with_suffix(".zip.tmp")
    tmp.write_bytes(data)
    added: list[str] = []
    with zipfile.ZipFile(tmp, "a", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob("*")):
            if not f.is_file() or f.name.startswith("._"):
                continue
            rel = f.relative_to(run_dir).as_posix()
            arc = f"run_dir/{rel}"
            zf.write(f, arc)
            added.append(arc)
        exec_root = paths.exec_outputs(spec)
        if exec_root.is_dir():
            for f in sorted(exec_root.rglob("*")):
                if f.is_file() and not f.name.startswith("._"):
                    arc = f"exec_outputs/{f.relative_to(exec_root).as_posix()}"
                    zf.write(f, arc)
                    added.append(arc)
        for src, arc in ((Path(__file__), "case_study/run_case_study.py"),
                         (HERE / "trace_numbers.py", "case_study/trace_numbers.py"),
                         (HERE / "README.md", "case_study/README.md"),
                         (KIT_DIR / "README.md", "kit/README.md"),
                         (KIT_DIR / "kit_manifest.json", "kit/kit_manifest.json")):
            if src.is_file():
                zf.write(src, arc)
                added.append(arc)
        analysis_env = {"python": ANALYSIS_PYTHON,
                        "pip_freeze": _cmd([ANALYSIS_PYTHON, "-m", "pip", "freeze"],
                                           timeout=120).splitlines()} if spec.execution else {}
        zf.writestr("metadata/analysis_environment.json", json.dumps(analysis_env, indent=2))
        zf.writestr("metadata/x4_package_index.json", json.dumps({
            "built_at": _dt.datetime.now().isoformat(), "run_id": spec.run_id,
            "extended_entries": added,
            "note": "backend.repropack output plus run_dir/ (all iteration directories, "
                    "prompts, events), exec_outputs/ (complete outputs of every execution "
                    "unit: code.py, stdout.txt, stderr.txt, work/; CODE_EXECUTION.json "
                    "blocks[].persisted_to names the unit), case_study/ scripts and kit/ "
                    "provenance; the kit arrays themselves are not included (see "
                    "kit/kit_manifest.json md5s)",
        }, indent=2))
    tmp.replace(out)
    return out


# ── Sandbox smoke test ───────────────────────────────────────────────────

SMOKE_CODE = '''```python
import json, os, sys, time
t0 = time.time()
import numpy as np, scipy, pandas as pd, sklearn, statsmodels
KIT = {kit!r}
genes = pd.read_csv(os.path.join(KIT, "genes.tsv"), sep="\\t")
A = np.load(os.path.join(KIT, "attention_layer_mean.npy"), mmap_mode="r")
H = np.load(os.path.join(KIT, "attention_heads_f32.npy"), mmap_mode="r")
head = np.asarray(H[0, 0])
edges = pd.read_csv(os.path.join(KIT, "trrust_edges.tsv"), sep="\\t")
denied = {{}}
import pwd
real_home = pwd.getpwuid(os.getuid()).pw_dir
for label, target in (("results_dir", {results!r}), ("home_codex_auth", os.path.join(real_home, ".codex", "auth.json"))):
    try:
        open(target).read(1)
        denied[label] = False
    except Exception as exc:
        denied[label] = type(exc).__name__
try:
    open(os.path.join(KIT, "x4_write_probe.txt"), "w").write("x")
    denied["kit_write"] = False
except Exception as exc:
    denied["kit_write"] = type(exc).__name__
try:
    import socket
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    denied["network"] = False
except Exception as exc:
    denied["network"] = type(exc).__name__
res = {{"python": sys.version.split()[0], "numpy": np.__version__, "n_genes": int(len(genes)),
       "layer_mean_shape": list(A.shape), "heads_shape": list(H.shape),
       "n_trrust_edges": int(len(edges)), "head00_finite": bool(np.isfinite(head).any()),
       "denied": denied, "seconds": round(time.time() - t0, 2)}}
print(json.dumps(res))
json.dump(res, open("results.json", "w"))
```'''


async def smoke_sandbox(paths: Paths) -> dict[str, Any]:
    """Run a tiny script with EXACTLY the run config of the executed runs."""
    from backend.orchestrator.code_executor import CodeExecutor

    cfg = run_config(RUN_SPECS["sol_A"], persist_root=paths.smoke / "exec_outputs")
    ex = CodeExecutor.from_run_config(cfg)
    code = SMOKE_CODE.format(kit=str(KIT_DIR),
                             results=str(REPO / "experiments/case_study/results/SUMMARY.md"))
    report = await ex.run_artifact(code)
    out = {"time": _dt.datetime.now().isoformat(), "config": ex.config_dict(),
           "summary": report.summary(), "report": report.to_dict(),
           "feedback": report.to_feedback()}
    paths.smoke.mkdir(parents=True, exist_ok=True)
    (paths.smoke / "sandbox_smoke.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# ── Status ────────────────────────────────────────────────────────────────


async def status_all(paths: Paths, keys: list[str]) -> list[dict[str, Any]]:
    from backend.database import get_run, init_db
    await init_db()
    out = []
    for k in keys:
        spec = RUN_SPECS[k]
        out.append(summarize_run(spec, paths, await get_run(spec.run_id)))
    return out


# ── CLI ───────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def exclusive_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit(f"another run_case_study.py holds {path}; not starting a second one")
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def _setup_logging(paths: Paths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(paths.logs / "x4_runs.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def _keys(arg: Optional[str]) -> list[str]:
    keys = [k.strip() for k in (arg or ",".join(DEFAULT_ORDER)).split(",") if k.strip()]
    bad = [k for k in keys if k not in RUN_SPECS]
    if bad:
        raise SystemExit(f"unknown run key(s): {bad}; choose from {list(RUN_SPECS)}")
    return keys


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("print-task")
    p.add_argument("--variant", choices=["executed", "plan_only"], default="executed")
    sub.add_parser("smoke-sandbox")
    p = sub.add_parser("run")
    p.add_argument("--runs", default=None, help=f"comma-separated keys (default {DEFAULT_ORDER})")
    p.add_argument("--max-repairs", type=int, default=4)
    p.add_argument("--no-trace", action="store_true",
                   help="do not run trace_numbers.py after the last run")
    p = sub.add_parser("status")
    p.add_argument("--runs", default=None)
    p = sub.add_parser("repack")
    p.add_argument("--runs", default=None)
    args = ap.parse_args(argv)

    os.environ.setdefault("TMPDIR", TMPDIR_DEFAULT)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    paths = Paths(Path(args.data_dir))

    if args.cmd == "print-task":
        sys.stdout.write(build_task_text(args.variant == "executed"))
        return 0

    paths.data.mkdir(parents=True, exist_ok=True)
    setup_backend(paths.db)

    if args.cmd == "smoke-sandbox":
        out = asyncio.run(smoke_sandbox(paths))
        print(out["feedback"])
        print(f"saved: {paths.smoke / 'sandbox_smoke.json'}")
        return 0 if out["summary"]["passed"] == 1 else 1

    if args.cmd == "status":
        print(json.dumps(asyncio.run(status_all(paths, _keys(args.runs))), indent=2, default=str))
        return 0

    if args.cmd == "repack":
        for k in _keys(args.runs):
            print(build_repropack(RUN_SPECS[k], paths))
        return 0

    # run
    keys = _keys(args.runs)
    _setup_logging(paths)
    with exclusive_lock(paths.lock):
        logger.info("X4 chain start: %s (pid %d)", keys, os.getpid())
        summaries = []
        for k in keys:
            t0 = time.time()
            st = asyncio.run(run_one(RUN_SPECS[k], paths, max_repairs=args.max_repairs))
            st["wall_seconds_this_invocation"] = round(time.time() - t0, 1)
            summaries.append(st)
            logger.info("X4 run %s finished: %s %s", k, st.get("status"), st.get("stop_reason"))
            (paths.data / f"status_{k}.json").write_text(json.dumps(st, indent=2, default=str))
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "run_status.json").write_text(json.dumps(summaries, indent=2, default=str))
        if not args.no_trace:
            try:
                import trace_numbers  # noqa: E402  (same directory)
            except ImportError:
                sys.path.insert(0, str(HERE))
                import trace_numbers  # type: ignore
            out = trace_numbers.main(["--data-dir", str(paths.data)])
            logger.info("trace_numbers exit %s", out)
        logger.info("X4 chain done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
