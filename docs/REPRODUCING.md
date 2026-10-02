# Reproducing the MI-Workbench results

This file explains how to rebuild the environment, run the test suite and
re-run each experiment reported in the PLOS ONE manuscript.

All commands run from the repository root.

## 1. Environment

MI-Workbench needs Python 3.11 or newer. The test suite was last run with
Python 3.11.14. There are three requirement files:

| File | Purpose |
|------|---------|
| `requirements.txt` | Core runtime for the backend and CLI (pure Python). |
| `requirements-dev.txt` | Test suite: pytest, pytest-asyncio, httpx, hypothesis, pytest-cov. It includes `requirements.txt`. |
| `requirements-analysis.txt` | Optional evaluation and figure scripts: numpy, scipy, pandas, scikit-learn, matplotlib, statsmodels. |

With conda:

```bash
conda create -n mi_workbench python=3.11
conda activate mi_workbench
pip install -r requirements-dev.txt
pip install -r requirements-analysis.txt   # only for the analysis scripts
```

With a plain virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Runs that use real providers also need the matching agent CLI installed and
authenticated: `claude` (Claude Code), `codex` (Codex CLI) or `gemini`
(Gemini CLI). The mock provider needs neither a CLI nor network access.

## 2. Test suite

```bash
python -m pytest -q
```

`pytest.ini` sets `testpaths = backend/tests` and strict asyncio mode. The
suite should finish with no failures, and the process should exit by itself
once the summary is printed. The tests make no network or model calls.

On macOS with the repository on an external volume, you can point `TMPDIR`
at the same volume and turn off bytecode caching:

```bash
TMPDIR=/path/on/volume/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

## 3. Codebase statistics (Table 3)

These are the exact commands used for Table 3. Re-run them on the frozen
revision; the figures change whenever the code does.

```bash
# Backend source files, excluding tests
find backend -name '*.py' -not -path '*/tests/*' -not -path '*/__pycache__/*' -not -name '._*' | wc -l
# Backend lines of code, excluding tests
find backend -name '*.py' -not -path '*/tests/*' -not -path '*/__pycache__/*' -not -name '._*' -print0 | xargs -0 cat | wc -l
# Test files, and test functions (should equal the number pytest collects)
ls backend/tests/test_*.py | wc -l
python -m pytest --collect-only -q | tail -1
# Test lines of code
find backend/tests -maxdepth 1 -name '*.py' -not -name '._*' -print0 | xargs -0 cat | wc -l
# Prompt templates
find prompts -name '*.yaml' -not -name '._*' | wc -l
# API route decorators, excluding tests
grep -rhE '^\s*@(router|app)\.(get|post|put|delete|patch|websocket)\(' backend --include='*.py' --exclude-dir=tests | wc -l
```

## 4. Experiments

Every experiment caches its raw model outputs. A results JSON drives each
table and figure. Model identifiers, CLI versions and reasoning-effort
settings are stored with the outputs. Each experiment's README is the full
reference; the commands below are copied from it.

Common setup (from `automation/mi-workbench`; the repository lives on an
exFAT drive, hence no bytecode and a `TMPDIR` on the same drive):

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="/Volumes/Crucial X6/tmp_miw"
PY=/Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python   # platform env (no numpy)
APY=/Users/ihorkendiukhov/anaconda3/bin/python                    # analysis env (numpy, matplotlib, ...)
```

Raw data goes to `experiments_data/` (gitignored). Real-model runs need an
authenticated Codex CLI.

### 4.1 Planted-flaw critique benchmark (X1)

Reference: `experiments/benchmark/README.md`; contract: `experiments/benchmark/ANALYSIS_PLAN.md`.

- Inputs: `experiments/benchmark/` (artifacts, ground truth, frozen analysis plan)
- Codex condition: `run_reviews.py --codex-tools-off web_only` (default; the
  frozen pilot condition: web search disabled, the user's `config.toml` and
  `$CODEX_HOME/AGENTS.md` still apply) or `strict` (no shell, no
  apps/plugins/MCP servers, `config.toml` ignored). The tools-off arguments
  and the `AGENTS.md` hashes of the Codex home are part of the cache
  signature: records made under another condition are reported as
  `CACHE_MISMATCH` and never reused, and `analyze.py` excludes units whose
  records mix conditions. A reply without a usable critique block (no
  recognisable format, an echoed template, a truncated block) is a failed
  attempt and is retried; it never counts as a clean review.

Commands:

```bash
# 1. freeze (verify the frozen benchmark against benchmark_manifest.json)
$PY experiments/benchmark/freeze.py --check

# 2. reviewer calls (8 calls per model x item x repeat; resumable, cached calls are skipped)
#    The reported run (experiments_data/benchmark_runs_v1, strict tools-off as recorded in
#    every record's request.cli_extra_args), one process per model:
$PY experiments/benchmark/run_reviews.py --models gpt-5.6-sol --effort medium --repeats 3 \
    --concurrency 3 --codex-tools-off strict --data-dir experiments_data/benchmark_runs_v1
$PY experiments/benchmark/run_reviews.py --models gpt-5.5 --effort medium --repeats 3 \
    --concurrency 2 --codex-tools-off strict --data-dir experiments_data/benchmark_runs_v1
$PY experiments/benchmark/run_reviews.py --models gpt-5.6-luna --effort low --repeats 3 \
    --concurrency 2 --codex-tools-off strict --data-dir experiments_data/benchmark_runs_v1

# 3. judge (two judges, effort high; experiments/benchmark/judge_loop.sh repeats this
#    until every complete unit is judged)
$PY experiments/benchmark/judge.py --judge-models gpt-5.6-sol,gpt-5.5 --effort high \
    --concurrency 2 --codex-tools-off strict --data-dir experiments_data/benchmark_runs_v1

# 4. analysis -> experiments/benchmark/results/ (benchmark_results.json + CSV tables)
$APY experiments/benchmark/analyze.py --judges gpt-5.6-sol,gpt-5.5 \
    --data-dir experiments_data/benchmark_runs_v1
#    sensitivity (TF-IDF merge at the all-data best threshold) -> results/sensitivity_tfidf/
$APY experiments/benchmark/analyze.py --judges gpt-5.6-sol,gpt-5.5 \
    --data-dir experiments_data/benchmark_runs_v1 --merge-method tfidf --merge-threshold 0.075 \
    --out-dir experiments/benchmark/results/sensitivity_tfidf

# tests (fake adapters, no model calls)
$PY -m pytest experiments/benchmark/test_benchmark_harness.py -q -p no:cacheprovider
```

- Outputs: `experiments_data/benchmark_runs_v1/{reviews,judgments,logs}/`
  (raw calls and judgments) and `experiments/benchmark/results/`
  (`benchmark_results.json`, `recall_by_condition.csv`, `recall_by_family.csv`,
  `recall_by_type.csv`, `severity_recall_by_condition.csv`, `contrasts.csv`,
  `discrimination.csv`, `false_alarms_clean.csv`, `unmatched_composition.csv`,
  `cost_by_condition.csv`, `call_stats.csv`, `per_unit_*.csv`).
- Merge of the severity outcomes: pinned to the platform merge's default at
  freeze time (`jaccard`, threshold 0.5), as the analysis plan specifies;
  `analyze.py` does not follow the later platform defaults (see 4.2). Re-running
  the analysis command above reproduces every CSV in `results/` byte for byte.
- Size: 3 models x 18 items x 3 repeats x 8 calls = 1,296 reviewer calls, plus
  2 judges x 162 units = 324 judge calls. The analysis runs in about 1.5 min.

### 4.2 Merge evaluation (X2, offline on X1 raw outputs)

Reference: `experiments/benchmark/README.md`, section "5. Merge evaluation (X2)".

- `merge_eval.py` uses the platform defaults (`same_lens_merge=False`:
  cross-lens merging only). LLM adjudications are cached with a request
  signature (adjudicator, effort, adjudicator prompt hash, Codex arguments
  and `AGENTS.md` hashes) and their groups are re-derived from the cached raw
  answer with the current parser; repaired answers are reported
  (`repaired_instances`, metrics also without them) and invalid partitions
  count as failures. `default_threshold` in the output is each method's
  pre-calibration threshold (jaccard 0.5, tfidf 0.3, embedding 0.7), so a
  re-run reproduces the file whatever the current platform defaults are.

```bash
$APY experiments/benchmark/merge_eval.py --judge gpt-5.6-sol \
    --adjudicator-model gpt-5.6-sol --adjudicator-effort medium \
    --data-dir experiments_data/benchmark_runs_v1
```

- Outputs: `experiments/benchmark/results/merge_eval.json`; cached adjudications
  in `experiments_data/benchmark_runs_v1/merge_eval/llm/`.
- Result and platform consequence: held-out F1 llm 0.824, tfidf 0.814, jaccard
  0.804 (162 panel instances, 4,556 labelled cross-lens pairs). The platform
  default `consensus_similarity_method` is therefore `llm`, with a fallback to
  `tfidf` at 0.1 when an adjudication fails; the jaccard / tfidf default
  thresholds are 0.1 (folds selected 0.075–0.1; see the *Calibration* note in the
  top-level `README.md`).

### 4.3 Stopping-rule and grade calibration (X3)

Reference: `experiments/stopping/README.md`.

```bash
# tests (scripted adapters; ~15 s) and the engine feature's tests
$PY -m pytest experiments/stopping/test_stopping.py backend/tests/test_engine_seed.py -q -p no:cacheprovider

# mock smoke through the real engine (platform MockAdapter)
$PY experiments/stopping/run_loops.py --provider mock --items A01-flawed --data-dir "$TMPDIR/x3_mock" --backoff 0

# real: one unit, then everything (complete units are skipped)
$PY experiments/stopping/run_loops.py --configs sol-medium --items A01-flawed
$PY experiments/stopping/run_loops.py --configs sol-medium
$PY experiments/stopping/run_loops.py --configs luna-low
$PY experiments/stopping/run_loops.py --configs sol-medium,luna-low --status

# judge (after the loops) and replay
$PY experiments/stopping/judge_states.py --configs sol-medium --concurrency 3
$PY experiments/stopping/judge_states.py --status
$PY experiments/stopping/replay.py           # -> results/stopping_results.json, results/stopping_rules.csv

# everything, resumable, in the background
nohup experiments/stopping/run_all.sh > experiments_data/stopping_runs/logs/run_all.out 2>&1 &
```

- Outputs: `experiments_data/stopping_runs/<config>/<item>/` (engine artifacts,
  calls, events, `trajectory.json`), `experiments_data/stopping_runs/judgments/`
  and `experiments/stopping/results/{stopping_results.json,stopping_rules.csv}`.
- Merge: every trajectory uses cross-lens `jaccard` at 0.5 (`run_loops.X3_MERGE`,
  the platform default when X3 started), recorded per panel in `CONSENSUS.json`.
- Size: 2 configurations x 12 items x 11 engine iterations (6 three-lens panels +
  5 executor revisions) = 23 Codex calls per trajectory, plus 6 judge calls per
  trajectory. Units run one at a time (at most 3 calls in flight).
- Result and platform consequence: pooled over both configurations (24
  trajectories), the pre-specified selection (lowest premature-stop rate among
  rules that stop before the horizon in >= 50 % of runs, ties by fewer
  revisions) selects a fixed budget of four revisions (`fixed_k4`, premature-stop
  rate 2/24 = 0.083; the adaptive any-of-three rule: 3/24 = 0.125 at about three
  revisions). The platform default is therefore `revision_budget: 4` with
  `convergence_enabled: false` (top-level `README.md`, *Default stopping policy
  (calibrated)*). `backend/tests/test_revision_budget.py` checks that
  `revision_budget=k` on a seeded panel-first loop ends at the replay's `fixed_k`
  state E_k (plus the panel that reviews it).
- The fixed horizon needs the revision budget disabled: with the platform
  default (4) a seeded trajectory ends after P4 (stop reason `revision_budget`)
  instead of P5. A trajectory run config therefore has to set
  `revision_budget: null` (like `convergence_enabled: false`).

### 4.4 Executed case study (X4)

References: `experiments/case_study/README.md` (kit and independent validation)
and `experiments/case_study/agent_runs/README.md` (agent runs).

Analysis kit (separate model environment with torch, transformers and h5py,
not part of the requirement files above):

```bash
# kit (about 40 min on an M2 Pro when the GPU is shared; resumable, checkpoints every 100 cells)
mkdir -p "/Volumes/Crucial X6/tmp_miw"
export TMPDIR="/Volumes/Crucial X6/tmp_miw"; export OMP_NUM_THREADS=4
nohup /Users/ihorkendiukhov/anaconda3/envs/subproject02-evalbias-rev/bin/python -W ignore \
    experiments/case_study/build_kit.py >> experiments_data/case_study_kit/build_kit.log 2>&1 &
```

Agent runs (real platform runner, `reviewer_consensus` with LLM adjudication,
verified execution in `sandbox_exec`, persisted execution outputs):

```bash
# task texts
$PY experiments/case_study/agent_runs/run_case_study.py print-task --variant executed
$PY experiments/case_study/agent_runs/run_case_study.py print-task --variant plan_only

# sandbox smoke test with exactly the executed runs' config
$PY experiments/case_study/agent_runs/run_case_study.py smoke-sandbox

# launch / resume the whole chain (sol_A, sol_B, gpt55, sol_planonly), then trace_numbers
nohup caffeinate -i $PY experiments/case_study/agent_runs/run_case_study.py run \
    >> experiments_data/case_study_agent_runs/logs/x4_chain.out 2>&1 &
echo $! > experiments_data/case_study_agent_runs/logs/x4_chain.pid

# progress
$PY experiments/case_study/agent_runs/run_case_study.py status

# a subset / reproducibility packages again
$PY experiments/case_study/agent_runs/run_case_study.py run --runs gpt55,sol_planonly
$PY experiments/case_study/agent_runs/run_case_study.py repack --runs sol_A

# number tracing (also runs automatically at the end of `run`)
$PY experiments/case_study/agent_runs/trace_numbers.py

# tests (offline: fake adapter, real runner/engine/consensus/sandbox)
$PY -m pytest experiments/case_study/agent_runs/test_case_study_agent_runs.py -q -p no:cacheprovider
```

- Outputs: `experiments_data/case_study_kit/` (kit, `kit_manifest.json`);
  `experiments_data/case_study_agent_runs/` (`miw_case_study.db`,
  `workspaces/<key>/runs/x4_<key>/` with every iteration directory,
  `exec_outputs/<key>/` with the complete outputs of every execution,
  `repropacks/<key>.zip`); `experiments/case_study/agent_runs/results/`
  (`run_status.json`, `trace_summary.json`, `trace_per_iteration.csv`,
  `trace_details/`, `SYNTHESIS.md`, `case_synthesis.json`).
- Reproducibility packages: `backend.repropack` now includes every iteration
  directory and the persisted execution outputs itself
  (`miw run repropack <run_id> [--max-file-mb N]`); `run_case_study.py repack`
  additionally adds the prompts, events, case-study scripts and kit README /
  manifest.

### 4.5 Independent computational validation (X5)

Reference: `experiments/case_study/README.md` (step 2; reads only the kit).

```bash
OMP_NUM_THREADS=4 /Users/ihorkendiukhov/anaconda3/bin/python experiments/case_study/independent_validation.py
```

- Outputs: `experiments/case_study/results/independent_validation.json`,
  `fig_case_validation.{png,pdf}`, `SUMMARY.md`. Deterministic (seed 20261001).

### 4.6 Orchestration overhead and concurrency (X6)

Reference: `experiments/orchestration/README.md`.

```bash
# 1 + 2: mock verification and overhead (about 8 minutes on an idle machine;
#        overhead uses two storage devices)
$PY scripts/run_experiments.py \
    --storage internal_apfs=/private/tmp/miw_overhead \
    --storage "external_exfat=/Volumes/Crucial X6/tmp_miw/overhead_exfat"
#   options: --only {all,mock,overhead} --levels 1,2,5,10,20 --iterations 6 --delays 0,1.0
#            --repeats 5 --preset reviewer_consensus --keep-workdir

# 3: real-backend concurrency: 15 panels = 45 Codex calls (about 4 minutes)
$PY scripts/real_concurrency.py

# smoke test of the live consensus path (3 calls)
$PY scripts/real_consensus_probe.py

# figures (PNG previews) and derived numbers
$APY scripts/make_figures.py

# tests
$PY -m pytest backend/tests/test_experiments.py -q -p no:cacheprovider
```

- Outputs: `experiments/orchestration/mock_verification.json`, `overhead.json`,
  `real_concurrency.json`, `raw/`, `smoke/real_consensus_probe.json`,
  `figures/`.
- Merge: these experiments pin cross-lens `jaccard` at 0.5
  (`backend.analysis.experiments.EXPERIMENT_MERGE`; the real-concurrency and
  probe scripts likewise), the setting of the recorded results, so every
  overhead iteration is one adapter round (no adjudicator call).
- The overhead run checks database / artifact integrity after every batch; the
  database layer retries writes on lock contention (busy timeout
  `MIW_DB_BUSY_TIMEOUT`, default 30 s, plus bounded retries), so a loaded host
  or a slow external drive no longer fails runs with "database is locked".

### 4.7 Figures

PLOS ONE figures (Fig2–Fig5, S1 Fig) from the results above, into
`paper/plos/figures/` (TIFF, PDF, PNG preview and `FIGURE_NOTES.md`):

```bash
scripts/make_plos_figures.sh                 # all figures + FIGURE_NOTES.md
scripts/make_plos_figures.sh --only Fig4     # a subset (notes file not rewritten)
```

Inputs: Fig2 `experiments/benchmark/results/` (benchmark CSVs and JSON); Fig3
`experiments/benchmark/results/merge_eval.json`; Fig4
`experiments/stopping/results/`; Fig5
`experiments/case_study/results/independent_validation.json`; S1
`experiments/orchestration/{overhead.json,real_concurrency.json}`. The script
uses the analysis interpreter (`PY=` overrides it) and puts matplotlib and temp
files under `/Volumes/Crucial X6/tmp_miw` (`MIW_SCRATCH`). A figure whose
inputs are missing or incomplete is skipped with a message. PNG previews of the
orchestration results alone: `$APY scripts/make_figures.py`.
