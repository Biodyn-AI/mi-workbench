# Orchestration experiments (X6, E9)

This directory answers three questions:

1. Does the merge code do what it is specified to do? (mock implementation check)
2. What does the orchestration layer itself cost per iteration when runs go through the real runner, database and artifact path? (mock backend)
3. How do concurrent real reviewer panels behave in wall time, latency, failures and retries? (Codex backend)

Question 1 is implementation verification only; it is not evidence about reviewer quality. Reviewer quality is measured on the planted-flaw benchmark in `experiments/benchmark` (X1/X2).

## Contents

| File | Produced by | What it holds |
|---|---|---|
| `mock_verification.json` | `scripts/run_experiments.py --only mock` | merge-semantics pass/fail checks, panel-size ablation, exact-string role overlap, mock loop trajectory |
| `overhead.json` | `scripts/run_experiments.py --only overhead` | runner-path control-plane overhead, two storage devices, mock delay 0 s and 1 s, integrity check after every batch |
| `replicates/overhead_run1_repeats3.json` | earlier run of the same command (3 repeats per level, no load-average fields) | run-to-run variability |
| `real_concurrency.json` | `scripts/real_concurrency.py` | K = 1, 2, 4, 8 concurrent 3-lens Codex panels: per-level summary, per-panel results and per-call records (no raw text) |
| `raw/real_concurrency_K<k>.jsonl` | same | one line per call attempt: raw model output, CLI stderr, parsed CLI events |
| `smoke/real_consensus_probe.json` | `scripts/real_consensus_probe.py` | one real panel (smoke test of the live path): raw outputs, tokens, model, CLI version |
| `figures/overhead.png`, `figures/real_concurrency.png` | `scripts/make_figures.py` | PNG previews; final PLOS TIFFs are made later |
| `figures/derived_summary.json` | same | the numbers quoted below, including overhead as a percentage of real call latency |

## Commands

Run from `automation/mi-workbench`.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="/Volumes/Crucial X6/tmp_miw"
PY=/Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python   # platform env (no numpy)
APY=/Users/ihorkendiukhov/anaconda3/bin/python                    # analysis env (matplotlib)

# 1 + 2: mock verification and overhead (about 8 minutes; overhead uses two storage devices)
$PY scripts/run_experiments.py \
    --storage internal_apfs=/private/tmp/miw_overhead \
    --storage "external_exfat=/Volumes/Crucial X6/tmp_miw/overhead_exfat"
#   options: --only {all,mock,overhead} --levels 1,2,5,10,20 --iterations 6 --delays 0,1.0
#            --repeats 5 --preset reviewer_consensus --keep-workdir

# 3: real-backend concurrency: 15 panels = 45 Codex calls (about 4 minutes)
$PY scripts/real_concurrency.py
#   --levels 1,2,4,8 --model gpt-5.6-luna --effort low --artifact experiments/benchmark/artifacts/A01-clean.md
#   --max-retries 2 --retry-base-delay 10 --lens-timeout 600 --pause 10 --force --dry-run

# smoke test of the live consensus path (3 calls)
$PY scripts/real_consensus_probe.py [--model gpt-5.5 --effort medium --artifact ...]

# figures (PNG previews) and derived numbers
$APY scripts/make_figures.py

# tests (fast; real and slow tests are opt-in with MIW_RUN_REAL=1 / MIW_RUN_SLOW=1)
$PY -m pytest backend/tests/test_experiments.py -q -p no:cacheprovider
```

## 1. Mock implementation verification (`mock_verification.json`)

**What is checked.** The merge is pinned to cross-lens `jaccard` at 0.5 (`EXPERIMENT_MERGE`), the deterministic platform code these checks target. The mock adapter returns fixed critique sets: three lenses with 3, 3 and 2 critiques at tier 1, one of them raised word for word by both the rigour and adversarial lenses. Ground truth comes from running each lens alone. `merge_semantics_checks` confirms five things:

- the merged panel contains each distinct critique string exactly once (7 = |exact union|);
- the shared critique carries both lenses in `raised_by`;
- the shared critique is escalated one level (HIGH to CRITICAL), because two distinct lenses meet the consensus threshold;
- single-lens critiques keep their severity;
- panel sizes 1, 2, 3 give 3, 5, 7 distinct critiques.

All five checks pass. A negative-control test (`test_merge_semantics_checks_detect_broken_escalation`) disables escalation and confirms that the checks then fail.

**Role overlap** compares critiques by exact string equality, so the field is `exact_string_set_overlap`; it is not a semantic or token-level similarity. Values: 0.2 / 0.0 / 0.0. These overlaps are a property of how the mock was written, not of real reviewers.

**Convergence profile** is a single deterministic trajectory (`n_runs = 1`). The mock's text depends only on its global call counter. Seeding changes only delays and token jitter, so repeated runs are copies, not samples; replays with seeds 0, 1, 2 are identical (`determinism_check.identical = true`). The trajectory stops at iteration 7 with `converged:output_similar`. Within a loop, the mock's quality tier advances per call, so the three lenses of one panel see different tiers. The trajectory therefore exercises the stopping code but carries no information about convergence with real models; that is X3.

## 2. Control-plane overhead through the real runner (`overhead.json`)

**Question.** How much wall time does the orchestration layer add per iteration? This covers loading the run, building prompts, parsing and merging critiques, persisting each iteration to SQLite, writing artifacts and `run_meta.json`. The question is asked with a mock backend so that everything else is zero (delay 0 s) or a known constant (delay 1 s).

**Method.**

- N runs (N = 1, 2, 5, 10, 20) of the `reviewer_consensus` loop are created with `create_run` and started together with `backend.orchestrator.runner.execute_run`, registered in the runner's task table as `POST /api/runs` does.
- Each run has 6 fixed iterations (convergence off): 3 executor calls and 3 three-lens panels, so 12 adapter calls. The merge is pinned to cross-lens `jaccard` at 0.5 (`backend.analysis.experiments.EXPERIMENT_MERGE`, recorded as `merge` in `overhead.json`), the platform default when these results were first recorded; with the current default (LLM adjudication) every panel would add one sequential adjudicator call.
- The database is a real SQLite file (WAL) and the artifacts are real files in a temporary workspace.
- 5 repeats per level, after one untimed warm-up run.
- Every adapter call is timed per run. The conservative overhead used in the figure is (run wall time − 6 × mock delay) / 6. It counts event-loop stalls as overhead. The `overhead_ms_per_iteration` field instead subtracts the measured adapter intervals.
- After every batch, `check_integrity` checks:
  - `PRAGMA integrity_check` returns `ok` and `PRAGMA foreign_key_check` is empty;
  - there are no orphaned iteration or run rows and no orphaned run directories;
  - every run's iteration rows are numbered 1..6 without duplicates and match `runs.current_iteration`, the `iter_NNNN` directories on disk (each holding `<role>_output.md`) and `run_meta.json`.
- The sweep was run on the internal SSD (APFS) and on the external USB drive that holds the repository (exFAT, served on this macOS version by a user-space file-system driver).

**Results** (p50 over runs, conservative definition; `figures/derived_summary.json`):

| storage | mock delay | N = 1 | N = 5 | N = 20 | throughput at N = 20 |
|---|---|---|---|---|---|
| internal APFS | 0 s | 18 ms/it | 53 ms/it | 241 ms/it | 87 it/s |
| internal APFS | 1 s | 26 ms/it | 76 ms/it | 153 ms/it | 17.1 it/s |
| external exFAT | 0 s | 24 ms/it | 42 ms/it | 152 ms/it | 89 it/s |
| external exFAT | 1 s | 120 ms/it | 303 ms/it | 1113 ms/it | 9.0 it/s |

- All 104 integrity checks passed: 2 storages × 2 delays × (1 warm-up + 25 batches). At the end each database held 191 runs and 1142 iterations, consistent between DB rows and disk.
- Process CPU time was 4–21 ms per iteration across all settings. Most of the remaining overhead is waiting, not computing.

**Interpretation.**

- *Fixed cost per iteration.* With one run and a zero-latency backend, an iteration costs about 18–24 ms of orchestration. Most of that is SQLite connection handling and commits; a profiling run on APFS showed the iteration-persist transaction plus connection open/close as the largest items.
- *Ceiling under concurrency.* The aggregate ceiling is about 75–95 iterations/s, and it does not rise with more runs. The runs share one event loop, and connection open/close is serialised: a workaround for the SQLite 3.51.1 deadlock in this environment, see `backend/database.py`. Per-run latency therefore grows roughly linearly with N when the backend is free.
- *With a backend that takes time.* With a 1-s backend, overhead stays small on the internal SSD (about 0.4 % of a real reviewer call at N = 20).
- *On the external exFAT drive* it grows to about 1.1 s per iteration at N = 20. Artifact writes are synchronous on the event loop, and each costs milliseconds on this drive, so stalls delay every run's resumption. The delay-1 s numbers on exFAT are higher than the delay-0 s numbers in both replicates; idle gaps between writes appear to make the USB drive slower per operation. That is an inference from the pattern, not a measurement.

**Relative to real model latency.** The median real reviewer call (gpt-5.6-luna, effort low, K = 1) took 36.9 s. Conservative orchestration overhead per iteration is:

- internal SSD: 0.05–0.07 % of one call at N = 1 and ≤ 0.65 % at N = 20;
- external exFAT drive: up to 3.0 % at N = 20 with the 1-s backend.

The control plane is not the bottleneck for LLM-driven runs; provider latency is.

**Limitations.**

- The machine was shared with other jobs during all measurements. The 1-minute load average was 120–185 on 10 cores; see `loadavg_start` per level. The earlier replicate (`replicates/`) gave 30 ms/it on APFS and 115 ms/it on exFAT at N = 1, delay 0 s. Absolute numbers are therefore upper bounds with large run-to-run variance on the external drive; the order of magnitude and the comparison with LLM latency are robust.
- The mock backend makes no network calls, and its outputs are smaller than real ones: real lens outputs were 3.8–8.3 kB.
- One machine, one Python process.

## 3. Real-backend concurrency (`real_concurrency.json`)

**Question.** When several reviewer panels run at once against a real provider, what happens to panel wall time, per-call latency and failure rates, and does the platform's retry path behave?

**Method.**

- K = 1, 2, 4, 8 panels started together, one level after another with a 10-s pause, giving 15 panels and 45 calls.
- Each panel is one `ConsensusReviewer.run_panel` call (rigour, adversarial, biological-plausibility lenses) over the clean benchmark artifact A01-clean.md (6.9 kB).
- Backend: one shared Codex adapter, gpt-5.6-luna, effort low, `allow_tools=False` (read-only sandbox), web search off, empty working directory per panel.
- Prompts are identical to X1.
- Retries use the panel's own `run_with_retry`: up to 2 retries on transient errors, 10 s base backoff. Every attempt is recorded.

**Results.**

| K | panels | calls (attempts) | retries / failures | call latency p50 (p95) | panel wall p50 | throughput |
|---|---|---|---|---|---|---|
| 1 | 1 | 3 (3) | 0 / 0 | 36.9 s (39.5) | 39.5 s | 4.6 calls/min |
| 2 | 2 | 6 (6) | 0 / 0 | 36.0 s (40.8) | 40.8 s | 8.8 calls/min |
| 4 | 4 | 12 (12) | 0 / 0 | 52.7 s (57.9) | 55.6 s | 12.4 calls/min |
| 8 | 8 | 24 (24) | 0 / 0 | 38.6 s (45.0) | 44.3 s | 30.6 calls/min |

- All 45 calls succeeded on the first attempt, with no rate-limit errors, timeouts or retries.
- All outputs parsed as JSON. No agent tool events occurred.
- Per call: about 14.4 k input tokens, 1.5 k output tokens (about 320 of them reasoning). The cached share of the input grew with concurrency, from 0 to 6.8 k per call at K = 8.
- Every panel reached grade D with 27–32 merged critiques.
- No critique was merged across lenses (`multi_reviewer = 0` in all 15 panels): the Jaccard 0.5 merge (the platform default at the time; pinned in `real_concurrency.py`) found no duplicates in real reviewer text. X2 evaluated this question; the platform default is now LLM adjudication, with calibrated thresholds of 0.1 for the lexical methods (`experiments/benchmark/results/merge_eval.json`).

**Interpretation.**

- Throughput scaled close to linearly up to K = 8: 30.6 vs an ideal 36.4 calls/min, which is 84 %. The panel is network/provider-bound, and the orchestrator adds no measurable queueing at this scale.
- The slower K = 4 level (call p50 52.7 s) is not monotonic in K: K = 8 was faster again. With one batch per level, we read it as provider or CLI latency variance rather than a concurrency effect.
- Every Codex CLI call logged a non-fatal models-cache error on stderr. At K = 4 each call also logged "failed to refresh available models: timeout waiting for child process to exit" about 5 s after start.

**Tail event (smoke probe, run separately right after).** In `smoke/real_consensus_probe.json`, two of three calls took about 365 s instead of about 37 s. Their stderr ends with a Codex models-cache error logged at the end of the stall. The calls still succeeded. Such CLI-side stalls are bounded by the lens timeout (default 900 s; 600 s here) and are not caused by the orchestrator, but they dominate panel wall time when they occur.

**Limitations.**

- One provider, model and effort; one artifact; one batch per level.
- K ≤ 8, so 24 simultaneous CLI processes. Provider rate limits were not reached, so this run does not show where they start.
- Codex prepends its own harness instructions and the user-level `AGENTS.md`; hashes are recorded in `environment.codex_env`.
- Scalability claims should be limited to what is shown: no failures and near-linear throughput up to 8 concurrent panels.
