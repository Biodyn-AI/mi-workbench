# Stopping-rule and grade calibration with real loops (X3)

The contract is the "Stopping-rule calibration (real loops)" section of
`experiments/benchmark/ANALYSIS_PLAN.md`. The items are the 12 flawed benchmark artifacts
(`experiments/benchmark/artifacts/A??-flawed.md`); their planted flaws are in
`experiments/benchmark/ground_truth/`. `run_loops.py` checks the frozen benchmark manifest
before it starts.

## Design

Each trajectory is one run of the real platform engine
(`backend.orchestrator.engine.LoopEngine`) on the `reviewer_consensus` preset: a
three-lens consensus panel (rigour, adversarial, bio-plausibility) alternating with the
executor, through the Codex CLI adapter. The original write-up is passed as the run-config
value `seed_executor_output`, so the engine treats it as the executor's iteration-0
output. That makes a panel the first step. The horizon is fixed at 5 executor revisions:

```
iteration:  1   2   3   4   5   6   7   8   9   10  11
step:       P0  E1  P1  E2  P2  E3  P3  E4  P4  E5  P5
```

`P_j` is the panel reviewing executor state `E_j`; `E0` is the original flawed write-up.

### The one backend change: `seed_executor_output`

This is the only backend change, in `backend/orchestrator/engine.py`. It is tested in
`backend/tests/test_engine_seed.py` and documented in the run-config table of the top-level
`README.md`.

- The seed text becomes the latest executor submission:
  - it is the artifact under review for the first panel;
  - it is the "YOUR PREVIOUS SUBMISSION (iteration 0)" in the first revision request;
  - it is the first executor output for the `output_similar` signal.
- It does not count as an engine iteration.
- A fresh run starts at the plan step after the first executor step.
- On resume the seed is re-applied before the stored iterations are restored.
- A non-string value, or a loop with no executor step, ends the run with
  `failed:invalid_config`.

### Run configuration of every trajectory (`run_loops.run_config`)

| Key | Value | Why |
|---|---|---|
| `seed_executor_output` | the original write-up | start with P0 |
| `convergence_enabled` | `false` | no early stopping; `grade_at_least` unset |
| `budget_max_tokens` / `budget_max_cost` | 1e10 / 1e6 | budgets never fire |
| `max_iterations` (run field) | 11 | horizon: 6 panels + 5 revisions |
| `executor_allow_tools`, `reviewer_allow_tools` | `false` | Codex strict tools-off for every call: read-only sandbox, no shell, no web search, no apps, plugins or MCP, and `--ignore-user-config` |
| `reasoning_effort`, model | per config | passed explicitly on every call |
| `adapter_timeout`, `consensus_lens_timeout` | 1500 s | |
| `max_retries` | 3 | transient adapter errors (engine and panel) |
| `consensus_partial_policy` | `fail` (last try of a step: `no_stop`) | see "Resume" below |
| `consensus_gate` | `false` | gating is replayed offline |
| `consensus_similarity_method` / `consensus_similarity_threshold` | `jaccard` / 0.5 (`run_loops.X3_MERGE`) | the platform default when X3 started, pinned for every configuration (the platform default has since become LLM adjudication with calibrated thresholds, see `experiments/benchmark/results/merge_eval.json`). Added to the engine run config outside the cache signature, so existing units stay valid; recorded in each `CONSENSUS.json` and in `unit.json` (`platform.consensus_merge_used`; units created before the pin record `platform.consensus_similarity_method_default: jaccard`). Raw lens outputs are stored, so merging and grades can be recomputed offline with any method |

**Executor instructions.** These are the run's task text, `stopping_common.EXECUTOR_TASK`.
- The executor is revising a write-up in response to reviewer feedback.
- It cannot run analyses or access data.
- It may:
  - correct or withdraw claims;
  - rephrase or weaken conclusions;
  - add caveats and limitations;
  - fix technical statements;
  - specify analyses that must be run first, marked `[PENDING ANALYSIS]`.
- It must not report new numerical results or claim that new analyses were performed.
- It outputs the complete revised write-up every time.

The engine wraps this text in its own revision request: the feedback since the last
submission, the previous submission, and "Original task: …".

**Reviewers** get the platform's three lens prompts and the neutral review framing:
`=== REVIEW REQUEST ===`, the task as "context only", and `=== ARTIFACT UNDER REVIEW ===`
with the current state. They never see the revision framing, the previous submission or
earlier feedback. The tests check this, and so did the real smoke run.

**Isolation.** Each call runs in a fresh, empty working directory under
`$TMPDIR/miw_stop_ws`, outside the repository. The engine's workspace (the unit directory,
used for artifacts and resume) is never handed to the CLI. The user-level
`~/.codex/AGENTS.md` is still injected by Codex, as in X1; its hash is in the unit
signature.

**Concurrency.** Units run one at a time. A panel makes 3 concurrent calls and an executor
step makes 1, so this harness never has more than 3 calls in flight.

### Configurations

| Name | Model | Effort | Role |
|---|---|---|---|
| `sol-medium` | gpt-5.6-sol | medium | primary (executor and panel) |
| `luna-low` | gpt-5.6-luna | low | second configuration, run after the primary |

## Files

| Path | What it is |
|---|---|
| `stopping_common.py` | Constants, executor task text, trajectory geometry, item loading (the benchmark's `common.py`, loaded as `miw_bench_common`), new-number heuristic |
| `run_loops.py` | Runs and resumes the trajectories; builds `trajectory.json` |
| `judge_states.py` | One judge call per executor state E0..E5 |
| `prompts/judge_states.md` | Judge rubric (its system prompt) |
| `replay.py` | Offline rule replay, outcomes, correlations → `results/stopping_results.json`, `results/stopping_rules.csv` |
| `run_all.sh` | The resumable chain: loops (sol) → judge → replay → loops (luna) → judge → replay |
| `test_stopping.py` | Tests with scripted / fake adapters; no real model calls |

Raw data lives under `experiments_data/stopping_runs/` (gitignored):

```
<config>/<item>/unit.json                  cache signature + platform provenance (git head, engine.py hash, merge default, merge used)
<config>/<item>/seed_executor_output.md    E0
<config>/<item>/runs/<config>__<item>/iter_NNNN/   engine artifacts: executor_output.md, CONSENSUS.json,
                                           reviewer_output.md, adversarial_reviewer_output.md,
                                           bio_plausibility_checker_output.md (raw lens outputs),
                                           consensus_merger_output.md / EVAL.md, *_feedback.md
<config>/<item>/runs/<config>__<item>/run_meta.json   runner.build_run_meta (stop_reason, per-iteration provenance, convergence metrics)
<config>/<item>/iterations.jsonl           every completed IterationResult (engine order)
<config>/<item>/calls/call_NNNN_iterNNNN_<role>.json  every adapter call: full user prompt, system-prompt hash, raw output, usage, command, CLI cwd
<config>/<item>/calls/system_prompts/<sha>.txt
<config>/<item>/events.jsonl, attempts.jsonl, failed_attempts/   engine events; failed tries and their archived files
<config>/<item>/states/E<k>.md, trajectory.json   written when the trajectory is complete
judgments/<judge-model>/<config>/<item>/E<k>.json one judge record per state
logs/run_loops.log, logs/judge_states.log, logs/run_all.log
```

`trajectory.json` (schema `miw-stopping-trajectory/1`) contains:
- `observed_order` (must equal `expected_order`) and `stop_reason` (`max_iterations`).
- `states[k]`: text path, hash, length, executor-output similarity to the previous state
  (`similarity_to_previous`, the platform's word-Jaccard signal), the numbers absent from E0
  (`new_numbers`), and the `[PENDING ANALYSIS]` count.
- `panels[j]`:
  - merged grade and grade score, Critical and High counts, merged critique count,
    severity histogram;
  - `unresolved_critical`, panel status (`complete` / `partial`), `grade_valid` and failed
    lenses;
  - similarity method and threshold;
  - per lens: status, parse method and severity counts;
  - tokens and wall time.
- `steps[n]`: one row per engine iteration, with tokens, wall time, model and effort.
- `engine_similarity_scores` and `similarity_matches_engine`: the engine's own convergence
  metrics must match the recomputed similarities.
- `stop_events` and `failed_attempts`.

## Commands

```bash
cd automation/mi-workbench
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="/Volumes/Crucial X6/tmp_miw"
PY=/Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python

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

Exit codes: 0 means everything requested is complete; 1 means some unit or state is still
incomplete (rerun to retry it); 2 means fatal (an auth or quota error, or the engine fell
back to an inline prompt), and `run_all.sh` aborts.

### Resume

Resume works at step level:
- A step can fail after the engine's own retries, for example a Codex `[empty_output]` or a
  panel still partial after its in-step retry.
- The failure ends the engine run. The failed iteration's files move to
  `failed_attempts/`, the failure goes into `attempts.jsonl`, and the run continues through
  the engine's own resume path (`run_loop(resume_from=N)`).
- Each step gets up to `--max-attempts-per-step` tries (default 3).
- The last try uses `consensus_partial_policy=no_stop` and runs that step alone. A panel
  that stays partial is then kept, flagged `grade_valid=false`; replay skips it, as the
  engine would. Later steps go back to the strict policy.
- Killing the script loses at most the step in flight. Rerunning resumes from the last
  completed iteration.
- A unit whose `unit.json` signature differs is reported as `SIGNATURE_MISMATCH` and is
  never overwritten. The signature covers model, effort, horizon, task, seed and prompt
  hashes, run config and the Codex AGENTS.md hash.

## Judging (`judge_states.py`)

For each item and state E0..E5 there is one judge call: `gpt-5.6-sol`, effort `high`,
strict tools-off, fresh empty working directory.
- **Input:** the planted flaws (id, type, flawed passage, description), the ORIGINAL
  write-up for reference and the CURRENT state. The judge is not told the revision index.
- **Flaw labels:**
  - `resolved`: the error is gone and nothing depends on it, with no new results;
    withdrawing a claim or making it conditional on a marked pending analysis counts.
  - `unresolved`: this includes reworded flaws, generic caveats on top of a conclusion that
    still stands, and deleted sentences whose dependent conclusions remain.
  - `resolved_by_fabrication`: the fix relies on new quantitative results, analyses or data
    absent from the original.
- **New errors:** `fabrication` (every distinct fabricated result, including those that
  resolve a flaw), `technical`, `overclaim`, `inconsistency`, `other`.
- **Not fabrication:** arithmetic derived from the original's numbers and citing well-known
  literature.
- **Strict JSON:** every flaw exactly once, valid labels and categories. An invalid answer
  gets one re-ask; transient failures are retried with backoff.
- **Caching:** by input hash (judge model, effort, rubric, user prompt). A changed input
  moves the old record to `superseded/` and the state is judged again.
- E0 is judged too, as a sanity check: all planted flaws should be `unresolved`
  (`judge_sanity` in the results).

## Replay (`replay.py`)

Every rule is replayed with the platform's own `replay_stopping_rule` on the event sequence
the engine saw: E0 at iteration 0, then P0 at 1, E1 at 2, … P5 at 11. Executor events carry
the full text. Panel events carry the grade, Critical and High counts, the partial flag and
`unresolved_critical` (used for the gate). Rules are built with
`StoppingRule.from_config` on top of `DEFAULT_STOPPING_RULE`.

| Rule | Definition |
|---|---|
| `default_any_of_3` | the current platform default: any of grade-stable / no-Critical-High / output-similar; window 3, min 4 iterations, similarity ≥ 0.9 |
| `grade_stable_only`, `no_critical_high_only`, `output_similar_only` | each signal alone |
| `all_of_3` | all three |
| `noCH_and_(grade_stable_or_similar)` | `convergence_required_signals=no_critical_high`, `convergence_rule=any` |
| `noCH_and_(grade_stable_or_similar)+gate` | the same with `gate=True` (the consensus gate) |
| `fixed_k1` … `fixed_k5` | stop after k revisions |
| `*@window2`, `output_similar_only@sim0.8/0.95` | sensitivity only, never selected |

**Definitions.**
- The state at a stop is the latest executor state `E_k`, with `k = stop iteration // 2`.
  A rule that never fires runs to `E5`.
- "Before the horizon" means `k_stop < 5`.
- Premature stop: the rule stopped before the horizon while at least one planted flaw is
  `unresolved` in `E_k`. In the strict variant, `resolved_by_fabrication` also counts as not
  resolved.
- Each rule also reports, at the stop:
  - mean unresolved flaws;
  - flaws resolved by fabrication and new fabrication errors;
  - cycles (revisions), panel reviews and tokens consumed;
  - the k distribution.

**Selection** (pre-registered): among rules that stop before the horizon in at least 50%
of runs, pick the one with the lowest premature-stop rate; ties go to fewer cycles. It is
computed over the plan's rules including the fixed-k rules (`selection`), and over the
adaptive rules only (`selection_adaptive_only`).

**Also reported:**
- Per-revision means: unresolved flaws, the fraction with everything resolved, grade score,
  Critical/High, similarity, fabrication.
- Spearman correlations between the panel signals at `P_k` (grade score, Critical+High,
  Critical, merged critiques, executor similarity) and unresolved flaws in `E_k`. Each is
  pooled over item × k, with a 95% item-cluster bootstrap CI and the mean within-item ρ.
- Correlations between the change in grade and the change in unresolved flaws.
- An oracle reference: the first state in which every flaw is resolved.
- Resolution by flaw type.
- A cross-check of the new-number heuristic against the judge's fabrication labels.

## Notes and deviations

- The seed counts as the first executor output for the similarity signal, so the earliest
  `output_similar` stop with window 3 is at E3. `min_iterations` counts engine iterations,
  with P0 = 1.
- Reviewers see the executor's task text, including "cannot run new analyses", labelled as
  context only. This is the platform's normal framing, but it may make reviewers more
  willing to accept `[PENDING ANALYSIS]` withdrawals than reviewers of a free-form analysis
  would be.
- The executor runs with tools off (no shell, no data) by design.
- Grades and Critical/High counts come from the X3 merge, cross-lens `jaccard` at 0.5
  (the platform default when X3 started, pinned as `run_loops.X3_MERGE`; the platform
  default is now LLM adjudication). X1 and X2 found that this merges little, so counts
  track critique volume. The raw lens outputs are kept so the replay can be redone under
  another merge method.
