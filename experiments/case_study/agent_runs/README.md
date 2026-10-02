# X4: executed case study with verified execution (real Geneformer kit)

Question given to the agents:

> Does Geneformer V2-104M attention encode TF->target regulation beyond co-expression,
> expression-rank proximity and hub (degree) structure?

The agents work only from the read-only analysis kit
`experiments_data/case_study_kit/` (see the kit `README.md`). The author-side independent
validation (`../results/`) is never shown to them: it is not in the task text (a test checks
that no validation number and no decimal result literal appears in it) and the sandbox cannot
read it (checked by `smoke-sandbox`).

## Design

| | |
|---|---|
| Engine | the real platform runner (`backend.orchestrator.runner.execute_run` driving `LoopEngine`), Codex CLI adapter, own SQLite DB |
| Loop | preset `reviewer_consensus`: executor <-> three-lens consensus panel (rigour, adversarial, bio-plausibility; 3 concurrent calls) + LLM adjudication of duplicate critiques (`consensus_similarity_method: llm`, one call after the lenses) |
| Horizon | fixed: 5 executor iterations, E1 P1 ... E5 P5 = 10 iterations (`max_iterations` 10, `convergence_enabled: false`, no `grade_at_least`, token budget 50M) |
| Verified execution | `code_execution_enabled: true`, backend `sandbox_exec` (Seatbelt: no network, writes confined to the per-execution temp dir, reads limited to system/interpreter paths and the kit), interpreter `/Users/ihorkendiukhov/anaconda3/bin/python` (numpy/scipy/pandas/sklearn/statsmodels), 900 s wall / 1800 s CPU per execution, single combined script, report shows stdout up to 24 KB and written text files up to 24 KB each / 48 KB in total with `results.json` read first; the complete outputs of every execution (full stdout/stderr, code, every written file) are kept in `exec_outputs/<key>/` (`code_execution_persist_dir`) |
| Tools | executor agent tools OFF (verified-execution default: every number must come from the sandbox); reviewers and adjudicator strictly tools-off (`--sandbox read-only`, no shell, no web search, user config ignored) |
| Models | `model` and `reasoning_effort` passed explicitly on every call |

Runs (sequential; each panel already issues 3 concurrent Codex calls):

| key | run_id | model | effort | code execution | task text |
|---|---|---|---|---|---|
| `sol_A` | `x4_sol_A` | gpt-5.6-sol | medium | sandbox_exec | `task_text_executed.md` |
| `sol_B` | `x4_sol_B` | gpt-5.6-sol | medium | sandbox_exec | `task_text_executed.md` (replicate) |
| `gpt55` | `x4_gpt55` | gpt-5.5 | medium | sandbox_exec | `task_text_executed.md` |
| `sol_planonly` | `x4_sol_planonly` | gpt-5.6-sol | medium | **off** (control), executor tools explicitly off | `task_text_plan_only.md` (= executed text minus its "COMPUTING ENVIRONMENT AND REPORTING RULE" section) |

The task text (frozen copies `task_text_executed.md`, `task_text_plan_only.md`; a test keeps them
identical to `build_task_text`) contains the kit path, a description of every kit file with its
orientation and semantics (adapted from the kit README), the analysis expectations (effect sizes
with uncertainty, baselines and nulls for each named confounder, multiple-testing control, null
results, verdict separating evidence / inference / hypothesis) and, for the executed runs, the
sandbox limits and the rule that only numbers produced by executed code may be reported. The
platform appends its verified-execution addendum (`prompts/executor/mi_executor_code.yaml`) to
the executor system prompt.

## Commands (from `automation/mi-workbench`)

```bash
PY=/Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python
export TMPDIR="/Volumes/Crucial X6/tmp_miw" PYTHONDONTWRITEBYTECODE=1

# task texts
$PY experiments/case_study/agent_runs/run_case_study.py print-task --variant executed
$PY experiments/case_study/agent_runs/run_case_study.py print-task --variant plan_only

# sandbox smoke test with exactly the executed runs' config (reads the kit, imports numpy,
# checks that the validation results, ~/.codex, kit writes and the network are denied)
$PY experiments/case_study/agent_runs/run_case_study.py smoke-sandbox

# launch / resume the whole chain (sol_A, sol_B, gpt55, sol_planonly), then trace_numbers
nohup caffeinate -i $PY experiments/case_study/agent_runs/run_case_study.py run \
    >> experiments_data/case_study_agent_runs/logs/x4_chain.out 2>&1 &
echo $! > experiments_data/case_study_agent_runs/logs/x4_chain.pid

# progress
$PY experiments/case_study/agent_runs/run_case_study.py status          # DB view, all runs
tail -f experiments_data/case_study_agent_runs/logs/x4_runs.log
ls experiments_data/case_study_agent_runs/workspaces/sol_A/runs/x4_sol_A/

# a subset / reproducibility packages again
$PY experiments/case_study/agent_runs/run_case_study.py run --runs gpt55,sol_planonly
$PY experiments/case_study/agent_runs/run_case_study.py repack --runs sol_A

# number tracing (also runs automatically at the end of `run`)
$PY experiments/case_study/agent_runs/trace_numbers.py

# tests (offline: fake adapter, real runner/engine/consensus/sandbox)
$PY -m pytest experiments/case_study/agent_runs/test_case_study_agent_runs.py -q -p no:cacheprovider
```

Resuming: re-run the same `run` command. Completed runs are skipped. An interrupted run
(process killed) or a failed one (adapter or panel failure after the engine's own retries) is
repaired: iteration rows after the last completed iteration are deleted from the DB, their
directories are moved to `<run_dir>/x4_failed_attempts/<timestamp>/`, and the run resumes from
the last completed iteration, so the E/P horizon stays intact. Each repair is logged in
`<run_dir>/x4_repairs.jsonl`; within one invocation a run is repaired at most `--max-repairs`
times (default 4, waits 1, 5, 15, 30 min). A run whose task text or config differs from its
`x4_run_spec.json` is refused. A file lock (`run.lock`) prevents two concurrent chains.

## Outputs

`experiments_data/case_study_agent_runs/` (gitignored):

- `miw_case_study.db`: runs, iterations (per-iteration provider, model, effort, CLI version,
  tokens, duration, grade, code-execution summary, agent-tool provenance) and per-call telemetry.
- `workspaces/<key>/runs/x4_<key>/`:
  - `iter_NNNN/`: executor iterations `executor_output.md` (full output), parsed artifacts
    (`MECH.md`, `EVAL.md`, ...), `CODE_EXECUTION.json` (full report incl. stdout, stderr,
    captured files and `results.json`), `RUN_LOG.md` (the report as the reviewers saw it);
    panel iterations `reviewer_output.md`, `adversarial_reviewer_output.md`,
    `bio_plausibility_checker_output.md`, `consensus_merger_output.md`, `EVAL.md`,
    `CONSENSUS.json`, `consensus_merger_feedback.md` (exact text queued for the executor).
  - `run_meta.json` (platform provenance), `x4_run_spec.json` (spec, config, task hash, Codex CLI
    version, `~/.codex/AGENTS.md` md5, project AGENTS.md files on the path (none), kit manifest md5,
    analysis-interpreter versions, git head), `x4_task.md`, `x4_prompts/` (exact prompt of every
    adapter call + `index.jsonl` with model, effort, tools flag, tokens, and whether the user prompt
    contained the execution report), `x4_events.jsonl` (engine events), `x4_repairs.jsonl`,
    `x4_invocations.jsonl`.
- `exec_outputs/<key>/<timestamp>_<id>/`: complete outputs of each execution unit (`code.py`,
  `stdout.txt`, `stderr.txt`, `work/` = every file the code wrote); `CODE_EXECUTION.json`
  `blocks[].persisted_to` names the unit of an iteration.
- `repropacks/<key>.zip`: `backend.repropack` package extended with `run_dir/` (all iteration
  directories, prompts, events), `exec_outputs/`, the case-study scripts and the kit
  README/manifest.
- `archive/pilot_v0_e1p1_no_persist/`: the first launch of `sol_A` (E1 + P1), stopped to add
  output persistence (see its `NOTE.md`).
- `logs/x4_runs.log`, `logs/x4_chain.out`, `logs/x4_chain.pid`, `status_<key>.json`,
  `smoke/sandbox_smoke.json`.

`experiments/case_study/agent_runs/results/`:

- `run_status.json`: final status of every run of the last chain invocation.
- `trace_summary.json`, `trace_per_iteration.csv`, `trace_details/<key>.json` (from
  `trace_numbers.py`).

## trace_numbers.py

Two write-up channels per executor iteration: `agent_text` (the executor response minus fenced
code) and `generated_reports` (`*.md` files written by the executed code; in the first live
iteration the agent answered with code only and its code wrote MECH.md / EVAL.md / ... with
formatted results, which is what the reviewers read). A generated-report number is `computed`
unless the executed code contains it as a literal; a literal is `literal_matches_output` if
it equals a value in stdout or a non-narrative output file, else `hard_coded` (untraceable).
`combined_result` adds both channels. In `agent_text`, every number (`result`; `small_integer` |v| <= 12; `parameter`: seeds,
permutation / bootstrap counts, thresholds, CI level, chance; `threshold`: conventional
significance levels next to p/q/FDR/alpha; excluded: `task_given` literals from the task text,
`list_marker`, `index` (layer/head/iteration/section/critique), `year`). A number is traceable if
it equals, after rounding to the shown precision, a number in that iteration's complete executed
stdout or any text file the code wrote (`results.json` etc., from `exec_outputs/`; the capped
report is the fallback) ("current") or in an earlier executor iteration's executed output
("earlier"); percent <-> fraction and sign-agnostic matches are allowed and flagged. For
the plan-only control every result number is untraceable by definition. Caveats recorded per
iteration: truncated stdout / files (an untraceable number may sit in the omitted part) and
traceable numbers that also appear as a literal in the executed code (printed back, not
computed). The classification is heuristic; every number is listed with its category, match
and context in `trace_details/<key>.json` for audit.

## Platform fixes made for this experiment (verified-execution path only)

- `backend/orchestrator/sandbox/capture.py` + `code_executor.py`: `results.json` (top level of the
  work directory) is captured before any other written file. Before, files whose names sort
  earlier (`MECH.md`, `a_table.csv`, ...) could exhaust the 16 KB text budget, and `results.json`
  was then reported as "text capture budget exhausted" although the prompt tells the reviewers to
  check every number against it.
- `code_executor.py`: new run-config keys `code_execution_capture_text_bytes` (4000),
  `code_execution_capture_total_bytes` (16000) and `code_execution_capture_max_files` (50);
  before, a `results.json` above 4 KB was always truncated and the caps could not be raised.
- `code_executor.py` + `sandbox/capture.py`: new opt-in run-config keys
  `code_execution_persist_dir` / `code_execution_persist_max_mb` (256). Before, the work
  directory was deleted after each execution and only the head/tail of stdout and capped file
  previews survived, so reported numbers could not be audited against the complete executed
  output (in the pilot, E1's `results.json` was 187 KB, of which 24 KB were kept). With the key
  set, each unit keeps `code.py`, full `stdout.txt` / `stderr.txt` and `work/` (regular files only;
  symlinks / hard links / special files are never followed or copied; byte budget per unit), and
  `CODE_EXECUTION.json` records `persisted_to`. The sandboxed code cannot write there.
  Regression tests: `backend/tests/test_code_executor_capture.py`. The keys are documented
  in the main README's `code_execution_*` table and in the module docstring.

## Known limitations

- When these packages were built, `backend.repropack` collected only top-level run-directory
  files, so `build_repropack` adds the iteration directories (`run_dir/`) and the persisted
  execution outputs (`exec_outputs/`) itself. The generator now packages both
  (`iterations/`, `code_execution/`, with a per-file size cap), so packages rebuilt with
  `repack` hold them twice.
- The Codex CLI injects `~/.codex/AGENTS.md` (the user's global persona) into every call, as in
  the X1 benchmark; its md5 is recorded per run.
- Codex reports no cost; token counts are provider-reported (input includes cached input).
- Memory is not limited by `sandbox_exec` on macOS (only wall/CPU time, file size, open files).
