# Planted-flaw critique benchmark (X1 / X2)

The contract is `ANALYSIS_PLAN.md`. It was fixed before any reviewer call.

## Contents

| Path | What it is |
|---|---|
| `artifacts/` | 18 analysis write-ups: A01–A12 flawed, plus clean A01, A03, A05, A06, A09, A11 |
| `ground_truth/` | One JSON per artifact: flaw id, type, family, verbatim quote, description, detection criterion |
| `verification/` | Leak and consistency checks for each scenario |
| `benchmark_manifest.json` | SHA-256 of every artifact, ground-truth file and the plan, plus counts. Written by `freeze.py` |
| `prompts/judge.md` | The judge rubric (system prompt for `judge.py`) |
| `common.py` | Items, prompts, adapters, parsing, merging, cache paths |
| `codex_wrapper.py` | Codex CLI shim that adds `-c web_search="disabled"` to `codex exec` |
| `freeze.py` / `run_reviews.py` / `judge.py` / `analyze.py` / `merge_eval.py` | The pipeline, described below |
| `test_benchmark_harness.py` | Tests with fake adapters. No real LLM calls |
| `results/` | `benchmark_results.json`, `merge_eval.json` and CSV tables |

Raw data is written to `experiments_data/benchmark_runs/`, which is gitignored:

```
reviews/<model>/<item>/r<k>/<tag>.json          one reviewer call (request, raw output, parse, tokens, timings)
judgments/<judge>/<model>/<item>/r<k>.json      one judge call per unit (shuffled critiques, labels, raw output)
merge_eval/llm/<adjudicator>/<model>/<item>/r<k>.json   LLM-adjudication groups for X2
logs/run_reviews.log, logs/judge.log            one line per call
```

## Environments

- **Platform env** (`mi_workbench`: pydantic, pyyaml, no numpy). Runs `freeze.py`, `run_reviews.py`, `judge.py` and the tests.
- **Analysis env** (`/Users/ihorkendiukhov/anaconda3/bin/python`). Runs `analyze.py` and `merge_eval.py`.

All scripts use only the standard library plus the repository's `backend` package, so either env works for any of them. Set `PYTHONDONTWRITEBYTECODE=1`, because the drive is exFAT. Set `TMPDIR` to a directory on the external drive, because the per-call empty working directories are created under `$TMPDIR/miw_bench_ws`.

```bash
cd automation/mi-workbench
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="/Volumes/Crucial X6/tmp_miw"
PY=/Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python
APY=/Users/ihorkendiukhov/anaconda3/bin/python
```

## Commands

### 1. Freeze

Validate the benchmark and hash it:

```bash
$PY experiments/benchmark/freeze.py            # writes benchmark_manifest.json (aborts on any defect)
$PY experiments/benchmark/freeze.py --check    # verify only
```

`freeze.py` checks four things:
- every quote is a verbatim substring of its artifact;
- each of the 12 flaw types occurs exactly 4 times;
- flawed items have 4 flaws from 4 distinct families, and clean items have none;
- families match the types, and ids are unique.

If any check fails it aborts and writes nothing. An existing manifest is never replaced silently: `--force` is required, and every condition run on a changed artifact must then be re-run. `run_reviews.py` and `judge.py` refuse to run if the files differ from the manifest.

### 2. Reviewer calls

Each unit of work is (model, item, repeat). Every unit gets 8 calls on the same input:

| Tags | Prompt |
|---|---|
| `rig1`–`rig3` | `reviewer/mi_reviewer` |
| `cmb1`–`cmb3` | `reviewer/mi_reviewer_combined` |
| `adv` | `adversarial_reviewer/adversarial_reviewer` |
| `bio` | `biological_plausibility/bio_plausibility_checker` |

Pilot run:

```bash
$PY experiments/benchmark/run_reviews.py --models gpt-5.6-sol,gpt-5.6-luna --effort medium \
    --repeats 1 --items A01-flawed,A05-clean --concurrency 4
```

Full run (all 18 items, all 8 calls, 3 repeats):

```bash
$PY experiments/benchmark/run_reviews.py --models gpt-5.6-sol,gpt-5.5 --effort medium \
    --repeats 3 --concurrency 4
```

Options:

| Option | Default |
|---|---|
| `--items` | all 18 |
| `--calls` | `rig1,rig2,rig3,cmb1,cmb2,cmb3,adv,bio` |
| `--concurrency` | 4 |
| `--timeout` (seconds per call) | 900 |
| `--max-attempts` (new attempts per invocation) | 3 |
| `--backoff` (first retry delay, seconds; doubles each time, capped at 240 s) | 15 |
| `--data-dir` | — |
| `--codex-home` (sets `CODEX_HOME` for the CLI subprocesses) | — |
| `--workspace-root` | — |
| `--dry-run` | — |

How the run behaves:
- **Models.** The provider is inferred from the model name: `gpt-*` uses Codex, `claude-*` uses Claude Code, `gemini-*` uses Gemini. Use `provider:model` to force a provider.
- **Calls.** Each call uses the backend adapter with `model`, `reasoning_effort` and `allow_tools=False`. For Codex that means `--sandbox read-only --ephemeral --json`.
- **Working directory.** Each call starts with `--cd` set to a fresh empty directory outside the repository, which is deleted afterwards.
- **Resuming.** Successful cached calls are skipped. Failed calls are retried.
- **Cache mismatches.** A cached record made with a different model, effort or prompt is never overwritten. It is reported as `CACHE_MISMATCH`.
- **Scheduling.** The order is repeat → item (seeded order) → model → call. A partial run therefore completes whole units and stays balanced across models.

### 3. Judge

```bash
$PY experiments/benchmark/judge.py --judge-models gpt-5.6-sol --effort high --concurrency 2
$PY experiments/benchmark/judge.py --judge-models gpt-5.6-sol,gpt-5.5 --effort high   # second judge (kappa)
```

Each unit is judged once all 8 of its calls have succeeded. The judge receives:
- the artifact;
- the planted-flaw list (id, passage, description, detection criterion), which is empty for clean items;
- the union of parsed critiques, shuffled and numbered. The shuffle is seeded by `sha256(model|item|repeat)`. Each critique shows only its description and proposed fix: no lens, call or severity.

Label validation:
- every index must be labelled exactly once;
- flaw ids must be valid;
- a `none` label needs a `none_type` of `substantive`, `generic` or `incorrect`.

An invalid answer is re-asked once. A failed or invalid unit is retried on the next invocation. A judgment is recomputed automatically if any of its inputs change.

### 4. Analysis

```bash
$APY experiments/benchmark/analyze.py --judges gpt-5.6-sol[,gpt-5.5]
```

The results follow `ANALYSIS_PLAN.md`:
- **Recall.** Recall per condition (C1–C5), per model and pooled. Also recall by family and by type, and per flaw, including the T1 attention-symmetry instances A01-F4, A02-F3 and A07-F2.
- **Severity-aware recall.** Uses the platform merge (`merge_parsed_critiques`, jaccard 0.5, consensus threshold 2): the platform default at freeze time, pinned in `common.DEFAULT_MERGE_METHOD` / `DEFAULT_MERGE_THRESHOLD`.
- **Unique contributions** of each C5 lens.
- **Discrimination.** AUROC of flawed vs clean items for the merged grade (ordinal) and the severity score (8/4/2/1/0), plus a paired flawed−clean comparison.
- **False alarms.** Critical/High critiques on clean items.
- **Unmatched critiques**, split into substantive, generic and incorrect.
- **Cost.** Calls, input/cached/output tokens and wall time per condition.
- **Call statistics.** Success rate, parse methods and critiques per call.
- **Judge agreement.** Cohen's κ at critique level and at item-flaw level, each with a bootstrap CI.
- **Planned contrasts.** C5−C1, C5−C3, C5−C4, C4−C2 and C3−C1, using a paired item-clustered bootstrap (10,000 resamples, repeats kept within item). Holm correction is applied within each model and within the pooled analysis.
- **Sensitivity analyses:**
  - judge 2;
  - the intersection of judges 1 and 2;
  - reviewer identity for escalation set to `role` instead of `tag`.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--bootstrap` | 10000 | resamples for the planned contrasts |
| `--bootstrap-descriptive` | 2000 | resamples for descriptive CIs and κ |
| `--merge-method` / `--merge-threshold` | `jaccard` / 0.5 | the merge used for the severity outcomes. Pinned to the platform merge's default settings at freeze time (jaccard, threshold 0.5), as pre-specified in `ANALYSIS_PLAN.md`; they do not follow the later platform defaults (LLM adjudication; calibrated thresholds 0.1, from the merge evaluation below). A threshold left unset resolves to the freeze-time default of the chosen method (jaccard 0.5, tfidf 0.3, embedding 0.7). Other values are sensitivity analyses (e.g. `results/sensitivity_tfidf/`: `--merge-method tfidf --merge-threshold 0.075`) |
| `--lens-identity` | `tag` | `tag` means each call is a distinct reviewer, so a self-ensemble can escalate like a panel |

Partial data is fine. Every table reports n (units and items).

### 5. Merge evaluation (X2)

```bash
$APY experiments/benchmark/merge_eval.py --judge gpt-5.6-sol \
    --adjudicator-model gpt-5.6-sol --adjudicator-effort medium
```

Ground truth is built within each C5 panel instance. Two cross-lens critiques that are matched to the same planted flaw form a duplicate pair. Two matched to different flaws form a non-duplicate pair. Critiques labelled `none` are excluded.

Methods:
- `jaccard`, `tfidf` and `embedding` use the platform's average-linkage grouping.
- `embedding` runs only if sentence-transformers and a locally cached model exist. Otherwise it is skipped, never downloaded.
- `llm` uses `LLMAdjudicator`, with one cached call per instance.

Outputs:
- pairwise precision, recall and F1, with thresholds chosen by leave-one-scenario-out cross-validation;
- ARI over flaw-matched critiques;
- escalation validity;
- the recommended default: best held-out F1, ties broken by cost.

`default_threshold` / `at_default_threshold` in `merge_eval.json` refer to each method's threshold before the calibration (jaccard 0.5, tfidf 0.3, embedding 0.7; `similarity.UNCALIBRATED_THRESHOLDS`), so a re-run reproduces the file whatever the current platform defaults are.

Result (`results/merge_eval.json`, judge gpt-5.6-sol; 162 panel instances, 4,556 labelled cross-lens pairs): held-out F1 llm 0.824, tfidf 0.814, jaccard 0.804 (embedding skipped). The folds selected thresholds 0.075–0.1 for both jaccard and tfidf; at the uncalibrated jaccard 0.5 recall was 0.012 (tfidf 0.3: 0.617). The platform adopted these results: the default `consensus_similarity_method` is `llm` (pre-specified winner), with a fallback to `tfidf` at 0.1 when the adjudication fails, and the jaccard / tfidf default thresholds are 0.1 (the upper end of the fold-selected range). The pre-specified benchmark analyses keep the freeze-time merge (see `--merge-method` above).

### Tests

```bash
TMPDIR="/Volumes/Crucial X6/tmp_miw" PYTHONDONTWRITEBYTECODE=1 $PY -m pytest \
    experiments/benchmark/test_benchmark_harness.py -q -p no:cacheprovider
```

## Notes and known limitations

- **What Codex adds to the prompt.**
  - Codex puts its own harness instructions and tool definitions in front of our prompt.
  - It also adds the user-level `$CODEX_HOME/AGENTS.md`.
  - Each record stores `request.codex_env`: hashes of `AGENTS.md` and `config.toml`.
  - Codex starts every call in the empty working directory, so no project `AGENTS.md` is read.
  - For a clean run, point `--codex-home` at a separate, logged-in Codex home that has no `AGENTS.md` and a minimal `config.toml`.
- **Tool use.**
  - The backend adapter's `allow_tools=False` maps to `--sandbox read-only`. That still leaves Codex's live web-search tool on, and the pilot saw the bio lens use it.
  - The harness therefore turns web search off by default (`--codex-web-search disabled`). `codex_wrapper.py`, which is used as the adapter's binary, inserts `-c web_search="disabled"` after `codex exec`; the arguments are recorded in `request.cli_extra_args`, and records made with different extra arguments count as cache mismatches.
  - Read-only shell commands remain possible.
  - Every agent tool item (command execution, web search, MCP call) is recorded in `tool_items`, captured from the CLI's JSONL stdout. `tool_events` counts the `item.started` events.
- **Sampling.** No temperature is set, because the CLIs do not expose it. Each record notes "CLI defaults".
