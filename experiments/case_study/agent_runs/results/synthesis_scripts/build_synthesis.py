"""Build case_synthesis.json (machine-readable) for the four X4 case-study runs.

Every number is read programmatically from the run artefacts and carries its source
path relative to experiments_data/case_study_agent_runs (validation numbers: relative
path to experiments/case_study/results/independent_validation.json from the same base).
Narrative fields were written after reading the artefacts.
"""
import json, os, re, datetime
import pandas as pd
import numpy as np

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/automation/mi-workbench/experiments_data/case_study_agent_runs"
VAL_ABS = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/automation/mi-workbench/experiments/case_study/results/independent_validation.json"
VAL_REL = "../../experiments/case_study/results/independent_validation.json"
OUT = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/automation/mi-workbench/experiments/case_study/agent_runs/results/case_synthesis.json"
RUNS = ["sol_A", "sol_B", "gpt55", "sol_planonly"]
V = json.load(open(VAL_ABS))


def rel(p):
    return p.replace(BASE + "/", "")


def rundir(run):
    return f"workspaces/{run}/runs/x4_{run}"


def jget(d, keypath):
    cur = d
    for k in keypath.split("/"):
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur


def num(v, source, key=None, note=None):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        v = int(v)
    o = {"value": v, "source": source}
    if key is not None:
        o["key"] = key
    if note:
        o["note"] = note
    return o


def vnum(keypath, note=None):
    return num(jget(V, keypath), VAL_REL, keypath, note)


# ---------------------------------------------------------------- exec mapping
EXEC = {}
for run in RUNS[:3]:
    EXEC[run] = {}
    for e, it in enumerate((1, 3, 5, 7, 9), 1):
        ce = json.load(open(f"{BASE}/{rundir(run)}/iter_{it:04d}/CODE_EXECUTION.json"))
        EXEC[run][e] = rel(ce["blocks"][0]["persisted_to"])


def rj(run, e, keypath, note=None):
    p = f"{EXEC[run][e]}/work/results.json"
    d = json.load(open(f"{BASE}/{p}"))
    return num(jget(d, keypath), p, keypath, note)


def so(run, e, name, note=None):
    p = f"{EXEC[run][e]}/stdout.txt"
    for line in open(f"{BASE}/{p}"):
        if line.startswith(name + "="):
            raw = line.strip().split("=", 1)[1]
            try:
                val = json.loads(raw)
            except Exception:
                try:
                    val = float(raw)
                except Exception:
                    val = raw
            return num(val, p, f"stdout:{name}", note)
    raise KeyError(name)


def tsv(run, e, fname, query_col, query_val, col, note=None):
    p = f"{EXEC[run][e]}/work/{fname}"
    t = pd.read_csv(f"{BASE}/{p}", sep="\t")
    row = t[t[query_col] == query_val].iloc[0]
    return num(row[col], p, f"{fname}[{query_col}=={query_val}].{col}", note)


# ---------------------------------------------------------------- run-level data
def run_level(run):
    rd = rundir(run)
    meta_p = f"{rd}/run_meta.json"
    m = json.load(open(f"{BASE}/{meta_p}"))
    spec = json.load(open(f"{BASE}/{rd}/x4_run_spec.json"))
    t0 = datetime.datetime.fromisoformat(m["started_at"])
    t1 = datetime.datetime.fromisoformat(m["completed_at"])
    ex_tok = sum(it["token_usage"] for it in m["iterations"] if it["role"] == "executor")
    pa_tok = sum(it["token_usage"] for it in m["iterations"] if it["role"] != "executor")
    calls = sum((it.get("agent_tools") or {}).get("calls", 0) for it in m["iterations"])
    n_prompts = sum(1 for _ in open(f"{BASE}/{rd}/x4_prompts/index.jsonl"))
    failed = sum((it.get("code_execution") or {}).get("failed", 0) for it in m["iterations"])
    timed = sum((it.get("code_execution") or {}).get("timed_out", 0) for it in m["iterations"])
    return {
        "run_id": m["run_id"],
        "model": num(m["model"], meta_p, "model"),
        "reasoning_effort": num(m["reasoning_effort"], meta_p, "reasoning_effort"),
        "provider": num(m["provider"], meta_p, "provider"),
        "cli_versions": num(m["cli_versions"], meta_p, "cli_versions"),
        "loop_preset": num(m["loop_preset"], meta_p, "loop_preset"),
        "task_variant": num(spec["task_variant"], f"{rd}/x4_run_spec.json", "task_variant"),
        "status": num(m["status"], meta_p, "status"),
        "stop_reason": num(m["stop_reason"], meta_p, "stop_reason"),
        "total_iterations": num(m["total_iterations"], meta_p, "total_iterations"),
        "total_tokens": num(m["total_tokens"], meta_p, "total_tokens"),
        "total_input_tokens": num(m["total_input_tokens"], meta_p, "total_input_tokens"),
        "total_output_tokens": num(m["total_output_tokens"], meta_p, "total_output_tokens"),
        "total_cached_input_tokens": num(m["total_cached_input_tokens"], meta_p, "total_cached_input_tokens"),
        "executor_tokens_sum": num(ex_tok, meta_p, "sum(iterations[role=executor].token_usage)"),
        "panel_tokens_sum": num(pa_tok, meta_p, "sum(iterations[role=consensus_merger].token_usage)"),
        "agent_calls_recorded": num(calls, meta_p, "sum(iterations[].agent_tools.calls)",
                                     "5 executor calls + 5 panels x (3 lenses + 1 adjudicator)"),
        "prompt_files_logged": num(n_prompts, f"{rd}/x4_prompts/index.jsonl", "line count",
                                    "gpt55 logged 28: the three P1 lens prompts were issued twice with identical sha256 (adapter-level re-issue); CONSENSUS.json records 1 attempt" if run == "gpt55" else None),
        "run_wall_clock_min": num(round((t1 - t0).total_seconds() / 60, 1), meta_p, "completed_at - started_at"),
        "executor_tools_setting": num(m["config"]["effective_tool_settings"], meta_p, "config.effective_tool_settings"),
        "code_execution_enabled": num(m["config"]["code_execution_enabled"], meta_p, "config.code_execution_enabled"),
        "code_execution_backend": num(m["config"].get("code_execution_backend"), meta_p, "config.code_execution_backend"),
        "code_execution_limits": num({"wall_s": m["config"].get("code_execution_timeout"), "cpu_s": m["config"].get("code_execution_cpu_seconds")}, meta_p, "config.code_execution_timeout / code_execution_cpu_seconds"),
        "failed_executions": num(failed, meta_p, "sum(iterations[].code_execution.failed)"),
        "timed_out_executions": num(timed, meta_p, "sum(iterations[].code_execution.timed_out)"),
        "grade_history": num(m["convergence"]["grade_history"], meta_p, "convergence.grade_history"),
    }


# ---------------------------------------------------------------- executor iteration mechanics
def exec_mech(run, e):
    it = 2 * e - 1
    rd = rundir(run)
    meta_p = f"{rd}/run_meta.json"
    m = json.load(open(f"{BASE}/{meta_p}"))
    mi = [x for x in m["iterations"] if x["iteration"] == it][0]
    rec = {
        "step": f"E{e}", "loop_iteration": it,
        "executor_call": {
            "duration_s": num(mi["duration_seconds"], meta_p, f"iterations[{it-1}].duration_seconds",
                               "includes LLM generation and the sandboxed execution"),
            "input_tokens": num(mi["input_tokens"], meta_p, f"iterations[{it-1}].input_tokens"),
            "output_tokens": num(mi["output_tokens"], meta_p, f"iterations[{it-1}].output_tokens"),
        },
    }
    eo = f"{rd}/iter_{it:04d}/executor_output.md"
    txt = open(f"{BASE}/{eo}").read()
    fences = len(re.findall(r"^```", txt, re.M))
    outside = re.sub(r"```.*?```", "", txt, flags=re.S).strip()
    rec["executor_response"] = {
        "bytes": num(len(txt.encode()), eo, "file size"),
        "fenced_blocks": num(fences // 2, eo, "count of ``` pairs"),
        "prose_chars_outside_code": num(len(outside), eo, "chars outside fenced code"),
    }
    if run == "sol_planonly":
        rec["execution"] = {"executed": False, "note": "code execution disabled (control)"}
        return rec
    cep = f"{rd}/iter_{it:04d}/CODE_EXECUTION.json"
    ce = json.load(open(f"{BASE}/{cep}"))
    b = ce["blocks"][0]
    pers = rel(b["persisted_to"])
    code = open(f"{BASE}/{pers}/code.py").read()
    err = open(f"{BASE}/{pers}/stderr.txt").read().strip()
    work = sorted(f for f in os.listdir(f"{BASE}/{pers}/work") if not f.startswith("._"))
    rec["execution"] = {
        "executed": True,
        "exit_code": num(b["exit_code"], cep, "blocks[0].exit_code"),
        "success": num(b["success"], cep, "blocks[0].success"),
        "timed_out": num(b["timed_out"], cep, "blocks[0].timed_out"),
        "limit_hit": num(b["limit_hit"], cep, "blocks[0].limit_hit"),
        "signal": num(b["signal"], cep, "blocks[0].signal"),
        "exec_wall_s": num(round(b["duration_seconds"], 1), cep, "blocks[0].duration_seconds"),
        "stdout_bytes": num(b["stdout_bytes"], cep, "blocks[0].stdout_bytes"),
        "stderr_last_line": num(err.splitlines()[-1] if err else "", f"{pers}/stderr.txt", "last line"),
        "persisted_to": pers,
        "code_lines": num(code.count("\n") + (0 if code.endswith("\n") else 1), f"{pers}/code.py", "line count"),
        "files_written": num(work, f"{pers}/work", "directory listing"),
        "results_json_written": "results.json" in work,
    }
    return rec


# ---------------------------------------------------------------- panel records
TOP3 = {
 "sol_A": {
  1: ["Repeat the analysis within cell types (or a hierarchical cell type/tissue/donor model) - not possible with the pooled kit attention.",
      "Recompute TRRUST degree covariates without the evaluated edge (fold-wise) or from an independent network; the 0.9877 baseline AUROC is outcome leakage.",
      "Audit TRRUST/DoRothEA overlap by PMID/source; edge-exclusion alone does not make DoRothEA an independent validation."],
  2: ["Redesign storage to stay below the 64 MiB per-file limit and rerun; no number in the templated reports is supported by a successful run.",
      "Use immune-context labels (ChIP/perturbation/motif-accessibility) or report generic-database recovery only as annotation discrimination.",
      "Repeat the 36-way layer/orientation selection inside every bootstrap replicate, or use disjoint selection and evaluation sets."],
  3: ["Cell-type-specific attention and matched labels; until then restrict to context-agnostic annotation discrimination.",
      "Add a no-attention option to selection and use one model class for selection and final refit (all 36 variants worsened selection log-loss by >= 0.0893).",
      "Repeat the within-TF permutation hundreds of times with full refit/selection (the single permutation's dAUROC 0.0185 exceeded attention's 0.0151)."],
  4: ["Deduplicate DoRothEA evidence against TRRUST and run the same bootstrap/nulls, or label DoRothEA an unvalidated exploratory sensitivity analysis.",
      "State the verdict explicitly as a negative result (dAUROC -0.0106, dAUPRC 0.0005, both nulls non-significant) instead of 'unresolved'.",
      "Per-cell/cell-type attention with active-TF filtering and context validation; otherwise database-annotation interpretation only."],
  5: ["Permute attention only among nuisance-matched pairs (the within-TF alignment permutation is not a conditional null).",
      "Run the same 15-split averaging for every null replicate (observed mean of 15 splits vs single-split nulls: p = 0.25/0.1875/0.4545 uncalibrated).",
      "Replace t-intervals over 15 overlapping splits by a TF-cluster bootstrap of the full nested pipeline, or drop confidence-interval language."]},
 "sol_B": {
  1: ["Provenance audit and validation on a source-independent immune perturbation/ChIP benchmark (DoRothEA-exclusive is not independent).",
      "Stratify by cell type and donor, require within-stratum expression, and report the specific pairs driving each supported score.",
      "Compare positives and controls over identical cell sets / cell-type strata, and add degree-preserving rewiring constrained on nuisances."],
  2: ["Estimate effects within cell-type/tissue strata on identical cell sets, or restrict to an unlocalized aggregate association (TRRUST max |SMD| 0.988).",
      "Constrain degree-preserving swaps within co-presence/expression/rank/context strata (the p = 0.00995 null tests unmatched mean log attention).",
      "The matched estimand covers 9/424 TRRUST and 76/1949 DoRothEA edges: audit retained vs excluded edges, caliper sensitivity, overlap weighting."],
  3: ["Do not interpret aggregate effects unless balance is achieved (max TF-macro |SMD| 1.279 TRRUST, 1.106 DoRothEA).",
      "Remove layer-0-attention-derived covariates from control selection (outcome-informed matching).",
      "Fix the graph-swap code (signature lookup returns None for already-rewired edges) and generate many mixed rewired graphs."],
  4: ["Recompute within cell-type x tissue strata with donor replication, or restrict the conclusion to aggregate associations.",
      "Gate prediction on balance and adequate counts (4 TRRUST edges, max |SMD| 2.557): the AUROC comparisons are uninterpretable.",
      "Add independent-network degree and run the degree-preserving null, or state that hub bias is unresolved."],
  5: ["Remove the __file__ dependency and rerun; the failed execution produced no result.",
      "Prespecify biological support criteria; audit flags are by construction (supporting_assay_available set from PMID presence).",
      "The exact binary co-presence gate is unnecessarily stringent; compare weighting/continuous adjustment before declaring non-identifiability."]},
 "gpt55": {
  1: ["Replace pair-level Welch t-tests (p down to 1.4e-17) with TF-cluster-aware tests before BH.",
      "Use leave-one-edge-out or external degree covariates (circular degree-only AUROC 0.954).",
      "Report positive and negative effects separately: the verdict used the smallest-p variant regardless of sign while 5 positive-SMD variants had q < 0.05."],
  2: ["Make the script finish within 900 s (vectorize, checkpoint) and report only numbers from completed files.",
      "Frame any positive attention result as secondary to degree/hub structure (LOO-degree AUPRC far above attention).",
      "Complete a family max-statistic null or prespecify the family; the partial table shows only negative BH-significant variants."],
  3: ["Use immune-specific references or cell-type stratification (TRRUST is context-generic).",
      "Increase family max-statistic permutations beyond 30, or treat the 22 negative BH findings as exploratory.",
      "Label selected-variant CIs as post-selection, or nest selection inside an outer held-out-TF split."],
  4: ["Annotate the evidence context of top pairs (only 3/20 flagged by the immune keyword panel) or restrict to context-generic association.",
      "Use >= 1000 matched within-TF permutations with continuous/caliper matching and balance diagnostics.",
      "Use fold-local topology (degree) features in held-out-TF CV; the globally computed LOO degree leaks test-fold labels."],
  5: ["State that the verified partial run shows no significant attention signal beyond topology (top q ~0.285; LOO-degree AUPRC 0.0553 vs best attention AUPRC 0.0096).",
      "Finish within the 900 s limit (timed out at 900.87 s before results.json) and report only produced numbers.",
      "Stratify or exclude housekeeping/mitochondrial/ribosomal targets and validate top pairs with immune-context evidence."]},
 "sol_planonly": {
  1: ["Rerun with executable access and report cell-type-stratified or cell-type-adjusted results.",
      "Define the pair universe and confounder-preserving nulls; report incremental AUROC/AUPRC with bootstrap CIs.",
      "Fit cross-validated confounder-only vs confounder-plus-attention models with leakage-safe splits."],
  2: ["Rerun where the kit can be read and computed on, then report the full preregistered analysis.",
      "Report TRRUST/DoRothEA-specific and mode-specific results and state what attention direction can imply.",
      "Evaluate within adequately represented cell types."],
  3: ["Rerun with Python access, or classify the run as an execution failure rather than a research answer.",
      "Report cell-type-stratified TF-target evaluations with within-type presence checks.",
      "Condition on rank distance, co-presence, expression mean/variance and dropout."],
  4: ["Rerun with programmatic kit access and report adjusted effect sizes, CIs, nulls, FDR and a verdict.",
      "Check top pairs against TRRUST/DoRothEA and orthogonal binding/perturbation evidence.",
      "Evaluate within cell types / lineages."],
  5: ["Evaluate attention against co-expression, expression/rank, co-presence and degree baselines and label/degree-preserving nulls.",
      "Rerun with kit access and report prespecified scores, adjusted effects, CIs and BH-FDR tests.",
      "The kit has only pooled attention: regenerate per-cell attention or restrict to pooled association (bio lens noting a kit limit)."]},
}

ABSENT_DATA_PAT = re.compile(r"per-cell|cell-type-specific attention|within (adequately|sufficiently)|cell[- ]type(-| )strat|within cell[- ]type|chip-seq|perturb|pretraining|raw (per-cell )?expression|expression variance|phospho|protein|independent (interaction )?network|immune-context|immune-cell-type|context-matched", re.I)


def panel(run, p):
    it = 2 * p
    cp = f"{rundir(run)}/iter_{it:04d}/CONSENSUS.json"
    c = json.load(open(f"{BASE}/{cp}"))
    r = c["report"]
    crit = c["critiques"]
    top = []
    for k in range(3):
        x = crit[k]
        fix = (x["required_fix"] or "").strip()
        first = re.split(r"(?<=[.;])\s", fix)[0]
        top.append({
            "rank": k + 1, "severity": x["severity"], "category": x["category"],
            "raised_by": x["raised_by"], "summary": TOP3[run][p][k],
            "required_fix_first_sentence_verbatim": first[:400],
            "source": cp, "key": f"critiques[{k}]",
        })
    n_absent = sum(1 for x in crit if ABSENT_DATA_PAT.search((x["required_fix"] or "") + " " + x["description"]))
    n_absent_ch = sum(1 for x in crit if x["severity"] in ("critical", "high") and ABSENT_DATA_PAT.search((x["required_fix"] or "") + " " + x["description"]))
    return {
        "step": f"P{p}", "loop_iteration": it,
        "grade": num(c["grade"], cp, "grade"),
        "severity_histogram": num(r["severity_histogram"], cp, "report.severity_histogram"),
        "total_merged_critiques": num(r["total_merged"], cp, "report.total_merged"),
        "n_groups_multi_lens": num(r["n_groups_multi_lens"], cp, "report.n_groups_multi_lens"),
        "raw_critique_counts_per_lens": num(r["raw_critique_counts"], cp, "report.raw_critique_counts"),
        "panel_complete": num(r["panel_complete"], cp, "report.panel_complete"),
        "attempts": num(len(c["attempts"]), cp, "len(attempts)"),
        "critiques_requesting_data_absent_from_kit_heuristic": num(n_absent, cp, "keyword heuristic over critiques[].description+required_fix",
                                                                  "keywords: per-cell / cell-type-stratified / ChIP / perturbation / pretraining / raw expression variance / protein / independent network / immune-context"),
        "of_which_critical_or_high": num(n_absent_ch, cp, "same heuristic, severity critical|high"),
        "top3_required_fixes": top,
        "lens_overall_assessments": num(c["consensus_meta"].get("overall_assessments"), cp, "consensus_meta.overall_assessments"),
    }


# ---------------------------------------------------------------- narrative per executor iteration
def code_ref(run, e, lines, what):
    return {"source": f"{EXEC[run][e]}/code.py", "lines": lines, "what": what}


NARR = {}

# ============================ sol_A
NARR["sol_A"] = {
 1: dict(
  design="Universe: 235 TRRUST TFs x genes with co-presence >= 100 cells (310228 ordered pairs, 423 positives). Scores per layer and per head: log A[TF,g], log A[g,TF] and their mean in log space (log-geometric). Nuisance: L2-logistic model on 16 features (signed/abs Pearson and Spearman, log rank distance, log co-presence, expression, presence, mean rank, and TRRUST-derived TF out-degree / target in-degree), 5-fold TF-grouped CV; conditional one-step score test with TF-cluster sandwich SE; BH over 468 tests; one degree-aware shuffled-attention control per score; DoRothEA (TRRUST edges removed) as 'external' check.",
  estimand="Incremental grouped-CV AUPRC of nuisance+attention over nuisance-only, and the conditional odds ratio per SD of the attention score.",
  numbers=lambda: [
   ("n_candidate_pairs", so("sol_A", 1, "n_candidate_pairs")),
   ("n_trrust_positive_pairs", so("sol_A", 1, "n_trrust_positive_pairs")),
   ("nuisance-only grouped-CV AUROC (includes circular TRRUST degree)", so("sol_A", 1, "baseline_oof_auroc")),
   ("nuisance-only grouped-CV AUPRC", so("sol_A", 1, "baseline_oof_auprc")),
   ("selected record", so("sol_A", 1, "best_record_id")),
   ("selected dAUPRC vs nuisance", so("sol_A", 1, "best_delta_oof_auprc")),
   ("selected conditional OR per SD", so("sol_A", 1, "best_conditional_or_per_sd")),
   ("selected OR 95% CI low (TF-cluster sandwich)", rj("sol_A", 1, "best_trrust_layer_variant/odds_ratio_ci95_low")),
   ("selected OR 95% CI high", rj("sol_A", 1, "best_trrust_layer_variant/odds_ratio_ci95_high")),
   ("selected BH q (468 tests)", so("sol_A", 1, "best_joint_bh_q")),
   ("layer discoveries BH", so("sol_A", 1, "n_layer_discoveries_bh")),
   ("head discoveries BH", so("sol_A", 1, "n_head_discoveries_bh")),
   ("DoRothEA non-overlap dAUPRC", so("sol_A", 1, "external_delta_oof_auprc")),
   ("DoRothEA non-overlap dAUPRC 95% CI (TF bootstrap, 200)", so("sol_A", 1, "external_delta_oof_auprc_ci95")),
   ("raw AUROC, layer 2 log-geometric (max of 36 layer scores)", rj("sol_A", 1, "layer_tests/8/raw_auroc")),
   ("raw AUROC, layer 2 TF-query (validation qTF idx2 = 0.664)", rj("sol_A", 1, "layer_tests/6/raw_auroc")),
   ("univariate |Spearman| AUROC", rj("sol_A", 1, "baseline/univariate/spearman_absolute/auroc")),
   ("univariate degree_sum AUROC (circular)", rj("sol_A", 1, "baseline/univariate/degree_sum/auroc")),
  ],
  flags=lambda: {"all_success_criteria_met": so("sol_A", 1, "all_success_criteria_met")},
  statement="The analysis does not detect attention information beyond the measured co-expression, rank, co-presence, expression, and degree confounders at the pre-registered multiplicity-controlled threshold. This is a null result, not evidence that no such information exists.",
  statement_source="work/MECH.md",
  derivation=("computed", code_ref("sol_A", 1, "522-526", "success = BH q < alpha and dAUPRC > 0 and DoRothEA CI low > 0")),
  changes="Initial iteration.",
  linked=[],
 ),
 2: dict(
  design="Not executed past setup. Planned (from code): leave-one-edge-out degrees, 1000-network degree-preserving edge-swap null, 2000 crossed TF/target bootstraps, stratum analyses, a candidate x test residual memmap.",
  estimand="n/a (crashed before any result)",
  numbers=lambda: [],
  flags=lambda: {},
  statement="No report produced (process exited with OSError [Errno 27] File too large while creating a 310228 x 468 float32 memmap, about 581 MB, above the 64 MiB RLIMIT_FSIZE).",
  statement_source="stderr.txt",
  derivation=("n/a", code_ref("sol_A", 2, "328-334", "np.memmap(shape=(n_candidates, n_attention_tests)) -> exceeds per-file limit")),
  changes="Responded to P1: leave-one-edge-out degrees (P1#2 leakage), degree-preserving network null and permutation-based negative controls (P1 critique[3]), crossed TF/target bootstrap (P1 critique[6]); code grew from 1025 to 2233 lines.",
  linked=["P1#2", "P1 critique[3] negative_control", "P1 critique[6] two-way clustering"],
 ),
 3: dict(
  design="Tests processed one at a time (no memmap). TF-disjoint train / selection / evaluation partitions; gradient-boosted nuisance with leave-source-out target degree, rank-shape, truncation and sequence-length features; association screen of 468 scores (residual contrast in residual-SD units, two-way TF/target cluster variance, Benjamini-Yekutieli); predictive selection of one layer/orientation on the selection partition and evaluation on 26 held-out TFs; one within-TF alignment permutation. Degree-preserving null removed.",
  estimand="Held-out dAUROC/dAUPRC of the selected attention-augmented model over the flexible nuisance model; BY-adjusted association screen.",
  numbers=lambda: [
   ("attention tests screened", so("sol_A", 3, "n_attention_tests")),
   ("BY discoveries", so("sol_A", 3, "n_by_fdr_discoveries")),
   ("strongest screen effect (residual-SD units)", so("sol_A", 3, "best_association_effect")),
   ("its two-way cluster 95% CI (unadjusted for selection)", so("sol_A", 3, "best_association_ci95")),
   ("its BY q", so("sol_A", 3, "best_association_by_q")),
   ("selected record", so("sol_A", 3, "selected_record_id")),
   ("held-out baseline AUROC", rj("sol_A", 3, "predictive_evaluation/baseline_auroc")),
   ("held-out dAUROC", so("sol_A", 3, "heldout_delta_auroc")),
   ("held-out dAUROC 95% CI (crossed bootstrap, 500)", so("sol_A", 3, "heldout_delta_auroc_ci95")),
   ("held-out dAUPRC", so("sol_A", 3, "heldout_delta_auprc")),
   ("held-out dAUPRC 95% CI", so("sol_A", 3, "heldout_delta_auprc_ci95")),
   ("evaluation TFs / positives", rj("sol_A", 3, "predictive_evaluation/n_evaluation_positive_pairs", "84 positives over 26 TFs")),
   ("single within-TF permutation control dAUROC", rj("sol_A", 3, "predictive_negative_control/delta_auroc_vs_nuisance")),
   ("single permutation control dAUPRC", so("sol_A", 3, "negative_control_delta_auprc")),
   ("selection log-loss improvement of the selected (best) variant", rj("sol_A", 3, "predictive_evaluation/selection_balanced_logloss_improvement")),
  ],
  flags=lambda: {"causal_claim_supported": so("sol_A", 3, "causal_claim_supported"),
                 "immune_context_specific_claim_supported": so("sol_A", 3, "immune_context_specific_claim_supported")},
  statement="The post-hoc TF-disjoint evaluation does not establish a robust incremental pooled-attention advantage over the measured flexible nuisance baseline on both annotation-discrimination metrics.",
  statement_source="work/MECH.md",
  derivation=("statement computed (both CI lows > 0 required); causal/immune flags hard-coded False", code_ref("sol_A", 3, "1146-1165", "if dAUPRC CI low > 0 and dAUROC CI low > 0 -> positive statement else null statement")),
  changes="P2#1 fixed (storage redesigned, run succeeded). P2#3 addressed by disjoint selection/evaluation partitions. P2 high critiques on the edge-swap chain (statistic mismatch, mixing) answered by removing the degree-preserving null altogether; context claims removed (P2#2).",
  linked=["P2#1", "P2#2", "P2#3", "P2 critique[5] null_model", "P2 critique[10] statistics"],
 ),
 4: dict(
  design="Selection family of 36 layer/orientation scores plus an explicit no-attention option, one model protocol for selection and refit; TF split with zero-positive TFs in every partition; degrees from training/development labels only; baseline families; 200 within-TF alignment permutations and 100 degree-preserving rewirings (swaps within co-presence x rank-distance tertiles), each rerunning the full pipeline; DoRothEA non-overlap sensitivity; manifest MD5 check.",
  estimand="Held-out dAUROC/dAUPRC of the selected model vs nuisance-only, compared with complete-pipeline alignment and degree-preserving nulls.",
  numbers=lambda: [
   ("selected record", so("sol_A", 4, "selected_record_id")),
   ("held-out baseline AUROC", rj("sol_A", 4, "predictive_evaluation/baseline_auroc")),
   ("held-out dAUROC", so("sol_A", 4, "heldout_delta_auroc")),
   ("held-out dAUPRC", so("sol_A", 4, "heldout_delta_auprc")),
   ("alignment permutations completed", so("sol_A", 4, "alignment_permutations_completed")),
   ("alignment null empirical p (dAUROC)", rj("sol_A", 4, "alignment_permutation_null/one_sided_empirical_p_delta_auroc")),
   ("alignment null empirical p (dAUPRC)", rj("sol_A", 4, "alignment_permutation_null/one_sided_empirical_p_delta_auprc")),
   ("degree rewires completed", so("sol_A", 4, "degree_rewires_completed")),
   ("degree null empirical p (dAUROC)", rj("sol_A", 4, "degree_preserving_null/one_sided_empirical_p_delta_auroc")),
   ("degree null empirical p (dAUPRC)", rj("sol_A", 4, "degree_preserving_null/one_sided_empirical_p_delta_auprc")),
   ("observed-minus-null dAUROC interval (labelled ci95, actually null quantiles)", so("sol_A", 4, "degree_observed_minus_null_delta_auroc_ci95")),
   ("DoRothEA non-overlap sensitivity dAUROC (no CI)", rj("sol_A", 4, "dorothea_nonoverlap_sensitivity/delta_auroc")),
   ("DoRothEA non-overlap sensitivity dAUPRC (no CI)", rj("sol_A", 4, "dorothea_nonoverlap_sensitivity/delta_auprc")),
   ("manifest MD5 entries checked (bug: 0)", rj("sol_A", 4, "input_integrity/manifest_md5_entries_checked")),
   ("input_manifest_all_checked_md5_match", so("sol_A", 4, "input_manifest_all_checked_md5_match")),
  ],
  flags=lambda: {"causal_claim_supported": so("sol_A", 4, "causal_claim_supported"),
                 "immune_context_specific_claim_supported": so("sol_A", 4, "immune_context_specific_claim_supported"),
                 "tf_to_target_directionality_supported": so("sol_A", 4, "tf_to_target_directionality_supported")},
  statement="The corrected post-hoc analysis permits the nuisance-only option and tests pooled attention only as context-agnostic annotation discrimination. It cannot establish active immune TF-to-target regulation or causality ...",
  statement_source="work/results.json verdict.statement",
  derivation=("hard-coded (all verdict flags and the statement are literals)", code_ref("sol_A", 4, "1298-1310", "verdict dict with literal False flags and fixed statement")),
  changes="P3#2 fixed (no-attention option, consistent learner). P3#3 fixed (200 complete-pipeline permutations). P3 high critiques fixed: outcome-conditioned TF split, transductive degree leakage; degree-preserving null re-introduced because P3 flagged its absence (it had been removed in E3 after P2 criticised it). P3#1 (cell-type attention) answered only by an 'unavailable_analyses' entry. PATCH_NOTES claimed input hashes were verified, but 0 manifest entries were checked.",
  linked=["P3#1", "P3#2", "P3#3", "P3 critique[6] leakage", "P3 critique[7] null_model", "P3 critique[8] leakage"],
 ),
 5: dict(
  design="15 repeated nested TF-group splits (each redraws negatives, refits nuisance + stacking, reselects among 36 scores + no-attention); multiverse of 12 configurations x 3 splits; separate TF-query-only family; 15 alignment permutations and 10 degree-preserving rewirings (swaps within binary joint strata of co-presence, rank distance, target expression, |coexpression|), each running ONE nested split; cell-type/donor/tissue co-presence fractions added to the nuisance model; manifest parsing repaired; DoRothEA predictive metric retired.",
  estimand="Mean over 15 outer splits of held-out dAUROC/dAUPRC (nuisance+selected attention vs nuisance), with t-intervals over splits and empirical null p-values.",
  numbers=lambda: [
   ("repeated nested splits", so("sol_A", 5, "repeated_nested_splits_completed")),
   ("mean baseline AUROC", rj("sol_A", 5, "primary_nested_evaluation/mean_baseline_auroc")),
   ("mean dAUROC", so("sol_A", 5, "mean_delta_auroc")),
   ("dAUROC repeated-split 95% CI", so("sol_A", 5, "delta_auroc_repeated_split_mean_ci95")),
   ("mean dAUPRC", so("sol_A", 5, "mean_delta_auprc")),
   ("dAUPRC repeated-split 95% CI (excludes 0)", so("sol_A", 5, "delta_auprc_repeated_split_mean_ci95")),
   ("TF-query-only mean dAUROC", rj("sol_A", 5, "primary_orientation_only/mean_delta_auroc")),
   ("TF-query-only dAUROC 95% CI", rj("sol_A", 5, "primary_orientation_only/delta_auroc_repeated_split_mean_ci95")),
   ("TF-query-only mean dAUPRC", rj("sol_A", 5, "primary_orientation_only/mean_delta_auprc")),
   ("TF-query-only dAUPRC 95% CI", rj("sol_A", 5, "primary_orientation_only/delta_auprc_repeated_split_mean_ci95")),
   ("alignment null replicates", rj("sol_A", 5, "alignment_permutation_null/replicates_completed")),
   ("alignment null p (dAUROC)", so("sol_A", 5, "alignment_empirical_p_auroc")),
   ("alignment null p (dAUPRC)", so("sol_A", 5, "alignment_empirical_p_auprc")),
   ("degree null replicates", rj("sol_A", 5, "degree_preserving_null/replicates_completed")),
   ("degree null p (dAUROC)", so("sol_A", 5, "degree_empirical_p_auroc")),
   ("degree null p (dAUPRC)", so("sol_A", 5, "degree_empirical_p_auprc")),
   ("multiverse configurations positive on both metrics (of 12)", rj("sol_A", 5, "multiverse/configurations_positive_for_both_mean_metrics")),
   ("manifest md5 entries matching", so("sol_A", 5, "manifest_md5_entries_matching")),
  ],
  flags=lambda: {"incremental_information_beyond_measured_nuisance_detected": so("sol_A", 5, "incremental_information_beyond_measured_nuisance_detected"),
                 "dorothea_independent_corroboration_supported": so("sol_A", 5, "dorothea_independent_corroboration_supported"),
                 "causal_claim_supported": so("sol_A", 5, "causal_claim_supported"),
                 "tf_to_target_directionality_supported": rj("sol_A", 5, "verdict/tf_to_target_directionality_supported")},
  statement="For this dataset and the context-agnostic TRRUST annotation endpoint, the repeated nested analysis found no supported incremental information from pooled Geneformer attention beyond the measured nuisance model.",
  statement_source="work/results.json verdict.statement",
  derivation=("hard-coded: verdict flags and statement are literals written before the run; the computed dAUPRC CI [0.0017, 0.0076] excludes 0 and did not affect the verdict", code_ref("sol_A", 5, "1438-1455, 1833-1837", "verdict dict literals; print('incremental_information_beyond_measured_nuisance_detected=False')")),
  changes="P4#2 implemented literally: the verdict was changed to an explicit negative result, but as a hard-coded literal. P4#1 answered by retiring the DoRothEA metric (E4 had shown DoRothEA dAUROC +0.029, dAUPRC +0.014). P4 critiques on the inspected fixed split / multiverse (critique[3]) and the mislabelled 'ci95' null quantiles (critique[4]) were fixed; the manifest bug (critique[5]) was fixed (10/10). The null replicate counts fell from 200/100 (E4) to 15/10.",
  linked=["P4#1", "P4#2", "P4 critique[3] p_hacking", "P4 critique[4] statistical_theater", "P4 critique[5] reproducibility"],
 ),
}

# ============================ sol_B
NARR["sol_B"] = {
 1: dict(
  design="Matched case-control design: each TRRUST edge (424) and each DoRothEA edge not in TRRUST (1949) matched to 5 database-absent same-TF controls by Euclidean nearest neighbour on 9 covariates (signed/abs Pearson, Spearman, log rank distance, log co-presence, target expression, target presence, target mean rank, log target in-degree); co-presence >= 30. Effect = conditional AUC - 0.5 of each score (12 layer means + 144 heads x forward / reverse / symmetric / directional = 624 per benchmark); TF-cluster bootstrap (2000), TF sign-flip test (10000), BH over 1248 tests; 'replicated' = BH-supported positive in both benchmarks.",
  estimand="Within-TF matched conditional AUC - 0.5 (positives vs matched controls) per attention score.",
  numbers=lambda: [
   ("TRRUST matched positives", so("sol_B", 1, "trrust_matched_positive_edges")),
   ("DoRothEA-exclusive matched positives", so("sol_B", 1, "dorothea_matched_positive_edges")),
   ("BH family size", so("sol_B", 1, "multiple_testing_family_size")),
   ("TRRUST BH-supported tests", so("sol_B", 1, "trrust_supported_tests")),
   ("DoRothEA BH-supported tests", so("sol_B", 1, "dorothea_supported_tests")),
   ("replicated tests", so("sol_B", 1, "replicated_test_count")),
   ("best TRRUST effect (head_L03_H05_forward)", rj("sol_B", 1, "benchmark_summary/trrust/best_effect")),
   ("its TF-cluster 95% CI low", rj("sol_B", 1, "benchmark_summary/trrust/best_ci_low")),
   ("its CI high", rj("sol_B", 1, "benchmark_summary/trrust/best_ci_high")),
   ("best DoRothEA effect (layer_mean_L03_forward)", rj("sol_B", 1, "benchmark_summary/dorothea_exclusive/best_effect")),
   ("its CI low", rj("sol_B", 1, "benchmark_summary/dorothea_exclusive/best_ci_low")),
   ("its CI high", rj("sol_B", 1, "benchmark_summary/dorothea_exclusive/best_ci_high")),
   ("residual imbalance: target in-degree cAUC-0.5 after matching (TRRUST)", rj("sol_B", 1, "matching_diagnostics/trrust.target_in_degree/effect")),
   ("residual imbalance: target in-degree (DoRothEA)", rj("sol_B", 1, "matching_diagnostics/dorothea_exclusive.target_in_degree/effect")),
   ("max |SMD| after matching (TRRUST)", rj("sol_B", 1, "benchmark_summary/trrust/maximum_absolute_matched_smd")),
   ("max |SMD| after matching (DoRothEA)", rj("sol_B", 1, "benchmark_summary/dorothea_exclusive/maximum_absolute_matched_smd")),
   ("descriptive raw AUROC TRRUST layer_mean_L02_forward", rj("sol_B", 1, "descriptive_universe_metrics/trrust/layer_mean_L02_forward/auroc")),
   ("TRRUST layer-mean directional test L01 effect (BH-supported)", rj("sol_B", 1, "attention_tests/trrust.layer_mean_L01_directional/effect")),
   ("DoRothEA layer-mean directional test L03 effect (BH-supported)", rj("sol_B", 1, "attention_tests/dorothea_exclusive.layer_mean_L03_directional/effect")),
  ],
  flags=lambda: {"verdict": so("sol_B", 1, "verdict")},
  statement="At least one pre-specified attention score had a positive, BH-controlled matched effect in both benchmarks. This is direct evidence of replicated association beyond the measured matching variables, but it is not evidence that attention causally implements regulation.",
  statement_source="work/MECH.md",
  derivation=("computed", code_ref("sol_B", 1, "707-728", "verdict_code from replicated_test_ids / supported_by_benchmark")),
  changes="Initial iteration.",
  linked=[],
 ),
 2: dict(
  design="Strict 1:1 matching with exact union-database target in-degree, 0.5-SD calipers and cell-type/tissue/donor co-presence profiles; balance gate |SMD| <= 0.1; locked primary score (layer-0 forward) with all other scores exploratory; matched-label permutation (4000); partial conjunction across benchmarks; multiway bootstrap; held-out-TF nuisance vs nuisance+attention; degree-preserving graph rewiring (200) on unmatched mean log attention.",
  estimand="Locked layer-0 forward matched effect in each benchmark and its partial-conjunction p; held-out incremental AUC.",
  numbers=lambda: [
   ("TRRUST matched edges (of 424)", so("sol_B", 2, "trrust_matched_positive_edges")),
   ("DoRothEA matched edges (of 1949)", so("sol_B", 2, "dorothea_matched_positive_edges")),
   ("TRRUST max |SMD|", so("sol_B", 2, "trrust_maximum_absolute_smd")),
   ("DoRothEA max |SMD|", so("sol_B", 2, "dorothea_maximum_absolute_smd")),
   ("TRRUST primary effect", so("sol_B", 2, "trrust_primary_effect")),
   ("TRRUST primary 95% CI low", rj("sol_B", 2, "primary_prespecified/trrust/ci_low")),
   ("TRRUST primary 95% CI high", rj("sol_B", 2, "primary_prespecified/trrust/ci_high")),
   ("DoRothEA primary effect", so("sol_B", 2, "dorothea_primary_effect")),
   ("DoRothEA primary 95% CI low", rj("sol_B", 2, "primary_prespecified/dorothea_exclusive/ci_low")),
   ("DoRothEA primary 95% CI high", rj("sol_B", 2, "primary_prespecified/dorothea_exclusive/ci_high")),
   ("partial-conjunction p", so("sol_B", 2, "primary_partial_conjunction_p")),
   ("held-out incremental AUC TRRUST", so("sol_B", 2, "trrust_incremental_heldout_auc")),
   ("held-out incremental AUC DoRothEA", so("sol_B", 2, "dorothea_incremental_heldout_auc")),
   ("degree-preserving null p (unmatched mean log attention, TRRUST, 200 rewires)", rj("sol_B", 2, "degree_preserving_graph_null/trrust/rewired_p_value")),
   ("observed macro mean log attention TRRUST", rj("sol_B", 2, "degree_preserving_graph_null/trrust/observed_macro_mean_log_attention")),
   ("rewired null mean TRRUST", rj("sol_B", 2, "degree_preserving_graph_null/trrust/rewired_null_mean")),
  ],
  flags=lambda: {"verdict": so("sol_B", 2, "verdict")},
  statement="The supplied aggregate matrices do not establish that attention encodes TF-to-target regulation beyond the named confounders ... This is an inconclusive result, not evidence of absence.",
  statement_source="work/MECH.md",
  derivation=("hard-coded: code comment 'even favorable score statistics cannot answer the research question affirmatively'", code_ref("sol_B", 2, "1429-1439", "verdict_code = 'not_established_with_supplied_aggregates' unconditionally")),
  changes="P1 critiques[5]/[6] (verdict contradicted by residual target in-degree imbalance) answered by exact degree matching + calipers + balance gate; P1#3 (identical cells) and P1#2 (cell-type strata) answered by context-profile matching and an unconditional 'not established' verdict; P1#1 by a provenance audit; P1 critique[4] by partial conjunction. Strict matching retained 9/424 TRRUST edges.",
  linked=["P1#1", "P1#2", "P1#3", "P1 critique[4] p_hacking", "P1 critique[5] overclaiming", "P1 critique[6] stat_bio_alignment"],
 ),
 3: dict(
  design="Same-TF global optimal assignment retaining all positives; TF-macro balance; target incoming-attention centrality and additive row/column attention covariates (derived from layer-0 attention); constrained degree-preserving swaps (must preserve co-presence/correlation/rank/context strata); 20x repeated grouped CV; all orientations exploratory; Benjamini-Yekutieli and joint max-statistic across 624 scores x 2 benchmarks.",
  estimand="Aggregate matched effect (cAUC - 0.5) by orientation; repeated-CV incremental AUC.",
  numbers=lambda: [
   ("TRRUST matched edges", so("sol_B", 3, "trrust_matched_edges")),
   ("TRRUST max TF-macro |SMD|", so("sol_B", 3, "trrust_maximum_absolute_tf_macro_smd")),
   ("DoRothEA max TF-macro |SMD|", so("sol_B", 3, "dorothea_exclusive_maximum_absolute_tf_macro_smd")),
   ("TRRUST forward aggregate effect", so("sol_B", 3, "trrust_forward_aggregate_effect")),
   ("its 95% CI low", rj("sol_B", 3, "aggregate_attention_associations/trrust/forward/ci_low")),
   ("its 95% CI high", rj("sol_B", 3, "aggregate_attention_associations/trrust/forward/ci_high")),
   ("DoRothEA forward aggregate effect", so("sol_B", 3, "dorothea_exclusive_forward_aggregate_effect")),
   ("its 95% CI low", rj("sol_B", 3, "aggregate_attention_associations/dorothea_exclusive/forward/ci_low")),
   ("its 95% CI high", rj("sol_B", 3, "aggregate_attention_associations/dorothea_exclusive/forward/ci_high")),
   ("TRRUST held-out increment mean (20x grouped CV)", so("sol_B", 3, "trrust_heldout_auc_increment_mean")),
   ("DoRothEA held-out increment mean", so("sol_B", 3, "dorothea_exclusive_heldout_auc_increment_mean")),
   ("constrained degree-null accepted swaps TRRUST (of 21200 attempts)", rj("sol_B", 3, "constrained_degree_preserving_null/trrust/accepted_swaps")),
   ("joint-family supported scores", so("sol_B", 3, "exploratory_joint_family_supported_count")),
  ],
  flags=lambda: {"verdict": so("sol_B", 3, "verdict"), "analysis_script_saved": so("sol_B", 3, "analysis_script_saved")},
  statement="The supplied kit does not identify whether Geneformer attention encodes active TF-to-target regulation beyond the named confounders. Attention was already averaged over pair-specific co-present cell sets ...",
  statement_source="work/MECH.md",
  derivation=("hard-coded", code_ref("sol_B", 3, "1716", "verdict_code = 'not_identifiable_from_supplied_aggregate_attention' unconditionally")),
  changes="P2#3 (9-edge estimand) answered by optimal matching that keeps all edges at the cost of balance (max |SMD| rose to 1.279); P2#2 answered by nuisance-constrained swaps (0/424 accepted, partly a code defect); P2#1 (identical cells) answered by declaring non-identifiability. Attention-derived covariates were added to the matching (later flagged as leakage).",
  linked=["P2#1", "P2#2", "P2#3"],
 ),
 4: dict(
  design="Caliper matching on non-attention covariates with exact union-database in-degree, balance gate and minimum-sample gates; effects withheld when gates fail; TF-cluster out-of-bag bootstrap that refits the prediction pipeline; corrected graph-swap slot signatures (not run because gate failed); script saved with SHA-256 and a --self-check rerun.",
  estimand="Balanced matched effect (withheld) and OOB-bootstrap held-out incremental AUC.",
  numbers=lambda: [
   ("TRRUST edges after balance trimming", so("sol_B", 4, "trrust_post_balance_trim_edges")),
   ("TRRUST TF clusters", so("sol_B", 4, "trrust_tf_clusters")),
   ("TRRUST max TF-macro |SMD|", so("sol_B", 4, "trrust_maximum_absolute_tf_macro_smd")),
   ("TRRUST incremental AUC mean", so("sol_B", 4, "trrust_attention_increment_mean")),
   ("TRRUST incremental AUC CI low (41 draws)", so("sol_B", 4, "trrust_attention_increment_ci_low")),
   ("TRRUST incremental AUC CI high", so("sol_B", 4, "trrust_attention_increment_ci_high")),
   ("DoRothEA edges after trimming", so("sol_B", 4, "dorothea_exclusive_post_balance_trim_edges")),
   ("DoRothEA max TF-macro |SMD|", so("sol_B", 4, "dorothea_exclusive_maximum_absolute_tf_macro_smd")),
   ("DoRothEA incremental AUC mean", so("sol_B", 4, "dorothea_exclusive_attention_increment_mean")),
   ("DoRothEA CI low", so("sol_B", 4, "dorothea_exclusive_attention_increment_ci_low")),
   ("DoRothEA CI high", so("sol_B", 4, "dorothea_exclusive_attention_increment_ci_high")),
  ],
  flags=lambda: {"verdict": so("sol_B", 4, "verdict"), "trrust_balance_gate_passed": so("sol_B", 4, "trrust_balance_gate_passed"),
                 "dorothea_exclusive_balance_gate_passed": so("sol_B", 4, "dorothea_exclusive_balance_gate_passed")},
  statement="The supplied aggregate kit does not identify TF-to-target regulation beyond co-expression, rank/expression structure and degree. ... Aggregate effects are withheld when the prespecified balance gate fails.",
  statement_source="work/results.json verdict.text",
  derivation=("hard-coded", code_ref("sol_B", 4, "1681", "verdict_code = 'not_identifiable_from_supplied_aggregate_attention' unconditionally")),
  changes="P3#2 fixed (attention-derived covariates removed). P3#3 fixed in code (invariant slot signatures) but the null was not executed. P3#1 implemented as a hard balance gate, which failed (TRRUST collapsed to 4 edges). P3 critique[12] fixed (script saved). Held-out prediction ran on 4 TRRUST edges, giving a degenerate [0, 0] interval.",
  linked=["P3#1", "P3#2", "P3#3", "P3 critique[12] reproducibility"],
 ),
 5: dict(
  design="Planned (from code): corrected degree-preserving graph null on all-layer forward log attention with tertile nuisance constraints (30 draws requested per benchmark), biological audit of all edges, self-copy of the script via Path(__file__).",
  estimand="n/a (crashed)",
  numbers=lambda: [],
  flags=lambda: {},
  statement="No report produced: NameError: name '__file__' is not defined (line 1012) after 72.2 s; partial files: 7 TRRUST graph-null draws, audit tables.",
  statement_source="stderr.txt",
  derivation=("hard-coded in the (unexecuted) results dict", code_ref("sol_B", 5, "987", "'code': 'not_identifiable_from_supplied_aggregate_attention'")),
  changes="Responded to P4#2/P4#3 by replacing the matched prediction with a graph-null contrast, and to reproducibility requests by self-copying the script; the new self-copy used __file__, which is undefined when the orchestrator runs the code, so the run crashed (E4 had worked around this).",
  linked=["P4#2", "P4#3", "P4 critique[18] reproducibility"],
 ),
}

# ============================ gpt55
NARR["gpt55"] = {
 1: dict(
  design="Universe: 126 TRRUST TFs with >= 1 edge x co-present genes (166446 pairs, 424 positives). 468 scores (36 layer means + 432 heads; tf_query, target_query, sym_mean): raw AUROC/AUPRC; linear residualisation on 12 covariates incl. TRRUST-derived degree; residual SMD with pair-level Welch t-test; BH over 468; TF bootstrap (300) for the selected variant; within-TF label permutation (200); reversed-edge control; DoRothEA transfer.",
  estimand="Residual standardized mean difference (TRRUST positives vs candidates) of each attention score after linear covariate residualisation; selected variant = smallest residual p.",
  numbers=lambda: [
   ("n_candidate_pairs", so("gpt55", 1, "n_candidate_pairs")),
   ("BH-significant tests (q<0.05)", so("gpt55", 1, "n_bh_significant_q_lt_0_05")),
   ("of which positive residual SMD", tsv_count("gpt55", 1, positive=True)),
   ("selected (smallest p) variant", so("gpt55", 1, "best_attention_variant")),
   ("its residual SMD (negative)", so("gpt55", 1, "best_attention_residual_smd")),
   ("its BH q", so("gpt55", 1, "best_attention_residual_q_bh_all_attention_tests")),
   ("its SMD 95% CI low (TF bootstrap)", so("gpt55", 1, "best_all_residual_smd_ci_low")),
   ("its SMD 95% CI high", so("gpt55", 1, "best_all_residual_smd_ci_high")),
   ("positive-SMD q<0.05 example head_L1_H5_target_query SMD", tsv("gpt55", 1, "attention_tests.tsv", "variant", "head_L1_H5_target_query", "residual_smd")),
   ("... its q", tsv("gpt55", 1, "attention_tests.tsv", "variant", "head_L1_H5_target_query", "residual_q_bh_all_attention_tests")),
   ("max layer-mean raw AUROC (layer_mean_L2_tf_query)", tsv("gpt55", 1, "attention_tests.tsv", "variant", "layer_mean_L2_tf_query", "auroc")),
   ("baseline |Pearson| AUROC", so("gpt55", 1, "baseline_coexpr_abs_pearson_auroc")),
   ("baseline rank proximity AUROC", so("gpt55", 1, "baseline_rank_proximity_auroc")),
   ("baseline degree-only AUROC (circular)", so("gpt55", 1, "baseline_degree_only_auroc")),
   ("combined confound logistic AUROC (in-sample)", so("gpt55", 1, "baseline_combined_confound_auroc")),
   ("reversed-edge control AUROC", so("gpt55", 1, "negative_control_reversed_edges_auroc")),
   ("DoRothEA transfer AUROC", so("gpt55", 1, "dorothea_transfer_auroc")),
   ("within-TF permutation p (200)", so("gpt55", 1, "negative_control_within_tf_permutation_p")),
  ],
  flags=lambda: {},
  statement="The verdict is: does not provide BH-FDR significant evidence that attention carries TF-target signal beyond measured confounders. This is correlational evidence from precomputed attention summaries, not a causal intervention on the model.",
  statement_source="work/MECH.md",
  derivation=("computed, but from the smallest-p variant regardless of sign", code_ref("gpt55", 1, "359-363", "positive verdict iff best-by-p variant has q<0.05 AND SMD>0")),
  changes="Initial iteration.",
  linked=[],
 ),
 2: dict(
  design="Planned: TF-cluster sign-flip permutation p per test (1999), LOO TRRUST degree, matched within-TF null (500), family max-statistic (80), bootstrap 200, separate best-positive / best-negative selection.",
  estimand="TF-balanced residual SMD with cluster-aware p (not completed).",
  numbers=lambda: [
   ("partial attention_tests.tsv: min BH q (TF-cluster sign-flip)", tsv_min("gpt55", 2, "attention_tests.tsv", "tf_cluster_permutation_q_bh_all_attention_tests")),
   ("partial baseline: LOO-degree AUROC", tsv("gpt55", 2, "baseline_metrics.tsv", "score", "loo_degree_only", "auroc")),
   ("partial baseline: LOO-degree AUPRC", tsv("gpt55", 2, "baseline_metrics.tsv", "score", "loo_degree_only", "auprc")),
   ("partial baseline: combined confound held-out-TF CV AUROC", tsv("gpt55", 2, "baseline_metrics.tsv", "score", "combined_confound_logistic_heldout_tf_cv", "auroc")),
  ],
  flags=lambda: {},
  statement="No report produced: killed at the 900 s wall-clock limit (SIGKILL); only attention_tests.tsv and baseline_metrics.tsv were written.",
  statement_source="CODE_EXECUTION.json",
  derivation=("n/a", code_ref("gpt55", 2, "25-28", "N_BOOT=200, N_CLUSTER_PERM=1999, N_MAXSTAT_PERM=80, N_MATCHED_PERM=500")),
  changes="P1#1 (cluster-aware tests), P1#2 (LOO degree) and P1#3 (separate positive/negative selection) all implemented; the permutation load exceeded the 900 s limit.",
  linked=["P1#1", "P1#2", "P1#3"],
 ),
 3: dict(
  design="Vectorised, reduced permutations with stage checkpoints; TF-balanced (per-TF mean) residual SMD as primary with TF-cluster permutation p and BH over 468; separate best positive / best negative; LOO-degree baseline; held-out-TF incremental CV; matched within-TF null (~100) and family max-statistic (30).",
  estimand="TF-balanced residual SMD of the best positive variant; success requires BH q<0.05, matched p<0.05, max-stat p<0.05 and AUPRC above the LOO-degree baseline by a practical threshold.",
  numbers=lambda: [
   ("BH-significant positive tests", so("gpt55", 3, "n_bh_significant_positive_q_lt_0_05")),
   ("BH-significant negative tests", so("gpt55", 3, "n_bh_significant_negative_q_lt_0_05")),
   ("best positive variant", so("gpt55", 3, "best_positive_variant")),
   ("its TF-balanced SMD", so("gpt55", 3, "best_positive_tf_cluster_mean_smd")),
   ("its SMD 95% CI low (post-selection)", rj("gpt55", 3, "best_positive_tf_cluster_mean_smd_ci_low")),
   ("its SMD 95% CI high", rj("gpt55", 3, "best_positive_tf_cluster_mean_smd_ci_high")),
   ("its BH q", so("gpt55", 3, "best_positive_tf_cluster_permutation_q_bh_all_attention_tests")),
   ("matched within-TF null p", so("gpt55", 3, "negative_control_matched_within_tf_permutation_best_positive_p")),
   ("family max-stat p (30 permutations)", so("gpt55", 3, "negative_control_maxstat_family_permutation_best_positive_p")),
   ("best positive AUPRC", so("gpt55", 3, "best_positive_auprc")),
   ("LOO-degree AUPRC", so("gpt55", 3, "baseline_loo_degree_only_auprc")),
   ("incremental CV dAUPRC (attention+confounds - confounds)", so("gpt55", 3, "incremental_cv_best_positive_attention_minus_confound_auprc_mean_bootstrap")),
  ],
  flags=lambda: {"primary_positive_success_after_bh_matched_maxstat_and_degree_threshold": so("gpt55", 3, "primary_positive_success_after_bh_matched_maxstat_and_degree_threshold"),
                 "degree_baseline_auprc_exceeds_best_positive_attention_auprc": so("gpt55", 3, "degree_baseline_auprc_exceeds_best_positive_attention_auprc")},
  statement="Primary verdict: No positive attention association passed the full pre-specified evidence bar after BH-FDR, matched-null, max-statistic, and degree-baseline checks. ... The leave-one-edge-out degree baseline exceeded the selected positive attention AUPRC, so hub/database structure is the dominant verified signal.",
  statement_source="work/MECH.md",
  derivation=("computed", code_ref("gpt55", 3, "744-752, 873-877", "positive_success conjunction; verdict string conditional on it")),
  changes="P2#1 fixed (runtime reduced, checkpoints). P2#3 implemented (family max-stat, but only 30 permutations). P2 critique[9] fixed (TF-balanced estimand used for both effect and test). Cluster-aware inference removed all positive BH-significant results seen in E1.",
  linked=["P2#1", "P2#3", "P2 critique[9] claim_validity"],
 ),
 4: dict(
  design="As E3 plus 4 cell-type composition covariates, 499-permutation family max-statistic, 100 matched within-TF permutations, nested attention selection inside held-out-TF CV, split-aware held-out-TF degree baseline, degree relabelled 'TRRUST topology null', kit manifest md5.",
  estimand="Same as E3; plus nested-selection held-out-TF incremental AUPRC.",
  numbers=lambda: [
   ("BH-significant positive tests", so("gpt55", 4, "n_bh_significant_positive_q_lt_0_05")),
   ("BH-significant negative tests", so("gpt55", 4, "n_bh_significant_negative_q_lt_0_05")),
   ("best positive variant", so("gpt55", 4, "best_positive_variant")),
   ("its TF-balanced SMD", so("gpt55", 4, "best_positive_tf_cluster_mean_smd")),
   ("its SMD 95% CI low (post-selection, 80 bootstraps)", rj("gpt55", 4, "best_positive_tf_cluster_mean_smd_ci_low")),
   ("its SMD 95% CI high", rj("gpt55", 4, "best_positive_tf_cluster_mean_smd_ci_high")),
   ("its BH q", so("gpt55", 4, "best_positive_tf_cluster_permutation_q_bh_all_attention_tests")),
   ("matched within-TF null p (100)", so("gpt55", 4, "negative_control_matched_within_tf_permutation_best_positive_p")),
   ("family max-stat p (499)", so("gpt55", 4, "negative_control_maxstat_family_permutation_best_positive_p")),
   ("best positive AUROC", rj("gpt55", 4, "best_positive_auroc")),
   ("best positive AUPRC", so("gpt55", 4, "best_positive_auprc")),
   ("prevalence", so("gpt55", 4, "candidate_positive_rate")),
   ("LOO-degree ('topology null') AUROC", rj("gpt55", 4, "baseline_loo_degree_only_trrust_topology_null_auroc")),
   ("LOO-degree AUPRC", so("gpt55", 4, "baseline_loo_degree_only_trrust_topology_null_auprc")),
   ("split-aware held-out-TF degree AUPRC", so("gpt55", 4, "baseline_split_aware_degree_only_heldout_tf_cv_auprc")),
   ("nested-selection incremental CV dAUPRC", so("gpt55", 4, "nested_selection_incremental_cv_attention_minus_confound_auprc_mean_bootstrap")),
   ("its 95% CI low", rj("gpt55", 4, "nested_selection_incremental_cv_attention_minus_confound_auprc_ci_low")),
   ("its 95% CI high", rj("gpt55", 4, "nested_selection_incremental_cv_attention_minus_confound_auprc_ci_high")),
   ("reversed-edge control AUROC (selected head)", rj("gpt55", 4, "negative_control_reversed_edges_best_positive_auroc")),
   ("DoRothEA transfer AUROC (selected head)", rj("gpt55", 4, "dorothea_transfer_best_positive_auroc")),
   ("|Pearson| baseline AUROC", rj("gpt55", 4, "baseline_coexpr_abs_pearson_auroc")),
   ("max layer-mean raw AUROC (layer_mean_L2_tf_query)", tsv("gpt55", 4, "attention_tests.tsv", "variant", "layer_mean_L2_tf_query", "auroc")),
  ],
  flags=lambda: {"primary_positive_success_after_bh_matched_maxstat_and_degree_threshold": so("gpt55", 4, "primary_positive_success_after_bh_matched_maxstat_and_degree_threshold"),
                 "degree_topology_null_auprc_exceeds_best_positive_attention_auprc": so("gpt55", 4, "degree_topology_null_auprc_exceeds_best_positive_attention_auprc"),
                 "split_aware_degree_heldout_tf_auprc_exceeds_best_positive_attention_auprc": so("gpt55", 4, "split_aware_degree_heldout_tf_auprc_exceeds_best_positive_attention_auprc")},
  statement="Primary verdict: No positive attention association passed the full evidence bar after BH-FDR, matched-null, powered max-statistic, and degree-baseline checks. ... This is evidence that TRRUST label topology/degree dominates the verified predictive signal, not proof of biological hub regulation in the sampled cells.",
  statement_source="work/MECH.md",
  derivation=("computed", code_ref("gpt55", 4, "956-964, 1114-1118", "positive_success requires SMD>0, q<0.05, attention AUPRC - LOO-degree AUPRC > 0.001, max-stat p<0.05, matched p<0.05")),
  changes="P3#2 fixed (max-stat 30 -> 499). P3#3 fixed (CIs labelled post-selection; nested selection added). P3 critique[3] (degree relabelled as topology null + split-aware degree) and critique[4] (cell-type composition covariates) implemented. P3#1 answered only by documenting unavailability.",
  linked=["P3#1", "P3#2", "P3#3", "P3 critique[3] leakage", "P3 critique[4] confounder"],
 ),
 5: dict(
  design="Planned: 199 within-TF label permutations per test as primary p, 1000 matched permutations with continuous nearest-neighbour matching, fold-local topology features in held-out-TF CV, biological audit tables.",
  estimand="Same as E4 (not completed).",
  numbers=lambda: [
   ("partial attention_tests.tsv: min BH q", tsv_min("gpt55", 5, "attention_tests.tsv", "primary_within_tf_label_permutation_q_bh_all_attention_tests")),
   ("partial: max attention AUPRC (head_L2_H5_sym_mean)", tsv("gpt55", 5, "attention_tests.tsv", "variant", "head_L2_H5_sym_mean", "auprc")),
   ("partial: fold-local leakage-free confound CV AUROC", tsv("gpt55", 5, "baseline_metrics.tsv", "score", "combined_confound_logistic_heldout_tf_cv_leakage_free_fold_local_topology", "auroc")),
   ("partial: confound + best positive attention CV AUROC", tsv("gpt55", 5, "baseline_metrics.tsv", "score", "confound_plus_best_positive_attention_heldout_tf_cv_leakage_free_fold_local_topology", "auroc")),
   ("last checkpoint elapsed s (stage matched_null_continuous_nn)", num(json.load(open(f"{BASE}/{EXEC['gpt55'][5]}/work/checkpoint_status.json"))["elapsed_seconds"], f"{EXEC['gpt55'][5]}/work/checkpoint_status.json", "elapsed_seconds")),
  ],
  flags=lambda: {},
  statement="No report produced: killed at the 900 s wall-clock limit during the continuous nearest-neighbour matched null.",
  statement_source="CODE_EXECUTION.json; work/checkpoint_status.json",
  derivation=("n/a", code_ref("gpt55", 5, "29-32", "N_LABEL_PERM=199, N_BOOT=60, N_MATCHED_PERM=1000, N_CV_BOOT=60")),
  changes="P4#2 implemented (1000 matched permutations, continuous NN matching) and P4#3 implemented (fold-local topology; its partial output shows confounds+attention CV AUROC below confounds-only). The added matched-null cost exceeded the 900 s limit.",
  linked=["P4#2", "P4#3"],
 ),
}

# plan-only
PLAN_NOTE = {
 1: "Refused: workspace read-only, no shell/Python; asked whether to proceed with a plan only. No plan, no numbers.",
 3: "Refused again; offered a 'plan-only, non-evidential response' if explicitly authorised. No plan, no numbers.",
 5: "Refused; listed available tools as apply_patch, update_plan and image viewing; requested shell/Python, kit read access and write access.",
 7: "Labelled 'Execution failure'; no scientific verdict; listed minimum capabilities for a rerun.",
 9: "Formally marked the iteration 'FAILED and excluded from scientific synthesis'; listed analyses it could not compute.",
}


def tsv_count(run, e, positive=True):
    p = f"{EXEC[run][e]}/work/attention_tests.tsv"
    t = pd.read_csv(f"{BASE}/{p}", sep="\t")
    q = "residual_q_bh_all_attention_tests"
    n = int(((t[q] < 0.05) & ((t.residual_smd > 0) if positive else (t.residual_smd < 0))).sum())
    return num(n, p, f"count({q}<0.05 & residual_smd{'>' if positive else '<'}0)")


def tsv_min(run, e, fname, col):
    p = f"{EXEC[run][e]}/work/{fname}"
    t = pd.read_csv(f"{BASE}/{p}", sep="\t")
    return num(float(t[col].min()), p, f"min({col})")


def build_iter(run, e):
    rec = exec_mech(run, e)
    if run == "sol_planonly":
        it = 2 * e - 1
        rec["what_was_computed"] = "Nothing (no code, no numbers)."
        rec["behaviour"] = num(PLAN_NOTE[it], f"{rundir(run)}/iter_{it:04d}/executor_output.md", "full text")
        return rec
    n = NARR[run][e]
    rec["design"] = n["design"]
    rec["primary_estimand"] = n["estimand"]
    rec["key_numbers"] = [dict(label=l, **v) for l, v in n["numbers"]()]
    rec["report_flags"] = n["flags"]()
    rec["report_statement_quote"] = {"text": n["statement"], "source": f"{EXEC[run][e]}/{n['statement_source']}" if not n['statement_source'].startswith('CODE') else f"{rundir(run)}/iter_{2*e-1:04d}/CODE_EXECUTION.json"}
    rec["verdict_derivation"] = {"type": n["derivation"][0], **n["derivation"][1]}
    rec["changes_vs_previous"] = n["changes"]
    rec["linked_panel_fixes"] = n["linked"]
    return rec


# ---------------------------------------------------------------- validation reference values (0-based layer indices)
def validation_refs():
    c = "c_edge_recovery/trrust"
    return {
     "layer_index_convention": "0-based in independent_validation.json and in all agent runs; SUMMARY.md prints 1-based (L1..L12). Layer index 2 here = 'L3' in SUMMARY.md.",
     "claim1": {"best_sym_auroc_idx2": vnum(f"{c}/per_layer/sym/2/auroc"), "ci95": vnum(f"{c}/per_layer/sym/2/auroc_ci95"),
                "auprc_idx2": vnum(f"{c}/per_layer/sym/2/auprc"), "base_rate": vnum(f"{c}/candidates/base_rate"),
                "min_sym_auroc_over_layers_idx7": vnum(f"{c}/per_layer/sym/7/auroc"), "its_ci95": vnum(f"{c}/per_layer/sym/7/auroc_ci95"),
                "verdict": "supported: sym AUROC CI excludes 0.5 in all 12 layers (TF bootstrap); but degree-preserving null mean at idx2 = 0.645"},
     "claim2": {"abs_spearman_auroc": vnum(f"{c}/baselines/abs_spearman/auroc"), "delta_sym_idx2": vnum(f"{c}/delta_vs_coexpression/sym/2"),
                "variant_layers_above_coexpression_bh": vnum("verdicts/attention recovers TRRUST edges above co-expression/variant_layers_above_coexpression_bh"),
                "verdict": vnum("verdicts/attention recovers TRRUST edges above co-expression/verdict")},
     "claim3": {"post_hoc_degree_matched_variant_layers_bh": vnum("verdicts/attention recovers TRRUST edges above co-expression/post_hoc_degree_matched_variant_layers_bh"),
                "post_hoc_text": vnum("verdicts/attention recovers TRRUST edges above co-expression/post_hoc"),
                "degree_matched_delta_observed_sym_idx2": vnum(f"{c}/permutation_null/degree_matched_delta_vs_coexpression_post_hoc/per_variant/sym/2/delta_auroc_observed"),
                "degree_matched_null_delta_mean_sym_idx2": vnum(f"{c}/permutation_null/degree_matched_delta_vs_coexpression_post_hoc/per_variant/sym/2/null_delta_mean"),
                "degree_matched_p_bh36_sym_idx2": vnum(f"{c}/permutation_null/degree_matched_delta_vs_coexpression_post_hoc/per_variant/sym/2/p_bh_36"),
                "degree_null_mean_auroc_sym_idx2": vnum("verdicts/reference-degree bias inflates recovery/best_sym_null_mean"),
                "fraction_of_excess_auroc_reproduced_by_degree_null_per_layer": vnum("verdicts/reference-degree bias inflates recovery/sym_fraction_of_excess_auroc_reproduced_by_null_per_layer"),
                "loo_target_indegree_auroc": vnum(f"{c}/baselines/target_indegree_loo/auroc"),
                "verdict": "not supported: degree-matched advantage over co-expression passes BH in 0/36 layer x variant tests"},
     "claim4": {"post_hoc_combined_beyond_coexpr_rank_degree_bh": vnum("verdicts/attention recovers TRRUST edges above co-expression/post_hoc_combined_beyond_coexpr_rank_degree_bh"),
                "trrust_best_sym_idx0_auroc": vnum(f"{c}/marginal_vs_pair_specific_post_hoc/interaction_residualised/sym/0/auroc"),
                "trrust_best_sym_idx0_degree_null_mean": vnum(f"{c}/marginal_vs_pair_specific_post_hoc/interaction_residualised/sym/0/null_mean"),
                "trrust_best_sym_idx0_p_bh_36": vnum(f"{c}/marginal_vs_pair_specific_post_hoc/interaction_residualised/sym/0/p_bh_36"),
                "trrust_n_pass_bh_of_36": num(sum(1 for v in ("sym", "qTF", "qTarget") for x in jget(V, f"{c}/marginal_vs_pair_specific_post_hoc/interaction_residualised/{v}") if x["p_bh_36"] < 0.05), VAL_REL, f"count({c}/marginal_vs_pair_specific_post_hoc/interaction_residualised/*/p_bh_36 < 0.05)"),
                "dorothea_n_pass_bh_of_36": num(sum(1 for v in ("sym", "qTF", "qTarget") for x in jget(V, "c_edge_recovery/dorothea_abc/marginal_vs_pair_specific_post_hoc/interaction_residualised/" + v) if x["p_bh_36"] < 0.05), VAL_REL, "count(c_edge_recovery/dorothea_abc/marginal_vs_pair_specific_post_hoc/interaction_residualised/*/p_bh_36 < 0.05)"),
                "dorothea_sym_excess_over_null_range": num([round(min(x["auroc"] - x["null_mean"] for x in jget(V, "c_edge_recovery/dorothea_abc/marginal_vs_pair_specific_post_hoc/interaction_residualised/sym")), 4), round(max(x["auroc"] - x["null_mean"] for x in jget(V, "c_edge_recovery/dorothea_abc/marginal_vs_pair_specific_post_hoc/interaction_residualised/sym")), 4)], VAL_REL, "min/max over layers of auroc - null_mean, c_edge_recovery/dorothea_abc/marginal_vs_pair_specific_post_hoc/interaction_residualised/sym"),
                "verdict": "TRRUST: not supported (0/36 BH); DoRothEA A-C: small pair-specific component passes BH in most layer x variant tests"},
     "claim5": {"layer_spearman_range": vnum("verdicts/attention is approximately symmetric/layer_spearman_range"),
                "layer_median_rel_asym_range": vnum("verdicts/attention is approximately symmetric/layer_median_rel_asym_range"),
                "per_head_spearman_median": vnum("verdicts/attention is approximately symmetric/per_head_spearman_median"),
                "verdict": vnum("verdicts/attention is approximately symmetric/verdict")},
     "claim6": {"trrust_layers_wilcoxon_bh_lt_0.05": [l["layer"] for l in jget(V, "b_directionality/trrust/per_layer") if l["wilcoxon_p_bh"] < 0.05],
                "trrust_layers_edge_vs_nonedge_mwu_bh_lt_0.05": [l["layer"] for l in jget(V, "b_directionality/trrust/per_layer") if l["edges_vs_nonedge_mwu_p_bh"] < 0.05],
                "dorothea_layers_negative_median_bh_lt_0.05": [l["layer"] for l in jget(V, "b_directionality/dorothea_abc/per_layer") if l["wilcoxon_p_bh"] < 0.05 and l["median_log2_ratio_qTF_over_qTarget"] < 0],
                "trrust_median_log2_ratio_idx11": vnum("b_directionality/trrust/per_layer/11/median_log2_ratio_qTF_over_qTarget"),
                "trrust_nonedge_median_idx11": vnum("b_directionality/trrust/per_layer/11/nonedge_control_median_log2_ratio"),
                "source": VAL_REL + " b_directionality/*/per_layer",
                "verdict": "no pre-specified verdict; descriptive: TF-query>target-query on TRRUST edges in 6/12 layers but edges differ from non-edges in only 4/12, and DoRothEA edges show the opposite sign in 7 layers -> no consistent TF->target direction"},
     "claim7": {"verdict": "not tested (observational kit, no intervention); caveats list only correlational limits", "source": VAL_REL + " caveats"},
    }


# ---------------------------------------------------------------- comparison table
def comparison():
    S = lambda r, e: EXEC[r][e]
    rows = []
    def row(claim, val, cells):
        rows.append({"claim": claim, "validation": val, "runs": cells})
    row("1 attention recovers TRRUST edges above chance",
        "supported (sym AUROC 0.673 [0.631, 0.708] at layer idx 2; all 12 layer CIs above 0.5; AUPRC 0.0037 vs base rate 0.0014)",
        {"sol_A": {"final_iteration": "E5", "concluded": "E5 reports only increments over a nuisance model. E1 results.json lists raw AUROCs (max 0.672, layer 2 log-geometric; TF-query/target-query AUROCs within 0.001 of the validation) without a conclusion", "mark": "not addressed",
                   "numbers": [rj("sol_A", 1, "layer_tests/8/raw_auroc"), rj("sol_A", 1, "layer_tests/6/raw_auroc")]},
         "sol_B": {"final_iteration": "E5 failed; last successful E4", "concluded": "E4 withheld all effects (balance gate failed). E1 reported a descriptive AUROC of 0.664 for TRRUST layer_mean_L02_forward without a conclusion.", "mark": "not addressed",
                   "numbers": [rj("sol_B", 1, "descriptive_universe_metrics/trrust/layer_mean_L02_forward/auroc")]},
         "gpt55": {"final_iteration": "E5 failed; last successful E4", "concluded": "no explicit conclusion; attention_tests.tsv lists raw AUROC/AUPRC for all 468 scores (layer means identical to the validation's 126-TF universe within 5e-8; max 0.663 layer_mean_L2_tf_query); report emphasises the selected head's AUPRC lift over prevalence (0.00065)", "mark": "not addressed (numbers agree)",
                   "numbers": [tsv("gpt55", 4, "attention_tests.tsv", "variant", "layer_mean_L2_tf_query", "auroc"), rj("gpt55", 4, "best_positive_auprc_lift_over_prevalence")]}})
    row("2 attention beats co-expression baselines on raw recall",
        "supported (sym idx2 minus |Spearman| AUROC 0.128 [0.081, 0.174], BH p 0.0045; 14 layer x variant tests pass BH)",
        {"sol_A": {"final_iteration": "E5", "concluded": "No attention-vs-co-expression comparison in E5. E4 compared the full nuisance+attention model with a co-expression-only model (held-out AUROC 0.491), which is not a raw comparison.", "mark": "not addressed",
                   "numbers": [rj("sol_A", 4, "baseline_families/coexpression_only/auroc")]},
         "sol_B": {"final_iteration": "E4", "concluded": "No comparison of attention with co-expression baselines in any iteration (co-expression entered only as matching covariates).", "mark": "not addressed", "numbers": []},
         "gpt55": {"final_iteration": "E4", "concluded": "not stated; both numbers reported (|Pearson| AUROC 0.538, max layer-mean attention AUROC 0.663)", "mark": "not addressed (numbers agree)",
                   "numbers": [rj("gpt55", 4, "baseline_coexpr_abs_pearson_auroc"), tsv("gpt55", 4, "attention_tests.tsv", "variant", "layer_mean_L2_tf_query", "auroc")]}})
    row("3 advantage over co-expression survives degree (hub) structure",
        "not supported (degree-matched advantage 0/36 BH; degree-preserving null reproduces 64%-102% of attention's excess AUROC; LOO target in-degree AUROC 0.829)",
        {"sol_A": {"final_iteration": "E5", "concluded": "indirectly: incremental gain not distinguishable from a degree-preserving rewiring null (empirical p 0.4545 for dAUROC and dAUPRC, 10 rewires; minimum attainable p 0.0909)", "mark": "agree (indirect, underpowered)",
                   "numbers": [so("sol_A", 5, "degree_empirical_p_auroc"), rj("sol_A", 5, "degree_preserving_null/replicates_completed")]},
         "sol_B": {"final_iteration": "E4", "concluded": "E4 did not run its graph null ('not_run_balance_gate_failed'). E1 had claimed association beyond matched confounders while target in-degree remained imbalanced (cAUC-0.5 0.217); E2's unconstrained degree-preserving null gave p 0.00995 for raw mean log attention", "mark": "not addressed",
                   "numbers": [rj("sol_B", 1, "matching_diagnostics/trrust.target_in_degree/effect"), rj("sol_B", 2, "degree_preserving_graph_null/trrust/rewired_p_value")]},
         "gpt55": {"final_iteration": "E4", "concluded": "'TRRUST label topology/degree dominates the verified predictive signal' (LOO-degree AUPRC 0.0553 and split-aware degree AUPRC 0.0330 both exceed best positive attention AUPRC 0.0032)", "mark": "agree (direction; operationalised as attention alone vs degree alone)",
                   "numbers": [so("gpt55", 4, "baseline_loo_degree_only_trrust_topology_null_auprc"), so("gpt55", 4, "baseline_split_aware_degree_only_heldout_tf_cv_auprc"), so("gpt55", 4, "best_positive_auprc")]}})
    row("4 pair-specific signal after co-expression, expression, rank-distance adjustment and degree-preserving null",
        "TRRUST: not supported (0/36 BH; best sym idx0 AUROC 0.612 vs null 0.575, BH p 0.0719). DoRothEA A-C: small component (24/36 BH; excess -0.004 to 0.039 AUROC)",
        {"sol_A": {"final_iteration": "E5", "concluded": "incremental_information_beyond_measured_nuisance_detected = False (hard-coded literal); mean dAUROC 0.012 [-0.0018, 0.0258], dAUPRC 0.0046 [0.0017, 0.0076]; alignment-null p 0.25/0.1875 (15 reps), degree-null p 0.4545 (10 reps). DoRothEA metric retired in E5 (E4: dAUROC +0.029, dAUPRC +0.014, no CI).", "mark": "agree for TRRUST (verdict not computed from data); DoRothEA not addressed",
                   "numbers": [so("sol_A", 5, "incremental_information_beyond_measured_nuisance_detected"), so("sol_A", 5, "mean_delta_auprc"), so("sol_A", 5, "delta_auprc_repeated_split_mean_ci95"), rj("sol_A", 4, "dorothea_nonoverlap_sensitivity/delta_auroc")]},
         "sol_B": {"final_iteration": "E4", "concluded": "verdict 'not_identifiable_from_supplied_aggregate_attention' (hard-coded); TRRUST increment 0.0 [0.0, 0.0] on 4 edges; DoRothEA -0.0041 [-0.042, 0.022] on 50 edges. E1 (not final) had concluded 'replicated_association_beyond_measured_confounders' (15 scores).", "mark": "not addressed (declared non-identifiable); E1 disagreed for TRRUST",
                   "numbers": [so("sol_B", 4, "verdict"), so("sol_B", 4, "trrust_post_balance_trim_edges"), so("sol_B", 4, "dorothea_exclusive_attention_increment_mean"), so("sol_B", 1, "replicated_test_count")]},
         "gpt55": {"final_iteration": "E4", "concluded": "primary_positive_success = 0: best positive TF-balanced SMD 0.189 [0.088, 0.291] (post-selection), BH q 0.0585, matched p 0.109, max-stat p 0.866; nested CV dAUPRC -0.0004 [-0.0023, 0.0013]. DoRothEA transfer of the selected head AUROC 0.519 (not a pair-specific adjusted test).", "mark": "agree for TRRUST; DoRothEA not comparable",
                   "numbers": [so("gpt55", 4, "primary_positive_success_after_bh_matched_maxstat_and_degree_threshold"), so("gpt55", 4, "best_positive_tf_cluster_permutation_q_bh_all_attention_tests"), so("gpt55", 4, "nested_selection_incremental_cv_attention_minus_confound_auprc_mean_bootstrap")]}})
    row("5 attention (approximately) symmetric / direction readable from attention",
        "attention is not symmetric (layer Spearman(A_ij, A_ji) 0.16-0.32; per-head median 0.10)",
        {r: {"final_iteration": f, "concluded": "not measured; no statement that attention is symmetric (forward, reverse and symmetrised scores treated as separate variants)", "mark": "not addressed (no symmetry claim)", "numbers": []}
         for r, f in (("sol_A", "E5"), ("sol_B", "E4"), ("gpt55", "E4"))})
    row("6 TF->target directionality",
        "no consistent direction: TF-query > target-query on TRRUST edges (Wilcoxon BH<0.05) in layers 2,3,8,9,10,11 but edges differ from non-edges in layers 2,3,4,11 only; DoRothEA edges negative in layers 1-7",
        {"sol_A": {"final_iteration": "E5", "concluded": "tf_to_target_directionality_supported = False (hard-coded literal; not tested); text: 'provides no evidence for TF-to-target direction'", "mark": "agree in conclusion (untested)", "numbers": [rj("sol_A", 5, "verdict/tf_to_target_directionality_supported")]},
         "sol_B": {"final_iteration": "E4", "concluded": "'No orientation is privileged as biological information flow' (E3/E4). E1 had BH-supported directional (log forward - log reverse) matched effects: TRRUST layer_mean_L01 0.066, DoRothEA layer_mean_L03 0.048; the validation's DoRothEA layer-3 edges have a lower forward/reverse ratio than non-edges (different control set)", "mark": "not addressed (final); E1 not directly comparable",
                   "numbers": [rj("sol_B", 1, "attention_tests/trrust.layer_mean_L01_directional/effect"), rj("sol_B", 1, "attention_tests/dorothea_exclusive.layer_mean_L03_directional/effect")]},
         "gpt55": {"final_iteration": "E4", "concluded": "tf_query called 'the directional TF-to-target readout', others 'orientation controls'; reversed-edge control for the selected head AUROC 0.490 (p 0.626); no directionality conclusion", "mark": "not addressed", "numbers": [rj("gpt55", 4, "negative_control_reversed_edges_best_positive_auroc")]}})
    row("7 causal interpretation",
        "not tested; observational summaries only",
        {"sol_A": {"final_iteration": "E5", "concluded": "causal_claim_supported = False (hard-coded)", "mark": "agree", "numbers": [so("sol_A", 5, "causal_claim_supported")]},
         "sol_B": {"final_iteration": "E4", "concluded": "'These findings are correlational. No ablation, activation patching, model rerun or biological intervention was available, so causation is not claimed.'", "mark": "agree", "numbers": []},
         "gpt55": {"final_iteration": "E4", "concluded": "'Any positive or negative result here is correlational ... does not show that a head or layer causally mediates regulatory prediction.'", "mark": "agree", "numbers": []}})
    return rows


# ---------------------------------------------------------------- cross-check of computations
def crosscheck():
    out = {}
    t = pd.read_csv(f"{BASE}/{EXEC['gpt55'][4]}/work/attention_tests.tsv", sep="\t")
    lm = t[t.family == "layer_mean"]
    mp = {"tf_query": "qTF", "target_query": "qTarget", "sym_mean": "sym"}
    d = []
    for k, v in mp.items():
        a = lm[lm.direction == k].sort_values("layer").auroc.values
        b = np.array([x["auroc"] for x in jget(V, f"c_edge_recovery/trrust_rows_tfs_with_edge/per_layer/{v}")])
        d += list(np.abs(a - b))
    out["gpt55_E4_layer_mean_auroc_vs_validation_trrust_rows_tfs_with_edge_max_abs_diff"] = num(float(max(d)), f"{EXEC['gpt55'][4]}/work/attention_tests.tsv vs {VAL_REL} c_edge_recovery/trrust_rows_tfs_with_edge/per_layer", "36 layer x orientation AUROCs")
    r = json.load(open(f"{BASE}/{EXEC['sol_A'][1]}/work/results.json"))
    lt = pd.DataFrame(r["layer_tests"])
    d = []
    for k, v in {"tf_query_target_key": "qTF", "target_query_tf_key": "qTarget"}.items():
        a = lt[lt.variant == k].sort_values("layer_index").raw_auroc.values
        b = np.array([x["auroc"] for x in jget(V, f"c_edge_recovery/trrust/per_layer/{v}")])
        d += list(np.abs(a - b))
    out["sol_A_E1_qTF_qTarget_auroc_vs_validation_trrust_max_abs_diff"] = num(float(max(d)), f"{EXEC['sol_A'][1]}/work/results.json layer_tests vs {VAL_REL} c_edge_recovery/trrust/per_layer", "24 AUROCs; universes differ (co-presence >= 100 vs >= 50)")
    out["gpt55_abs_pearson_auroc"] = so("gpt55", 1, "baseline_coexpr_abs_pearson_auroc")
    out["validation_abs_pearson_auroc_same_universe"] = vnum("c_edge_recovery/trrust_rows_tfs_with_edge/baselines/abs_pearson/auroc")
    out["gpt55_rank_proximity_auroc"] = so("gpt55", 1, "baseline_rank_proximity_auroc")
    out["validation_neg_rank_distance_auroc_same_universe"] = vnum("c_edge_recovery/trrust_rows_tfs_with_edge/baselines/neg_rank_distance/auroc")
    out["interpretation"] = "Where agents and the validation computed the same quantity, the numbers coincide; disagreements are in design and interpretation, not arithmetic."
    return out


def cs(run, p):
    return f"{rundir(run)}/iter_{2*p:04d}/CONSENSUS.json"


def curated():
    A, B, G = EXEC["sol_A"], EXEC["sol_B"], EXEC["gpt55"]
    final = {
     "sol_A": {"iteration_used": "E5 (succeeded)", "verdict_flags": {"incremental_information_beyond_measured_nuisance_detected": so("sol_A", 5, "incremental_information_beyond_measured_nuisance_detected"), "causal_claim_supported": so("sol_A", 5, "causal_claim_supported")},
               "statement": "no supported incremental information from pooled Geneformer attention beyond the measured nuisance model (TRRUST annotation endpoint)",
               "computed_support": [so("sol_A", 5, "mean_delta_auroc"), so("sol_A", 5, "delta_auroc_repeated_split_mean_ci95"), so("sol_A", 5, "mean_delta_auprc"), so("sol_A", 5, "delta_auprc_repeated_split_mean_ci95"), so("sol_A", 5, "alignment_empirical_p_auprc"), so("sol_A", 5, "degree_empirical_p_auprc")],
               "caveat": "verdict literal written before execution; the computed dAUPRC CI excludes 0; nulls have 15 and 10 replicates"},
     "sol_B": {"iteration_used": "E5 crashed; last successful E4", "verdict_flags": {"verdict": so("sol_B", 4, "verdict")},
               "statement": "not identifiable from the supplied aggregate (pooled) attention",
               "computed_support": [so("sol_B", 4, "trrust_post_balance_trim_edges"), so("sol_B", 4, "trrust_attention_increment_mean"), so("sol_B", 4, "dorothea_exclusive_attention_increment_mean")],
               "caveat": "verdict code assigned unconditionally from E2 onward; E1 had concluded 'replicated_association_beyond_measured_confounders'"},
     "gpt55": {"iteration_used": "E5 timed out; last successful E4", "verdict_flags": {"primary_positive_success_after_bh_matched_maxstat_and_degree_threshold": so("gpt55", 4, "primary_positive_success_after_bh_matched_maxstat_and_degree_threshold")},
               "statement": "no positive attention association passed BH, matched-null, max-statistic and degree checks; TRRUST label topology/degree dominates the verified predictive signal",
               "computed_support": [so("gpt55", 4, "best_positive_tf_cluster_permutation_q_bh_all_attention_tests"), so("gpt55", 4, "negative_control_maxstat_family_permutation_best_positive_p"), so("gpt55", 4, "nested_selection_incremental_cv_attention_minus_confound_auprc_mean_bootstrap")],
               "caveat": "verdict computed from results; E5 partial outputs (no BH-significant test, min q 0.285) are consistent"},
     "sol_planonly": {"iteration_used": "E1-E5", "statement": "no analysis; refused in every iteration citing missing execution tools and a read-only workspace", "computed_support": [], "caveat": "control"},
    }
    bugs = [
     {"run": "sol_A", "caught_in": "P1 critique[1] (top#2), 3 lenses", "issue": "Nuisance model used TRRUST out/in-degree computed from the outcome labels including the evaluated edge (circular).", "evidence": [so("sol_A", 1, "baseline_oof_auroc")], "verified": f"{A[1]}/code.py lines 106-107 read genes.tsv trrust_*_degree_in_G", "fixed_in": "E3 (leave-source-out target degree, code lines 233-254) and E4 (training/development-label degrees)", "status": "fixed"},
     {"run": "sol_A", "caught_in": "P1 critique[9]", "issue": "Code comment claims squared terms allow curvature; none were in the design matrix.", "evidence": [], "verified": f"{A[1]}/code.py lines 114-150", "fixed_in": "E3 (residualisation with linear + squared terms, per P3 critique[12])", "status": "fixed"},
     {"run": "sol_A", "caught_in": "P1 critique[5]", "issue": "Conditional score test used unpenalised information while the nuisance probabilities came from an L2-penalised (C=1) logistic fit.", "evidence": [], "verified": f"{A[1]}/code.py lines 165-170, 241-249", "fixed_in": "E3 replaced the inference procedure", "status": "replaced"},
     {"run": "sol_A", "caught_in": "P1 critique[17]", "issue": "StandardScaler fitted on the full candidate universe before grouped CV.", "evidence": [], "verified": f"{A[1]}/code.py lines 157-158", "fixed_in": "not checked", "status": "unknown"},
     {"run": "sol_A", "caught_in": "P1 critique[3]; again P3 top#3", "issue": "Negative control = one shuffled realisation with a parametric p-value; in E3 the single within-TF permutation's dAUROC exceeded the attention dAUROC.", "evidence": [rj("sol_A", 3, "predictive_negative_control/delta_auroc_vs_nuisance"), so("sol_A", 3, "heldout_delta_auroc")], "verified": "results.json values", "fixed_in": "E4 (200 complete-pipeline permutations)", "status": "fixed after two panels"},
     {"run": "sol_A", "caught_in": "P2 top#1, 3 lenses", "issue": "E2 crashed: 310228 x 468 float32 memmap exceeds the 64 MiB per-file limit.", "evidence": [num("OSError: [Errno 27] File too large", f"{A[2]}/stderr.txt", "last line")], "verified": f"{A[2]}/code.py lines 328-334", "fixed_in": "E3", "status": "fixed"},
     {"run": "sol_A", "caught_in": "P3 top#2", "issue": "Selection forced an attention variant although all 36 worsened selection log-loss, and the selection and final learners differed.", "evidence": [rj("sol_A", 3, "predictive_evaluation/selection_balanced_logloss_improvement")], "verified": "results.json", "fixed_in": "E4 (no_attention candidate, one protocol)", "status": "fixed"},
     {"run": "sol_A", "caught_in": "P3 critique[6]", "issue": "Outcome-conditioned TF split (all zero-positive TFs in training).", "evidence": [rj("sol_A", 4, "partitions/evaluation/n_zero_positive_tfs")], "verified": "E4 partitions include zero-positive TFs", "fixed_in": "E4", "status": "fixed"},
     {"run": "sol_A", "caught_in": "P3 critique[8]", "issue": "Predictive target degree computed from the complete TRRUST network, including evaluation TFs (transductive leakage).", "evidence": [], "verified": "PATCH_NOTES E4", "fixed_in": "E4", "status": "fixed"},
     {"run": "sol_A", "caught_in": "P4 critique[5]", "issue": "Manifest MD5 verification checked 0 entries (parser lost file names) while PATCH_NOTES claimed all inputs verified.", "evidence": [rj("sol_A", 4, "input_integrity/manifest_md5_entries_checked")], "verified": "results.json", "fixed_in": "E5", "status": "fixed", "fix_evidence": [so("sol_A", 5, "manifest_md5_entries_matching")]},
     {"run": "sol_A", "caught_in": "P4 critique[4]", "issue": "Null-distribution quantiles labelled as 95% confidence intervals.", "evidence": [], "verified": "results.json E4 key names *_ci95", "fixed_in": "E5 (reference_interval95)", "status": "fixed"},
     {"run": "sol_A", "caught_in": "P5 top#2", "issue": "Observed statistic = mean over 15 nested splits; each null replicate = one nested split, so empirical p-values compare different sampling variances.", "evidence": [so("sol_A", 5, "alignment_empirical_p_auroc"), so("sol_A", 5, "degree_empirical_p_auroc")], "verified": f"{A[5]}/code.py lines 918-935 (one nested_pipeline call per replicate)", "fixed_in": "none (last panel)", "status": "not fixed"},
     {"run": "sol_A", "caught_in": "P5 critique[14]", "issue": "10 degree-null replicates: minimum attainable p = 1/11 = 0.0909, so the degree null could not reach 0.05.", "evidence": [rj("sol_A", 5, "degree_preserving_null/replicates_completed")], "verified": "results.json", "fixed_in": "none", "status": "not fixed"},
     {"run": "sol_B", "caught_in": "P1 critiques[5],[6]", "issue": "Positive verdict despite residual imbalance: matched target in-degree still separates positives from controls more than any attention score does.", "evidence": [rj("sol_B", 1, "matching_diagnostics/trrust.target_in_degree/effect"), rj("sol_B", 1, "benchmark_summary/trrust/best_effect")], "verified": "results.json", "fixed_in": "E2 (exact degree matching, balance gate, verdict withdrawn)", "status": "fixed (verdict withdrawn; design then collapsed to 9 edges)"},
     {"run": "sol_B", "caught_in": "P1 critique[4]", "issue": "Ordinary BH over 1248 tests used to claim cross-benchmark replication.", "evidence": [so("sol_B", 1, "multiple_testing_family_size")], "verified": "code verdict uses replicated_test_ids from per-test BH", "fixed_in": "E2 (partial conjunction)", "status": "fixed"},
     {"run": "sol_B", "caught_in": "P3 top#2", "issue": "Controls selected with layer-0-attention-derived covariates and the same attention then tested (outcome-informed matching).", "evidence": [], "verified": "E3 results attention_centrality_adjustment.target_incoming_attention_included = true", "fixed_in": "E4", "status": "fixed", "fix_evidence": [rj("sol_B", 4, "matching_design/attention_derived_covariates_used")]},
     {"run": "sol_B", "caught_in": "P3 top#3", "issue": "Graph-swap code compares proposed signatures with original_signatures.get(edge), which is None for already-rewired edges, so rewired slots can never swap again (0/424 accepted).", "evidence": [rj("sol_B", 3, "constrained_degree_preserving_null/trrust/accepted_swaps")], "verified": f"{B[3]}/code.py lines 1071-1098", "fixed_in": "E4 code (slot-invariant signatures) but the null was not run (balance gate failed)", "status": "fixed in code, not exercised", "fix_evidence": [rj("sol_B", 4, "corrected_degree_preserving_null/trrust/slot_signature_is_invariant")]},
     {"run": "sol_B", "caught_in": "P3 critique[12]", "issue": "analysis_script_saved=false while RUN_SUMMARY instructed users to rerun the saved script.", "evidence": [so("sol_B", 3, "analysis_script_saved")], "verified": "stdout", "fixed_in": "E4 (script saved with SHA-256); the E5 rewrite of this step used __file__ and crashed", "status": "fixed, then regressed"},
     {"run": "sol_B", "caught_in": "P4 critique[8]", "issue": "METHOD.md says each bootstrap re-matches, but the code reuses initial_matchings.", "evidence": [], "verified": f"{B[4]}/code.py lines 1372-1394, 1484", "fixed_in": "none (E5 crashed)", "status": "not fixed"},
     {"run": "sol_B", "caught_in": "P4 top#2", "issue": "Held-out prediction on 4 TRRUST edges / 4 TFs gave a degenerate [0, 0] interval presented with the same structure as a result.", "evidence": [so("sol_B", 4, "trrust_post_balance_trim_edges"), so("sol_B", 4, "trrust_prediction_bootstrap_draws")], "verified": "stdout", "fixed_in": "E5 redesign (crashed)", "status": "not fixed"},
     {"run": "sol_B", "caught_in": "P5 top#1", "issue": "E5 NameError: __file__ undefined in the orchestrator.", "evidence": [num("NameError: name '__file__' is not defined. Did you mean: '__name__'?", f"{B[5]}/stderr.txt", "last line")], "verified": f"{B[5]}/code.py line 1012", "fixed_in": "none (last panel)", "status": "not fixed"},
     {"run": "gpt55", "caught_in": "P1 top#3", "issue": "Verdict logic took the smallest-p variant regardless of sign (a negative effect) while 5 positive-SMD variants had BH q < 0.05.", "evidence": [so("gpt55", 1, "best_attention_residual_smd"), tsv_count("gpt55", 1, True)], "verified": f"{G[1]}/code.py lines 359-363 and attention_tests.tsv", "fixed_in": "E3 (best positive and best negative reported separately)", "status": "fixed"},
     {"run": "gpt55", "caught_in": "P1 top#1", "issue": "Pair-level Welch t-tests over 166446 dependent pairs (p down to 1.4e-17).", "evidence": [so("gpt55", 1, "best_attention_residual_p")], "verified": "stdout", "fixed_in": "E3 (TF-cluster permutation p-values)", "status": "fixed"},
     {"run": "gpt55", "caught_in": "P1 top#2", "issue": "Circular TRRUST degree covariates (degree-only AUROC 0.954).", "evidence": [so("gpt55", 1, "baseline_degree_only_auroc")], "verified": "stdout", "fixed_in": "E3 (LOO degree), E4 (relabelled topology null + split-aware degree), E5 (fold-local topology; partial output only)", "status": "fixed progressively"},
     {"run": "gpt55", "caught_in": "P1 critique[9]", "issue": "Combined confound baseline evaluated in-sample.", "evidence": [so("gpt55", 1, "baseline_combined_confound_auroc")], "verified": "stdout", "fixed_in": "E2/E3 (held-out-TF CV)", "status": "fixed", "fix_evidence": [rj("gpt55", 3, "baseline_combined_confound_logistic_heldout_tf_cv_auroc")]},
     {"run": "gpt55", "caught_in": "P2 top#1", "issue": "E2 timed out at 900 s.", "evidence": [], "verified": "CODE_EXECUTION.json", "fixed_in": "E3", "status": "fixed"},
     {"run": "gpt55", "caught_in": "P3 top#2", "issue": "Family max-statistic null with 30 permutations.", "evidence": [rj("gpt55", 3, "n_family_maxstat_permutations_requested")], "verified": "results.json", "fixed_in": "E4 (499)", "status": "fixed", "fix_evidence": [rj("gpt55", 4, "n_family_maxstat_permutations_requested")]},
     {"run": "gpt55", "caught_in": "P4 top#3", "issue": "Held-out-TF CV used globally computed LOO degree features (test-fold labels leak).", "evidence": [rj("gpt55", 4, "baseline_combined_confound_logistic_heldout_tf_cv_auroc")], "verified": f"{G[4]}/code.py line 632 (baseline_x = full covariate matrix)", "fixed_in": "E5 (fold-local topology computed before the timeout)", "status": "partially fixed", "fix_evidence": [tsv("gpt55", 5, "baseline_metrics.tsv", "score", "combined_confound_logistic_heldout_tf_cv_leakage_free_fold_local_topology", "auroc")]},
     {"run": "gpt55", "caught_in": "P5 top#2", "issue": "E5 timed out at 900.9 s during the continuous NN matched null.", "evidence": [], "verified": "CODE_EXECUTION.json; checkpoint_status.json", "fixed_in": "none (last panel)", "status": "not fixed"},
    ]
    panel_errors = [
     {"run": "gpt55", "panel": "P2", "critique": "critiques[21] (medium, statistics)", "claim": "The partial attention_tests.tsv 'visibly ends with an invalid q value of 0 in the last shown row'.", "why_incorrect": "The report showed a truncated row; the actual q of that row (head_L8_H9_sym_mean) is 0.606 and no q in the file is 0.",
      "evidence": [tsv("gpt55", 2, "attention_tests.tsv", "variant", "head_L8_H9_sym_mean", "tf_cluster_permutation_q_bh_all_attention_tests"), tsv_min("gpt55", 2, "attention_tests.tsv", "tf_cluster_permutation_q_bh_all_attention_tests")], "source": cs("gpt55", 2)},
     {"run": "sol_B", "panel": "P5", "critique": "critiques[20] (medium, reproducibility)", "claim": "Script name 'analysis_iteration9.py' is wrong 'despite this being Iteration 10'.", "why_incorrect": "The executor step was loop iteration 9 (its prompt says 'Iteration 9'); iteration 10 is the panel itself.", "evidence": [], "source": cs("sol_B", 5)},
     {"run": "sol_B", "panel": "P4", "critique": "critiques[12] (medium, regulatory_relationship)", "claim": "Fix asks to 'Verify the TRRUST record's regulatory mode' for HIF1A-COX4I1.", "why_incorrect": "HIF1A-COX4I1 is a DoRothEA (confidence A) edge and is absent from the kit's trrust_edges.tsv (database misattribution; the biological point about the COX4 isoform switch is not contested here).", "evidence": [], "source": cs("sol_B", 4)},
    ]
    panel_inconsistencies = [
     {"run": "sol_A", "sequence": "P2 critiques[5],[10] attacked the E2 degree-preserving edge-swap chain -> E3 removed the degree-preserving null -> P3 critique[7] (high) attacked its absence -> E4 re-added it.", "sources": [cs("sol_A", 2), cs("sol_A", 3)]},
     {"run": "sol_A", "sequence": "P4 top#2 required stating an explicit negative verdict -> E5 hard-coded 'no supported incremental information' -> P5 critique[18] asked to frame the result as inconclusive because the nulls are underpowered and the dAUPRC interval excludes 0.", "sources": [cs("sol_A", 4), cs("sol_A", 5)]},
     {"run": "sol_B", "sequence": "P1 top#3, P2 top#1 and P3 critique[4] asked for identical-cell-set comparisons -> E2-E4 added exact co-presence-signature gates -> P5 top#3 called the exact gate 'unnecessarily stringent'.", "sources": [cs("sol_B", 1), cs("sol_B", 2), cs("sol_B", 3), cs("sol_B", 5)]},
    ]
    not_caught = [
     {"run": "sol_A", "issue": "Verdict flags and statement are literals in the code (E4 and E5); in E5 incremental_information_beyond_measured_nuisance_detected=False was fixed before execution while the computed dAUPRC CI excluded 0.", "evidence": [so("sol_A", 5, "delta_auprc_repeated_split_mean_ci95"), code_ref("sol_A", 5, "1438-1442, 1833-1835", "literal verdict")], "panel_note": "P5 critique[18] noted the tension between the positive AUPRC interval and the negative verdict but not that the verdict was hard-coded."},
     {"run": "sol_B", "issue": "verdict_code assigned unconditionally from E2 onward (E2 comment: 'even favorable score statistics cannot answer the research question affirmatively').", "evidence": [code_ref("sol_B", 2, "1429-1432", "unconditional verdict"), code_ref("sol_B", 4, "1681", "unconditional verdict")], "panel_note": "No panel critique identifies the unconditional assignment; searching all CONSENSUS.json files for 'hard-coded' finds only audit flags and the script name."},
     {"run": "gpt55", "issue": "E3/E4 success criterion requires the standalone attention AUPRC to exceed the LOO-degree AUPRC by 0.001, which tests 'better than degree alone' rather than 'information beyond degree' (the nested incremental CV is the relevant test).", "evidence": [code_ref("gpt55", 4, "956-964", "positive_success conjunction")], "panel_note": "Panels discussed the practical gap to the degree baseline but not the logic of this criterion."},
     {"run": "sol_A", "issue": "E5 TF-query-only family gave dAUROC CI entirely below 0 and dAUPRC CI entirely above 0 (mixed-sign result).", "evidence": [rj("sol_A", 5, "primary_orientation_only/delta_auroc_repeated_split_mean_ci95"), rj("sol_A", 5, "primary_orientation_only/delta_auprc_repeated_split_mean_ci95")], "panel_note": "P5 mentions the TF-query family only in a multiplicity critique."},
     {"run": "all", "issue": "No run measured attention symmetry (A_ij vs A_ji), and no panel requested it; the validation finds layer Spearman(A_ij, A_ji) 0.16-0.32.", "evidence": [vnum("verdicts/attention is approximately symmetric/layer_spearman_range")], "panel_note": "sol_A P3 critique[4] asked to test whether the forward orientation preferentially recovers TF->target edges; E4/E5 answered with a literal tf_to_target_directionality_supported=False instead of a test."},
    ]
    symmetry = {
     "method": "Regex search for 'symmetr', 'asymmetr', 'A_ij|A_{ij}|A[i,j]', 'bi-?directional' and '(attention|matrix|A) (is|are|was) (approximately )?symmetric' over every executor_output.md (20 executor steps) and every code-generated *.md and *.json under exec_outputs/<run>/*/work/.",
     "hits": {"sol_A": {"files_searched": 44, "symmetr_lines": 1066, "asymmetr_lines": 2, "A_ij": 0, "bidirectional": 0, "symmetry_assertions": 0},
              "sol_B": {"files_searched": 29, "symmetr_lines": 678, "asymmetr_lines": 0, "A_ij": 0, "bidirectional": 0, "symmetry_assertions": 0},
              "gpt55": {"files_searched": 26, "symmetr_lines": 5, "asymmetr_lines": 0, "A_ij": 0, "bidirectional": 0, "symmetry_assertions": 0},
              "sol_planonly": {"files_searched": 5, "symmetr_lines": 0, "asymmetr_lines": 0, "A_ij": 0, "bidirectional": 0, "symmetry_assertions": 0}},
     "finding": "No run asserted that attention is symmetric. Every 'symmetr' hit is a score-variant name (symmetric_log_geometric, sym_mean, *_symmetric), a definition of the symmetrised score (mean of forward and reverse, or of their logs), or a statement that forward and reverse readings are treated 'symmetrically' as exploratory variants (sol_B E3, E4). The two 'asymmetr' hits are a sol_A E3 section title about label-dependent degree asymmetry. All runs treated A as directed (query -> key) and tested both readings; none quantified the asymmetry itself.",
     "examples": [num("Attention was read as directed TF-query attention `A[layer, head, TF, target]`, target-query attention `A[layer, head, target, TF]`, and a symmetric mean.", f"{rundir('gpt55')}/iter_0001/executor_output.md", "line 371"),
                  num("orientations_treated_symmetrically: forward, reverse, symmetric, directional", f"{B[3]}/work/results.json", "configuration.orientations_treated_symmetrically"),
                  num("mean of log forward and log reverse attention, treated as nondirectional association", f"{A[4]}/work/results.json", "orientation.symmetric_interpretation")],
    }
    plan = {
     "executor_output_tokens": [num(json.load(open(f"{BASE}/{rundir('sol_planonly')}/run_meta.json"))["iterations"][i]["output_tokens"], f"{rundir('sol_planonly')}/run_meta.json", f"iterations[{i}].output_tokens") for i in (0, 2, 4, 6, 8)],
     "numbers_reported": num(0, "../../experiments/case_study/agent_runs/results/trace_summary.json", "runs.sol_planonly.combined_result.total"),
     "plan_provided": False,
     "behaviour_per_iteration": {f"E{k}": num(PLAN_NOTE[2*k-1], f"{rundir('sol_planonly')}/iter_{2*k-1:04d}/executor_output.md", "full text") for k in range(1, 6)},
     "panel_response": "Grade F in all five panels; every lens wrote that refusing to fabricate was appropriate but that the artefact answers nothing. P1 critique[3] asked for 'a complete preregistered plan explicitly labeled as non-evidential'; the executor never provided one. P2 critique[8], P3 critique[10] and P4 critique[12] noted that read-only workspace access does not by itself prevent returning results in the response. P5 (6 critiques, 0 multi-lens groups; adversarial lens raised 0) noted the kit only has pooled attention.",
     "panel_sources": [cs("sol_planonly", p) for p in range(1, 6)],
     "note": "The plan-only task text equals the executed text minus its 'COMPUTING ENVIRONMENT AND REPORTING RULE' section, so it still asked for 'a quantitative analysis'; with no tools and no execution the executor treated the task as infeasible.",
    }
    failure = {
     "execution_failures": [
      {"run": "sol_A", "step": "E2", "type": "file-size limit (RLIMIT_FSIZE 64 MiB)", "exec_wall_s": 5.1, "source": f"{rundir('sol_A')}/iter_0003/CODE_EXECUTION.json"},
      {"run": "sol_B", "step": "E5", "type": "NameError (__file__)", "exec_wall_s": 72.2, "source": f"{rundir('sol_B')}/iter_0009/CODE_EXECUTION.json"},
      {"run": "gpt55", "step": "E2", "type": "wall-clock timeout 900 s", "exec_wall_s": 900.1, "source": f"{rundir('gpt55')}/iter_0003/CODE_EXECUTION.json"},
      {"run": "gpt55", "step": "E5", "type": "wall-clock timeout 900 s", "exec_wall_s": 900.9, "source": f"{rundir('gpt55')}/iter_0009/CODE_EXECUTION.json"}],
     "near_limit": {"run": "sol_A", "step": "E4", "exec_wall_s": 814.3, "limit_s": 900, "source": f"{rundir('sol_A')}/iter_0007/CODE_EXECUTION.json"},
     "final_iteration_failed": {"sol_A": False, "sol_B": True, "gpt55": True},
     "scope_creep": "Panels of the three executed runs returned 15-24 merged critiques each (18 of all 20 panels graded F); executors tried to address most of them in a single script, which grew from 1025 to 2233 lines (sol_A), 1096 to 2211 (sol_B) and 489 to 1359 (gpt55). Two of the four failures (sol_A E2, gpt55 E5) followed iterations that added large permutation/bootstrap or storage machinery in response to null-model critiques; sol_B E5 failed in code added for a reproducibility request.",
     "design_drift": "sol_B moved from a 5:1 matched design on all 424 TRRUST edges (E1) to exact/caliper matching that retained 9 (E2) and 4 (E4) TRRUST edges, and from a computed positive verdict (E1) to an unconditional non-identifiability verdict (E2-E5).",
     "infeasible_requests": "Critiques requiring data absent from the kit (per-cell or cell-type-specific attention, immune ChIP/perturbation labels, pretraining manifest, raw expression variance, protein activity) appear in every panel (keyword heuristic counts per panel in runs.*.panel_steps[].critiques_requesting_data_absent_from_kit_heuristic).",
     "executor_prose": "All 15 executed-run executor responses were a single fenced Python block with no prose; every reported number therefore lives in code-generated reports (trace_summary: 0 agent-text numbers; 460/149/3 generated-report numbers for sol_A/sol_B/gpt55, all traceable).",
     "trace_summary_source": "../../experiments/case_study/agent_runs/results/trace_summary.json",
    }
    return {"final_conclusions": final, "panel_caught_bugs": bugs, "panel_incorrect_critiques": panel_errors,
            "panel_inconsistencies_across_iterations": panel_inconsistencies, "issues_not_caught_by_panels": not_caught,
            "symmetry_search": symmetry, "plan_only_control": plan, "failure_modes": failure}


def main():
    syn = {
        "generated_by": "experiments/case_study/agent_runs/results/synthesis_scripts/build_synthesis.py (then build_md.py for SYNTHESIS.md), from artefacts under experiments_data/case_study_agent_runs",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "path_base": BASE,
        "path_note": "All 'source' paths are relative to path_base; validation numbers cite " + VAL_REL + " (same base).",
        "layer_index_convention": "All layer indices are the 0-based kit array index used by the agents and by independent_validation.json; SUMMARY.md uses 1-based labels.",
        "question": "Does Geneformer V2-104M attention encode TF->target regulation beyond co-expression, expression-rank proximity and hub (degree) structure?",
        "runs": {},
        "validation_reference": validation_refs(),
        "comparison_final_vs_validation": comparison(),
        "computation_crosscheck": crosscheck(),
    }
    for run in RUNS:
        R = {"run_level": run_level(run), "executor_iterations": [], "panel_steps": []}
        for e in range(1, 6):
            R["executor_iterations"].append(build_iter(run, e))
        for p in range(1, 6):
            R["panel_steps"].append(panel(run, p))
        syn["runs"][run] = R
    syn.update(curated())
    syn["code_lines_per_iteration"] = {r: [it["execution"]["code_lines"]["value"] for it in syn["runs"][r]["executor_iterations"]] for r in RUNS[:3]}
    syn["exec_wall_s_per_iteration"] = {r: [it["execution"]["exec_wall_s"]["value"] for it in syn["runs"][r]["executor_iterations"]] for r in RUNS[:3]}
    json.dump(syn, open(OUT, "w"), indent=1, default=str)
    return syn


if __name__ == "__main__":
    s = main()
    print("written", OUT)
