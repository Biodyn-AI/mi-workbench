"""Verification pass: apply the corrections found by the independent verifier to case_synthesis.json in place.

Run once after build_synthesis.py. Re-running build_synthesis.py overwrites these corrections, so this
script must be re-applied after any regeneration. Every new number below was re-read from the cited file.
"""
import json
import datetime

OUT = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/automation/mi-workbench/experiments/case_study/agent_runs/results/case_synthesis.json"
VAL = "../../experiments/case_study/results/independent_validation.json"
S = json.load(open(OUT))
if "verification" in S:
    raise SystemExit("verification corrections already applied")


def run_cmp(claim_idx, run):
    return S["comparison_final_vs_validation"][claim_idx]["runs"][run]


# ---- computation cross-check: sol_A max diff is 0.00103, not <= 0.001
cc = S["computation_crosscheck"]
cc["sol_A_E1_qTF_qTarget_auroc_vs_validation_trrust_max_abs_diff"]["note"] = (
    "max abs diff 0.00103, i.e. within 0.0011 (the earlier text said 'within 0.001')")
cc["interpretation"] = (
    "Where agents and the validation computed the same quantity on the same universe (gpt55, 126-TF universe) the "
    "numbers coincide to <= 5e-8; sol_A (co-presence >= 100 vs >= 50) differs by <= 0.0011. Disagreements are in "
    "design and interpretation, not arithmetic.")

# ---- validation reference: extra numbers used by the corrected comparison
vr = S["validation_reference"]
vr["claim2"]["delta_auprc_sym_idx2_vs_abs_spearman"] = {
    "value": 0.001057047, "ci95": [-0.001540866, 0.004673606], "source": VAL,
    "key": "c_edge_recovery/trrust/delta_vs_coexpression/sym/2/delta_auprc_vs_abs_spearman(_ci95)",
    "note": "the AUROC advantage over co-expression is significant; the AUPRC advantage is not"}
vr["claim3"]["raw_attention_vs_degree_null"] = {
    "value": {"best_sym_idx2_auroc": 0.6730247, "null_mean": 0.6454671, "p_empirical_max_over_36": 0.003996004,
              "n_layer_variant_beating_degree_null_bh": 12},
    "source": VAL,
    "key": "verdicts/attention recovers TRRUST edges above co-expression/{best_layer_any_variant,variant_layers_beating_degree_null_bh}; verdicts/reference-degree bias inflates recovery/best_sym_null_mean",
    "note": "raw attention AUROC does exceed the degree-preserving null; what fails (post hoc) is the advantage over co-expression beyond the degree-null difference"}
vr["claim3"]["verdict"] = (
    "not supported (post-hoc test, added after inspection): degree-matched advantage over co-expression passes BH in 0/36 "
    "layer x variant tests (sym idx2: observed 0.128 vs null 0.132, BH p 0.995). Raw attention itself beats the "
    "degree-preserving null (sym idx2 0.673 vs 0.645, max-stat p 0.004; 12/36 BH), but the null reproduces 64%-102% "
    "of the excess AUROC.")
vr["prespecified_statements_note"] = (
    "The seven claims are this synthesis's decomposition. The validation pre-specifies four statements: (1) symmetric: "
    "not supported (= claim 5); (2) recovers TRRUST above co-expression: supported (claims 1-2; claims 3-4 are its "
    "post-hoc qualifications); (3) explained by expression rank proximity: not supported (median |Spearman| 0.242; "
    "no row of its own; every run used rank distance only as a covariate); (4) reference-degree bias inflates "
    "recovery: supported (no row of its own; gpt55 E4's degree-dominance verdict maps onto it).")
vr["statement4_degree_bias"] = {
    "value": {"verdict": "supported", "degree_product_loo_auroc": 0.8044909, "degree_product_loo_auprc": 0.0556,
              "target_indegree_loo_auroc": 0.8287729,
              "same_126tf_universe_degree_product_loo_auprc": 0.0562, "same_126tf_universe_target_indegree_loo_auroc": 0.8295},
    "source": VAL,
    "key": "verdicts/reference-degree bias inflates recovery; c_edge_recovery/{trrust,trrust_rows_tfs_with_edge}/baselines/*_loo"}

# ---- comparison table
c0 = run_cmp(0, "sol_A")
c0["concluded"] = c0["concluded"].replace("max 0.672, layer 2 log-geometric", "layer-mean max 0.672, layer 2 log-geometric; head max 0.674").replace("within 0.001 of the validation", "within 0.0011 of the validation")

S["comparison_final_vs_validation"][1]["claim"] = "2 attention beats co-expression baselines on raw edge ranking (AUROC)"
S["comparison_final_vs_validation"][1]["validation"] = (
    "supported for AUROC (sym idx2 minus |Spearman| AUROC 0.128 [0.081, 0.174], BH p 0.0045; 14 layer x variant tests "
    "pass BH); the AUPRC difference (0.0011 [-0.0015, 0.0047]) is not significant")

S["comparison_final_vs_validation"][2]["validation"] = vr["claim3"]["verdict"] + " LOO target in-degree AUROC 0.829."
c = run_cmp(2, "sol_A")
c["mark"] = "not directly comparable"
c["concluded"] = (
    "sol_A never compared attention with co-expression. Its degree-preserving rewiring null (E5: empirical p 0.4545 "
    "for dAUROC and dAUPRC, 10 rewires, minimum attainable p 0.0909; E4: p 0.70/0.56 with 100 rewires) tests the "
    "increment of attention over a nuisance model that already contains co-expression, rank and leakage-safe degree, "
    "i.e. the claim-4 estimand, not claim 3's advantage-over-co-expression versus a degree-matched null. Direction "
    "consistent with the validation.")
c = run_cmp(2, "gpt55")
c["mark"] = "not directly comparable (agrees with the validation's pre-specified 'reference-degree bias inflates recovery')"
c["concluded"] = (
    "'TRRUST label topology/degree dominates the verified predictive signal' (LOO-degree AUPRC 0.0553 and split-aware "
    "degree AUPRC 0.0330 both exceed best positive attention AUPRC 0.0032). This compares attention alone with degree "
    "alone; it does not test whether attention's advantage over co-expression survives a degree-matched null. It "
    "matches the validation's statement 4 (same 126-TF universe: LOO degree-product AUPRC 0.0562, LOO target "
    "in-degree AUROC 0.830, both above attention).")

c = run_cmp(3, "sol_A")
c["mark"] = ("agree for TRRUST, on E4's evidence (200/100-replicate nulls, p 0.56-0.73); E5's verdict is hard-coded "
             "and E5's nulls could not reach p < 0.05; DoRothEA not addressed")
c["concluded"] = c["concluded"] + (
    " Neither E5 null could reject at 0.05 (minimum attainable p 1/16 = 0.0625 with 15 alignment permutations, 1/11 = "
    "0.0909 with 10 rewirings). The dAUPRC t-interval is over 15 overlapping splits, and the null replicates "
    "themselves average dAUPRC 0.0022 (alignment) and 0.0030 (degree); observed-minus-null reference intervals include "
    "0. Estimand differs from the validation's (incremental held-out prediction vs residualised pair-specific AUROC "
    "against a degree-preserving null).")
c = run_cmp(3, "gpt55")
c["mark"] = "agree for TRRUST (different estimand); DoRothEA not comparable"
c["concluded"] = c["concluded"] + (
    " Estimand: TF-balanced residual SMD after linear residualisation on 22 covariates including LOO degree, plus "
    "nested-CV increment (126-TF universe), vs the validation's row/column-removed, covariate-residualised AUROC "
    "against a degree-preserving null. gpt55 also reports 25 negative BH-significant variants (attention lower on "
    "edges after residualisation); the validation has no corresponding test, and this is not examined here.")

c = run_cmp(5, "sol_A")
c["mark"] = "not tested (hard-coded flag; consistent with the validation's lack of a consistent direction)"
c = run_cmp(5, "sol_B")
c["concluded"] = c["concluded"].replace(
    "'No orientation is privileged as biological information flow' (E3/E4).",
    "'No orientation is privileged as biological information flow' (E3 MECH.md); E4 METHOD.md: reverse, symmetric "
    "and directional readings 'remain computational alternatives, not biological information-flow directions'.")

# ---- final conclusions
S["final_conclusions"]["sol_A"]["caveat"] = (
    "verdict literal written before execution; the computed dAUPRC t-interval (15 overlapping splits) excludes 0, but "
    "null replicates average dAUPRC 0.0022-0.0030 and observed-minus-null intervals include 0; nulls have 15 and 10 "
    "replicates, so neither could reach p < 0.05 (min 0.0625 / 0.0909): inconclusive rather than negative")

# ---- per-iteration fields
S["runs"]["sol_A"]["executor_iterations"][3]["verdict_derivation"]["type"] = (
    "hard-coded literals, but only scope flags (causal, immune context, directionality, beyond all named and unmeasured "
    "confounders) and a scope statement; E4 set no flag on the primary incremental-information question. Causal and "
    "immune-context claims are not testable with the kit; directionality was testable but not tested.")
S["runs"]["gpt55"]["run_level"]["prompt_files_logged"]["note"] = (
    "gpt55 logged 28: the three P1 lens prompts were issued twice with identical sha256 because the first issue failed "
    "after ~47 s with '[network] stream disconnected before completion' and the adapter retried; CONSENSUS.json "
    "records 1 attempt")

# ---- panel-caught bugs
for b in S["panel_caught_bugs"]:
    if b["run"] == "sol_A" and b["caught_in"] == "P1 critique[17]":
        b["fixed_in"] = ("E3 (predictive pipeline fits StandardScaler on training rows only, code.py line 767; E4 "
                         "fit_stack line 465; E5 fit_stack line 579). E3's descriptive association screen still "
                         "standardises over the full universe (line 319) but has no held-out evaluation.")
        b["status"] = "fixed"
        b["verified"] += "; verifier checked E3-E5 code"
    if b["run"] == "sol_A" and b["caught_in"] == "P5 critique[14]":
        b["issue"] = ("15 alignment and 10 degree-null replicates: minimum attainable p = 1/16 = 0.0625 and 1/11 = "
                      "0.0909, so neither null could reach 0.05 (the critique reported both minima but drew the "
                      "conclusion only for the degree null).")

# ---- incorrect critiques: add context for the iteration-number error
for x in S["panel_incorrect_critiques"]:
    if x["run"] == "sol_B" and x["panel"] == "P5":
        x["why_incorrect"] += (" The lens prompt itself was headed 'REVIEW REQUEST (Iteration 10)', so the reviewer "
                               "conflated its own iteration with the executor's.")

# ---- inconsistencies across iterations: reclassify
inc = S["panel_inconsistencies_across_iterations"]
inc[0]["classification"] = "not a panel contradiction (executor's choice)"
inc[0]["sequence"] = (
    "P2 critiques[5],[10] criticised the implementation of the E2 degree-preserving edge-swap chain (statistic not "
    "recomputed per rewired network; no burn-in or mixing diagnostics) and asked for it to be fixed -> E3 removed the "
    "null instead -> P3 critique[7] (high) asked for one -> E4 re-added it. The panel never asked for removal.")
inc[1]["classification"] = "reversal of framing after the evidence changed (not a pure contradiction)"
inc[1]["sequence"] = (
    "P4 top#2 (on E4: dAUROC -0.0106, dAUPRC 0.0005, 200/100-replicate nulls non-significant) required an explicit "
    "negative verdict -> E5 hard-coded 'no supported incremental information' -> P5 critique[18] (on E5: positive "
    "dAUPRC interval, 15/10-replicate nulls) asked for an 'inconclusive' framing.")
inc[2]["classification"] = "partial reversal"
inc[2]["sequence"] = (
    "P1 top#3 ('identical cell sets, or at minimum within cell-type and sequence-length strata'), P2 top#1 ('identical "
    "positive/control cell sets, or restrict the biological conclusion to an unlocalized aggregate association') and "
    "P3 critique[4] asked for same-cell comparisons -> E2-E4 used exact co-presence-signature requirements and treated "
    "their failure as non-identifiability -> P5 top#3 called the exact gate 'unnecessarily stringent'. P1 and P2 had "
    "offered fallbacks (strata, or an aggregate-only conclusion) that the executor did not take.")

# ---- issues not caught
S["issues_not_caught_by_panels"][0]["issue"] = (
    "In E5 the primary flag incremental_information_beyond_measured_nuisance_detected=False and the statement are "
    "literals fixed before execution, while the computed dAUPRC CI excluded 0. E4's literals were scope flags "
    "(causal, immune context, directionality, unmeasured confounders), not a verdict on the primary question.")

# ---- symmetry search
S["symmetry_search"]["finding"] = S["symmetry_search"]["finding"].replace("(sol_B E3, E4)", "(sol_B E3)")

# ---- failure modes
fm = S["failure_modes"]
fm["scope_creep"] = fm["scope_creep"].replace(
    "Two of the four failures (sol_A E2, gpt55 E5) followed iterations that added large permutation/bootstrap or "
    "storage machinery in response to null-model critiques",
    "Three of the four failures (sol_A E2, gpt55 E2, gpt55 E5) occurred in steps that added large "
    "permutation/bootstrap or storage machinery in response to inference and null-model critiques")
fm["file_size_limit_disclosure"] = (
    "The 64 MiB per-file limit was not in the task text (which states only the 900 s wall and 1800 s CPU limits). It "
    "was stated in the E1 execution report embedded in sol_A's E2 prompt ('max_file_mb: RLIMIT_FSIZE=67108864 bytes "
    "per file'), and E2 still allocated a ~581 MB memmap.")
fm["source_file_size_limit_disclosure"] = [
    "../../experiments/case_study/agent_runs/task_text_executed.md",
    "workspaces/sol_A/runs/x4_sol_A/x4_prompts/iter_0003__executor__20261001T115217_673024.prompt.md (line 166)"]

# ---- sol_A E4 change note: P2 asked for a fix, not removal
e4 = S["runs"]["sol_A"]["executor_iterations"][3]
e4["changes_vs_previous"] = e4["changes_vs_previous"].replace(
    "(it had been removed in E3 after P2 criticised it)",
    "(it had been removed in E3 after P2 criticised its implementation; P2 had asked for a fix, not removal)")

# ---- verification record
S["verification"] = {
    "verified_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "by": "independent verifier (re-opened every source; recomputed counts with python)",
    "automated": "Of the 918 pre-verification {value, source, key} entries, 654 were re-resolved programmatically "
                 "(stdout, results.json, validation JSON, run_meta, CODE_EXECUTION.json, CONSENSUS.json, tsv "
                 "queries) and 155 more by recomputation (file sizes, fence counts, prose chars, line counts, "
                 "directory listings, stderr last lines, attempts, lens assessments): 0 mismatches. The rest "
                 "(run-level sums, keyword-heuristic counts, count/min over tsv/validation, quotes, and 10 "
                 "paraphrases of the plan-only outputs) were recomputed or read by hand; no numeric error found.",
    "changes": "see SYNTHESIS.md section 'Verification notes'",
    "generator_note": "build_synthesis.py / build_md.py reproduce the pre-verification files; "
                      "synthesis_scripts/verify_patch_json.py re-applies the JSON corrections.",
}
S["generated_at"] = S["generated_at"] + " (verified " + S["verification"]["verified_at"] + ")"

json.dump(S, open(OUT, "w"), indent=1, default=str)
print("patched")
