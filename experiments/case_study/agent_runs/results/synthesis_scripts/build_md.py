import json

RES = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/automation/mi-workbench/experiments/case_study/agent_runs/results"
S = json.load(open(f"{RES}/case_synthesis.json"))
R = S["runs"]
L = []
w = L.append


def v(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def fmt_int(x):
    return f"{int(x):,}"


def hist(h):
    order = ["critical", "high", "medium", "low", "info"]
    return "/".join(str(h.get(k, 0)) for k in order)


w("# Case-study agent runs (X4): synthesis of four executed loops")
w("")
w("Question given to the agents: *Does Geneformer V2-104M attention encode TF->target regulation beyond co-expression, expression-rank proximity and hub (degree) structure?*")
w("")
w("Runs: `sol_A`, `sol_B` (gpt-5.6-sol, medium effort, verified execution), `gpt55` (gpt-5.5, medium, verified execution), `sol_planonly` (gpt-5.6-sol, verified execution disabled). Loop: `reviewer_consensus`, E1 P1 ... E5 P5 (executor step, then a three-lens panel with LLM-adjudicated merging). Each executed step is one Python script run in the macOS `sandbox_exec` backend (900 s wall, 1800 s CPU, 64 MiB per written file).")
w("")
w("Conventions. Layer indices are the 0-based kit array index used by the agents and by `independent_validation.json` (the validation `SUMMARY.md` prints them 1-based, so index 2 here is its 'L3'). Every number below is in `case_synthesis.json` with its source file, given relative to `experiments_data/case_study_agent_runs/`. 'Validation' means the author-side independent analysis in `experiments/case_study/results/`. Loop iteration numbers: E*k* = iteration 2*k*-1, P*k* = iteration 2*k*.")
w("")

# ------------------------------------------------------------------ 1
w("## 1. Summary")
w("")
w("1. **Where the agents and the validation computed the same quantity, the numbers agree.** gpt55's 36 layer-mean AUROCs (126-TF universe) equal the validation's to within 5e-8; sol_A's TF-query and target-query AUROCs (co-presence >= 100 instead of >= 50) are within 0.001; the |Pearson| (0.5376) and rank-proximity (0.4890) baselines match to four or more decimals. The runs differ from the validation in design and interpretation, not in arithmetic.")
w("2. **Final conclusions.** sol_A (E5): no supported incremental information beyond the measured nuisance model. This verdict is a literal written into the code before the run, and the run's own dAUPRC interval, [0.0017, 0.0076], excludes 0. gpt55 (E4, last successful step): no positive association passed BH, matched-null, max-statistic and degree checks, and TRRUST degree/topology dominates the predictive signal; this verdict was computed. sol_B (E4, last successful step): 'not identifiable from the supplied aggregate attention', assigned unconditionally from E2 onward. sol_planonly refused in all five iterations and produced no plan and no numbers.")
w("3. **Against the validation (seven claims; details in section 4).** No final iteration stated the two claims the validation supports, namely that attention recovers TRRUST edges above chance and that it beats co-expression on raw recall. sol_A and gpt55 agree with the validation that, for TRRUST, nothing survives degree structure or the full adjustment (claims 3 and 4). No final iteration measured symmetry or drew a directionality conclusion (sol_B E1 tested a forward-minus-reverse score and gpt55 used a reversed-edge control, but neither was carried into a final claim). No run claimed attention is symmetric; the validation finds it is not. All runs declined causal claims. No final iteration evaluated the small pair-specific DoRothEA signal the validation detects (24/36 tests pass BH). sol_A had a positive DoRothEA estimate in E4 (dAUROC +0.029, no CI) and retired it in E5 after a panel critique.")
w("4. **Execution reliability.** There were 4 failed executions out of 15: one file-size limit, one NameError and two 900 s timeouts. Both sol_B and gpt55 ended on a failed step, so their final panels reviewed code without a results.json (gpt55 P5 used partial intermediate tables).")
w("5. **Panel.** The panel found real defects, most of them fixed in the next step: label leakage through TRRUST-derived degree in all three runs, a sign-agnostic verdict rule (gpt55), a broken graph-swap routine (sol_B), a silent manifest-check failure (sol_A) and invalid nulls. It also made at least three factually wrong critiques, contradicted itself across iterations in sol_A and sol_B, raised requests for data the kit does not contain in every panel, and never noticed that verdicts were hard-coded (sol_A E4-E5, sol_B E2-E5). Grades were F in 18 of 20 panels; the exceptions were gpt55 P3 and P4 (D).")
w("")

# ------------------------------------------------------------------ 2
w("## 2. Run overview")
w("")
w("| run | model / effort | execution | executor tools | tokens total (input / output; cached) | executor / panel tokens | agent calls | wall-clock | failed executions | grades P1-P5 | final conclusion (step) |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
fc = S["final_conclusions"]
for r in ["sol_A", "sol_B", "gpt55", "sol_planonly"]:
    rl = R[r]["run_level"]
    ts = v(rl["executor_tools_setting"])
    tools = f"executor_allow_tools={ts['executor_allow_tools']} ({ts['executor_allow_tools_source']})"
    calls = f"{v(rl['agent_calls_recorded'])}" + (f" ({v(rl['prompt_files_logged'])} prompt files)" if v(rl['prompt_files_logged']) != v(rl['agent_calls_recorded']) else "")
    exe = f"{v(rl['code_execution_backend'])}" if v(rl["code_execution_enabled"]) else "off"
    w(f"| {r} | {v(rl['model'])} / {v(rl['reasoning_effort'])} | {exe} | {tools} | {fmt_int(v(rl['total_tokens']))} ({fmt_int(v(rl['total_input_tokens']))} / {fmt_int(v(rl['total_output_tokens']))}; {fmt_int(v(rl['total_cached_input_tokens']))}) | {fmt_int(v(rl['executor_tokens_sum']))} / {fmt_int(v(rl['panel_tokens_sum']))} | {calls} | {v(rl['run_wall_clock_min'])} min | {v(rl['failed_executions'])} (timeouts {v(rl['timed_out_executions'])}) | {' '.join(v(rl['grade_history']))} | {fc[r]['statement']} [{fc[r]['iteration_used']}] |")
w("")
w("Source: `workspaces/<run>/runs/x4_<run>/run_meta.json`. Codex reports no cost. Input tokens include cached input. 'Agent calls' = 5 executor calls + 5 x (3 lenses + 1 adjudicator). gpt55 logged 28 prompt files because its three P1 lens prompts were issued twice with identical content (CONSENSUS.json records one attempt).")
w("")

# ------------------------------------------------------------------ 3
w("## 3. Iteration-by-iteration record")
w("")
w("All 15 executor responses in the executed runs were a single fenced Python block with no prose (0 numbers in agent text, `results/trace_summary.json`). The reviewers therefore read the code together with the execution report (stdout plus up to 48 KB of written files, `results.json` first). Every reported number comes from code-generated files.")
w("")


def mech_row(it):
    ex = it["execution"]
    status = "OK" if v(ex["success"]) else ("TIMEOUT (SIGKILL)" if v(ex["timed_out"]) else "FAILED: " + v(ex["stderr_last_line"]).split(". ")[0][:60])
    return f"exit {v(ex['exit_code'])}, {status}", v(ex["exec_wall_s"]), v(ex["code_lines"]), v(it["executor_call"]["duration_s"])


EX_TEXT = {
 "sol_A": [
  ("Universe 235 TFs x genes, co-presence >= 100 (310,228 pairs, 423 positives). Estimand: grouped-CV dAUPRC of nuisance+attention over nuisance-only, plus conditional OR per SD. Nuisance-only AUROC 0.9877 (inflated by circular TRRUST degree). Best layer_7_tf_query: dAUPRC 0.0020, OR 0.929 [0.830, 1.040], BH q 0.873; 0/36 layer and 0/432 head discoveries. DoRothEA dAUPRC -0.00008 [-0.00045, 0.00016]. Raw AUROC up to 0.672 (layer 2 log-geometric) in results.json.",
   "`all_success_criteria_met=False` (computed): 'does not detect attention information beyond the measured ... confounders ... a null result'"),
  ("No result: OSError [Errno 27] File too large when creating a 310,228 x 468 float32 memmap (~581 MB > 64 MiB).", "none"),
  ("TF-disjoint train/selection/evaluation; 468-score association screen: 0 BY discoveries (best effect 0.236 [0.073, 0.398], selection-unadjusted, BY q 1). Selected layer_8_target_query: held-out dAUROC 0.0151 [0.0016, 0.0320], dAUPRC -0.0224 [-0.0607, 0.0091] (26 TFs, 84 positives). A single within-TF permutation gave dAUROC 0.0185, dAUPRC -0.0693.",
   "statement computed (needs both CIs > 0): 'does not establish a robust incremental ... advantage'; `causal_claim_supported=False` (literal)"),
  ("No-attention option plus 36 scores; selected layer_7_symmetric: held-out dAUROC -0.0106, dAUPRC 0.0005. 200 alignment permutations: p 0.68 / 0.73; 100 degree-preserving rewirings: p 0.70 / 0.56. DoRothEA non-overlap sensitivity dAUROC +0.029, dAUPRC +0.014 (no CI). Manifest check: 0 entries checked.",
   "`causal_claim_supported`, `immune_context_specific_claim_supported`, `tf_to_target_directionality_supported` all False (literals)"),
  ("15 repeated nested TF splits: mean baseline AUROC 0.817; mean dAUROC 0.0120 [-0.0018, 0.0258]; mean dAUPRC 0.0046 [0.0017, 0.0076]. TF-query only: dAUROC -0.0096 [-0.0160, -0.0032], dAUPRC 0.0031 [0.0002, 0.0060]. Alignment null (15 reps) p 0.25 / 0.1875; degree null (10 reps) p 0.4545 / 0.4545. 6/12 multiverse configurations positive on both metrics. Manifest 10/10.",
   "`incremental_information_beyond_measured_nuisance_detected=False`, `causal_claim_supported=False`, `tf_to_target_directionality_supported=False`, all **literals written before execution**"),
 ],
 "sol_B": [
  ("Matched case-control design: 5 same-TF database-absent controls per edge on 9 covariates incl. log target in-degree (424 TRRUST, 1,949 DoRothEA-exclusive). Effect = conditional AUC - 0.5 over 1,248 scores; BH. Supported tests: 40 TRRUST, 72 DoRothEA, 15 replicated. Best TRRUST 0.098 [0.054, 0.142]; best DoRothEA 0.081 [0.050, 0.113]. Residual imbalance after matching: target in-degree 0.217 / 0.180; max abs SMD 0.380 / 0.357.",
   "`verdict=replicated_association_beyond_measured_confounders` (computed)"),
  ("Exact degree matching + calipers + context profiles kept 9/424 TRRUST and 76/1,949 DoRothEA edges (max abs SMD 0.988 / 0.369). Locked layer-0 forward: TRRUST 0.028 [-0.068, 0.124]; DoRothEA 0.129 [0.093, 0.164]; partial-conjunction p 0.466. Held-out increment -0.012 / +0.001. Unconstrained degree-preserving null on mean log attention p 0.00995 (observed -3.408 vs null mean -3.440).",
   "`verdict=not_established_with_supplied_aggregates` (**unconditional**)"),
  ("Optimal same-TF matching kept all edges, but max TF-macro abs SMD was 1.279 / 1.106. Forward effect TRRUST 0.025 [-0.002, 0.056], DoRothEA 0.035 [0.014, 0.053]. 20x grouped-CV increment -0.0017 / +0.0001. Constrained degree null: 0/424 swaps accepted. 0 joint-family supported.",
   "`verdict=not_identifiable_from_supplied_aggregate_attention` (**unconditional**)"),
  ("Caliper matching with balance gate: TRRUST 4 edges / 4 TFs (max abs SMD 2.557), DoRothEA 50 / 24 (0.449), so both gates failed and effects were withheld. Increment TRRUST 0.0 [0.0, 0.0] (41 draws), DoRothEA -0.0041 [-0.0421, 0.0218].",
   "`verdict=not_identifiable_from_supplied_aggregate_attention` (**unconditional**)"),
  ("No result: NameError `__file__` (line 1012) after 72.2 s; 7 partial TRRUST graph-null draws.", "none"),
 ],
 "gpt55": [
  ("Universe: 126 TFs with >= 1 TRRUST edge (166,446 pairs, 424 positives). Estimand: residual SMD after linear residualisation on 12 covariates incl. circular degree; pair-level Welch t; BH over 468. 39 q < 0.05 (5 positive, 34 negative). Selected by smallest p: head_L8_H10_sym_mean, SMD -0.103 [-0.222, -0.041], q 6.7e-15. Baselines: circular degree AUROC 0.954, in-sample combined 0.974, abs Pearson 0.538, rank proximity 0.489. Max layer-mean AUROC 0.663 (L2 tf_query). Reversed edges 0.540; DoRothEA transfer 0.504.",
   "computed but sign-agnostic: 'does not provide BH-FDR significant evidence ...'"),
  ("No result: killed at 900.1 s. Partial: LOO-degree AUROC 0.853 / AUPRC 0.0553; held-out combined confound AUROC 0.895; min BH q 0.011 (TF-cluster sign-flip).", "none"),
  ("TF-balanced residual SMD with TF-cluster permutation p: 0 positive and 22 negative BH-significant. Best positive head_L9_H4_tf_query SMD 0.186 [0.080, 0.289] (post-selection), q 0.121, matched p 0.196, max-stat p 0.871 (30 permutations). AUPRC 0.0032 vs LOO-degree 0.0553. Incremental CV dAUPRC -0.0002.",
   "`primary_positive_success...=0` (computed): 'hub/database structure is the dominant verified signal'"),
  ("Plus cell-type composition covariates: 0 positive and 25 negative BH-significant. Best positive SMD 0.189 [0.088, 0.291], q 0.0585, matched p 0.109 (100), max-stat p 0.866 (499). AUROC 0.563, AUPRC 0.0032 (prevalence 0.0025). LOO-degree AUROC 0.853 / AUPRC 0.0553; split-aware degree AUPRC 0.0330. Nested-selection dAUPRC -0.0004 [-0.0023, 0.0013]. Reversed edges 0.490; DoRothEA transfer 0.519.",
   "`primary_positive_success...=0` (computed): 'TRRUST label topology/degree dominates the verified predictive signal'"),
  ("No result: killed at 900.9 s in the continuous-NN matched null. Partial: min BH q 0.285; max AUPRC 0.0096; fold-local held-out CV AUROC 0.836 for confounds vs 0.832 with attention added.", "none"),
 ],
}

for r in ["sol_A", "sol_B", "gpt55"]:
    w(f"### 3.{['sol_A','sol_B','gpt55'].index(r)+1} {r}")
    w("")
    w("| step | execution | exec wall (s) | code lines | executor step (s) | what was computed (estimand; key numbers with intervals) | code-generated verdict |")
    w("|---|---|---|---|---|---|---|")
    for k, it in enumerate(R[r]["executor_iterations"]):
        st, wall, loc, dur = mech_row(it)
        comp, verd = EX_TEXT[r][k]
        w(f"| E{k+1} | {st} | {wall} | {loc} | {dur} | {comp} | {verd} |")
    w("")
    w("What changed and why (links to the preceding panel's critiques; `P1#2` = second item of P1's top-3):")
    w("")
    for k, it in enumerate(R[r]["executor_iterations"]):
        if k == 0:
            continue
        w(f"- **E{k+1}** ({', '.join(it['linked_panel_fixes'])}): {it['changes_vs_previous']}")
    w("")
    w("| panel | grade | critical/high/medium/low/info | merged critiques | multi-lens groups | raw per lens (rev/adv/bio) | top-3 required fixes |")
    w("|---|---|---|---|---|---|---|")
    for p in R[r]["panel_steps"]:
        raw = v(p["raw_critique_counts_per_lens"])
        tops = "<br>".join(f"{t['rank']}. [{t['severity']}, {t['category']}, {len(t['raised_by'])} lens{'es' if len(t['raised_by'])>1 else ''}] {t['summary']}" for t in p["top3_required_fixes"])
        w(f"| {p['step']} | {v(p['grade'])} | {hist(v(p['severity_histogram']))} | {v(p['total_merged_critiques'])} | {v(p['n_groups_multi_lens'])} | {raw['reviewer']}/{raw['adversarial_reviewer']}/{raw['bio_plausibility_checker']} | {tops} |")
    w("")

w("### 3.4 sol_planonly (control, execution disabled)")
w("")
w("| step | executor output tokens | behaviour |")
w("|---|---|---|")
for k, it in enumerate(R["sol_planonly"]["executor_iterations"]):
    w(f"| E{k+1} | {v(it['executor_call']['output_tokens'])} | {v(it['behaviour'])} |")
w("")
w("| panel | grade | critical/high/medium/low/info | merged critiques | multi-lens groups | raw per lens (rev/adv/bio) | top-3 required fixes |")
w("|---|---|---|---|---|---|---|")
for p in R["sol_planonly"]["panel_steps"]:
    raw = v(p["raw_critique_counts_per_lens"])
    tops = "<br>".join(f"{t['rank']}. [{t['severity']}, {t['category']}] {t['summary']}" for t in p["top3_required_fixes"])
    w(f"| {p['step']} | {v(p['grade'])} | {hist(v(p['severity_histogram']))} | {v(p['total_merged_critiques'])} | {v(p['n_groups_multi_lens'])} | {raw['reviewer']}/{raw['adversarial_reviewer']}/{raw['bio_plausibility_checker']} | {tops} |")
w("")

# ------------------------------------------------------------------ 4
w("## 4. Final iterations vs the independent validation")
w("")
w("Iteration used: sol_A E5. For sol_B and gpt55, E5 failed, so E4 (the last successful step) is compared. Marks: **agree**, **disagree**, **not addressed**. Validation numbers come from `experiments/case_study/results/independent_validation.json` (layer index 0-based).")
w("")
w("| # | claim | validation verdict and numbers | sol_A (E5) | sol_B (E4) | gpt55 (E4) |")
w("|---|---|---|---|---|---|")
for row in S["comparison_final_vs_validation"]:
    cells = []
    for r in ["sol_A", "sol_B", "gpt55"]:
        c = row["runs"][r]
        cells.append(f"**{c['mark']}**: {c['concluded']}")
    num_, name = row["claim"].split(" ", 1)
    w(f"| {num_} | {name} | {row['validation']} | {cells[0]} | {cells[1]} | {cells[2]} |")
w("")
w("Interpretation. The validation's headline is that raw recovery is real (sym AUROC 0.673 at layer index 2) but mostly reproduced by degree structure: a degree-preserving null reproduces 64%-102% of the excess AUROC. After co-expression, rank and degree are removed, no TRRUST signal survives; a few hundredths of AUROC remain with the larger DoRothEA set. sol_A and gpt55 reach the TRRUST half of that conclusion through incremental-prediction designs. gpt55 reaches it with a verdict computed from its numbers. sol_A reaches it through a verdict fixed in advance, while its own primary dAUPRC interval excludes 0, which, given 15/10-replicate nulls, makes its result inconclusive rather than negative. Neither reports the positive half (attention beats co-expression on raw recall), although gpt55's and sol_A E1's tables contain the numbers. sol_B's E1, the only positive verdict among the 15 executed steps, matched on target in-degree but left a residual degree imbalance larger than the attention effect; the panel caught this and the claim was withdrawn. The panel's push toward identical-cell matching then drove the design to 4 TRRUST edges and an unconditional non-identifiability verdict, which does not answer the question.")
w("")
w("DoRothEA. The validation finds a pair-specific component with DoRothEA A-C (24/36 tests pass BH; sym excess over the degree null -0.004 to 0.039 AUROC). No final iteration tested this: sol_A retired its DoRothEA estimate, gpt55 transferred one selected head (AUROC 0.519), and sol_B withheld effects. sol_A E4 (dAUROC +0.029, dAUPRC +0.014, no interval) and sol_B E2 (layer-0 forward 0.129 [0.093, 0.164] on 76 matched edges, balance gate failed) point the same way as the validation but were not carried forward.")
w("")

# ------------------------------------------------------------------ 5
sy = S["symmetry_search"]
w("## 5. Did any run assert that attention is symmetric?")
w("")
w("No. " + sy["method"])
w("")
w("| run | files searched | lines with 'symmetr' | 'asymmetr' | 'A_ij' | 'bidirectional' | assertions of symmetry |")
w("|---|---|---|---|---|---|---|")
for r, h in sy["hits"].items():
    w(f"| {r} | {h['files_searched']} | {h['symmetr_lines']} | {h['asymmetr_lines']} | {h['A_ij']} | {h['bidirectional']} | {h['symmetry_assertions']} |")
w("")
w(sy["finding"] + " No panel asked for a symmetry measurement. sol_A P3 asked for a forward-vs-reverse test of directionality (critique[4]); sol_A answered with the literal flag `tf_to_target_directionality_supported=False` rather than a test. For reference, the validation finds head-mean attention is not symmetric (layer Spearman(A_ij, A_ji) 0.16-0.32, per-head median 0.10).")
w("")

# ------------------------------------------------------------------ 6
w("## 6. Errors the panel caught in executed code, and whether they were fixed")
w("")
w("Each item was checked against the code or outputs (column 'verified').")
w("")
w("| run | caught in | issue | verified | fixed in | status |")
w("|---|---|---|---|---|---|")
for b in S["panel_caught_bugs"]:
    w(f"| {b['run']} | {b['caught_in']} | {b['issue']} | {b['verified']} | {b['fixed_in']} | {b['status']} |")
w("")
w("Counts: of the 29 checked items, 19 were fixed (including fixed-after-two-panels, fixed-progressively and replaced), 1 was partially fixed, 1 was fixed in code but never exercised, 1 was fixed and then regressed, 6 were not fixed (4 raised by the last panel, 2 because the next execution crashed), and 1 was not checked. Leakage through TRRUST-derived degree occurred independently in all three executed runs at E1 and was flagged by P1 each time.")
w("")

# ------------------------------------------------------------------ 7
w("## 7. Panel critiques that were wrong, inconsistent, or missing")
w("")
w("Factually incorrect critiques:")
w("")
for e in S["panel_incorrect_critiques"]:
    w(f"- **{e['run']} {e['panel']} {e['critique']}**: {e['claim']} {e['why_incorrect']}")
w("")
w("Contradictions across iterations, where the executor followed one panel and the next panel reversed it:")
w("")
for e in S["panel_inconsistencies_across_iterations"]:
    w(f"- **{e['run']}**: {e['sequence']}")
w("")
w("Requests for data the kit does not contain (per-cell or cell-type-specific attention, immune ChIP/perturbation labels, pretraining-corpus manifest, raw expression variance, TF protein activity) appear in every panel. A keyword heuristic counts 3-12 such critiques per executed-run panel, 1-8 of them critical or high (per-panel counts in the JSON). Such demands were usually the top-1 or top-2 item: cell-type stratification was the top-1 fix in sol_A P1/P3, sol_B P2/P4 and gpt55 P3. Executors answered with `unavailable_analyses` flags (sol_A, gpt55) or by declaring the question non-identifiable (sol_B).")
w("")
w("Defects the panel did not catch:")
w("")
for e in S["issues_not_caught_by_panels"]:
    w(f"- **{e['run']}**: {e['issue']} {e['panel_note']}")
w("")

# ------------------------------------------------------------------ 8
pc = S["plan_only_control"]
w("## 8. Plan-only control")
w("")
w(f"The control never reported a number and never produced an analysis plan. Executor output was {', '.join(str(v(x)) for x in pc['executor_output_tokens'])} tokens across E1-E5. In every step it said it lacked a shell/Python tool and a writable workspace and declined to produce results (section 3.4). {pc['panel_response']} {pc['note']} The control therefore shows that the executed runs' numbers depend on verified execution. It does not show what an unexecuted plan would have looked like, because none was written. Tokens: 386,702 in total, 6.0 min wall-clock.")
w("")

# ------------------------------------------------------------------ 9
fm = S["failure_modes"]
w("## 9. Failure modes")
w("")
w("| run | step | failure | exec wall (s) |")
w("|---|---|---|---|")
for f in fm["execution_failures"]:
    w(f"| {f['run']} | {f['step']} | {f['type']} | {f['exec_wall_s']} |")
w("")
w(f"Near miss: sol_A E4 ran for {fm['near_limit']['exec_wall_s']} s of the {fm['near_limit']['limit_s']} s limit.")
w("")
w("Lines of executed code per step (E1-E5):")
w("")
w("| run | E1 | E2 | E3 | E4 | E5 |")
w("|---|---|---|---|---|---|")
for r, locs in S["code_lines_per_iteration"].items():
    w(f"| {r} | " + " | ".join(str(x) for x in locs) + " |")
w("")
w("Execution wall time per step (s):")
w("")
w("| run | E1 | E2 | E3 | E4 | E5 |")
w("|---|---|---|---|---|---|")
for r, ws in S["exec_wall_s_per_iteration"].items():
    w(f"| {r} | " + " | ".join(str(x) for x in ws) + " |")
w("")
w("- **Scope creep.** " + fm["scope_creep"])
w("- **Design drift.** " + fm["design_drift"])
w("- **Hard-coded verdicts.** sol_A E4-E5 and sol_B E2-E5 wrote verdict flags and statements as literals, so the reported conclusion could not respond to the computed numbers. In sol_A E5 it did not: the primary dAUPRC interval excludes 0, yet the flag says no incremental information. The number trace (`trace_summary.json`) classifies only numbers, so it does not flag these boolean literals.")
w("- **Underpowered nulls after time pressure.** sol_A cut its null replicates from 200/100 (E4) to 15/10 (E5). With 10 rewirings the smallest attainable p is 0.0909, so its degree null cannot reject at 0.05.")
w("- **Executor prose.** " + fm["executor_prose"])
w("")

# ------------------------------------------------------------------ 10
w("## 10. Limitations of this synthesis")
w("")
w("- One replicate per configuration except gpt-5.6-sol (2 replicates). sol_A and sol_B diverged completely in design (incremental prediction vs matched case-control) and in final verdict, so between-replicate variance is large and model comparisons are not warranted.")
w("- The comparison maps each run's own estimand onto the validation's claims. For claims 3 and 4, sol_A and gpt55 test incremental prediction rather than the validation's degree-matched and residualised AUROC tests, so 'agree' means the same direction of conclusion, not the same test.")
w("- Bug verification covered the specific code lines cited. 'Fixed' means the next executed code no longer contains the defect, or the outputs show the corrected behaviour; the corrected code was not independently re-run.")
w("- The count of requests for absent data is a keyword heuristic. The top-3 fixes are the first three merged critiques in the order the platform stores them (severity first); the one-line summaries are paraphrases, and the verbatim first sentence of each fix is in the JSON.")
w("")
w("Files: `case_synthesis.json` (all numbers with sources), this file. Generated from the run artefacts; nothing outside `experiments/case_study/agent_runs/results/` was modified.")

txt = "\n".join(L) + "\n"
for a, b in (("|SMD|", "abs SMD"), ("|Pearson|", "abs Pearson"), ("|Spearman|", "abs Spearman")):
    txt = txt.replace(a, b)
open(f"{RES}/SYNTHESIS.md", "w").write(txt)
print("written", len("\n".join(L)))
