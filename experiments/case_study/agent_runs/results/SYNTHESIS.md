# Case-study agent runs (X4): synthesis of four executed loops

Question given to the agents: *Does Geneformer V2-104M attention encode TF->target regulation beyond co-expression, expression-rank proximity and hub (degree) structure?*

Runs: `sol_A`, `sol_B` (gpt-5.6-sol, medium effort, verified execution), `gpt55` (gpt-5.5, medium, verified execution), `sol_planonly` (gpt-5.6-sol, verified execution disabled). Loop: `reviewer_consensus`, E1 P1 ... E5 P5 (executor step, then a three-lens panel with LLM-adjudicated merging). Each executed step is one Python script run in the macOS `sandbox_exec` backend (900 s wall, 1800 s CPU, 64 MiB per written file).

Conventions. Layer indices are the 0-based kit array index used by the agents and by `independent_validation.json` (the validation `SUMMARY.md` prints them 1-based, so index 2 here is its 'L3'). Every number below is in `case_synthesis.json` with its source file, given relative to `experiments_data/case_study_agent_runs/`. 'Validation' means the author-side independent analysis in `experiments/case_study/results/`. Loop iteration numbers: E*k* = iteration 2*k*-1, P*k* = iteration 2*k*.

## 1. Summary

1. **Where the agents and the validation computed the same quantity, the numbers agree.** gpt55's 36 layer-mean AUROCs (126-TF universe) equal the validation's to within 5e-8; sol_A's TF-query and target-query AUROCs (co-presence >= 100 instead of >= 50) are within 0.0011; gpt55's abs Pearson (0.5376) and rank-proximity (0.4890) baselines match the validation's same-universe values to four or more decimals. The runs differ from the validation in design and interpretation, not in arithmetic.
2. **Final conclusions.** sol_A (E5): no supported incremental information beyond the measured nuisance model. This verdict is a literal written into the code before the run. The run's own dAUPRC interval, [0.0017, 0.0076], excludes 0, but it is a t-interval over 15 overlapping splits, the null replicates themselves average dAUPRC 0.0022-0.0030, and neither null (15 and 10 replicates) could reach p < 0.05. The result is therefore inconclusive rather than negative. gpt55 (E4, last successful step): no positive association passed BH, matched-null, max-statistic and degree checks, and TRRUST degree/topology dominates the predictive signal; this verdict was computed. sol_B (E4, last successful step): 'not identifiable from the supplied aggregate attention', assigned unconditionally from E2 onward. sol_planonly refused in all five iterations and produced no plan and no numbers.
3. **Against the validation (seven claims; details in section 4).** No final iteration stated the validation's two positive recovery findings: attention ranks TRRUST edges above chance, and above co-expression by AUROC. For TRRUST, sol_A and gpt55 agree with the validation that no attention signal survives the full adjustment for co-expression, rank and degree (claim 4). sol_A's agreement rests on E4's properly powered nulls, not on E5. gpt55's degree-dominance verdict also agrees with the validation's pre-specified statement that reference-degree bias inflates recovery. Neither run tested claim 3 as the validation defines it (attention's advantage over co-expression against a degree-matched null), so claim 3 is marked not directly comparable for both. No final iteration measured symmetry or drew a directionality conclusion (sol_B E1 tested a forward-minus-reverse score and gpt55 used a reversed-edge control, but neither was carried into a final claim). No run claimed attention is symmetric; the validation finds it is not. All runs declined causal claims. No final iteration evaluated the small pair-specific DoRothEA signal the validation detects (24/36 tests pass BH). sol_A had a positive DoRothEA estimate in E4 (dAUROC +0.029, no CI) and retired it in E5 after a panel critique.
4. **Execution reliability.** There were 4 failed executions out of 15: one file-size limit, one NameError and two 900 s timeouts. Both sol_B and gpt55 ended on a failed step, so their final panels reviewed code without a results.json (gpt55 P5 used partial intermediate tables).
5. **Panel.** The panel found real defects, most of them fixed in the next step: circular TRRUST-derived degree covariates at E1 in all three runs (flagged as leakage in sol_A and gpt55, as residual target in-degree imbalance in sol_B), a verdict rule that tested only the smallest-p variant regardless of sign (gpt55), a broken graph-swap routine (sol_B), a silent manifest-check failure (sol_A) and invalid nulls. It also made at least three factually wrong critiques. It reversed its own framing once in sol_A, after the evidence had changed, and partly reversed itself once in sol_B. It raised requests for data the kit does not contain in every panel. It never noticed that the primary verdict was hard-coded (sol_A E5, sol_B E2-E5). Grades were F in 18 of 20 panels; the exceptions were gpt55 P3 and P4 (D).

## 2. Run overview

| run | model / effort | execution | executor tools | tokens total (input / output; cached) | executor / panel tokens | agent calls | wall-clock | failed executions | grades P1-P5 | final conclusion (step) |
|---|---|---|---|---|---|---|---|---|---|---|
| sol_A | gpt-5.6-sol / medium | sandbox_exec | executor_allow_tools=False (default) | 944,622 (798,489 / 146,133; 147,456) | 277,189 / 667,433 | 25 | 61.2 min | 1 (timeouts 0) | F F F F F | no supported incremental information from pooled Geneformer attention beyond the measured nuisance model (TRRUST annotation endpoint) [E5 (succeeded)] |
| sol_B | gpt-5.6-sol / medium | sandbox_exec | executor_allow_tools=False (default) | 977,474 (838,235 / 139,239; 153,600) | 278,655 / 698,819 | 25 | 43.1 min | 1 (timeouts 0) | F F F F F | not identifiable from the supplied aggregate (pooled) attention [E5 crashed; last successful E4] |
| gpt55 | gpt-5.5 / medium | sandbox_exec | executor_allow_tools=False (default) | 961,748 (864,168 / 97,580; 208,384) | 252,684 / 709,064 | 25 (28 prompt files) | 90.7 min | 2 (timeouts 2) | F F D D F | no positive attention association passed BH, matched-null, max-statistic and degree checks; TRRUST label topology/degree dominates the verified predictive signal [E5 timed out; last successful E4] |
| sol_planonly | gpt-5.6-sol / medium | off | executor_allow_tools=False (config) | 386,702 (367,552 / 19,150; 203,392) | 134,819 / 251,883 | 25 | 6.0 min | 0 (timeouts 0) | F F F F F | no analysis; refused in every iteration citing missing execution tools and a read-only workspace [E1-E5] |

Source: `workspaces/<run>/runs/x4_<run>/run_meta.json`. Codex reports no cost. Input tokens include cached input. 'Agent calls' = 5 executor calls + 5 x (3 lenses + 1 adjudicator). gpt55 logged 28 prompt files because its three P1 lens prompts were issued twice with identical content. The first issue failed after about 47 s with a network stream disconnect and the adapter retried; CONSENSUS.json records one attempt.

## 3. Iteration-by-iteration record

All 15 executor responses in the executed runs were a single fenced Python block with no prose (0 numbers in agent text, `results/trace_summary.json`). The reviewers therefore read the code together with the execution report (stdout plus up to 48 KB of written files, `results.json` first). Every reported number comes from code-generated files.

### 3.1 sol_A

| step | execution | exec wall (s) | code lines | executor step (s) | what was computed (estimand; key numbers with intervals) | code-generated verdict |
|---|---|---|---|---|---|---|
| E1 | exit 0, OK | 317.4 | 1025 | 600.782 | Universe 235 TFs x genes, co-presence >= 100 (310,228 pairs, 423 positives). Estimand: grouped-CV dAUPRC of nuisance+attention over nuisance-only, plus conditional OR per SD. Nuisance-only AUROC 0.9877 (inflated by circular TRRUST degree). Best layer_7_tf_query: dAUPRC 0.0020, OR 0.929 [0.830, 1.040], BH q 0.873; 0/36 layer and 0/432 head discoveries. DoRothEA dAUPRC -0.00008 [-0.00045, 0.00016]. Raw layer-mean AUROC up to 0.672 (layer 2 log-geometric; head maximum 0.674) in results.json. | `all_success_criteria_met=False` (computed): 'does not detect attention information beyond the measured ... confounders ... a null result' |
| E2 | exit 1, FAILED: OSError: [Errno 27] File too large | 5.1 | 2233 | 503.172 | No result: OSError [Errno 27] File too large when creating a 310,228 x 468 float32 memmap (~581 MB > 64 MiB). | none |
| E3 | exit 0, OK | 70.9 | 1714 | 423.646 | TF-disjoint train/selection/evaluation; 468-score association screen: 0 BY discoveries (best effect 0.236 [0.073, 0.398], selection-unadjusted, BY q 1). Selected layer_8_target_query: held-out dAUROC 0.0151 [0.0016, 0.0320], dAUPRC -0.0224 [-0.0607, 0.0091] (26 TFs, 84 positives). A single within-TF permutation gave dAUROC 0.0185, dAUPRC -0.0693. | statement computed (needs both CIs > 0): 'does not establish a robust incremental ... advantage'; `causal_claim_supported=False` (literal) |
| E4 | exit 0, OK | 814.3 | 1694 | 1167.581 | No-attention option plus 36 scores; selected layer_7_symmetric: held-out dAUROC -0.0106, dAUPRC 0.0005. 200 alignment permutations: p 0.68 / 0.73; 100 degree-preserving rewirings: p 0.70 / 0.56. DoRothEA non-overlap sensitivity dAUROC +0.029, dAUPRC +0.014 (no CI). Manifest check: 0 entries checked. | `causal_claim_supported`, `immune_context_specific_claim_supported`, `tf_to_target_directionality_supported` (and `beyond_all_named_and_unmeasured_confounders_supported`) all False (literals). These are scope flags; E4 set no flag on the primary question. |
| E5 | exit 0, OK | 117.0 | 1837 | 474.988 | 15 repeated nested TF splits: mean baseline AUROC 0.817; mean dAUROC 0.0120 [-0.0018, 0.0258]; mean dAUPRC 0.0046 [0.0017, 0.0076]. TF-query only: dAUROC -0.0096 [-0.0160, -0.0032], dAUPRC 0.0031 [0.0002, 0.0060]. Alignment null (15 reps) p 0.25 / 0.1875; degree null (10 reps) p 0.4545 / 0.4545 (minimum attainable p 0.0625 and 0.0909, so neither null could reach 0.05; null-replicate mean dAUPRC 0.0022 and 0.0030). 6/12 multiverse configurations positive on both metrics. Manifest 10/10. | `incremental_information_beyond_measured_nuisance_detected=False`, `causal_claim_supported=False`, `tf_to_target_directionality_supported=False`, all **literals written before execution** |

What changed and why (links to the preceding panel's critiques; `P1#2` = second item of P1's top-3):

- **E2** (P1#2, P1 critique[3] negative_control, P1 critique[6] two-way clustering): Responded to P1: leave-one-edge-out degrees (P1#2 leakage), degree-preserving network null and permutation-based negative controls (P1 critique[3]), crossed TF/target bootstrap (P1 critique[6]); code grew from 1025 to 2233 lines.
- **E3** (P2#1, P2#2, P2#3, P2 critique[5] null_model, P2 critique[10] statistics): P2#1 fixed (storage redesigned, run succeeded). P2#3 addressed by disjoint selection/evaluation partitions. P2 high critiques on the edge-swap chain (statistic mismatch, mixing) answered by removing the degree-preserving null altogether; context claims removed (P2#2).
- **E4** (P3#1, P3#2, P3#3, P3 critique[6] leakage, P3 critique[7] null_model, P3 critique[8] leakage): P3#2 fixed (no-attention option, consistent learner). P3#3 fixed (200 complete-pipeline permutations). P3 high critiques fixed: outcome-conditioned TF split, transductive degree leakage; degree-preserving null re-introduced because P3 flagged its absence (it had been removed in E3 after P2 criticised its implementation; P2 had asked for a fix, not removal). P3#1 (cell-type attention) answered only by an 'unavailable_analyses' entry. PATCH_NOTES claimed input hashes were verified, but 0 manifest entries were checked.
- **E5** (P4#1, P4#2, P4 critique[3] p_hacking, P4 critique[4] statistical_theater, P4 critique[5] reproducibility): P4#2 implemented literally: the verdict was changed to an explicit negative result, but as a hard-coded literal. P4#1 answered by retiring the DoRothEA metric (E4 had shown DoRothEA dAUROC +0.029, dAUPRC +0.014). P4 critiques on the inspected fixed split / multiverse (critique[3]) and the mislabelled 'ci95' null quantiles (critique[4]) were fixed; the manifest bug (critique[5]) was fixed (10/10). The null replicate counts fell from 200/100 (E4) to 15/10.

| panel | grade | critical/high/medium/low/info | merged critiques | multi-lens groups | raw per lens (rev/adv/bio) | top-3 required fixes |
|---|---|---|---|---|---|---|
| P1 | F | 7/3/10/0/0 | 20 | 9 | 14/10/7 | 1. [critical, tissue_specificity, 3 lenses] Repeat the analysis within cell types (or a hierarchical cell type/tissue/donor model) - not possible with the pooled kit attention.<br>2. [critical, stat_bio_alignment, 3 lenses] Recompute TRRUST degree covariates without the evaluated edge (fold-wise) or from an independent network; the 0.9877 baseline AUROC is outcome leakage.<br>3. [critical, leakage, 2 lenses] Audit TRRUST/DoRothEA overlap by PMID/source; edge-exclusion alone does not make DoRothEA an independent validation. |
| P2 | F | 5/6/10/0/0 | 21 | 4 | 12/10/4 | 1. [critical, execution_failure, 3 lenses] Redesign storage to stay below the 64 MiB per-file limit and rerun; no number in the templated reports is supported by a successful run.<br>2. [critical, tissue_specificity, 2 lenses] Use immune-context labels (ChIP/perturbation/motif-accessibility) or report generic-database recovery only as annotation discrimination.<br>3. [critical, statistical_theater, 2 lenses] Repeat the 36-way layer/orientation selection inside every bootstrap replicate, or use disjoint selection and evaluation sets. |
| P3 | F | 4/9/9/1/1 | 24 | 6 | 13/10/7 | 1. [critical, tissue_specificity, 2 lenses] Cell-type-specific attention and matched labels; until then restrict to context-agnostic annotation discrimination.<br>2. [critical, p_hacking, 2 lenses] Add a no-attention option to selection and use one model class for selection and final refit (all 36 variants worsened selection log-loss by >= 0.0893).<br>3. [critical, null_model, 2 lenses] Repeat the within-TF permutation hundreds of times with full refit/selection (the single permutation's dAUROC 0.0185 exceeded attention's 0.0151). |
| P4 | F | 5/7/5/0/0 | 17 | 6 | 11/9/5 | 1. [critical, leakage, 3 lenses] Deduplicate DoRothEA evidence against TRRUST and run the same bootstrap/nulls, or label DoRothEA an unvalidated exploratory sensitivity analysis.<br>2. [critical, stat_bio_alignment, 3 lenses] State the verdict explicitly as a negative result (dAUROC -0.0106, dAUPRC 0.0005, both nulls non-significant) instead of 'unresolved'.<br>3. [critical, biological_implausibility, 2 lenses] Per-cell/cell-type attention with active-TF filtering and context validation; otherwise database-annotation interpretation only. |
| P5 | F | 3/4/12/1/0 | 20 | 3 | 12/7/4 | 1. [critical, negative_control, 2 lenses] Permute attention only among nuisance-matched pairs (the within-TF alignment permutation is not a conditional null).<br>2. [critical, statistics, 2 lenses] Run the same 15-split averaging for every null replicate (observed mean of 15 splits vs single-split nulls: p = 0.25/0.1875/0.4545 uncalibrated).<br>3. [critical, statistical_theater, 2 lenses] Replace t-intervals over 15 overlapping splits by a TF-cluster bootstrap of the full nested pipeline, or drop confidence-interval language. |

### 3.2 sol_B

| step | execution | exec wall (s) | code lines | executor step (s) | what was computed (estimand; key numbers with intervals) | code-generated verdict |
|---|---|---|---|---|---|---|
| E1 | exit 0, OK | 12.5 | 1096 | 274.071 | Matched case-control design: 5 same-TF database-absent controls per edge on 9 covariates incl. log target in-degree (424 TRRUST, 1,949 DoRothEA-exclusive). Effect = conditional AUC - 0.5 for 624 scores x 2 benchmarks (1,248 tests); BH over all 1,248. Supported tests: 40 TRRUST, 72 DoRothEA, 15 replicated. Best TRRUST 0.098 [0.054, 0.142]; best DoRothEA 0.081 [0.050, 0.113]. Residual imbalance after matching: target in-degree 0.217 / 0.180; max abs SMD 0.380 / 0.357. | `verdict=replicated_association_beyond_measured_confounders` (computed) |
| E2 | exit 0, OK | 119.5 | 1850 | 468.649 | Exact degree matching + calipers + context profiles kept 9/424 TRRUST and 76/1,949 DoRothEA edges (max abs SMD 0.988 / 0.369). Locked layer-0 forward: TRRUST 0.028 [-0.068, 0.124]; DoRothEA 0.129 [0.093, 0.164]; partial-conjunction p 0.466. Held-out increment -0.012 / +0.001. Unconstrained degree-preserving null on mean log attention p 0.00995 (observed -3.408 vs null mean -3.440). | `verdict=not_established_with_supplied_aggregates` (**unconditional**) |
| E3 | exit 0, OK | 85.0 | 2211 | 460.161 | Optimal same-TF matching kept all edges, but max TF-macro abs SMD was 1.279 / 1.106. Forward effect TRRUST 0.025 [-0.002, 0.056], DoRothEA 0.035 [0.014, 0.053]. 20x grouped-CV increment -0.0017 / +0.0001. Constrained degree null: 0/424 swaps accepted. 0 joint-family supported. | `verdict=not_identifiable_from_supplied_aggregate_attention` (**unconditional**) |
| E4 | exit 0, OK | 104.8 | 1841 | 444.54 | Caliper matching with balance gate: TRRUST 4 edges / 4 TFs (max abs SMD 2.557), DoRothEA 50 / 24 (0.449), so both gates failed and effects were withheld. Increment TRRUST 0.0 [0.0, 0.0] (41 draws), DoRothEA -0.0041 [-0.0421, 0.0218]. | `verdict=not_identifiable_from_supplied_aggregate_attention` (**unconditional**) |
| E5 | exit 1, FAILED: NameError: name '__file__' is not defined | 72.2 | 1369 | 371.372 | No result: NameError `__file__` (line 1012) after 72.2 s; 7 partial TRRUST graph-null draws. | none |

What changed and why (links to the preceding panel's critiques; `P1#2` = second item of P1's top-3):

- **E2** (P1#1, P1#2, P1#3, P1 critique[4] p_hacking, P1 critique[5] overclaiming, P1 critique[6] stat_bio_alignment): P1 critiques[5]/[6] (verdict contradicted by residual target in-degree imbalance) answered by exact degree matching + calipers + balance gate; P1#3 (identical cells) and P1#2 (cell-type strata) answered by context-profile matching and an unconditional 'not established' verdict; P1#1 by a provenance audit; P1 critique[4] by partial conjunction. Strict matching retained 9/424 TRRUST edges.
- **E3** (P2#1, P2#2, P2#3): P2#3 (9-edge estimand) answered by optimal matching that keeps all edges at the cost of balance (max abs SMD rose to 1.279); P2#2 answered by nuisance-constrained swaps (0/424 accepted, partly a code defect); P2#1 (identical cells) answered by declaring non-identifiability. Attention-derived covariates were added to the matching (later flagged as leakage).
- **E4** (P3#1, P3#2, P3#3, P3 critique[12] reproducibility): P3#2 fixed (attention-derived covariates removed). P3#3 fixed in code (invariant slot signatures) but the null was not executed. P3#1 implemented as a hard balance gate, which failed (TRRUST collapsed to 4 edges). P3 critique[12] fixed (script saved). Held-out prediction ran on 4 TRRUST edges, giving a degenerate [0, 0] interval.
- **E5** (P4#2, P4#3, P4 critique[18] reproducibility): Responded to P4#2/P4#3 by replacing the matched prediction with a graph-null contrast, and to reproducibility requests by self-copying the script; the new self-copy used __file__, which is undefined when the orchestrator runs the code, so the run crashed (E4 had worked around this).

| panel | grade | critical/high/medium/low/info | merged critiques | multi-lens groups | raw per lens (rev/adv/bio) | top-3 required fixes |
|---|---|---|---|---|---|---|
| P1 | F | 7/12/4/0/0 | 23 | 9 | 15/9/8 | 1. [critical, validation_independence, 2 lenses] Provenance audit and validation on a source-independent immune perturbation/ChIP benchmark (DoRothEA-exclusive is not independent).<br>2. [critical, biological_implausibility, 2 lenses] Stratify by cell type and donor, require within-stratum expression, and report the specific pairs driving each supported score.<br>3. [critical, null_model, 2 lenses] Compare positives and controls over identical cell sets / cell-type strata, and add degree-preserving rewiring constrained on nuisances. |
| P2 | F | 4/11/9/0/0 | 24 | 7 | 15/9/8 | 1. [critical, context_mismatch, 3 lenses] Estimate effects within cell-type/tissue strata on identical cell sets, or restrict to an unlocalized aggregate association (TRRUST max abs SMD 0.988).<br>2. [critical, null_model, 2 lenses] Constrain degree-preserving swaps within co-presence/expression/rank/context strata (the p = 0.00995 null tests unmatched mean log attention).<br>3. [critical, statistical_theater, 2 lenses] The matched estimand covers 9/424 TRRUST and 76/1949 DoRothEA edges: audit retained vs excluded edges, caliper sensitivity, overlap weighting. |
| P3 | F | 5/8/9/1/0 | 23 | 6 | 12/9/10 | 1. [critical, stat_bio_alignment, 3 lenses] Do not interpret aggregate effects unless balance is achieved (max TF-macro abs SMD 1.279 TRRUST, 1.106 DoRothEA).<br>2. [critical, leakage, 2 lenses] Remove layer-0-attention-derived covariates from control selection (outcome-informed matching).<br>3. [critical, null_model, 2 lenses] Fix the graph-swap code (signature lookup returns None for already-rewired edges) and generate many mixed rewired graphs. |
| P4 | F | 4/5/12/0/0 | 21 | 4 | 12/8/7 | 1. [critical, tissue_specificity, 3 lenses] Recompute within cell-type x tissue strata with donor replication, or restrict the conclusion to aggregate associations.<br>2. [critical, stat_bio_alignment, 3 lenses] Gate prediction on balance and adequate counts (4 TRRUST edges, max abs SMD 2.557): the AUROC comparisons are uninterpretable.<br>3. [critical, confounder, 2 lenses] Add independent-network degree and run the degree-preserving null, or state that hub bias is unresolved. |
| P5 | F | 5/11/5/0/0 | 21 | 6 | 14/8/6 | 1. [critical, reproducibility, 3 lenses] Remove the __file__ dependency and rerun; the failed execution produced no result.<br>2. [critical, biological_implausibility, 2 lenses] Prespecify biological support criteria; audit flags are by construction (supporting_assay_available set from PMID presence).<br>3. [critical, overclaiming, 2 lenses] The exact binary co-presence gate is unnecessarily stringent; compare weighting/continuous adjustment before declaring non-identifiability. |

### 3.3 gpt55

| step | execution | exec wall (s) | code lines | executor step (s) | what was computed (estimand; key numbers with intervals) | code-generated verdict |
|---|---|---|---|---|---|---|
| E1 | exit 0, OK | 165.0 | 489 | 307.777 | Universe: 126 TFs with >= 1 TRRUST edge (166,446 pairs, 424 positives). Estimand: residual SMD after linear residualisation on 12 covariates incl. circular degree; pair-level Welch t; BH over 468. 39 q < 0.05 (5 positive, 34 negative). Selected by smallest p: head_L8_H10_sym_mean, SMD -0.103 [-0.222, -0.041], q 6.7e-15. Baselines: circular degree AUROC 0.954, in-sample combined 0.974, abs Pearson 0.538, rank proximity 0.489. Max layer-mean AUROC 0.663 (L2 tf_query). Reversed edges 0.540; DoRothEA transfer 0.504. | computed, but only for the smallest-p variant, which had a negative SMD. The 5 positive q < 0.05 variants were ignored: 'does not provide BH-FDR significant evidence ...' |
| E2 | exit -9, TIMEOUT (SIGKILL) | 900.1 | 829 | 1120.638 | No result: killed at 900.1 s. Partial: LOO-degree AUROC 0.853 / AUPRC 0.0553; held-out combined confound AUROC 0.895; min BH q 0.011 (TF-cluster sign-flip). | none |
| E3 | exit 0, OK | 548.5 | 1083 | 810.375 | TF-balanced residual SMD with TF-cluster permutation p: 0 positive and 22 negative BH-significant. Best positive head_L9_H4_tf_query SMD 0.186 [0.080, 0.289] (post-selection), q 0.121, matched p 0.196, max-stat p 0.871 (30 permutations). AUPRC 0.0032 vs LOO-degree 0.0553. Incremental CV dAUPRC -0.0002. | `primary_positive_success...=0` (computed): 'hub/database structure is the dominant verified signal' |
| E4 | exit 0, OK | 650.9 | 1359 | 1559.916 | Plus cell-type composition covariates: 0 positive and 25 negative BH-significant. Best positive SMD 0.189 [0.088, 0.291], q 0.0585, matched p 0.109 (100), max-stat p 0.866 (499). AUROC 0.563, AUPRC 0.0032 (prevalence 0.0025). LOO-degree AUROC 0.853 / AUPRC 0.0553; split-aware degree AUPRC 0.0330. Nested-selection dAUPRC -0.0004 [-0.0023, 0.0013]. Reversed edges 0.490; DoRothEA transfer 0.519. | `primary_positive_success...=0` (computed): 'TRRUST label topology/degree dominates the verified predictive signal' |
| E5 | exit -9, TIMEOUT (SIGKILL) | 900.9 | 1337 | 1283.278 | No result: killed at 900.9 s in the continuous-NN matched null. Partial: min BH q 0.285; max AUPRC 0.0096; fold-local held-out CV AUROC 0.836 for confounds vs 0.832 with attention added. | none |

What changed and why (links to the preceding panel's critiques; `P1#2` = second item of P1's top-3):

- **E2** (P1#1, P1#2, P1#3): P1#1 (cluster-aware tests), P1#2 (LOO degree) and P1#3 (separate positive/negative selection) all implemented; the permutation load exceeded the 900 s limit.
- **E3** (P2#1, P2#3, P2 critique[9] claim_validity): P2#1 fixed (runtime reduced, checkpoints). P2#3 implemented (family max-stat, but only 30 permutations). P2 critique[9] fixed (TF-balanced estimand used for both effect and test). Cluster-aware inference removed all positive BH-significant results seen in E1.
- **E4** (P3#1, P3#2, P3#3, P3 critique[3] leakage, P3 critique[4] confounder): P3#2 fixed (max-stat 30 -> 499). P3#3 fixed (CIs labelled post-selection; nested selection added). P3 critique[3] (degree relabelled as topology null + split-aware degree) and critique[4] (cell-type composition covariates) implemented. P3#1 answered only by documenting unavailability.
- **E5** (P4#2, P4#3): P4#2 implemented (1000 matched permutations, continuous NN matching) and P4#3 implemented (fold-local topology; its partial output shows confounds+attention CV AUROC below confounds-only). The added matched-null cost exceeded the 900 s limit.

| panel | grade | critical/high/medium/low/info | merged critiques | multi-lens groups | raw per lens (rev/adv/bio) | top-3 required fixes |
|---|---|---|---|---|---|---|
| P1 | F | 4/4/11/0/1 | 20 | 4 | 10/8/6 | 1. [critical, statistics, 2 lenses] Replace pair-level Welch t-tests (p down to 1.4e-17) with TF-cluster-aware tests before BH.<br>2. [critical, confounder, 2 lenses] Use leave-one-edge-out or external degree covariates (circular degree-only AUROC 0.954).<br>3. [critical, p_hacking, 2 lenses] Report positive and negative effects separately: the verdict used the smallest-p variant regardless of sign while 5 positive-SMD variants had q < 0.05. |
| P2 | F | 6/4/13/1/0 | 24 | 5 | 12/9/9 | 1. [critical, execution_failure, 3 lenses] Make the script finish within 900 s (vectorize, checkpoint) and report only numbers from completed files.<br>2. [critical, stat_bio_alignment, 2 lenses] Frame any positive attention result as secondary to degree/hub structure (LOO-degree AUPRC far above attention).<br>3. [critical, p_hacking, 2 lenses] Complete a family max-statistic null or prespecify the family; the partial table shows only negative BH-significant variants. |
| P3 | D | 1/4/5/4/1 | 15 | 3 | 6/8/4 | 1. [critical, tissue_specificity, 2 lenses] Use immune-specific references or cell-type stratification (TRRUST is context-generic).<br>2. [high, null_model, 2 lenses] Increase family max-statistic permutations beyond 30, or treat the 22 negative BH findings as exploratory.<br>3. [high, p_hacking, 2 lenses] Label selected-variant CIs as post-selection, or nest selection inside an outer held-out-TF split. |
| P4 | D | 0/3/8/3/1 | 15 | 2 | 6/7/4 | 1. [high, biological_implausibility, 2 lenses] Annotate the evidence context of top pairs (only 3/20 flagged by the immune keyword panel) or restrict to context-generic association.<br>2. [high, null_model, 2 lenses] Use >= 1000 matched within-TF permutations with continuous/caliper matching and balance diagnostics.<br>3. [high, leakage, 1 lens] Use fold-local topology (degree) features in held-out-TF CV; the globally computed LOO degree leaks test-fold labels. |
| P5 | F | 7/2/8/0/0 | 17 | 7 | 10/10/6 | 1. [critical, stat_bio_alignment, 3 lenses] State that the verified partial run shows no significant attention signal beyond topology (top q ~0.285; LOO-degree AUPRC 0.0553 vs best attention AUPRC 0.0096).<br>2. [critical, execution_validity, 3 lenses] Finish within the 900 s limit (timed out at 900.87 s before results.json) and report only produced numbers.<br>3. [critical, biological_implausibility, 2 lenses] Stratify or exclude housekeeping/mitochondrial/ribosomal targets and validate top pairs with immune-context evidence. |

### 3.4 sol_planonly (control, execution disabled)

| step | executor output tokens | behaviour |
|---|---|---|
| E1 | 865 | Refused: workspace read-only, no shell/Python; asked whether to proceed with a plan only. No plan, no numbers. |
| E2 | 717 | Refused again; offered a 'plan-only, non-evidential response' if explicitly authorised. No plan, no numbers. |
| E3 | 690 | Refused; listed available tools as apply_patch, update_plan and image viewing; requested shell/Python, kit read access and write access. |
| E4 | 802 | Labelled 'Execution failure'; no scientific verdict; listed minimum capabilities for a rerun. |
| E5 | 1164 | Formally marked the iteration 'FAILED and excluded from scientific synthesis'; listed analyses it could not compute. |

| panel | grade | critical/high/medium/low/info | merged critiques | multi-lens groups | raw per lens (rev/adv/bio) | top-3 required fixes |
|---|---|---|---|---|---|---|
| P1 | F | 5/3/7/0/0 | 15 | 5 | 9/8/3 | 1. [critical, context_mismatch] Rerun with executable access and report cell-type-stratified or cell-type-adjusted results.<br>2. [critical, null_model] Define the pair universe and confounder-preserving nulls; report incremental AUROC/AUPRC with bootstrap CIs.<br>3. [critical, overclaiming] Fit cross-validated confounder-only vs confounder-plus-attention models with leakage-safe splits. |
| P2 | F | 4/3/2/0/0 | 9 | 3 | 9/1/2 | 1. [critical, incomplete_analysis] Rerun where the kit can be read and computed on, then report the full preregistered analysis.<br>2. [critical, biology] Report TRRUST/DoRothEA-specific and mode-specific results and state what attention direction can imply.<br>3. [critical, tissue_specificity] Evaluate within adequately represented cell types. |
| P3 | F | 8/2/5/0/0 | 15 | 3 | 15/2/1 | 1. [critical, noncompletion] Rerun with Python access, or classify the run as an execution failure rather than a research answer.<br>2. [critical, context_mismatch] Report cell-type-stratified TF-target evaluations with within-type presence checks.<br>3. [critical, confounder] Condition on rank distance, co-presence, expression mean/variance and dropout. |
| P4 | F | 3/9/6/0/0 | 18 | 3 | 16/2/4 | 1. [critical, stat_bio_alignment] Rerun with programmatic kit access and report adjusted effect sizes, CIs, nulls, FDR and a verdict.<br>2. [critical, regulatory_relationship] Check top pairs against TRRUST/DoRothEA and orthogonal binding/perturbation evidence.<br>3. [critical, tissue_specificity] Evaluate within cell types / lineages. |
| P5 | F | 2/4/0/0/0 | 6 | 0 | 5/0/1 | 1. [critical, negative_control] Evaluate attention against co-expression, expression/rank, co-presence and degree baselines and label/degree-preserving nulls.<br>2. [critical, claim_validity] Rerun with kit access and report prespecified scores, adjusted effects, CIs and BH-FDR tests.<br>3. [high, tissue_specificity] The kit has only pooled attention: regenerate per-cell attention or restrict to pooled association (bio lens noting a kit limit). |

## 4. Final iterations vs the independent validation

Iteration used: sol_A E5. For sol_B and gpt55, E5 failed, so E4 (the last successful step) is compared. Marks: **agree**, **disagree**, **not addressed**, **not directly comparable** (the run reached a conclusion on a different estimand). Validation numbers come from `experiments/case_study/results/independent_validation.json` (layer index 0-based).

The seven claims are this synthesis's decomposition, not the validation's own structure. The validation pre-specifies four statements:

1. Attention is approximately symmetric: not supported (claim 5 here).
2. Attention recovers TRRUST edges above co-expression: supported (claims 1-2; claims 3-4 are its post-hoc qualifications).
3. Attention is explained by expression rank proximity: not supported (median |Spearman| over layers 0.242). It has no row here; every run used rank distance only as a covariate.
4. Reference-degree bias inflates recovery: supported. It has no row of its own; gpt55 E4's degree-dominance verdict maps onto it.

| # | claim | validation verdict and numbers | sol_A (E5) | sol_B (E4) | gpt55 (E4) |
|---|---|---|---|---|---|
| 1 | attention recovers TRRUST edges above chance | supported (sym AUROC 0.673 [0.631, 0.708] at layer idx 2; all 12 layer CIs above 0.5; AUPRC 0.0037 vs base rate 0.0014) | **not addressed**: E5 reports only increments over a nuisance model. E1 results.json lists raw AUROCs (layer-mean max 0.672, layer 2 log-geometric; head max 0.674; TF-query/target-query AUROCs within 0.0011 of the validation) without a conclusion | **not addressed**: E4 withheld all effects (balance gate failed). E1 reported a descriptive AUROC of 0.664 for TRRUST layer_mean_L02_forward without a conclusion. | **not addressed (numbers agree)**: no explicit conclusion; attention_tests.tsv lists raw AUROC/AUPRC for all 468 scores (layer means identical to the validation's 126-TF universe within 5e-8; max 0.663 layer_mean_L2_tf_query); report emphasises the selected head's AUPRC lift over prevalence (0.00065) |
| 2 | attention beats co-expression baselines on raw edge ranking (AUROC) | supported for AUROC (sym idx2 minus abs Spearman AUROC 0.128 [0.081, 0.174], BH p 0.0045; 14 layer x variant tests pass BH); the AUPRC difference (0.0011 [-0.0015, 0.0047]) is not significant | **not addressed**: No attention-vs-co-expression comparison in E5. E4 compared the full nuisance+attention model with a co-expression-only model (held-out AUROC 0.491), which is not a raw comparison. | **not addressed**: No comparison of attention with co-expression baselines in any iteration (co-expression entered only as matching covariates). | **not addressed (numbers agree)**: not stated; both numbers reported (abs Pearson AUROC 0.538, max layer-mean attention AUROC 0.663) |
| 3 | advantage over co-expression survives degree (hub) structure | not supported, by a post-hoc test added after inspection: the degree-matched advantage over co-expression passes BH in 0/36 tests (sym idx2: observed 0.128 vs null 0.132, BH p 0.995). Raw attention itself beats the degree-preserving null (sym idx2 0.673 vs 0.645, max-stat p 0.004; 12/36 pass BH), but the null reproduces 64%-102% of attention's excess AUROC. LOO target in-degree alone reaches AUROC 0.829. | **not directly comparable**: sol_A never compared attention with co-expression. Its degree-preserving rewiring null (E5: p 0.4545 for dAUROC and dAUPRC, 10 rewires, minimum attainable p 0.0909; E4: p 0.70/0.56, 100 rewires) tests the increment over a nuisance model that already contains co-expression, rank and leakage-safe degree. That is the claim-4 estimand. The direction is consistent with the validation. | **not addressed**: E4 did not run its graph null ('not_run_balance_gate_failed'). E1 had claimed association beyond matched confounders while target in-degree remained imbalanced (cAUC-0.5 0.217); E2's unconstrained degree-preserving null gave p 0.00995 for raw mean log attention | **not directly comparable; agrees with validation statement 4 (degree bias)**: 'TRRUST label topology/degree dominates the verified predictive signal' (LOO-degree AUPRC 0.0553 and split-aware degree AUPRC 0.0330 both exceed best positive attention AUPRC 0.0032). This compares attention alone with degree alone. It does not test whether attention's advantage over co-expression survives a degree-matched null. It matches the validation's 'reference-degree bias inflates recovery' (same 126-TF universe: LOO degree-product AUPRC 0.0562, LOO target in-degree AUROC 0.830). |
| 4 | pair-specific signal after co-expression, expression, rank-distance adjustment and degree-preserving null | TRRUST: not supported (0/36 BH; best sym idx0 AUROC 0.612 vs null 0.575, BH p 0.0719). DoRothEA A-C: small component (24/36 BH; excess -0.004 to 0.039 AUROC) | **agree for TRRUST, on E4's evidence; DoRothEA not addressed**: E5 has incremental_information_beyond_measured_nuisance_detected = False (hard-coded literal). Its numbers: mean dAUROC 0.012 [-0.0018, 0.0258], dAUPRC 0.0046 [0.0017, 0.0076] (t-interval over 15 overlapping splits); alignment-null p 0.25/0.1875 (15 reps), degree-null p 0.4545 (10 reps). Neither E5 null could reject at 0.05 (minimum p 0.0625 and 0.0909). The null replicates average dAUPRC 0.0022 and 0.0030, and the observed-minus-null intervals include 0. The informative evidence is E4's: 200 alignment permutations and 100 rewirings, p 0.68/0.73 and 0.70/0.56. The estimand (incremental held-out prediction) differs from the validation's. The DoRothEA metric was retired in E5 (E4: dAUROC +0.029, dAUPRC +0.014, no CI). | **not addressed (declared non-identifiable); E1 disagreed for TRRUST**: verdict 'not_identifiable_from_supplied_aggregate_attention' (hard-coded); TRRUST increment 0.0 [0.0, 0.0] on 4 edges; DoRothEA -0.0041 [-0.042, 0.022] on 50 edges. E1 (not final) had concluded 'replicated_association_beyond_measured_confounders' (15 scores). | **agree for TRRUST (different estimand); DoRothEA not comparable**: primary_positive_success = 0: best positive TF-balanced SMD 0.189 [0.088, 0.291] (post-selection), BH q 0.0585, matched p 0.109, max-stat p 0.866; nested CV dAUPRC -0.0004 [-0.0023, 0.0013]. The estimand is a residual SMD after linear adjustment for 22 covariates including LOO degree (126-TF universe), not the validation's row/column-removed residual AUROC against a degree-preserving null. gpt55 also reports 25 negative BH-significant variants, for which the validation has no counterpart; they are not examined here. DoRothEA transfer of the selected head: AUROC 0.519 (not a pair-specific adjusted test). |
| 5 | attention (approximately) symmetric / direction readable from attention | attention is not symmetric (layer Spearman(A_ij, A_ji) 0.16-0.32; per-head median 0.10) | **not addressed (no symmetry claim)**: not measured; no statement that attention is symmetric (forward, reverse and symmetrised scores treated as separate variants) | **not addressed (no symmetry claim)**: not measured; no statement that attention is symmetric (forward, reverse and symmetrised scores treated as separate variants) | **not addressed (no symmetry claim)**: not measured; no statement that attention is symmetric (forward, reverse and symmetrised scores treated as separate variants) |
| 6 | TF->target directionality | no consistent direction: TF-query > target-query on TRRUST edges (Wilcoxon BH<0.05) in layers 2,3,8,9,10,11 but edges differ from non-edges in layers 2,3,4,11 only; DoRothEA edges negative in layers 1-7 | **not tested (hard-coded flag consistent with the validation)**: tf_to_target_directionality_supported = False (hard-coded literal; no forward-vs-reverse test); text: 'provides no evidence for TF-to-target direction' | **not addressed (final); E1 not directly comparable**: 'No orientation is privileged as biological information flow' (E3 MECH.md). E4 METHOD.md says the reverse, symmetric and directional readings 'remain computational alternatives, not biological information-flow directions'. E1 had BH-supported directional (log forward - log reverse) matched effects: TRRUST layer_mean_L01 0.066, DoRothEA layer_mean_L03 0.048; the validation's DoRothEA layer-3 edges have a lower forward/reverse ratio than non-edges (different control set) | **not addressed**: tf_query called 'the directional TF-to-target readout', others 'orientation controls'; reversed-edge control for the selected head AUROC 0.490 (p 0.626); no directionality conclusion |
| 7 | causal interpretation | not tested; observational summaries only | **agree**: causal_claim_supported = False (hard-coded) | **agree**: 'These findings are correlational. No ablation, activation patching, model rerun or biological intervention was available, so causation is not claimed.' | **agree**: 'Any positive or negative result here is correlational ... does not show that a head or layer causally mediates regulatory prediction.' |

Interpretation. The validation's headline is that raw recovery is real (sym AUROC 0.673 at layer index 2) but mostly reproduced by degree structure: a degree-preserving null reproduces 64%-102% of the excess AUROC. After co-expression, rank and degree are removed, no TRRUST signal survives; a few hundredths of AUROC remain with the larger DoRothEA set.

- **sol_A and gpt55** reach the TRRUST half of that conclusion through covariate-adjusted designs: incremental prediction for both, plus a residualised SMD for gpt55.
- **gpt55** reaches it with a verdict computed from its numbers.
- **sol_A** reaches it through a verdict fixed in advance. Its E5 dAUPRC interval excludes 0, but the null replicates average half to two-thirds of the observed increment (0.0022 and 0.0030 vs 0.0046) and the E5 nulls could not reach p < 0.05. E5 alone is therefore inconclusive. E4's properly powered nulls (p 0.56-0.73) are what support the negative.
- **Neither** reports the positive half (attention beats co-expression on raw AUROC), although gpt55's and sol_A E1's tables contain the numbers.
- **sol_B's E1**, the only positive verdict among the 15 executed steps, matched on target in-degree but left a residual degree imbalance larger than the attention effect. The panel caught this and the claim was withdrawn. The panel then asked for exact or calipered degree balance (P1) and same-cell or context matching (P1-P3). The executor added context-profile calipers and a hard balance gate, which reduced the design to 9 (E2) and then 4 (E4) TRRUST edges. Together with the unconditional non-identifiability verdict, this means the run does not answer the question.

DoRothEA. The validation finds a pair-specific component with DoRothEA A-C (24/36 tests pass BH; sym excess over the degree null -0.004 to 0.039 AUROC). No final iteration tested this: sol_A retired its DoRothEA estimate, gpt55 transferred one selected head (AUROC 0.519), and sol_B withheld effects. Three earlier estimates point the same way as the validation but were not carried forward:

- sol_A E4: dAUROC +0.029, dAUPRC +0.014, no interval.
- sol_B E2: layer-0 forward 0.129 [0.093, 0.164] on 76 matched edges; the balance gate failed.
- sol_B E3: forward 0.035 [0.014, 0.053] on all 1,949 edges, with max TF-macro abs SMD 1.106, so balance was not achieved.

## 5. Did any run assert that attention is symmetric?

No. Regex search for 'symmetr', 'asymmetr', 'A_ij|A_{ij}|A[i,j]', 'bi-?directional' and '(attention|matrix|A) (is|are|was) (approximately )?symmetric' over every executor_output.md (20 executor steps) and every code-generated *.md and *.json under exec_outputs/<run>/*/work/.

| run | files searched | lines with 'symmetr' | 'asymmetr' | 'A_ij' | 'bidirectional' | assertions of symmetry |
|---|---|---|---|---|---|---|
| sol_A | 44 | 1066 | 2 | 0 | 0 | 0 |
| sol_B | 29 | 678 | 0 | 0 | 0 | 0 |
| gpt55 | 26 | 5 | 0 | 0 | 0 | 0 |
| sol_planonly | 5 | 0 | 0 | 0 | 0 | 0 |

No run asserted that attention is symmetric. Every 'symmetr' hit is a score-variant name (symmetric_log_geometric, sym_mean, *_symmetric), a definition of the symmetrised score (mean of forward and reverse, or of their logs), or a statement that forward and reverse readings are treated 'symmetrically' as exploratory variants (sol_B E3). The two 'asymmetr' hits are a sol_A E3 section title about label-dependent degree asymmetry. All runs treated A as directed (query -> key) and tested both readings; none quantified the asymmetry itself. No panel asked for a symmetry measurement. sol_A P3 asked for a forward-vs-reverse test of directionality (critique[4]); sol_A answered with the literal flag `tf_to_target_directionality_supported=False` rather than a test. For reference, the validation finds head-mean attention is not symmetric (layer Spearman(A_ij, A_ji) 0.16-0.32, per-head median 0.10).

## 6. Errors the panel caught in executed code, and whether they were fixed

Each item was checked against the code or outputs (column 'verified').

| run | caught in | issue | verified | fixed in | status |
|---|---|---|---|---|---|
| sol_A | P1 critique[1] (top#2), 3 lenses | Nuisance model used TRRUST out/in-degree computed from the outcome labels including the evaluated edge (circular). | exec_outputs/sol_A/20261001T114521_9f7bcc7d/code.py lines 106-107 read genes.tsv trrust_*_degree_in_G | E3 (leave-source-out target degree, code lines 233-254) and E4 (training/development-label degrees) | fixed |
| sol_A | P1 critique[9] | Code comment claims squared terms allow curvature; none were in the design matrix. | exec_outputs/sol_A/20261001T114521_9f7bcc7d/code.py lines 114-150 | E3 (residualisation with linear + squared terms, per P3 critique[12]) | fixed |
| sol_A | P1 critique[5] | Conditional score test used unpenalised information while the nuisance probabilities came from an L2-penalised (C=1) logistic fit. | exec_outputs/sol_A/20261001T114521_9f7bcc7d/code.py lines 165-170, 241-249 | E3 replaced the inference procedure | replaced |
| sol_A | P1 critique[17] | StandardScaler fitted on the full candidate universe before grouped CV. | exec_outputs/sol_A/20261001T114521_9f7bcc7d/code.py lines 157-158; the verifier checked E3-E5 | E3: the predictive pipeline fits scalers on training rows only (code line 767); E4 line 465 and E5 line 579 do the same. E3's descriptive association screen still standardises over the full universe (line 319) but has no held-out evaluation. | fixed |
| sol_A | P1 critique[3]; again P3 top#3 | Negative control = one shuffled realisation with a parametric p-value; in E3 the single within-TF permutation's dAUROC exceeded the attention dAUROC. | results.json values | E4 (200 complete-pipeline permutations) | fixed after two panels |
| sol_A | P2 top#1, 3 lenses | E2 crashed: 310228 x 468 float32 memmap exceeds the 64 MiB per-file limit. | exec_outputs/sol_A/20261001T120035_7c020ce5/code.py lines 328-334 | E3 | fixed |
| sol_A | P3 top#2 | Selection forced an attention variant although all 36 worsened selection log-loss, and the selection and final learners differed. | results.json | E4 (no_attention candidate, one protocol) | fixed |
| sol_A | P3 critique[6] | Outcome-conditioned TF split (all zero-positive TFs in training). | E4 partitions include zero-positive TFs | E4 | fixed |
| sol_A | P3 critique[8] | Predictive target degree computed from the complete TRRUST network, including evaluation TFs (transductive leakage). | PATCH_NOTES E4 | E4 | fixed |
| sol_A | P4 critique[5] | Manifest MD5 verification checked 0 entries (parser lost file names) while PATCH_NOTES claimed all inputs verified. | results.json | E5 | fixed |
| sol_A | P4 critique[4] | Null-distribution quantiles labelled as 95% confidence intervals. | results.json E4 key names *_ci95 | E5 (reference_interval95) | fixed |
| sol_A | P5 top#2 | Observed statistic = mean over 15 nested splits; each null replicate = one nested split, so empirical p-values compare different sampling variances. | exec_outputs/sol_A/20261001T123825_0addd673/code.py lines 918-935 (one nested_pipeline call per replicate) | none (last panel) | not fixed |
| sol_A | P5 critique[14] | 15 alignment and 10 degree-null replicates give minimum attainable p values of 1/16 = 0.0625 and 1/11 = 0.0909, so neither null could reach 0.05. The critique reported both minima but drew the conclusion only for the degree null. | results.json | none | not fixed |
| sol_B | P1 critiques[5],[6] | Positive verdict despite residual imbalance: matched target in-degree still separates positives from controls more than any attention score does. | results.json | E2 (exact degree matching, balance gate, verdict withdrawn) | fixed (verdict withdrawn; design then collapsed to 9 edges) |
| sol_B | P1 critique[4] | Ordinary BH over 1248 tests used to claim cross-benchmark replication. | code verdict uses replicated_test_ids from per-test BH | E2 (partial conjunction) | fixed |
| sol_B | P3 top#2 | Controls selected with layer-0-attention-derived covariates and the same attention then tested (outcome-informed matching). | E3 results attention_centrality_adjustment.target_incoming_attention_included = true | E4 | fixed |
| sol_B | P3 top#3 | Graph-swap code compares proposed signatures with original_signatures.get(edge), which is None for already-rewired edges, so rewired slots can never swap again (0/424 accepted). | exec_outputs/sol_B/20261001T130416_fb49b45e/code.py lines 1071-1098 | E4 code (slot-invariant signatures) but the null was not run (balance gate failed) | fixed in code, not exercised |
| sol_B | P3 critique[12] | analysis_script_saved=false while RUN_SUMMARY instructed users to rerun the saved script. | stdout | E4 (script saved with SHA-256); the E5 rewrite of this step used __file__ and crashed | fixed, then regressed |
| sol_B | P4 critique[8] | METHOD.md says each bootstrap re-matches, but the code reuses initial_matchings. | exec_outputs/sol_B/20261001T131352_a547f4e6/code.py lines 1372-1394, 1484 | none (E5 crashed) | not fixed |
| sol_B | P4 top#2 | Held-out prediction on 4 TRRUST edges / 4 TFs gave a degenerate [0, 0] interval presented with the same structure as a result. | stdout | E5 redesign (crashed) | not fixed |
| sol_B | P5 top#1 | E5 NameError: __file__ undefined in the orchestrator. | exec_outputs/sol_B/20261001T132233_dfb1dd31/code.py line 1012 | none (last panel) | not fixed |
| gpt55 | P1 top#3 | Verdict logic took the smallest-p variant regardless of sign (a negative effect) while 5 positive-SMD variants had BH q < 0.05. | exec_outputs/gpt55/20261001T132742_0fff3492/code.py lines 359-363 and attention_tests.tsv | E3 (best positive and best negative reported separately) | fixed |
| gpt55 | P1 top#1 | Pair-level Welch t-tests over 166446 dependent pairs (p down to 1.4e-17). | stdout | E3 (TF-cluster permutation p-values) | fixed |
| gpt55 | P1 top#2 | Circular TRRUST degree covariates (degree-only AUROC 0.954). | stdout | E3 (LOO degree), E4 (relabelled topology null + split-aware degree), E5 (fold-local topology; partial output only) | fixed progressively |
| gpt55 | P1 critique[9] | Combined confound baseline evaluated in-sample. | stdout | E2/E3 (held-out-TF CV) | fixed |
| gpt55 | P2 top#1 | E2 timed out at 900 s. | CODE_EXECUTION.json | E3 | fixed |
| gpt55 | P3 top#2 | Family max-statistic null with 30 permutations. | results.json | E4 (499) | fixed |
| gpt55 | P4 top#3 | Held-out-TF CV used globally computed LOO degree features (test-fold labels leak). | exec_outputs/gpt55/20261001T142158_1700cb21/code.py line 632 (baseline_x = full covariate matrix) | E5 (fold-local topology computed before the timeout) | partially fixed |
| gpt55 | P5 top#2 | E5 timed out at 900.9 s during the continuous NN matched null. | CODE_EXECUTION.json; checkpoint_status.json | none (last panel) | not fixed |

Counts: of the 29 items, 20 were fixed (including fixed-after-two-panels, fixed-progressively and replaced), 1 was partially fixed, 1 was fixed in code but never exercised, 1 was fixed and then regressed, and 6 were not fixed (4 raised by the last panel, 2 because the next execution crashed). Circular TRRUST-derived degree covariates appeared independently in all three executed runs at E1. P1 flagged them as leakage in sol_A and gpt55. In sol_B, P1 flagged the resulting residual target in-degree imbalance after matching, not leakage.

## 7. Panel critiques that were wrong, inconsistent, or missing

Factually incorrect critiques:

- **gpt55 P2 critiques[21] (medium, statistics)**: The partial attention_tests.tsv 'visibly ends with an invalid q value of 0 in the last shown row'. The report showed a truncated row; the actual q of that row (head_L8_H9_sym_mean) is 0.606 and no q in the file is 0.
- **sol_B P5 critiques[20] (medium, reproducibility)**: Script name 'analysis_iteration9.py' is wrong 'despite this being Iteration 10'. The executor step was loop iteration 9 (its prompt says 'Iteration 9'); iteration 10 is the panel itself. The lens prompt was headed 'REVIEW REQUEST (Iteration 10)', so the reviewer conflated its own iteration with the executor's.
- **sol_B P4 critiques[12] (medium, regulatory_relationship)**: Fix asks to 'Verify the TRRUST record's regulatory mode' for HIF1A-COX4I1. HIF1A-COX4I1 is a DoRothEA (confidence A) edge and is absent from the kit's trrust_edges.tsv (database misattribution; the biological point about the COX4 isoform switch is not contested here).

Apparent reversals across iterations. On re-reading the critiques, none is a clean self-contradiction by the panel:

- **sol_A, degree null: not a panel contradiction.** P2 critiques[5],[10] criticised how the E2 degree-preserving edge-swap chain was implemented: the statistic was not recomputed per rewired network, and there were no burn-in or mixing diagnostics. Both asked for it to be fixed. E3 removed the null instead. P3 critique[7] (high) then asked for one, and E4 re-added it. The panel never asked for removal; the executor chose it.
- **sol_A, verdict framing: a reversal after the evidence changed.** P4 top#2 was written on E4's results (dAUROC -0.0106, dAUPRC 0.0005, 200/100-replicate nulls non-significant) and required an explicit negative verdict. E5 hard-coded 'no supported incremental information'. P5 critique[18] was written on E5's results (positive dAUPRC interval, 15/10-replicate nulls) and asked for an 'inconclusive' framing.
- **sol_B, same-cell matching: a partial reversal.** P1 top#3 asked for 'identical cell sets, or at minimum within cell-type and sequence-length strata'. P2 top#1 asked for 'identical positive/control cell sets, or restrict the biological conclusion to an unlocalized aggregate association'. P3 critique[4] asked for same-cell comparisons. E2-E4 used exact co-presence-signature requirements and treated their failure as proof of non-identifiability. P5 top#3 called the exact gate 'unnecessarily stringent'. P1 and P2 had offered fallbacks (strata, or an aggregate-only conclusion) that the executor did not take.

Requests for data the kit does not contain (per-cell or cell-type-specific attention, immune ChIP/perturbation labels, pretraining-corpus manifest, raw expression variance, TF protein activity) appear in every panel. A keyword heuristic counts 3-12 such critiques per executed-run panel, 1-8 of them critical or high (per-panel counts in the JSON). Such demands were usually the top-1 or top-2 item: cell-type stratification was the top-1 fix in sol_A P1/P3, sol_B P2/P4 and gpt55 P3. Executors answered with `unavailable_analyses` flags (sol_A, gpt55) or by declaring the question non-identifiable (sol_B).

Defects the panel did not catch:

- **sol_A**: In E5 the primary flag incremental_information_beyond_measured_nuisance_detected=False and the statement are literals fixed before execution, while the computed dAUPRC CI excluded 0. E4's literals were scope flags (causal, immune context, directionality, unmeasured confounders), not a verdict on the primary question. P5 critique[18] noted the tension between the positive AUPRC interval and the negative verdict but not that the verdict was hard-coded.
- **sol_B**: verdict_code assigned unconditionally from E2 onward (E2 comment: 'even favorable score statistics cannot answer the research question affirmatively'). No panel critique identifies the unconditional assignment; searching all CONSENSUS.json files for 'hard-coded' finds only audit flags and the script name.
- **gpt55**: E3/E4 success criterion requires the standalone attention AUPRC to exceed the LOO-degree AUPRC by 0.001, which tests 'better than degree alone' rather than 'information beyond degree' (the nested incremental CV is the relevant test). Panels discussed the practical gap to the degree baseline but not the logic of this criterion.
- **sol_A**: E5 TF-query-only family gave dAUROC CI entirely below 0 and dAUPRC CI entirely above 0 (mixed-sign result). P5 mentions the TF-query family only in a multiplicity critique.
- **all**: No run measured attention symmetry (A_ij vs A_ji), and no panel requested it; the validation finds layer Spearman(A_ij, A_ji) 0.16-0.32. sol_A P3 critique[4] asked to test whether the forward orientation preferentially recovers TF->target edges; E4/E5 answered with a literal tf_to_target_directionality_supported=False instead of a test.

## 8. Plan-only control

The control never reported a number and never produced an analysis plan. Executor output was 865, 717, 690, 802, 1164 tokens across E1-E5. In every step it said it lacked a shell/Python tool and a writable workspace and declined to produce results (section 3.4). Grade F in all five panels; every lens wrote that refusing to fabricate was appropriate but that the artefact answers nothing. P1 critique[3] asked for 'a complete preregistered plan explicitly labeled as non-evidential'; the executor never provided one. P2 critique[8], P3 critique[10] and P4 critique[12] noted that read-only workspace access does not by itself prevent returning results in the response. P5 (6 critiques, 0 multi-lens groups; adversarial lens raised 0) noted the kit only has pooled attention. The plan-only task text equals the executed text minus its 'COMPUTING ENVIRONMENT AND REPORTING RULE' section, so it still asked for 'a quantitative analysis'; with no tools and no execution the executor treated the task as infeasible. The control therefore shows that, without code execution, this executor produced neither numbers nor a plan: it did not fabricate results. It does not show what an unexecuted plan would have looked like, because none was written. In E1 and E2 it asked for authorisation to write a plan, and it did not act on P1's suggestion to submit one. Tokens: 386,702 in total, 6.0 min wall-clock.

## 9. Failure modes

| run | step | failure | exec wall (s) |
|---|---|---|---|
| sol_A | E2 | file-size limit (RLIMIT_FSIZE 64 MiB) | 5.1 |
| sol_B | E5 | NameError (__file__) | 72.2 |
| gpt55 | E2 | wall-clock timeout 900 s | 900.1 |
| gpt55 | E5 | wall-clock timeout 900 s | 900.9 |

Near miss: sol_A E4 ran for 814.3 s of the 900 s limit. The 64 MiB per-file limit was not in the task text, which states only the 900 s wall and 1800 s CPU limits. It was, however, stated in the E1 execution report embedded in sol_A's E2 prompt ('max_file_mb: RLIMIT_FSIZE=67108864 bytes per file'), and E2 still allocated a memmap of about 581 MB.

Lines of executed code per step (E1-E5):

| run | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| sol_A | 1025 | 2233 | 1714 | 1694 | 1837 |
| sol_B | 1096 | 1850 | 2211 | 1841 | 1369 |
| gpt55 | 489 | 829 | 1083 | 1359 | 1337 |

Execution wall time per step (s):

| run | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| sol_A | 317.4 | 5.1 | 70.9 | 814.3 | 117.0 |
| sol_B | 12.5 | 119.5 | 85.0 | 104.8 | 72.2 |
| gpt55 | 165.0 | 900.1 | 548.5 | 650.9 | 900.9 |

- **Scope creep.** Panels of the three executed runs returned 15-24 merged critiques each (18 of all 20 panels graded F); executors tried to address most of them in a single script, which grew from 1025 to 2233 lines (sol_A), 1096 to 2211 (sol_B) and 489 to 1359 (gpt55). Three of the four failures (sol_A E2, gpt55 E2, gpt55 E5) occurred in steps that added large permutation/bootstrap or storage machinery in response to inference and null-model critiques; sol_B E5 failed in code added for a reproducibility request.
- **Design drift.** sol_B moved from a 5:1 matched design on all 424 TRRUST edges (E1) to exact/caliper matching that retained 9 (E2) and 4 (E4) TRRUST edges, and from a computed positive verdict (E1) to an unconditional non-identifiability verdict (E2-E5).
- **Hard-coded verdicts.** sol_A E5 (primary flag and statement) and sol_B E2-E5 (verdict code) wrote the answer to the research question as a literal, so the reported conclusion could not respond to the computed numbers. In sol_A E5 it did not: the primary dAUPRC interval excludes 0, yet the flag says no incremental information. sol_A E4's literals were scope flags (causal, immune context, directionality), mostly for claims the kit cannot support. sol_B's E2 code comment states the reason: 'even favorable score statistics cannot answer the research question affirmatively'. The number trace (`trace_summary.json`) classifies only numbers, so it does not flag these literals.
- **Underpowered nulls after time pressure.** sol_A cut its null replicates from 200/100 (E4) to 15/10 (E5). With 15 alignment permutations the smallest attainable p is 0.0625, and with 10 rewirings it is 0.0909, so neither E5 null could reject at 0.05.
- **Executor prose.** All 15 executed-run executor responses were a single fenced Python block with no prose; every reported number therefore lives in code-generated reports (trace_summary: 0 agent-text numbers; 460/149/3 generated-report numbers for sol_A/sol_B/gpt55, all traceable).

## 10. Limitations of this synthesis

- One replicate per configuration except gpt-5.6-sol (2 replicates). sol_A and sol_B diverged completely in design (incremental prediction vs matched case-control) and in final verdict, so between-replicate variance is large and model comparisons are not warranted.
- The comparison maps each run's own estimand onto the validation's claims. Where a run reached a conclusion on a clearly different estimand (claim 3 for sol_A and gpt55), the mark is 'not directly comparable'. For claim 4, sol_A and gpt55 test incremental prediction or residual SMDs rather than the validation's row/column-removed residual AUROC against a degree-preserving null. 'Agree' there means the same direction of conclusion, not the same test. Universes also differ: sol_A uses co-presence >= 100 (310,228 pairs), gpt55 the 126 TFs with an edge (166,446 pairs), sol_B co-presence >= 30 with matched controls, and the validation co-presence >= 50 (310,435 pairs).
- Claim 3 ('not supported') rests on a post-hoc test in the validation. The validation's pre-specified verdict for 'recovers TRRUST edges above co-expression' is 'supported', and raw attention does beat the degree-preserving null.
- Bug verification covered the specific code lines cited. 'Fixed' means the next executed code no longer contains the defect, or the outputs show the corrected behaviour; the corrected code was not independently re-run.
- The count of requests for absent data is a keyword heuristic. The top-3 fixes are the first three merged critiques in the order the platform stores them (severity first); the one-line summaries are paraphrases, and the verbatim first sentence of each fix is in the JSON.

Files: `case_synthesis.json` (all numbers with sources), this file. Generated from the run artefacts; nothing outside `experiments/case_study/agent_runs/results/` was modified.

## Verification notes

An independent verifier re-opened the source files and checked every number and factual statement in this file and in `case_synthesis.json`. Counts were recomputed with python. The corrections below were made in place in both files. `synthesis_scripts/verify_patch_json.py` re-applies the JSON corrections, and the edits to this file were made by hand. Re-running `build_synthesis.py` / `build_md.py` would restore the pre-verification versions.

### What was checked

**Run-level metadata** (`run_meta.json`): tokens (total, input, output, cached, executor vs panel), agent calls, prompt-file counts, wall-clock, failed and timed-out executions, grade histories and tool settings for all four runs. All matched.

**Execution records** (`CODE_EXECUTION.json` and `exec_outputs`): exit codes, limit hits, wall times, code line counts, stderr last lines, written files, and the absence of prose in the 15 executed responses. All matched.

**Key numbers in the per-step tables.** These were re-read from `results.json`, `stdout.txt` and the tsv tables, including the partial outputs of gpt55 E2/E5 and sol_B E5:

- sol_A E1, E3, E4, E5
- sol_B E1, E2, E3, E4
- gpt55 E1, E3, E4

The automated pass covered 654 sourced entries resolved by key and 155 recomputed entries, with 0 mismatches. The remaining entries (run-level sums, heuristic counts, quotes, plan-only paraphrases) were checked by hand.

**Verdict code paths:**

- sol_A E1 and E3 verdicts are conditional. sol_A E4 and E5 use literals (E5 lines 1439-1443 and 1834).
- sol_B E1 is conditional. sol_B E2-E5 assign the verdict unconditionally.
- gpt55 E1 selects the smallest-q variant regardless of sign. gpt55 E3 and E4 use a computed conjunction (E4 lines 956-964).

**Panels** (`CONSENSUS.json`): grade, severity histogram, merged count, multi-lens groups, raw counts per lens, and the severity, category and lens count of each top-3 item. All 20 panels matched. The top-3 paraphrases were compared with the verbatim fixes, and every critique cited in sections 3 and 6-8 was re-read.

**Validation numbers:** checked against `independent_validation.json` and `SUMMARY.md`, including the layer conversion (0-based vs 1-based), the 14/36 and 24/36 counts, the 64%-102% range, and the directionality layer lists.

**Cross-checks recomputed:**

- gpt55's layer-mean AUROCs match the validation's to a maximum difference of 4.7e-8.
- sol_A's match to a maximum difference of 0.00103.

**Section 5 searches:** the symmetry regex counts (44/29/26/5 files; 1066/678/5/0 lines). A further search found that no panel critique requests a symmetry measurement.

**Section 7 search:** the hard-coded-verdict search across all lens outputs.

**Other spot checks:** the wrong-critique evidence (E2 q of 0.606; HIF1A-COX4I1 present only in `dorothea_abc_edges.tsv` and labelled dorothea_exclusive by sol_B; the iteration headers in the prompts), the plan-only outputs and lens assessments, and the diff between the two task texts.

### What was changed

**1. Claim 3 is now 'not directly comparable' for sol_A and gpt55 (previously 'agree').**

- The validation's claim 3 is a post-hoc test: attention's AUROC advantage over co-expression against a degree-matched null.
- sol_A never compared attention with co-expression. Its rewiring null tests the increment over a nuisance model that already contains degree, which is the claim-4 estimand.
- gpt55 compared attention alone with degree alone. That matches the validation's pre-specified statement 4 ('reference-degree bias inflates recovery': supported), which it agrees with.
- The validation column now notes that raw attention does beat the degree-preserving null (0.673 vs 0.645, max-stat p 0.004). The earlier summary wording, 'nothing survives degree structure', was wrong.

**2. Claim 4 for sol_A: still 'agree for TRRUST', with a corrected basis.**

- The agreement rests on E4's properly powered nulls (200/100 replicates, p 0.56-0.73).
- Neither E5 null could reach p < 0.05: the minimum attainable p is 1/16 = 0.0625 for the alignment null, not only 1/11 = 0.0909 for the degree null as the text had said.
- The E5 dAUPRC interval that 'excludes 0' is a t-interval over 15 overlapping splits. The null replicates average dAUPRC 0.0022 (alignment) and 0.0030 (degree), against 0.0046 observed, and the observed-minus-null intervals include 0.
- The 'inconclusive, not negative' reading of E5 stands, with this clarification.

**3. Claim 4 for gpt55:** marked as a different estimand. Its 25 negative BH-significant variants are noted.

**4. Claim 6 for sol_A:** changed from 'agree in conclusion' to 'not tested'. The flag is a literal and no forward-vs-reverse test was run.

**5. Claim 2:** relabelled as 'raw edge ranking (AUROC)' instead of 'raw recall'. The validation's AUPRC advantage over co-expression is not significant (0.0011 [-0.0015, 0.0047]).

**6. Section 4 framing:** now states that the seven claims are this synthesis's decomposition of the validation's four pre-specified statements. The rank-proximity statement (not supported) and the degree-bias statement (supported) have no row of their own.

**7. Numbers and wording corrected:**

- sol_A's AUROC agreement is 'within 0.0011', not 'within 0.001'.
- sol_A E1's 0.672 is the layer-mean maximum; the head maximum is 0.674.
- sol_B E1 ran 624 scores x 2 benchmarks = 1,248 tests, not '1,248 scores'.
- The gpt55 E1 verdict was not 'sign-agnostic': the rule requires SMD > 0, but it tests only the smallest-p variant, which was negative.
- sol_B's 'No orientation is privileged' quote is from E3 only; E4's equivalent is in METHOD.md.
- The 'treated symmetrically' statement is from sol_B E3 only.
- The duplicated gpt55 P1 prompts were network-error retries.

**8. Panel bug table:**

- sol_A P1 critique[17] (StandardScaler) was 'not checked'. The verifier confirmed it was fixed in E3-E5, so the counts are now 20 fixed and 0 unchecked.
- The sol_A P5 critique[14] row now notes both null minima.

**9. Degree leakage:** 'leakage in all three runs' is corrected. The circular degree covariates were present in all three runs, but sol_B's P1 flagged residual target in-degree imbalance, not leakage.

**10. 'Contradictions' reclassified:**

- sol_A's degree-null episode is not a panel contradiction: P2 asked for the null to be fixed, and the executor removed it.
- sol_A's verdict-framing reversal followed a change in the evidence between E4 and E5.
- sol_B's same-cell episode is a partial reversal: P1 and P2 had offered fallbacks.
- The summary now says 'reversed its framing once in sol_A and partly reversed itself once in sol_B'.

**11. Hard-coded verdicts narrowed.**

- sol_A E4's literals are scope flags (causal, immune context, directionality, unmeasured confounders), not a verdict on the primary question.
- The primary verdict is hard-coded in sol_A E5 and sol_B E2-E5.

**12. Failure modes:**

- Three of the four failures, not two, came in steps that added permutation/bootstrap or storage machinery: gpt55 E2 was killed in the permutation stages.
- The 64 MiB file limit was absent from the task text but present in the E1 execution report shown to sol_A's E2.

**13. sol_B interpretation:** the collapse to 4 TRRUST edges is now attributed to exact/calipered degree balance plus context calipers and a hard balance gate, not only to 'identical-cell matching'. sol_B E3's DoRothEA estimate was added to the DoRothEA paragraph.

### Remaining caveats

- **Unreplicated and partly paraphrased.** The verifier did not re-run any agent code. The 'fixed' statuses rest on reading the next step's code and outputs. The top-3 summaries and the plan-only behaviour descriptions are paraphrases, checked against the source text.
- **The seven-claim mapping is a judgement call.** Even the 'agree' marks compare different estimands, universes (co-presence >= 30/50/100; 235 vs 126 TFs) and multiplicity families.
- **One unresolved discrepancy.** gpt55's 25 negative residual-SMD findings have no counterpart in the validation. Whether they come from residualising on LOO degree was not investigated.
- **Panel-search negatives depend on wording.** 'No panel noticed hard-coded verdicts' and 'no panel asked for symmetry' come from keyword searches of the critiques and lens outputs. A paraphrased mention could have been missed.
- **No independent judgement of the validation.** Its own post-hoc analyses were taken at face value.
