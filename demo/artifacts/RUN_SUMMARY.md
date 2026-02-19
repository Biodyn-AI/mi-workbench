# Run Summary: Synthetic GRN Recovery from Transformer Attention

**Run ID:** demo-run-001
**Date:** 2026-01-15
**Iterations:** 3 (executor x2, reviewer x1)
**Total tokens:** 42,800
**Total cost:** $0.85
**Status:** Completed (Grade B)

## Task

Analyze attention patterns in a 6-layer transformer trained on synthetic gene
expression data (100 genes, 500 cells) to determine whether attention weights
recover known regulatory edges from a synthetic ground-truth network (150 edges).

## What Happened

### Iteration 1: Initial Execution
The executor designed and ran the complete analysis pipeline:
- Extracted attention from all 6 layers and 24 heads.
- Computed AUROC for attention, correlation, and mean expression baselines.
- Ran degree-preserving null model (1000 permutations).
- Performed residualization analysis.
- Produced initial MECH.md, EVAL.md, XP.md, METHOD.md.

### Iteration 2: Review
The reviewer identified two issues:
1. **MEDIUM:** Per-head analysis was missing. Added in revision.
2. **LOW:** Bootstrap CIs were computed on 500 resamples; increased to 1000.

Overall grade: C+ (required per-head analysis before acceptance).

### Iteration 3: Revision
The executor added per-head AUROC analysis and strengthened bootstrap CIs.
Reviewer accepted with grade B.

## Key Takeaways

1. **Attention works, but so does correlation.** Layer 4 attention achieves AUROC
   0.672, but pairwise correlation achieves 0.658 -- not significantly different
   (p = 0.34). The model learns co-expression, not regulation per se.

2. **Residualization is the decisive test.** After removing co-expression signal,
   attention drops to AUROC 0.531 (78.6% signal loss). Correlation barely changes
   after removing attention (2.4% signal loss). This asymmetry is the core finding.

3. **No head specialization for regulation.** All 24 heads show similar AUROC
   (range 0.487-0.634), with no evidence of dedicated "regulatory heads."

4. **Null models are essential.** The degree-preserving null confirms the signal is
   real (Z = 14.0), ruling out degree-distribution artifacts.

## Limitations and Caveats

- This is synthetic data. Real biological systems have more complex regulatory
  dynamics, noise structures, and incomplete ground truth.
- The 100-gene scale is much smaller than real transcriptomes (20,000+ genes).
- Single training run; reproducibility across random seeds not assessed.
- The masked gene prediction objective may not be the best proxy for learning
  regulatory relationships.

## Artifacts Produced

| Artifact | Path | Size |
|----------|------|------|
| MECH.md | artifacts/MECH.md | 4.2 KB |
| EVAL.md | artifacts/EVAL.md | 3.8 KB |
| XP.md | artifacts/XP.md | 3.1 KB |
| METHOD.md | artifacts/METHOD.md | 3.5 KB |
| RUN_SUMMARY.md | artifacts/RUN_SUMMARY.md | 2.9 KB |

## Suggested Follow-Ups

1. **Scale to 1000 genes** -- Does the co-expression dominance persist at larger scale?
2. **Real data validation** -- Repeat on Tabula Sapiens with TRRUST as ground truth.
3. **Alternative training objectives** -- Test regulatory-aware fine-tuning.
4. **Cross-cell-type analysis** -- Does attention structure differ between cell types?
5. **Causal intervention** -- Ablate top regulatory heads and measure perturbation
   prediction impact.
