# Mechanistic Findings: Attention Recovery of Synthetic Regulatory Edges

**Experiment:** Synthetic GRN recovery from 6-layer transformer attention
**Date:** 2026-01-15
**Status:** Iteration 3 (post-review revision)

## Summary

We investigated whether attention weights in a 6-layer, 4-head transformer trained on
synthetic gene expression data (100 genes, 500 cells) recover known regulatory edges
from a synthetic ground-truth network (150 edges). The model was trained on rank-value
encoded expression vectors using a masked gene prediction objective.

## Key Findings

### Finding 1: Layer 4 attention maximally recovers regulatory structure

Attention weights aggregated from Layer 4 (0-indexed: Layer 3) show the highest AUROC
for distinguishing true regulatory pairs from non-regulatory pairs.

| Layer | AUROC | AUPRC | Top-100 Precision |
|-------|-------|-------|--------------------|
| L1    | 0.523 | 0.167 | 0.18               |
| L2    | 0.561 | 0.198 | 0.22               |
| L3    | 0.598 | 0.234 | 0.29               |
| **L4** | **0.672** | **0.312** | **0.38**   |
| L5    | 0.641 | 0.287 | 0.34               |
| L6    | 0.612 | 0.251 | 0.30               |

Best layer AUROC = 0.672 (95% CI: [0.638, 0.706], bootstrap n=1000).

### Finding 2: Co-expression correlation matches or exceeds attention

Pearson correlation of gene expression across the 500 cells achieves AUROC = 0.658
(95% CI: [0.623, 0.693]), which overlaps with the attention-based AUROC confidence
interval. A paired permutation test yields p = 0.34 (n = 4950 gene pairs), indicating
no statistically significant difference.

| Method          | AUROC | 95% CI          | p vs. random |
|-----------------|-------|-----------------|--------------|
| Attention (L4)  | 0.672 | [0.638, 0.706]  | < 0.001      |
| Correlation     | 0.658 | [0.623, 0.693]  | < 0.001      |
| Mean expression | 0.621 | [0.585, 0.657]  | < 0.001      |
| Random          | 0.500 | [0.472, 0.528]  | --           |

### Finding 3: Attention signal is largely explained by co-expression

After residualizing attention weights against pairwise expression correlation, the
residual attention AUROC drops to 0.531 (95% CI: [0.497, 0.565]), a reduction of
0.141 (21.0% of the original signal). Conversely, residualizing correlation against
attention yields AUROC = 0.642 (reduction of 0.016, 2.4% of original signal).

This asymmetry indicates that attention weights primarily capture co-expression
structure, with minimal independent regulatory information.

### Finding 4: Head-level analysis shows specialization but not for regulation

Per-head AUROC analysis across all 24 heads (6 layers x 4 heads):
- Range: 0.487 - 0.634
- Mean: 0.552 (SD: 0.038)
- Top head: L4_H2 (AUROC = 0.634)
- Bottom head: L1_H3 (AUROC = 0.487)

No individual head significantly outperforms whole-layer aggregation, suggesting
that regulatory signal (to the extent it exists) is distributed rather than
concentrated in specific heads.

## Negative Controls

1. **Degree-preserving null:** Curveball algorithm (1000 permutations) of the
   ground-truth network preserving in- and out-degree. Attention AUROC on null
   networks: mean 0.504 (SD: 0.012). The real AUROC of 0.672 exceeds all 1000
   null instances (empirical p < 0.001).

2. **Shuffled attention null:** Row-permuted attention matrix destroys gene-gene
   structure while preserving marginal distributions. Shuffled AUROC: 0.501
   (SD: 0.009). Confirms the signal is in the pairwise structure.

## Limitations

- Synthetic data may not reflect the complexity of real gene regulation.
- The ground-truth network is fully known, which inflates apparent AUROC relative
  to real-world incomplete networks.
- Only 100 genes; scaling behavior is unknown.
- Single training run; stochastic variation not assessed.

## Evidence Pointers

- Attention extraction: `outputs/attention_scores_layer_all.npy`
- AUROC computation: `outputs/auroc_per_layer.json`
- Residualization analysis: `outputs/residualization_results.json`
- Per-head analysis: `outputs/per_head_auroc.json`
- Null model results: `outputs/null_model_results.json`
