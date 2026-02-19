# Evaluation Report: Synthetic GRN Recovery from Transformer Attention

**Experiment:** Synthetic GRN recovery
**Date:** 2026-01-15
**Iteration:** 3

## Evaluation Metrics Summary

### Primary Metrics

| Metric                | Attention (L4) | Correlation | Mean Expr | Random |
|-----------------------|-----------------|-------------|-----------|--------|
| AUROC                 | 0.672           | 0.658       | 0.621     | 0.500  |
| AUPRC                 | 0.312           | 0.298       | 0.241     | 0.030  |
| Top-100 Precision     | 0.38            | 0.35        | 0.27      | 0.03   |
| Top-50 Precision      | 0.44            | 0.42        | 0.30      | 0.03   |

Base rate: 150 / 4950 = 0.030 (150 true edges among 4950 possible pairs of 100 genes).

### Statistical Tests

#### Test 1: Attention vs. Random Baseline
- Null hypothesis: AUROC(attention) = 0.5
- Test: One-sample permutation test (10,000 permutations)
- Result: AUROC = 0.672, p < 0.001
- Conclusion: Attention significantly exceeds random baseline.

#### Test 2: Attention vs. Correlation
- Null hypothesis: AUROC(attention) = AUROC(correlation)
- Test: Paired permutation test on per-pair scores (n = 4950)
- Result: delta_AUROC = 0.014, p = 0.34
- Conclusion: No significant difference. Attention does not outperform correlation.

#### Test 3: Residualized Attention
- Null hypothesis: Residual attention AUROC = 0.5
- Test: Permutation test (10,000 permutations)
- Result: Residual AUROC = 0.531, p = 0.048
- Note: Marginal significance; 78.6% of the original attention signal is accounted
  for by co-expression. After BH-FDR correction across 8 tests, this becomes
  non-significant (adjusted p = 0.096).

#### Test 4: Residualized Correlation
- Null hypothesis: Residual correlation AUROC = 0.5
- Test: Permutation test (10,000 permutations)
- Result: Residual AUROC = 0.642, p < 0.001
- Conclusion: Correlation retains most of its signal after removing attention.

#### Test 5: Layer Selection (L4 vs. others)
- Null hypothesis: L4 AUROC = max(other layers AUROC) by chance
- Test: Split-sample validation (250/250 cell split, 100 repetitions)
- Result: L4 selected in 67/100 splits; mean selected-layer AUROC = 0.661
- Conclusion: L4 selection is reasonably stable but not deterministic.

#### Test 6: Per-Head AUROC Distribution
- Null hypothesis: Per-head AUROCs are uniformly distributed
- Test: Kolmogorov-Smirnov test against U(0.45, 0.70)
- Result: D = 0.142, p = 0.71
- Conclusion: Cannot reject uniform distribution; no strong head specialization.

#### Test 7: Degree-Preserving Null
- Null hypothesis: Attention AUROC on real network = attention AUROC on degree-matched null
- Test: Empirical null distribution from 1000 curveball permutations
- Result: Real AUROC = 0.672, null mean = 0.504 (SD = 0.012), Z = 14.0
- Conclusion: Attention signal far exceeds degree-preserving null.

#### Test 8: Expression Mean Baseline
- Null hypothesis: AUROC(attention) = AUROC(mean expression)
- Test: Paired permutation test (n = 4950)
- Result: delta_AUROC = 0.051, p = 0.003 (BH-adjusted p = 0.012)
- Conclusion: Attention significantly outperforms mean expression baseline.

### Multiple Testing Correction

All 8 tests corrected using Benjamini-Hochberg FDR at alpha = 0.05.

| Test | Raw p    | BH-adjusted p | Significant? |
|------|----------|---------------|--------------|
| 1    | < 0.001  | < 0.001       | Yes          |
| 2    | 0.340    | 0.389         | No           |
| 3    | 0.048    | 0.096         | No           |
| 4    | < 0.001  | < 0.001       | Yes          |
| 5    | --       | --            | (descriptive)|
| 6    | 0.710    | 0.710         | No           |
| 7    | < 0.001  | < 0.001       | Yes          |
| 8    | 0.003    | 0.012         | Yes          |

### Interpretation

The attention mechanism learns gene-gene relationships that significantly exceed
random chance and the mean-expression baseline. However, the signal is statistically
indistinguishable from simple pairwise correlation, and residualization analysis
confirms that co-expression explains the vast majority of attention's apparent
regulatory signal. This is consistent with the model learning co-expression
structure during pre-training rather than causal regulatory relationships.

## Reproducibility Checklist

- [x] Random seeds set (seed=42 for all stochastic operations)
- [x] Data split documented (full dataset, no train/test split for this analysis)
- [x] Software versions: Python 3.11, PyTorch 2.1, NumPy 1.26.4, scikit-learn 1.3
- [x] Hardware: Apple M2 Pro, 16GB RAM, MPS backend
- [x] All intermediate results saved to outputs/
- [x] Multiple testing correction applied
