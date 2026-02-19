# Experiment Plan: Synthetic GRN Recovery from Transformer Attention

**Task:** Analyze attention patterns in a 6-layer transformer on synthetic gene
expression data to test if attention weights recover known regulatory edges.

## Hypotheses

### H1 (Primary): Attention weights recover regulatory structure
- Attention-derived gene-gene scores achieve AUROC > 0.5 against the known
  synthetic regulatory network.
- Pre-registered success threshold: AUROC > 0.60.

### H2 (Secondary): Attention outperforms simple co-expression
- Attention-derived AUROC exceeds Pearson correlation AUROC by a statistically
  significant margin (paired permutation test, alpha = 0.05 after BH correction).

### H3 (Exploratory): Layer-specific regulatory specialization
- At least one layer has AUROC significantly higher than the layer average,
  suggesting regulatory specialization in the transformer architecture.

### H4 (Null expectation): Co-expression explains most of the attention signal
- After residualizing attention against co-expression, the residual AUROC should
  drop substantially (> 50% signal loss), consistent with attention capturing
  co-expression rather than independent regulatory information.

## Experimental Design

### Dataset
- 100 synthetic genes with known regulatory network (150 directed edges).
- 500 synthetic cells generated from a stochastic gene expression model
  parameterized by the regulatory network.
- Expression values rank-value encoded to match Geneformer input format.

### Model
- 6-layer, 4-head transformer with hidden dimension 256.
- Trained on masked gene prediction (15% masking rate) for 50 epochs.
- Training: Adam optimizer, lr=1e-4, batch_size=32.

### Analysis Pipeline

1. **Attention extraction** (est. 5 min)
   - Forward pass all 500 cells through the trained model.
   - Extract attention weights from all 6 layers, all 4 heads.
   - Aggregate: average across cells, then average across heads per layer.

2. **AUROC computation** (est. 2 min)
   - For each layer: compute AUROC of aggregated attention vs. ground-truth edges.
   - For correlation baseline: compute pairwise Pearson correlation, then AUROC.
   - For mean expression baseline: compute mean per gene, absolute difference, AUROC.

3. **Statistical testing** (est. 10 min)
   - Bootstrap CIs for all AUROCs (n=1000).
   - Paired permutation test: attention vs. correlation (n=10000).
   - Degree-preserving null model (curveball, n=1000).

4. **Residualization analysis** (est. 5 min)
   - Regress attention weights on correlation; take residuals.
   - Regress correlation on attention weights; take residuals.
   - Compute AUROC for both sets of residuals.

5. **Per-head analysis** (est. 5 min)
   - Compute AUROC for each of 24 individual heads.
   - Test for head specialization (KS test against uniform).

### Controls

| Control Type | Description | Expected Result |
|-------------|-------------|-----------------|
| Positive    | Ground-truth network self-AUROC | 1.0 |
| Negative (random) | Shuffled attention matrix | AUROC ~ 0.5 |
| Negative (degree) | Degree-preserving null network | AUROC ~ 0.5 on null |
| Trivial baseline | Pairwise Pearson correlation | AUROC comparable to attention |
| Trivial baseline | Mean expression level | AUROC < attention |

### Success Criteria
- H1 supported if: AUROC > 0.60 and p < 0.05 vs. random (BH-corrected).
- H2 supported if: delta_AUROC > 0 and p < 0.05 (BH-corrected).
- H3 supported if: best layer AUROC > mean + 2*SD across layers.
- H4 supported if: residual attention AUROC < 0.55 (> 50% signal loss).

## Timeline and Budget

| Step | Estimated time | Estimated tokens |
|------|---------------|-----------------|
| Attention extraction | 5 min | 5,000 |
| AUROC computation | 2 min | 3,000 |
| Statistical testing | 10 min | 10,000 |
| Residualization | 5 min | 5,000 |
| Per-head analysis | 5 min | 5,000 |
| Report writing | 10 min | 15,000 |
| **Total** | **37 min** | **43,000** |

## Checkpoints and Resume Strategy

Intermediate results saved after each step:
- `outputs/attention_scores_layer_all.npy` (after step 1)
- `outputs/auroc_per_layer.json` (after step 2)
- `outputs/statistical_tests.json` (after step 3)
- `outputs/residualization_results.json` (after step 4)
- `outputs/per_head_auroc.json` (after step 5)

If interrupted, resume from the last completed checkpoint.
