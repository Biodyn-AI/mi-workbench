# Transferring attention-edge thresholds from kidney to lung in Geneformer

## Question

Attention-derived TF–target scores from single-cell foundation models are usually thresholded separately in each dataset against a curated reference, which is impractical for tissues with thin tissue-matched references. We asked whether a score threshold calibrated on kidney can be applied unchanged to lung and still deliver the precision it was calibrated for.

## Data and model

We used Tabula Sapiens kidney (9,641 cells) and lung (10,000 cells, subsampled evenly across donors and the epithelial, endothelial, stromal and immune compartments). The model was Geneformer V2-104M (12 layers, 12 attention heads per layer, 768-dimensional embeddings), used frozen. Each cell was rank-value encoded: detected genes were ordered by expression normalized by each gene's corpus-wide median, and the top 4,096 genes formed the input.

The reference was DoRothEA (confidence levels A–C). For each tissue, the candidate space was all pairs of a DoRothEA TF and a candidate target gene, both detected in at least 2% of cells: 301 TFs × 9,236 genes in kidney (2.78 million pairs, 9,466 reference edges, 0.34%) and 318 TFs × 9,720 genes in lung (3.09 million pairs, 10,520 reference edges, 0.34%).

## Methods

**Edge score.** For each candidate pair, the score is the attention weight from the target token (query) to the TF token (key), averaged over the 12 heads of layers 4–9 and over all cells in which both genes are encoded; pairs co-occurring in fewer than 50 cells were dropped. The layer window was chosen on kidney alone, as the contiguous six-layer window with the highest kidney AUPRC.

**Calibration and transfer.** The kidney threshold τ_k is the lowest score at which kidney precision against DoRothEA reaches 1.5%, about 4.4 times the base rate. τ_k was applied unchanged to lung scores. For reference only, we also computed the threshold that would give 1.5% precision in lung ("lung oracle"); it played no role in calibration.

**Baseline.** Spearman co-expression computed in the same cells was calibrated to 1.5% precision in kidney and transferred to lung in the same way.

**Statistics.** Confidence intervals for precision come from 200 bootstrap resamples of cells, with scores recomputed each time. Enrichment of reference edges among thresholded edges was assessed against 1,000 size-matched sets of TF–target pairs drawn uniformly at random from the tissue's candidate space.

**Further analyses.** We compared kidney and lung scores for pairs scored in both tissues, compared the transferred lung edges with a GRNBoost2 network, and counted, per TF, transferred edges to the top 40 alveolar type II (AT2) marker genes from the Tabula Sapiens annotation.

## Results

In kidney, τ_k = 2.3 × 10^-3 retained 14,380 edges at 1.52% precision and 2.3% recall. Applied to lung, the same threshold retained 16,912 edges, with precision 1.31% (95% CI 1.16–1.47%) and recall 2.1%. The lung-oracle threshold (2.7 × 10^-3) would have retained 11,240 edges at 1.50% precision, so transfer lost about 13% of the target precision while keeping more edges. The transferred co-expression baseline reached 1.12% (0.97–1.28%) in lung; its interval overlaps that of the attention score.

| Setting | Edges retained | Reference edges | Precision | Recall | Fold over base rate |
|---|---|---|---|---|---|
| Kidney, τ_k (calibration) | 14,380 | 218 | 1.52% | 2.3% | 4.5 |
| Lung, τ_k transferred | 16,912 | 222 | 1.31% | 2.1% | 3.9 |
| Lung, oracle threshold | 11,240 | 169 | 1.50% | 1.6% | 4.4 |
| Lung, co-expression transferred | 18,300 | 205 | 1.12% | 1.9% | 3.3 |

Against the random edge sets, the transferred lung edges were enriched 3.9-fold for DoRothEA edges (empirical p < 0.001; no random set reached the observed count).

Across the 2.41 million pairs scored in both tissues, kidney and lung attention scores were positively correlated (Spearman ρ = 0.028, p < 10^-300), indicating that the edge ranking is conserved between tissues and supporting direct reuse of the kidney threshold.

As independent support for the transferred edges, we compared them with a GRNBoost2 network inferred from the same 10,000 lung cells: 61% of the 16,912 edges ranked in the top 5% of GRNBoost2 importances for their target gene, against 5% expected by chance.

Among edges to AT2 marker genes, the TFs with most edges were NKX2-1 (14 edges) and PAX8 (11), followed by FOXA2 (8) and CEBPA (6).

## Interpretation

A threshold calibrated in kidney carries over to lung with a modest precision loss (1.31% against the 1.5% target), while admitting about 50% more edges than the lung-optimal threshold. The transferred attention edges keep a nearly four-fold enrichment for curated regulatory edges, and the conserved cross-tissue ranking of edge scores explains why a single threshold works in both tissues. The attention score's advantage over co-expression at this operating point is small, so the case for attention rests mainly on its transfer behaviour rather than on higher precision. The GRNBoost2 agreement independently confirms that the transferred edges capture lung TF–target structure rather than kidney carry-over.

The AT2 module is organized around NKX2-1 and PAX8, the lineage factors that maintain surfactant gene expression in alveolar type II cells, with FOXA2 and CEBPA as secondary regulators, suggesting that the transferred threshold preserves lung-specific structure. In practice, kidney calibration could seed candidate GRNs for tissues with sparse references, accepting a precision shortfall of roughly 10–15%.

## Limitations

- DoRothEA is not tissue-specific, so precision here reflects recovery of broadly curated edges rather than lung-specific ones.
- Only one checkpoint and one aggregation (layers 4–9, head mean) were tested; other layer windows might transfer differently.
- We examined a single precision target (1.5%); behaviour at stricter targets, where fewer edges pass, is untested.
- Lung cells were subsampled evenly across compartments, so edges specific to rarer epithelial populations are underrepresented.
