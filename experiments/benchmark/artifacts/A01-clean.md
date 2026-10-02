# Attention heads in Geneformer V2-104M and TRRUST TF–target edges in human PBMCs

## Question

Do individual attention heads of a pretrained single-cell foundation model assign higher attention to known transcription factor (TF)–target pairs than to other gene pairs, and does any such signal remain once co-expression and expression level are accounted for? We tested the attention heads of Geneformer V2-104M against TRRUST in human peripheral blood mononuclear cells (PBMCs).

## Data and model

We used a public 10x Chromium v3 PBMC dataset from four healthy donors. After standard quality control (200–4,000 detected genes, <15% mitochondrial reads, doublet removal with scDblFinder) we retained 31,862 cells and randomly subsampled 8,000 cells (2,000 per donor). Cells were assigned to five major populations (CD4 T, CD8 T, NK, B, monocytes) by marker-based clustering.

The model was Geneformer V2-104M (12 layers, 12 attention heads per layer, 768-dimensional hidden states), pretrained with a masked-gene objective and used without fine-tuning. Cells were tokenized with the Geneformer V2 tokenizer: genes are ranked by expression after normalization by total counts and by the gene's non-zero median across the pretraining corpus, and the ranked gene tokens form the input sequence. The 4,096-token context covered all detected genes in every retained cell.

The reference was TRRUST v2 (human), restricted to the 2,214 genes detected in at least 10% of cells, which left 118 TFs with at least one TRRUST target inside this universe.

## Methods

For each cell, layer and head we extracted the attention matrix A = softmax(QK^T/√d). Each row of A sums to one over keys, and A is generally asymmetric (A_ij ≠ A_ji) because queries and keys use separate projections and the softmax normalizes each query row separately. We scored a directed edge TF→target by the attention from the target token (query) to the TF token (key), i.e. the TF's weight in the update of the target's representation; this orientation was fixed in advance, with the reverse analysed as a sensitivity check. Each head's pair score was the mean attention across cells in which both genes were present; pairs co-present in fewer than 50 cells were dropped (0.6%), and the same pair set was used for every score.

The candidate set comprised all directed pairs from the 118 TFs to every other gene in the universe that passed the co-presence filter (259,567 of 261,134 pairs), of which 1,036 were TRRUST edges (base rate 0.40%). For each head we computed AUPRC (reported as a ratio to the base rate), AUROC and fold enrichment among the top 1% of scores. Significance used 10,000 degree-preserving rewirings of TRRUST (edge swaps preserving each TF's out-degree and each target's in-degree); empirical p-values on AUPRC were adjusted across the 144 heads with the Benjamini–Hochberg procedure, and heads with q < 0.05 were called enriched. 95% confidence intervals came from 1,000 bootstrap resamples of TFs together with their edges.

Attention between gene tokens can track co-expression and, under rank-value encoding, expression level, so we built two baselines from the same cells: within-population co-expression (Spearman correlation of log-normalized expression in each of the five populations, averaged) and an expression-level score (mean rank position of the two genes in the tokenized sequences). To test for information beyond these, we regressed each head's pair scores on co-expression, both genes' mean rank positions and both genes' detection rates, and repeated the rewiring test, with the same correction, on the residuals. To check robustness to the choice of TFs, we split them at random into two halves and computed each head's AUPRC ratio in each. A randomly initialized Geneformer V2-104M served as an architecture and tokenization control.

## Results

Across heads, TRRUST recovery was modest on average (median AUROC 0.53, IQR 0.51–0.56; median AUPRC 1.2 times base rate). The randomly initialized model gave a median AUROC of 0.50, with no head above 1.3 times base rate.

On raw attention, 11 of 144 heads were enriched at q < 0.05, eight of them in layers 7–10. The strongest, L9H4, reached an AUPRC of 2.9 times base rate (95% CI 2.4–3.5; AUROC 0.63) and a 3.2-fold enrichment in its top 1% of scores. As the maximum of 144 heads its point estimate is optimistic, but head rankings were reproducible across the two TF halves (Spearman ρ = 0.58 across heads), and L9H4 ranked in the top five in both.

The baselines were competitive: co-expression alone came close to the best head (Table 1). Attention in the 11 enriched heads correlated with co-expression across pairs (Spearman ρ = 0.38–0.56). After residualization, three heads remained enriched, with residual AUPRC ratios of 1.3–1.5. In the reverse orientation, six heads were enriched on raw scores and one on residuals.

| Score | Raw AUPRC ratio (95% CI) | Raw q | Residual AUPRC ratio (95% CI) | Residual q |
|---|---|---|---|---|
| L9H4 | 2.9 (2.4–3.5) | 0.004 | 1.5 (1.2–1.9) | 0.014 |
| L8H11 | 2.5 (2.0–3.0) | 0.004 | 1.4 (1.1–1.7) | 0.029 |
| L10H2 | 2.3 (1.9–2.8) | 0.004 | 1.3 (1.1–1.6) | 0.043 |
| Co-expression | 2.7 (2.2–3.2) | – | – | – |
| Expression level | 1.5 (1.2–1.8) | – | – | – |

L9H4's top raw TRRUST edges (CIITA→HLA-DRA, RFX5→HLA-DRB1, SPI1→CSF1R) were among the most co-expressed pairs and fell in rank after residualization; top residual edges included STAT1→IRF1 and TBX21→IFNG.

## Interpretation

Most of the TRRUST signal in Geneformer attention is shared with co-expression: a within-population correlation comes close to the strongest head, and after adjustment only three heads retain an enrichment of 1.3–1.5 times base rate. The randomly initialized model showed no comparable enrichment. Three heads in layers 8–10 additionally show a small association with curated TF–target pairs not accounted for by co-expression, expression level or detection rate. Stronger enrichment in the target-as-query orientation is compatible with these heads weighting regulator tokens when target representations are updated, but this is an association; testing whether the heads contribute to masked-gene predictions requires interventions such as head ablation or activation patching. The practical implication is modest: at roughly 1.5 times a 0.40% base rate, even the best head gives low precision, so these heads are candidates for mechanistic follow-up rather than a standalone GRN method.

## Limitations

- Four donors from a single dataset and tissue; results may differ in other tissues or in disease.
- TRRUST is incomplete, so some high-scoring pairs absent from it may be genuine interactions.
- The residualization is linear, so non-linear dependence on co-expression could leave shared signal in the residuals.
- Only one model size was examined.
