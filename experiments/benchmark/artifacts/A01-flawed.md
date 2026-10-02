# Attention heads in Geneformer V2-104M recover TRRUST TF–target edges in human PBMCs

## Question

Do individual attention heads of a pretrained single-cell foundation model assign preferentially high attention to known transcription factor (TF)–target pairs? We asked whether any of the attention heads of Geneformer V2-104M rank TRRUST TF–target pairs above other gene pairs in human peripheral blood mononuclear cells (PBMCs), and whether such heads concentrate in particular layers.

## Data and model

We used a public 10x Chromium v3 PBMC dataset from four healthy donors. After standard quality control (200–4,000 detected genes, <15% mitochondrial reads, doublet removal with scDblFinder) we retained 31,862 cells and randomly subsampled 8,000 cells (2,000 per donor) for attention extraction. Cells were assigned to five major populations (CD4 T, CD8 T, NK, B, monocytes) by marker-based clustering.

The model was Geneformer V2-104M (12 layers, 12 attention heads per layer, 768-dimensional hidden states), pretrained with a masked-gene objective and used without fine-tuning. Cells were tokenized with the Geneformer V2 tokenizer: each cell's genes are ranked by expression after normalization by total counts and by the gene's non-zero median across the pretraining corpus, and the ranked gene tokens form the input sequence. The 4,096-token context covered all detected genes in every retained cell.

The reference was TRRUST v2 (human). We restricted the candidate gene universe to the 2,214 genes detected in at least 10% of cells, which left 118 TFs with at least one TRRUST target inside the universe.

## Methods

For each cell, layer and head we extracted the full attention matrix, A = softmax(QK^T/√d). Because scaled dot-product attention is symmetric between the two tokens (A_ij = A_ji), a single head cannot distinguish TF-to-target from target-to-TF attention; we therefore took one attention value per gene pair and evaluated TRRUST edges as unordered pairs. For each head, the pair score was the mean attention across cells in which both genes were present in the input sequence; pairs co-present in fewer than 50 cells were dropped (0.6% of pairs), and the same pair set was used for every head.

The candidate set comprised all unordered pairs containing at least one of the 118 TFs that passed the co-presence filter (252,706 of 254,231 pairs), of which 1,021 were TRRUST pairs (base rate 0.40%). For each head we computed AUROC, AUPRC and the fold enrichment of TRRUST pairs among the top 1% of scores. Significance was assessed with 1,000 degree-preserving rewirings of the TRRUST network (edge swaps that keep each gene's number of TRRUST partners fixed), recomputing AUPRC for each rewired reference; the empirical p-value was the fraction of rewired AUPRCs at least as large as the observed one. As a control for architecture and tokenization, we repeated the full pipeline with a randomly initialized Geneformer V2-104M.

## Results

Across heads, TRRUST recovery was modest on average: median AUROC 0.53 (interquartile range 0.51–0.56) and median AUPRC 0.0047, 1.2 times the 0.0040 base rate. The randomly initialized model gave a median AUROC of 0.50, and none of its heads exceeded an AUROC of 0.53 or an AUPRC of 1.3 times the base rate.

Against the degree-preserving null, 19 of the 144 heads showed significant enrichment (permutation p < 0.05), 14 of them in layers 7–10 (Table 1); we refer to these 19 as regulatory heads. The strongest head, L9H4, reached an AUROC of 0.64 and an AUPRC of 0.0121, a 3.0-fold improvement over the base rate, with a 3.4-fold enrichment of TRRUST pairs in its top 1% of scores.

| Head | AUROC | AUPRC (× base rate) | Top-1% fold enrichment | Permutation p |
|---|---|---|---|---|
| L9H4 | 0.64 | 0.0121 (3.0×) | 3.4 | 0.002 |
| L8H11 | 0.62 | 0.0104 (2.6×) | 2.9 | 0.004 |
| L10H2 | 0.61 | 0.0093 (2.3×) | 2.6 | 0.007 |
| L7H6 | 0.60 | 0.0086 (2.2×) | 2.4 | 0.011 |
| L9H9 | 0.59 | 0.0079 (2.0×) | 2.1 | 0.018 |

Recomputed within each donor, L9H4 gave AUPRCs between 0.0109 and 0.0133. The highest-scoring TRRUST pairs in L9H4 were CIITA–HLA-DRA, RFX5–HLA-DRB1, SPI1–CSF1R, STAT1–IRF1 and CEBPB–IL1B. Layers 1–4 contributed a single regulatory head, layers 5–6 none, and the final two layers four.

## Interpretation

Most heads carry little TRRUST information, but a subset in the middle-to-late layers ranks known TF–target pairs well above the base rate, and this pattern is absent from a randomly initialized network with the same architecture and tokenization. Because the null preserves each gene's number of TRRUST partners, the enrichment cannot be attributed to highly connected genes attracting attention in general; it reflects pair-specific regulatory relationships between TFs and their targets. The top pairs correspond to well-characterized immune programs (MHC class II transactivation by CIITA and RFX5, SPI1-dependent myeloid genes, interferon-inducible IRF1).

The layer distribution is also informative. Early layers, which operate closest to the rank-ordered input, show almost no enrichment, whereas layers 7–10 concentrate the signal. These heads are the mechanism through which Geneformer uses TF–target regulatory relationships to predict masked genes, and they constitute the model's internal regulatory circuit. This makes the 19 regulatory heads natural starting points for extracting cell-type-specific GRNs from Geneformer, for example by restricting attention aggregation to these heads and to cells of a single population.

## Limitations

- Four donors from a single dataset and tissue; head-level enrichment may differ in other tissues or in disease.
- TRRUST is incomplete, so some high-scoring pairs that are absent from TRRUST may be genuine interactions counted as false positives.
- Scores were averaged over all cells, which can dilute population-specific attention patterns; per-population analyses were not performed.
- Only one model size was examined.
