# A donor-age axis in Geneformer cell embeddings

## Question

Single-cell foundation models are pretrained without donor metadata, yet donor age shifts immune-cell composition and gene expression. We asked whether donor age can be decoded linearly from Geneformer embeddings of peripheral blood mononuclear cells (PBMCs) of donors not seen during fitting, how that signal is distributed across cell types, and which genes co-vary with the resulting aging axis.

## Data and model

We analysed a population-scale PBMC cohort profiled with 10x Genomics 3' chemistry: 981 donors aged 19–97 years (median 64), of whom 31 were aged 90 or older. After QC (≥400 genes, <12% mitochondrial reads, doublets removed) we subsampled up to 150 cells per donor, giving 142,870 cells. Cells were annotated with the Azimuth PBMC reference into seven types (CD4 T, CD8 T, NK, B, CD14 monocytes, CD16 monocytes, dendritic cells).

The model was Geneformer V2-104M (12 layers, 12 attention heads, 768-dimensional hidden states), used frozen. Each cell was rank-value encoded (genes ordered by median-normalized expression, up to 4,096 tokens), and the cell embedding was the mean of gene-token embeddings from the penultimate layer.

## Methods

**Split.** Donors, not cells, were assigned to five cross-validation folds stratified by age decile, so no donor contributed cells to both training and test data, and every cell received an out-of-fold prediction. The ridge penalty was chosen by an inner donor-grouped cross-validation within each training fold.

**Age regression.** Ridge regression predicted donor age from the 768-dimensional embedding, after regressing sex and sequencing pool out of the embeddings with coefficients estimated on each fold's training donors. We report Pearson r and mean absolute error (MAE) for held-out cells, and for donor-level predictions obtained by averaging each donor's cell predictions. Two baselines used the same folds: ridge on the top 50 principal components of log-normalized expression (PCA fitted on training donors only), and a composition-only model using each donor's fractions of the seven cell types. Separate within-type models used the same donor folds. Model differences are reported as Δr with 95% CIs from a paired bootstrap over donors.

**Aging axis and gene correlates.** In each fold the aging axis is the unit vector of the ridge coefficients fitted on the training donors; held-out cells were projected onto their fold's axis. Within each cell type we computed Spearman correlations between these out-of-fold projections and log-normalized expression of the 12,400 genes detected in at least 1% of cells, controlling the false discovery rate with Benjamini–Hochberg across all 86,800 gene × cell-type tests, and we discuss only genes with |ρ| ≥ 0.2.

**Oldest donors.** A class-weighted L2-regularized logistic regression on donor-mean embeddings classified donors aged ≥90 (31 of 981, prevalence 0.032) against all others, using the same donor folds. We report AUPRC against its chance level (the prevalence), AUROC, balanced accuracy and sensitivity at 90% specificity.

Uncertainty was estimated with 1,000 bootstrap resamples of donors.

## Results

**Age regression.** Held-out cell predictions correlated with donor age at r = 0.49 (95% CI 0.47–0.51), MAE 12.3 years. Averaging per donor gave r = 0.71 (0.68–0.74), MAE 9.2 years. At the donor level, the expression-PC baseline reached r = 0.64 (0.60–0.67), MAE 10.1 years, and the composition-only model r = 0.52 (0.47–0.56), MAE 11.6 years. The embedding's gain over the PC baseline was Δr = 0.07 (0.03–0.11).

| Cell type | Cells | r (cell level) | r (donor level) | MAE, donor level (years) |
|---|---|---|---|---|
| CD8 T | 24,560 | 0.52 | 0.66 | 9.9 |
| CD4 T | 49,030 | 0.44 | 0.61 | 10.4 |
| NK | 15,590 | 0.38 | 0.55 | 11.0 |
| CD14 monocytes | 28,200 | 0.35 | 0.52 | 11.3 |
| B | 14,510 | 0.31 | 0.47 | 11.8 |
| CD16 monocytes | 6,770 | 0.27 | 0.40 | 12.3 |
| Dendritic cells | 4,210 | 0.19 | 0.29 | 13.0 |

**Oldest donors.** AUPRC was 0.19 (0.10–0.31) against a chance level of 0.032, AUROC 0.83 (0.76–0.89) and balanced accuracy 0.72 at a 0.5 cutoff. At 90% specificity, 16 of the 31 oldest donors were detected (sensitivity 0.52), together with 95 false positives among 950 younger donors, a precision of 14%.

**Gene correlates.** In total, 7,208 gene × cell-type pairs passed FDR < 0.05, most with |ρ| < 0.1. In CD8 T cells the strongest positive correlates were GZMH (ρ = 0.33), NKG7 (0.27) and KLRD1 (0.24), and the strongest negative correlates were CCR7 (−0.36), LEF1 (−0.31), SELL (−0.29) and CD28 (−0.26). CDKN2A also correlated positively (ρ = 0.20); its product p16INK4a is a CDK4/6 inhibitor that blocks G1-to-S progression and accumulates in T cells with age, consistent with more cell-cycle-arrested, senescence-like cells. In CD14 monocytes, S100A8 (0.21) and S100A9 (0.20) were the leading positive correlates.

## Interpretation

Donor age is linearly decodable from Geneformer embeddings of donors held out from fitting, at a moderate level: donor-level r = 0.71 with a typical error of about nine years. The gain over an expression-PC baseline is small (Δr = 0.07), and cell-type proportions alone reach r = 0.52, so much of the decodable information likely reflects the known age-related shift in blood composition, and the embedding adds only a modest increment over the PC baseline. Within-type decoding shows that some age information remains after fixing cell type, most in CD8 T cells.

Along the axis, CD8 T cells show lower naive and co-stimulatory gene expression (CCR7, LEF1, SELL, CD28) and higher cytotoxic gene expression (GZMH, NKG7), and monocytes show higher alarmin expression (S100A8, S100A9). These associations match established features of immune aging, but with |ρ| ≤ 0.36 they leave most expression variance unexplained, and they describe what co-varies with the axis, not which inputs the model relies on.

For the oldest donors, the embedding carries signal above chance (AUPRC about six times the prevalence), but most donors flagged at 90% specificity are younger than 90, so it is not a usable individual-level classifier.

## Limitations

- All donors come from a single cohort and geographic population; ancestry-related differences in immune aging are not addressed.
- Only 10x 3' data were analysed; performance on other chemistries, or on fresh versus cryopreserved samples, is unknown.
- Cell-type labels come from reference mapping and may misassign transitional states, which affects the within-type estimates.
- The model was used frozen; fine-tuning might strengthen or reorganize the age signal.
