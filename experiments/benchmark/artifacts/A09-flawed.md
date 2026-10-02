# A donor-age axis in Geneformer cell embeddings

## Question

Single-cell foundation models are pretrained without donor metadata, yet donor age shifts immune-cell composition and gene expression. We asked whether donor age can be decoded linearly from Geneformer embeddings of peripheral blood mononuclear cells (PBMCs), how that signal is distributed across cell types, and which genes co-vary with the resulting aging axis.

## Data and model

We analysed a population-scale PBMC cohort profiled with 10x Genomics 3' chemistry: 981 donors aged 19–97 years (median 64), of whom 31 were aged 90 or older. After QC (≥400 genes, <12% mitochondrial reads, doublets removed) we subsampled up to 150 cells per donor, giving 142,870 cells. Cells were annotated with the Azimuth PBMC reference into seven types (CD4 T, CD8 T, NK, B, CD14 monocytes, CD16 monocytes, dendritic cells).

The model was Geneformer V2-104M (12 layers, 12 attention heads, 768-dimensional hidden states), used frozen. Each cell was rank-value encoded (genes ordered by median-normalized expression, up to 4,096 tokens), and the cell embedding was the mean of gene-token embeddings from the penultimate layer.

## Methods

**Split.** Cells were randomly split, stratified by cell type, into a training set (80%, 114,296 cells) and a test set (20%, 28,574 cells); the ridge penalty was chosen by 5-fold cross-validation within the training cells.

**Age regression.** Ridge regression predicted donor age from the 768-dimensional embedding, after regressing sex and sequencing pool out of the embeddings with coefficients estimated on the training cells. We report Pearson r and mean absolute error (MAE) for test cells, and for donor-level predictions obtained by averaging each donor's test-cell predictions. A baseline used the same model on the top 50 principal components of log-normalized expression (PCA fitted on the training cells). Separate models were fitted within each cell type.

**Aging axis and gene correlates.** The aging axis is the unit vector of the pooled ridge coefficients. Within each cell type we computed Spearman correlations between cells' axis projections and log-normalized expression of the 12,400 genes detected in at least 1% of cells, controlling the false discovery rate with Benjamini–Hochberg across all 86,800 gene × cell-type tests.

**Oldest donors.** An L2-regularized logistic regression on the embeddings classified cells from donors aged ≥90 against all other cells, using the same split.

Uncertainty was estimated with 1,000 bootstrap resamples of test cells.

## Results

**Age regression.** Test-cell predictions correlated with donor age at r = 0.84 (95% CI 0.83–0.84), MAE 7.6 years. Averaging per donor gave r = 0.93 (0.92–0.94), MAE 4.9 years; predictions were compressed toward the cohort median, with the largest errors for donors under 30 and over 85. The expression-PC baseline reached r = 0.71 (0.70–0.72), MAE 10.4 years, at the cell level. Within-type models ranged from r = 0.63 in dendritic cells to 0.86 in CD8 T cells:

| Cell type | Test cells | r (cell level) | MAE (years) |
|---|---|---|---|
| CD8 T | 4,912 | 0.86 | 7.1 |
| CD4 T | 9,806 | 0.81 | 8.0 |
| NK | 3,118 | 0.79 | 8.4 |
| CD14 monocytes | 5,640 | 0.77 | 8.9 |
| B | 2,902 | 0.74 | 9.5 |
| CD16 monocytes | 1,354 | 0.72 | 9.8 |
| Dendritic cells | 842 | 0.63 | 11.2 |

**Oldest donors.** The same embeddings identified cells from donors aged ≥90 (4,563 of 142,870 cells, 3.2%) with 97.4% test accuracy.

**Gene correlates.** In total, 9,412 gene × cell-type pairs passed FDR < 0.05. In CD8 T cells the strongest positive correlates were GZMH (ρ = 0.38), NKG7 (0.31) and KLRD1 (0.29), and the strongest negative correlates were CCR7 (−0.41), LEF1 (−0.36), SELL (−0.33) and CD28 (−0.30). CDKN2A (p16INK4a) also correlated positively with the axis in CD8 T cells (ρ = 0.21), consistent with its role in driving G1-to-S entry in the proliferating effector-memory pool that expands with age. In CD14 monocytes, S100A8 (0.24) and IL1B (0.19) were the leading positive correlates.

## Interpretation

Donor age is strongly and linearly decodable from Geneformer embeddings: donor-level predictions track chronological age to within about five years, and the signal is present in every major cell type, strongest in CD8 T cells. The embedding also outperforms the expression-PC baseline by a wide margin (r = 0.84 versus 0.71), indicating that it organizes age-related variation more compactly than the leading axes of expression. It is sensitive enough to single out cells from the oldest donors, which makes it a candidate readout of extreme age.

Taken together, the results show that Geneformer has learned to compute donor age from the immune-aging program: the model uses the loss of naive and co-stimulatory genes (CCR7, LEF1, SELL, CD28) and the gain of cytotoxic genes (GZMH, NKG7) to place cells along its aging axis. The monocyte correlates (S100A8, IL1B) fit the low-grade inflammatory component of immune aging.

## Limitations

- All donors come from a single cohort and geographic population; ancestry-related differences in immune aging are not addressed.
- Only 10x 3' data were analysed; performance on other chemistries, or on fresh versus cryopreserved samples, is unknown.
- Cell-type labels come from reference mapping and may misassign transitional states, which affects the within-type estimates.
- The model was used frozen; fine-tuning might strengthen or reorganize the age signal.
