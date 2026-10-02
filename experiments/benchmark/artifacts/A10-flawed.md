# Attention hub genes in Geneformer as candidate drivers of ulcerative colitis

## Question

Central network genes are often proposed as disease drivers, but expression-derived networks mostly reflect co-expression. We asked which genes gain or lose attention-hub status in a pretrained single-cell foundation model between inflamed and non-inflamed colonic mucosa of patients with ulcerative colitis (UC), whether attention hubs are enriched for known inflammatory bowel disease (IBD) genes, and which genes should be prioritized as candidate drivers for perturbation experiments.

## Data and model

We used a public 10x Chromium 3' v3 dataset of colonic biopsies from 18 UC patients, each contributing one inflamed and one macroscopically non-inflamed biopsy. The dataset is not part of the model's pretraining corpus. After quality control (300–6,000 detected genes, <25% mitochondrial reads, doublet removal) we retained 158,240 cells, assigned to compartments by marker-based clustering. We analysed the epithelial and immune compartments, subsampling up to 400 cells per sample and compartment (13,870 epithelial and 14,210 immune cells).

The model was Geneformer V2-316M (18 layers, 18 attention heads per layer, 1,152-dimensional hidden states), pretrained with a masked-gene objective on about 100 million human single-cell transcriptomes and used frozen. Cells were rank-value encoded: counts were normalized by total counts and by each gene's non-zero median in the pretraining corpus, and genes were ordered by the normalized value (up to 4,096 tokens, covering all detected genes in 99.6% of cells). Each compartment's gene universe comprised genes detected in at least 5% of its cells (3,904 epithelial, 3,466 immune).

## Methods

**Hub score.** We used layer 12 of 18, fixed in advance. Because a layer's multi-head output is the head-averaged attention matrix applied to the value vectors, the mean of the 18 head-specific matrices is the effective attention pattern of the layer, and we scored hubs on this head-averaged matrix. For gene g in a cell, the raw hub score was the total attention g receives from all other genes, the sum over queries q ≠ g of Ā_qg; because every query row sums to one, these scores average at most 1 across the genes of a cell. Attention received depends strongly on input position, so within each compartment we regressed log raw hub scores on rank position (natural spline, 5 degrees of freedom) and the cell's number of detected genes, and used the residuals as adjusted hub scores. A gene's sample-level score was its mean adjusted score over the sample's cells containing it (at least 10 cells).

**Co-expression comparison.** We computed each gene's co-expression degree (number of partners with Spearman |ρ| > 0.2 across cells of the compartment) and correlated it with the gene's mean adjusted hub score.

**Differential hubs.** For each gene in each compartment we compared sample-level adjusted hub scores between inflamed and non-inflamed biopsies with a two-sided paired Wilcoxon signed-rank test across patients, requiring scores in both biopsies of at least 12 patients. Effect sizes are the median within-patient difference (Δ) in units of the gene's between-sample standard deviation.

**IBD gene enrichment.** The reference was the 412 genes with a curated association to ulcerative colitis or IBD in DisGeNET, 188 of which fell in the union of the two gene universes (5,212 genes). We took the 100 genes with the highest mean adjusted hub score in inflamed biopsies in each compartment and tested enrichment for curated IBD genes with a one-sided hypergeometric test against the union of the two gene universes; as an empirical check, we drew 10,000 random gene sets of the same size uniformly from that union.

## Results

Raw hub scores fell steeply with input rank (Spearman ρ = −0.74), and the adjustment removed this dependence (residual ρ = −0.02). Adjusted hub scores correlated moderately with co-expression degree (ρ = 0.31 in epithelial and 0.27 in immune cells).

Overall, 3,655 epithelial and 3,263 immune genes met the scoring requirement. Of the 6,918 gene–compartment tests, 612 reached p < 0.05 (371 epithelial, 241 immune), and we refer to these as differential hubs and carry them forward as candidates. Their median |Δ| was 0.58 standard deviations (interquartile range 0.44–0.79), and 402 of the 612 gained hub status in inflamed tissue.

Table 1. Largest changes in adjusted hub score (inflamed minus non-inflamed biopsy).

| Gene | Compartment | Δ (SD units) | Paired p | Curated IBD gene |
|---|---|---|---|---|
| DUOX2 | Epithelial | +1.42 | 1.5e-5 | yes |
| LCN2 | Epithelial | +1.31 | 2.3e-5 | yes |
| REG1A | Epithelial | +1.18 | 7.6e-5 | no |
| NKX2-5 | Epithelial | −1.09 | 1.1e-4 | no |
| HNF4A | Epithelial | −0.97 | 3.1e-4 | yes |
| OSM | Immune | +1.36 | 2.3e-5 | yes |
| IL1B | Immune | +1.24 | 3.8e-5 | yes |
| TREM1 | Immune | +1.12 | 1.4e-4 | yes |

CDX2 also lost hub status in the epithelium (Δ = −0.71, p = 0.004).

The top inflamed-tissue hubs comprised 193 unique genes, of which 27 were curated IBD genes, against 7.0 expected (3.9-fold; hypergeometric p = 7.0e-10). None of the 10,000 random gene sets contained more than 19 curated IBD genes.

## Interpretation

Attention hubs overlap only partly with co-expression hubs (ρ ≈ 0.3). The 3.9-fold enrichment of inflamed-tissue hubs for curated IBD genes indicates that the model's attention concentrates on disease-relevant genes rather than on arbitrary expressed genes, which supports using hub changes to rank candidates beyond the known IBD genes.

In the epithelium, gained hubs are dominated by the inflammatory colonocyte program of active UC (DUOX2, LCN2, REG1A). The genes that lost hub status form a coherent lineage module: CDX2, HNF4A and NKX2-5, core transcription factors of the absorptive colonocyte program, decline in centrality in inflamed epithelium, in line with the loss of differentiated colonocytes in active disease. In immune cells, OSM, IL1B and TREM1, all expressed by inflammatory monocytes and macrophages, gained hub status; high mucosal OSM expression has been associated with non-response to anti-TNF therapy.

Hub changes are associations with the inflamed state, not evidence of a causal role. We therefore propose REG1A and NKX2-5, which changed hub status without being curated IBD genes, for CRISPR perturbation in cytokine-exposed colonic organoids, with DUOX2 and HNF4A as positive controls.

## Limitations

- One UC cohort was analysed; Crohn's disease and ileal tissue were not examined.
- Cells were analysed at compartment level; epithelial and immune subtypes were not scored separately.
- Only one layer was analysed; hub rankings may differ at other depths.
- Subsampling under-represents rare epithelial types (tuft and enteroendocrine cells), whose genes rarely met the scoring requirement.
