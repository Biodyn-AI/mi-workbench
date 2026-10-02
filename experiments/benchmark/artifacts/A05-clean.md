# Sparse-autoencoder features on the scGPT residual stream align with curated biological pathways

## Question

Sparse autoencoders (SAEs) decompose a model's hidden states into a dictionary of sparsely active features. We asked whether SAE features learned on the residual stream of scGPT correspond to curated biological pathways, what fraction of the dictionary can be annotated this way, and which immune programs are represented.

## Data and model

We used an immune-cell atlas (blood, bone marrow, spleen and lymph node; 22 donors) generated after the data snapshot used for scGPT pretraining. After quality control (at least 400 detected genes, less than 10% mitochondrial reads, doublet removal) we kept 212,000 cells. Cells from 18 donors (174,300 cells) were used to train the SAE, and cells from the remaining 4 donors (37,700 cells) were held out for all feature characterization.

The model was the scGPT whole-human checkpoint (12 transformer layers, 8 attention heads, 512-dimensional hidden states), pretrained on about 33 million human cells and used frozen. scGPT represents a cell as a set of gene tokens, each combined with an embedding of that gene's binned expression value (51 bins); the order of gene tokens carries no expression information. For cells with more than 1,200 detected genes we sampled 1,200 non-zero genes at random, and we collected residual-stream vectors after layer 8 at all gene-token positions, excluding the <cls> token. The layer was fixed in advance as a mid-to-late layer.

## Methods

**SAE training.** We trained a TopK SAE (k = 32 active features per token) with a dictionary of 8,192 features (16× the hidden size) on 40 million residual-stream vectors sampled from the training donors. The dictionary size and k were chosen on a validation split of the training donors by reconstruction quality and dead-feature rate.

**Feature–gene profiles.** On the held-out donors, we computed for each feature its mean activation at each gene's token positions, restricted to the 9,850 genes observed in at least 50 held-out cells, and took the 50 genes with the highest mean activation as the feature's top genes. Features whose gene-level mean activation was strongly correlated with the gene's mean log-normalized expression (Spearman ρ > 0.6) were treated as expression-level features and set aside.

**Enrichment.** Each feature's top-50 set was tested against 1,665 gene sets (1,615 Reactome pathways with 10–500 genes in the background, plus the 50 MSigDB Hallmark sets) with a one-sided hypergeometric test, using the 9,850 observed genes as background. We applied Benjamini-Hochberg (BH) correction jointly over all feature–gene set tests. A feature was called pathway-aligned if at least one gene set reached BH q < 0.05 with a fold enrichment of at least 3 and at least 4 overlapping genes, and it was labelled with its best such gene set.

**Matched null.** Genes that are highly expressed or well studied belong to many curated gene sets, so any list enriched for them will match pathways easily. For every feature we therefore drew 100 null 50-gene sets in which each top gene was replaced by a random background gene from the same decile of annotation degree (number of the 1,665 gene sets containing the gene) and the same decile of mean expression. Null sets, and for reference 10,000 random 50-gene sets drawn uniformly from the background, were scored with the p-value cut-off that the BH procedure produced for the features and the same fold-enrichment and overlap requirements.

## Results

The SAE explained 87% of residual-stream variance on held-out cells, and 254 features (3.1%) never activated, leaving 7,938 live features. Of these, 702 were expression-level features, leaving 7,236 features and 12,047,940 feature–gene set tests.

BH correction corresponded to a nominal cut-off of p ≤ 1.9e-4, under which 5,818 of the 7,236 features (80.4%) were pathway-aligned, compared with 41.7% of degree- and expression-matched null sets and 1.8% of uniformly drawn sets, so SAE features exceed the matched null by 38.7 percentage points (95% bootstrap CI over features 37.2–40.2). The median best-set −log10 p was 8.7 (interquartile range 4.9–15.2) for SAE features, 3.9 (3.0–5.1) for matched null sets and 2.8 (2.3–3.4) for uniform sets. Matched lists thus reproduce about half of the features' advantage in alignment rate over uniform lists, reflecting annotation and expression bias. The most frequent labels were translation and ribosome biogenesis, respiratory electron transport, interferon signalling, cell-cycle checkpoints and antigen presentation.

Table 1. Representative aligned features (held-out donors; all shown associations have BH q < 1e-10).

| Feature | Top genes (5 of 50) | Best gene set | Fold enrichment | p |
|---|---|---|---|---|
| 1187 | RPL3, RPS6, RPL13A, RPS3A, EEF1A1 | Reactome: Eukaryotic translation elongation | 50.3 | 1.4e-36 |
| 5640 | ISG15, IFI6, IFIT3, MX1, OAS1 | Hallmark: Interferon alpha response | 38.6 | 2.8e-26 |
| 771 | NDUFA4, COX7C, UQCRB, NDUFB3, ATP5MC3 | Reactome: Respiratory electron transport | 32.2 | 4.8e-22 |
| 6402 | MKI67, TOP2A, CENPF, NUSAP1, TPX2 | Hallmark: G2M checkpoint | 21.1 | 2.6e-23 |
| 2913 | HLA-DRA, CD74, HLA-DPA1, HLA-DPB1, HLA-DQA1 | Reactome: Interferon gamma signaling | 28.8 | 2.9e-16 |

## Interpretation

Most live features pass the corrected enrichment criterion, but gene lists matched on annotation degree and expression pass it at about half that rate. Alignment beyond this bias, an excess of about 39 percentage points, is therefore substantial but far from universal in the layer-8 residual stream, and judging alignment against uniform random lists would overstate it about twofold. Some features may track programs that Reactome and Hallmark do not capture.

Several features map onto programs with clear immunological meaning. Feature 5640 captures the type I interferon response, feature 6402 proliferating cells, and feature 771 oxidative metabolism. Feature 2913, whose top genes are HLA-DRA, CD74, HLA-DPA1, HLA-DPB1 and HLA-DQA1 and which is most active in B cells, monocytes and conventional dendritic cells, is an MHC class II antigen-presentation feature of professional antigen-presenting cells, which display peptides to CD4+ T cells. Its best match to interferon-gamma signalling is consistent with the induction of class II genes by IFN-γ through CIITA.

These results suggest that SAE features provide a partial pathway-level vocabulary for describing scGPT's internal states. Whether downstream computation in scGPT depends on these features is a separate question that would require ablating or steering them.

## Limitations

- One layer and one dictionary size were analysed; features may split or merge at other dictionary sizes, which would change the per-feature labels.
- Top-50 gene sets are an arbitrary cut-off; features with broad activation profiles may be under-annotated by a fixed-size list.
- Reactome and Hallmark cover immune programs unevenly; features aligned with poorly curated processes would appear unannotated.
