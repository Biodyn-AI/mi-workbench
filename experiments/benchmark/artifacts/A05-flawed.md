# Sparse-autoencoder features on the scGPT residual stream align with curated biological pathways

## Question

Sparse autoencoders (SAEs) decompose a model's hidden states into a dictionary of sparsely active features. We asked whether SAE features learned on the residual stream of scGPT correspond to curated biological pathways, what fraction of the dictionary can be annotated this way, and which immune programs are represented.

## Data and model

We used an immune-cell atlas (blood, bone marrow, spleen and lymph node; 22 donors) generated after the data snapshot used for scGPT pretraining. After quality control (at least 400 detected genes, less than 10% mitochondrial reads, doublet removal) we kept 212,000 cells. Cells from 18 donors (174,300 cells) were used to train the SAE, and cells from the remaining 4 donors (37,700 cells) were held out for all feature characterization.

The model was the scGPT whole-human checkpoint (12 transformer layers, 8 attention heads, 512-dimensional hidden states), pretrained on about 33 million human cells and used frozen. Because scGPT orders each cell's genes by expression rank before tokenization, we kept the first 512 gene tokens of every cell, which correspond to its 512 most highly expressed genes, and collected residual-stream vectors at those positions after layer 8. The layer was fixed in advance as a mid-to-late layer.

## Methods

**SAE training.** We trained a TopK SAE (k = 32 active features per token) with a dictionary of 8,192 features (16× the hidden size) on 40 million residual-stream vectors sampled from the training donors. The dictionary size and k were chosen on a validation split of the training donors by reconstruction quality and dead-feature rate.

**Feature–gene profiles.** On the held-out donors, we computed for each feature its mean activation at each gene's token positions, restricted to the 9,850 genes observed in at least 50 held-out cells, and took the 50 genes with the highest mean activation as the feature's top genes. Features whose gene-level mean activation was strongly correlated with the gene's mean log-normalized expression (Spearman ρ > 0.6) were treated as expression-level features and set aside.

**Enrichment.** Each feature's top-50 set was tested against 1,665 gene sets (1,615 Reactome pathways with 10–500 genes in the background, plus the 50 MSigDB Hallmark sets) with a one-sided hypergeometric test, using the 9,850 observed genes as background. A feature was called pathway-aligned if its best gene set reached p < 0.01, and it was labelled with that gene set.

**Comparison with chance.** To ask whether alignment exceeds what arbitrary gene lists achieve, we drew 10,000 random 50-gene sets uniformly from the 9,850 background genes and recorded, for each feature and each random set, the −log10 p-value of its best gene set.

## Results

The SAE explained 87% of residual-stream variance on held-out cells, and 254 features (3.1%) never activated, leaving 7,938 live features. Of these, 702 were expression-level features, leaving 7,236 features for the pathway analysis.

In total, 6,981 of the 7,236 features (96.5%) were pathway-aligned. The most frequent labels were translation and ribosome biogenesis, respiratory electron transport, interferon signalling, cell-cycle checkpoints and antigen presentation. Table 1 shows representative features.

Table 1. Representative features and their best-matching gene sets (held-out donors).

| Feature | Top genes (5 of 50) | Best gene set | Fold enrichment | p |
|---|---|---|---|---|
| 1187 | RPL3, RPS6, RPL13A, RPS3A, EEF1A1 | Reactome: Eukaryotic translation elongation | 50.3 | 1.4e-36 |
| 5640 | ISG15, IFI6, IFIT3, MX1, OAS1 | Hallmark: Interferon alpha response | 38.6 | 2.8e-26 |
| 771 | NDUFA4, COX7C, UQCRB, NDUFB3, ATP5MC3 | Reactome: Respiratory electron transport | 32.2 | 4.8e-22 |
| 6402 | MKI67, TOP2A, CENPF, NUSAP1, TPX2 | Hallmark: G2M checkpoint | 21.1 | 2.6e-23 |
| 2913 | HLA-DRA, CD74, HLA-DPA1, HLA-DPB1, HLA-DQA1 | Reactome: Interferon gamma signaling | 28.8 | 2.9e-16 |

Features aligned with pathways far more strongly than random gene lists did: the median best-set −log10 p was 8.7 (interquartile range 4.9–15.2) for SAE features versus 2.8 (2.3–3.4) for the uniformly drawn random sets. This separation shows that the pathway structure of the dictionary is not something that arbitrary gene lists of the same size reproduce.

## Interpretation

Nearly the whole live dictionary aligns with at least one curated gene set, so pathway-level structure is pervasive in the layer-8 residual stream of scGPT. Several features map onto programs with clear immunological meaning. Feature 5640 captures the type I interferon response, feature 6402 proliferating cells, and feature 771 oxidative metabolism.

Feature 2913, whose top genes are HLA-DRA, CD74, HLA-DPA1, HLA-DPB1 and HLA-DQA1 and which is most active in B cells, monocytes and conventional dendritic cells, is an MHC class I antigen-presentation feature that reflects peptide loading for recognition by CD8+ T cells. Its best match to interferon-gamma signalling is consistent with the induction of this program by IFN-γ.

These results suggest that SAE features provide a pathway-level vocabulary for describing scGPT's internal states. Whether downstream computation in scGPT depends on these features is a separate question that would require ablating or steering them.

## Limitations

- One layer and one dictionary size were analysed; features may split or merge at other dictionary sizes, which would change the per-feature labels.
- Top-50 gene sets are an arbitrary cut-off; features with broad activation profiles may be under-annotated by a fixed-size list.
- Reactome and Hallmark cover immune programs unevenly; features aligned with poorly curated processes would appear unannotated.
