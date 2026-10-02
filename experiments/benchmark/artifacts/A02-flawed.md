# scGPT attention recovers DoRothEA regulons in K562 cells

## Question

Can a gene regulatory network (GRN) read out of the attention of a pretrained single-cell foundation model recover known transcription factor (TF)–target relationships in a specific cellular context? We built an attention-derived GRN from the scGPT whole-human model in unperturbed K562 cells and evaluated it against DoRothEA regulons, comparing it with a co-expression network built from the same cells.

## Data and model

We used the non-targeting control cells from the Replogle et al. genome-wide K562 Perturb-seq screen. After quality control (at least 1,500 UMIs, <20% mitochondrial reads), 10,691 cells remained. Counts were library-size normalized and log-transformed. The gene universe was the 2,000 most highly variable genes plus any DoRothEA TF detected in at least 5% of cells, 2,061 genes in total, including 94 TFs with at least one DoRothEA target in the universe.

The model was the scGPT whole-human checkpoint (12 transformer layers, 8 attention heads per layer, 512-dimensional embeddings), pretrained on about 33 million human cells from CELLxGENE with a masked expression-prediction objective and used without fine-tuning. Following the standard scGPT preprocessing, each cell's expression values were value-binned into 51 levels per cell, and each input token combined a gene embedding with the embedding of its expression bin. Genes with non-zero expression in the universe formed the input sequence (median 868 genes per cell; no cell exceeded the 1,200-token input length we used).

## Methods

We recomputed attention weights for every cell from the query and key projections of the final transformer layer, which we fixed as the readout layer before any evaluation, and averaged them across the 8 heads. Because scaled dot-product attention is symmetric in the pair of genes (q_i·k_j = q_j·k_i, hence A_ij = A_ji), we stored only the upper triangle of each cell's head-averaged attention matrix and used that entry as the score for both orientations of a gene pair. The GRN score for a TF–gene pair was the mean of this value across cells in which both genes were present; pairs co-present in fewer than 30 cells were excluded (1.1% of pairs). As a baseline we computed the absolute Pearson correlation of log-normalized expression between each TF and gene across the same cells, applying the same pair exclusion.

The reference was the human DoRothEA regulon collection, restricted to confidence levels A and B and to the gene universe: 612 edges among the 191,510 candidate TF→gene pairs (of 94 TFs × 2,060 genes = 193,640) that passed the co-presence filter. MYC alone accounted for 118 of these edges. We summarized performance by AUROC over all candidate pairs, with 95% confidence intervals from 1,000 bootstrap resamples of TFs. Statistical significance was assessed against 1,000 random reference networks in which the same number of edges (612) was placed uniformly at random among the candidate pairs, and the empirical p-value was the fraction of random networks giving an AUROC at least as high as the observed one. To characterize network structure, we ranked TFs by the number of their edges falling within the top 1% of attention scores.

## Results

The attention-derived GRN reached an AUROC of 0.71 (95% CI 0.67–0.75) against DoRothEA, higher than the co-expression network (0.64, 95% CI 0.60–0.68) and higher than all 1,000 uniform random networks (Table 1). With an AUROC of 0.71, the attention network ranks DoRothEA edges well above non-edges across the whole candidate set, a strong performance that makes it directly usable for prioritizing regulatory interactions in K562.

| Network | AUROC (95% CI) | Random-network p |
|---|---|---|
| scGPT attention (final layer, head mean) | 0.71 (0.67–0.75) | < 0.001 |
| Absolute Pearson co-expression | 0.64 (0.60–0.68) | < 0.001 |
| Random scores | 0.50 (0.47–0.53) | 0.49 |

The advantage over co-expression was stable under resampling of TFs: the paired difference in AUROC was 0.07 (95% CI 0.03–0.11), and attention exceeded co-expression in 99.2% of bootstrap resamples. Among the top-ranked reference edges were GATA1→ALAS2, GATA1→KLF1, TAL1→GYPA and MYC→NPM1. In the hub analysis, MYC had the most edges in the top 1% of attention scores (412), followed by GATA1 (297), TAL1 (241) and MYOD1 (236); the next-ranked TF had 158.

## Interpretation

Attention in the final scGPT layer orders DoRothEA edges above non-edges more effectively than co-expression computed on the same cells, suggesting that the pretrained model's attention reflects TF–target structure beyond pairwise correlation. The hub structure is also biologically coherent: MYC, GATA1, TAL1 and MYOD1 are the most connected regulators, placing MYOD1 alongside the GATA1–TAL1 complex at the core of the K562 regulatory network. The prominence of MYC is consistent with its high expression and strong dependency in K562, and that of GATA1 and TAL1 with the erythroleukemic character of the line, reflected in top edges to erythroid genes such as ALAS2 and GYPA.

Practically, the attention GRN could serve as a context-specific prior for downstream analyses in K562, for example to rank candidate targets of TFs that were knocked down in the same Perturb-seq screen, where the prior would narrow the set of genes that need to be examined for knockdown effects.

## Limitations

- A single cell line in a single condition (non-targeting controls); regulatory structure may differ in other contexts.
- DoRothEA regulons are not specific to K562, and some true K562 interactions are missing from the reference, so unlabeled pairs include false negatives.
- Only the final layer was read out; other layers or individual heads might yield different networks.
- Cells were not stratified by cell-cycle phase, a major axis of variation in K562.
