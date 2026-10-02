# Single attention heads in Geneformer are needed for masked prediction of hematopoietic lineage genes

## Question

Masked-gene prediction requires Geneformer to infer which gene occupies a hidden position from the rest of the cell's ranked gene list. We asked whether individual attention heads are needed for predicting hematopoietic lineage marker genes, whether their contribution is specific to lineage genes, and what the attention pattern of the most important head looks like.

## Data and model

We used a public human bone-marrow scRNA-seq dataset from 8 healthy donors that is not among the studies in the Geneformer pretraining corpus. After quality control (at least 500 detected genes, less than 10% mitochondrial reads, doublet removal) we retained 41,600 cells annotated as HSC/MPP (4,720), erythroid (9,850), megakaryocyte (1,240), myeloid (11,300), B (5,960) and T/NK (8,530). Donors 1–4 (20,500 cells) formed the discovery set and donors 5–8 (21,100 cells) the confirmation set.

Lineage gene sets were fixed before any model run from published marker lists: erythroid (24 genes, e.g. HBB, HBA1, ALAS2, GYPA, AHSP, CA1), megakaryocyte (18), myeloid (26), B (22) and HSC/progenitor (20; e.g. CD34, CRHBP, HLF, AVP, MECOM).

The model was Geneformer V2-104M (12 layers, 12 heads per layer, 768-dimensional hidden states, 64 dimensions per head), pretrained with masked-gene prediction on rank-value encoded transcriptomes and used frozen.

## Methods

**Masked prediction.** For each cell we masked a random 15% of input tokens, with five masking replicates per cell, and recorded the cross-entropy and top-1 accuracy of the model's prediction of the masked gene's identity. Lineage-gene positions were evaluated in cells of the matching lineage.

**Head ablation.** We mean-ablated one head at a time: the head's output (its attention-weighted sum of value vectors, before the output projection) was replaced at every position by its mean over 2,000 reference cells from the discovery donors.

**Screen and confirmation.** In discovery donors we computed, for each cell, the change in mean cross-entropy on lineage-gene positions between the ablated and intact model, and tested each of the 720 head–lineage combinations (144 heads × 5 lineages) with a paired Wilcoxon signed-rank test across cells, applying Benjamini-Hochberg (BH) correction across all 720 tests. Combinations with BH q < 0.05 were re-tested in the confirmation donors, with BH correction across the re-tested set. Because thousands of cells make very small effects significant, we treated a combination as practically relevant only if its confirmation ΔCE exceeded 0.05 nats (about 2% of baseline loss).

**Expression-matched controls.** Prediction loss depends strongly on a gene's position in the rank-value input and on how often it is detected. For selectivity analyses we therefore paired every lineage gene in each cell with non-lineage control genes from the same cell within ±3 input-rank positions and in the same detection-rate decile, and fitted a per-gene regression of ΔCE on lineage membership with input rank and detection rate as covariates.

**Attention analysis.** For confirmed heads we extracted attention matrices in cells of the corresponding lineage and computed the fraction of attention mass that lineage-gene query positions place on transcription factor (TF) key positions, compared with the same query–key pairs in the other 143 heads.

## Results

Most heads were individually dispensable: the median absolute change in lineage-gene cross-entropy across the 720 combinations was 0.004 nats (intact baseline 2.94 nats for erythroid genes). Twenty-three combinations passed BH correction in discovery and 19 were confirmed, of which only three exceeded the 0.05-nat relevance threshold (Table 1).

Table 1. Selected head–lineage effects of mean ablation. ΔCE, change in cross-entropy on lineage-gene positions (nats, 95% bootstrap CI over cells); Δtop-1, change in top-1 accuracy in confirmation donors.

| Head | Lineage | ΔCE discovery | ΔCE confirmation | Δtop-1 | BH q (discovery) |
|---|---|---|---|---|---|
| L7.H4 | Erythroid | +0.41 (0.38–0.44) | +0.39 (0.36–0.42) | −0.063 | <1e-50 |
| L9.H2 | Myeloid | +0.14 (0.12–0.16) | +0.12 (0.10–0.14) | −0.021 | <1e-30 |
| L5.H11 | B | +0.09 (0.07–0.11) | +0.08 (0.06–0.10) | −0.014 | 3e-18 |
| L3.H9 | HSC/progenitor | +0.006 (0.004–0.008) | +0.005 (0.003–0.007) | −0.003 | 4e-9 |

The largest effect came from L7.H4 in erythroid cells, where ablation lowered top-1 accuracy on erythroid genes from 0.412 to 0.349. Erythroid genes sit near the top of the rank-value input in erythroid cells (median input rank 9), and for rank- and detection-matched control genes in the same cells ablating L7.H4 raised cross-entropy by 0.22 nats (0.20–0.24). The erythroid-specific excess was therefore 0.19 nats (0.15–0.23), and in the per-gene regression the lineage coefficient was 0.17 nats (0.12–0.22) after adjusting for rank and detection. Rank and detection thus explain about half of the raw contrast with all other masked genes (0.41 versus 0.06 nats).

Ablating L3.H9 changed top-1 accuracy on HSC/progenitor genes by −0.003 (95% CI −0.004 to −0.002), about 1% of the 0.284 baseline, and cross-entropy by 0.006 nats. It passed BH correction because of the large number of cells but lies far below the relevance threshold.

In erythroid cells of the confirmation donors, L7.H4 placed 31% of the attention mass of erythroid-gene queries on TF keys, versus a median of 6% (range 2–14%) for the same query–key pairs in the other 143 heads. The TF keys receiving the most attention from erythroid-gene queries were GATA1 (10.8% of attention mass), KLF1 (7.2%), TAL1 (5.9%) and NFE2 (4.6%), which together form the core transcriptional regulators of the erythroid program.

## Interpretation

Single-head ablation leaves most of Geneformer's masked prediction intact, which points to substantial redundancy across its 144 heads, but a few heads carry large, reproducible contributions. L7.H4 stands out: removing it costs about 0.4 nats on erythroid genes in two independent donor sets, and roughly half of that effect remains erythroid-specific after matching on input rank and detection.

The attention pattern of L7.H4 is consistent with the head moving information from erythroid TF tokens to erythroid-gene positions, but attention weights alone do not show that this route carries the signal the head contributes to prediction. Blocking attention from erythroid-gene queries to TF keys would test this directly. L9.H2 and L5.H11 show smaller but confirmed effects on myeloid and B-cell genes, respectively, with excesses over matched controls of 0.05 (0.03–0.07) and 0.03 (0.01–0.05) nats, so their lineage selectivity is modest.

## Limitations

- We used only mean ablation; zero or resample ablation can give different effect sizes, particularly for heads with large mean outputs.
- Lineage gene sets are short marker lists; programs defined by broader gene sets might involve additional heads.
- Heads were ablated one at a time, so redundant pairs of heads that compensate for each other would be missed.
