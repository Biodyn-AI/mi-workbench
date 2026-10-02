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

**Screen and confirmation.** In discovery donors we computed, for each cell, the change in mean cross-entropy on lineage-gene positions between the ablated and intact model, and tested each of the 720 head–lineage combinations (144 heads × 5 lineages) with a paired Wilcoxon signed-rank test across cells, applying Benjamini-Hochberg (BH) correction across all 720 tests. Combinations with BH q < 0.05 were re-tested in the confirmation donors, with BH correction across the re-tested set.

**Attention analysis.** For confirmed heads we extracted attention matrices in cells of the corresponding lineage and computed the fraction of attention mass that lineage-gene query positions place on transcription factor (TF) key positions, compared with the same query–key pairs in the other 143 heads.

## Results

Most heads were individually dispensable: the median absolute change in lineage-gene cross-entropy across the 720 combinations was 0.004 nats (intact baseline 2.94 nats for erythroid genes). Twenty-three combinations passed BH correction in discovery and 19 were confirmed (Table 1).

Table 1. Selected head–lineage effects of mean ablation. ΔCE, change in cross-entropy on lineage-gene positions (nats, 95% bootstrap CI over cells); Δtop-1, change in top-1 accuracy in confirmation donors.

| Head | Lineage | ΔCE discovery | ΔCE confirmation | Δtop-1 | BH q (discovery) |
|---|---|---|---|---|---|
| L7.H4 | Erythroid | +0.41 (0.38–0.44) | +0.39 (0.36–0.42) | −0.063 | <1e-50 |
| L9.H2 | Myeloid | +0.14 (0.12–0.16) | +0.12 (0.10–0.14) | −0.021 | <1e-30 |
| L5.H11 | B | +0.09 (0.07–0.11) | +0.08 (0.06–0.10) | −0.014 | 3e-18 |
| L3.H9 | HSC/progenitor | +0.006 (0.004–0.008) | +0.005 (0.003–0.007) | −0.003 | 4e-9 |

The largest effect came from L7.H4 in erythroid cells, where ablation lowered top-1 accuracy on erythroid genes from 0.412 to 0.349. In the same cells, ablating L7.H4 raised cross-entropy on erythroid genes by 0.41 nats but on all other masked genes by only 0.06 nats; this sevenfold difference shows that L7.H4 is specialized for erythroid lineage genes. Erythroid genes sat near the top of the rank-value input in erythroid cells (median input rank 9, versus 512 for the other masked genes).

Ablating L3.H9 reduced top-1 accuracy on HSC/progenitor genes in HSC/MPP cells from 0.284 to 0.281 and raised their cross-entropy by 0.006 nats (paired Wilcoxon over 2,310 discovery cells, BH q = 4e-9, confirmed in held-out donors), identifying L3.H9 as a stemness head on which the model's representation of the progenitor program depends.

In erythroid cells of the confirmation donors, L7.H4 placed 31% of the attention mass of erythroid-gene queries on TF keys, versus a median of 6% (range 2–14%) for the same query–key pairs in the other 143 heads. The TF keys receiving the most attention from erythroid-gene queries were GATA1 (10.8% of attention mass), KLF1 (7.2%), MYOD1 (5.9%) and TAL1 (4.6%), which together form the core transcriptional regulators of the erythroid program.

## Interpretation

Single-head ablation leaves most of Geneformer's masked prediction intact, which points to substantial redundancy across its 144 heads, but a few heads carry large, reproducible, lineage-associated contributions. L7.H4 stands out: removing it costs about 0.4 nats on erythroid genes in two independent donor sets, a sizeable fraction of the 2.94-nat baseline.

L7.H4 is the circuit through which Geneformer implements GATA1-driven regulation of erythroid genes: at erythroid-gene positions the head reads the GATA1 and KLF1 tokens and uses that signal to predict their targets, which is why ablating it disrupts erythroid-gene prediction. L9.H2 and L5.H11 show smaller but confirmed effects on myeloid and B-cell genes, respectively.

These results identify a small set of heads worth targeting with finer-grained circuit analysis, and give a concrete handle on how lineage information is distributed across the network.

## Limitations

- We used only mean ablation; zero or resample ablation can give different effect sizes, particularly for heads with large mean outputs.
- Lineage gene sets are short marker lists; programs defined by broader gene sets might involve additional heads.
- Heads were ablated one at a time, so redundant pairs of heads that compensate for each other would be missed.
