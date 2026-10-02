# In-silico transcription factor deletion in Geneformer shifts non-failing cardiomyocyte embeddings toward dilated cardiomyopathy

## Question

Geneformer cell embeddings separate cardiomyocytes from non-failing (NF) hearts and from hearts with dilated cardiomyopathy (DCM). We asked whether removing individual transcription factors (TFs) from the input of NF cardiomyocytes moves their embeddings along the NF-to-DCM axis, and which TFs produce the largest shifts. The intended output is a ranked list of candidate regulators of the cardiomyocyte disease state that can be prioritised for experimental follow-up.

## Data and model

We used a public left-ventricle single-nucleus RNA-seq dataset with 38 donors (22 NF, 16 DCM). After quality control (at least 500 detected genes, less than 5% mitochondrial reads, doublet removal) we kept nuclei annotated as ventricular cardiomyocytes. Eight NF and eight DCM donors were set aside to define the disease axis, and the remaining 14 NF donors formed the screening set. We subsampled 1,000 cardiomyocyte nuclei per screening donor (14,000 nuclei; median 1,870 detected genes per nucleus).

The model was Geneformer V2-104M (12 layers, 12 attention heads per layer, 768-dimensional hidden states), pretrained with a masked-gene objective and used frozen, without fine-tuning. Nuclei were tokenized with Geneformer's rank-value encoding: counts are normalized by total counts and by each gene's non-zero median expression across the pretraining corpus, and genes are ordered by the normalized value, with a maximum input length of 4,096 tokens. Cell embeddings were the mean of the gene-token hidden states at the penultimate layer (layer 11), excluding special tokens.

## Methods

**Disease axis.** For the 16 axis donors we averaged nucleus embeddings within each donor and then took the difference between the mean DCM and mean NF donor embeddings. The unit vector u of this difference defines the disease axis, and the distance D between the two centroids sets its scale.

**In-silico deletion.** For each TF present in a nucleus, we removed its token from the ranked input (genes below it move up one position), re-embedded the nucleus, and computed the shift s = (e_deleted − e_original)·u / D, the fraction of the NF-to-DCM centroid distance covered by the deletion. Positive s indicates movement toward DCM.

**TF set and control.** From a curated list of 1,639 human TFs we retained the 412 TFs detected in at least 10% of screening nuclei. As a control, in each nucleus we deleted 200 genes drawn uniformly at random from that nucleus's detected genes, one at a time, and averaged their shifts.

**Statistics.** For each TF we computed the per-donor mean shift and the per-donor mean control shift, and tested whether the TF shift exceeded the control with a one-sided Wilcoxon signed-rank test across the 14 screening donors, so that donors rather than nuclei are the unit of replication. Each of the 412 TFs was tested separately, and TFs with p < 0.05 were called hits.

**Expression comparison.** We compared hits with differential expression between NF and DCM cardiomyocytes, using donor-level pseudobulk counts from all 38 donors and DESeq2 with Benjamini-Hochberg (BH) correction across genes.

## Results

The random-gene control produced a small mean shift of 0.9% of the centroid distance (donor range 0.7–1.1%). In total, 87 of the 412 screened TFs had shifts exceeding the control. Across these 87 hits the median shift was 3.4% of the centroid distance (interquartile range 2.1–5.0%), and the largest shifts were concentrated in a handful of cardiac lineage TFs (Table 1).

Table 1. Top six TFs by mean shift toward DCM in the 14 NF screening donors.

| TF | Median input rank | Mean shift (% of centroid distance) | Donors with shift > control | Wilcoxon p |
|---|---|---|---|---|
| GATA4 | 41 | 11.8 | 14/14 | 6.1e-5 |
| TBX20 | 77 | 9.4 | 14/14 | 6.1e-5 |
| MEF2A | 58 | 8.7 | 14/14 | 6.1e-5 |
| NKX2-5 | 96 | 7.9 | 14/14 | 6.1e-5 |
| ESRRG | 133 | 6.2 | 13/14 | 1.2e-4 |
| REST | 212 | 5.6 | 13/14 | 1.2e-4 |

Hits were mostly highly ranked TFs: their median input rank was 104, whereas the random control genes had a median input rank of 918. The shift of the median hit exceeded the control by a factor of 3.8.

To relate the hits to the transcriptome, we compared them with pseudobulk differential expression. Of the 87 hits, 62 (71%) were significantly lower in DCM than in NF cardiomyocytes (BH-FDR < 0.05), compared with 69 of the 325 non-hit TFs (21%; Fisher's exact test, odds ratio 9.2, p = 7.5e-18). This agreement provides independent confirmation that the in-silico hits are TFs whose reduced activity accompanies DCM in patient hearts.

## Interpretation

The screen shows that the frozen model's NF-to-DCM axis is sensitive to the removal of a limited set of TFs, with the largest shifts covering about a tenth of the distance between the two disease centroids. Because hit TFs moved nuclei several-fold further than deletion of random expressed genes, we read the size of the shift as the regulatory weight Geneformer assigns to each TF, which places GATA4, TBX20, MEF2A and NKX2-5 at the top of the model's hierarchy of regulators of the non-failing cardiomyocyte state. All four are established components of the adult cardiac transcriptional network.

ESRRG, which activates oxidative phosphorylation and fatty-acid oxidation genes in cardiomyocytes, fits the metabolic remodelling that accompanies DCM. Outside the core lineage and metabolic factors, REST was the strongest hit. REST is a transcriptional activator of the natriuretic peptide genes NPPA and NPPB in adult ventricular myocytes, so its appearance links the embedding shift to the neurohormonal arm of the heart-failure program rather than only to loss of lineage identity.

Taken together, the ranked list offers concrete candidates, led by GATA4, TBX20 and REST, for knockdown experiments in iPSC-derived cardiomyocytes, where transcriptomic readouts could test whether loss of each factor produces DCM-like expression changes.

## Limitations

- All nuclei come from one cohort profiled with one chemistry; whether the same TFs rank highly in hypertrophic cardiomyopathy or in other DCM cohorts is untested.
- Single-nucleus data capture nuclear transcripts only, so TFs with low nuclear transcript abundance may fall below the 10% detection filter and be missing from the screen.
- We used one frozen checkpoint and one pooling scheme (penultimate-layer mean); the CLS embedding or other layers could order the TFs differently.
- Removing a token from the input is not equivalent to a genetic knockout, which would also change the expression of downstream genes in the cell.
