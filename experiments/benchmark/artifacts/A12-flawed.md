# Geneformer gene-embedding similarity and STRING physical protein–protein interactions

## Question

Proteins that physically interact tend to act in the same processes, and single-cell foundation models learn gene representations from millions of transcriptomes. We asked whether the cosine similarity of Geneformer's contextual gene embeddings reflects physical protein–protein interactions (PPIs) catalogued in STRING, which interacting pairs are closest in embedding space, and whether highly similar pairs absent from STRING are credible candidate interactions.

## Data and model

We used Tabula Sapiens (10x 3' data). After quality control (at least 500 detected genes, less than 15% mitochondrial reads, doublet removal) we subsampled 2,000 cells from each of 15 tissues (30,000 cells), stratified by annotated cell type within tissue.

The model was Geneformer V2-104M (12 layers, 12 attention heads per layer, 768-dimensional hidden states), used frozen. Cells were rank-value encoded: expression was normalized by total counts and by each gene's non-zero median across the pretraining corpus, genes were ordered by normalized expression, and inputs were truncated at 4,096 tokens.

The reference was the STRING v12 human physical-interaction subnetwork at combined score ≥ 700.

## Methods

**Gene embeddings.** Because Geneformer is pretrained with next-token prediction, learning to predict each gene from the genes ranked above it, the representation of a gene is formed at the position that precedes it; we therefore took the layer-11 hidden state at position i − 1 as the embedding of the gene at position i. A gene's embedding was the mean of these states over all cells containing the gene, restricted to the 8,214 genes present in at least 100 cells. Layer 11 (the penultimate layer) was fixed before analysis. Embeddings were mean-centred across genes before computing pairwise cosine similarities, to remove the shared component that otherwise makes all cosines positive.

**Reference comparison.** All 33.7 million gene pairs were scored, of which 51,890 were STRING physical pairs (base rate 0.154%). We compared cosine similarity between STRING pairs and all other pairs with a two-sided Mann–Whitney test. To evaluate ranking at the top, we computed the precision of the 10,000 most similar pairs and the genome-wide area under the precision–recall curve (AUPRC). Significance of top-10,000 precision was assessed against 1,000 degree-preserving rewirings of the STRING network (edge swaps that keep each gene's number of partners fixed).

**Candidate interactions.** Pairs among the top 10,000 with no STRING interaction at any confidence score were treated as candidate novel interactions. For the 500 most similar candidates we performed in-silico deletion in Geneformer: in up to 300 cells expressing both genes, we removed the token of gene A, re-embedded the cell and measured the change in gene B's embedding (1 − cosine between original and perturbed states). The shift was compared with the median shift produced by deleting each of 20 random genes matched to gene A on input rank.

## Results

Across all scored pairs, STRING physical pairs had a mean cosine similarity of 0.019 (SD 0.141), compared with 0.012 (SD 0.139) for other pairs (Mann–Whitney p = 1e-29), showing that physical interaction partners are systematically closer in Geneformer's embedding space.

The signal was concentrated at the top of the ranking. The 10,000 most similar pairs contained 1,318 STRING physical pairs, a precision of 13.2%, 86-fold above the 0.154% base rate; the highest precision among the 1,000 degree-preserving rewirings was 2.9% (empirical p < 0.001). Genome-wide AUPRC was 0.0093, 6.0 times the base rate. Most top-ranked STRING pairs fell within a few large, stable complexes (Table 1).

Table 1. STRING physical pairs among the 10,000 most similar gene pairs, grouped by complex.

| Complex | STRING pairs in top 10,000 | Median cosine |
|---|---|---|
| Cytosolic ribosome | 488 | 0.71 |
| Respiratory chain complex I | 131 | 0.64 |
| Proteasome | 97 | 0.61 |
| CCT/TRiC chaperonin | 24 | 0.58 |
| MCM helicase | 14 | 0.55 |
| Other | 564 | 0.51 |

Of the remaining 8,682 top pairs, 2,146 had no STRING interaction at any score; 58% of these involved at least one mitochondrial ribosomal or respiratory-chain gene. In the in-silico deletion test, deleting gene A shifted gene B's embedding more than the median matched random deletion for 381 of the 500 candidates (76%; binomial p = 5e-33 against 50%), which provides independent confirmation that these high-similarity pairs are genuine interactions missing from STRING.

## Interpretation

Geneformer's embedding geometry tracks physical interactions, most strongly at the top of the similarity ranking, where subunits of large stable complexes dominate. The degree-preserving null shows that this top-of-list enrichment is not explained by highly connected genes pairing with many partners. Because the reference was restricted to STRING's physical subnetwork, which excludes co-expression evidence, the enrichment shows that embedding similarity reflects protein–protein interaction itself rather than transcriptional co-variation of the interacting genes.

The genome-wide AUPRC of six times the base rate is far below the top-of-list enrichment, so embedding similarity is useful mainly for its highest-ranked pairs rather than as a genome-wide interaction score. Within that range, the candidate pairs provide a list of predicted interactions for experimental testing, for example by co-immunoprecipitation or proximity labelling, with priority for pairs that link uncharacterized genes to known complexes.

## Limitations

- Tabula Sapiens profiles healthy adult tissues from a small number of donors, so interactions specific to development or disease states are unlikely to be represented.
- Embeddings were averaged over tissues and cell types, which can dilute interactions that occur in only one context.
- Only one model size and one layer were examined.
- STRING's physical subnetwork includes interactions transferred from other organisms, so some reference pairs may not hold in human cells.
