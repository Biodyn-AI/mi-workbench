# Immune cell identity is linearly decodable from Geneformer V1-10M layer embeddings in Tabula Sapiens

## Question

How much immune cell-type information is linearly accessible in each layer of a small pretrained single-cell foundation model when probes are evaluated on donors they were not trained on, at which depth does it peak, and do the model's embeddings carry more cell-type information than a standard expression representation?

## Data and model

We used the immune compartment of Tabula Sapiens (v1), which profiles cells from 15 donors across multiple tissues, including blood, bone marrow, spleen, lymph node, thymus and lung. We kept the 12 immune cell types with at least 300 annotated cells (consortium annotations) that were present in at least five donors, and capped each type at 8,000 randomly sampled cells, giving 61,420 cells; class sizes ranged from 412 (plasmacytoid dendritic cells) to 8,000.

The model was Geneformer V1-10M (6 transformer layers, 4 attention heads, 256-dimensional hidden states, 2,048-token context), pretrained with masked-gene prediction on Genecorpus-30M (about 30 million human single-cell transcriptomes) and used without fine-tuning. Geneformer uses rank-value encoding: each cell is represented as a sequence of gene tokens ordered by expression after normalization by total counts and by each gene's non-zero median across the pretraining corpus, so expression enters the model only through this ordering and the truncation to the top 2,048 genes, not as values. Cell embeddings were obtained at each of the seven hidden states (layer 0, the input embeddings, and layers 1–6) by mean-pooling over gene tokens. Because averaging largely discards the pairing between each gene and its position, the layer-0 embedding mainly records which genes are among a cell's top-ranked genes, whereas deeper layers can mix rank-position information into the gene representations.

## Methods

We used nested cross-validation grouped by donor. The outer loop had five folds, each holding out three donors; within each outer training set, an inner five-fold donor-grouped cross-validation selected the regularization strength and the layer. For each layer we trained an L2-regularized multinomial logistic regression probe on standardized embeddings. Highly variable gene selection, PCA and standardization were fit on outer-training cells only. Because classes are imbalanced, the primary metric was macro-F1, with balanced accuracy and per-class F1 as secondary metrics.

As an expression baseline we trained the same probe on the first 50 principal components of log-normalized expression of 2,000 highly variable genes. A randomly initialized Geneformer V1-10M served as an architecture and tokenization control. Held-out predictions from all outer folds were pooled, and 95% confidence intervals were obtained by resampling donors (2,000 bootstrap resamples). We prespecified two contrasts: selected layer versus expression baseline, and selected layer versus final layer.

## Results

The inner loop selected layer 4 in four of five outer folds and layer 5 in one. Decodability on held-out donors rose with depth to a plateau over layers 3–5 and declined slightly at layer 6 (Table 1).

| Representation | Macro-F1 (95% CI) | Balanced accuracy |
|---|---|---|
| Layer 0 (input embeddings) | 0.742 (0.713–0.770) | 0.756 |
| Layer 2 | 0.842 (0.816–0.866) | 0.853 |
| Layer 4 | 0.874 (0.851–0.895) | 0.883 |
| Layer 6 | 0.852 (0.828–0.874) | 0.862 |
| Randomly initialized model, layer 4 | 0.683 (0.651–0.714) | 0.698 |
| Expression PCA (50 PCs) | 0.869 (0.845–0.891) | 0.879 |

Layers 1, 3 and 5 reached 0.801, 0.866 and 0.871. The selected-layer probe and the expression baseline performed comparably (macro-F1 0.873 vs 0.869; difference 0.004, 95% CI −0.012 to 0.020). The final layer was lower than the selected-layer probe by 0.021 (95% CI 0.009–0.033). Pretrained embeddings exceeded the randomly initialized model by 0.19 at layer 4. Across outer folds, layer-4 macro-F1 ranged from 0.842 to 0.903, with most of the variation coming from rare classes. Per-class F1 at layer 4 was highest for plasma cells (0.97), mast cells (0.95) and neutrophils (0.94), and lowest for conventional dendritic cells (0.71), CD8+ T cells (0.78), plasmacytoid dendritic cells (0.78) and CD4+ T cells (0.80); most errors were between the two T-cell classes and between conventional dendritic cells and classical monocytes.

## Interpretation

Immune cell identity is linearly decodable from Geneformer V1-10M embeddings on donors the probe never saw, far above a randomly initialized network with the same architecture and tokenization (a difference of 0.19 macro-F1, a large effect). However, the best layer is practically equivalent to a 50-component expression baseline, and the confidence interval for their difference is centred near zero and excludes differences larger than about 0.02, so these data give no evidence that the embeddings contain more cell-type information than standard expression features. The layer profile is a broad plateau over layers 3–5 with a modest decline at layer 6. Because layer-0 mean pooling discards rank-position information, the rise from layer 0 should not be read purely as integration across genes; part of it may reflect deeper layers exposing expression-rank information that the pooled input embedding lacks.

Decodability shows that cell-type information is present and linearly accessible in these layers; it does not show that the model relies on this information for masked-gene prediction. Testing that would require interventions, for example projecting out the probe directions at layer 4 and measuring the change in masked-gene prediction accuracy. The small decline at layer 6 is compatible with the last layer specializing toward the pretraining objective, but this remains a hypothesis. For annotating immune cells from new donors, layers 3–5 perform similarly to an expression baseline, and the choice between them can be made on convenience.

## Limitations

- Tabula Sapiens annotations are coarse for some lineages (for example, T-cell subsets beyond CD4+ and CD8+), so probe accuracy reflects the granularity of the labels.
- Capping large classes at 8,000 cells changes the class distribution relative to the atlas.
- With 15 donors, donor-level confidence intervals are fairly wide, and the donor composition of rare classes differs between folds.
- Mean pooling is only one way to form cell embeddings; other pooling schemes could shift the layer profile.
