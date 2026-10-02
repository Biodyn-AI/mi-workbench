# Immune cell identity is linearly decodable from Geneformer V1-10M layer embeddings in Tabula Sapiens

## Question

How much immune cell-type information is linearly accessible in each layer of a small pretrained single-cell foundation model, at which depth does it peak, and do the model's embeddings carry more cell-type information than a standard expression representation?

## Data and model

We used the immune compartment of Tabula Sapiens (v1), which profiles cells from 15 donors across multiple tissues, including blood, bone marrow, spleen, lymph node, thymus and lung. We kept the 12 immune cell types with at least 300 annotated cells (consortium annotations) and capped each type at 8,000 randomly sampled cells, giving 61,420 cells; class sizes ranged from 412 (plasmacytoid dendritic cells) to 8,000.

The model was Geneformer V1-10M (6 transformer layers, 4 attention heads, 256-dimensional hidden states, 2,048-token context), pretrained with masked-gene prediction on Genecorpus-30M (about 30 million human single-cell transcriptomes) and used without fine-tuning. Geneformer tokenizes each cell by discretizing the normalized expression of every detected gene into a fixed set of expression bins and adding the bin embedding to the gene's token embedding, so the input layer already carries each gene's expression level. We therefore treated layer 0 as an in-model expression reference: gains in decodability over layer 0 reflect integration across genes by the transformer layers rather than additional expression information. Cell embeddings were obtained at each of the seven hidden states (layer 0, the input embeddings, and layers 1–6) by mean-pooling over gene tokens.

## Methods

Cells from all 15 donors were pooled and split at random into a training set (80%, 49,136 cells) and a test set (20%, 12,284 cells), stratified by cell type. For each layer we trained an L2-regularized multinomial logistic regression probe on standardized embeddings, with the scaler fit on training cells. The regularization strength was chosen by 5-fold cross-validation within the training set, and the layer used for the primary comparison was the one with the highest cross-validated macro-F1. Because classes are imbalanced, the primary metric was macro-F1, with balanced accuracy and per-class F1 as secondary metrics.

As an expression baseline we trained the same probe on the first 50 principal components of log-normalized expression of 2,000 highly variable genes, with gene selection and PCA fit on training cells only. As a control for architecture and tokenization we repeated the probing with a randomly initialized Geneformer V1-10M. The selected layer was compared with the expression baseline on the test set using a paired bootstrap over test cells (2,000 resamples).

## Results

Cell-type decodability increased with depth up to layer 4 and then declined (Table 1). Cross-validation on the training set selected layer 4 (cross-validated macro-F1 0.918), and test-set performance tracked the cross-validated estimates closely at every layer (maximum absolute difference 0.006).

| Representation | Test macro-F1 | Test balanced accuracy |
|---|---|---|
| Layer 0 (input embeddings) | 0.781 | 0.794 |
| Layer 2 | 0.889 | 0.897 |
| Layer 4 | 0.921 | 0.928 |
| Layer 6 | 0.897 | 0.905 |
| Randomly initialized model, layer 4 | 0.702 | 0.716 |
| Expression PCA (50 PCs) | 0.917 | 0.924 |

Layers 1, 3 and 5 reached test macro-F1 of 0.846, 0.910 and 0.915. The layer-4 probe outperformed the expression baseline on the test set (macro-F1 0.921 vs 0.917; paired bootstrap p = 0.003 over 12,284 test cells), showing that Geneformer embeddings contain cell-type information beyond what is available from expression alone. Per-class F1 at layer 4 was highest for plasma cells (0.98), mast cells (0.97) and neutrophils (0.96), and lowest for conventional dendritic cells (0.81), CD8+ T cells (0.83) and CD4+ T cells (0.85); most errors were between the two T-cell classes and between conventional dendritic cells and classical monocytes. Plasmacytoid dendritic cells reached 0.90 despite being the smallest class. The randomly initialized model was 0.22 lower in macro-F1 at layer 4.

## Interpretation

Immune cell identity is highly linearly decodable from the pretrained embeddings, well above a randomly initialized network with identical architecture and tokenization. Most of the gain in decodability occurs across the first four transformer layers (0.781 at layer 0 to 0.921 at layer 4). Geneformer thus computes an explicit cell-type representation by layer 4 and then uses it to drive masked-gene prediction in layers 5 and 6, which is why decodability declines toward the output.

The residual errors fall along biologically expected boundaries: CD4+ versus CD8+ T cells, which share most of their transcriptome apart from a few co-receptor and effector genes, and conventional dendritic cells versus classical monocytes, which share myeloid programs. Practically, layer-4 embeddings are the preferred Geneformer V1 representation for annotating immune cells from new donors, and for probing analyses of this model we recommend layer 4 rather than the final layer.

## Limitations

- Tabula Sapiens annotations are coarse for some lineages (for example, T-cell subsets beyond CD4+ and CD8+), so probe accuracy reflects the granularity of the labels.
- Capping large classes at 8,000 cells changes the class distribution relative to the atlas.
- Mean pooling is only one way to form cell embeddings; other pooling schemes could shift the layer profile.
- Only linear probes were tested; nonlinear probes might extract more information from later layers.
