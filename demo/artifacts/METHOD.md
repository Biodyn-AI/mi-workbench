# Methodology: Attention-Based GRN Recovery from Synthetic Data

## Data Generation

### Synthetic Regulatory Network
We constructed a synthetic gene regulatory network (GRN) with 100 genes and 150
directed regulatory edges. The network was generated using a preferential attachment
model (Barabasi-Albert, m=3) to produce realistic degree distributions with hub
transcription factors. Edge signs (activating/repressing) were assigned uniformly
at random with probability 0.7/0.3 respectively.

### Synthetic Expression Data
Gene expression data for 500 cells was generated using a stochastic kinetic model:

1. Initialize expression vector x(0) from a log-normal distribution
   (mu=1.0, sigma=0.5).
2. For each simulation step t=1,...,T:
   - Compute regulatory input: r_i = sum_j W_ij * x_j(t-1) for each gene i
   - Update expression: x_i(t) = max(0, x_i(t-1) + alpha * tanh(r_i) + noise)
   - noise ~ N(0, sigma_noise=0.1)
3. Sample 500 cells from the stationary distribution (t > 100).

This process ensures that the expression data contains correlations induced by
the regulatory network, but also includes stochastic variation.

### Rank-Value Encoding
Following the Geneformer encoding scheme, each cell's expression profile was
converted to a rank-ordered gene list. Genes were sorted by expression value in
descending order, and each gene's position in the ranked list was used as its
input token.

## Model Architecture and Training

### Architecture
- 6-layer transformer encoder (BERT-style).
- 4 attention heads per layer.
- Hidden dimension: 256.
- Intermediate dimension: 512.
- Vocabulary: 100 gene tokens + [CLS] + [MASK] + [PAD].

### Pre-training
- Objective: Masked gene prediction (mask 15% of tokens per cell).
- Optimizer: Adam (lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01).
- Batch size: 32.
- Epochs: 50.
- Training loss converged to 2.14 (cross-entropy).

## Attention Extraction

### Procedure
For each of the 500 cells:
1. Forward pass through the trained model with output_attentions=True.
2. Extract attention weight matrices A^{l,h} of shape (N_genes, N_genes) for each
   layer l in {1,...,6} and head h in {1,...,4}.
3. Average across heads to obtain per-layer attention: A^l = (1/4) * sum_h A^{l,h}.
4. Average across cells to obtain mean attention: A^l_mean = (1/500) * sum_c A^l_c.

### Symmetrization
Since the ground-truth network is directed but attention is asymmetric (A_ij != A_ji),
we symmetrize by taking: S_ij = (A_ij + A_ji) / 2. The evaluation is then performed
on undirected edges.

## Evaluation Framework

### AUROC Computation
For each method (attention per layer, correlation, mean expression):
1. Flatten the upper triangle of the score matrix into a vector of 4950 gene pairs.
2. Label each pair as positive (edge in ground truth) or negative.
3. Compute AUROC using scikit-learn's roc_auc_score.

### Baseline Methods
- **Pearson correlation:** Pairwise Pearson correlation across 500 cells, absolute value.
- **Mean expression:** Absolute difference in mean expression between gene pairs.
- **Random:** Uniform random scores U(0,1).

### Statistical Testing
All tests use alpha = 0.05 with Benjamini-Hochberg FDR correction across 8 tests.

- **Bootstrap confidence intervals:** 1000 bootstrap resamples of the 4950 gene pairs,
  computing AUROC on each resample.
- **Paired permutation test:** For comparing two methods on the same gene pairs. Permute
  the method labels 10,000 times, compute delta-AUROC each time, and derive the
  two-sided p-value.
- **Degree-preserving null:** Curveball algorithm applied to the ground-truth network
  1000 times. Each null network preserves the in-degree and out-degree of every node
  while randomizing edge targets.

### Residualization
To decompose the attention signal:
1. For each gene pair (i,j), regress attention_ij on correlation_ij using OLS.
2. Residual = attention_ij - predicted_attention_ij.
3. Compute AUROC on the residuals.
4. Repeat in the reverse direction (regress correlation on attention).

The fraction of signal explained is: 1 - (residual_AUROC - 0.5) / (original_AUROC - 0.5).

## Software and Hardware

- Python 3.11.5
- PyTorch 2.1.2 (MPS backend on Apple M2 Pro)
- NumPy 1.26.4
- scikit-learn 1.3.2
- SciPy 1.11.4
- Random seed: 42 (set for PyTorch, NumPy, and Python random module)
