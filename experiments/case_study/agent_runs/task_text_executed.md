RESEARCH QUESTION
Does Geneformer V2-104M attention encode TF->target regulation beyond co-expression, expression-rank proximity and hub (degree) structure?

Answer this question with a quantitative analysis of the precomputed analysis kit described
below. The model itself is not available; everything the analysis needs is in the kit.

KIT DIRECTORY (read-only)
KIT = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/automation/mi-workbench/experiments_data/case_study_kit"

HOW THE KIT WAS BUILT
- Model: Geneformer V2-104M (Hugging Face ctheodoris/Geneformer, snapshot
  fcd26c45fc30fba1989e586bdc46bc366dda8655), BertForMaskedLM, 12 layers x 12 heads, eager
  attention, eval mode (no dropout).
- Cells: 1000 cells from a Tabula Sapiens immune subset, restricted to the 10x 3' v3 assay (a
  single UMI chemistry), sampled stratified by cell_type with a cap of 32 cells per type (types with
  fewer cells contribute all of them; leftover slots go to randomly chosen larger types; seed
  20261001). Counts are the raw integer UMI counts.
- Tokenisation (Geneformer V2 rank-value encoding): per cell, counts / n_counts x 10000 (n_counts =
  total raw counts over all 60,606 genes), divided by the gene's Geneformer non-zero median; genes
  in the token dictionary with a non-zero value are sorted in descending order (stable sort) and
  truncated to 4094 genes; <cls> is prepended and <eos> appended (at most 4096 tokens).
- Gene rank r = token position in the sequence (1 = most highly ranked gene; <cls> is position 0).
- Gene set G (1322 genes): the 1200 genes most frequently present in the 1000 tokenised sequences
  (ties broken by mean normalised expression, then Ensembl ID), plus every TRRUST TF present in at
  least 30% of cells; ordered by presence frequency (descending). Index i in every matrix = row i
  of genes.tsv (0-based gene_index).
- Attention: for every cell and layer, the attention tensor (heads x tokens x tokens) was sliced to
  the rows and columns of the G genes present in that cell; per-head and head-averaged sums were
  accumulated over cells and divided by the number of cells in which both genes are co-present.
  "Co-present in cell c" means that both genes appear in cell c's (truncated) token sequence.
- Symbol -> Ensembl mapping for TRRUST and DoRothEA used the Geneformer gene-name dictionary
  first, then a unique match on the dataset's feature names.

FILES (every matrix is indexed by gene_index from genes.tsv)
- genes.tsv (1322 rows): gene_index, ensembl_id, symbol, token_id, is_tf (gene is a TRRUST
  source), is_dorothea_tf, selection (top_frequency or trrust_tf_ge30pct), n_cells_present,
  presence_frac, mean_norm_expr (mean counts/n_counts x 1e4 over all cells, zeros included),
  mean_log1p_norm_expr, mean_median_scaled_expr, mean_rank_when_present (mean token position over
  the cells where the gene is present), geneformer_median, trrust_out_degree_in_G,
  trrust_in_degree_in_G (non-self TRRUST edges with both ends in G).
- cells.tsv (1000 rows): cell_index, obs_name, h5ad_row, cell_type, donor_id, assay, tissue,
  n_counts_raw_allgenes, obs_total_counts, n_tokenizable_nonzero (before truncation),
  seq_len_with_special, truncated, n_G_present.
- tokenized_cells.npz (ragged arrays): input_ids (all token sequences concatenated, special tokens
  included) with offsets (cell c = input_ids[offsets[c]:offsets[c+1]]); g_index, g_position and
  g_offsets (for cell c, the G genes present and their token positions:
  g_index[g_offsets[c]:g_offsets[c+1]] and g_position[g_offsets[c]:g_offsets[c+1]]).
- attention_layer_mean.npy, float32 [12, 1322, 1322]: A[l, i, j] = mean attention weight FROM
  query gene i TO key gene j in layer l (l = 0..11 is the array index of the encoder layer, first
  layer first), averaged over the 12 heads and over the cells in which genes i and j are
  co-present. Rows are NOT renormalised: each full attention row sums to 1 over all tokens of that
  cell (including <cls>, <eos> and genes outside G), so a row of A restricted to G is not a
  probability distribution over G. NaN where copresence_counts == 0. The diagonal is
  self-attention.
- attention_heads_f32.npy, float32 [12, 12, 1322, 1322] (about 1 GB): A[l, h, i, j], the same
  definition for head h of layer l (no head averaging). Use this file for per-head statistics.
- attention_heads.npy, float16 [12, 12, 1322, 1322]: the same per-head means stored in float16. A
  large fraction of the values fall below the float16 normal range (6.1e-5), where relative
  precision is reduced; prefer the float32 file.
- copresence_counts.npy, int32 [1322, 1322]: C[i, j] = number of cells in which genes i and j are
  co-present (symmetric; the diagonal is the number of cells in which gene i is present).
- rank_distance_mean.npy, float32 [1322, 1322]: mean over co-present cells of |r_i - r_j|
  (token-position distance). NaN where C == 0.
- coexpr_pearson.npy, float32 [1322, 1322]: Pearson correlation across the 1000 cells of
  log1p(counts / n_counts x 1e4) (zeros included).
- coexpr_spearman.npy, float32 [1322, 1322]: Spearman correlation of the same vectors (average
  ranks for ties).
- trrust_edges.tsv (433 rows): unique TRRUST (TF -> target) pairs with both genes in G; columns
  tf, target, tf_ensembl, target_ensembl, tf_index, target_index, mode (';'-joined unique modes:
  Activation, Repression, Unknown), pmids, n_records, is_self (autoregulation).
- dorothea_abc_edges.tsv (2118 rows): unique DoRothEA (TF -> target) pairs with confidence A, B or
  C and both genes in G (best confidence kept); columns tf, target, tf_ensembl, target_ensembl,
  tf_index, target_index, confidence, is_self.
- attention_per_cell_log.tsv (1000 rows): cell_index, seq_len, n_G_present, max_rowsum_dev (a
  row-sum sanity check of the attention tensor), sec.
- prepare_summary.json: sampling strata, symbol-mapping statistics, gene-set construction counts.
- kit_manifest.json: provenance, md5 sums, package versions, sanity checks and runtimes of the kit
  build.
- README.md: the kit's own description (the information above).
- _checkpoint/: raw accumulators used while the kit was built; not needed.

ORIENTATION
TRRUST and DoRothEA edges are directed (TF -> target) and A is directed (query -> key). A
TF -> target pair (t, g) can be read as A[l, t, g] (TF as query), A[l, g, t] (target as query) or
a symmetrised combination. State which reading(s) you use and why.

WHAT THE ANALYSIS MUST DELIVER
1. Define the attention edge score(s), the candidate pair universe (which TF-gene pairs count as
   negatives) and the evaluation metric(s), and justify these choices.
2. Baselines and null models that address each confounder named in the question: co-expression
   (coexpr_*.npy), expression-rank proximity (rank_distance_mean.npy; co-presence and expression
   level are related nuisance variables) and hub / degree structure (TF out-degree, target
   in-degree). Test whether attention carries TF->target information beyond these confounders, not
   only whether it beats random pairs.
3. Effect sizes with uncertainty (for example bootstrap confidence intervals, stating the
   resampling unit), not p-values alone.
4. Multiple-testing control across the layers, heads and score variants you test (state the family
   and the procedure).
5. Null and negative results reported with the same care as positive ones, and a verdict on the
   research question that separates direct evidence, inference and hypothesis.

COMPUTING ENVIRONMENT AND REPORTING RULE
- Your analysis code is executed by the orchestrator in a sandbox: no network access; it can read
  only the kit directory and write only to its current working directory; numpy, scipy, pandas,
  scikit-learn and statsmodels are available; BLAS/OpenMP use a single thread; each execution has
  a wall-clock limit of 900 s (CPU-time limit 1800 s).
  np.load(path, mmap_mode="r") reads single heads of attention_heads_f32.npy without loading the
  whole array.
- Only numbers produced by your executed code may be reported. Every number in the write-up must be
  printed by the code or written to results.json. Do not report numbers from memory, from the
  literature or from estimation, and do not report numbers from a run that failed.
