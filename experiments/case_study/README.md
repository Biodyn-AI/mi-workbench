# Case study: Geneformer attention vs TF->target regulation

Question: does Geneformer V2-104M attention encode TF->target regulation beyond co-expression,
expression-rank proximity and hub degree?

Two author-side scripts, both deterministic (seed 20261001):

| step | script | environment | output |
|---|---|---|---|
| 1. build the kit | `build_kit.py` | `/Users/ihorkendiukhov/anaconda3/envs/subproject02-evalbias-rev/bin/python` (torch + transformers, MPS) | `../../experiments_data/case_study_kit/` (large, not in git; see the kit `README.md` and `kit_manifest.json`) |
| 2. validate | `independent_validation.py` | `/Users/ihorkendiukhov/anaconda3/bin/python` (numpy/scipy/sklearn/pandas/statsmodels/matplotlib) | `results/independent_validation.json`, `results/fig_case_validation.{png,pdf}`, `results/SUMMARY.md` |

## Commands (run from `automation/mi-workbench/`)

```bash
# 1. kit (about 40 min on an M2 Pro when the GPU is shared; resumable, checkpoints every 100 cells)
mkdir -p "/Volumes/Crucial X6/tmp_miw"
export TMPDIR="/Volumes/Crucial X6/tmp_miw"; export OMP_NUM_THREADS=4
nohup /Users/ihorkendiukhov/anaconda3/envs/subproject02-evalbias-rev/bin/python -W ignore \
    experiments/case_study/build_kit.py >> experiments_data/case_study_kit/build_kit.log 2>&1 &

# 2. validation (reads only the kit)
OMP_NUM_THREADS=4 /Users/ihorkendiukhov/anaconda3/bin/python experiments/case_study/independent_validation.py
```

`build_kit.py --stage {prepare,attention,finalize}` runs one stage; `--force` rebuilds from scratch;
`--max-cells N` stops the attention stage early (smoke test). `independent_validation.py --kit DIR --out DIR`
points at another kit; `--min-copresence/--n-perm/--n-boot` exist only for smoke tests on small kits.

## Design decisions worth knowing
- Counts come from `raw/X` (integer UMIs). `X` in the source h5ad is log-normalised.
- Cells are restricted to `10x 3' v3` (17,113 of 20,000 cells) so that co-expression is not driven by mixing
  Smart-seq2 read counts with 10x UMIs; then stratified by `cell_type` (water-filling cap, 41 types).
- The encoder is run layer by layer (verified identical to the full forward pass, max |dA| = 0), so only one
  layer's attention tensor is in memory at a time.
- Per-head means are stored twice: float16 (`attention_heads.npy`, as specified) and float32
  (`attention_heads_f32.npy`) because roughly a quarter of per-head means fall below the float16 normal range;
  the validation uses the float32 file.
- Primary edge score (pre-specified): symmetrised attention (A[TF,g] + A[g,TF]) / 2; the two directed readings
  are reported alongside.
