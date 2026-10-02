# Activation patching localizes an interferon-response circuit in scGPT

## Question

When scGPT predicts the expression of interferon-stimulated genes (ISGs) in IFN-β-treated immune cells, which internal components carry the stimulation information to that prediction? We used activation patching to test whether a small set of heads and MLP blocks accounts for most of the difference between the model's predictions for stimulated and control cells.

## Data and model

We used the IFN-β PBMC dataset of Kang et al., in which PBMCs from each of eight donors were left untreated or stimulated with IFN-β for 6 h. After QC (≥500 detected genes, <10% mitochondrial reads, predicted doublets removed), 12,946 control and 12,712 stimulated cells remained across seven annotated immune cell types.

The model was the scGPT whole-human checkpoint (12 transformer layers, 8 attention heads per layer, 512-dimensional embeddings), used frozen. scGPT represents a cell as gene tokens, each paired with a value-binned expression level (51 bins), and can predict the values of masked genes from the rest of the cell.

## Methods

**Readout.** So that clean and corrupted runs differ only in expression values, every cell was encoded with one 1,200-gene panel: 1,176 highly variable genes plus 24 ISGs fixed a priori from the Hallmark interferon-alpha response set (ISG15, IFI27, IFIT2, IFIT3, MX1, OAS1, RSAD2, IFI44L and others). The 24 ISG values were masked, and the readout R is the model's mean predicted expression bin across them. STAT1, IRF7 and IRF9 were kept as unmasked inputs.

**Pairs and splits.** We subsampled 9,800 stimulated cells and paired each with a distinct control cell of the same donor and cell type. Pairs from all eight donors were pooled and randomly assigned 70/30 to a discovery set (6,860 pairs) and an evaluation set (2,940 pairs).

**Patching.** For each of the 96 heads and 12 MLP blocks, we ran the control cell while replacing that component's output with its output on the stimulated partner, and computed normalized recovery (R_patched − R_control) / (R_stim − R_control). On the discovery set, components were ranked by mean recovery, and the circuit was defined as the smallest top-ranked set whose joint patching recovered at least 80% of the effect. Necessity was tested by mean-ablating the circuit in stimulated runs (outputs replaced by their mean over control cells). As a reference we patched 500 random component sets of the same size.

**Route analysis.** To find which token-to-token routes inside circuit heads matter, we zeroed selected attention weights and renormalized. Because scaled dot-product attention is symmetric, A_ij = A_ji, the weight a head places from token i onto token j equals the reverse, so we ablated each unordered pair among the 40 most-attended tokens once (780 pairs per head) and report routes as undirected gene pairs.

**Probe check.** To confirm that the circuit carries the stimulation state, we trained an L2-regularized logistic regression on the concatenated, position-averaged outputs of the circuit components, scored by 5-fold cross-validation on the discovery cells, and compared it with the same probe on random component sets of equal size.

**Resting-blood check.** In an unstimulated single-donor PBMC dataset (10x Genomics PBMC 10k; 10,390 cells after the same QC), cells were labelled ISG-high if their measured Hallmark interferon-alpha module score exceeded the 99th percentile of Kang control cells (83 cells, 0.8%). The circuit score is the projection of the summed circuit outputs onto the discovery-set mean stimulated-minus-control difference; its threshold was fixed in advance at the midpoint between the discovery-set stimulated and control means.

## Results

Unpatched, the mean readout was 29.6 bins for stimulated and 11.8 for control cells. Most single components recovered less than 2% of this gap. The top nine were the smallest set reaching the 80% criterion (the top eight reached 0.77) and form the circuit.

| Component | Single-component recovery (discovery) | 95% bootstrap CI |
|---|---|---|
| L7H1 | 0.21 | 0.19–0.23 |
| L8H3 | 0.17 | 0.15–0.19 |
| MLP7 | 0.14 | 0.12–0.16 |
| MLP9 | 0.12 | 0.10–0.14 |
| L10H4 | 0.11 | 0.09–0.13 |
| L6H5 | 0.09 | 0.08–0.10 |
| L7H6 | 0.08 | 0.07–0.09 |
| L9H0 | 0.07 | 0.06–0.08 |
| L4H2 | 0.06 | 0.05–0.07 |

Jointly patching the nine components recovered 0.86 of the gap on discovery pairs and 0.84 (95% CI 0.81–0.87) on evaluation pairs, against 0.06 ± 0.04 (maximum 0.19) for random nine-component sets. The joint effect is smaller than the sum of single effects (1.05), as expected when components interact. Mean-ablating the circuit in stimulated runs removed 71% (68–74%) of the stimulated–control gap, compared with 5% for random sets. On evaluation pairs, recovery was highest in CD14 monocytes (0.89) and lowest in B cells (0.74).

The highest-effect routes in L7H1 and L8H3 were STAT1–IFIT3, IRF7–ISG15 and IRF9–MX1; ablating the ten highest-effect pairs in L7H1 removed 58% of that head's single-component recovery.

The probe on circuit outputs separated stimulated from control cells with 98.1% ± 0.4% cross-validated accuracy (classes balanced by construction), compared with 71.3% ± 6.2% for random component sets.

In resting blood, the pre-set circuit-score threshold classified cells as ISG-high or not with 99.3% accuracy.

## Interpretation

Nine of 108 components are sufficient to transfer most of the stimulation signal to the ISG readout and are needed for most of it: patching and ablation agree, and random sets of the same size do little. The circuit spans layers 4–10, including two MLP blocks, consistent with stimulation information from unmasked inputs such as STAT1 and IRF7 being combined in the middle of the network. Because faithfulness on evaluation pairs (0.84) matches discovery (0.86), the circuit generalizes beyond the cells used to find it rather than being tied to particular individuals.

The route analysis narrows the heads' contribution to a few gene pairs involving STAT1, IRF7 and IRF9, although which member of each pair is read from cannot be determined from these routes. These are statements about the model's computation, not about regulation. The near-perfect probe accuracy confirms that the circuit components carry the stimulation state, and the resting-blood result suggests that the same components track interferon activity outside the stimulation experiment.

## Limitations

- The fixed 1,200-gene panel departs from scGPT's usual input of each cell's expressed genes; the circuit under native tokenization could differ.
- Mean ablation can push activations off-distribution; resample ablation was not tried.
- Only one stimulus and time point (IFN-β, 6 h) was analysed; IFN-γ or other cytokines may use different components.
- The readout is masked-value prediction; components supporting cell embeddings used downstream may differ.
