# Fine-tuned scGPT versus a linear baseline for predicting responses to unseen CRISPRi perturbations

## Question

Single-cell foundation models are increasingly used to predict how cells respond to genetic perturbations that were not observed during training. We asked whether scGPT, fine-tuned on a genome-scale CRISPR interference (CRISPRi) Perturb-seq screen, predicts the transcriptional response to held-out perturbations more accurately than a linear model built from gene embeddings derived from the training data alone.

## Data and model

We used the Replogle et al. K562 essential-gene CRISPRi Perturb-seq screen. We kept perturbations with at least 50 cells and at least 50% knockdown of the target transcript relative to non-targeting controls, leaving 1,062 perturbations (171,340 cells) and 10,691 non-targeting control cells. Counts were normalized to 10,000 per cell and log1p-transformed. The gene set comprised the 5,000 most variable genes, computed on control cells and training perturbations only, plus the 741 targeted genes not already among them (5,741 genes). Perturbations were split at random into 850 training and 212 test perturbations; the training perturbations were further split into 744 for fitting and 106 for validation.

The model was the scGPT whole-human checkpoint (12 transformer layers, 8 attention heads, 512-dimensional embeddings), pretrained on about 33 million human cells from CELLxGENE. To limit overfitting, we froze the gene and value embeddings and the first 8 of the 12 transformer layers and fine-tuned the last four layers together with the expression decoder. We otherwise followed the scGPT perturbation fine-tuning protocol: each input is a control cell's expression profile with a learned perturbation flag added to the token of the targeted gene, and the model is trained to output the expression of all 5,741 genes in a randomly paired cell carrying that perturbation, with a mean-squared-error loss.

## Methods

**Linear baseline.** We built a pseudobulk change matrix (genes × fitting perturbations) by subtracting the control mean from each perturbation's mean profile and took its top K principal components. Each gene was represented by its loadings on these components, and each perturbation by the loadings of its target gene. The expression change was predicted as a bilinear function of the two embeddings, with the weight matrix fitted by ridge regression.

**Reference baselines.** The no-change baseline predicts the control mean for every perturbation. The training-mean baseline predicts, for every perturbation, the mean change across fitting perturbations, which captures the response shared by many essential-gene knockdowns.

**Model selection.** For scGPT we searched learning rate (1e-4, 5e-5, 1e-5) and number of epochs (up to 20); for the linear baseline, K (10–100) and the ridge penalty. For both, the configuration with the best primary metric on the 106 validation perturbations was kept (scGPT: learning rate 5e-5, epoch 11; linear: K = 40). The 212 test perturbations were evaluated once, after selection. scGPT predictions for a perturbation were averaged over 300 randomly drawn control cells.

**Evaluation.** Because a single knockdown changes few genes, correlations on absolute expression are dominated by between-gene differences in baseline expression. We therefore scored predicted changes relative to the control mean (Δ). The primary metric was the Pearson correlation between predicted and observed Δ over each test perturbation's 20 most strongly changed genes, ranked by observed absolute change against controls. Secondary metrics were Pearson Δ over all genes, the mean squared error (MSE) over the top 20 genes, and the fraction of these 20 genes whose direction of change was predicted correctly. Two comparisons on the primary metric were prespecified, scGPT versus the linear baseline and scGPT versus the training mean, using paired Wilcoxon signed-rank tests across test perturbations with Holm correction; 95% CIs came from 2,000 bootstrap resamples of test perturbations. Strong responders were perturbations with at least 100 differentially expressed genes against controls (Wilcoxon rank-sum test per gene, Benjamini–Hochberg FDR < 0.05 within each perturbation).

## Results

Fine-tuned scGPT and the linear baseline performed similarly, and both exceeded the training-mean baseline by a small margin (Table 1).

Table 1. Accuracy on the 212 test perturbations (primary metric with 95% CI).

| Method | Pearson Δ, top 20 genes | Pearson Δ, all genes | MSE, top 20 genes | Direction correct, top 20 genes |
|---|---|---|---|---|
| scGPT (fine-tuned) | 0.54 (0.50–0.58) | 0.29 | 0.087 | 0.79 |
| Linear baseline | 0.57 (0.53–0.61) | 0.31 | 0.081 | 0.81 |
| Training mean | 0.49 (0.45–0.53) | 0.26 | 0.094 | 0.76 |
| No change | – | – | 0.171 | – |

The scGPT minus linear difference on the primary metric was −0.03 (95% CI −0.06 to 0.00; Holm-adjusted p = 0.09), and the scGPT minus training-mean difference was +0.05 (0.02 to 0.08; Holm-adjusted p = 0.004). Among the 57 strong responders, primary-metric values were 0.61 (scGPT), 0.63 (linear) and 0.52 (training mean); among the 155 weak responders, 0.51, 0.55 and 0.48. For reference, Pearson correlation on absolute expression exceeded 0.98 for every method, including the no-change baseline (0.986), and did not separate them.

The test set included HSPA5, whose knockdown gave a strong response (412 DE genes), including induction of the unfolded protein response (UPR) target genes DDIT3, HERPUD1, DNAJB9 and SEL1L (observed log2 fold changes +1.2 to +2.3). Both models predicted increases for all four genes but underestimated their size (scGPT +0.5 to +1.1; linear +0.6 to +1.3). For 49 of the 57 strong responders, both models likewise predicted a smaller mean absolute change over the top 20 genes than observed.

## Interpretation

On this screen, fine-tuned scGPT did not predict unseen knockdowns more accurately than a linear model fitted to the same training perturbations. The confidence interval (−0.06 to 0.00) is compatible with no difference or a small linear advantage, not with a meaningful scGPT advantage. Both models improve on the training-mean baseline by only 0.05–0.08, so much of what is predictable for an unseen essential-gene knockdown is the generic response shared across knockdowns; perturbation-specific structure is captured only partly.

The HSPA5 case fits this picture. Depleting BiP, which binds and holds the ERN1, EIF2AK3 and ATF6 sensors inactive, activates all three UPR branches; both models predicted the correct direction for the UPR targets but compressed the magnitude. For screens of this type, the linear baseline is the cheaper choice with equal or better accuracy.

## Limitations

- One cell line and modality (CRISPRi in K562) were examined; primary cells and CRISPR activation are untested.
- Combinatorial perturbations were not evaluated.
- We did not fine-tune a randomly initialized scGPT, so the contribution of pretraining itself is not isolated.
- Excluding perturbations with <50% knockdown may favour strong, well-defined responses.
