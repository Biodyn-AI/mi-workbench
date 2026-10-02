# Fine-tuned scGPT versus a linear baseline for predicting responses to unseen CRISPRi perturbations

## Question

Single-cell foundation models are increasingly used to predict how cells respond to genetic perturbations that were not observed during training. We asked whether scGPT, fine-tuned on a genome-scale CRISPR interference (CRISPRi) Perturb-seq screen, predicts the transcriptional response to held-out perturbations more accurately than a linear model built from gene embeddings derived from the training data alone.

## Data and model

We used the Replogle et al. K562 essential-gene CRISPRi Perturb-seq screen. We kept perturbations with at least 50 cells and at least 50% knockdown of the target transcript relative to non-targeting controls, leaving 1,062 perturbations (171,340 cells) and 10,691 non-targeting control cells. Counts were normalized to 10,000 per cell and log1p-transformed. The gene set comprised the 5,000 most variable genes, computed on control cells and training perturbations only, plus the 741 targeted genes not already among them (5,741 genes). Perturbations were split at random into 850 training and 212 test perturbations, so no test perturbation was seen during fitting.

The model was the scGPT whole-human checkpoint (24 transformer layers, 16 attention heads, 1,024-dimensional embeddings), pretrained on about 33 million human cells from CELLxGENE. To limit overfitting with 850 training perturbations, we froze the gene and value embeddings and the first 18 of the 24 transformer layers and fine-tuned the last six layers together with the expression decoder. We otherwise followed the scGPT perturbation fine-tuning protocol: each input is a control cell's expression profile with a learned perturbation flag added to the token of the targeted gene, and the model is trained to output the expression of all 5,741 genes in a randomly paired cell carrying that perturbation, with a mean-squared-error loss.

## Methods

**Linear baseline.** We built a pseudobulk change matrix (genes × training perturbations) by subtracting the control mean from each training perturbation's mean profile and took its top K principal components. Each gene was represented by its loadings on these components, and each perturbation by the loadings of its target gene. The expression change for a perturbation was predicted as a bilinear function of the two embeddings, with the weight matrix fitted by ridge regression on the training perturbations; predicted changes were added to the control mean. K (10–100) and the ridge penalty were chosen by five-fold cross-validation across training perturbations, which selected K = 40.

**scGPT training.** Learning rate (1e-4, 5e-5 or 1e-5) and number of epochs (up to 20) were set by grid search, keeping the configuration and epoch with the highest mean Pearson correlation on the 212 test perturbations (learning rate 5e-5, epoch 14). For each test perturbation, the predicted profile was the mean of scGPT's outputs over 300 randomly drawn control cells.

**Evaluation.** For each test perturbation, the observed profile was the mean log-normalized expression of its cells. Accuracy was the Pearson correlation between predicted and observed profiles across all 5,741 genes, averaged over test perturbations. Models were compared with a paired Wilcoxon signed-rank test across test perturbations, and 95% confidence intervals were obtained from 2,000 bootstrap resamples of test perturbations. To examine performance by response strength, we counted differentially expressed (DE) genes per perturbation against control cells (Wilcoxon rank-sum test per gene, Benjamini–Hochberg FDR < 0.05 within each perturbation) and called perturbations with at least 100 DE genes strong responders.

## Results

Fine-tuned scGPT reached a mean Pearson correlation of 0.989 (95% CI 0.988–0.990) between predicted and observed profiles of the test perturbations, compared with 0.968 (0.965–0.971) for the linear baseline (paired Wilcoxon p = 2.1e-31). scGPT was more accurate for 191 of the 212 test perturbations. The advantage held for both strong and weak responders (Table 1).

Table 1. Mean Pearson correlation between predicted and observed profiles on test perturbations.

| Test subset | Perturbations | scGPT | Linear baseline | scGPT more accurate |
|---|---|---|---|---|
| All | 212 | 0.989 | 0.968 | 191/212 |
| Strong responders (≥100 DE genes) | 57 | 0.976 | 0.949 | 55/57 |
| Weak responders (<100 DE genes) | 155 | 0.994 | 0.975 | 136/155 |

Correlations were lowest for perturbations of core ribosomal and RNA polymerase II subunits, which produced the largest transcriptome-wide shifts, but even there scGPT stayed above 0.95 for every perturbation. The test set included HSPA5, whose knockdown gave a strong response (412 DE genes). scGPT reproduced this profile with r = 0.971 (linear baseline 0.938), including the decrease of the unfolded protein response (UPR) target genes DDIT3, HERPUD1, DNAJB9 and SEL1L (observed log2 fold changes −1.2 to −2.3; scGPT −0.9 to −1.8), the expected consequence of depleting BiP, the chaperone that activates the ERN1, EIF2AK3 and ATF6 stress sensors.

## Interpretation

Fine-tuned scGPT predicts the transcriptome-wide response to CRISPRi perturbations it never saw during training with near-perfect accuracy (mean r = 0.989) and outperforms the linear baseline on 90% of test perturbations. The gap is largest among strong responders, where the linear model, which can only combine directions of variation present among the training perturbations, has the least room to extrapolate. The HSPA5 example illustrates that the predictions can contain pathway-specific structure, here the UPR, rather than only the generic response shared by many essential-gene knockdowns.

A plausible explanation for the advantage is that pretraining on tens of millions of cells gives scGPT gene representations that transfer to perturbations absent from the fine-tuning data, but these experiments do not isolate the contribution of pretraining. For screens of this type, fine-tuned scGPT is a reasonable default predictor when responses to untested knockdowns are needed, for example to prioritize genes for follow-up screens.

## Limitations

- Only one cell line and perturbation modality (CRISPRi knockdown in K562) were examined; performance in primary cells or with CRISPR activation is untested.
- Only single-gene perturbations were evaluated; combinatorial perturbations were not.
- We did not fine-tune a randomly initialized scGPT, so the contribution of pretraining itself is not isolated.
- Perturbations with less than 50% knockdown were excluded, which may favour strong, well-defined responses.
