# Planted-flaw critique benchmark — analysis plan

This plan is fixed before any reviewer call is made. The frozen benchmark files are hashed
(`benchmark_manifest.json`, SHA-256 per artifact and ground-truth file); any later change
to an artifact invalidates the hash and requires re-running every condition on it.

## Items

- 12 flawed analysis write-ups (A01–A12), each carrying 4 planted flaws, one from each of
  4 of the 5 families (statistics S1–S3, confounding C1–C3, causal/logical L1–L2,
  technical T1–T2, biological B1–B2). Every one of the 12 flaw types occurs exactly 4 times
  (48 planted flaws).
- 6 clean counterparts (A01, A03, A05, A06, A09, A11) with all four flaws corrected and no
  instance of any taxonomy type.
- Ground truth per item: flaw id, type, family, verbatim quote, description, detection
  criterion.

## Reviewer calls

For each item × model × repeat, eight independent calls are made on the same input:

| Call | Prompt | Tag |
|---|---|---|
| 1–3 | rigour lens (`reviewer/mi_reviewer`) | `rig1`, `rig2`, `rig3` |
| 4–6 | combined-checklist reviewer (`reviewer/mi_reviewer_combined`) | `cmb1`, `cmb2`, `cmb3` |
| 7 | adversarial lens (`adversarial_reviewer/adversarial_reviewer`) | `adv` |
| 8 | biological-plausibility lens (`biological_plausibility/bio_plausibility_checker`) | `bio` |

The reviewer receives the task statement ("Review the following mechanistic-interpretability
analysis of a single-cell foundation model") and the artifact text only. Tools are
disabled; each call is stateless.

Conditions (constructed from the calls; calls are matched for C3–C5):

| Condition | Calls | Description |
|---|---|---|
| C1 single rigour | rig1 | one lens-specialised reviewer |
| C2 single combined | cmb1 | one reviewer given the union of all three checklists |
| C3 rigour ×3 | rig1–3 | self-ensemble of the rigour lens |
| C4 combined ×3 | cmb1–3 | self-ensemble of the combined reviewer |
| C5 panel | rig1, adv, bio | three distinct lenses |

All multi-call conditions are merged with the same merge function and settings, so
conditions differ only in which calls they contain. Token usage (input, output) is recorded
per call and reported per condition.

Models: every available model is run with the same fixed reasoning-effort setting; the
CLI's sampling defaults are used (temperature is not user-settable in the CLIs) and
recorded. Repeats: 3 per item × model.

## Judging

One judge call per (item, model, repeat) receives the item's ground-truth flaw list and the
numbered, shuffled union of all parsed critiques from the eight calls, stripped of lens and
call identity. For each critique the judge returns either the id of the planted flaw it
identifies (strict criterion: the critique must identify the specific problem stated in the
detection criterion) or `none`; `none` critiques are further labelled `substantive`
(a specific, plausible methodological concern not in the ground truth), `generic`
(boilerplate that would apply to any analysis), or `incorrect` (factually wrong or
mischaracterises the analysis). Clean items are judged with an empty flaw list.

Two judges from different model families where available (otherwise two different models).
Agreement: Cohen's κ on critique-level labels (flaw id vs none) and on item-level
flaw-detected indicators per condition. Primary analysis uses judge 1; all headline results
are re-computed with judge 2 and with the intersection (both judges agree) as sensitivity
analyses.

## Outcomes

Primary:
- **Flaw recall** of a condition on an item = fraction of the item's planted flaws
  identified by at least one critique from the condition's calls.

Secondary:
- Recall by family and by flaw type (incl. the attention-symmetry T1 instances).
- Severity-aware recall: fraction of planted flaws identified by a critique of merged
  severity High or Critical.
- Unique contribution of each panel lens (flaws found only by that lens within C5).
- Discrimination: AUROC separating flawed from clean items using (a) the merged letter
  grade (ordinal) and (b) a severity score (Critical 8, High 4, Medium 2, Low 1, Info 0,
  summed over merged critiques); paired flawed–clean comparison on the 6 matched scenarios.
- Critical/High critique count on clean items (false-alarm burden).
- Composition of unmatched critiques (substantive / generic / incorrect).
- Cost: calls, input and output tokens, wall time.

## Statistics

- Condition contrasts (planned): C5 vs C1, C5 vs C3, C5 vs C4, C4 vs C2, C3 vs C1.
- Paired, item-clustered bootstrap (10,000 resamples of items; repeats kept within item)
  for differences in mean recall; two-sided p-values from the bootstrap distribution;
  Holm correction over the five planned contrasts within each model; pooled-over-models
  results reported alongside per-model results.
- Judge agreement: Cohen's κ with bootstrap 95% CI.
- All other analyses are descriptive with 95% bootstrap CIs.

## Merge evaluation (offline)

- Ground truth: within one panel instance (C5 calls), two critiques from different lenses
  are duplicates if the judge mapped both to the same planted flaw; non-duplicates if they
  map to different planted flaws. Critiques mapped to `none` are excluded from pair labels.
- Methods: lexical Jaccard (legacy), normalised TF-IDF cosine, sentence-embedding cosine
  (if a model is available), LLM adjudication.
- Metrics: pairwise precision, recall, F1; threshold selected by leave-one-scenario-out
  cross-validation (maximise F1 on the training scenarios, evaluate on the held-out one).
- Escalation validity: proportion of multi-lens merged groups that correspond to a planted
  flaw vs single-lens groups; recall of planted flaws raised by ≥2 lenses that the method
  escalates.
- Default method and threshold for the platform = best held-out F1, ties broken by cost.

## Stopping-rule calibration (real loops)

- Executor → panel loops on the 12 flawed items, fixed horizon of 5 executor–panel cycles,
  no early stopping. The executor may revise text, correct or withdraw claims, and mark
  analyses that would require new computation as pending; it must not report new numbers.
- After every executor revision a judge labels each planted flaw `resolved`, `unresolved`,
  or `resolved_by_fabrication` (claims new quantitative results that were not computed).
- Recorded per cycle: merged grade, Critical/High count, executor-output similarity.
- Offline replay of stopping rules on the recorded trajectories: current any-of-three;
  each signal alone; all three; no-Critical/High ∧ (grade-stable ∨ similar); the latter
  with the advancement gate; fixed 1–5 cycles.
- Outcomes per rule: premature-stop rate (stops while ≥1 planted flaw unresolved),
  mean unresolved flaws at stop, cycles consumed. The platform default is the rule with the
  lowest premature-stop rate among those that stop before the horizon in ≥50% of runs;
  ties broken by fewer cycles.
