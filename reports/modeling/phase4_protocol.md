# Phase 4 protocol fixed before tuning or validation comparison

Continue the confirmed-discharge scoring policy and original frozen partitions. Phase 4
requires an existing lock: it will not reconstruct or reallocate data. Test bytes are hashed
for integrity only; test records, labels, probabilities and metrics are never loaded.
Assignment keys may be checked for patient isolation. Existing Phase 3 records report only
validation scoring. Historical full-cohort EDA means the test is not fully unseen conceptually.

## Training-only design

Five StratifiedGroupKFold folds, shuffled with seed 42 on canonical encounter ordering.
Verify unique encounter coverage, patient isolation, both classes and absolute prevalence
tolerance 0.01 for every fit/holdout subset. Fit all transformations inside each fold.
The chosen folds are shared across comparisons; no seed search. AP is primary because the
constant-score reference is only about 0.11 and false positives matter for this rare outcome.
Report mean, sample SD, minimum/maximum AP, ROC-AUC, Brier and threshold-0.50 metrics.
Fold SD is descriptive, not a confidence interval. Reusing folds for selection creates
optimism: this is not nested CV. Validation is reserved for the fixed candidates' comparison.

Compare raw utilization, three any-use indicators alone, and raw plus totals/indicators with
fixed baseline settings for both logistic and boosting. Prefer raw when within 0.001 AP of
the highest allowed configuration; otherwise use the highest mean. This simplicity margin
is a development convention, not a clinical utility estimate. Compare raw versus grouped
diagnoses on top of raw utilization as separate training-CV sensitivities. They are NEVER
eligible for tuning/final selection without a scoring-time audit; none exists. Do not add
medication/lab/payer experiments or future appearance-based histories.

Tune unweighted L2 logistic C on [0.001,100] (log scale), maximum 30 trials. L1/elastic-net
are not justified by the small primary design and add solver/convergence complexity.
Tune histogram boosting with learning rate [0.025,0.15] (log), 75/125/150/200/300 iterations,
7/15/31 leaves, minimum leaf 20/30/50/100/200 and L2 0/0.1/1/10/30, maximum 50 trials.
Early stopping stays disabled. Leaf count bounds complexity; max_depth is left unset and
max_bins stays 255 to retain categorical compatibility. Include baseline settings as the
first Optuna trial. Use TPE seed 42, sequential trials, no pruning and two numerical threads.
After at least 15/25 logistic/boosting trials, stop after 10/15 trials without a 0.0002
improvement. Record every completed fold and trial, including durations. Fail on convergence
warnings. One fixed five-fold random forest is a reference; no forest search is planned.
No SMOTE: existing evidence concerns operating thresholds/probability quality, not a
demonstrated need for synthetic observations.

## Calibration and one external development comparison

For each tuned primary family fit uncalibrated, sigmoid and isotonic candidates using only
training. CalibratedClassifierCV uses the explicit patient-disjoint folds and ensemble=False:
out-of-fold responses fit the calibrator, then one base pipeline fits the full training set.
This retains one inference model rather than a five-model ensemble. Hyperparameters were
chosen using training CV, so calibration responses are not independent of that selection;
the external validation check is essential. No validation labels enter training/calibration.
Seal model hashes and chosen hyperparameters BEFORE opening validation. Score that fixed
candidate set once and persist its predictions locally. Re-execution verifies/reuses those
predictions; changed models/policy cannot silently replace the comparison.

Use 1,000 paired patient-cluster bootstrap resamples (seed 42) for AP/Brier differences.
Within a family, retain uncalibrated unless calibration improves Brier by at least 0.0001,
its paired Brier-difference interval is wholly below zero, and AP loss is no more than 0.001.
If both qualify, choose lower Brier. These are prespecified development criteria, not proof
of clinical calibration. Compare both methods even if neither qualifies.

Prefer boosting as primary only if its selected version beats logistic by >=0.005 validation
AP with a paired interval wholly above zero, has no more than 0.0002 Brier disadvantage,
and its training CV mean is at least logistic's with no more than twice its AP SD. Otherwise
prefer logistic's simpler inference and interpretation, retaining boosting as challenger.
Report training/inference time, encoded dimension and model size alongside this rule.
Do not interpret a small AP difference as business benefit. Calibration/model choice on
validation makes this a development result, not an unbiased final performance claim.

## Diagnostics and stopping point

Explore a fixed 0.00–1.00 threshold grid in 0.01 steps. Report encounter metrics and unique
patients with any flagged validation encounter; these are historical aggregate counts,
not a simultaneous outreach caseload. Do not choose a threshold or optimize F1/capacity.
Subgroup diagnostics use the prespecified illustrative threshold 0.10: no/any/3+ prior
inpatient use and recorded race/gender/age. Suppress performance estimates below 200
encounters or 30 positives/negatives; retain counts and missingness. No fairness conclusion
or future-derived single/repeat-patient feature is supported.

Local MLflow SQLite tracks configuration, folds, trial/final metrics, seed, Git revision and
artifact paths. Optuna studies, individual predictions and development models remain ignored.
Keep original Phase 3 artifacts intact. No test evaluation, SHAP, operational threshold,
full fairness analysis, business-capacity/ROI work, API or deployment belongs in this phase.

References: [group CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[calibration](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html),
[Optuna reproducibility](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results),
[local MLflow database](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/).
