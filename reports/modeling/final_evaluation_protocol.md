# Frozen final evaluation protocol

This decision record was prepared **before opening the frozen test table**. Its Git commit
must exist before `make final-eval` is permitted. Model/feature/calibration/policy choices
cannot change in response to test results. The source/evaluation code commit is
`5714a2961f1346f7b505f439e6da2e4298b9b41c`; the training commit is `a48c68b10e0339e95c934ea4ab6b1ee7ff00a904`. The freeze commit itself
is resolved and verified from this file's Git history and recorded with the final result.

## Fixed models and data

Primary: **uncalibrated Logistic Regression**, raw utilization configuration, ten original
allowed source predictors, L2/lbfgs, C=0.09988151348099303, seed 42, no class weights.
Challenger: **uncalibrated Histogram Gradient Boosting**, explicitly included for a single
prespecified comparison. Exact parameters, preprocessing, feature names, library versions,
training provenance and both binary SHA-256 values are in
[the model specification](final_model_specification.json).
The actual trained pipelines are reused byte-for-byte; no train+validation refit occurs.
The [final feature policy](final_feature_policy.md) keeps timing-uncertain families excluded.

The original frozen patient split and all hashes remain binding. Test contains 13,549
encounters and 9,757 patients according to the existing allocation record. No test outcomes,
probabilities or errors were used to choose any decision here. Earlier full-cohort EDA did
examine eventual test outcomes before partitioning; this is a held-out model evaluation,
not a fully untouched confirmatory study.

## Frozen operational policy

Prioritize the **top 10% highest-risk eligible discharge encounters** in an available outreach
batch. Capacity is a **portfolio scenario assumption**, not measured hospital staffing.
Rank descending by predicted probability; equal scores use ascending SHA-256 of
`42:<encounter_id>`. The key is only an outcome-independent tie-break, never a predictor or
chronology. Select exactly `floor(0.10 * N)` encounters. Report unique people separately;
the data cannot identify simultaneous caseload or a daily/weekly batch schedule.

The corresponding probability cutoff may differ in test because ranking uses that batch's
unlabeled scores. This is an application of the frozen capacity rule, not post-test threshold
optimization. Do not switch to a validation probability cutoff, optimize F1, use subgroup
thresholds or change capacity after test. The pooled historical evaluation cannot establish
performance in real-time batches with a changing mix of discharges.

Validation: 1,359 targeted encounters / 871 unique historical patients captured 321 of 1,499
readmissions (21.4143% recall, 23.6203% precision, 2.14143x lift). Top 5% has higher precision
but captures only 13.2755%; larger capacities increase coverage at declining precision.
Ten percent is an interpretable limited-outreach demonstration, not an empirically optimal
cost/utility choice. Under this policy 732 readmissions without prior inpatient use were
missed; this limitation does not trigger a model or policy change. Risk prioritization must
not be interpreted as denying standard care to unflagged patients.

## Metrics, uncertainty and diagnostics declared before test

Report AP (non-interpolated), ROC-AUC, Brier, mean risk/prevalence, quantile reliability,
precision/recall curves, selected-policy recall/precision/specificity, confusion counts,
encounters/unique people flagged, lift and gains. Report the fixed 5/10/15/20/25% capacity
grid and ten risk groups as descriptive sensitivity summaries only; 10% stays primary.
Scale encounter-level yields to an illustrative 10,000-discharge scenario. No prevented
readmissions, causal effect, cost savings or scaled unique-person counts are estimated.

Use 1,000 paired patient-cluster bootstrap draws, seed 42, resampling all encounters of
each patient together. Apply the frozen ranking rule within each weighted draw; report
percentile 95% intervals for AP, ROC-AUC, Brier, recall, precision and lift. The intervals
condition on fitted models and exclude retraining/selection and temporal/site uncertainty.

Apply identical validation/test subgroup definitions from `configs/phase5.yaml`: no/any
prior inpatient and emergency use, 0–2/3+ inpatient use, recorded age bands/gender/race,
stay lengths 1–2/3–4/5–7/8–14 days, historical destination/type/source codes, and specialty/
payer missingness. Payer and race are audit-only. Preserve Unknown and counts; suppress
performance below 200 encounters, 100 patients, 30 positives or 30 negatives. Report
cluster-sandwich 95% ratio intervals for recall, precision, prevalence and mean prediction
minus observed outcome. Flag mean probability bias as material only at >=2 percentage
points with an interval excluding zero. These criteria neither certify fairness nor rule
out within-group miscalibration. No demographic recalibration or new subgroup cuts follow.

Limited error diagnostics compare false negatives/positives, low-utilization misses,
destination, stay and other declared groups with validation. Characterization does not
authorize refitting. Explanations use the already-frozen models and validation examples.
Linear SHAP is checked against logistic scores; permutation SHAP of the actual categorical
booster replaces its unsupported TreeSHAP adapter. Explanations are predictive, not causal.

## One-time execution and repeatable review

`make final-eval` requires this committed protocol, the committed JSON lock/specification,
unchanged evaluation code/configuration/dependencies, exact original model bytes and the
original partition hashes. It writes an exclusive start marker before parsing test and
calls each prespecified model's prediction method once. Saved predictions remain ignored.
Subsequent local analysis reuses verified cached predictions; an incomplete scoring attempt
cannot automatically retry. A fresh checkout containing published final results refuses
new test scoring. CI verifies the archived results and synthetic guard/reproducibility tests;
it does not repeat the real test prediction pass.

Final model metadata and an ignored copy of the primary pipeline are created after successful
evaluation. Test findings cannot change these decisions. No service or deployment is built.
