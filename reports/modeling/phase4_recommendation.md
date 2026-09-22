# Phase 4 recommendation from validation evidence

Prioritize **histogram gradient boosting and logistic regression**. Keep random forest as a
lower-priority reference, not a required tuning track: its AP advantage over logistic is
uncertain, while boosting has a modest positive paired difference. The best validation model
is a development candidate only. No final test evaluation or production model exists.

1. Carry forward the frozen patient assignments and confirmed-discharge feature contract.
   Use patient-grouped training folds for future parameter comparisons. Preserve validation
   for bounded development decisions and keep test locked until the final pipeline is fixed.
2. Start the logistic track from separate raw utilization counts: the total/any-use package
   showed no benefit. Retain all three counts; do not infer a strict monotonic constraint.
   Validate whether redundant derived terms are also unnecessary for boosting inside training.
3. Tune a small regularization/complexity range only in the next authorized phase. Favor the
   two observed promising families over importing many similarly motivated boosters now.
4. Assess calibration with group-aware out-of-fold training predictions. Weighted logistic
   probability distortion needs attention if weighting is pursued; AP currently offers no reason
   to prefer it. No current evidence justifies SMOTE.
5. Later choose an outreach threshold using costs/capacity and development data. At 0.50,
   boosting misses 99.53% of positive encounters, including most high-utilization cases.
   Do not mistake low default-threshold recall for absence of ranking value, or balanced-model
   recall for an improved precision–recall frontier.
6. Require a timing audit before adding final diagnoses, billing, lab or treatment summaries.
   Grouped diagnoses are more compact, but raw versus grouped performance is inconclusive.
   Encounter-summary gain is negligible. Neither sensitivity result overrides the contract.
7. Review subgroup calibration/error rates with patient uncertainty after choosing a meaningful
   development-only operating policy. Current tiny positive-call counts are inadequate for
   stable subgroup conclusions. Excluding race as a predictor does not establish fairness.

Unresolved operational questions: actual source-field arrival timestamps, availability of
confirmed destination before outreach allocation, suitability of mixed rehabilitation cases,
out-of-network outcome ascertainment and contemporary external validation. Cohort scope stays
fixed throughout current development; changing it requires an explicit versioned redesign.

No tuning, resampling, final calibration, SHAP, threshold optimization, API or deployment was
performed in Phase 3. Do not start those steps automatically.
