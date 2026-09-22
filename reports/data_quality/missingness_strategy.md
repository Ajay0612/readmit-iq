# Missingness strategy

Source `?` values are normalized to missing on a copied eligible table. No mean/mode
imputation, rare-level merging, scaling or encoding is fit on the full dataset.

## Ranked coverage and treatment

| variable | raw_question_marks | raw_unknown_pct | cohort_unknown_count | cohort_unknown_pct | treatment |
| --- | --- | --- | --- | --- | --- |
| weight | 98569 | 96.858 | 87689 | 96.678 | Exclude from first modeling pass; retain for sensitivity, no imputation. |
| medical_specialty | 49949 | 49.082 | 43794 | 48.283 | Explicit Unknown; retain for ablation; monitor care-access proxies. |
| payer_code | 40256 | 39.557 | 33090 | 36.482 | Explicit Unknown; retain for ablation; monitor care-access proxies. |
| admission_type_id | 0 | 10.216 | 8671 | 9.560 | Keep source categories/counts; verify runtime cutoff, no scaling yet. |
| admission_source_id | 0 | 6.944 | 6149 | 6.779 | Keep source categories/counts; verify runtime cutoff, no scaling yet. |
| race | 2273 | 2.234 | 2085 | 2.299 | Keep source categories/Unknown; subgroup audits; no causal inference. |
| diag_3 | 1423 | 1.398 | 1307 | 1.441 | Preserve strings/Unknown; verify cutoff, then test broad grouping. |
| diag_2 | 358 | 0.352 | 334 | 0.368 | Preserve strings/Unknown; verify cutoff, then test broad grouping. |
| diag_1 | 21 | 0.021 | 15 | 0.017 | Preserve strings/Unknown; verify cutoff, then test broad grouping. |
| gender | 0 | 0.003 | 3 | 0.003 | Keep source categories/Unknown; subgroup audits; no causal inference. |

## Is unknown status associated with the outcome?

| feature | level | encounters | positives | rate_pct | ci_low_pct | ci_high_pct | sparse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weight | Observed | 3013 | 335 | 11.118 | 9.959 | 12.278 | False |
| weight | Unknown | 87689 | 9719 | 11.083 | 10.835 | 11.332 | False |
| medical_specialty | Observed | 46908 | 5007 | 10.674 | 10.343 | 11.005 | False |
| medical_specialty | Unknown | 43794 | 5047 | 11.524 | 11.177 | 11.872 | False |
| payer_code | Observed | 57612 | 6364 | 11.046 | 10.729 | 11.363 | False |
| payer_code | Unknown | 33090 | 3690 | 11.151 | 10.780 | 11.523 | False |
| admission_type_id | Observed | 82031 | 9109 | 11.104 | 10.846 | 11.362 | False |
| admission_type_id | Unknown | 8671 | 945 | 10.898 | 10.202 | 11.595 | False |
| admission_source_id | Observed | 84553 | 9412 | 11.131 | 10.878 | 11.385 | False |
| admission_source_id | Unknown | 6149 | 642 | 10.441 | 9.598 | 11.284 | False |
| race | Observed | 88617 | 9888 | 11.158 | 10.911 | 11.405 | False |
| race | Unknown | 2085 | 166 | 7.962 | 6.720 | 9.203 | False |
| diag_3 | Observed | 89395 | 9971 | 11.154 | 10.912 | 11.396 | False |
| diag_3 | Unknown | 1307 | 83 | 6.350 | 4.269 | 8.432 | False |
| diag_2 | Observed | 90368 | 10028 | 11.097 | 10.854 | 11.340 | False |
| diag_2 | Unknown | 334 | 26 | 7.784 | 4.757 | 10.812 | False |
| diag_1 | Observed | 90687 | 10050 | 11.082 | 10.839 | 11.325 | False |
| diag_1 | Unknown | 15 | 4 | 26.667 | — | — | True |
| gender | Observed | 90699 | 10054 | 11.085 | 10.842 | 11.328 | False |
| gender | Unknown | 3 | 0 | 0.000 | — | — | True |

Confidence intervals use patient-cluster variance. Small groups have no normal interval.
These descriptive differences do not show why a value is missing; no MCAR/MAR/MNAR claim
or causal interpretation can be established from this extract.

## Decisions for Phase 3

- Weight: only 3,013 eligible encounters have recorded values. Unknown and recorded groups
  have very similar outcome rates. Omit from the first model; do not invent weights for the
  remaining population. A small observed subset cannot establish absence of within-weight
  associations. An explicit coverage ablation can be considered after the first model.
- Medical specialty: keep explicit Unknown and test a with/without-feature ablation.
  Missingness has an observed rate difference, but can encode hospital documentation habits.
- Payer: preserve Unknown and retain for a timed-availability/access-proxy ablation. Its
  missingness alone shows little difference, which does not prove the whole feature useless.
- Race: preserve Unknown, never infer or mode-fill race. Missingness relates to observed
  outcomes and warrants subgroup/coverage audits; no fairness claim follows from EDA.
- Diagnoses: retain explicit Unknown at encoding time. Missing primary diagnoses are too
  sparse for dependable inference. Do not replace absent diagnoses with a normal category.
- Admission codes: map documented unavailable/NULL/not-mapped codes to explicit Unknown
  inside the future pipeline, preserving the raw codes for audit. All primary-cohort
  discharge destinations are known by design.
- Gender Unknown/Invalid: preserve an unknown category and its three rows; do not infer gender.
- Lab None: retain as Not measured, distinct from an unknown result or a normal test.
- Low-frequency categorical levels: learn any grouping threshold on training data only.

Supported unknown-minus-observed differences (percentage points):

| feature | difference_pp | CI_low_pp | CI_high_pp |
| --- | --- | --- | --- |
| weight | -0.035 | -1.220 | 1.150 |
| medical_specialty | 0.850 | 0.377 | 1.324 |
| payer_code | 0.105 | -0.382 | 0.593 |
| admission_type_id | -0.206 | -0.947 | 0.535 |
| admission_source_id | -0.691 | -1.570 | 0.189 |
| race | -3.196 | -4.462 | -1.931 |
| diag_3 | -4.803 | -6.878 | -2.728 |
| diag_2 | -3.312 | -6.324 | -0.301 |
