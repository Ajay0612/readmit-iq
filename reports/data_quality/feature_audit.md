# Feature audit and leakage review

All 50 original fields are audited below. Detailed source definitions, raw/cohort dtypes,
raw cardinality, observed low-cardinality values, missing markers and policy decisions
are machine-readable in `feature_audit.csv`. Semantic definitions also remain in
[`docs/data_dictionary.md`](../../docs/data_dictionary.md). No predictor matrix is built.

## Scoring contract

Score at completed discharge, after destination confirmation. 'Definitely safe' below
means the source definition is antecedent to the outcome; it does not prove live-system
availability because this extract has no field-level timestamps. For retrospective
encounter summaries, finalized diagnoses and billing fields, verify actual arrival time
before deployment. Preserve questionable fields for Phase 3 sensitivity experiments.

Discharge disposition is not mechanically future leakage at this scoring time. It defines
eligibility and may encode destination/care intensity, but may be finalized late or change
under an intervention. Compare models with and without it, conditional on the same cohort.
Diagnosis counts/codes and medication/procedure summaries may be finalized after discharge;
they belong in a timing-sensitive ablation. No post-readmission measurement is knowingly
used. An admission-time use case would require a substantially different feature set.

## Clear modeling exclusions

`encounter_id`, `patient_nbr`: identifiers. `readmitted` and the added `readmitted_lt30`: outcomes.
`examide`, `citoglipton`: constant. `weight`: initially omit for sparse coverage, retain for
sensitivity. These are modeling recommendations; columns remain in the analytical file.
Full-dataset encounter counts/repeat status and future patient outcomes are analysis-only
and must never become predictor columns. Do not infer chronology from encounter IDs.

## Per-variable review

Unknown percentages include `?`, parser NA, administrative unknown codes and invalid gender,
as applicable. Not-measured lab `None` is reported separately and is not missingness.

### Identifiers

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| encounter_id | identifier | 90702 | 0.000 | 0 |
| patient_nbr | identifier | 65044 | 0.000 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| encounter_id | excluded: identity leakage | Yes, but never a predictor | Keep for audit/grouping only; no ID-order chronology. |
| patient_nbr | excluded: identity leakage | Yes, but never a predictor | Keep for audit/grouping only; no ID-order chronology. |

### Demographics

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| race | Categorical | 6 | 2.299 | 0 |
| gender | Categorical | 3 | 0.003 | 0 |
| age | Categorical | 10 | 0.000 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| race | definitely safe by source semantics | Expected before discharge; source has no timestamps | Keep source categories/Unknown; subgroup audits; no causal inference. |
| gender | definitely safe by source semantics | Expected before discharge; source has no timestamps | Keep source categories/Unknown; subgroup audits; no causal inference. |
| age | definitely safe by source semantics | Expected before discharge; source has no timestamps | Keep source categories/Unknown; subgroup audits; no causal inference. |

### Encounter information

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| weight | Categorical | 10 | 96.678 | 0 |
| admission_type_id | Categorical | 8 | 9.560 | 0 |
| discharge_disposition_id | Categorical | 10 | 0.000 | 0 |
| admission_source_id | Categorical | 17 | 6.779 | 0 |
| time_in_hospital | Integer | 14 | 0.000 | 0 |
| num_lab_procedures | Integer | 117 | 0.000 | 0 |
| num_procedures | Integer | 7 | 0.000 | 0 |
| num_medications | Integer | 75 | 0.000 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| weight | questionable: sparse coverage | If measured during stay; operational timing unverified | Exclude from first modeling pass; retain for sensitivity, no imputation. |
| admission_type_id | definitely safe by source semantics | Admission information or elapsed stay is known at completed discharge | Keep source categories/counts; verify runtime cutoff, no scaling yet. |
| discharge_disposition_id | questionable: discharge snapshot | Usable only if this final encounter value exists at scoring | Retain for cohort; compare models with/without disposition in Phase 3. |
| admission_source_id | definitely safe by source semantics | Admission information or elapsed stay is known at completed discharge | Keep source categories/counts; verify runtime cutoff, no scaling yet. |
| time_in_hospital | definitely safe by source semantics | Admission information or elapsed stay is known at completed discharge | Keep source categories/counts; verify runtime cutoff, no scaling yet. |
| num_lab_procedures | questionable: discharge snapshot | Usable only if this final encounter value exists at scoring | Keep source categories/counts; verify runtime cutoff, no scaling yet. |
| num_procedures | questionable: discharge snapshot | Usable only if this final encounter value exists at scoring | Keep source categories/counts; verify runtime cutoff, no scaling yet. |
| num_medications | questionable: discharge snapshot | Usable only if this final encounter value exists at scoring | Keep source categories/counts; verify runtime cutoff, no scaling yet. |

### Administrative variables

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| payer_code | Categorical | 18 | 36.482 | 0 |
| medical_specialty | Categorical | 73 | 48.283 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| payer_code | questionable: billing finalization | Coverage information may be known; final billing payer unverified | Explicit Unknown; retain for ablation; monitor care-access proxies. |
| medical_specialty | definitely safe by source semantics | Admitting physician specialty, if recorded | Explicit Unknown; retain for ablation; monitor care-access proxies. |

### Utilization history

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| number_outpatient | Integer | 39 | 0.000 | 0 |
| number_emergency | Integer | 32 | 0.000 | 0 |
| number_inpatient | Integer | 20 | 0.000 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| number_outpatient | definitely safe by source semantics | Defined as the year before this encounter | Keep separate counts; evaluate nonlinear effects; no clipping yet. |
| number_emergency | definitely safe by source semantics | Defined as the year before this encounter | Keep separate counts; evaluate nonlinear effects; no clipping yet. |
| number_inpatient | definitely safe by source semantics | Defined as the year before this encounter | Keep separate counts; evaluate nonlinear effects; no clipping yet. |

### Diagnoses

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| diag_1 | Categorical | 709 | 0.017 | 0 |
| diag_2 | Categorical | 741 | 0.368 | 0 |
| diag_3 | Categorical | 777 | 1.441 | 0 |
| number_diagnoses | Integer | 16 | 0.000 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| diag_1 | questionable: retrospective coding | Clinical diagnoses may be known; finalized billing codes may be late | Preserve strings/Unknown; verify cutoff, then test broad grouping. |
| diag_2 | questionable: retrospective coding | Clinical diagnoses may be known; finalized billing codes may be late | Preserve strings/Unknown; verify cutoff, then test broad grouping. |
| diag_3 | questionable: retrospective coding | Clinical diagnoses may be known; finalized billing codes may be late | Preserve strings/Unknown; verify cutoff, then test broad grouping. |
| number_diagnoses | questionable: retrospective coding | Clinical diagnoses may be known; finalized billing codes may be late | Keep numeric diagnosis count; verify coding cutoff, then test nonlinear effects. |

### Lab indicators

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| max_glu_serum | Categorical | 4 | 0.000 | 86216 |
| A1Cresult | Categorical | 4 | 0.000 | 75443 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| max_glu_serum | questionable: result availability | Only usable if the result exists by discharge; timing unverified | Preserve None as not measured; categorical, no numeric imputation. |
| A1Cresult | questionable: result availability | Only usable if the result exists by discharge; timing unverified | Preserve None as not measured; categorical, no numeric imputation. |

### Medications

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| metformin | Categorical | 4 | 0.000 | 0 |
| repaglinide | Categorical | 4 | 0.000 | 0 |
| nateglinide | Categorical | 4 | 0.000 | 0 |
| chlorpropamide | Categorical | 4 | 0.000 | 0 |
| glimepiride | Categorical | 4 | 0.000 | 0 |
| acetohexamide | Categorical | 2 | 0.000 | 0 |
| glipizide | Categorical | 4 | 0.000 | 0 |
| glyburide | Categorical | 4 | 0.000 | 0 |
| tolbutamide | Categorical | 2 | 0.000 | 0 |
| pioglitazone | Categorical | 4 | 0.000 | 0 |
| rosiglitazone | Categorical | 4 | 0.000 | 0 |
| acarbose | Categorical | 4 | 0.000 | 0 |
| miglitol | Categorical | 4 | 0.000 | 0 |
| troglitazone | Categorical | 2 | 0.000 | 0 |
| tolazamide | Categorical | 2 | 0.000 | 0 |
| examide | Categorical | 1 | 0.000 | 0 |
| citoglipton | Categorical | 1 | 0.000 | 0 |
| insulin | Categorical | 4 | 0.000 | 0 |
| glyburide-metformin | Categorical | 4 | 0.000 | 0 |
| glipizide-metformin | Categorical | 2 | 0.000 | 0 |
| glimepiride-pioglitazone | Categorical | 2 | 0.000 | 0 |
| metformin-rosiglitazone | Categorical | 2 | 0.000 | 0 |
| metformin-pioglitazone | Categorical | 2 | 0.000 | 0 |
| change | Categorical | 2 | 0.000 | 0 |
| diabetesMed | Categorical | 2 | 0.000 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| metformin | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| repaglinide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| nateglinide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| chlorpropamide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| glimepiride | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| acetohexamide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| glipizide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| glyburide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| tolbutamide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| pioglitazone | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| rosiglitazone | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| acarbose | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| miglitol | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| troglitazone | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| tolazamide | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| examide | excluded: constant, not leakage | In-stay treatment may be known; source aggregation cutoff unverified | Exclude from modeling; keep original field for source audit. |
| citoglipton | excluded: constant, not leakage | In-stay treatment may be known; source aggregation cutoff unverified | Exclude from modeling; keep original field for source audit. |
| insulin | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| glyburide-metformin | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| glipizide-metformin | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| glimepiride-pioglitazone | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| metformin-rosiglitazone | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| metformin-pioglitazone | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| change | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |
| diabetesMed | questionable: full-stay reconstruction | In-stay treatment may be known; source aggregation cutoff unverified | Keep categorical; separate No from missing; no treatment-effect claims. |

### Outcome

| variable | semantic_type | cohort_cardinality | cohort_unknown_pct | cohort_not_measured_count |
| --- | --- | --- | --- | --- |
| readmitted | Categorical | 3 | 0.000 | 0 |

| variable | leakage_review | availability_at_discharge | treatment |
| --- | --- | --- | --- |
| readmitted | excluded: direct target leakage | No: requires post-discharge follow-up | Outcome only; exclude from every predictor matrix. |
