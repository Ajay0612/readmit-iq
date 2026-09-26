# ReadmitIQ model and service card

**Version:** `1.0.0-frozen-portfolio`. **Use:** local decision-support portfolio demonstration
of retrospective readmission risk and batch-relative outreach prioritization. Built from
historical 1999–2008 hospital data; not validated for clinical decisions, diagnosis or patient
care. No claim of HIPAA compliance is made. Examples and serving verification use synthetic inputs.

## Population, target and scoring moment

The UCI Diabetes 130-US Hospitals dataset contains encounters with diabetes-related diagnoses.
The original eligible cohort has 90,702 encounters from 65,044 patients, split by patient with
zero overlap. The binary target is recorded readmission **less than 30 days** (`<30` versus
`>30`/`NO`). The exact day-30 boundary and outside-hospital follow-up cannot be independently
verified. Score after the discharge destination is confirmed. Destination-based exclusions
reuse the Phase 2 rules; cohort membership and reliable field arrival cannot be verified from
the service's ten inputs alone.

## Frozen model and predictors

Uncalibrated logistic regression: L2/lbfgs, `C=0.09988151348099303`, seed 42, no class weights.
Training uses 63,563 encounters only. Scaling, category normalization, rare pooling and one-hot
encoding were fitted on training and are unchanged. No refit, calibration or new features were
introduced in Phase 6.

The exact predictors are `time_in_hospital`, `number_inpatient`, `number_emergency`,
`number_outpatient`, `age`, `gender`, `admission_type_id`, `admission_source_id`,
`discharge_disposition_id`, `medical_specialty`. IDs, outcomes, race, diagnoses, medications,
labs and payer do not enter inference. The strict [API contract](local_demo.md#api-v1-contract)
documents valid ranges, categories, missing markers and eligibility.

## Archived final performance

These are the existing Phase 5 results, not a new evaluation:

| Metric | Frozen final test |
|---|---:|
| Average precision | 0.20837 |
| ROC-AUC | 0.65181 |
| Brier | 0.094101 |
| Recall at top 10% | 23.54% |
| Precision at top 10% | 25.85% |
| Lift | 2.36× |

On 13,549 test encounters, the policy selected 1,354 and surfaced 350 of 1,487 recorded
readmissions. Most readmissions were missed. Consult the [final report](../reports/modeling/final_test_report.md)
for patient-bootstrap uncertainty, error analysis and the limitations of held-out evaluation
after earlier full-cohort exploration. Metrics establish neither clinical utility nor savings.

## Policy and explanation

Rank an available eligible encounter batch by unrounded risk and select `floor(0.10 × N)`.
Ties use the original ascending SHA-256 of seed `42:` plus stable encounter/request ID.
Batches smaller than ten select zero. Capacity is a **portfolio assumption**, not hospital
staffing. Single-record scoring returns `outreach_selected: null`; probability alone is not
an outreach recommendation. There is no automatic patient-level deduplication or repeat-contact rule.

Prior inpatient use and discharge destination are major model associations. The linked
[SHAP/coefficient report](../reports/modeling/explainability_report.md) explains predictions,
not causes. The UI links it rather than computing request-time SHAP.

## Limitations and excluded uses

Half of test readmissions had no prior inpatient history. Recall in that group was only **7.6%**
versus **39.6%** with prior use; 56 of its 57 captured events involved rehabilitation destination
22. Low predicted risk is not reassurance. Contemporary external/temporal validation, clinical
review of rehabilitation coding, coverage/field-timing audits and prospective intervention
evaluation are required before considering any patient-care application.

Historical selection, missingness, incomplete follow-up, limited timestamps and uncertain
transportability remain. Subgroup estimates use minimum-size rules and patient-cluster
intervals; small groups and conditional uncertainty prevent fairness or subgroup-calibration
certification. Excluding race does not remove demographic proxies. No causal, prevention,
financial-benefit or hospital production-readiness claim is made.

## Provenance and operation

Training commit: `a48c68b10e0339e95c934ea4ab6b1ee7ff00a904`.
Freeze commit: `ce2fcdb5c79256f088e82af45d97bb6d51a0360d`.
Artifact SHA-256: `ff82996f1f49008dec373655d53cf95a2ce5d940cb7dbc2bd1e25ea6fe447fa8`.
The [committed specification](../reports/modeling/final_model_specification.json) and
[metadata](../reports/modeling/final_model_metadata.json) are the source of model/policy truth.

The loader verifies exact bytes and runtime compatibility before deserialization and loads
once per process. Docker receives the ignored binary by a local read-only mount. API limits
are 1,000 records / 1 MiB. The UI uses the same API; it has no model instance. Structured logs
omit payloads and identifiers. Authentication, TLS, hospital integration, clinical monitoring
and cloud hosting are outside this local demonstration. See [local setup](local_demo.md) and
[verification evidence](../reports/serving/phase6_verification.md).
