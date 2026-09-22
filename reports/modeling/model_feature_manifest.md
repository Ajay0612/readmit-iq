# Model feature manifest

| source | transformed | type | status | reason |
| --- | --- | --- | --- | --- |
| encounter_id | none | categorical | Excluded | Traceability only; never an order surrogate |
| patient_nbr | none | categorical | Excluded | Grouping only; no future patient aggregates |
| race | none | categorical | Excluded | Audit-only; predictive inclusion deferred |
| gender | gender | categorical | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| age | age | categorical | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| weight | none | categorical | Excluded | 96.68% missing in Phase 2; no initial predictive use |
| admission_type_id | admission_type_id | categorical | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| discharge_disposition_id | discharge_disposition_id | categorical | Primary allowed | Conditional on confirmed destination; without-disposition ablation included |
| admission_source_id | admission_source_id | categorical | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| time_in_hospital | time_in_hospital | numeric | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| payer_code | payer_code | categorical | Sensitivity only | Billing/result/full-stay finalization time unverified |
| medical_specialty | medical_specialty | categorical | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| num_lab_procedures | num_lab_procedures | numeric | Sensitivity only | Billing/result/full-stay finalization time unverified |
| num_procedures | num_procedures | numeric | Sensitivity only | Billing/result/full-stay finalization time unverified |
| num_medications | num_medications | numeric | Sensitivity only | Billing/result/full-stay finalization time unverified |
| number_outpatient | number_outpatient | numeric | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| number_emergency | number_emergency | numeric | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| number_inpatient | number_inpatient | numeric | Primary allowed | Antecedent/as-of-discharge source semantics; see scoring-time contract |
| diag_1 | diag_1_group or raw diag_1 | categorical | Sensitivity only | Final coding time unverified; broad documented groups versus train-pooled raw codes |
| diag_2 | diag_2_group or raw diag_2 | categorical | Sensitivity only | Final coding time unverified; broad documented groups versus train-pooled raw codes |
| diag_3 | diag_3_group or raw diag_3 | categorical | Sensitivity only | Final coding time unverified; broad documented groups versus train-pooled raw codes |
| number_diagnoses | number_diagnoses | numeric | Sensitivity only | Billing/result/full-stay finalization time unverified |
| max_glu_serum | max_glu_serum | categorical | Sensitivity only | Billing/result/full-stay finalization time unverified |
| A1Cresult | A1Cresult | categorical | Sensitivity only | Billing/result/full-stay finalization time unverified |
| metformin | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| repaglinide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| nateglinide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| chlorpropamide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| glimepiride | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| acetohexamide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| glipizide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| glyburide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| tolbutamide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| pioglitazone | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| rosiglitazone | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| acarbose | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| miglitol | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| troglitazone | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| tolazamide | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| examide | none | categorical | Excluded | Constant |
| citoglipton | none | categorical | Excluded | Constant |
| insulin | insulin | categorical | Sensitivity only | Billing/result/full-stay finalization time unverified |
| glyburide-metformin | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| glipizide-metformin | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| glimepiride-pioglitazone | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| metformin-rosiglitazone | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| metformin-pioglitazone | active/changed diabetes drug-field counts | categorical | Sensitivity derivation only | Count documented status fields; do not interpret as independent ingredients or change events |
| change | change | categorical | Sensitivity only | Billing/result/full-stay finalization time unverified |
| diabetesMed | diabetesMed | categorical | Sensitivity only | Billing/result/full-stay finalization time unverified |
| readmitted | none | categorical | Excluded | Original outcome: direct leakage |
| readmitted_lt30 | y only | binary target | Excluded from predictors | Validated target; direct leakage if used as X |
| number_inpatient, number_emergency, number_outpatient | total_prior_utilization | numeric | Primary engineered candidate | Fixed encounter-local arithmetic; raw-count ablation tests joint added value |
| number_inpatient | any_prior_inpatient | binary | Primary engineered candidate | Fixed encounter-local arithmetic; raw-count ablation tests joint added value |
| number_emergency | any_prior_emergency | binary | Primary engineered candidate | Fixed encounter-local arithmetic; raw-count ablation tests joint added value |
| number_outpatient | any_prior_outpatient | binary | Primary engineered candidate | Fixed encounter-local arithmetic; raw-count ablation tests joint added value |
| 21 nonconstant drug-status columns | active_diabetes_drug_fields | numeric | Sensitivity only | Active = Steady/Up/Down; changed = Up/Down; No counts zero |
| 21 nonconstant drug-status columns | changed_diabetes_drug_fields | numeric | Sensitivity only | Active = Steady/Up/Down; changed = Up/Down; No counts zero |

Primary: 10 source fields plus 4 fixed utilization derivations (14 columns before encoding). No identifiers, outcomes or full-dataset patient aggregates are passed to estimators. All source fields remain in ignored analytical data for traceability.

Linear and random-forest pipelines use training-pooled one-hot categories; only linear numeric columns are standardized. Histogram boosting uses ordinal category IDs explicitly marked categorical, not numeric ordering. All numerical counts are complete under the validated source contract, so no numeric imputer is fitted. Missing/nonfinite numeric counts fail explicitly; revise the contract if future data requires a train-only imputer.

Categories seen fewer than 100 times in training map to Other; the count cutoff was fixed before validation. Training frequencies are saved separately. Unknown, Not measured and Invalid are reserved levels; age bands remain intact. Unseen values map safely to Other. Age is categorical rather than a fabricated exact age or a forced linear ordinal trend.

Clinical diagnosis ranges follow [Strack et al. Table 3](https://onlinelibrary.wiley.com/doi/10.1155/2014/781670): 390–459/785 circulatory, 460–519/786 respiratory, 520–579/787 digestive, 580–629/788 genitourinary, 250.xx diabetes, 800–999 injury, 710–739 musculoskeletal, 140–239 neoplasms, otherwise Other. The same documented grouping applied to secondary/tertiary fields is an explicit project extension. Decimal suffixes use the numeric code family; short numeric codes remain in their valid broad family. E/V supplementary codes map to residual Other, missing to Unknown and malformed codes to Invalid. This is broad grouping, not validation of every ICD code against a complete codebook. A primary-diabetes indicator is omitted because the grouped category already contains exactly that information.

The two medication counts describe source columns, not distinct ingredients, doses, prescriptions at discharge or numbers of change events. Combination-product fields count once; constants are omitted. Never count No as active or Unknown as no treatment.
