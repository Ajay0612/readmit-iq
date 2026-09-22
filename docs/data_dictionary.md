# Data dictionary

Definitions below come from the [UCI metadata](https://archive.ics.uci.edu/api/dataset?id=296),
retrieved for the original [dataset release](https://doi.org/10.24432/C5230J), CC BY 4.0.
Counts and loaded types are calculated by this repository and saved in `column_profile.csv`.

The column names and roles match all 50 columns in the downloaded file. Source type is
semantic, not a promise about the CSV parser dtype. Administrative codes are categorical.
IDs and ICD-9 diagnosis strings are preserved as strings. Medication spellings, including
`citoglipton`, are preserved without asserting a corrected drug identity.

For medication-status fields, UCI defines dosage increase, dosage decrease, unchanged dosage,
or not prescribed. The raw category spellings are `Up`, `Down`, `Steady`, and `No`.
For labs, `None` means not measured; `Norm` denotes the source normal-range category.
Raw `?` means missing. A1c/glucose category intervals are source descriptions, not
independently validated numerical lab values.

| Raw column | Role / source type | Source description |
|---|---|---|
| `encounter_id` | ID / identifier | Unique identifier of an encounter |
| `patient_nbr` | ID / identifier | Unique identifier of a patient |
| `race` | Feature / Categorical | Values: Caucasian, Asian, African American, Hispanic, and other |
| `gender` | Feature / Categorical | Values: male, female, and unknown/invalid |
| `age` | Feature / Categorical | Grouped in 10-year intervals: [0, 10), [10, 20),..., [90, 100) |
| `weight` | Feature / Categorical | Weight in pounds. |
| `admission_type_id` | Feature / Categorical | Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available |
| `discharge_disposition_id` | Feature / Categorical | Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available |
| `admission_source_id` | Feature / Categorical | Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital |
| `time_in_hospital` | Feature / Integer | Integer number of days between admission and discharge |
| `payer_code` | Feature / Categorical | Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay |
| `medical_specialty` | Feature / Categorical | Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon |
| `num_lab_procedures` | Feature / Integer | Number of lab tests performed during the encounter |
| `num_procedures` | Feature / Integer | Number of procedures (other than lab tests) performed during the encounter |
| `num_medications` | Feature / Integer | Number of distinct generic names administered during the encounter |
| `number_outpatient` | Feature / Integer | Number of outpatient visits of the patient in the year preceding the encounter |
| `number_emergency` | Feature / Integer | Number of emergency visits of the patient in the year preceding the encounter |
| `number_inpatient` | Feature / Integer | Number of inpatient visits of the patient in the year preceding the encounter |
| `diag_1` | Feature / Categorical | The primary diagnosis (coded as first three digits of ICD9); 848 distinct values |
| `diag_2` | Feature / Categorical | Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values |
| `diag_3` | Feature / Categorical | Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values |
| `number_diagnoses` | Feature / Integer | Number of diagnoses entered to the system |
| `max_glu_serum` | Feature / Categorical | Indicates the range of the result or if the test was not taken. Values: >200, >300, normal, and none if not measured |
| `A1Cresult` | Feature / Categorical | Indicates the range of the result or if the test was not taken. Values: >8 if the result was greater than 8%, >7 if the result was greater than 7% but less than 8%, normal if the result was less than 7%, and none if not measured. |
| `metformin` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `repaglinide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `nateglinide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `chlorpropamide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `glimepiride` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `acetohexamide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `glipizide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `glyburide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `tolbutamide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `pioglitazone` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `rosiglitazone` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `acarbose` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `miglitol` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `troglitazone` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `tolazamide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `examide` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `citoglipton` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `insulin` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `glyburide-metformin` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `glipizide-metformin` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `glimepiride-pioglitazone` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `metformin-rosiglitazone` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `metformin-pioglitazone` | Feature / Categorical | Medication status; uses the common dosage/prescription categories described above. |
| `change` | Feature / Categorical | Indicates if there was a change in diabetic medications (either dosage or generic name). Values: change and no change |
| `diabetesMed` | Feature / Categorical | Indicates if there was any diabetic medication prescribed. Values: yes and no |
| `readmitted` | Target / Categorical | Days to inpatient readmission. Values: <30 if the patient was readmitted in less than 30 days, >30 if the patient was readmitted in more than 30 days, and No for no record of readmission. |

## Interpretation notes

- `weight` is grouped in pounds. It is 96.86% missing in this raw release.
- `number_outpatient`, `number_emergency` and `number_inpatient` describe the year preceding the encounter.
- Medication measures describe the encounter. They must not be described as a verified post-discharge medication list.
- `readmitted` uses the literal `<30`, `>30` and `NO` categories. Exact timestamps are absent.
- Original administrative descriptions and observed code counts are available in
  `reports/data_quality/id_mapping.json` and `category_counts.json`. These are taken from
  `IDS_mapping.csv`, including its explicit `NULL` and `Not Mapped` categories.
- Some metadata descriptions list category counts that differ from the observed release.
  Use `column_profile.csv` for observed cardinality; do not force the raw data to match prose.

Reference: [Strack et al., 2014, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/).
