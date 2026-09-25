# Final feature availability policy

The frozen primary remains the Phase 4 uncalibrated logistic pipeline with **ten source
predictors** and no new features. Prediction occurs at confirmed discharge after destination
is known. The cohort and feature definitions are unchanged.

| Inputs | Availability supported by source semantics | Operational qualification |
|---|---|---|
| Age band, recorded gender | Demographics describe the encounter and normally precede discharge | Verify the contemporaneous registration snapshot; recorded gender is not an inferred identity |
| Admission type and source | Describe admission before the discharge event | Administrative corrections and code mappings are institution dependent |
| Medical specialty | Defined as the admitting physician's specialty | Unknown remains explicit; the extract does not establish when this field was entered |
| Inpatient, emergency and outpatient counts | Source explicitly defines visits in the preceding year | Reproduce the look-back window and capture boundary; fragmented external history can be missed |
| Time in hospital | Elapsed stay is conceptually known at confirmed discharge | Validate day-count conventions and prohibit later corrected billing values |
| Discharge destination | Available by definition of the chosen confirmed-destination workflow | A conditional input: real-time availability is NOT verified; do not score earlier or substitute later finalized codes |

No predictor's actual EHR arrival timestamp is verified in this extract. “Available” here means
consistent with source semantics and the stated scoring moment, not proven readiness at a named
hospital. Admission information/demographics and elapsed stay are conceptually antecedent; their
recording, coverage and corrections still require a site audit. The portfolio assumption must not
be silently carried into clinical deployment.

Diagnoses/counts, medications, labs, procedures and payer remain excluded from the primary model.
The prior diagnosis sensitivities cannot override missing timing evidence. Payer missingness and
race may be examined for retrospective evaluation, but never enter predictions. IDs are used only
for linkage, clustering and an outcome-independent tie-break; they are not model predictors or
chronology. Outcomes, future patient aggregates, sparse weight and constants remain excluded.
Raw prior-utilization counts are retained; the tested totals/indicators were not retained.

The primary pipeline standardizes its four numeric columns using training means/SDs, normalizes
documented categorical missing/unknown codes, pools rare categories using training frequencies,
and one-hot encodes six categorical columns. No transformation is refitted in Phase 5.
The challenger retains its original native-categorical preprocessing and frozen training weights.

Evidence: [original scoring contract](scoring_time_contract.md),
[reviewed source field audit](../data_quality/feature_audit.md),
[Phase 4 feature comparison](tuning_report.md), and
[UCI variable definitions](https://doi.org/10.24432/C5230J).
These sources support the conservative feature policy; none supplies field-level arrival times.
