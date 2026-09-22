# Phase 3 scoring-time contract

Produce the recorded <30-day readmission risk **at confirmed discharge**, once the destination
is agreed and before the outreach list is finalized. This is one discharge-time use case,
not an admission-time or earlier-stay model. No score may use a later encounter or outcome.
The source labels do not identify avoidable/unplanned events or complete external follow-up.

The extract has no field-level timestamps. Source semantics support the primary allowlist;
they do not prove operational availability. A live implementation must reconstruct the same
as-of-discharge snapshot. If the destination is not yet confirmed, this model is not ready to
score; excluding disposition alone would not establish an earlier-stay model.

| Family | Status | Decision and evidence |
|---|---|---|
| Age band, gender | Allowed | Recorded demographics; categorical age avoids invented midpoint precision and a forced linear gradient. |
| Race | Excluded as predictor; retained for audit | Conservative development scope; no fairness claim or inferred attributes. Future predictive use needs a separate documented review. |
| Admission type/source, admitting specialty | Allowed | Antecedent admission information under UCI definitions; normalize documented unknown codes. |
| Prior inpatient/emergency/outpatient visits | Allowed | Source explicitly defines the preceding year. Derived totals/any-use indicators use only these three fields. |
| Time in hospital | Allowed | Elapsed inpatient days at confirmed discharge; source's retrospective value must match that cutoff operationally. |
| Discharge disposition | Allowed conditionally | Known destination is part of this scoring workflow. Source documentation describes the destination but gives no arrival timestamp. Do not substitute a later corrected billing disposition. Run one without-disposition ablation on the same cohort. |
| Diagnosis codes/count | Uncertain; sensitivity only | Final billed codes may arrive late. Compare documented broad groups against raw strings, but do not promote based on validation gain without a timing audit. |
| Procedures, lab-procedure and medication counts | Uncertain; sensitivity only | Completed-stay aggregate counts can be reconstructed retrospectively; arrival at scoring is unverified. |
| Medication statuses, change, diabetesMed, insulin | Uncertain; sensitivity only | Source reports prescriptions/dose changes during the encounter, not a verified discharge medication list. |
| A1c/glucose summaries | Uncertain; sensitivity only | Result may not be available at scoring. Preserve Not measured separately from Unknown and Normal. |
| Payer | Uncertain; sensitivity only | Coverage may be known, but final billing value and access-proxy stability are unverified. |
| IDs, outcomes, future patient aggregates | Excluded | Identity/target leakage. Patient ID is only an allocation/audit key. No chronology inferred from encounter ID. |
| Weight; examide/citoglipton | Excluded | Weight has inadequate coverage for initial modeling; medication fields are constant. |

Freeze the existing 90,702-encounter cohort: retain mixed rehabilitation for coordinated
postacute outreach and exclude unknown destination because eligibility cannot be confirmed.
These are scope decisions, not prevalence/performance optimizations. The Phase 2 sensitivity
analysis remains historical; do not redefine the frozen cohort using validation or test results.

**Primary candidate inputs:** age, gender, admission_type_id, admission_source_id,
discharge_disposition_id, medical_specialty, time_in_hospital and the three prior-utilization
counts, with four encounter-local utilization derivations. All other fields are excluded
from primary models. Separate sensitivity experiments cannot become the development winner
solely through a higher score.

**No future patient history:** repeated-patient status, total appearances in the extract,
patient-wide outcome rates and ordered histories reconstructed from IDs are prohibited.
We cannot recover true chronology without encounter dates.

Sources: [UCI variable definitions](https://doi.org/10.24432/C5230J) and
[Strack et al. (2014), feature definitions and diagnosis grouping](https://onlinelibrary.wiley.com/doi/10.1155/2014/781670).
The conditional workflow and conservative primary/sensitivity distinction are project decisions;
neither source verifies real-time EHR availability.
