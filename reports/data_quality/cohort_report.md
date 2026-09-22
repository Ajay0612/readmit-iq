# Analytical cohort: discharge outreach

Predict the source `<30` readmission label for confirmed live discharges to home,
outpatient follow-up, or selected postacute settings. Scoring occurs after the destination
is confirmed and the available encounter record is assembled. This is a project-specific
outreach population, not a recreation of a CMS quality measure.

## Eligibility rationale

- Death-coded dispositions cannot support live-patient outreach. Hospice is excluded
  because its goals require a separate care pathway, not because readmission is impossible.
  The hospice rows actually contain 43 `<30` labels; death-coded rows contain zero.
- Explicit hospital/inpatient transfers do not identify the completed episode needed for
  post-discharge outreach. No timestamps allow transfer chains to be joined. Codes 5
  and 27 have broad descriptions; their conservative exclusion is a project assumption.
- Codes 9 and 12 do not confirm discharge; 15 is a within-institution swing-bed continuation.
  Those cases need a separately specified transition-of-care workflow.
- Unknown, unmapped and insufficiently specified destinations cannot verify eligibility.
  They are excluded from the primary cohort, not recoded as safe home discharges. The
  retained-unknown sensitivity quantifies this assumption's population effect.
- SNF, intermediate-care, nursing-facility and rehabilitation destinations remain eligible
  for coordinated postacute follow-up. Code 22 includes hospital rehab units, so its mixed
  setting is an explicit uncertainty; a no-rehabilitation sensitivity is included.
- Left-against-medical-advice encounters are retained: the hospital may still offer outreach.
  No age-based exclusion or first-encounter-per-patient filtering is imposed.

These decisions were based on category meaning rather than optimizing label prevalence.
No care benefit or avoidability is inferred from the label. All raw bytes are unchanged.

## Exclusions (mutually exclusive)

| reason | codes | encounters | positives | rationale |
| --- | --- | --- | --- | --- |
| death | 11, 19, 20, 21 | 1652 | 0 | Death-coded encounters are outside a live-discharge outreach population. |
| hospice | 13, 14 | 771 | 43 | Hospice needs a separate goal-concordant pathway; readmission is possible but is not the standard outreach objective. |
| inpatient_transfer | 2, 5, 10, 23, 27, 28, 29 | 3874 | 670 | Continued hospital or unspecified inpatient care; a completed discharge episode cannot be reconstructed without dates. |
| discharge_not_confirmed | 9, 12, 15 | 87 | 39 | Still admitted, ambiguous still-patient status, or within-institution swing-bed continuation. |
| unknown_destination | 18, 25, 26, 30 | 4680 | 551 | Destination is unknown or insufficiently specified to confirm eligibility; retain separately for sensitivity analysis. |

## Flow and prevalence

| stage | removed_at_stage | encounters | patients | positives | negatives | prevalence_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Raw source | 0 | 101766 | 71518 | 11357 | 90409 | 11.160 |
| After death | 1652 | 100114 | 70439 | 11357 | 88757 | 11.344 |
| After hospice | 771 | 99343 | 69990 | 11314 | 88029 | 11.389 |
| After inpatient_transfer | 3874 | 95469 | 67785 | 10644 | 84825 | 11.149 |
| After discharge_not_confirmed | 87 | 95382 | 67749 | 10605 | 84777 | 11.118 |
| After unknown_destination | 4680 | 90702 | 65044 | 10054 | 80648 | 11.085 |

## Sensitivity to cohort assumptions

| cohort | encounters | patients | positives | negatives | prevalence_pct |
| --- | --- | --- | --- | --- | --- |
| Raw source | 101766 | 71518 | 11357 | 90409 | 11.160 |
| Exclude death/hospice only | 99343 | 69990 | 11314 | 88029 | 11.389 |
| Primary cohort | 90702 | 65044 | 10054 | 80648 | 11.085 |
| Primary + unknown destinations | 95382 | 67749 | 10605 | 84777 | 11.118 |
| Primary without mixed rehabilitation | 88709 | 63998 | 9502 | 79207 | 10.711 |
| Community destinations only | 73892 | 55284 | 7345 | 66547 | 9.940 |

A changed prevalence describes the chosen population; it is not evidence that a model
improves. The mixed rehabilitation setting materially changes the population's rate,
so its inclusion should be settled with the intended workflow before Phase 3.

## Every source disposition code

| code | description | encounters | positives | decision |
| --- | --- | --- | --- | --- |
| 1 | Discharged to home | 60234 | 5602 | eligible |
| 2 | Discharged/transferred to another short term hospital | 2128 | 342 | inpatient_transfer |
| 3 | Discharged/transferred to SNF | 13954 | 2046 | eligible |
| 4 | Discharged/transferred to ICF | 815 | 104 | eligible |
| 5 | Discharged/transferred to another type of inpatient care institution | 1184 | 247 | inpatient_transfer |
| 6 | Discharged/transferred to home with home health service | 12902 | 1638 | eligible |
| 7 | Left AMA | 623 | 90 | eligible |
| 8 | Discharged/transferred to home under care of Home IV provider | 108 | 15 | eligible |
| 9 | Admitted as an inpatient to this hospital | 21 | 9 | discharge_not_confirmed |
| 10 | Neonate discharged to another hospital for neonatal aftercare | 6 | 0 | inpatient_transfer |
| 11 | Expired | 1642 | 0 | death |
| 12 | Still patient or expected to return for outpatient services | 3 | 2 | discharge_not_confirmed |
| 13 | Hospice / home | 399 | 19 | hospice |
| 14 | Hospice / medical facility | 372 | 24 | hospice |
| 15 | Discharged/transferred within this institution to Medicare approved swing bed | 63 | 28 | discharge_not_confirmed |
| 16 | Discharged/transferred/referred another institution for outpatient services | 11 | 0 | eligible |
| 17 | Discharged/transferred/referred to this institution for outpatient services | 14 | 0 | eligible |
| 18 | NULL | 3691 | 459 | unknown_destination |
| 19 | Expired at home. Medicaid only, hospice. | 8 | 0 | death |
| 20 | Expired in a medical facility. Medicaid only, hospice. | 2 | 0 | death |
| 21 | Expired, place unknown. Medicaid only, hospice. | 0 | 0 | death |
| 22 | Discharged/transferred to another rehab fac including rehab units of a hospital . | 1993 | 552 | eligible |
| 23 | Discharged/transferred to a long term care hospital. | 412 | 30 | inpatient_transfer |
| 24 | Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare. | 48 | 7 | eligible |
| 25 | Not Mapped | 989 | 92 | unknown_destination |
| 26 | Unknown/Invalid | 0 | 0 | unknown_destination |
| 27 | Discharged/transferred to a federal health care facility. | 5 | 0 | inpatient_transfer |
| 28 | Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital | 139 | 51 | inpatient_transfer |
| 29 | Discharged/transferred to a Critical Access Hospital (CAH). | 0 | 0 | inpatient_transfer |
| 30 | Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere | 0 | 0 | unknown_destination |

Descriptions are from the checksum-verified `IDS_mapping.csv`, including unused codes.
These historical source mappings take precedence over modern code lists.

## Target and reproducibility

`<30 → 1`, `>30 → 0`, `NO → 0`. Unexpected labels fail before any cohort exclusion.
This approximates the business question using recorded less-than-30-day readmission.
The inclusive day-30 boundary, out-of-network events, planned readmissions and complete
follow-up are unverified. The original label remains alongside `readmitted_lt30`.
Rules are versioned in `configs/phase2.yaml` and applied by `data/cohort.py`.
`data/interim/eligible_encounters.csv` and `cohort_membership.csv` are ignored by Git.
Only fixed `? → missing` normalization and the target were added; no learned transform
was fit. Administrative codes and lab `None` remain intact.

## Sources and scope

- [UCI release and variable descriptions](https://doi.org/10.24432/C5230J).
- [Strack et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/): the original
  analysis excluded death/hospice and used one encounter per patient. This cohort retains
  repeated encounters and therefore does not reproduce that study population.
- [CMS/Yale 2011 draft methodology, section 2.3.2](https://www.cms.gov/medicare/quality-initiatives-patient-assessment-instruments/mms/downloads/mmshospital-wideall-conditionreadmissionrate.pdf):
  supports distinguishing live discharges and acute transfer episodes. Its Medicare-age
  and AMA quality-measure rules are not adopted for this different outreach use case.
