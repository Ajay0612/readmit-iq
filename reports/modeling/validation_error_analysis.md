# Validation errors at the selected outreach capacity

The top-10% policy captures 321 of 1,499 readmissions and leaves **1,178 false negatives**;
1,038 of the 1,359 targeted encounters have no recorded <30-day readmission. This is a
limited-capacity prioritization rule. An unflagged discharge must still receive standard care.

| cohort | encounters | patients | mean_probability | median_stay | no_prior_inpatient_fraction | no_prior_emergency_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| all_readmissions | 1499 | 1190 | 0.1393 | 4.0000 | 0.5197 | 0.8426 |
| false_negative | 1178 | 1038 | 0.1036 | 4.0000 | 0.6214 | 0.8582 |
| false_positive | 1038 | 755 | 0.2426 | 5.0000 | 0.1513 | 0.7187 |
| low_utilization_false_negative | 732 | 725 | 0.0917 | 4.0000 | 1.0000 | 0.9153 |
| low_utilization_true_positive | 47 | 47 | 0.2389 | 4.0000 | 1.0000 | 0.9787 |

## Why prior history matters—and what it misses

**779/1,499 (52.0%)** readmissions occur without recorded prior inpatient use. Only **47/779
(6.0%)** are captured, compared with **274/720 (38.1%)** with any prior use. The Phase 4
35.2%/88.2% figures used probability >=0.10 and flagged 41% of encounters; they are not
comparable budgets. The sharper gap here follows the chosen 10% capacity, not a changed model.

The 732 low-utilization misses are **62.1% of all false negatives**, from 725 historical
patients. Mean predicted risk is 9.17%; 91.5% also have no prior emergency visits.
Home destination code 1 accounts for 57.5%, skilled nursing code 3 for 23.4%, and home health
code 6 for 16.4%. Specialty is unknown for 51.4%; payer for 39.2% (payer is audit-only).
By contrast, **46 of the 47 captured low-utilization readmissions** have rehabilitation
destination code 22, with one skilled-nursing discharge. This is strong care-setting dependence;
it does not establish that rehabilitation causes risk or that transfer-related labels are preventable.

The frozen model already uses available age bands, admission context, length of stay,
specialty and destination. These can surface some low-utilization cases, but neither primary
logistic nor the fixed challenger solves this failure: the booster captures only 45 such
readmissions at the same capacity. Prior utilization is a strong historical signal, yet the
dataset cannot establish why an individual with little recorded history returns. Fragmented
outside-hospital history, acute severity and unmeasured social/support factors are possible
limitations, not demonstrated causes. Timing-sensitive diagnoses/labs/medications are not
promoted; no new feature or model change follows this analysis.

## Full comparison of error characteristics

False negatives are more commonly home discharges (60.7% versus 27.3% of false positives),
while false positives include more skilled nursing (33.2% versus 19.1%) and rehabilitation
(19.4% versus 0.17%). Short stays of 1–2 days account for 28.3% of false negatives versus
14.9% of false positives; stays of 8–14 days account for 15.5% versus 24.5%.
Age 70–89 accounts for 44.2% of false negatives and 50.4% of false positives. Emergency
admission type 1 is common in both (55.7%/58.0%); emergency-room source code 7 is more
common among false positives (68.3% versus 59.3%). These are proportions within error
cohorts, not age- or setting-specific error rates; the subgroup table supplies denominators.

The [complete characteristics table](phase5/validation/error_characteristics.csv) compares
all readmissions, false negatives, false positives and both low-utilization error strata
across the predeclared age, stay, destination, source/type and missingness groups.
Fractions below are within each error cohort; differences describe selection, not causal effects.
Demographics receive the separate minimum-size-controlled responsible-ML analysis.

| cohort | group | level | encounters | fraction |
| --- | --- | --- | --- | --- |
| all_readmissions | prior_inpatient | None | 779 | 0.5197 |
| all_readmissions | prior_inpatient | Any | 720 | 0.4803 |
| false_negative | prior_inpatient | None | 732 | 0.6214 |
| false_negative | prior_inpatient | Any | 446 | 0.3786 |
| false_positive | prior_inpatient | Any | 881 | 0.8487 |
| false_positive | prior_inpatient | None | 157 | 0.1513 |
| low_utilization_false_negative | prior_inpatient | None | 732 | 1.0000 |
| low_utilization_true_positive | prior_inpatient | None | 47 | 1.0000 |
| all_readmissions | prior_emergency | None | 1263 | 0.8426 |
| all_readmissions | prior_emergency | Any | 236 | 0.1574 |
| false_negative | prior_emergency | None | 1011 | 0.8582 |
| false_negative | prior_emergency | Any | 167 | 0.1418 |
| false_positive | prior_emergency | None | 746 | 0.7187 |
| false_positive | prior_emergency | Any | 292 | 0.2813 |
| low_utilization_false_negative | prior_emergency | None | 670 | 0.9153 |
| low_utilization_false_negative | prior_emergency | Any | 62 | 0.0847 |
| low_utilization_true_positive | prior_emergency | None | 46 | 0.9787 |
| low_utilization_true_positive | prior_emergency | Any | 1 | 0.0213 |
| all_readmissions | length_of_stay | 3–4 | 458 | 0.3055 |
| all_readmissions | length_of_stay | 1–2 | 395 | 0.2635 |
| all_readmissions | length_of_stay | 5–7 | 392 | 0.2615 |
| all_readmissions | length_of_stay | 8–14 | 254 | 0.1694 |
| false_negative | length_of_stay | 3–4 | 356 | 0.3022 |
| false_negative | length_of_stay | 1–2 | 333 | 0.2827 |
| false_negative | length_of_stay | 5–7 | 306 | 0.2598 |
| false_negative | length_of_stay | 8–14 | 183 | 0.1553 |
| false_positive | length_of_stay | 5–7 | 334 | 0.3218 |
| false_positive | length_of_stay | 3–4 | 295 | 0.2842 |
| false_positive | length_of_stay | 8–14 | 254 | 0.2447 |
| false_positive | length_of_stay | 1–2 | 155 | 0.1493 |
| low_utilization_false_negative | length_of_stay | 3–4 | 210 | 0.2869 |
| low_utilization_false_negative | length_of_stay | 1–2 | 208 | 0.2842 |
| low_utilization_false_negative | length_of_stay | 5–7 | 203 | 0.2773 |
| low_utilization_false_negative | length_of_stay | 8–14 | 111 | 0.1516 |
| low_utilization_true_positive | length_of_stay | 3–4 | 19 | 0.4043 |
| low_utilization_true_positive | length_of_stay | 8–14 | 12 | 0.2553 |
| low_utilization_true_positive | length_of_stay | 1–2 | 8 | 0.1702 |
| low_utilization_true_positive | length_of_stay | 5–7 | 8 | 0.1702 |
| all_readmissions | specialty_missing | Unknown | 776 | 0.5177 |
| all_readmissions | specialty_missing | Recorded | 723 | 0.4823 |
| false_negative | specialty_missing | Unknown | 600 | 0.5093 |
| false_negative | specialty_missing | Recorded | 578 | 0.4907 |
| false_positive | specialty_missing | Recorded | 533 | 0.5135 |
| false_positive | specialty_missing | Unknown | 505 | 0.4865 |
| low_utilization_false_negative | specialty_missing | Unknown | 376 | 0.5137 |
| low_utilization_false_negative | specialty_missing | Recorded | 356 | 0.4863 |
| low_utilization_true_positive | specialty_missing | Unknown | 27 | 0.5745 |
| low_utilization_true_positive | specialty_missing | Recorded | 20 | 0.4255 |
| all_readmissions | payer_missing | Recorded | 930 | 0.6204 |
| all_readmissions | payer_missing | Unknown | 569 | 0.3796 |
| false_negative | payer_missing | Recorded | 723 | 0.6138 |
| false_negative | payer_missing | Unknown | 455 | 0.3862 |
| false_positive | payer_missing | Recorded | 723 | 0.6965 |
| false_positive | payer_missing | Unknown | 315 | 0.3035 |
| low_utilization_false_negative | payer_missing | Recorded | 445 | 0.6079 |
| low_utilization_false_negative | payer_missing | Unknown | 287 | 0.3921 |
| low_utilization_true_positive | payer_missing | Recorded | 35 | 0.7447 |
| low_utilization_true_positive | payer_missing | Unknown | 12 | 0.2553 |
