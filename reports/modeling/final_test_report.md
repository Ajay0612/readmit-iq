# One-time frozen final test evaluation

The evaluation used the committed [protocol](final_evaluation_protocol.md), freeze commit
`ce2fcdb5c79256f088e82af45d97bb6d51a0360d`, and code commit `5714a2961f1346f7b505f439e6da2e4298b9b41c`.
The original test table was parsed once; each prespecified pipeline predicted once. All
subsequent statistics/figures use saved predictions or aggregate outputs. Models, features,
calibration, top-10% capacity and subgroup definitions were not changed after scoring.
The primary remains uncalibrated logistic; the challenger is descriptive, not a replacement.

## Generalization and uncertainty

Test performance broadly aligns with development: logistic AP is 0.20837 versus validation
0.19678, ROC-AUC 0.65181 versus 0.65459, and Brier 0.094101 versus 0.095065. The modest AP
increase is compatible with the reported sampling uncertainty; it is not evidence of a
new intervention benefit. Policy recall increases from 21.41% to 23.54% and precision from
23.62% to 25.85%, with the original policy unchanged.

| partition | AP | ROC_AUC | Brier |
| --- | --- | --- | --- |
| Training CV (selected; not nested) | 0.2124 | — | — |
| Validation | 0.1968 | 0.6546 | 0.0951 |
| Final test | 0.2084 | 0.6518 | 0.0941 |

Training CV AP SD was 0.007680; those reused selection folds do not provide an unbiased
final performance estimate. Compare test with validation descriptively, without choosing a
new model from their ordering. Patient-bootstrap intervals below condition on the fitted
model; 1,000 paired whole-patient draws, seed 42, rerank the fixed capacity in each draw.

| metric | ci_low | ci_high | valid_replicates |
| --- | --- | --- | --- |
| average_precision | 0.1870 | 0.2328 | 1000 |
| roc_auc | 0.6351 | 0.6693 | 1000 |
| brier | 0.0898 | 0.0983 | 1000 |
| recall | 0.2129 | 0.2591 | 1000 |
| precision | 0.2281 | 0.2889 | 1000 |
| lift | 2.1298 | 2.5920 | 1000 |

| model | average_precision | roc_auc | brier | recall | precision | specificity | tp | fp | fn | tn | lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic__uncalibrated | 0.2084 | 0.6518 | 0.0941 | 0.2354 | 0.2585 | 0.9168 | 350 | 1004 | 1137 | 11058 | 2.3553 |
| boosting__uncalibrated | 0.2131 | 0.6592 | 0.0937 | 0.2313 | 0.2541 | 0.9163 | 344 | 1010 | 1143 | 11052 | 2.3149 |

The [paired challenger comparisons](phase5/test/paired_comparisons.csv) use the same patient
draws. Boosting's AP advantage is +0.00474 (95% paired interval −0.00211 to +0.01101).
Its ROC-AUC and Brier are better, but at the selected capacity it captures 344 readmissions
versus logistic's 350; the capacity-recall difference is uncertain. These secondary results
do not reopen the prespecified primary selection. The uncertainty estimates exclude
 temporal/site shift, retraining and selection.
Test was isolated from model fitting/tuning; earlier full-cohort EDA means it was not fully
unseen in the broader analysis. No equivalence or clinical utility conclusion follows.

## Frozen operational result

Of 13,549 test encounters from 9,757 patients, the fixed
policy targets **1,354 encounters / 863
unique historical people**, capturing **350/1487 readmissions
(23.54%)**, with **25.85% precision** and **2.355× lift**.
Specificity is 91.68%; false positives 1,004, false negatives
1,137. The observed cutoff is 0.166509110; no probability cutoff
was reoptimized. Floor rounding targets 9.99336% of the partition.

| capacity | cutoff | targeted_encounters | targeted_patients | tp | fn | recall | precision | lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0500 | 0.2191 | 677 | 467 | 205 | 1282 | 0.1379 | 0.3028 | 2.7591 |
| 0.1000 | 0.1665 | 1354 | 863 | 350 | 1137 | 0.2354 | 0.2585 | 2.3553 |
| 0.1500 | 0.1420 | 2032 | 1304 | 452 | 1035 | 0.3040 | 0.2224 | 2.0268 |
| 0.2000 | 0.1303 | 2709 | 1765 | 560 | 927 | 0.3766 | 0.2067 | 1.8835 |
| 0.2500 | 0.1219 | 3387 | 2233 | 646 | 841 | 0.4344 | 0.1907 | 1.7379 |

At the observed test mix, per 10,000 eligible discharges, about
**999.3 outreach encounters** surface
**258.3** readmissions versus
**109.7** under random targeting:
**148.6 additional surfaced events**.
Expected recorded readmissions: 1097.5.
Contacts per surfaced event: 3.87. This scales observed
encounter-level yields, not unique people, prevented readmissions or savings.

## Predeclared failure and subgroup diagnostics

No prior inpatient history accounts for 748/1487
readmissions (50.3%). Recall is
**7.6% without prior use** versus
**39.6% with prior use**. The low-utilization concern persists.
The 691 missed low-utilization events are 60.8% of all 1,137 false negatives, involving
688 historical patients. Of these misses, 60.8% are discharged home, 22.1% to skilled nursing
and 15.6% with home health; 50.1% have unknown specialty and 35.5% unknown payer.
Among the 57 captured low-utilization readmissions, 56 have rehabilitation destination 22.
This reproduces the validation concern about destination and history dependence. Recall
for 3+ prior inpatient visits is 87.5%, compared with 87.6% on validation.
These groups use the original cuts and suppression rules; no new test-driven features or
thresholds follow. [Error profiles](phase5/test/error_profiles.csv) and
[all error characteristics](phase5/test/error_characteristics.csv) report destinations,
age, stay, admission context and missingness. The
[responsible ML report](../responsible_ml.md) applies identical demographic and probability-bias checks.

Test female/male recall is 23.6%/23.5%. African American/Caucasian recall is 23.5%/23.9%,
so the larger validation racial recall gap does not reproduce descriptively. Cluster
intervals remain wide; this is not a fairness finding. Hispanic test performance is
suppressed (14 positive labels), as are Asian, Other and Unknown race performance estimates.
No supported test group meets the prespecified material mean-bias criterion. Prior-use
mean risk is 1.21 percentage points below observed prevalence (CI −2.30 to −0.11), while
no-prior-use mean risk is 0.58 points above (CI 0.02 to 1.14); neither reaches the declared
2-point magnitude criterion. Do not interpret the absence of flags as proof of calibration.

The highest-risk decile has 350 events and 25.85% observed risk; the lowest has 82 events
and 6.05% risk. Top two deciles capture 37.66% of events. Lower decile rates are not strictly
monotonic, and the ranking does not separate a clinically safe low-risk group.

## Artifacts and repeatability

The frozen train-only pipeline is copied byte-for-byte to ignored
`models/final/readmit_iq_logistic.joblib`; no train+validation refit occurs. The
[metadata](final_model_metadata.json) binds the feature schema, versions, training/freeze
commits, binary hash, policy and validation/test results. Individual predictions, raw data
and model binaries are not committed. [Publication hashes](phase5/test/publication_manifest.json)
protect every final aggregate table. Notebook 06 and CI verify these archived results,
without repeating the real prediction pass. A fresh checkout refuses `make final-eval` once
published results exist; synthetic tests demonstrate repeatability without accessing real test.

Current clinical use remains unsupported for the specific reasons documented in the feature
policy and responsible-ML report. Phase 6 may package a local demonstration only after
separate authorization; no serving, dashboard or deployment was added in Phase 5.
