# Phase 4 development model selection

All external metrics are **validation performance** on the frozen 13,590 encounters / 9,757 patients. Test has not been scored. Hyperparameters and feature configurations were selected from training-only patient-grouped CV before validation was opened. See [the prespecified protocol](phase4_protocol.md).

**Primary: Logistic regression / uncalibrated. Challenger: Histogram boosting / uncalibrated.**

Boosting minus logistic AP is +0.00194, with 95% paired patient-bootstrap interval [-0.00427, +0.00839]. Brier difference is -0.000461 [-0.000840, -0.000116]. Negative Brier differences favor boosting.

The AP ranking difference is statistically difficult to distinguish on this validation set.

| Prespecified boosting selection condition | Passed |
| --- | --- |
| validation_ap_gain | False |
| paired_ap_interval_positive | False |
| brier_acceptable | True |
| training_cv_mean | True |
| training_cv_stability | True |

Boosting must improve validation AP by >=0.005 with an interval above zero, have no more than 0.0002 Brier disadvantage, and match/exceed logistic's CV mean with at most twice its fold SD. Otherwise the simpler logistic model is preferred. This is a development selection convention, not a clinical-utility estimate. Boosting's lower Brier and/or better fold stability remain reasons to retain a challenger, even when the AP evidence is insufficient to justify primary status.

| model | average_precision | roc_auc | brier | recall | precision | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic__uncalibrated | 0.196781 | 0.654593 | 0.095065 | 0.006671 | 0.322581 | 0.013072 | 0.998263 |
| boosting__uncalibrated | 0.198721 | 0.661488 | 0.094604 | 0.000667 | 1.000000 | 0.001333 | 1.000000 |
| forest__uncalibrated | 0.199690 | 0.660228 | 0.094797 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| phase3_logistic_raw | 0.196410 | 0.653037 | 0.095083 | 0.006671 | 0.312500 | 0.013063 | 0.998180 |
| phase3_boosting | 0.202180 | 0.662312 | 0.094497 | 0.004670 | 0.538462 | 0.009259 | 0.999504 |

All operating metrics above use 0.50 as a reference only. Confusion matrices:

| model | tn | fp | fn | tp |
| --- | --- | --- | --- | --- |
| logistic__uncalibrated | 12070 | 21 | 1489 | 10 |
| boosting__uncalibrated | 12091 | 0 | 1498 | 1 |
| forest__uncalibrated | 12091 | 0 | 1499 | 0 |
| phase3_logistic_raw | 12069 | 22 | 1489 | 10 |
| phase3_boosting | 12085 | 6 | 1492 | 7 |

Reference-model comparisons against the tuned logistic candidate:

| model | reference | ap_difference | brier_difference | ap_ci_low | ap_ci_high | brier_ci_low | brier_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase3_boosting | logistic__uncalibrated | 0.005399 | -0.000568 | -0.001160 | 0.012209 | -0.000954 | -0.000226 |
| forest__uncalibrated | logistic__uncalibrated | 0.002909 | -0.000268 | -0.003897 | 0.011559 | -0.000730 | 0.000213 |

The Phase 3 booster remains an important historical benchmark; Phase 4 does not claim that tuning improved it. The fixed forest is not promoted based on a small AP difference. The primary/challenger tracks follow the declared training/validation rule rather than a post-hoc hyperparameter search against these outcomes.

## Complexity and reproducibility

| model | configuration | encoded_features | fit_seconds | inference_batch_rows | inference_median_ms | artifact_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| logistic__uncalibrated | raw | 77.000000 | 0.367068 | 1024.000000 | 8.399167 | 4127.000000 |
| boosting__uncalibrated | raw | 10.000000 | 0.515198 | 1024.000000 | 11.193709 | 58489.000000 |
| forest__uncalibrated | raw | 77.000000 | 3.484426 | 1024.000000 | 39.857833 | 2252031.000000 |

Inference timings are medians of seven warmed batches of 1,024 training feature rows with two numerical threads on this environment; they are not production latency SLAs. Fit time excludes the search and includes calibration where applicable. Logistic has direct coefficients but still requires noncausal interpretation; boosted/forest trees are more complex. All models exclude IDs/race/outcomes, fit rare categories inside training folds, and use the same confirmed-discharge contract. The original frozen split guard remains strict. Model computations may vary slightly across hardware.

## Exploratory threshold behavior

| model | threshold | recall | precision | f1 | specificity | encounters_flagged | encounter_fraction_flagged | patients_flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic__uncalibrated | 0.050000 | 0.995330 | 0.111535 | 0.200592 | 0.017037 | 13377 | 0.984327 | 9577 |
| logistic__uncalibrated | 0.100000 | 0.606404 | 0.163079 | 0.257034 | 0.614176 | 5574 | 0.410155 | 3608 |
| logistic__uncalibrated | 0.150000 | 0.267512 | 0.219967 | 0.241421 | 0.882392 | 1823 | 0.134143 | 1144 |
| logistic__uncalibrated | 0.200000 | 0.162108 | 0.268212 | 0.202079 | 0.945166 | 906 | 0.066667 | 588 |
| logistic__uncalibrated | 0.300000 | 0.056704 | 0.335968 | 0.097032 | 0.986105 | 253 | 0.018617 | 142 |
| logistic__uncalibrated | 0.500000 | 0.006671 | 0.322581 | 0.013072 | 0.998263 | 31 | 0.002281 | 14 |
| boosting__uncalibrated | 0.050000 | 0.992662 | 0.113242 | 0.203293 | 0.036308 | 13140 | 0.966887 | 9369 |
| boosting__uncalibrated | 0.100000 | 0.657105 | 0.160711 | 0.258259 | 0.574560 | 6129 | 0.450993 | 3946 |
| boosting__uncalibrated | 0.150000 | 0.346231 | 0.209021 | 0.260673 | 0.837565 | 2483 | 0.182708 | 1552 |
| boosting__uncalibrated | 0.200000 | 0.168112 | 0.268943 | 0.206897 | 0.943346 | 937 | 0.068948 | 567 |
| boosting__uncalibrated | 0.300000 | 0.052035 | 0.310757 | 0.089143 | 0.985692 | 251 | 0.018469 | 172 |
| boosting__uncalibrated | 0.500000 | 0.000667 | 1.000000 | 0.001333 | 1.000000 | 1 | 0.000074 | 1 |

The complete 0.00–1.00 grid is diagnostic. No threshold is selected and no capacity, lift/gains or ROI optimization is performed. Unique patients flagged counts anyone with a flagged validation encounter across the historical extract; without dates this is not a simultaneous outreach population.

## Initial subgroup and demographic diagnostics

| group | level | encounters | positives | average_precision | recall | precision | false_negative_rate | estimates_suppressed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior_inpatient | none | 9134 | 779 | 0.137422 | 0.351733 | 0.141237 | 0.648267 | False |
| prior_inpatient | any | 4456 | 720 | 0.241796 | 0.881944 | 0.174739 | 0.118056 | False |
| prior_inpatient | 3+ | 886 | 226 | 0.327048 | 1.000000 | 0.255079 | 0.000000 | False |
| race | AfricanAmerican | 2599 | 254 | 0.146799 | 0.535433 | 0.134520 | 0.464567 | False |
| race | Asian | 100 | 8 | — | — | — | — | True |
| race | Caucasian | 10062 | 1154 | 0.211413 | 0.624783 | 0.169448 | 0.375217 | False |
| race | Hispanic | 320 | 36 | 0.230062 | 0.666667 | 0.196721 | 0.333333 | False |
| race | Other | 193 | 20 | — | — | — | — | True |
| race | Unknown | 316 | 27 | — | — | — | — | True |
| gender | Female | 7376 | 816 | 0.190069 | 0.606618 | 0.160976 | 0.393382 | False |
| gender | Male | 6214 | 683 | 0.206547 | 0.606149 | 0.165666 | 0.393851 | False |
| age | [0-10) | 30 | 0 | — | — | — | — | True |
| age | [10-20) | 130 | 7 | — | — | — | — | True |
| age | [20-30) | 227 | 23 | — | — | — | — | True |
| age | [30-40) | 488 | 44 | 0.214673 | 0.613636 | 0.184932 | 0.386364 | False |
| age | [40-50) | 1374 | 165 | 0.260819 | 0.600000 | 0.245050 | 0.400000 | False |
| age | [50-60) | 2334 | 223 | 0.211545 | 0.470852 | 0.200765 | 0.529148 | False |
| age | [60-70) | 2977 | 336 | 0.225377 | 0.598214 | 0.171795 | 0.401786 | False |
| age | [70-80) | 3412 | 379 | 0.175762 | 0.643799 | 0.145152 | 0.356201 | False |
| age | [80-90) | 2213 | 275 | 0.169494 | 0.687273 | 0.141573 | 0.312727 | False |
| age | [90-100) | 405 | 47 | 0.133031 | 0.595745 | 0.129032 | 0.404255 | False |

Diagnostics use the prespecified illustrative threshold 0.10, not a recommended operating point. None/any/3+ prior inpatient groups overlap intentionally; they use only provided prior-year history. No single/repeat-patient feature is built from future appearances. Rates are suppressed below 200 encounters or 30 positives/negatives. Unknown race/gender remain visible, and counts are retained even where estimates are suppressed. Passing a sample-size cutoff does not establish precision: small retained groups can still be unstable. These unadjusted summaries have no subgroup uncertainty bands and do not establish fairness, discrimination or clinical causality.

## Limits and artifacts

The 1,000 bootstrap draws resample whole validation patients, keeping all their encounters together and pairing every model on the same weights. These intervals exclude retraining/selection uncertainty and are exploratory across multiple comparisons. Validation has now informed feature development (Phase 3), calibration choice and development selection; it is not a final unbiased performance estimate. Prior full-cohort EDA indirectly exposed eventual test outcomes. No temporal or hospital holdout can be constructed from this extract. Contemporary availability, outcome ascertainment and external validation remain unresolved.

The selected development artifact is `models/development/phase4/logistic__uncalibrated.joblib`; challenger is `boosting__uncalibrated.joblib`. Models and individual predictions remain ignored. There is no production artifact and no final test result.
