# Phase 4 tuning report

All external metrics are **validation performance** on the frozen 13,590 encounters / 9,757 patients. Test has not been scored. Hyperparameters and feature configurations were selected from training-only patient-grouped CV before validation was opened. See [the prespecified protocol](phase4_protocol.md).

## Group-aware CV

Five StratifiedGroupKFold folds, shuffle enabled, seed 42, stable encounter ordering. Each training encounter appears in exactly one holdout; every patient's encounters remain together. All fit/holdout patient overlaps are zero. No seed search occurred.

| fold | role | encounters | patients | positives | prevalence | patient_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | holdout | 12713 | 9113 | 1414 | 0.111225 | 0 |
| 1 | holdout | 12712 | 9117 | 1414 | 0.111233 | 0 |
| 2 | holdout | 12713 | 9152 | 1413 | 0.111146 | 0 |
| 3 | holdout | 12712 | 9085 | 1413 | 0.111155 | 0 |
| 4 | holdout | 12713 | 9063 | 1414 | 0.111225 | 0 |

AP is primary because the positive prevalence is about 11%; the constant-score AP baseline equals prevalence. ROC-AUC and probability error provide complementary evidence. CV SD describes fold variability; overlapping training folds are not independent replicates. Feature selection and tuning share folds, so selected CV scores are optimistic and are not nested-CV performance estimates.

## Search space and selected parameters

Unweighted L2 logistic: C in [0.001,100] on a log scale, lbfgs, maximum 2,000 iterations. Histogram boosting: learning rate [0.025,0.15] on a log scale, iterations 75/125/150/200/300, leaves 7/15/31, minimum leaf 20/30/50/100/200, L2 0/0.1/1/10/30. Depth is unset, max_bins=255, and internal early stopping is disabled. Both include baseline parameters in trial 0. Sequential TPE uses seed 42 and two numerical threads. No SMOTE, class-weight search, L1/elastic-net search or forest tuning.

**Logistic regression:** 15 trials; best trial 12; stopped by prespecified training-CV plateau.

```json
{
  "C": 0.09988151348099303
}
```

**Histogram boosting:** 30 trials; best trial 14; stopped by prespecified training-CV plateau.

```json
{
  "learning_rate": 0.04938583890481199,
  "max_iter": 75,
  "max_leaf_nodes": 15,
  "min_samples_leaf": 50,
  "l2_regularization": 30.0
}
```

Maximum budgets were 30 logistic and 50 boosting trials. The prespecified plateau rule checks improvements of at least 0.0002 after 15/25 minimum trials and 10/15 trials of patience. Random forest is one fixed 200-tree/depth-12/min-leaf-20 reference.

| Family | Features | CV AP mean | CV AP SD | Fold AP min | Fold AP max | Validation AP | ROC-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | raw | 0.212434 | 0.007680 | 0.205121 | 0.225514 | 0.196781 | 0.654593 | 0.095065 |
| Histogram boosting | raw | 0.220207 | 0.005272 | 0.212383 | 0.227111 | 0.198721 | 0.661488 | 0.094604 |
| Forest reference | raw | 0.217605 | 0.005375 | 0.211854 | 0.223439 | 0.199690 | 0.660228 | 0.094797 |

## Controlled feature comparison

| family | configuration | cv_average_precision_mean | cv_average_precision_sd | eligible |
| --- | --- | --- | --- | --- |
| logistic | raw | 0.212191 | 0.008265 | True |
| logistic | indicators | 0.182266 | 0.006331 | True |
| logistic | raw_engineered | 0.211126 | 0.007796 | True |
| logistic | diagnosis_grouped | 0.212213 | 0.007699 | False |
| logistic | diagnosis_raw | 0.213028 | 0.006965 | False |
| boosting | raw | 0.218772 | 0.004638 | True |
| boosting | indicators | 0.192949 | 0.003775 | True |
| boosting | raw_engineered | 0.219738 | 0.004828 | True |
| boosting | diagnosis_grouped | 0.224065 | 0.006758 | False |
| boosting | diagnosis_raw | 0.212723 | 0.007984 | False |

Matched fold differences (descriptive, not confidence intervals):

| family | configuration | reference | mean_ap_difference | sd_ap_difference | min_ap_difference | max_ap_difference |
| --- | --- | --- | --- | --- | --- | --- |
| boosting | raw_engineered | raw | 0.000967 | 0.001259 | -0.000665 | 0.002598 |
| boosting | indicators | raw | -0.025822 | 0.005324 | -0.030750 | -0.017357 |
| boosting | diagnosis_grouped | raw | 0.005293 | 0.002833 | 0.000769 | 0.008346 |
| boosting | diagnosis_raw | raw | -0.006048 | 0.006000 | -0.011701 | 0.001924 |
| boosting | diagnosis_raw | diagnosis_grouped | -0.011342 | 0.004271 | -0.016417 | -0.004887 |
| logistic | raw_engineered | raw | -0.001065 | 0.001407 | -0.002710 | 0.000646 |
| logistic | indicators | raw | -0.029926 | 0.005735 | -0.036404 | -0.023739 |
| logistic | diagnosis_grouped | raw | 0.000022 | 0.001393 | -0.001957 | 0.001226 |
| logistic | diagnosis_raw | raw | 0.000836 | 0.006490 | -0.007779 | 0.008184 |
| logistic | diagnosis_raw | diagnosis_grouped | 0.000815 | 0.006170 | -0.006858 | 0.007427 |

Encoded dimensions across training folds:

| family | configuration | min_encoded_features | max_encoded_features |
| --- | --- | --- | --- |
| boosting | diagnosis_grouped | 14 | 14 |
| boosting | diagnosis_raw | 14 | 14 |
| boosting | indicators | 10 | 10 |
| boosting | raw | 10 | 10 |
| boosting | raw_engineered | 14 | 14 |
| logistic | diagnosis_grouped | 113 | 114 |
| logistic | diagnosis_raw | 339 | 345 |
| logistic | indicators | 76 | 77 |
| logistic | raw | 76 | 77 |
| logistic | raw_engineered | 80 | 81 |

Logistic uses one-hot levels, while boosting retains ordinal-encoded categorical columns; their dimension counts are not directly comparable measures of capacity. Raw diagnosis categories are pooled using training-only rarity rules, so the comparison is bounded and manageable. Grouping reduces logistic dimensionality but is not universally more predictive.

- Logistic regression selected **raw**. Raw + engineered minus raw AP: -0.00107; indicators-only minus raw: -0.02993. Tuning on the selected configuration changed CV AP by +0.00024.
- Histogram boosting selected **raw**. Raw + engineered minus raw AP: +0.00097; indicators-only minus raw: -0.02582. Tuning on the selected configuration changed CV AP by +0.00144.

The 0.001 simplicity margin was fixed before fitting. Small engineered gains within that margin do not justify additional inputs. Keep raw counts instead of automatically discarding their magnitude or retaining redundant derivations.

Diagnosis experiments are timing-restricted sensitivities, not contenders. Grouped diagnoses can help boosting in these folds while raw codes need not; logistic differences are small. Their source arrival time is unverified, and the comparison does not authorize promoting them. No diagnosis candidate was tuned or scored on validation in Phase 4.

## Did tuning improve external development performance?

| model | reference | ap_difference | brier_difference | ap_ci_low | ap_ci_high | brier_ci_low | brier_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic__uncalibrated | phase3_logistic_raw | 0.000370 | -0.000018 | -0.000261 | 0.001009 | -0.000053 | 0.000017 |
| boosting__uncalibrated | phase3_boosting | -0.003459 | 0.000107 | -0.007771 | 0.000902 | -0.000083 | 0.000292 |

The logistic reference uses Phase 3 raw utilization. The boosting reference uses Phase 3 raw + engineered utilization, so that validation comparison combines feature selection and tuning; it does not isolate hyperparameters. The CV comparison above holds the selected configuration fixed. A CV gain does not guarantee a validation gain. The candidate set was not retuned after these results.

CV trial/fold metrics, parameters and durations are in [optuna_trials.csv](phase4/optuna_trials.csv) and [tuning_fold_metrics.csv](phase4/tuning_fold_metrics.csv). Local MLflow stores run provenance in ignored mlruns/phase4/tracking.sqlite. Individual predictions and study/model objects are ignored development artifacts.
