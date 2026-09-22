# Phase 3 validation baselines

All results below use the frozen **validation** partition. Test has not been evaluated. The scoring-time allowlist is fixed before comparison. Sensitivity models are ineligible for primary selection.

| name | average_precision | roc_auc | recall | precision | f1 | specificity | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| histogram_boosting | 0.202 | 0.662 | 0.005 | 0.538 | 0.009 | 1.000 | 0.094 |
| random_forest | 0.197 | 0.658 | 0.000 | 0.000 | 0.000 | 1.000 | 0.095 |
| logistic | 0.194 | 0.653 | 0.004 | 0.462 | 0.008 | 0.999 | 0.095 |
| logistic_balanced | 0.194 | 0.654 | 0.532 | 0.172 | 0.260 | 0.683 | 0.226 |
| naive_prevalence | 0.110 | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.098 |

Average precision (AP) is the primary metric: it summarizes precision/recall across thresholds and its constant-score reference equals validation prevalence. It is sklearn average_precision_score, not trapezoidal PR area. ROC-AUC is complementary; accuracy is not a selection metric. Precision/recall/F1/specificity and confusion counts use **0.50 only**, with no threshold search. Precision is reported as zero when there are no positive predictions. No SMOTE, calibration fitting or hyperparameter search occurred.

| name | tn | fp | fn | tp | threshold | mean_probability | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| histogram_boosting | 12085 | 6 | 1492 | 7 | 0.500 | 0.109 | 0.110 |
| random_forest | 12091 | 0 | 1499 | 0 | 0.500 | 0.110 | 0.110 |
| logistic | 12084 | 7 | 1493 | 6 | 0.500 | 0.109 | 0.110 |
| logistic_balanced | 8256 | 3835 | 702 | 797 | 0.500 | 0.465 | 0.110 |
| naive_prevalence | 12091 | 0 | 1499 | 0 | 0.500 | 0.111 | 0.110 |

## Controlled comparisons

| comparison | model | reference | ap_difference | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| Logistic vs prevalence | logistic | naive_prevalence | 0.084 | 0.070 | 0.101 |
| All prior utilization | logistic | logistic_no_utilization | 0.039 | 0.027 | 0.055 |
| Add total and any-use indicators | logistic | logistic_raw_utilization | -0.002 | -0.005 | 0.000 |
| Discharge destination | logistic | logistic_no_disposition | 0.014 | 0.008 | 0.021 |
| Balanced class weights | logistic_balanced | logistic | -0.000 | -0.001 | 0.001 |
| Random forest vs logistic | random_forest | logistic | 0.003 | -0.003 | 0.011 |
| Boosting vs logistic | histogram_boosting | logistic | 0.008 | 0.002 | 0.014 |
| Add grouped diagnoses (uncertain timing) | logistic_diagnosis_groups | logistic | 0.002 | -0.002 | 0.005 |
| Raw vs grouped diagnosis representation | logistic_diagnosis_raw | logistic_diagnosis_groups | 0.002 | -0.006 | 0.010 |
| Add encounter summaries (uncertain timing) | logistic_encounter_summaries | logistic | 0.000 | -0.002 | 0.003 |

Intervals are paired percentile intervals from 300 patient bootstrap resamples, seed 42. They describe validation sampling variation, not external transportability or uncertainty from retraining. Multiple comparisons are exploratory; narrow numerical rankings are not proof of a final winner.

## Timing-sensitive diagnostic comparisons

| name | average_precision | roc_auc | recall | precision | f1 | specificity | brier | encoded_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_diagnosis_raw | 0.198 | 0.657 | 0.009 | 0.412 | 0.018 | 0.998 | 0.095 | 382 |
| logistic_diagnosis_groups | 0.196 | 0.658 | 0.004 | 0.300 | 0.008 | 0.999 | 0.095 | 118 |
| logistic_encounter_summaries | 0.195 | 0.653 | 0.005 | 0.350 | 0.009 | 0.999 | 0.095 | 135 |

Raw diagnosis codes achieve 0.198 AP versus 0.196 for broad groups, but the paired difference interval includes zero (−0.006 to +0.010). Raw encoding has 382 columns versus 118 with grouping. Grouping is compact; superiority is not established. Neither representation is promoted without an as-of-scoring diagnosis audit. The encounter-summary sensitivity adds only 0.0004 AP (interval −0.0024 to +0.0034); it provides no current reason to broaden the timing-sensitive allowlist.

## Decisions supported by these results

- Logistic regression improves AP by 0.084 over the prevalence baseline (1.76× AP).
- Prior utilization adds 0.039 AP (95% paired interval 0.027–0.055): the Phase 2 association contributes predictive value.
- Total-plus-any-use additions reduce AP by 0.002 versus retaining raw counts alone (interval −0.0052 to +0.0001). Do not retain that package by default in Phase 4; this joint ablation does not identify which individual derived term is responsible. The total is also exactly redundant with the three counts in a linear representation, although it can change regularization.
- Disposition adds 0.014 AP (0.008–0.021), conditional on the confirmed-discharge contract. Predictiveness does not establish arrival time.
- Histogram boosting improves AP by 0.0079 over the matched logistic baseline (0.0015–0.0143): a modest development gain. Random forest adds 0.0032 (−0.0034 to +0.0106), which is inconclusive.
- Balanced logistic weights leave AP essentially unchanged (−0.0004), increase recall from 0.004 to 0.532 and lower precision from 0.462 to 0.172 at the same arbitrary threshold. Brier worsens from 0.095 to 0.226. This does not establish superiority over an unweighted model at a different threshold, which was not tested.

## Initial probability and error checks

Validation prevalence is 11.03%. Mean predicted risk is 10.92% for boosting and 10.91% for unweighted logistic, but 46.53% for balanced logistic. The unweighted models have reasonable average probability levels; that alone does not prove calibration. Quantile curves reveal local deviations and have no uncertainty bands. Balanced probabilities markedly overstate absolute risk. Brier combines calibration and discrimination; no calibrator has been fit.

The boosting model at 0.50 produces only 13 positive predictions: 7 true positives and 6 false positives, with 1,492 false negatives. All 13 predictions belong to patients with 3+ provided prior inpatient visits; 219 of 226 positives in that group are still missed. Five false positives are home discharges and one is home-health; six total errors cannot establish a stable destination-specific pattern. Specialty-unknown encounters have 773 false negatives out of 776 positives versus 719/723 when observed. Payer-unknown misses are 566/569 versus 926/930 observed. At this threshold almost everyone is negative, so these counts are mainly evidence that 0.50 is not an operational outreach threshold, not evidence of a specific missingness mechanism or fairness. Risk means rise from 8.37% at no prior inpatient visits to 24.76% at 3+. Do not confuse these provided history fields with future repeat status.

## Scope and reproducibility

Eleven prespecified experiments: five baseline/primary runs, three core ablations and three timing-sensitive runs. Settings and seed live in configs/phase3.yaml; no validation-driven parameter search. Logistic uses C=1 and convergence checks; forest uses 200 trees/depth 12/min leaf 20; boosting uses 150 iterations/15 leaves/learning rate 0.05. Boosting early stopping is disabled to avoid an implicit encounter-level split. All vocabularies, rare pooling, scaling and estimators fit training only.

Use lightweight YAML + CSV/JSON tracking for this finite registry; MLflow is deferred until experiment volume warrants it. Development joblib pipelines and individual validation predictions are ignored by Git. No final model is declared. The package already supplies the three model families, so XGBoost/LightGBM/CatBoost dependencies were not added merely to fill a roster. No model family is assumed promising without observed evidence.

Limitations: full-cohort Phase 2 EDA indirectly viewed future test outcomes; historical selected diabetes encounters; incomplete out-of-network follow-up; no dates/hospital holdout; retrospective source timing; one repeatedly consulted validation set; and unadjusted subgroup error summaries. Final test remains isolated from subsequent development.

Implementation references: [sklearn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) and [one-hot encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
