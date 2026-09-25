# Frozen-model explanations: validation only

Prior inpatient use and discharge destination dominate logistic and the fixed boosting
challenger. Global mean absolute logistic SHAP contributions are 0.237 and 0.169 log odds;
age, admission source and specialty follow at about 0.064–0.071. Importance summarizes
variation in this validation population, not a causal effect or a clinical intervention target.

## Coefficients in defensible units

Numeric beta is divided by the training StandardScaler scale before exponentiation.
These odds ratios describe one original-unit increase, with other modeled fields fixed;
they do not multiply risk directly. A one-visit inpatient increment multiplies modeled
odds by 1.314; an extra hospital day by 1.013. Linear slopes are imposed by the frozen model.

| feature | coefficient | odds_ratio | training_scale |
| --- | --- | --- | --- |
| time_in_hospital | 0.0133 | 1.0134 | 2.9209 |
| number_inpatient | 0.2732 | 1.3142 | 1.2879 |
| number_emergency | 0.0408 | 1.0416 | 0.9976 |
| number_outpatient | -0.0031 | 0.9969 | 1.3223 |

Six categorical families are fully one-hot encoded after training-only rare pooling.
Individual dummy coefficients have no omitted-category baseline. The
[complete contrast table](phase5/validation/coefficient_contrasts.csv) subtracts the
declared reference coefficient **within each family** before calculating odds ratios:
age [50–60), Female, admission type 1, source 7, destination 1, InternalMedicine.
Pooled levels share coefficients; unseen levels use the original encoder behavior.
Do not compare an unscaled category coefficient with a standardized count coefficient.

## Technically checked SHAP

The [LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)
uses 128 deterministic training background rows and independent masking. Its 77 encoded
column contributions are summed into ten original feature families, preserving additivity.
All 13,590 validation records are explained in log-odds units; expit(base + sum) reproduces
the cached logistic probabilities. Maximum logit additivity error:
1.1e-15.

The installed [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
adapter did **not** reproduce this native-categorical histogram booster's actual decision
function: maximum error 2.539
log odds on 32 checked validation rows. Its adapter omits the estimator's categorical split
bitsets/internal representation. Passing the adapter's internal additivity check would not
validate these scores. The model is unchanged. We use permutation SHAP of the **actual**
categorical decision function, with the same training background, 256 seeded validation
rows and eight forward/reverse permutation pairs per row. Its maximum actual-output error
is 8e-15. Approximate attribution still
has sampling/background sensitivity; exact output additivity does not remove that uncertainty.

Independent masking can combine correlated inputs unrealistically. These are background-relative
predictive explanations, not causal effects or conditional clinical reasoning. The background
base is mean logit, not logit of mean risk. Per-encounter arrays and linkage IDs stay ignored.

| feature | logistic_mean_absolute_log_odds | logistic_matched_mean_absolute_log_odds | boosting_matched_mean_absolute_log_odds |
| --- | --- | --- | --- |
| time_in_hospital | 0.0303 | 0.0280 | 0.0408 |
| number_inpatient | 0.2369 | 0.2186 | 0.2863 |
| number_emergency | 0.0132 | 0.0141 | 0.0315 |
| number_outpatient | 0.0021 | 0.0023 | 0.0037 |
| age | 0.0705 | 0.0718 | 0.0588 |
| gender | 0.0305 | 0.0305 | 0.0109 |
| admission_type_id | 0.0293 | 0.0287 | 0.0087 |
| admission_source_id | 0.0643 | 0.0604 | 0.0323 |
| discharge_disposition_id | 0.1693 | 0.1566 | 0.1574 |
| medical_specialty | 0.0640 | 0.0604 | 0.0642 |

## Stability and disagreement

The matched 256-row comparison prevents unequal explanation samples from driving the model
comparison. Prior inpatient visits, emergency visits and length of stay have positive logistic
slopes and positive rank association with SHAP in both models. Raw outpatient visits have a
small negative logistic slope but a positive boosting rank association; both assign them
very little global importance. This disagreement is retained, not presented as a consistent
protective or harmful relationship. Categories have no meaningful ordinal direction, so
we compare their grouped importance rather than invent a rank correlation for category codes.

| feature | logistic_coefficient_per_unit | logistic_shap_rank_association | boosting_shap_rank_association |
| --- | --- | --- | --- |
| time_in_hospital | 0.0133 | 1.0000 | 0.9409 |
| number_inpatient | 0.2732 | 1.0000 | 0.8367 |
| number_emergency | 0.0408 | 1.0000 | 0.5359 |
| number_outpatient | -0.0031 | -1.0000 | 0.3534 |

## Three anonymous, purposive cases

Each case is the validation encounter nearest its case-type median predicted risk; these
are illustrations, not a random patient sample. [All ten additive contributions](phase5/validation/individual_explanations.csv)
are exported without patient/encounter IDs or source-row profiles.

* Targeted readmission, risk **24.3%**: destination contributes +1.203 log odds; no prior
  inpatient use contributes −0.203. A discharge-setting association dominates this score.
* Targeted non-readmission, risk **21.6%**: inpatient use +0.890 and emergency use +0.238
  dominate; specialty and destination partially offset them. A false positive does not
  establish that outreach would have been unnecessary.
* Missed readmission without inpatient history, risk **8.75%**: prior inpatient and
  destination contributions are −0.203 and −0.115, offset partly by age and admission source.
  A low predicted risk is not an assurance that readmission will not occur.

No test explanations or additional features were used to revise model selection.
