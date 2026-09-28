# Discussing ReadmitIQ in an interview

Use the results as retrospective project evidence. The capacity assumption is illustrative;
no hospital deployment, prevented readmissions or savings were demonstrated.

## 30-second explanation

“I built ReadmitIQ to study how a care-management team could prioritize limited discharge
follow-up. I analyzed 90,702 eligible encounters using patient-grouped validation, compared
model families and selected a simple logistic model using performance and uncertainty.
On the held-out test, targeting the highest-risk 10% surfaced 23.54% of recorded readmissions,
or 2.36 times random targeting. I then packaged the frozen model in a tested FastAPI,
Streamlit and Docker demo, while documenting its historical-data and low-utilization limitations.”

## 2-minute walkthrough

**Problem, about 20 seconds.** A risk score needs an operational decision. I assumed capacity
to review 10% of eligible discharge encounters and asked how much recorded readmission burden
that queue would contain. The assumption is a portfolio scenario, not measured hospital staffing.

**Approach, about 35 seconds.** I validated the public UCI diabetes dataset, defined discharge
eligibility, audited feature availability and separated patients across train, validation and
test. Preprocessing stayed inside patient-disjoint training folds. I compared logistic,
forest and histogram boosting, then evaluated uncertainty and calibration before freezing
the primary model and outreach rule.

**Result, about 35 seconds.** Logistic's held-out AP was 0.20837 and ROC-AUC 0.65181. The top
10% had 25.85% precision and captured 23.54% of events. At a hypothetical 10,000 discharges,
that scales to about 258 surfaced readmissions versus 110 under random targeting. This is
an estimate of queue yield; it does not tell us how many readmissions outreach would prevent.

**Judgment and engineering, about 30 seconds.** Boosting's validation advantage was small and
uncertain, so I retained logistic for interpretability. Error analysis showed only 7.6% recall
without prior inpatient use, versus 39.6% with it. I made that limitation prominent and built
a strict, hash-verified local API with a synthetic Streamlit walkthrough—not a clinical system.

## 5-minute technical walkthrough

| Time | Show | Explain |
|---|---|---|
| 0:00–0:40 | README result and gains chart | Define capacity, recall, precision and retrospective yield. |
| 0:40–1:25 | [Dataset/feature audit](../reports/data_quality/feature_audit.md) | Selected diabetes population; `<30` target; missingness, exclusions and confirmed-discharge timing. |
| 1:25–2:10 | [Split report](../reports/modeling/data_split_report.md) | Patient grouping, zero overlap, frozen hashes, preprocessing within CV folds; earlier EDA limits. |
| 2:10–3:00 | [Selection](../reports/modeling/model_selection_report.md) and [calibration](../reports/modeling/calibration_report.md) | Compare uncertainty, probability quality and complexity; distinguish validation from final test. |
| 3:00–3:45 | [Final report](../reports/modeling/final_test_report.md) | Frozen 10% rule, one-time evaluation, conditional bootstrap intervals and low-history errors. |
| 3:45–4:30 | [Explainability](../reports/modeling/explainability_report.md) | Prior inpatient use/destination; associations, not causal effects. |
| 4:30–5:00 | [Synthetic demo](demo_walkthrough.md) | Single probability versus batch decision; loader integrity, API validation, Docker and CI test boundaries. |

## Technical interview questions

**Why this dataset?**
It has a documented readmission label and public access, making the study reproducible. The
selection review compared MIMIC-IV, HCUP NRD and Synthea. The tradeoff is historical, selected
diabetes encounters with incomplete follow-up and no suitable temporal/site validation fields.

**Why patient-grouped splitting?**
One person can contribute several encounters. Separating encounters randomly could share
patient information across partitions. The frozen allocation has zero patient overlap; training
CV also keeps each patient's encounters together. That does not solve temporal transportability.

**Why average precision?**
The positive prevalence is about 11%. AP summarizes precision-recall ranking behavior and
focuses attention on finding positive events. A constant-score baseline has AP equal to
prevalence. I also report ROC-AUC, Brier and capacity-specific outcomes.

**Why not accuracy?**
Predicting no readmission almost everywhere would appear accurate in this imbalanced cohort
while failing the prioritization objective. I evaluated ranking, probability quality, missed
events and the workload associated with the selected queue.

**Why logistic regression?**
Boosting's validation AP gain was +0.00194, with a paired 95% interval of −0.00427 to +0.00839.
It did not satisfy the prespecified complexity rule. Logistic offered similar discrimination
and simpler interpretation. I do not claim the two models are statistically equivalent.

**Why not XGBoost?**
I did not run XGBoost, so I cannot claim it would be better or worse. The actual challenger
was scikit-learn histogram gradient boosting. Adding another library was not necessary to
demonstrate the documented comparison, and a new experiment would need a new evaluation plan.

**Why was class weighting rejected?**
In baseline validation, balanced logistic weighting raised mean predicted risk to 46.53%
against 11.03% observed prevalence and worsened Brier. Improved recall at an arbitrary 0.50
threshold did not justify those probabilities. The frozen model is unweighted.

**Why was calibration not retained?**
The rule required a sufficiently large Brier improvement, a favorable paired interval and
limited AP loss. Logistic sigmoid improvement was negligible. Isotonic reduced Brier by
0.000236 but lost 0.00600 AP. Neither qualified; “uncalibrated” does not mean perfectly calibrated.

**Why not use 0.50 as the threshold?**
It is poorly aligned with a limited-capacity review queue. The tuned logistic model identified
only 10 of 1,499 validation events at that reference threshold. A fixed percentage of a known
batch makes the assumed workload explicit; its realized probability cutoff changes by batch.

**How does top-10% ranking work?**
Validate every encounter and reject an invalid/ineligible batch as a whole. Rank unrounded
probabilities descending, break ties by ascending SHA-256 of `42:` plus the stable request ID,
then select `floor(0.10 × N)`. IDs are tracking/tie keys, never predictors. Below ten, select zero.

**What is lift?**
It compares the selected queue's precision with the cohort event rate: 25.85% divided by
approximately 10.975% gives 2.36×. It measures retrospective concentration relative to random
targeting, not a treatment effect. The finite-batch floor is retained in the exact calculation.

**What did SHAP reveal?**
Prior inpatient visits and discharge destination dominated the documented explanations.
Age, admission source and specialty also contributed. Logistic SHAP explains additive log-odds
contributions; the shared comparison uses the actual boosting pipeline. SHAP does not establish causality.

**Where does the model fail?**
Test recall is 7.6% without prior inpatient use versus 39.6% with it. Half of test events occur
without prior inpatient history, and nearly all captured events in that group involve rehab
destination 22. A low score cannot establish low need. Small subgroup estimates are suppressed.

**How is target leakage prevented?**
Targets and IDs never enter the feature matrix; patients are disjoint across partitions;
preprocessing fits within training folds; and model/policy choices are frozen before final
scoring. Timing-uncertain fields remain excluded. Earlier full-cohort EDA is an acknowledged
limitation, so I describe the design as held out from model development, not entirely untouched.

**Why is the application not clinically deployable?**
It lacks contemporary external/temporal validation, complete outcome capture, verified workflow
timing, prospective intervention evidence and hospital security/integration. The historical
diabetes sample does not establish current clinical utility. No HIPAA compliance is claimed.

**How is the artifact secured and verified?**
The application treats its committed specification and code as trusted. It checks exact model
size/SHA-256 before deserializing the same byte buffer, verifies metadata/features/runtime, and
fails closed on mismatch. Docker mounts the ignored artifact read-only. This integrity control
does not make arbitrary joblib files safe or replace access control and supply-chain governance.

**Why separate Streamlit and FastAPI?**
The UI uses the same schema, eligibility, inference and policy implementation as other clients,
with one loaded model in the API. Separate environments also prevent Streamlit's Arrow dependency
from changing pandas behavior in the frozen modeling stack. This is a local service separation.

**How would contemporary data improve the system?**
First define the intervention, capacity, outcome coverage and scoring moment with stakeholders.
Then use timestamped, appropriately governed data for temporal/site validation, probability and
subgroup monitoring, and field-arrival audits. Investigate low-history errors and repeated-contact
burden. Prospective shadow evaluation and an intervention study would assess usefulness and harm;
none of those studies were completed here.

## Evidence to keep available

[Case study](project_case_study.md) · [Frozen specification](../reports/modeling/final_model_specification.json)
· [Final evaluation](../reports/modeling/final_test_report.md) · [Serving tests and parity](../reports/serving/phase6_verification.md).
Historical phase reports retain the knowledge available at their original stage; use the final
report for headline results. Do not turn an uncertain comparison into an equivalence claim.
