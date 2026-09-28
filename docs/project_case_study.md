# ReadmitIQ: from readmission risk to a constrained review queue

## The business challenge

A readmission score becomes useful only when it supports a defined decision. ReadmitIQ asks
which discharge encounters a care-management team could review first when follow-up capacity
is limited. The portfolio scenario assumes capacity for the highest-risk 10% of eligible
encounters. It does not assume that every recorded readmission is preventable or that contact
produces a benefit. This distinction shapes both evaluation and presentation: the output is
a retrospective prioritization analysis and a local demonstration, not a hospital deployment.

The final policy surfaced 23.54% of recorded readmissions while selecting approximately 10%
of the held-out encounters, producing 2.36× lift over random targeting. Those results create
a concrete discussion about workload and missed events. They also make the model's limitations
visible: most readmissions remain outside the queue, and performance differs substantially
with prior inpatient history. [Archived final results](../reports/modeling/final_test_report.md).

## Understanding the data

The project uses the UCI Diabetes 130-US Hospitals dataset, covering 1999–2008. Its public,
documented encounter data supports a reproducible readmission study without the access barriers
of restricted hospital datasets. The starting extract contains 101,766 encounters. Documented
discharge exclusions leave 90,702 encounters from 65,044 patients. This is a selected diabetes
inpatient population, not a representative sample of every hospital patient.

The positive label is recorded readmission less than 30 days after discharge. The other source
labels, `>30` and `NO`, form the negative class. `NO` does not establish complete follow-up
across hospitals, and the exact day-30 boundary cannot be independently reconstructed. Source
hashes, code mappings, missing-value checks and a feature audit turn these qualifications into
documented contracts rather than treating a downloaded table as analysis-ready.
[Dataset review](dataset_selection.md) · [Data dictionary](data_dictionary.md).

## Designing evaluation around patients

Repeated encounters make row-level random splitting inappropriate: the same person's history
could appear on both sides of an evaluation. ReadmitIQ groups by patient, producing 63,563
training, 13,590 validation and 13,549 test encounters with zero patient overlap. The allocation,
source provenance and serialized partitions are protected by a frozen manifest. Reproduction
must match that manifest; it cannot silently create a more favorable split.

Within training, five patient-disjoint folds support feature comparisons and bounded tuning.
Scaling, category handling and other fitted preprocessing stay inside each fold. A committed
protocol fixes the selected model, features, calibration and outreach policy before the
one-time final prediction pass. This controls important leakage paths, but does not make the
study fully confirmatory: earlier full-cohort EDA exposed eventual test outcomes indirectly,
and development choices reused training folds and validation evidence.
[Split contract](../reports/modeling/data_split_report.md) · [Pretest protocol](../reports/modeling/final_evaluation_protocol.md).

## Exploring predictive signals

Exploration showed a stronger readmission gradient for prior inpatient and emergency use than
for outpatient visits. That supported retaining separate utilization counts rather than hiding
their differences in one total. Confirmed discharge destination also carried substantial
predictive information. These are associations: recorded utilization may reflect access and
capture, while destination may reflect clinical and institutional decisions.

The scoring moment is after destination is confirmed. The final ten inputs combine stay length,
three prior-utilization counts, age, gender, admission type and source, discharge disposition,
and specialty. IDs, outcomes and future patient aggregates are excluded. Race remains available
for auditing rather than prediction. Diagnosis experiments remain training-only sensitivities;
unverified timing prevents their promotion into the primary feature contract.
[Feature policy](../reports/modeling/final_feature_policy.md).

## Comparing model families

The study compares a prevalence baseline, logistic regression, random forest and histogram
gradient boosting. Average precision is the primary ranking metric because recorded positive
outcomes are uncommon, around 11%. Accuracy alone could reward predicting no readmission for
nearly everyone. ROC-AUC, Brier score, calibration diagnostics and capacity-specific metrics
provide complementary views instead of making AP the entire decision.

Seeded Optuna searches completed 15 logistic and 30 boosting trials under prespecified stopping
rules. Patient-bootstrap comparisons keep repeated encounters together and evaluate models on
paired resamples. Their intervals condition on fitted candidates; they do not include the
uncertainty of rerunning feature selection and training. The project does not claim that tuning
uniformly improved the earlier baselines. [Tuning evidence](../reports/modeling/tuning_report.md).

## Choosing logistic regression

Boosting achieved validation AP of 0.19872 versus logistic's 0.19678. The difference, +0.00194,
had a paired 95% interval from −0.00427 to +0.00839. Boosting's Brier score and fold stability
were better, but its AP evidence did not meet the declared complexity margin. Logistic became
the primary model because it offered similar discrimination with simpler interpretation;
boosting remained a challenger. Overlapping uncertainty is not proof of equivalence.

Calibration and class weighting were evaluated rather than added automatically. Balanced
weighting substantially inflated baseline predicted probabilities. Logistic isotonic calibration
improved Brier slightly but lost 0.00600 AP; sigmoid improvement was negligible. Neither satisfied
the retention rule. The final model is unweighted, uncalibrated L2 logistic regression with
`C=0.09988151348099303`, lbfgs and seed 42. Its training-fitted preprocessing and coefficients
remain unchanged. [Selection](../reports/modeling/model_selection_report.md) · [Calibration](../reports/modeling/calibration_report.md).

## Translating risk into an operational decision

The held-out test AP is 0.20837, ROC-AUC 0.65181 and Brier 0.094101. Under the frozen capacity
rule, 1,354 selected encounters surface 350 of 1,487 recorded readmissions: 23.54% recall and
25.85% precision. The distinction matters. Recall describes how many events enter the queue;
precision describes its concentration of events. Neither measures whether outreach helps.

Scaling the observed encounter yield to 10,000 eligible discharges gives approximately 258
surfaced readmissions under model targeting versus 110 with random targeting, at about 1,000
outreach encounters. The difference is approximately 149 additional surfaced events, not
prevented admissions or financial savings. The policy ranks a known batch and selects
`floor(0.10 × N)` with deterministic ties. A single probability cannot establish membership
in the top tenth of an unknown future batch. [Operational policy](../reports/modeling/operational_policy_report.md).

## Explaining predictions and investigating errors

Coefficient contrasts and SHAP identify prior inpatient use and discharge destination as the
largest documented model influences, followed by age, admission source and specialty. Existing
anonymous validation examples illustrate how these contributions combine; they are historical
analytical examples, distinct from the hand-authored synthetic inputs in the application.
Contributions explain predictions and do not identify causes or interventions.

The most consequential error pattern is low recall without prior inpatient use: 7.6%, compared
with 39.6% among those with prior use. Half of test readmissions belong to the no-history group.
Of its 57 captured events, 56 involve rehabilitation destination code 22. Demographic summaries
use suppression rules and patient-cluster intervals; they do not establish fairness. This
analysis makes failure modes part of the project story rather than relegating them to footnotes.
[Explanations](../reports/modeling/explainability_report.md) · [Responsible ML](../reports/responsible_ml.md).

## Engineering a reproducible demonstration

Streamlit calls a versioned FastAPI service that reuses the frozen pipeline and original ranking
implementation. Strict validation enforces the ten-feature schema and discharge eligibility.
Single scores return a probability with no outreach decision; batch requests return ranks and
the exact capacity selection. The loader checks SHA-256 before deserialization, verifies runtime
and metadata compatibility, and loads once per process. Docker mounts the ignored model read-only.

Separate API and UI environments preserve the modeling stack: Streamlit's Arrow dependency
otherwise changes pandas string storage and breaks split regression tests. Phase 6 verified 296
local tests, including four integration tests using the original artifact and synthetic inputs.
Service and direct-pipeline probabilities matched exactly. Public CI uses explicit test doubles
and missing-artifact rejection rather than claiming to possess the ignored binary.
[Deployment contract](local_demo.md) · [Serving evidence](../reports/serving/phase6_verification.md).

## Limitations and future work

Historical selection, incomplete follow-up, unavailable temporal/site validation, unverified
feature timing and earlier exploratory exposure constrain the findings. There is no prospective
intervention evidence, patient-level repeat-contact policy or clinical deployment validation.
Next work would require contemporary data, external and temporal assessment, workflow audits,
clinical review and prospective evaluation of contact burden and outcomes. These are proposed
requirements, not completed experiments. ReadmitIQ demonstrates a defensible analytical decision
and reliable local implementation while preserving those boundaries.
