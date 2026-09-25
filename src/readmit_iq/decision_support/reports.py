"""Evidence-based narratives from aggregate Phase 5 outputs; no test scoring or fitting."""

# Long Markdown links/table expressions are kept intact in narrative templates.
# ruff: noqa: E501

import json

import pandas as pd

from readmit_iq.decision_support.plots import PRIMARY, read_table


def markdown_table(table):
    """Small dependency-free renderer that preserves explicit missing/suppressed values."""

    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def write_validation_reports(root):
    out = root / "reports/modeling"
    coefficients = read_table(root, "validation", "coefficient_contrasts")
    numeric = coefficients.loc[
        coefficients.feature.isin(
            ["time_in_hospital", "number_inpatient", "number_emergency", "number_outpatient"]
        )
    ]
    influence = read_table(root, "validation", "global_feature_influence")
    directions = read_table(root, "validation", "explanation_direction_checks")
    verification = json.loads((out / "phase5/validation/explanation_verification.json").read_text())
    (out / "explainability_report.md").write_text(f"""# Frozen-model explanations: validation only

Prior inpatient use and discharge destination dominate logistic and the fixed boosting
challenger. Global mean absolute logistic SHAP contributions are 0.237 and 0.169 log odds;
age, admission source and specialty follow at about 0.064–0.071. Importance summarizes
variation in this validation population, not a causal effect or a clinical intervention target.

## Coefficients in defensible units

Numeric beta is divided by the training StandardScaler scale before exponentiation.
These odds ratios describe one original-unit increase, with other modeled fields fixed;
they do not multiply risk directly. A one-visit inpatient increment multiplies modeled
odds by 1.314; an extra hospital day by 1.013. Linear slopes are imposed by the frozen model.

{markdown_table(numeric[["feature", "coefficient", "odds_ratio", "training_scale"]])}

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
{verification["logistic_maximum_additivity_error"]:.2g}.

The installed [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
adapter did **not** reproduce this native-categorical histogram booster's actual decision
function: maximum error {verification["tree_adapter_check"]["maximum_log_odds_error"]:.3f}
log odds on 32 checked validation rows. Its adapter omits the estimator's categorical split
bitsets/internal representation. Passing the adapter's internal additivity check would not
validate these scores. The model is unchanged. We use permutation SHAP of the **actual**
categorical decision function, with the same training background, 256 seeded validation
rows and eight forward/reverse permutation pairs per row. Its maximum actual-output error
is {verification["boosting_maximum_additivity_error"]:.2g}. Approximate attribution still
has sampling/background sensitivity; exact output additivity does not remove that uncertainty.

Independent masking can combine correlated inputs unrealistically. These are background-relative
predictive explanations, not causal effects or conditional clinical reasoning. The background
base is mean logit, not logit of mean risk. Per-encounter arrays and linkage IDs stay ignored.

{markdown_table(influence)}

## Stability and disagreement

The matched 256-row comparison prevents unequal explanation samples from driving the model
comparison. Prior inpatient visits, emergency visits and length of stay have positive logistic
slopes and positive rank association with SHAP in both models. Raw outpatient visits have a
small negative logistic slope but a positive boosting rank association; both assign them
very little global importance. This disagreement is retained, not presented as a consistent
protective or harmful relationship. Categories have no meaningful ordinal direction, so
we compare their grouped importance rather than invent a rank correlation for category codes.

{markdown_table(directions)}

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
""")

    errors = read_table(root, "validation", "error_profiles", True)
    detail = read_table(root, "validation", "error_characteristics", True)
    overview = detail.loc[
        detail.group.isin(
            [
                "prior_inpatient",
                "prior_emergency",
                "length_of_stay",
                "specialty_missing",
                "payer_missing",
            ]
        )
    ]
    (
        out / "validation_error_analysis.md"
    ).write_text(f"""# Validation errors at the selected outreach capacity

The top-10% policy captures 321 of 1,499 readmissions and leaves **1,178 false negatives**;
1,038 of the 1,359 targeted encounters have no recorded <30-day readmission. This is a
limited-capacity prioritization rule. An unflagged discharge must still receive standard care.

{markdown_table(errors.drop(columns=["model", "partition"]))}

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

{markdown_table(overview.drop(columns=["model", "partition"]))}
""")

    capacities = read_table(root, "validation", "capacity", True)
    columns = [
        "capacity",
        "cutoff",
        "targeted_encounters",
        "targeted_patients",
        "tp",
        "fn",
        "recall",
        "precision",
        "lift",
    ]
    thresholds = read_table(root, "validation", f"{PRIMARY}_thresholds")
    (out / "operational_policy_report.md").write_text(f"""# Validation-selected operational policy

Prioritize the **top 10% highest-risk eligible discharge encounters**, scored at confirmed
discharge after destination is known. This is a **portfolio capacity scenario assumption**,
selected without actual staffing, intervention costs or clinical utility estimates. The user
accepted using a recommended illustrative scenario. It is not a proposed clinical standard.

Top 5% concentrates yield but captures only 13.3% of readmissions. Increasing to 10% captures
21.4%, with precision 23.6%; 15% captures 28.4% at 20.9% precision. Ten percent illustrates
a constrained, understandable staffing budget. No cost-optimal policy is claimed, and F1
was not optimized. Larger capacities below remain sensitivity summaries, not alternate
post-test candidates.

{markdown_table(capacities[columns])}

Rank scores descending, tie-break by SHA-256 of `42:<encounter_id>` ascending, then select
`floor(0.10*N)`. IDs only break ties; outcomes never allocate capacity. Validation cutoff is
0.169407398, but that probability is **not** the deployed rule. Each unlabeled batch's score
distribution determines its cutoff. The same fraction/tie policy will apply to frozen test.

**Prediction/outreach unit: encounter.** Validation targets 1,359 discharge encounters from
871 unique historical people. Patient counts deduplicate the observed partition; they cannot
be summed across capacities/deciles or scaled to 10,000 discharges. No encounter dates are
available to estimate simultaneous caseload, recurring contacts or daily/weekly batches.
Real scheduling, repeat-contact handling and the batch boundary require prospective workflow
design. Pooled historical ranking does not validate a streaming allocation system.

## Fixed probability thresholds are comparison points

{markdown_table(thresholds[["threshold", "targeted_encounters", "targeted_patients", "fraction_targeted", "recall", "precision", "specificity", "fp", "fn", "number_needed_to_contact"]])}

The listed 0.169407 comparison is rounded; the rank policy enforces capacity and handles ties
explicitly. A 0.50 threshold misses almost every event and has no useful capacity rationale.

## Illustrative operational scenario per 10,000 eligible discharges

Scale validation's encounter-level experience: **1,103 expected recorded readmissions**,
**1,000 outreach encounters**, **236 readmissions surfaced** by model targeting versus **110**
under random outreach, or **126 additional surfaced readmissions**. About **4.23 contacts per
observed readmission surfaced** is a retrospective yield measure, not number needed to treat.
These are expected scaled counts, not a forecast for a named hospital or unique-person counts.
No effect on readmission, prevention, cost saving or intervention effectiveness is estimated.

## Proposed use pending external validation

A care-management team could review an available discharge batch, prioritize its top 10%,
and verify patient context before offering follow-up. Unflagged patients retain usual care;
the model should not make diagnoses, deny services or determine discharge destinations.
The mixed rehabilitation population and strong destination signal require explicit review.
Actual staffing, field arrival times, outreach acceptance and outcomes must be established
before this retrospective scenario can support a hospital policy.

[Risk deciles](phase5/validation/risk_deciles.csv) include observed/predicted risk, unique
people, cumulative event capture and lift. Decile 1 is the highest-risk tenth, with 321
readmissions and 23.6% observed risk; ten deciles reconcile to all 13,590 encounters/1,499 events.
""")


def write_responsible_report(root, include_test=False):
    partitions = ["validation", "test"] if include_test else ["validation"]
    if include_test:
        from readmit_iq.decision_support.final_evaluation import verify_published_results

        verify_published_results(root)
    sections = []
    for partition in partitions:
        groups = read_table(root, partition, "subgroups", True)
        headline = groups.loc[
            groups.group.isin(["prior_inpatient", "prior_inpatient_high", "age", "gender", "race"])
        ]
        flagged = groups.loc[groups.material_probability_bias.eq(True)]
        sections.append(f"""## {partition.title()} diagnostics, frozen primary

{markdown_table(headline[["group", "level", "encounters", "patients", "positives", "suppressed", "prevalence", "average_precision", "recall", "recall_ci_low", "recall_ci_high", "precision", "false_negative_rate", "mean_probability"]])}

Material mean probability-bias flags: **{len(flagged)}** of {int((~groups.suppressed).sum())}
supported group rows. {("None meet the prespecified joint size/magnitude/interval criterion." if flagged.empty else markdown_table(flagged[["group", "level", "probability_bias", "probability_bias_ci_low", "probability_bias_ci_high"]]))}
The [complete table](modeling/phase5/{partition}/subgroups.csv) includes Brier, bias intervals,
all predeclared clinical/missingness groups and the challenger's identical diagnostics.
""")
    (root / "reports/responsible_ml.md").write_text(
        """# Responsible ML: specific findings and deployment limits

The primary model prioritizes repeated prior use much more strongly than first recorded
inpatient history. At validation's 10% budget, no-history recall is 6.0% (95% interval
4.4–7.7%), compared with 38.1% (33.7–42.4%) for any prior use and 87.6% for 3+ visits.
That gap persists despite reasonable average probability alignment: limited-capacity ranking
and calibration answer different questions. It is a major reason to retain standard care
for unflagged patients and evaluate a complementary clinical pathway prospectively.

Recorded female/male validation recall is 20.7%/22.3%, with overlapping cluster intervals.
African American/Caucasian recall is 16.1%/23.2%; the first interval is wide (10.0–22.3%).
Hispanic recall is 13.9% with only 36 positives and a very wide interval. These observational
differences can reflect prevalence, history capture, case mix, missingness and institutional
practice; they establish neither fairness nor unlawful discrimination. No equalized-odds or
causal fairness claim is supported. Race is evaluation-only; recorded binary gender is an
input whose justification and alternatives require local clinical governance.

## Prespecified uncertainty and suppression

Keep Unknown and raw counts. Suppress group performance below **200 encounters, 100 patients,
30 positives or 30 negatives**. Subgroups overlap; patient totals must not be added.
Ratio intervals use whole-patient cluster influence estimates, conditional on the fitted
model and the selected full-partition outreach list. They do not repeat model selection or
rerank subgroups independently. Unlike the headline bootstrap, they do not model outreach
cutoff uncertainty. AP and Brier are descriptive; no multiple-comparison-adjusted hypothesis
test is claimed. Small-group suppression does not establish safety for excluded groups.

Define mean probability bias as predicted mean minus observed prevalence. Flag it as material
only when absolute bias is >=2 percentage points **and** its cluster 95% interval excludes
zero. This is a declared descriptive screen, not proof of calibration. Mean alignment can
hide errors within risk bins; quantile reliability plots and the archived group tables must
be read together. No demographic threshold or subgroup recalibration is fitted.

"""
        + "\n".join(sections)
        + """
## Why this cannot yet guide clinical deployment

* The selected diabetes inpatient cohort dates to **1999–2008**. Coding, clinical treatment,
  discharge practice, population mix and outreach availability may have changed.
* This is observational data with incomplete outside-hospital follow-up; NO is no recorded
  readmission, not verified absence. The exact day-30 boundary cannot be audited from categories.
  No hospital IDs/dates support contemporary site/temporal transport assessment.
* Patient-group splitting prevents repeated-patient overlap (zero across partitions), but
  it is neither temporal nor external validation. Earlier full-cohort EDA exposed eventual
  test outcomes indirectly. Final test is held out from model development, not a fully
  untouched confirmatory study; bootstrap intervals omit that exploration/selection uncertainty.
* Demographics are historical recorded categories with missing/unknown values and small groups.
  Utilization can measure access and fragmented capture as well as illness. Destination and
  specialty can reflect institutional decisions; strong prediction does not imply a causal target.
* Confirmed discharge and destination availability are explicit assumptions. The extract has
  no field arrival timestamps. Audit real-time registration, look-back counts, final versus
  provisional destination, coding corrections and specialty before any shadow evaluation.
* Aggregate retrospective lift measures surfaced events, not preventability or benefit.
  Intervention effect, contact burden, repeated contacts and possible harms are unmeasured.
* Before use: clinical and patient review, current external/temporal validation, silent
  prospective scoring, workflow/capacity evaluation and an intervention study are needed.
  Monitor feature availability, prevalence, calibration, utilization/demographic error gaps,
  workload and outcomes with governance-defined review criteria. No monitoring service is built here.

All model, feature, calibration, subgroup and targeting definitions remain fixed after the
pretest protocol. Final diagnoses, labs, medications and payer remain excluded. This report
documents limitations; it does not authorize direct clinical use or deny care to low scores.
"""
    )


def write_final_report(root):
    from readmit_iq.decision_support.final_evaluation import verify_published_results

    summary = verify_published_results(root)
    val = read_table(root, "validation", "metrics", True).iloc[0]
    test = read_table(root, "test", "metrics", True).iloc[0]
    metrics = read_table(root, "test", "metrics")
    intervals = read_table(root, "test", "metric_intervals", True)
    capacity = read_table(root, "test", "capacity", True)
    groups = read_table(root, "test", "subgroups", True)
    prior = groups.loc[groups.group.eq("prior_inpatient")].set_index("level")
    scenario = summary["scenario"]
    comparison = pd.DataFrame(
        [
            {
                "partition": "Training CV (selected; not nested)",
                "AP": 0.212434,
                "ROC_AUC": None,
                "Brier": None,
            },
            {
                "partition": "Validation",
                "AP": val.average_precision,
                "ROC_AUC": val.roc_auc,
                "Brier": val.brier,
            },
            {
                "partition": "Final test",
                "AP": test.average_precision,
                "ROC_AUC": test.roc_auc,
                "Brier": test.brier,
            },
        ]
    )
    (
        root / "reports/modeling/final_test_report.md"
    ).write_text(f"""# One-time frozen final test evaluation

The evaluation used the committed [protocol](final_evaluation_protocol.md), freeze commit
`{summary["freeze_commit"]}`, and code commit `{summary["pretest_code_commit"]}`.
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

{markdown_table(comparison)}

Training CV AP SD was 0.007680; those reused selection folds do not provide an unbiased
final performance estimate. Compare test with validation descriptively, without choosing a
new model from their ordering. Patient-bootstrap intervals below condition on the fitted
model; 1,000 paired whole-patient draws, seed 42, rerank the fixed capacity in each draw.

{markdown_table(intervals.drop(columns=["model"]))}

{markdown_table(metrics[["model", "average_precision", "roc_auc", "brier", "recall", "precision", "specificity", "tp", "fp", "fn", "tn", "lift"]])}

The [paired challenger comparisons](phase5/test/paired_comparisons.csv) use the same patient
draws. Boosting's AP advantage is +0.00474 (95% paired interval −0.00211 to +0.01101).
Its ROC-AUC and Brier are better, but at the selected capacity it captures 344 readmissions
versus logistic's 350; the capacity-recall difference is uncertain. These secondary results
do not reopen the prespecified primary selection. The uncertainty estimates exclude
 temporal/site shift, retraining and selection.
Test was isolated from model fitting/tuning; earlier full-cohort EDA means it was not fully
unseen in the broader analysis. No equivalence or clinical utility conclusion follows.

## Frozen operational result

Of {int(test.encounters):,} test encounters from {int(test.patients):,} patients, the fixed
policy targets **{int(test.targeted_encounters):,} encounters / {int(test.targeted_patients):,}
unique historical people**, capturing **{int(test.tp)}/{int(test.positives)} readmissions
({test.recall:.2%})**, with **{test.precision:.2%} precision** and **{test.lift:.3f}× lift**.
Specificity is {test.specificity:.2%}; false positives {int(test.fp):,}, false negatives
{int(test.fn):,}. The observed cutoff is {test.realized_cutoff:.9f}; no probability cutoff
was reoptimized. Floor rounding targets {test.fraction_targeted:.5%} of the partition.

{markdown_table(capacity[["capacity", "cutoff", "targeted_encounters", "targeted_patients", "tp", "fn", "recall", "precision", "lift"]])}

At the observed test mix, per 10,000 eligible discharges, about
**{scenario["outreach_encounters"]:.1f} outreach encounters** surface
**{scenario["model_surfaced_readmissions"]:.1f}** readmissions versus
**{scenario["random_surfaced_readmissions"]:.1f}** under random targeting:
**{scenario["additional_surfaced_readmissions"]:.1f} additional surfaced events**.
Expected recorded readmissions: {scenario["expected_observed_readmissions"]:.1f}.
Contacts per surfaced event: {scenario["number_needed_to_contact"]:.2f}. This scales observed
encounter-level yields, not unique people, prevented readmissions or savings.

## Predeclared failure and subgroup diagnostics

No prior inpatient history accounts for {int(prior.loc["None", "positives"])}/{int(test.positives)}
readmissions ({prior.loc["None", "positives"] / test.positives:.1%}). Recall is
**{prior.loc["None", "recall"]:.1%} without prior use** versus
**{prior.loc["Any", "recall"]:.1%} with prior use**. The low-utilization concern persists.
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
""")
