"""Feature provenance, fixed-experiment evidence and a bounded Phase 4 recommendation."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from readmit_iq.analysis.reporting import markdown_table, write_json
from readmit_iq.modeling.features import (
    CORE_CATEGORICAL,
    DERIVED_UTILIZATION,
    DIAGNOSES,
    MEDICATIONS,
    SNAPSHOT_CATEGORICAL,
    SNAPSHOT_NUMERIC,
    UTILIZATION,
)
from readmit_iq.modeling.settings import settings


def feature_manifest(root: Path) -> None:
    audit = pd.read_csv(root / "reports/data_quality/feature_audit.csv", keep_default_na=False)
    rows = []
    primary = set(CORE_CATEGORICAL + UTILIZATION + ["time_in_hospital"])
    for variable in audit.itertuples():
        name = variable.variable
        kind = (
            ("numeric")
            if name in UTILIZATION + SNAPSHOT_NUMERIC + [("time_in_hospital"), ("number_diagnoses")]
            else ("categorical")
        )
        if name in primary:
            status = "Primary allowed"
            output = name
            reason = "Antecedent/as-of-discharge source semantics; see scoring-time contract"
            if name == "discharge_disposition_id":
                reason = (
                    "Conditional on confirmed destination; without-disposition ablation included"
                )
        elif name in DIAGNOSES:
            status, output = "Sensitivity only", f"{name}_group or raw {name}"
            reason = (
                "Final coding time unverified; broad documented groups versus "
                "train-pooled raw codes"
            )
        elif name in SNAPSHOT_CATEGORICAL + SNAPSHOT_NUMERIC + ["number_diagnoses"]:
            status, output = "Sensitivity only", name
            reason = "Billing/result/full-stay finalization time unverified"
        elif name in MEDICATIONS:
            status, output = (
                ("Sensitivity derivation only"),
                ("active/changed diabetes drug-field counts"),
            )
            reason = (
                "Count documented status fields; do not interpret as independent "
                "ingredients or change events"
            )
        else:
            status, output = "Excluded", "none"
            reason = {
                "encounter_id": "Traceability only; never an order surrogate",
                "patient_nbr": "Grouping only; no future patient aggregates",
                "readmitted": "Original outcome: direct leakage",
                "weight": "96.68% missing in Phase 2; no initial predictive use",
                "race": "Audit-only; predictive inclusion deferred",
                "examide": "Constant",
                "citoglipton": "Constant",
            }.get(name, "Outside allowlist")
        rows.append(dict(source=name, transformed=output, type=kind, status=status, reason=reason))
    rows.append(
        dict(
            source="readmitted_lt30",
            transformed="y only",
            type="binary target",
            status=("Excluded from predictors"),
            reason=("Validated target; direct leakage if used as X"),
        )
    )
    for name in DERIVED_UTILIZATION:
        rows.append(
            dict(
                source=(", ").join(UTILIZATION)
                if name.startswith("total")
                else ("number_") + name.removeprefix("any_prior_"),
                transformed=name,
                type="numeric" if name.startswith("total") else "binary",
                status="Primary engineered candidate",
                reason=(
                    "Fixed encounter-local arithmetic; raw-count ablation tests joint added value"
                ),
            )
        )
    for name in ["active_diabetes_drug_fields", "changed_diabetes_drug_fields"]:
        rows.append(
            dict(
                source=("21 nonconstant drug-status columns"),
                transformed=name,
                type=("numeric"),
                status=("Sensitivity only"),
                reason=("Active = Steady/Up/Down; changed = Up/Down; No counts zero"),
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(root / "reports/modeling/model_feature_manifest.csv", index=False)
    text = "# Model feature manifest\n\n" + markdown_table(result)
    text += (
        "\n\nPrimary: 10 source fields plus 4 fixed utilization derivations (14 "
        "columns before encoding). "
    )
    text += "No identifiers, outcomes or full-dataset patient aggregates are passed to estimators. "
    text += "All source fields remain in ignored analytical data for traceability.\n\n"
    text += (
        "Linear and random-forest pipelines use training-pooled one-hot "
        "categories; only linear numeric columns are standardized. "
    )
    text += (
        "Histogram boosting uses ordinal category IDs explicitly marked "
        "categorical, not numeric ordering. "
    )
    text += (
        "All numerical counts are complete under the validated source "
        "contract, so no numeric imputer is fitted. "
    )
    text += (
        "Missing/nonfinite numeric counts fail explicitly; revise the contract "
        "if future data requires a train-only imputer.\n\n"
    )
    text += (
        "Categories seen fewer than 100 times in training map to Other; the "
        "count cutoff was fixed before validation. "
    )
    text += (
        "Training frequencies are saved separately. Unknown, Not measured and "
        "Invalid are reserved levels; age bands remain intact. "
    )
    text += (
        "Unseen values map safely to Other. Age is categorical rather than a "
        "fabricated exact age or a forced linear ordinal trend.\n\n"
    )
    text += (
        "Clinical diagnosis ranges follow [Strack et al. Table "
        "3](https://onlinelibrary.wiley.com/doi/10.1155/2014/781670): "
    )
    text += (
        "390–459/785 circulatory, 460–519/786 respiratory, 520–579/787 "
        "digestive, 580–629/788 genitourinary, "
    )
    text += (
        "250.xx diabetes, 800–999 injury, 710–739 musculoskeletal, 140–239 "
        "neoplasms, otherwise Other. "
    )
    text += (
        "The same documented grouping applied to secondary/tertiary fields is "
        "an explicit project extension. "
    )
    text += (
        "Decimal suffixes use the numeric code family; short numeric codes "
        "remain in their valid broad family. "
    )
    text += (
        "E/V supplementary codes map to residual Other, missing to Unknown and "
        "malformed codes to Invalid. "
    )
    text += "This is broad grouping, not validation of every ICD code against a complete codebook. "
    text += (
        "A primary-diabetes indicator is omitted because the grouped category "
        "already contains exactly that information.\n\n"
    )
    text += (
        "The two medication counts describe source columns, not distinct "
        "ingredients, doses, prescriptions at discharge or numbers of change "
        "events. "
    )
    text += (
        "Combination-product fields count once; constants are omitted. Never "
        "count No as active or Unknown as no treatment.\n"
    )
    (root / "reports/modeling/model_feature_manifest.md").write_text(text)


def build_reports() -> None:
    root, policy = settings()
    out = root / policy["modeling"]["report_dir"]
    summary = json.loads((out / "phase3_summary.json").read_text())
    results = pd.read_csv(out / "baseline_model_results.csv")
    differences = pd.read_csv(out / "feature_ablation_results.csv")
    calibration = pd.read_csv(out / "calibration_bins.csv")
    errors = pd.read_csv(out / "validation_error_groups.csv", keep_default_na=False, na_values=[""])
    predictions = pd.read_csv(
        root / policy[("modeling")][("artifact_dir")] / ("validation_predictions.csv.gz")
    )
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    from readmit_iq.modeling.plots import create_figures

    figures = create_figures(
        results,
        differences,
        calibration,
        predictions,
        summary[("development_winner")],
        root / policy["modeling"]["figure_dir"],
    )
    feature_manifest(root)
    fields = [
        "name",
        "average_precision",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "specificity",
        "brier",
    ]
    primary = results.loc[results.role.isin(["primary", "baseline"])]
    by_name = results.set_index("name")
    # Narrative below is an analyst-reviewed snapshot, not automatically inferred prose.
    # Refuse to publish stale conclusions if a later experiment changes the evidence.
    reviewed_ap = {
        "naive_prevalence": 0.11030,
        "logistic": 0.19428,
        "logistic_balanced": 0.19387,
        "random_forest": 0.19747,
        "histogram_boosting": 0.20218,
        "logistic_raw_utilization": 0.19641,
        "logistic_no_utilization": 0.15492,
        "logistic_no_disposition": 0.17999,
        "logistic_diagnosis_groups": 0.19597,
        "logistic_diagnosis_raw": 0.19832,
        "logistic_encounter_summaries": 0.19469,
    }
    if any(
        abs(by_name.loc[name, "average_precision"] - ap) > 0.00001
        for name, ap in reviewed_ap.items()
    ):
        raise ValueError("Validation results changed: review narrative before regenerating reports")
    high_use = errors.loc[errors.group.eq("prior_inpatient") & errors.level.eq("3+")].iloc[0]
    if high_use["fn"] != 219 or high_use.predicted_positives != 13:
        raise ValueError("Error evidence changed: review narrative before publication")
    text = "# Phase 3 validation baselines\n\n"
    text += (
        "All results below use the frozen **validation** partition. Test has not been evaluated. "
    )
    text += (
        "The scoring-time allowlist is fixed before comparison. Sensitivity "
        "models are ineligible for primary selection.\n\n"
    )
    text += markdown_table(primary[fields])
    text += (
        "\n\nAverage precision (AP) is the primary metric: it summarizes "
        "precision/recall across thresholds and its constant-score reference "
        "equals validation prevalence. "
    )
    text += (
        "It is sklearn average_precision_score, not trapezoidal PR area. "
        "ROC-AUC is complementary; accuracy is not a selection metric. "
    )
    text += (
        "Precision/recall/F1/specificity and confusion counts use **0.50 "
        "only**, with no threshold search. "
    )
    text += (
        "Precision is reported as zero when there are no positive predictions. "
        "No SMOTE, calibration fitting or hyperparameter search occurred.\n\n"
    )
    text += markdown_table(
        primary[
            [
                ("name"),
                ("tn"),
                ("fp"),
                ("fn"),
                ("tp"),
                ("threshold"),
                ("mean_probability"),
                ("prevalence"),
            ]
        ]
    )
    text += "\n\n## Controlled comparisons\n\n" + markdown_table(differences)
    text += (
        "\n\nIntervals are paired percentile intervals from 300 patient "
        "bootstrap resamples, seed 42. "
    )
    text += (
        "They describe validation sampling variation, not external "
        "transportability or uncertainty from retraining. "
    )
    text += (
        "Multiple comparisons are exploratory; narrow numerical rankings are "
        "not proof of a final winner.\n\n"
    )
    text += ("## Timing-sensitive diagnostic comparisons\n\n") + markdown_table(
        results.loc[results.role.eq("sensitivity"), fields + [("encoded_features")]]
    )
    text += (
        "\n\nRaw diagnosis codes achieve 0.198 AP versus 0.196 for broad groups, "
        "but the paired difference interval includes zero "
    )
    text += (
        "(−0.006 to +0.010). Raw encoding has 382 columns versus 118 with "
        "grouping. Grouping is compact; superiority is not established. "
    )
    text += (
        "Neither representation is promoted without an as-of-scoring diagnosis "
        "audit. The encounter-summary sensitivity adds only 0.0004 AP "
    )
    text += (
        "(interval −0.0024 to +0.0034); it provides no current reason to "
        "broaden the timing-sensitive allowlist.\n\n"
    )
    text += "## Decisions supported by these results\n\n"
    text += "- Logistic regression improves AP by 0.084 over the prevalence baseline (1.76× AP).\n"
    text += (
        "- Prior utilization adds 0.039 AP (95% paired interval 0.027–0.055): "
        "the Phase 2 association contributes predictive value.\n"
    )
    text += (
        "- Total-plus-any-use additions reduce AP by 0.002 versus retaining "
        "raw counts alone (interval −0.0052 to +0.0001). "
    )
    text += (
        "Do not retain that package by default in Phase 4; this joint ablation "
        "does not identify which individual derived term is responsible. "
    )
    text += (
        "The total is also exactly redundant with the three counts in a linear "
        "representation, although it can change regularization.\n"
    )
    text += (
        "- Disposition adds 0.014 AP (0.008–0.021), conditional on the "
        "confirmed-discharge contract. Predictiveness does not establish "
        "arrival time.\n"
    )
    text += (
        "- Histogram boosting improves AP by 0.0079 over the matched logistic "
        "baseline (0.0015–0.0143): a modest development gain. "
    )
    text += "Random forest adds 0.0032 (−0.0034 to +0.0106), which is inconclusive.\n"
    text += (
        "- Balanced logistic weights leave AP essentially unchanged (−0.0004), "
        "increase recall from 0.004 to 0.532 "
    )
    text += (
        "and lower precision from 0.462 to 0.172 at the same arbitrary "
        "threshold. Brier worsens from 0.095 to 0.226. "
    )
    text += (
        "This does not establish superiority over an unweighted model at a "
        "different threshold, which was not tested.\n\n"
    )
    text += "## Initial probability and error checks\n\n"
    text += (
        "Validation prevalence is 11.03%. Mean predicted risk is 10.92% for "
        "boosting and 10.91% for unweighted logistic, "
    )
    text += (
        "but 46.53% for balanced logistic. The unweighted models have "
        "reasonable average probability levels; that alone does not prove "
        "calibration. "
    )
    text += (
        "Quantile curves reveal local deviations and have no uncertainty "
        "bands. Balanced probabilities markedly overstate absolute risk. "
    )
    text += "Brier combines calibration and discrimination; no calibrator has been fit.\n\n"
    text += (
        "The boosting model at 0.50 produces only 13 positive predictions: 7 "
        "true positives and 6 false positives, with 1,492 false negatives. "
    )
    text += (
        "All 13 predictions belong to patients with 3+ provided prior "
        "inpatient visits; 219 of 226 positives in that group are still "
        "missed. "
    )
    text += (
        "Five false positives are home discharges and one is home-health; six "
        "total errors cannot establish a stable destination-specific pattern. "
    )
    text += (
        "Specialty-unknown encounters have 773 false negatives out of 776 "
        "positives versus 719/723 when observed. "
    )
    text += (
        "Payer-unknown misses are 566/569 versus 926/930 observed. At this "
        "threshold almost everyone is negative, so these counts are "
    )
    text += (
        "mainly evidence that 0.50 is not an operational outreach threshold, "
        "not evidence of a specific missingness mechanism or fairness. "
    )
    text += (
        "Risk means rise from 8.37% at no prior inpatient visits to 24.76% at "
        "3+. Do not confuse these provided history fields with future repeat "
        "status.\n\n"
    )
    text += "## Scope and reproducibility\n\n"
    text += (
        "Eleven prespecified experiments: five baseline/primary runs, three "
        "core ablations and three timing-sensitive runs. "
    )
    text += "Settings and seed live in configs/phase3.yaml; no validation-driven parameter search. "
    text += (
        "Logistic uses C=1 and convergence checks; forest uses 200 trees/depth "
        "12/min leaf 20; boosting uses 150 iterations/15 leaves/learning rate "
        "0.05. "
    )
    text += "Boosting early stopping is disabled to avoid an implicit encounter-level split. "
    text += "All vocabularies, rare pooling, scaling and estimators fit training only.\n\n"
    text += (
        "Use lightweight YAML + CSV/JSON tracking for this finite registry; "
        "MLflow is deferred until experiment volume warrants it. "
    )
    text += (
        "Development joblib pipelines and individual validation predictions "
        "are ignored by Git. No final model is declared. "
    )
    text += (
        "The package already supplies the three model families, so "
        "XGBoost/LightGBM/CatBoost dependencies were not added merely to fill "
        "a roster. "
    )
    text += "No model family is assumed promising without observed evidence.\n\n"
    text += (
        "Limitations: full-cohort Phase 2 EDA indirectly viewed future test "
        "outcomes; historical selected diabetes encounters; "
    )
    text += (
        "incomplete out-of-network follow-up; no dates/hospital holdout; "
        "retrospective source timing; one repeatedly consulted validation set; "
    )
    text += (
        "and unadjusted subgroup error summaries. Final test remains isolated "
        "from subsequent development.\n\n"
    )
    text += (
        "Implementation references: [sklearn "
        "AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics."
        "average_precision_score.html), "
    )
    text += (
        "[histogram boosting](https://scikit-learn.org/stable/modules/generated"
        "/sklearn.ensemble.HistGradientBoostingClassifier.html) "
    )
    text += (
        "and [one-hot encoding](https://scikit-learn.org/stable/modules/generat"
        "ed/sklearn.preprocessing.OneHotEncoder.html).\n"
    )
    (out / "baseline_model_report.md").write_text(text)
    recommendation = """# Phase 4 recommendation from validation evidence

Prioritize **histogram gradient boosting and logistic regression**. Keep random forest as a
lower-priority reference, not a required tuning track: its AP advantage over logistic is
uncertain, while boosting has a modest positive paired difference. The best validation model
is a development candidate only. No final test evaluation or production model exists.

1. Carry forward the frozen patient assignments and confirmed-discharge feature contract.
   Use patient-grouped training folds for future parameter comparisons. Preserve validation
   for bounded development decisions and keep test locked until the final pipeline is fixed.
2. Start the logistic track from separate raw utilization counts: the total/any-use package
   showed no benefit. Retain all three counts; do not infer a strict monotonic constraint.
   Validate whether redundant derived terms are also unnecessary for boosting inside training.
3. Tune a small regularization/complexity range only in the next authorized phase. Favor the
   two observed promising families over importing many similarly motivated boosters now.
4. Assess calibration with group-aware out-of-fold training predictions. Weighted logistic
   probability distortion needs attention if weighting is pursued; AP currently offers no reason
   to prefer it. No current evidence justifies SMOTE.
5. Later choose an outreach threshold using costs/capacity and development data. At 0.50,
   boosting misses 99.53% of positive encounters, including most high-utilization cases.
   Do not mistake low default-threshold recall for absence of ranking value, or balanced-model
   recall for an improved precision–recall frontier.
6. Require a timing audit before adding final diagnoses, billing, lab or treatment summaries.
   Grouped diagnoses are more compact, but raw versus grouped performance is inconclusive.
   Encounter-summary gain is negligible. Neither sensitivity result overrides the contract.
7. Review subgroup calibration/error rates with patient uncertainty after choosing a meaningful
   development-only operating policy. Current tiny positive-call counts are inadequate for
   stable subgroup conclusions. Excluding race as a predictor does not establish fairness.

Unresolved operational questions: actual source-field arrival timestamps, availability of
confirmed destination before outreach allocation, suitability of mixed rehabilitation cases,
out-of-network outcome ascertainment and contemporary external validation. Cohort scope stays
fixed throughout current development; changing it requires an explicit versioned redesign.

No tuning, resampling, final calibration, SHAP, threshold optimization, API or deployment was
performed in Phase 3. Do not start those steps automatically.
"""
    (out / "phase4_recommendation.md").write_text(recommendation)
    probability_quality = []
    for name, group in calibration.groupby("model"):
        probability_quality.append(
            dict(
                model=name,
                quantile_ece=float(
                    np.average(
                        abs(group.mean_probability - group.observed_rate), weights=group.encounters
                    )
                ),
            )
        )
    write_json(
        out / ("phase3_artifacts.json"),
        dict(
            figures=figures,
            calibration_descriptive=probability_quality,
            test_evaluated=False,
            development_winner=summary[("development_winner")],
        ),
    )


if __name__ == "__main__":
    build_reports()
