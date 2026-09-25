"""Evidence-driven Phase 4 reports; all numerical comparisons come from saved outputs."""

import json

import pandas as pd

from readmit_iq.analysis.reporting import markdown_table as shared_markdown_table
from readmit_iq.analysis.reporting import write_json
from readmit_iq.optimization.contract import settings
from readmit_iq.optimization.plots import LABELS, create_figures


def markdown_table(frame: pd.DataFrame) -> str:
    """Retain small Brier changes and interval signs in development comparison tables."""
    return shared_markdown_table(
        frame.map(lambda value: f"{value:.6f}" if isinstance(value, float) else value).where(
            frame.notna()
        )
    )


def paired_feature_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in folds.groupby("family"):
        values = group.pivot(index="fold", columns="configuration", values="average_precision")
        for high, low in [
            ("raw_engineered", "raw"),
            ("indicators", "raw"),
            ("diagnosis_grouped", "raw"),
            ("diagnosis_raw", "raw"),
            ("diagnosis_raw", "diagnosis_grouped"),
        ]:
            difference = values[high] - values[low]
            rows.append(
                dict(
                    family=family,
                    configuration=high,
                    reference=low,
                    mean_ap_difference=difference.mean(),
                    sd_ap_difference=difference.std(),
                    min_ap_difference=difference.min(),
                    max_ap_difference=difference.max(),
                )
            )
    return pd.DataFrame(rows)


def build_reports() -> dict:
    root, policy = settings()
    out = root / policy["report_dir"]
    destination = root / "reports/modeling"
    summary = json.loads((out / "summary.json").read_text())
    metrics = pd.read_csv(out / "validation_metrics.csv").set_index("model")
    features = pd.read_csv(out / "feature_comparison.csv")
    folds = pd.read_csv(out / "cv_folds.csv")
    differences = pd.read_csv(out / "paired_comparisons.csv")
    thresholds = pd.read_csv(out / "thresholds.csv")
    subgroups = pd.read_csv(out / "subgroups.csv")
    feature_folds = pd.read_csv(out / "feature_fold_metrics.csv")
    feature_differences = paired_feature_summary(feature_folds)
    feature_differences.to_csv(out / "feature_paired_fold_differences.csv", index=False)
    figures = create_figures()
    chosen = summary["training"]
    primary, challenger = summary["primary"], summary["challenger"]
    labels = {
        name: f"{LABELS[name.split('__')[0]]} / {name.split('__')[1]}"
        for name in metrics.index
        if "__" in name
    }
    common = (
        "All external metrics are **validation performance** on the frozen 13,590 encounters "
        "/ 9,757 patients. Test has not been scored. Hyperparameters and feature configurations "
        "were selected from training-only patient-grouped CV before validation was opened. "
        "See [the prespecified protocol](phase4_protocol.md).\n\n"
    )
    comparison = []
    for family in ["logistic", "boosting", "forest"]:
        train = chosen[family]
        val = metrics.loc[f"{family}__uncalibrated"]
        comparison.append(
            {
                "Family": LABELS[family],
                "Features": train["configuration"],
                "CV AP mean": train["cv_average_precision_mean"],
                "CV AP SD": train["cv_average_precision_sd"],
                "Fold AP min": train["cv_ap_min"],
                "Fold AP max": train["cv_ap_max"],
                "Validation AP": val.average_precision,
                "ROC-AUC": val.roc_auc,
                "Brier": val.brier,
            }
        )
    text = "# Phase 4 tuning report\n\n" + common
    text += "## Group-aware CV\n\n"
    text += (
        "Five StratifiedGroupKFold folds, shuffle enabled, seed 42, stable encounter ordering. "
        "Each training encounter appears in exactly one holdout; every patient's encounters "
        "remain together. All fit/holdout patient overlaps are zero. No seed search occurred.\n\n"
    )
    text += markdown_table(folds.loc[folds.role.eq("holdout")]) + "\n\n"
    text += (
        "AP is primary because the positive prevalence is about 11%; the constant-score AP "
        "baseline equals prevalence. ROC-AUC and probability error provide complementary evidence. "
        "CV SD describes fold variability; overlapping training folds are not independent "
        "replicates. Feature selection and tuning share folds, so selected CV scores are "
        "optimistic and are not nested-CV performance estimates.\n\n"
    )
    text += "## Search space and selected parameters\n\n"
    text += (
        "Unweighted L2 logistic: C in [0.001,100] on a log scale, lbfgs, maximum 2,000 iterations. "
        "Histogram boosting: learning rate [0.025,0.15] on a log scale, iterations "
        "75/125/150/200/300, leaves 7/15/31, minimum leaf 20/30/50/100/200, "
        "L2 0/0.1/1/10/30. Depth is unset, max_bins=255, and internal early stopping is disabled. "
        "Both include baseline parameters in trial 0. Sequential TPE uses seed 42 and two "
        "numerical threads. No SMOTE, class-weight search, L1/elastic-net search "
        "or forest tuning.\n\n"
    )
    for family in ["logistic", "boosting"]:
        value = chosen[family]
        text += (
            f"**{LABELS[family]}:** {value['trials']} trials; best trial {value['best_trial']}; "
            f"stopped by {value['stop_reason']}.\n\n```json\n"
            + json.dumps(value["parameters"], indent=2)
            + "\n```\n\n"
        )
    text += (
        "Maximum budgets were 30 logistic and 50 boosting trials. The prespecified plateau "
        "rule checks improvements of at least 0.0002 after 15/25 minimum trials and 10/15 "
        "trials of patience. Random forest is one fixed "
        "200-tree/depth-12/min-leaf-20 reference.\n\n"
    )
    text += markdown_table(pd.DataFrame(comparison)) + "\n\n"
    text += "## Controlled feature comparison\n\n"
    text += (
        markdown_table(
            features[
                [
                    "family",
                    "configuration",
                    "cv_average_precision_mean",
                    "cv_average_precision_sd",
                    "eligible",
                ]
            ]
        )
        + "\n\n"
    )
    text += "Matched fold differences (descriptive, not confidence intervals):\n\n"
    text += markdown_table(feature_differences) + "\n\n"
    dimensions = (
        feature_folds.groupby(["family", "configuration"])
        .encoded_features.agg(min_encoded_features="min", max_encoded_features="max")
        .reset_index()
    )
    dimensions.to_csv(out / "feature_dimensions.csv", index=False)
    text += "Encoded dimensions across training folds:\n\n" + markdown_table(dimensions) + "\n\n"
    text += (
        "Logistic uses one-hot levels, while boosting retains ordinal-encoded categorical "
        "columns; their dimension counts are not directly comparable measures of capacity. "
        "Raw diagnosis categories are pooled using training-only rarity rules, so the "
        "comparison is bounded and manageable. Grouping reduces logistic dimensionality "
        "but is not universally more predictive.\n\n"
    )
    for family in ["logistic", "boosting"]:
        f = features.loc[features.family.eq(family)].set_index("configuration")
        raw = f.loc["raw", "cv_average_precision_mean"]
        engineered_gain = f.loc["raw_engineered", "cv_average_precision_mean"] - raw
        indicator_gain = f.loc["indicators", "cv_average_precision_mean"] - raw
        tuning_gain = (
            chosen[family]["cv_average_precision_mean"]
            - f.loc[chosen[family]["configuration"], "cv_average_precision_mean"]
        )
        text += (
            f"- {LABELS[family]} selected **{chosen[family]['configuration']}**. "
            f"Raw + engineered minus raw AP: {engineered_gain:+.5f}; "
            f"indicators-only minus raw: {indicator_gain:+.5f}. "
            f"Tuning on the selected configuration changed CV AP by {tuning_gain:+.5f}.\n"
        )
    text += (
        "\nThe 0.001 simplicity margin was fixed before fitting. Small engineered gains within "
        "that margin do not justify additional inputs. Keep raw counts instead of automatically "
        "discarding their magnitude or retaining redundant derivations.\n\n"
        "Diagnosis experiments are timing-restricted sensitivities, not contenders. Grouped "
        "diagnoses can help boosting in these folds while raw codes need not; logistic differences "
        "are small. Their source arrival time is unverified, and the comparison does not authorize "
        "promoting them. No diagnosis candidate was tuned or scored on validation in Phase 4.\n\n"
    )
    text += "## Did tuning improve external development performance?\n\n"
    text += (
        markdown_table(differences.loc[differences.reference.str.startswith("phase3_")]) + "\n\n"
    )
    text += (
        "The logistic reference uses Phase 3 raw utilization. The boosting reference uses "
        "Phase 3 raw + engineered utilization, so that validation comparison combines feature "
        "selection and tuning; it does not isolate hyperparameters. The CV comparison above "
        "holds the selected configuration fixed. A CV gain does not guarantee a validation gain. "
        "The candidate set was not retuned after these results.\n\n"
        "CV trial/fold metrics, parameters and durations are in "
        "[optuna_trials.csv](phase4/optuna_trials.csv) "
        "and [tuning_fold_metrics.csv](phase4/tuning_fold_metrics.csv). Local MLflow stores run "
        "provenance in ignored mlruns/phase4/tracking.sqlite. "
        "Individual predictions and study/model "
        "objects are ignored development artifacts.\n"
    )
    (destination / "tuning_report.md").write_text(text)

    text = "# Phase 4 calibration report\n\n" + common
    text += (
        "For each tuned family, sigmoid and isotonic calibration use CalibratedClassifierCV "
        "with the same explicit five patient-disjoint training folds and ensemble=False. "
        "Out-of-fold responses fit the calibrator; the base pipeline then fits all training rows. "
        "No validation labels enter either fit. Hyperparameters were selected using the same "
        "training population, so OOF responses are conditional on that selection.\n\n"
        "Reported CV metrics refer to the uncalibrated base estimator's tuning folds, not "
        "an independent CV evaluation of the complete calibration-selection procedure. "
        "Calibration methods are compared once on the external development validation set.\n\n"
    )
    calibrations = metrics.loc[metrics.family.isin(["logistic", "boosting"])]
    text += (
        markdown_table(
            calibrations.reset_index()[
                ["model", "average_precision", "roc_auc", "brier", "mean_probability", "prevalence"]
            ]
        )
        + "\n\n"
    )
    text += (
        markdown_table(differences.loc[differences.model.str.endswith(("__sigmoid", "__isotonic"))])
        + "\n\n"
    )
    text += (
        "Retain calibration only when Brier improves by at least 0.0001, the paired Brier "
        "interval is wholly below zero, and AP loss is <=0.001. If both qualify, choose the "
        "lower Brier. This rule is fixed in configs/phase4.yaml.\n\n"
    )
    for family, name in summary["family_choices"].items():
        text += f"- {LABELS[family]} retains **{name.split('__')[1]}**.\n"
    text += (
        "\nIsotonic may improve squared probability error while creating ties and reducing "
        "ranking precision; the table reports both consequences. Sigmoid need not improve "
        "probability error even when it preserves rank. No method is preferred by reputation. "
        "Mean-probability agreement alone is not evidence of complete calibration. Quantile "
        "curves have no uncertainty bands; Brier mixes calibration and discrimination.\n\n"
        "![Calibration comparison](../figures/modeling/phase4/03_calibration_comparison.png)\n"
    )
    (destination / "calibration_report.md").write_text(text)

    difference = summary["chosen_difference"]
    text = "# Phase 4 development model selection\n\n" + common
    text += f"**Primary: {labels[primary]}. Challenger: {labels[challenger]}.**\n\n"
    text += (
        f"Boosting minus logistic AP is {difference['ap_difference']:+.5f}, with 95% paired "
        f"patient-bootstrap interval [{difference['ap_ci_low']:+.5f}, "
        f"{difference['ap_ci_high']:+.5f}]. "
        f"Brier difference is {difference['brier_difference']:+.6f} "
        f"[{difference['brier_ci_low']:+.6f}, {difference['brier_ci_high']:+.6f}]. "
        "Negative Brier differences favor boosting.\n\n"
    )
    if difference["ap_ci_low"] <= 0 <= difference["ap_ci_high"]:
        text += (
            "The AP ranking difference is statistically difficult to distinguish "
            "on this validation set.\n\n"
        )
    text += (
        markdown_table(
            pd.DataFrame([summary["boosting_selection_checks"]])
            .T.reset_index()
            .rename(columns={"index": "Prespecified boosting selection condition", 0: "Passed"})
        )
        + "\n\n"
    )
    text += (
        "Boosting must improve validation AP by >=0.005 with an interval above zero, have "
        "no more than 0.0002 Brier disadvantage, and match/exceed logistic's CV mean with "
        "at most twice its fold SD. Otherwise the simpler logistic model is preferred. "
        "This is a development selection convention, not a clinical-utility estimate. "
        "Boosting's lower Brier and/or better fold stability remain reasons "
        "to retain a challenger, "
        "even when the AP evidence is insufficient to justify primary status.\n\n"
    )
    names = [primary, challenger, "forest__uncalibrated", "phase3_logistic_raw", "phase3_boosting"]
    text += (
        markdown_table(
            metrics.loc[names].reset_index()[
                [
                    "model",
                    "average_precision",
                    "roc_auc",
                    "brier",
                    "recall",
                    "precision",
                    "f1",
                    "specificity",
                ]
            ]
        )
        + "\n\n"
    )
    text += "All operating metrics above use 0.50 as a reference only. Confusion matrices:\n\n"
    text += (
        markdown_table(metrics.loc[names].reset_index()[["model", "tn", "fp", "fn", "tp"]]) + "\n\n"
    )
    text += "Reference-model comparisons against the tuned logistic candidate:\n\n"
    text += (
        markdown_table(
            differences.loc[differences.model.isin(["phase3_boosting", "forest__uncalibrated"])]
        )
        + "\n\n"
    )
    text += (
        "The Phase 3 booster remains an important historical benchmark; Phase 4 does not "
        "claim that tuning improved it. The fixed forest is not promoted based on a small AP "
        "difference. The primary/challenger tracks follow the declared training/validation "
        "rule rather than a post-hoc hyperparameter search against these outcomes.\n\n"
        "## Complexity and reproducibility\n\n"
    )
    text += (
        markdown_table(
            metrics.loc[[primary, challenger, "forest__uncalibrated"]].reset_index()[
                [
                    "model",
                    "configuration",
                    "encoded_features",
                    "fit_seconds",
                    "inference_batch_rows",
                    "inference_median_ms",
                    "artifact_bytes",
                ]
            ]
        )
        + "\n\n"
    )
    text += (
        "Inference timings are medians of seven warmed batches of 1,024 training feature rows "
        "with two numerical threads on this environment; they are not production latency SLAs. "
        "Fit time excludes the search and includes calibration where applicable. Logistic has "
        "direct coefficients but still requires noncausal interpretation; boosted/forest trees "
        "are more complex. All models exclude IDs/race/outcomes, fit rare categories inside "
        "training folds, and use the same confirmed-discharge contract. The original frozen "
        "split guard remains strict. Model computations may vary slightly across hardware.\n\n"
        "## Exploratory threshold behavior\n\n"
    )
    examples = thresholds.loc[thresholds.threshold.isin([0.05, 0.10, 0.15, 0.20, 0.30, 0.50])]
    text += (
        markdown_table(
            examples[
                [
                    "model",
                    "threshold",
                    "recall",
                    "precision",
                    "f1",
                    "specificity",
                    "encounters_flagged",
                    "encounter_fraction_flagged",
                    "patients_flagged",
                ]
            ]
        )
        + "\n\n"
    )
    text += (
        "The complete 0.00–1.00 grid is diagnostic. No threshold is selected and no capacity, "
        "lift/gains or ROI optimization is performed. Unique patients flagged counts anyone "
        "with a flagged validation encounter across the historical extract; without dates "
        "this is not a simultaneous outreach population.\n\n"
        "## Initial subgroup and demographic diagnostics\n\n"
    )
    primary_groups = subgroups.loc[subgroups.model.eq(primary)]
    text += (
        markdown_table(
            primary_groups[
                [
                    "group",
                    "level",
                    "encounters",
                    "positives",
                    "average_precision",
                    "recall",
                    "precision",
                    "false_negative_rate",
                    "estimates_suppressed",
                ]
            ]
        )
        + "\n\n"
    )
    text += (
        "Diagnostics use the prespecified illustrative threshold 0.10, not a recommended "
        "operating point. None/any/3+ prior inpatient groups overlap intentionally; they use "
        "only provided prior-year history. No single/repeat-patient feature is built from "
        "future appearances. Rates are suppressed below 200 encounters or 30 positives/negatives. "
        "Unknown race/gender remain visible, and counts are retained even where estimates are "
        "suppressed. Passing a sample-size cutoff does not establish precision: small retained "
        "groups can still be unstable. These unadjusted summaries have no subgroup uncertainty "
        "bands and do not establish fairness, discrimination or clinical causality.\n\n"
        "## Limits and artifacts\n\n"
        "The 1,000 bootstrap draws resample whole validation patients, keeping all their "
        "encounters together and pairing every model on the same weights. These intervals "
        "exclude retraining/selection uncertainty and are exploratory across multiple comparisons. "
        "Validation has now informed feature development (Phase 3), calibration choice and "
        "development selection; it is not a final unbiased performance estimate. Prior full-cohort "
        "EDA indirectly exposed eventual test outcomes. No temporal or hospital holdout can be "
        "constructed from this extract. Contemporary availability, outcome ascertainment and "
        "external validation remain unresolved.\n\n"
        f"The selected development artifact is `models/development/phase4/{primary}.joblib`; "
        f"challenger is `{challenger}.joblib`. Models and individual predictions remain ignored. "
        "There is no production artifact and no final test result.\n"
    )
    (destination / "model_selection_report.md").write_text(text)

    text = "# Phase 5 recommendation\n\n"
    text += (
        f"Carry **{labels[primary]}** as primary development model "
        f"and **{labels[challenger]}** as challenger.\n\n"
    )
    text += (
        "1. Freeze the development choice, preprocessing, eligibility and scoring-time contract "
        "before any separately authorized final test evaluation. Do not reopen tuning based on "
        "the modest validation differences. Preserve the original test hashes "
        "and patient isolation.\n"
        "2. Establish intervention capacity, false-positive/false-negative costs and outcome "
        "definitions with stakeholders before choosing an outreach threshold. The exploratory "
        "grid is not an operational policy or a business ROI estimate.\n"
        "3. Investigate why low-utilization patients have lower recall at the illustrative "
        "threshold. Audit subgroup calibration and clustered uncertainty before making fairness "
        "claims; race exclusion alone is insufficient. Retain missingness "
        "and suppress tiny groups.\n"
        "4. Verify actual as-of-discharge field arrival times, especially confirmed destination. "
        "Diagnosis signals remain sensitivity-only until this audit; no CV gain overrides timing.\n"
        "5. Plan deeper error analysis and explanations only in the next authorized phase. "
        "Avoid causal interpretations of coefficients or retrospective associations.\n"
        "6. Plan contemporary external/temporal validation with suitable data. This historical "
        "selected diabetes cohort lacks reliable timestamps, complete outside-hospital follow-up "
        "and a fully unseen exploratory history.\n\n"
        "Phase 4 ends here. No test scoring, final threshold, SHAP, full fairness/error audit, "
        "lift/gains, intervention-capacity optimization, ROI scenario, "
        "API or deployment was performed.\n"
    )
    (destination / "phase5_recommendation.md").write_text(text)
    record = dict(
        figures=figures,
        reports=[
            "tuning_report.md",
            "calibration_report.md",
            "model_selection_report.md",
            "phase5_recommendation.md",
        ],
        test_evaluated=False,
    )
    write_json(out / "artifacts.json", record)
    return record


if __name__ == "__main__":
    build_reports()
