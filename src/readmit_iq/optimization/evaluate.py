"""Score sealed candidates once on validation, then analyze the cached development predictions."""

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from readmit_iq.analysis.reporting import write_json
from readmit_iq.data.download import sha256, verify_file
from readmit_iq.modeling.metrics import calibration_bins, classification_metrics
from readmit_iq.optimization.contract import (
    BLOCKED,
    digest,
    load_partition,
    settings,
    verify_contract,
)
from readmit_iq.optimization.fit import fit_candidates
from readmit_iq.optimization.tracking import LocalTracker

LOGGER = logging.getLogger(__name__)
BASELINES = {
    "phase3_logistic_raw": "logistic_raw_utilization",
    "phase3_boosting": "histogram_boosting",
}


def score_validation_once() -> tuple[pd.DataFrame, dict, dict]:
    root, policy = settings()
    fitted = fit_candidates()
    artifacts = root / policy["artifact_dir"]
    manifest_path = artifacts / "validation_scores.json"
    prediction_path = artifacts / "validation_predictions.csv.gz"
    fitted_hash = digest(fitted)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fit_manifest_sha256"] != fitted_hash:
            raise ValueError("Scored candidates changed; refusing validation rescoring")
        verify_file(prediction_path, manifest["prediction_sha256"])
        return (
            pd.read_csv(
                prediction_path,
                dtype={"patient_nbr": "string", "encounter_id": "string"},
                keep_default_na=False,
            ),
            manifest,
            fitted,
        )
    attempt = artifacts / "validation_started.json"
    if attempt.exists():
        raise ValueError("Incomplete validation scoring needs explicit review; no automatic retry")
    # Benchmark on training features, so repeated inference timing does not rescore validation.
    training = load_partition(root, "train", "calibration")
    batch = training.drop(columns=BLOCKED).iloc[: policy["evaluation"]["inference_batch_size"]]
    models, timings = {}, {}
    for name in fitted["models"]:
        models[name] = joblib.load(artifacts / f"{name}.joblib")
        measurements = []
        with threadpool_limits(limits=policy["threads"]):
            models[name].predict_proba(batch)  # warmup, no performance metrics
            for _ in range(policy["evaluation"]["inference_repeats"]):
                start = time.perf_counter()
                models[name].predict_proba(batch)
                measurements.append(time.perf_counter() - start)
        timings[name] = dict(
            inference_batch_rows=len(batch),
            inference_median_ms=1000 * float(np.median(measurements)),
        )
    write_json(attempt, {"fit_manifest_sha256": fitted_hash, "candidate_names": list(models)})
    validation = load_partition(root, "validation", "evaluation")
    if set(training.patient_nbr) & set(validation.patient_nbr):
        raise ValueError("Development patient overlap")
    columns = [
        "encounter_id",
        "patient_nbr",
        "readmitted_lt30",
        "race",
        "gender",
        "age",
        "number_inpatient",
        "number_emergency",
    ]
    scored = validation[columns].rename(columns={"readmitted_lt30": "y"}).copy()
    X = validation.drop(columns=BLOCKED)
    for name, model in models.items():
        LOGGER.info("Single validation prediction pass: %s", name)
        with threadpool_limits(limits=policy["threads"]):
            scored[name] = model.predict_proba(X)[:, 1]
    baseline_path = root / "models/development/validation_predictions.csv.gz"
    baseline = pd.read_csv(baseline_path, dtype={"encounter_id": "string", "patient_nbr": "string"})
    for key in ["encounter_id", "patient_nbr", "y"]:
        if not np.array_equal(scored[key].to_numpy(), baseline[key].to_numpy()):
            raise ValueError("Phase 3 validation predictions do not align with frozen validation")
    for name, source in BASELINES.items():
        scored[name] = baseline[source].to_numpy()
    scored.to_csv(prediction_path, index=False)
    manifest = dict(
        fit_manifest_sha256=fitted_hash,
        prediction_sha256=sha256(prediction_path),
        baseline_predictions_sha256=sha256(baseline_path),
        prediction_columns=[*models, *BASELINES],
        timings=timings,
        encounters=len(scored),
        patients=scored.patient_nbr.nunique(),
        evaluated_partition="validation",
        score_passes=1,
        test_evaluated=False,
    )
    write_json(manifest_path, manifest)
    return scored, manifest, fitted


def cluster_bootstrap(
    scored: pd.DataFrame, names: list[str], replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired patient draws shared by all AP and Brier comparisons."""
    codes, patients = pd.factorize(scored.patient_nbr, sort=True)
    y = scored.y.to_numpy()
    predictions = {name: scored[name].to_numpy() for name in names}
    rng = np.random.default_rng(seed)
    ap, brier = [], []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        weight = counts[codes]
        ap.append(
            {
                name: average_precision_score(y, p, sample_weight=weight)
                for name, p in predictions.items()
            }
        )
        brier.append(
            {name: np.average((p - y) ** 2, weights=weight) for name, p in predictions.items()}
        )
    return pd.DataFrame(ap), pd.DataFrame(brier)


def paired_comparison(
    name: str, reference: str, metrics: pd.DataFrame, ap: pd.DataFrame, brier: pd.DataFrame
) -> dict:
    a, b = metrics.loc[name], metrics.loc[reference]
    row = {
        "model": name,
        "reference": reference,
        "ap_difference": a.average_precision - b.average_precision,
        "brier_difference": a.brier - b.brier,
    }
    for label, samples in [("ap", ap), ("brier", brier)]:
        delta = samples[name] - samples[reference]
        row[f"{label}_ci_low"] = float(delta.quantile(0.025))
        row[f"{label}_ci_high"] = float(delta.quantile(0.975))
    return row


def choose_calibration(
    family: str, metrics: pd.DataFrame, comparisons: pd.DataFrame, policy: dict
) -> str:
    base = f"{family}__uncalibrated"
    eligible = [base]
    for method in ["sigmoid", "isotonic"]:
        name = f"{family}__{method}"
        difference = comparisons.loc[
            comparisons.model.eq(name) & comparisons.reference.eq(base)
        ].iloc[0]
        if (
            difference.brier_difference <= -policy["minimum_brier_improvement"]
            and difference.brier_ci_high < 0
            and difference.ap_difference >= -policy["maximum_ap_loss"]
        ):
            eligible.append(name)
    return min(eligible, key=lambda name: metrics.loc[name, "brier"])


def select_primary(
    logistic: str,
    boosting: str,
    metrics: pd.DataFrame,
    difference: dict,
    training: dict,
    policy: dict,
) -> dict:
    """Apply the written decision rule; numerical differences alone cannot select complexity."""
    lr, gb = training["logistic"], training["boosting"]
    checks = dict(
        validation_ap_gain=difference["ap_difference"] >= policy["minimum_boosting_ap_gain"],
        paired_ap_interval_positive=difference["ap_ci_low"] > 0,
        brier_acceptable=metrics.loc[boosting, "brier"] - metrics.loc[logistic, "brier"]
        <= policy["maximum_brier_disadvantage"],
        training_cv_mean=gb["cv_average_precision_mean"] >= lr["cv_average_precision_mean"],
        training_cv_stability=gb["cv_average_precision_sd"]
        <= policy["maximum_sd_ratio"] * lr["cv_average_precision_sd"],
    )
    use_boosting = all(checks.values())
    return dict(
        primary=boosting if use_boosting else logistic,
        challenger=logistic if use_boosting else boosting,
        boosting_selection_checks={k: bool(v) for k, v in checks.items()},
        operational_threshold_selected=False,
        production_model=False,
    )


def threshold_table(scored: pd.DataFrame, names: list[str], step: float) -> pd.DataFrame:
    """Fixed exploration grid; never choose a maximum-F1 or capacity threshold."""
    rows = []
    for name in names:
        for threshold in np.round(np.arange(0, 1 + step / 2, step), 8):
            metrics = classification_metrics(
                scored.y.to_numpy(), scored[name].to_numpy(), threshold
            )
            flags = scored[name].ge(threshold)
            rows.append(
                dict(
                    model=name,
                    **{
                        k: metrics[k]
                        for k in [
                            "threshold",
                            "recall",
                            "precision",
                            "f1",
                            "specificity",
                            "tn",
                            "fp",
                            "fn",
                            "tp",
                        ]
                    },
                    encounters_flagged=int(flags.sum()),
                    encounter_fraction_flagged=float(flags.mean()),
                    patients_flagged=scored.loc[flags, "patient_nbr"].nunique(),
                    patient_fraction_flagged=scored.loc[flags, "patient_nbr"].nunique()
                    / scored.patient_nbr.nunique(),
                )
            )
    return pd.DataFrame(rows)


def subgroup_table(scored: pd.DataFrame, names: list[str], policy: dict) -> pd.DataFrame:
    groups = [
        ("prior_inpatient", "none", scored.number_inpatient.eq(0)),
        ("prior_inpatient", "any", scored.number_inpatient.gt(0)),
        ("prior_inpatient", "3+", scored.number_inpatient.ge(3)),
    ]
    for column in ["race", "gender", "age"]:
        values = (
            scored[column]
            .astype("string")
            .fillna("Unknown")
            .replace({"?": "Unknown", "": "Unknown", "Unknown/Invalid": "Unknown"})
        )
        groups.extend((column, level, values.eq(level)) for level in sorted(values.unique()))
    rows = []
    for name in names:
        for group, level, mask in groups:
            subset = scored.loc[mask]
            if subset.empty:
                continue
            positives = int(subset.y.sum())
            adequate = (
                len(subset) >= policy["subgroup_min_encounters"]
                and min(positives, len(subset) - positives) >= policy["subgroup_min_positives"]
            )
            row = dict(
                model=name,
                group=group,
                level=level,
                encounters=len(subset),
                patients=subset.patient_nbr.nunique(),
                positives=positives,
                prevalence=positives / len(subset),
                estimates_suppressed=not adequate,
                threshold=policy["subgroup_threshold"],
            )
            if adequate:
                metrics = classification_metrics(
                    subset.y.to_numpy(), subset[name].to_numpy(), row["threshold"]
                )
                row.update(
                    {
                        k: metrics[k]
                        for k in [
                            "average_precision",
                            "recall",
                            "precision",
                            "brier",
                            "tn",
                            "fp",
                            "fn",
                            "tp",
                        ]
                    }
                )
                row["false_negative_rate"] = 1 - metrics["recall"]
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate() -> dict:
    root, policy = settings()
    scored, scoring, fitted = score_validation_once()
    artifacts, out = root / policy["artifact_dir"], root / policy["report_dir"]
    training = json.loads((artifacts / "training_selection.json").read_text())
    names = scoring["prediction_columns"]
    rows = []
    for name in names:
        row = dict(
            model=name, **classification_metrics(scored.y.to_numpy(), scored[name].to_numpy())
        )
        if name in fitted["models"]:
            record = fitted["models"][name]
            row.update(record)
            row["cv_scope"] = "uncalibrated base estimator; training selection CV"
            row.update(scoring["timings"][name])
            row.update(
                {
                    k: v
                    for k, v in training["selected"][record["family"]].items()
                    if k.startswith("cv_")
                }
            )
            row["parameters"] = json.dumps(record["parameters"], sort_keys=True)
        else:
            row.update(
                family="phase3", calibration="uncalibrated", configuration="archived baseline"
            )
        rows.append(row)
    metrics = pd.DataFrame(rows).set_index("model")
    LOGGER.info(
        "Computing paired validation uncertainty: %s patient bootstrap draws",
        policy["evaluation"]["bootstrap_replicates"],
    )
    ap, brier = cluster_bootstrap(
        scored, names, policy["evaluation"]["bootstrap_replicates"], policy["random_seed"]
    )
    metrics["ap_ci_low"], metrics["ap_ci_high"] = ap.quantile(0.025), ap.quantile(0.975)
    metrics["brier_ci_low"], metrics["brier_ci_high"] = brier.quantile(0.025), brier.quantile(0.975)
    pairs = [
        (f"{family}__{method}", f"{family}__uncalibrated")
        for family in ["logistic", "boosting"]
        for method in ["sigmoid", "isotonic"]
    ]
    pairs += [
        ("boosting__uncalibrated", "logistic__uncalibrated"),
        ("logistic__uncalibrated", "phase3_logistic_raw"),
        ("boosting__uncalibrated", "phase3_boosting"),
        ("phase3_boosting", "logistic__uncalibrated"),
        ("forest__uncalibrated", "logistic__uncalibrated"),
    ]
    comparisons = pd.DataFrame([paired_comparison(a, b, metrics, ap, brier) for a, b in pairs])
    choices = {
        family: choose_calibration(family, metrics, comparisons, policy["calibration"])
        for family in ["logistic", "boosting"]
    }
    difference = paired_comparison(choices["boosting"], choices["logistic"], metrics, ap, brier)
    selection = select_primary(
        choices["logistic"],
        choices["boosting"],
        metrics,
        difference,
        training["selected"],
        policy["selection"],
    )
    selected_names = [selection["primary"], selection["challenger"]]
    metrics.reset_index().to_csv(out / "validation_metrics.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    threshold_table(scored, selected_names, policy["evaluation"]["threshold_step"]).to_csv(
        out / "thresholds.csv", index=False
    )
    subgroup_table(scored, selected_names, policy["evaluation"]).to_csv(
        out / "subgroups.csv", index=False
    )
    reliability = []
    for name in fitted["models"]:
        bins = calibration_bins(scored.y, scored[name])
        bins.insert(0, "model", name)
        reliability.append(bins)
    pd.concat(reliability).to_csv(out / "calibration_bins.csv", index=False)
    tracker = LocalTracker(root, policy, training["git_commit"])
    previous_path = out / "summary.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    reused_runs = (
        previous.get("mlflow_validation_runs", {})
        if previous.get("scoring_manifest") == scoring
        else {}
    )
    run_ids = {}
    for name, record in fitted["models"].items():
        if name in reused_runs:
            run_ids[name] = reused_runs[name]
            continue
        metric = metrics.loc[name]
        run_ids[name] = tracker.log(
            f"validation-{name}",
            {
                "family": record["family"],
                "configuration": record["configuration"],
                "calibration": record["calibration"],
                "hyperparameters": record["parameters"],
                "artifact_path": str((artifacts / f"{name}.joblib").relative_to(root)),
                "model_sha256": record["sha256"],
                "partition": "validation",
            },
            {
                **{
                    f"validation_{key}": metric[key]
                    for key in [
                        "average_precision",
                        "roc_auc",
                        "brier",
                        "recall",
                        "precision",
                        "f1",
                    ]
                },
                **{
                    key: metric[key]
                    for key in [
                        "cv_average_precision_mean",
                        "cv_average_precision_sd",
                        "cv_roc_auc_mean",
                        "fit_seconds",
                        "inference_median_ms",
                    ]
                },
            },
        )
    frozen = verify_contract(root)
    summary = dict(
        phase=4,
        **selection,
        family_choices=choices,
        chosen_difference=difference,
        training=training["selected"],
        evaluated_partition="validation",
        test_evaluated=False,
        validation_score_passes=scoring["score_passes"],
        bootstrap_replicates=len(ap),
        threshold_selected=False,
        test_sha256=frozen["files"]["test.csv.gz"],
        scoring_manifest=scoring,
        mlflow_validation_runs=run_ids,
        training_git_commit=training["git_commit"],
    )
    write_json(out / "summary.json", summary)
    write_json(
        out / "selection.json", {**selection, "family_choices": choices, "comparison": difference}
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate()
