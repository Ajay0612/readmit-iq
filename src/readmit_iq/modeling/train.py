"""Fixed Phase 3 experiment registry; train only, validate only, never evaluate the test."""

import logging
import time
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from readmit_iq.analysis.reporting import write_json
from readmit_iq.config import configure_logging
from readmit_iq.data.download import sha256
from readmit_iq.modeling.error_analysis import error_groups
from readmit_iq.modeling.metrics import bootstrap_ap, calibration_bins, classification_metrics
from readmit_iq.modeling.pipelines import build_pipeline
from readmit_iq.modeling.settings import settings
from readmit_iq.modeling.splitting import load_development, prepare_frozen_partitions

LOGGER = logging.getLogger(__name__)
BLOCKED = ["encounter_id", "patient_nbr", "readmitted", "readmitted_lt30"]


def run_experiments() -> dict:
    configure_logging()
    root, policy = settings()
    lock = prepare_frozen_partitions()
    train = load_development("train")
    validation = load_development("validation")
    assert set(train.patient_nbr).isdisjoint(validation.patient_nbr)
    X_train, X_valid = train.drop(columns=BLOCKED), validation.drop(columns=BLOCKED)
    y_train, y_valid = train.readmitted_lt30.to_numpy(), validation.readmitted_lt30.to_numpy()
    out = root / policy["modeling"]["report_dir"]
    artifacts = root / policy["modeling"]["artifact_dir"]
    artifacts.mkdir(parents=True, exist_ok=True)
    results, predictions, diagnostics = [], {}, []
    for experiment in policy["experiments"]:
        name = experiment["name"]
        LOGGER.info("Fitting %s: %s / %s", name, experiment["family"], experiment["variant"])
        pipeline = build_pipeline(experiment, policy)
        start = time.perf_counter()
        with (
            warnings.catch_warnings(record=True) as caught,
            threadpool_limits(limits=policy["modeling"]["threads"]),
        ):
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_valid)[:, 1]
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError(f"{name} failed convergence; resolve before comparing scores")
        elapsed = time.perf_counter() - start
        predictions[name] = probability
        metrics = classification_metrics(y_valid, probability, policy["modeling"]["threshold"])
        encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
        if encoder.n_fit_rows_ != len(train):
            raise ValueError("Preprocessing fitted on unexpected rows")
        n_features = len(pipeline.named_steps["preprocess"].get_feature_names_out())
        results.append(
            {
                **experiment,
                **metrics,
                "fit_predict_seconds": elapsed,
                "encoded_features": n_features,
            }
        )
        joblib.dump(pipeline, artifacts / f"{name}.joblib", compress=3)
        diagnostics.append(
            dict(
                name=name,
                warnings=[str(w.message) for w in caught],
                train_rows=len(train),
                validation_rows=len(validation),
                preprocessing_fit_rows=encoder.n_fit_rows_,
                model_parameters=pipeline.named_steps["model"].get_params(),
            )
        )
        if name == "logistic":
            pd.DataFrame(encoder.frequency_records_).to_csv(
                out / "training_category_frequencies.csv", index=False
            )
            coefficients = pd.DataFrame(
                {
                    "feature": pipeline.named_steps["preprocess"].get_feature_names_out(),
                    "coefficient": pipeline.named_steps["model"].coef_[0],
                }
            )
            coefficients.to_csv(out / "logistic_coefficients.csv", index=False)
        if name in {"logistic_diagnosis_raw", "logistic_encounter_summaries"}:
            pd.DataFrame(encoder.frequency_records_).to_csv(
                out / f"{name}_training_categories.csv", index=False
            )
        LOGGER.info(
            "%s validation AP=%.4f Brier=%.4f (%0.1fs)",
            name,
            metrics["average_precision"],
            metrics["brier"],
            elapsed,
        )
    results = pd.DataFrame(results).sort_values("average_precision", ascending=False)
    # Same patient bootstrap draws for every experiment; paired deltas retain covariance.
    LOGGER.info(
        "Computing %s paired patient bootstrap replicates",
        policy["modeling"]["bootstrap_replicates"],
    )
    bootstrap = bootstrap_ap(
        y_valid,
        validation.patient_nbr,
        predictions,
        policy["modeling"]["bootstrap_replicates"],
        policy["random_seed"],
    )
    results["ap_ci_low"] = results.name.map(bootstrap.quantile(0.025))
    results["ap_ci_high"] = results.name.map(bootstrap.quantile(0.975))
    results.to_csv(out / "baseline_model_results.csv", index=False)
    pairs = [
        ("logistic", "naive_prevalence", "Logistic vs prevalence"),
        ("logistic", "logistic_no_utilization", "All prior utilization"),
        ("logistic", "logistic_raw_utilization", "Add total and any-use indicators"),
        ("logistic", "logistic_no_disposition", "Discharge destination"),
        ("logistic_balanced", "logistic", "Balanced class weights"),
        ("random_forest", "logistic", "Random forest vs logistic"),
        ("histogram_boosting", "logistic", "Boosting vs logistic"),
        ("logistic_diagnosis_groups", "logistic", "Add grouped diagnoses (uncertain timing)"),
        (
            "logistic_diagnosis_raw",
            "logistic_diagnosis_groups",
            "Raw vs grouped diagnosis representation",
        ),
        ("logistic_encounter_summaries", "logistic", "Add encounter summaries (uncertain timing)"),
    ]
    by_name = results.set_index("name")
    differences = []
    for high, low, question in pairs:
        delta = bootstrap[high] - bootstrap[low]
        differences.append(
            dict(
                comparison=question,
                model=high,
                reference=low,
                ap_difference=by_name.loc[high, "average_precision"]
                - by_name.loc[low, "average_precision"],
                ci_low=delta.quantile(0.025),
                ci_high=delta.quantile(0.975),
            )
        )
    differences = pd.DataFrame(differences)
    differences.to_csv(out / "feature_ablation_results.csv", index=False)
    primary = results.loc[results.role.eq("primary")]
    winner = primary.iloc[0]["name"]
    # Rank families using the unweighted reference for clean model-family comparisons.
    families = results.loc[
        results.name.isin(["logistic", "random_forest", "histogram_boosting"])
    ].copy()
    calibration_names = list(
        dict.fromkeys([winner, "logistic", "random_forest", "logistic_balanced"])
    )
    calibration = []
    for name in calibration_names:
        bins = calibration_bins(y_valid, predictions[name])
        bins.insert(0, "model", name)
        calibration.append(bins)
    pd.concat(calibration).to_csv(out / "calibration_bins.csv", index=False)
    errors = error_groups(validation, predictions[winner], policy["modeling"]["threshold"])
    errors.to_csv(out / "validation_error_groups.csv", index=False)
    # Individual predictions and fitted objects are development artifacts, never committed data.
    pd.DataFrame(
        {
            "encounter_id": validation.encounter_id,
            "patient_nbr": validation.patient_nbr,
            "y": y_valid,
            **predictions,
        }
    ).to_csv(artifacts / "validation_predictions.csv.gz", index=False)
    summary = dict(
        phase=3,
        models_fitted=len(results),
        evaluated_partition="validation",
        test_evaluated=False,
        threshold=policy["modeling"]["threshold"],
        threshold_optimized=False,
        development_winner=winner,
        primary_family_ranking=families.name.tolist(),
        primary_feature_variant="primary",
        sensitivity_models_eligible_for_selection=False,
        train_encounters=len(train),
        validation_encounters=len(validation),
        seed=policy["random_seed"],
        sklearn_version=sklearn.__version__,
        split_manifest_sha256=sha256(root / policy["split"]["lock_report"]),
        frozen_test_sha256=lock["files"]["test.csv.gz"],
        experiment_policy_sha256=sha256(root / "configs/phase3.yaml"),
        implementation_sha256={
            name: sha256(root / "src/readmit_iq/modeling" / name)
            for name in [
                ("train.py"),
                ("features.py"),
                ("preprocessing.py"),
                ("pipelines.py"),
                ("metrics.py"),
                ("diagnoses.py"),
            ]
        },
        bootstrap=dict(
            replicates=policy["modeling"]["bootstrap_replicates"], unit="patient", paired=True
        ),
        diagnostics=diagnostics,
        tracking="Fixed 11-experiment YAML registry plus CSV/JSON and joblib; MLflow deferred",
    )
    write_json(out / "phase3_summary.json", summary)
    return summary


if __name__ == "__main__":
    run_experiments()
