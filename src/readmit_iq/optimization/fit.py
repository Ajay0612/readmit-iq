"""Refit and cross-fit calibrators on training only, then seal all candidates before scoring."""

import json
import logging
import time
import warnings

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from readmit_iq.analysis.reporting import write_json
from readmit_iq.data.download import sha256, verify_file
from readmit_iq.optimization.contract import (
    BLOCKED,
    digest,
    load_partition,
    settings,
    verify_contract,
)
from readmit_iq.optimization.cv import grouped_folds
from readmit_iq.optimization.pipelines import build_pipeline
from readmit_iq.optimization.tune import run_tuning

LOGGER = logging.getLogger(__name__)


def calibrated_pipeline(pipeline, method: str, folds: list):
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Only the prespecified training-only calibrators are allowed")
    return CalibratedClassifierCV(pipeline, method=method, cv=folds, ensemble=False, n_jobs=1)


def fit_candidates() -> dict:
    root, policy = settings()
    selection = run_tuning()
    artifacts = root / policy["artifact_dir"]
    path = artifacts / "fit_manifest.json"
    selection_hash = digest(selection)
    if path.exists():
        fitted = json.loads(path.read_text())
        if fitted["training_selection_sha256"] != selection_hash:
            raise ValueError("Fitted candidate contract changed")
        for name, record in fitted["models"].items():
            verify_file(artifacts / f"{name}.joblib", record["sha256"])
        return fitted
    if (artifacts / "validation_scores.json").exists():
        raise ValueError("Cannot refit candidates after validation scoring")
    frame = load_partition(root, "train", "calibration")
    folds, _, fold_hash = grouped_folds(
        frame, policy["cv"]["folds"], policy["random_seed"], policy["cv"]["prevalence_tolerance"]
    )
    if fold_hash != selection["fold_membership_sha256"]:
        raise ValueError("Calibration folds differ from the verified training folds")
    X, y = frame.drop(columns=BLOCKED), frame.readmitted_lt30.to_numpy()
    models = {}
    for family, chosen in selection["selected"].items():
        methods = policy["calibration"]["methods"] if family != "forest" else ["uncalibrated"]
        for method in methods:
            name = f"{family}__{method}"
            LOGGER.info("Training-only final fit: %s", name)
            pipeline = build_pipeline(
                family,
                chosen["configuration"],
                chosen["parameters"],
                policy["random_seed"],
                policy["features"]["rare_min_count"],
            )
            model = (
                pipeline
                if method == "uncalibrated"
                else calibrated_pipeline(pipeline, method, folds)
            )
            start = time.perf_counter()
            with threadpool_limits(limits=policy["threads"]), warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X, y)
            seconds = time.perf_counter() - start
            fitted_pipeline = (
                model if method == "uncalibrated" else model.calibrated_classifiers_[0].estimator
            )
            encoder = fitted_pipeline.named_steps["preprocess"].named_transformers_["categorical"]
            if encoder.n_fit_rows_ != len(frame):
                raise ValueError("Final preprocessing did not fit full training only")
            artifact = artifacts / f"{name}.joblib"
            joblib.dump(model, artifact, compress=3)
            models[name] = dict(
                family=family,
                calibration=method,
                configuration=chosen["configuration"],
                parameters=chosen["parameters"],
                sha256=sha256(artifact),
                artifact_bytes=artifact.stat().st_size,
                fit_seconds=seconds,
                fit_encounters=len(frame),
                encoded_features=len(
                    fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
                ),
            )
    verify_contract(root)
    fitted = dict(
        training_selection_sha256=selection_hash,
        models=models,
        fold_membership_sha256=fold_hash,
        calibration_partition="train",
        validation_opened=False,
        test_evaluated=False,
    )
    write_json(path, fitted)
    write_json(root / policy["report_dir"] / "fit_manifest.json", fitted)
    return fitted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit_candidates()
