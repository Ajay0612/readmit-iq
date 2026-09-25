"""Explicit one-time scoring of the original test, followed only by cached analysis."""

import argparse
import json
import shutil
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from readmit_iq.analysis.reporting import write_json
from readmit_iq.data.download import sha256, verify_file
from readmit_iq.decision_support.context import settings
from readmit_iq.decision_support.evaluation import evaluate_predictions
from readmit_iq.decision_support.freeze import LOCK, SPECIFICATION, verify_freeze
from readmit_iq.optimization.contract import digest


def score_once(root):
    """Called only by the explicit final-eval command after a verified Git freeze."""
    lock, spec, freeze_commit = verify_freeze(root, require_models=True)
    policy = lock["policy"]
    artifacts = root / policy["artifact_dir"]
    artifacts.mkdir(parents=True, exist_ok=True)
    prediction_path = artifacts / "test_predictions.csv.gz"
    manifest_path = artifacts / "test_scoring_manifest.json"
    start_path = artifacts / "test_evaluation_started.json"
    freeze_hash = sha256(root / LOCK)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["freeze_sha256"] != freeze_hash
            or manifest["model_sha256"] != lock["model_sha256"]
        ):
            raise ValueError("Cached final predictions belong to a different freeze")
        verify_file(prediction_path, manifest["prediction_sha256"])
        return (
            pd.read_csv(
                prediction_path,
                dtype={"encounter_id": "string", "patient_nbr": "string"},
                float_precision="round_trip",
                keep_default_na=False,
                na_values=[""],
            ),
            manifest,
            lock,
            spec,
        )
    if (root / policy["report_dir"] / "test/summary.json").exists():
        raise ValueError("Published final results exist; a fresh checkout may not rescore test")
    if start_path.exists():
        raise ValueError("Incomplete one-time test attempt requires review; no automatic retry")

    # Load verified model bytes before test. Hashes bind parameters, weights and preprocessing.
    models = {name: joblib.load(root / item["path"]) for name, item in spec["models"].items()}
    expected_names = policy["final_test"]["models"]
    if set(models) != set(expected_names):
        raise ValueError("Frozen model registry changed")
    for name, model in models.items():
        if model[-1].get_params(deep=False) != spec["models"][name]["exact_estimator_parameters"]:
            raise ValueError("Estimator parameters differ from the pre-test specification")
    test_path = root / "data/processed/phase3/test.csv.gz"
    verify_file(test_path, lock["frozen_files"]["test.csv.gz"])
    with start_path.open("x") as stream:
        json.dump(
            dict(
                started_utc=datetime.now(UTC).isoformat(),
                freeze_sha256=freeze_hash,
                freeze_commit=freeze_commit,
                models=expected_names,
            ),
            stream,
            indent=2,
        )
    features = (
        spec["models"][policy["primary"]]["numeric_features"]
        + spec["models"][policy["primary"]]["categorical_features"]
    )
    columns = ["encounter_id", "patient_nbr", "readmitted_lt30", "race", "payer_code", *features]
    frame = pd.read_csv(
        test_path,
        usecols=columns,
        dtype={"encounter_id": "string", "patient_nbr": "string"},
        keep_default_na=False,
        na_values=[""],
    ).rename(columns={"readmitted_lt30": "y"})
    if not frame.encounter_id.is_unique or frame.patient_nbr.isna().any():
        raise ValueError("Invalid frozen test keys")
    assignments = pd.read_csv(
        root / "data/processed/phase3/assignments.csv.gz",
        dtype={"encounter_id": "string", "patient_nbr": "string"},
    )
    assigned = assignments.loc[assignments.partition.eq("test")]
    if set(zip(frame.encounter_id, frame.patient_nbr, strict=True)) != set(
        zip(assigned.encounter_id, assigned.patient_nbr, strict=True)
    ):
        raise ValueError("Test rows do not match the frozen assignments")
    X = frame[features]
    prediction_calls = {}
    for name in expected_names:
        with threadpool_limits(limits=policy["threads"]):
            probabilities = models[name].predict_proba(X)
        if probabilities.shape != (len(frame), 2) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid frozen inference output")
        frame[name] = probabilities[:, 1]
        prediction_calls[name] = 1
    frame.to_csv(prediction_path, index=False)
    manifest = dict(
        completed_utc=datetime.now(UTC).isoformat(),
        freeze_commit=freeze_commit,
        pretest_code_commit=lock["pretest_code_commit"],
        freeze_sha256=freeze_hash,
        model_sha256=lock["model_sha256"],
        test_file_sha256=sha256(test_path),
        prediction_sha256=sha256(prediction_path),
        test_table_loads=1,
        prediction_calls=prediction_calls,
        encounters=len(frame),
        patients=frame.patient_nbr.nunique(),
        targeting=policy["targeting"],
        model_refitted=False,
        calibration_changed=False,
    )
    write_json(manifest_path, manifest)
    return frame, manifest, lock, spec


def run_final_evaluation():
    root, _ = settings()
    frame, scoring, lock, spec = score_once(root)
    policy = lock["policy"]
    out = root / policy["report_dir"] / "test"
    summary = evaluate_predictions(
        frame,
        {name: frame[name].to_numpy() for name in policy["final_test"]["models"]},
        policy,
        out,
        "test",
    )
    summary.update(
        freeze_commit=scoring["freeze_commit"],
        pretest_code_commit=scoring["pretest_code_commit"],
        scoring=scoring,
        decision_changes_after_test=False,
    )
    write_json(out / "summary.json", summary)
    # Copy original bytes; never serialize a refitted or changed pipeline as final.
    destination = root / "models/final/readmit_iq_logistic.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / spec["models"][policy["primary"]]["path"], destination)
    verify_file(destination, lock["model_sha256"][policy["primary"]])
    metadata = dict(
        version=spec["version"],
        primary=policy["primary"],
        inference_pipeline=str(destination.relative_to(root)),
        pipeline_sha256=sha256(destination),
        specification_sha256=sha256(root / SPECIFICATION),
        features=spec["models"][policy["primary"]]["numeric_features"]
        + spec["models"][policy["primary"]]["categorical_features"],
        training_commit=spec["training_commit"],
        freeze_commit=scoring["freeze_commit"],
        library_versions=spec["library_versions"],
        targeting=policy["targeting"],
        validation_metrics=spec["validation_metrics"],
        test_metrics=summary["metrics"],
        clinical_deployment_validated=False,
        model_refitted_after_test=False,
    )
    write_json(root / "reports/modeling/final_model_metadata.json", metadata)
    write_json(destination.with_suffix(".metadata.json"), metadata)
    result_files = [
        p for p in out.iterdir() if p.is_file() and p.name != "publication_manifest.json"
    ]
    publication = dict(
        freeze_sha256=sha256(root / LOCK),
        freeze_commit=scoring["freeze_commit"],
        files={str(p.relative_to(root)): sha256(p) for p in result_files},
        metadata_sha256=sha256(root / "reports/modeling/final_model_metadata.json"),
        primary=policy["primary"],
        targeting_sha256=digest(policy["targeting"]),
        test_table_loads=1,
        prediction_calls=scoring["prediction_calls"],
        test_refitting_performed=False,
    )
    write_json(out / "publication_manifest.json", publication)
    print(json.dumps(summary, indent=2))
    return summary


def verify_published_results(root):
    """Read aggregate evidence only. Safe for CI and notebook 06; no partition/model loading."""
    lock, _, freeze_commit = verify_freeze(root, require_models=False)
    out = root / lock["policy"]["report_dir"] / "test"
    published = json.loads((out / "publication_manifest.json").read_text())
    if (
        published["freeze_sha256"] != sha256(root / LOCK)
        or published["freeze_commit"] != freeze_commit
    ):
        raise ValueError("Published results do not belong to the committed freeze")
    if published["targeting_sha256"] != digest(lock["policy"]["targeting"]):
        raise ValueError("Published targeting differs from the frozen policy")
    for path, expected in published["files"].items():
        if not (root / path).resolve().is_relative_to(out.resolve()):
            raise ValueError("Unexpected final result path")
        verify_file(root / path, expected)
    verify_file(root / "reports/modeling/final_model_metadata.json", published["metadata_sha256"])
    summary = json.loads((out / "summary.json").read_text())
    scoring = summary["scoring"]
    if set(scoring["prediction_calls"]) != set(lock["policy"]["final_test"]["models"]):
        raise ValueError("Published prediction calls omit a frozen model")
    if summary["targeting"] != lock["policy"]["targeting"]:
        raise ValueError("Final results changed the selected policy")
    if scoring["test_table_loads"] != 1 or any(
        value != 1 for value in scoring["prediction_calls"].values()
    ):
        raise ValueError("Final evaluation was not the declared one-time comparison")
    if summary["decision_changes_after_test"] or summary["model_refitted"]:
        raise ValueError("Final results contain a prohibited development change")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-frozen-test", action="store_true")
    args = parser.parse_args()
    if not args.execute_frozen_test:
        parser.error("Real test scoring requires the explicit --execute-frozen-test command")
    run_final_evaluation()
