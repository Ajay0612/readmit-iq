"""Prespecified calibration/selection rules, patient uncertainty and diagnostic-only thresholds."""

import json

import numpy as np
import pandas as pd
import pytest

from readmit_iq.data.download import sha256
from readmit_iq.optimization import evaluate as evaluation
from readmit_iq.optimization.contract import digest, settings, verify_contract
from readmit_iq.optimization.evaluate import (
    choose_calibration,
    cluster_bootstrap,
    select_primary,
    subgroup_table,
    threshold_table,
)


def test_calibration_requires_brier_uncertainty_and_preserved_discrimination():
    _, policy = settings()
    names = ["logistic__uncalibrated", "logistic__sigmoid", "logistic__isotonic"]
    metrics = pd.DataFrame({"brier": [0.10, 0.099, 0.098]}, index=names)
    comparisons = pd.DataFrame(
        [
            dict(
                model=names[1],
                reference=names[0],
                brier_difference=-0.001,
                brier_ci_high=-0.0001,
                ap_difference=0,
            ),
            dict(
                model=names[2],
                reference=names[0],
                brier_difference=-0.002,
                brier_ci_high=-0.0002,
                ap_difference=-0.01,
            ),
        ]
    )
    assert choose_calibration("logistic", metrics, comparisons, policy["calibration"]) == names[1]
    comparisons.loc[0, "brier_ci_high"] = 0.0001
    assert choose_calibration("logistic", metrics, comparisons, policy["calibration"]) == names[0]


@pytest.mark.parametrize(
    "violation", [None, "small_gain", "uncertain", "brier", "cv_mean", "cv_sd"]
)
def test_primary_selection_obeys_every_prespecified_condition(violation):
    _, policy = settings()
    metrics = pd.DataFrame({"brier": [0.095, 0.094]}, index=["lr", "gb"])
    difference = {"ap_difference": 0.01, "ap_ci_low": 0.002}
    training = {
        "logistic": {"cv_average_precision_mean": 0.20, "cv_average_precision_sd": 0.01},
        "boosting": {"cv_average_precision_mean": 0.21, "cv_average_precision_sd": 0.01},
    }
    if violation == "small_gain":
        difference["ap_difference"] = 0.002
    elif violation == "uncertain":
        difference["ap_ci_low"] = -0.001
    elif violation == "brier":
        metrics.loc["gb", "brier"] = 0.10
    elif violation == "cv_mean":
        training["boosting"]["cv_average_precision_mean"] = 0.19
    elif violation == "cv_sd":
        training["boosting"]["cv_average_precision_sd"] = 0.03
    result = select_primary("lr", "gb", metrics, difference, training, policy["selection"])
    assert result["primary"] == ("gb" if violation is None else "lr")
    assert result["challenger"] != result["primary"]
    assert result["operational_threshold_selected"] is False
    assert result["production_model"] is False


def test_thresholds_count_encounters_and_unique_patients_without_selection():
    f = pd.DataFrame(
        {"patient_nbr": ["a", "a", "b", "c"], "y": [1, 0, 1, 0], "model": [0.8, 0.7, 0.4, 0.1]}
    )
    result = threshold_table(f, ["model"], 0.5)
    middle = result.loc[result.threshold.eq(0.5)].iloc[0]
    assert middle.encounters_flagged == 2 and middle.patients_flagged == 1
    assert middle.recall == 0.5 and middle.precision == 0.5
    assert list(result.threshold) == [0, 0.5, 1]


def test_patient_bootstrap_is_paired_deterministic_and_cluster_weighted():
    f = pd.DataFrame(
        {
            "patient_nbr": ["a", "a", "b", "b", "c", "c"],
            "y": [0, 1, 0, 1, 0, 1],
            "a": [0.1, 0.8, 0.2, 0.7, 0.4, 0.6],
        }
    )
    f["b"] = f.a
    ap, brier = cluster_bootstrap(f, ["a", "b"], 30, 42)
    again, _ = cluster_bootstrap(f, ["a", "b"], 30, 42)
    pd.testing.assert_frame_equal(ap, again)
    assert np.array_equal(ap.a, ap.b) and np.array_equal(brier.a, brier.b)
    # Every draw gives equal weight to each pair's positive and negative encounter.
    expected_possible_brier = {
        (x * 0.025 + y * 0.065 + z * 0.16) / 3
        for x in range(4)
        for y in range(4)
        for z in range(4)
        if x + y + z == 3
    }
    for value in brier.a:
        assert min(abs(value - reference) for reference in expected_possible_brier) < 1e-12


def test_small_demographic_groups_are_suppressed_but_counts_retained():
    _, policy = settings()
    f = pd.DataFrame(
        {
            "patient_nbr": ["a", "b", "c", "d"],
            "y": [0, 1, 0, 1],
            "model": [0.1, 0.2, 0.3, 0.4],
            "number_inpatient": [0, 0, 1, 3],
            "race": ["?", "Caucasian", "Asian", "Asian"],
            "gender": "Female",
            "age": "[50-60)",
        }
    )
    table = subgroup_table(f, ["model"], policy["evaluation"])
    assert table.estimates_suppressed.all()
    assert "Unknown" in table.level.values
    assert table.loc[table.group.eq("race"), "encounters"].sum() == len(f)
    assert "average_precision" not in table


@pytest.mark.parametrize("fault", [None, "changed_fit", "changed_predictions", "incomplete"])
def test_validation_cache_prevents_new_data_loading_or_rescoring(tmp_path, monkeypatch, fault):
    """Cached predictions are reusable only with the exact sealed models and bytes."""
    fitted = {"models": {"logistic__uncalibrated": {"sha256": "sealed-model"}}}
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    predictions = artifacts / "validation_predictions.csv.gz"
    pd.DataFrame({"patient_nbr": ["001"], "encounter_id": ["002"], "y": [1]}).to_csv(
        predictions, index=False
    )
    manifest = {
        "fit_manifest_sha256": digest(fitted),
        "prediction_sha256": sha256(predictions),
        "score_passes": 1,
    }
    if fault == "changed_fit":
        fitted["models"]["logistic__uncalibrated"]["sha256"] = "different-model"
    elif fault == "changed_predictions":
        predictions.write_bytes(b"changed cached predictions")
    if fault == "incomplete":
        (artifacts / "validation_started.json").write_text("{}")
    else:
        (artifacts / "validation_scores.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(evaluation, "settings", lambda: (tmp_path, {"artifact_dir": "artifacts"}))
    monkeypatch.setattr(evaluation, "fit_candidates", lambda: fitted)

    def forbidden(*args, **kwargs):
        pytest.fail("A cached/incomplete scoring attempt must not reopen data or load models")

    monkeypatch.setattr(evaluation, "load_partition", forbidden)
    monkeypatch.setattr(evaluation.joblib, "load", forbidden)
    if fault:
        with pytest.raises(ValueError):
            evaluation.score_validation_once()
    else:
        result, actual, _ = evaluation.score_validation_once()
        assert result.patient_nbr.tolist() == ["001"]
        assert actual["score_passes"] == 1


def test_contract_checks_hashes_and_assignments_without_parsing_any_partition(
    tmp_path, monkeypatch
):
    directory = tmp_path / "data/processed/phase3"
    reports = tmp_path / "reports/modeling"
    directory.mkdir(parents=True)
    reports.mkdir(parents=True)
    for name in ["train", "validation", "test"]:
        (directory / f"{name}.csv.gz").write_bytes(b"opaque partition, never parsed")
    pd.DataFrame(
        {
            "encounter_id": ["e1", "e2", "e3"],
            "patient_nbr": ["p1", "p2", "p3"],
            "partition": ["train", "validation", "test"],
        }
    ).to_csv(directory / "assignments.csv.gz", index=False)
    manifest = {
        "files": {p.name: sha256(p) for p in directory.iterdir()},
        "eligible_encounters": 3,
        "eligible_patients": 3,
        "patient_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }
    (directory / "lock.json").write_text(json.dumps(manifest))
    (reports / "split_manifest.json").write_text(json.dumps(manifest))
    (reports / "phase3_summary.json").write_text(
        json.dumps({"test_evaluated": False, "evaluated_partition": "validation"})
    )
    original_read = pd.read_csv
    reads = []

    def assignments_only(path, **kwargs):
        assert path.name == "assignments.csv.gz", "Partition parser must never be invoked"
        reads.append(path.name)
        return original_read(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", assignments_only)
    assert verify_contract(tmp_path) == manifest
    assert reads == ["assignments.csv.gz"]
    (directory / "test.csv.gz").write_bytes(b"unexpected byte change")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_contract(tmp_path)
