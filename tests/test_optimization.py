"""Patient-disjoint tuning, restricted features, train-only calibration and local tracking."""

from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from readmit_iq.optimization.contract import BLOCKED, load_partition, settings, verify_contract
from readmit_iq.optimization.cv import grouped_folds
from readmit_iq.optimization.fit import calibrated_pipeline
from readmit_iq.optimization.pipelines import DevelopmentFeatures, build_pipeline, schema
from readmit_iq.optimization.tracking import LocalTracker
from readmit_iq.optimization.tune import choose_configuration, cross_validate, optimize


@pytest.fixture
def population():
    n = 800
    i = np.arange(n)
    return pd.DataFrame(
        {
            "encounter_id": [f"e{v:04}" for v in i],
            "patient_nbr": [f"p{v // 2:04}" for v in i],
            "readmitted_lt30": ((i // 2) % 10 == 0).astype(int),
            "readmitted": "NO",
            "age": "[50-60)",
            "gender": np.where(i % 2, "Male", "Female"),
            "admission_type_id": 1,
            "admission_source_id": 7,
            "discharge_disposition_id": 1,
            "medical_specialty": np.where(i % 3, "?", "Cardiology"),
            "time_in_hospital": 1 + i % 6,
            "number_inpatient": i % 4,
            "number_emergency": i % 3,
            "number_outpatient": i % 2,
            "number_diagnoses": 5,
            "diag_1": "250.83",
            "diag_2": "428",
            "diag_3": "?",
            "race": "Caucasian",
            "future_encounters": 100,
        }
    )


def test_group_cv_complete_balanced_and_order_invariant(population):
    folds, table, fingerprint = grouped_folds(population)
    assert len(folds) == 5 and len(table) == 10
    holdouts = np.concatenate([b for _, b in folds])
    assert sorted(holdouts) == list(range(len(population)))
    for a, b in folds:
        assert set(population.patient_nbr.iloc[a]).isdisjoint(population.patient_nbr.iloc[b])
    assert table.patient_overlap.eq(0).all()
    assert (table.prevalence - population.readmitted_lt30.mean()).abs().max() <= 0.01
    _, _, reordered = grouped_folds(population.sample(frac=1, random_state=123))
    assert reordered == fingerprint


@pytest.mark.parametrize("fault", ["missing_patient", "duplicate_encounter", "one_class"])
def test_cv_rejects_invalid_population(population, fault):
    if fault == "missing_patient":
        population.loc[0, "patient_nbr"] = None
    elif fault == "duplicate_encounter":
        population.loc[1, "encounter_id"] = population.loc[0, "encounter_id"]
    else:
        population["readmitted_lt30"] = 0
    with pytest.raises(ValueError):
        grouped_folds(population)


@pytest.mark.parametrize(
    "partition,purpose",
    [
        ("test", "tuning"),
        ("test", "calibration"),
        ("test", "evaluation"),
        ("validation", "tuning"),
        ("validation", "calibration"),
        ("train", "evaluation"),
    ],
)
def test_illegal_partition_access_rejected_before_io(tmp_path, monkeypatch, partition, purpose):
    def fail_read(*args, **kwargs):
        pytest.fail("Forbidden request reached data loading")

    monkeypatch.setattr(pd, "read_csv", fail_read)
    with pytest.raises(ValueError, match="forbidden"):
        load_partition(tmp_path, partition, purpose)


def test_validation_requires_fitted_candidate_seal(tmp_path):
    with pytest.raises(ValueError, match="seal"):
        load_partition(tmp_path, "validation", "evaluation")


def test_phase4_never_reconstructs_missing_partitions(tmp_path):
    with pytest.raises(FileNotFoundError, match="no resplit"):
        verify_contract(tmp_path)


@pytest.mark.parametrize(
    "configuration", ["raw", "indicators", "raw_engineered", "diagnosis_grouped", "diagnosis_raw"]
)
def test_configuration_integrity_and_excluded_fields(population, configuration):
    model = DevelopmentFeatures(configuration)
    output = model.fit_transform(population)
    numeric, categorical = schema(configuration)
    assert list(output) == numeric + categorical
    assert not set(BLOCKED + ["race", "future_encounters"]) & set(output)
    changed = population.assign(
        patient_nbr="changed", readmitted_lt30=1, readmitted="<30", future_encounters=999
    )
    pd.testing.assert_frame_equal(output, model.transform(changed))
    if configuration == "raw":
        assert "number_inpatient" in output and "total_prior_utilization" not in output
    if configuration == "indicators":
        assert "number_inpatient" not in output and "total_prior_utilization" not in output
        assert set(output.any_prior_inpatient.unique()) <= {0, 1}
    if configuration.startswith("diagnosis"):
        assert "total_prior_utilization" not in output


def test_diagnosis_cannot_win_primary_feature_selection():
    _, policy = settings()
    table = pd.DataFrame(
        {
            "configuration": ["raw", "indicators", "raw_engineered", "diagnosis_raw"],
            "cv_average_precision_mean": [0.20, 0.19, 0.2005, 0.99],
        }
    )
    assert choose_configuration(table, policy) == "raw"
    table.loc[2, "cv_average_precision_mean"] = 0.21
    assert choose_configuration(table, policy) == "raw_engineered"


def test_cv_objective_is_deterministic_and_records_fit_rows(population):
    _, policy = settings()
    folds, _, _ = grouped_folds(population)
    X, y = population.drop(columns=BLOCKED), population.readmitted_lt30.to_numpy()
    first, rows = cross_validate("logistic", "raw", {"C": 0.1}, X, y, folds, policy)
    second, _ = cross_validate("logistic", "raw", {"C": 0.1}, X, y, folds, policy)
    assert first == pytest.approx(second, abs=1e-12)
    assert all(row["fit_encounters"] == 640 for row in rows)


def test_sequential_optuna_suggestions_and_scores_reproduce(population):
    _, original = settings()
    policy = deepcopy(original)
    policy["tuning"]["logistic"]["trials"] = 3
    folds, _, _ = grouped_folds(population)
    X, y = population.drop(columns=BLOCKED), population.readmitted_lt30.to_numpy()

    class Tracker:
        def log(self, *args, **kwargs):
            return "unit-test-run"

    a = optimize("logistic", "raw", X, y, folds, policy, Tracker())
    b = optimize("logistic", "raw", X, y, folds, policy, Tracker())
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert [t.value for t in a.trials] == pytest.approx([t.value for t in b.trials], abs=1e-12)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_training_group_calibration_and_serialization(population, tmp_path, method):
    folds, _, _ = grouped_folds(population)
    X, y = population.drop(columns=BLOCKED), population.readmitted_lt30.to_numpy()
    model = calibrated_pipeline(build_pipeline("logistic", "raw"), method, folds)
    assert model.ensemble is False and model.cv is folds
    with threadpool_limits(limits=2):
        model.fit(X, y)
        predicted = model.predict_proba(X)
    assert len(model.calibrated_classifiers_) == 1
    encoder = (
        model.calibrated_classifiers_[0]
        .estimator.named_steps["preprocess"]
        .named_transformers_["categorical"]
    )
    assert encoder.n_fit_rows_ == len(population)
    assert np.isfinite(predicted).all() and np.allclose(predicted.sum(axis=1), 1)
    path = tmp_path / "candidate.joblib"
    joblib.dump(model, path)
    with threadpool_limits(limits=2):
        np.testing.assert_allclose(joblib.load(path).predict_proba(X), predicted, atol=1e-12)


def test_mlflow_is_local_explicit_and_records_provenance(tmp_path, monkeypatch):
    _, policy = settings()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://must-not-be-used.invalid")
    tracker = LocalTracker(tmp_path, policy, "test-commit")
    assert tracker.uri.startswith("sqlite:///") and str(tmp_path) in tracker.uri
    run_id = tracker.log("unit", {"family": "logistic", "configuration": "raw"}, {"cv_ap": 0.2})
    run = tracker.client.get_run(run_id)
    assert run.data.metrics["cv_ap"] == 0.2
    assert run.data.tags["git_commit"] == "test-commit"
    assert run.data.tags["seed"] == "42"
    assert run.info.status == "FINISHED"


def test_mlflow_refuses_external_artifact_directory(tmp_path):
    _, policy = settings()
    policy["tracking_dir"] = "../escape"
    with pytest.raises(ValueError, match="inside"):
        LocalTracker(tmp_path, policy, "test")
