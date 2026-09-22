"""Leakage, clinical code boundaries, training-only vocabularies and fit/predict behavior."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from readmit_iq.data.load_data import load_id_mapping
from readmit_iq.modeling.diagnoses import diagnosis_group
from readmit_iq.modeling.features import (
    ADMIN_UNKNOWN,
    MEDICATIONS,
    FeatureBuilder,
    normalize_category,
)
from readmit_iq.modeling.metrics import classification_metrics
from readmit_iq.modeling.pipelines import build_pipeline
from readmit_iq.modeling.preprocessing import CategoryEncoder
from readmit_iq.modeling.settings import settings


@pytest.mark.parametrize(
    "code,expected",
    [
        ("250.83", "Diabetes"),
        ("250", "Diabetes"),
        ("249.9", "Other"),
        ("251", "Other"),
        ("390", "Circulatory"),
        ("459.9", "Circulatory"),
        ("460", "Respiratory"),
        ("785.4", "Circulatory"),
        ("786", "Respiratory"),
        ("787.1", "Digestive"),
        ("788", "Genitourinary"),
        ("519.9", "Respiratory"),
        ("520", "Digestive"),
        ("579.9", "Digestive"),
        ("580", "Genitourinary"),
        ("629.9", "Genitourinary"),
        ("630", "Other"),
        ("710", "Musculoskeletal"),
        ("739.9", "Musculoskeletal"),
        ("740", "Other"),
        ("140", "Neoplasms"),
        ("239.9", "Neoplasms"),
        ("800", "Injury"),
        ("999.9", "Injury"),
        ("V45.81", "Other"),
        ("E849.0", "Other"),
        ("38", "Other"),
        ("038.9", "Other"),
        ("?", "Unknown"),
        (pd.NA, "Unknown"),
        ("garbage", "Invalid"),
        ("1000", "Invalid"),
        ("0", "Invalid"),
        ("2500", "Invalid"),
    ],
)
def test_documented_diagnosis_boundaries(code, expected):
    assert diagnosis_group(code) == expected


@pytest.fixture
def encounters():
    n = 120
    f = pd.DataFrame(
        {
            "age": ["[50-60)", "[70-80)"] * (n // 2),
            "gender": ["Female", "Male"] * (n // 2),
            "admission_type_id": [1] * n,
            "admission_source_id": [7] * n,
            "discharge_disposition_id": [1, 3] * (n // 2),
            "medical_specialty": ["?"] * n,
            "time_in_hospital": [2, 6] * (n // 2),
            "number_inpatient": [0, 3] * (n // 2),
            "number_emergency": [1] * n,
            "number_outpatient": [2] * n,
            "diag_1": ["250.83", "428"] * (n // 2),
            "diag_2": ["V45.81"] * n,
            "diag_3": ["?"] * n,
            "number_diagnoses": [5] * n,
            "num_medications": [12] * n,
            "num_procedures": [0] * n,
            "num_lab_procedures": [40] * n,
            "change": ["Ch"] * n,
            "diabetesMed": ["Yes"] * n,
            "A1Cresult": ["None"] * n,
            "max_glu_serum": ["Norm"] * n,
            "payer_code": ["?"] * n,
            "encounter_id": range(n),
            "patient_nbr": range(n),
            "readmitted": ["<30"] * n,
            "readmitted_lt30": [1] * n,
            "future_encounters": [999] * n,
        }
    )
    for name in MEDICATIONS:
        f[name] = "No"
    f["insulin"] = "Up"
    f["metformin"] = "Steady"
    f["glyburide"] = "Down"
    return f


def test_feature_exclusions_and_no_target_dependence(encounters):
    builder = FeatureBuilder()
    first = builder.fit_transform(encounters)
    changed = encounters.assign(
        patient_nbr=0,
        encounter_id=0,
        readmitted="NO",
        readmitted_lt30=0,
        future_encounters=0,
        diag_1="999",
        payer_code="MC",
        insulin="No",
    )
    pd.testing.assert_frame_equal(first, builder.transform(changed))
    assert not {
        "patient_nbr",
        "encounter_id",
        "readmitted",
        "readmitted_lt30",
        "future_encounters",
        "diag_1",
        "insulin",
        "payer_code",
    } & set(first)


def test_utilization_uses_provided_history_and_preserves_counts(encounters):
    f = FeatureBuilder().fit_transform(encounters)
    assert f.total_prior_utilization.iloc[:2].to_list() == [3, 6]
    assert f.any_prior_inpatient.iloc[:2].to_list() == [0, 1]
    assert f.any_prior_emergency.eq(1).all() and f.any_prior_outpatient.eq(1).all()
    assert f.number_inpatient.iloc[:2].to_list() == [0, 3]
    assert not any(
        "prior" in c or c.startswith("number_")
        for c in FeatureBuilder("no_utilization").transform(encounters)
    )


def test_medication_counts_only_documented_active_changed_statuses(encounters):
    f = FeatureBuilder("encounter_summaries").fit_transform(encounters)
    assert f.active_diabetes_drug_fields.eq(3).all()
    assert f.changed_diabetes_drug_fields.eq(2).all()
    assert f.A1Cresult.eq("Not measured").all()
    encounters.loc[0, "insulin"] = "?"
    with pytest.raises(ValueError, match="Medication counts"):
        FeatureBuilder("encounter_summaries").transform(encounters)


def test_missing_numeric_contract_does_not_invent_values(encounters):
    encounters.loc[0, "time_in_hospital"] = np.nan
    with pytest.raises(ValueError, match="Numeric source contract"):
        FeatureBuilder().transform(encounters)


def test_administrative_unknowns_match_reviewed_source():
    from pathlib import Path

    maps = load_id_mapping(Path("data/raw/IDS_mapping.csv"))
    for name, codes in ADMIN_UNKNOWN.items():
        assert codes == {
            str(c)
            for c, label in maps[name].items()
            if label in {"NULL", "Not Available", "Not Mapped", "Unknown/Invalid"}
        }
    assert normalize_category(pd.Series(["None", "?", "Norm"]), "A1Cresult").tolist() == [
        "Not measured",
        "Unknown",
        "Norm",
    ]


@pytest.mark.parametrize("mode", ["onehot", "ordinal"])
def test_unseen_categories_do_not_change_training_vocabulary(mode):
    train = pd.DataFrame({"medical_specialty": ["common", "common", "rare", "Unknown"]})
    encoder = CategoryEncoder(mode, min_count=2).fit(train)
    categories = deepcopy(encoder.categories_)
    valid = pd.DataFrame({"medical_specialty": ["validation-only", "Unknown", "rare"]})
    result = encoder.transform(valid)
    assert result.shape[0] == 3
    assert encoder.n_fit_rows_ == 4 and encoder.categories_ == categories
    assert "validation-only" not in encoder.categories_[0] and "rare" not in encoder.categories_[0]
    grouped = encoder._group(valid)
    assert grouped.iloc[:, 0].tolist() == ["Other", "Unknown", "Other"]


@pytest.mark.parametrize("family", ["logistic", "forest", "boosting", "dummy"])
def test_pipeline_fit_predict_and_serialization(encounters, family, tmp_path):
    import joblib

    _, policy = settings()
    policy["modeling"]["forest"]["n_estimators"] = 3
    policy["modeling"]["boosting"]["max_iter"] = 3
    policy["modeling"]["rare_min_count"] = 2
    pipeline = build_pipeline({"family": family, "variant": "primary"}, policy)
    clone(pipeline)
    y = np.tile([0, 0, 0, 1], 30)
    with threadpool_limits(limits=1):
        pipeline.fit(encounters, y)
        valid = encounters.iloc[:4].assign(medical_specialty="validation-only")
        p = pipeline.predict_proba(valid)
        joblib.dump(pipeline, tmp_path / "pipeline.joblib")
        np.testing.assert_allclose(
            joblib.load(tmp_path / "pipeline.joblib").predict_proba(valid), p
        )
    assert p.shape == (4, 2) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert pipeline.named_steps["preprocess"].named_transformers_["categorical"].n_fit_rows_ == 120
    if family == "boosting":
        assert pipeline.named_steps["model"].early_stopping is False


def test_naive_metrics_and_reference_threshold():
    y = np.array([0, 0, 0, 1])
    m = classification_metrics(y, np.full(4, 0.25))
    assert m["average_precision"] == 0.25 and m["roc_auc"] == 0.5
    assert m["brier"] == 0.1875 and m["recall"] == 0 and m["specificity"] == 1
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 0, 1, 0)
