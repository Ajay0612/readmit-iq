"""Opt-in real-artifact checks: hand-authored synthetic encounters, never test records."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from readmit_iq.decision_support.ranking import target_mask
from readmit_iq.serving.api import create_app
from readmit_iq.serving.contract import FEATURES
from readmit_iq.serving.examples import LOW_HISTORY, synthetic_batch

pytestmark = pytest.mark.real_artifact


def test_real_artifact_api_preprocessing_and_probability_parity(real_model):
    seen = []

    def observe(frame):
        seen.append(frame.copy())
        return real_model.predict_proba(frame)

    observed = SimpleNamespace(
        metadata=real_model.metadata, artifact_verified=True, predict_proba=observe
    )
    payload = synthetic_batch(100)
    direct_frame = pd.DataFrame([r["features"] for r in payload["records"]], columns=FEATURES)
    expected_matrix = real_model._pipeline[:-1].transform(direct_frame)
    direct = real_model._pipeline.predict_proba(direct_frame)[:, 1]
    with TestClient(create_app(model_loader=lambda: observed)) as client:
        first = client.post("/v1/predict/batch", json=payload)
        assert first.status_code == 200
        second = client.post("/v1/predict/batch", json=payload)
        assert first.json() == second.json()
        assert client.get("/v1/model").json()["artifact_verified"] is True
    pd.testing.assert_frame_equal(seen[0], direct_frame)
    api_matrix = real_model._pipeline[:-1].transform(seen[0])
    np.testing.assert_array_equal(api_matrix.toarray(), expected_matrix.toarray())
    probabilities = [p["readmission_probability"] for p in first.json()["predictions"]]
    np.testing.assert_allclose(probabilities, direct, rtol=0, atol=1e-12)


def test_real_single_probability_and_missing_normalization(real_model):
    with TestClient(create_app(model_loader=lambda: real_model)) as client:
        records = [
            LOW_HISTORY,
            dict(
                LOW_HISTORY,
                medical_specialty=None,
                gender="?",
                admission_type_id=6,
                admission_source_id=17,
            ),
        ]
        for record in records:
            response = client.post("/v1/predict", json=record)
            assert response.status_code == 200
            direct = real_model._pipeline.predict_proba(pd.DataFrame([record], columns=FEATURES))[
                0, 1
            ]
            assert response.json()["readmission_probability"] == pytest.approx(direct, abs=1e-12)
            assert response.json()["outreach_selected"] is None


def test_real_batch_policy_matches_original_phase5_function(real_model):
    payload = synthetic_batch(1000)
    frame = pd.DataFrame([r["features"] for r in payload["records"]], columns=FEATURES)
    p = real_model._pipeline.predict_proba(frame)[:, 1]
    ids = [r["request_id"] for r in payload["records"]]
    expected = {
        key for key, chosen in zip(ids, target_mask(p, ids, 0.1, 42), strict=True) if chosen
    }
    with TestClient(create_app(model_loader=lambda: real_model)) as client:
        response = client.post("/v1/prioritize", json=payload)
    assert response.status_code == 200
    actual = {r["request_id"] for r in response.json()["predictions"] if r["selected_for_outreach"]}
    assert actual == expected and len(actual) == 100


def test_real_metadata_is_deeply_immutable(real_model):
    with pytest.raises(TypeError):
        real_model.metadata["version"] = "changed"
    with pytest.raises(TypeError):
        real_model.metadata["targeting"]["fraction"] = 0.5
