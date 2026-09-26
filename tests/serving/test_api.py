"""Strict transport, privacy, eligibility and original ranking without real data access."""

from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from readmit_iq.decision_support.ranking import ranked_indices
from readmit_iq.serving.api import create_app, logger
from readmit_iq.serving.contract import MAX_BATCH, MAX_BODY_BYTES
from readmit_iq.serving.errors import ModelIntegrityError
from readmit_iq.serving.examples import LOW_HISTORY, synthetic_batch
from readmit_iq.serving.logging import JsonFormatter


def test_health_and_metadata_identify_contract_double(client):
    assert client.get("/health").json() == dict(
        status="ok", model_loaded=True, model_version="contract-test-double"
    )
    metadata = client.get("/v1/model").json()
    assert metadata["feature_count"] == 10
    assert metadata["artifact_verified"] is False
    assert metadata["model_version"] == "contract-test-double"
    assert metadata["targeting_policy"]["fraction"] == 0.1
    assert "/Users/" not in str(metadata) and "inference_pipeline" not in metadata


def test_single_is_deterministic_and_has_no_outreach_decision(client, fake):
    first = client.post("/v1/predict", json=LOW_HISTORY)
    second = client.post("/v1/predict", json=LOW_HISTORY)
    assert first.status_code == 200 and first.json() == second.json()
    assert first.json()["outreach_selected"] is None
    assert first.json()["readmission_probability"] == 0.05
    assert first.json()["risk_percent"] == 5
    assert fake.calls == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("number_inpatient", -1),
        ("number_emergency", 1.5),
        ("number_outpatient", True),
        ("time_in_hospital", 0),
        ("time_in_hospital", 15),
        ("time_in_hospital", "3"),
        ("number_inpatient", 2**31),
        ("age", "50-60"),
        ("gender", "not-a-code"),
        ("admission_type_id", 99),
        ("admission_type_id", "1"),
        ("admission_type_id", True),
        ("admission_source_id", 16),
        ("discharge_disposition_id", 999),
        ("medical_specialty", "invented-specialty"),
        ("medical_specialty", 2),
    ],
)
def test_invalid_fields_rejected_before_model(client, fake, field, value):
    payload = dict(LOW_HISTORY, **{field: value})
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"
    assert fake.calls == 0


@pytest.mark.parametrize("field", list(LOW_HISTORY))
def test_each_feature_is_required(client, fake, field):
    payload = dict(LOW_HISTORY)
    payload.pop(field)
    assert client.post("/v1/predict", json=payload).status_code == 422
    assert fake.calls == 0


@pytest.mark.parametrize(
    "field", ["race", "patient_nbr", "encounter_id", "readmitted", "diag_1", "payer_code"]
)
def test_extra_features_are_forbidden(client, fake, field):
    assert (
        client.post("/v1/predict", json=dict(LOW_HISTORY, **{field: "PRIVATE"})).status_code == 422
    )
    assert fake.calls == 0


@pytest.mark.parametrize("missing", [None, "?", "", "Unknown"])
def test_documented_missing_categories_are_explicit(client, missing):
    payload = dict(
        LOW_HISTORY,
        gender=missing,
        medical_specialty=missing,
        admission_source_id=missing,
        admission_type_id=missing,
    )
    assert client.post("/v1/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "code,reason",
    [
        (11, "death"),
        (13, "hospice"),
        (2, "inpatient_transfer"),
        (9, "discharge_not_confirmed"),
        (18, "unknown_destination"),
    ],
)
def test_eligibility_rejects_entire_batch_before_inference(client, fake, code, reason):
    payload = synthetic_batch()
    payload["records"][7]["features"]["discharge_disposition_id"] = code
    response = client.post("/v1/prioritize", json=payload)
    assert response.status_code == 422 and fake.calls == 0
    assert response.json()["error"] == "ineligible_encounter"
    assert response.json()["details"][0]["code"] == reason
    assert response.json()["details"][0]["location"] == ["records", 7]


@pytest.mark.parametrize("code", [1, 3, 4, 6, 7, 8, 16, 17, 22, 24])
def test_original_included_destinations_remain_eligible(client, code):
    assert (
        client.post(
            "/v1/predict", json=dict(LOW_HISTORY, discharge_disposition_id=code)
        ).status_code
        == 200
    )


def test_batch_predictions_preserve_input_order_and_do_not_decide_outreach(client):
    payload = synthetic_batch()
    response = client.post("/v1/predict/batch", json=payload).json()
    assert [r["request_id"] for r in response["predictions"]] == [
        r["request_id"] for r in payload["records"]
    ]
    assert all(r["outreach_selected"] is None for r in response["predictions"])


@pytest.mark.parametrize("n", [1, 9, 10, 11, 19, 20, 99, 1000])
def test_prioritization_exact_floor_and_original_ties(client, n):
    payload = synthetic_batch(n)
    first = client.post("/v1/prioritize", json=payload)
    assert first.status_code == 200
    first = first.json()
    assert first["number_selected"] == n // 10
    assert sum(r["selected_for_outreach"] for r in first["predictions"]) == n // 10
    p = np.array([0.05 + 0.05 * r["features"]["number_inpatient"] for r in payload["records"]])
    ids = [r["request_id"] for r in payload["records"]]
    expected = [ids[i] for i in ranked_indices(p, ids, 42)]
    assert [r["request_id"] for r in first["predictions"]] == expected
    shuffled = deepcopy(payload)
    shuffled["records"].reverse()
    assert client.post("/v1/prioritize", json=shuffled).json() == first


@pytest.mark.parametrize("fault", ["empty", "too_many", "duplicate", "missing_id", "outcome"])
def test_malformed_batches(client, fake, fault):
    payload = synthetic_batch()
    if fault == "empty":
        payload["records"] = []
    if fault == "too_many":
        payload = synthetic_batch(MAX_BATCH + 1)
    if fault == "duplicate":
        payload["records"][1]["request_id"] = payload["records"][0]["request_id"]
    if fault == "missing_id":
        payload["records"][0].pop("request_id")
    if fault == "outcome":
        payload["records"][0]["y"] = 1
    assert client.post("/v1/prioritize", json=payload).status_code == 422
    assert fake.calls == 0


def test_body_limit_and_malformed_json(client, fake):
    assert client.post("/v1/predict", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413
    assert client.post("/v1/predict", content=b'{"age":').status_code == 422
    assert fake.calls == 0


def test_model_load_once_for_multiple_requests(fake):
    calls = []

    def loader():
        calls.append(1)
        return fake

    with TestClient(create_app(model_loader=loader)) as client:
        for _ in range(3):
            assert client.post("/v1/predict", json=LOW_HISTORY).status_code == 200
    assert calls == [1]


@pytest.mark.parametrize("code", ["model_missing", "model_hash_mismatch", "runtime_incompatible"])
def test_model_failure_is_unhealthy_and_never_falls_back(code):
    def loader():
        raise ModelIntegrityError(code, "Restore the verified frozen artifact.")

    with TestClient(create_app(model_loader=loader)) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/health").json()["model_loaded"] is False
        result = client.post("/v1/predict", json=LOW_HISTORY)
        assert result.status_code == 503 and result.json()["error"] == code


@pytest.mark.parametrize("fault", ["raise", "nan", "wrong_shape", "invalid_probabilities"])
def test_inference_errors_do_not_expose_payload_or_trace(client, fake, fault, monkeypatch):
    def broken(frame):
        if fault == "raise":
            raise ValueError("PRIVATE-PAYLOAD-123")
        if fault == "nan":
            return [[np.nan, np.nan]]
        if fault == "wrong_shape":
            return [[0.2]]
        return [[-1, 2]]

    monkeypatch.setattr(fake, "predict_proba", broken)
    result = client.post("/v1/predict", json=LOW_HISTORY)
    assert result.status_code == 500
    assert "PRIVATE" not in result.text and "Traceback" not in result.text


def test_logs_and_validation_errors_omit_inputs_identifiers_and_unknown_field_names(client, caplog):
    logger.addHandler(caplog.handler)
    try:
        payload = dict(LOW_HISTORY, **{"PRIVATE-FIELD": "PRIVATE-VALUE"})
        result = client.post("/v1/predict", json=payload)
        assert "PRIVATE" not in result.text
        client.post("/v1/prioritize", json=synthetic_batch())
        logs = "\n".join(JsonFormatter().format(record) for record in caplog.records)
        assert "PRIVATE" not in logs and "demo-" not in logs
        assert "number_inpatient" not in logs and "record_count" in logs
    finally:
        logger.removeHandler(caplog.handler)


def test_openapi_documents_required_fields_and_responses(client):
    doc = client.get("/openapi.json").json()
    assert set(doc["components"]["schemas"]["EncounterFeatures"]["required"]) == set(LOW_HISTORY)
    for path in ["/v1/predict", "/v1/predict/batch", "/v1/prioritize"]:
        assert {"200", "422", "503", "500"} <= set(doc["paths"][path]["post"]["responses"])
        schema = doc["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert schema["examples"]
    assert client.get("/docs").status_code == 200
