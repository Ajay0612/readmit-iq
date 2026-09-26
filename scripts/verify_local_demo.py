"""Smoke the actual local Docker/API/UI boundary with synthetic inputs only."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from readmit_iq.decision_support.ranking import ranked_indices
from readmit_iq.serving.contract import FEATURES
from readmit_iq.serving.examples import LOW_HISTORY, synthetic_batch
from readmit_iq.serving.model_loader import load_frozen_model


def main():
    root = Path(__file__).resolve().parents[1]
    model = load_frozen_model(root)
    url = "http://127.0.0.1:8000"
    statuses = {}
    for endpoint in ["/health", "/v1/model", "/docs", "/openapi.json"]:
        response = requests.get(url + endpoint, timeout=10)
        statuses[endpoint] = response.status_code
        assert response.status_code == 200
    metadata = requests.get(url + "/v1/model", timeout=10).json()
    assert metadata["artifact_verified"]
    assert metadata["artifact_sha256"] == model.metadata["pipeline_sha256"]
    response = requests.post(url + "/v1/predict", json=LOW_HISTORY, timeout=10)
    statuses["/v1/predict"] = response.status_code
    assert response.status_code == 200 and response.json()["outreach_selected"] is None
    differences = []
    selections = {}
    for count in [9, 20, 1000]:
        batch = synthetic_batch(count)
        frame = pd.DataFrame([r["features"] for r in batch["records"]], columns=FEATURES)
        direct = model.predict_proba(frame)[:, 1]
        response = requests.post(url + "/v1/predict/batch", json=batch, timeout=20)
        statuses["/v1/predict/batch"] = response.status_code
        assert response.status_code == 200
        actual = np.array([r["readmission_probability"] for r in response.json()["predictions"]])
        np.testing.assert_allclose(actual, direct, atol=1e-12, rtol=0)
        differences.append(float(np.max(np.abs(actual - direct))))
        response = requests.post(url + "/v1/prioritize", json=batch, timeout=20)
        statuses["/v1/prioritize"] = response.status_code
        assert response.status_code == 200
        ranking = response.json()
        identifiers = [r["request_id"] for r in batch["records"]]
        order = ranked_indices(direct, identifiers, seed=42)
        assert [r["request_id"] for r in ranking["predictions"]] == [identifiers[i] for i in order]
        assert ranking["number_selected"] == count // 10
        selections[str(count)] = ranking["number_selected"]
    ineligible = requests.post(
        url + "/v1/predict", json={**LOW_HISTORY, "discharge_disposition_id": 11}, timeout=10
    )
    assert ineligible.status_code == 422 and ineligible.json()["error"] == "ineligible_encounter"
    ui = requests.get("http://127.0.0.1:8501/_stcore/health", timeout=10)
    assert ui.status_code == 200
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "synthetic_inputs_only": True,
        "real_artifact_verified": True,
        "artifact_sha256": metadata["artifact_sha256"],
        "endpoint_http_status": statuses,
        "streamlit_health_status": ui.status_code,
        "ineligible_status": ineligible.status_code,
        "host_container_max_probability_difference": max(differences),
        "probability_absolute_tolerance": 1e-12,
        "original_ranking_identical": True,
        "batch_selections": selections,
        "limitations": "Local loopback, serial synthetic workload; no clinical or load validation",
    }
    output = root / "reports/serving/local_container_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
