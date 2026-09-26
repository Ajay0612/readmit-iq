"""Synthetic-only parity, policy and warmed service latency evidence; no data loading."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from readmit_iq.decision_support.freeze import verify_freeze
from readmit_iq.decision_support.ranking import ranked_indices
from readmit_iq.serving.api import create_app
from readmit_iq.serving.contract import CONTRACT, FEATURES
from readmit_iq.serving.examples import LOW_HISTORY, PATTERNS, synthetic_batch
from readmit_iq.serving.model_loader import load_frozen_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/serving/verification.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_freeze(root, require_models=False)
    model = load_frozen_model(root)
    results = {}
    with TestClient(create_app(model_loader=lambda: model)) as client:
        assert client.get("/health").json()["model_loaded"]
        assert client.get("/v1/model").json()["artifact_verified"]
        for size in [1, 100, 1000]:
            payload = LOW_HISTORY if size == 1 else synthetic_batch(size)
            endpoint = "/v1/predict" if size == 1 else "/v1/prioritize"
            samples = []
            for iteration in range(25):
                start = time.perf_counter()
                response = client.post(endpoint, json=payload)
                elapsed = (time.perf_counter() - start) * 1000
                assert response.status_code == 200
                if iteration >= 5:
                    samples.append(elapsed)
            results[str(size)] = {
                "endpoint": endpoint,
                "median_ms": round(statistics.median(samples), 3),
                "p95_ms": round(float(np.percentile(samples, 95)), 3),
                "minimum_ms": round(min(samples), 3),
                "maximum_ms": round(max(samples), 3),
                "request_bytes": len(json.dumps(payload).encode()),
            }
        payload = synthetic_batch(1000)
        frame = pd.DataFrame([r["features"] for r in payload["records"]], columns=FEATURES)
        direct = model.predict_proba(frame)[:, 1]
        first = client.post("/v1/predict/batch", json=payload).json()
        second = client.post("/v1/predict/batch", json=payload).json()
        actual = np.array([p["readmission_probability"] for p in first["predictions"]])
        np.testing.assert_allclose(actual, direct, atol=1e-12, rtol=0)
        assert first == second
        ranked = client.post("/v1/prioritize", json=payload).json()
        identifiers = [r["request_id"] for r in payload["records"]]
        order = ranked_indices(direct, identifiers, seed=42)
        assert [p["request_id"] for p in ranked["predictions"]] == [identifiers[i] for i in order]
        assert ranked["number_selected"] == 100
        small = client.post("/v1/prioritize", json=synthetic_batch(9)).json()
        assert small["number_selected"] == 0
        examples = {
            name: client.post("/v1/predict", json=features).json()["readmission_probability"]
            for name, features in PATTERNS.items()
        }
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "artifact_verified": True,
        "artifact_sha256": model.metadata["pipeline_sha256"],
        "serving_contract_sha256": hashlib.sha256(
            (root / "src/readmit_iq/serving/contract.json").read_bytes()
        ).hexdigest(),
        "model_version": model.metadata["version"],
        "protected_phase5_freeze_unchanged": True,
        "data_used": "Hand-authored synthetic demonstration inputs only",
        "final_test_scored": False,
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "libraries": {p: importlib.metadata.version(p) for p in CONTRACT["runtime_packages"]},
        },
        "parity": {
            "records": 1000,
            "absolute_tolerance": 1e-12,
            "maximum_probability_difference": float(np.max(np.abs(actual - direct))),
            "repeated_response_identical": True,
            "original_ranking_identical": True,
            "selected_of_1000": 100,
            "selected_of_9": 0,
        },
        "synthetic_pattern_probabilities": examples,
        "benchmark": {
            "scope": "TestClient full request validation, inference and JSON response; no network",
            "warmup_requests_per_size": 5,
            "measured_requests_per_size": 20,
            "concurrency": 1,
            "native_threads": 2,
            "sizes": results,
            "limitation": "Local warmed synthetic workload; not a throughput or clinical benchmark",
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
