"""API contract doubles are explicitly distinct from the real frozen artifact."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from readmit_iq.serving.api import create_app
from readmit_iq.serving.contract import FEATURES
from readmit_iq.serving.model_loader import immutable, load_frozen_model

ROOT = Path(__file__).resolve().parents[2]


class ContractDouble:
    artifact_verified = False

    def __init__(self):
        metadata = json.loads((ROOT / "reports/modeling/final_model_metadata.json").read_text())
        metadata.update(version="contract-test-double", pipeline_sha256="not-a-frozen-artifact")
        self.metadata = immutable(metadata)
        self.calls = 0
        self.frames = []

    def predict_proba(self, frame):
        assert tuple(frame.columns) == FEATURES
        self.calls += 1
        self.frames.append(frame.copy())
        p = np.minimum(0.05 + 0.05 * frame.number_inpatient.to_numpy(), 0.9)
        return np.column_stack([1 - p, p])


@pytest.fixture
def fake():
    return ContractDouble()


@pytest.fixture
def client(fake):
    with TestClient(create_app(model_loader=lambda: fake)) as client:
        yield client


@pytest.fixture(scope="session")
def real_model():
    artifact = ROOT / "models/final/readmit_iq_logistic.joblib"
    if not artifact.exists():
        if os.environ.get("READMITIQ_REQUIRE_REAL_MODEL") == "1":
            pytest.fail("Required real frozen artifact is missing; restore it without training")
        pytest.skip(
            "Ignored frozen artifact unavailable; contract tests use an explicit test double"
        )
    return load_frozen_model(ROOT)
