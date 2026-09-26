"""Fail closed before deserialization; fixtures never masquerade as the frozen model."""

import shutil
from pathlib import Path

import pytest

from readmit_iq.serving.contract import CONTRACT
from readmit_iq.serving.errors import ModelIntegrityError
from readmit_iq.serving.model_loader import load_frozen_model

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bundle(tmp_path):
    for relative in CONTRACT["files"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def test_missing_model_has_clear_restore_instruction(bundle, monkeypatch):
    monkeypatch.setattr(
        "readmit_iq.serving.model_loader.joblib.load",
        lambda *a: pytest.fail("Unpickled missing model"),
    )
    with pytest.raises(ModelIntegrityError, match="Restore the trusted frozen artifact"):
        load_frozen_model(bundle)


@pytest.mark.parametrize("fault", ["wrong_size", "wrong_hash"])
def test_corrupt_artifact_never_deserializes(bundle, monkeypatch, fault):
    path = bundle / "models/final/readmit_iq_logistic.joblib"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (4127 if fault == "wrong_hash" else 10))
    monkeypatch.setattr(
        "readmit_iq.serving.model_loader.joblib.load",
        lambda *a: pytest.fail("Unpickled unverified model"),
    )
    with pytest.raises(ModelIntegrityError) as caught:
        load_frozen_model(bundle)
    assert caught.value.code == "model_hash_mismatch"


@pytest.mark.parametrize(
    "relative",
    [
        "reports/modeling/final_model_specification.json",
        "reports/modeling/final_model_metadata.json",
        "configs/phase2.yaml",
        "src/readmit_iq/modeling/preprocessing.py",
        "src/readmit_iq/decision_support/ranking.py",
    ],
)
def test_changed_bundle_contract_fails_before_model_io(bundle, monkeypatch, relative):
    (bundle / relative).write_text("changed")
    monkeypatch.setattr(
        "readmit_iq.serving.model_loader.joblib.load",
        lambda *a: pytest.fail("Unpickled with invalid contract"),
    )
    with pytest.raises(ModelIntegrityError) as caught:
        load_frozen_model(bundle)
    assert caught.value.code == "bundle_integrity_failed"


def test_runtime_mismatch_rejected_before_deserialization(bundle, monkeypatch):
    monkeypatch.setattr(
        "readmit_iq.serving.model_loader.importlib.metadata.version", lambda name: "0.0"
    )
    with pytest.raises(ModelIntegrityError) as caught:
        load_frozen_model(bundle)
    assert caught.value.code == "runtime_incompatible"
