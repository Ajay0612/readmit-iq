"""Synthetic tests of the committed freeze and one-time scorer; never access real test data."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from readmit_iq.data.download import sha256
from readmit_iq.decision_support import final_evaluation as final
from readmit_iq.decision_support import freeze
from readmit_iq.optimization.contract import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fail_io(*args, **kwargs):
    pytest.fail("A forbidden final-evaluation request reached data/model loading")


def test_missing_freeze_blocks_before_test_io(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "read_csv", fail_io)
    monkeypatch.setattr(final.joblib, "load", fail_io)
    with pytest.raises(ValueError, match="committed pre-test freeze"):
        final.score_once(tmp_path)


@pytest.fixture
def committed_freeze(tmp_path, monkeypatch):
    policy = {"targeting": {"fraction": 0.1}}
    specification = {"targeting": policy["targeting"]}
    write_json(tmp_path / freeze.SPECIFICATION, specification)
    protocol = tmp_path / freeze.PROTOCOL
    protocol.write_text("A fixed model and 10% policy, before test")
    code = tmp_path / "evaluation.py"
    code.write_text("# fixed implementation")
    manifest = tmp_path / "reports/modeling/split_manifest.json"
    manifest.write_text("{}")
    lock = {
        "pretest_code_commit": "code-commit",
        "protocol_sha256": sha256(protocol),
        "specification_sha256": sha256(tmp_path / freeze.SPECIFICATION),
        "split_manifest_sha256": sha256(manifest),
        "implementation": {"evaluation.py": sha256(code)},
        "policy": policy,
        "policy_sha256": digest(policy),
        "test_evaluated_at_freeze": False,
    }
    write_json(tmp_path / freeze.LOCK, lock)
    blobs = {
        f"freeze-commit:{path}": (tmp_path / path).read_bytes()
        for path in [freeze.PROTOCOL, freeze.SPECIFICATION, freeze.LOCK]
    }
    blobs["code-commit:evaluation.py"] = code.read_bytes()

    def git_stub(root, *args):
        if args[0] == "log":
            return b"freeze-commit\n"
        if args[0] == "show":
            return blobs[args[1]]
        if args[0] == "merge-base":
            assert args[1:] == ("--is-ancestor", "code-commit", "freeze-commit")
            return b""
        pytest.fail("Unexpected Git operation")

    monkeypatch.setattr(freeze, "git", git_stub)
    return tmp_path, lock, blobs


def test_committed_freeze_verifies_without_loading_any_partition(committed_freeze, monkeypatch):
    root, lock, _ = committed_freeze
    monkeypatch.setattr(pd, "read_csv", fail_io)
    monkeypatch.setattr(final.joblib, "load", fail_io)
    actual, _, commit = freeze.verify_freeze(root, require_models=False)
    assert actual == lock and commit == "freeze-commit"


@pytest.mark.parametrize(
    "path", [freeze.PROTOCOL, freeze.SPECIFICATION, freeze.LOCK, "evaluation.py"]
)
def test_changed_decision_or_implementation_is_rejected(committed_freeze, path):
    root, _, _ = committed_freeze
    p = root / path
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError):
        freeze.verify_freeze(root, require_models=False)


def test_uncommitted_protocol_is_rejected(committed_freeze, monkeypatch):
    root, _, _ = committed_freeze
    monkeypatch.setattr(freeze, "git", lambda *args: b"")
    with pytest.raises(ValueError, match="not been committed"):
        freeze.verify_freeze(root, require_models=False)


def test_missing_git_ancestry_is_rejected(committed_freeze, monkeypatch):
    root, _, _ = committed_freeze
    original = freeze.git

    def git_stub(root, *args):
        if args[0] == "merge-base":
            raise subprocess.CalledProcessError(1, args)
        return original(root, *args)

    monkeypatch.setattr(freeze, "git", git_stub)
    with pytest.raises(ValueError, match="ancestry"):
        freeze.verify_freeze(root, require_models=False)


def test_changed_frozen_model_bytes_are_rejected(committed_freeze, monkeypatch):
    root, lock, blobs = committed_freeze
    model = root / "candidate.joblib"
    model.write_bytes(b"frozen learned weights and preprocessing")
    spec = {
        "targeting": lock["policy"]["targeting"],
        "models": {"lr": {"path": "candidate.joblib"}},
        "library_versions": {},
    }
    write_json(root / freeze.SPECIFICATION, spec)
    lock.update(
        model_sha256={"lr": sha256(model)}, specification_sha256=sha256(root / freeze.SPECIFICATION)
    )
    write_json(root / freeze.LOCK, lock)
    for path in [freeze.SPECIFICATION, freeze.LOCK]:
        blobs[f"freeze-commit:{path}"] = (root / path).read_bytes()
    monkeypatch.setattr(freeze, "verify_contract", lambda root: {})
    freeze.verify_freeze(root, require_models=True)
    model.write_bytes(b"changed model or preprocessing")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        freeze.verify_freeze(root, require_models=True)


@pytest.fixture
def scoring_fixture(tmp_path, monkeypatch):
    n = 20
    frame = pd.DataFrame(
        {
            "encounter_id": [f"e{i}" for i in range(n)],
            "patient_nbr": [f"p{i // 2}" for i in range(n)],
            "readmitted_lt30": np.arange(n) % 2,
            "race": "Unknown",
            "payer_code": "?",
            "number_inpatient": np.arange(n) % 4,
        }
    )
    directory = tmp_path / "data/processed/phase3"
    directory.mkdir(parents=True)
    frame.to_csv(directory / "test.csv.gz", index=False)
    frame[["encounter_id", "patient_nbr"]].assign(partition="test").to_csv(
        directory / "assignments.csv.gz", index=False
    )
    policy = {
        "artifact_dir": "models/development/phase5",
        "report_dir": "reports/modeling/phase5",
        "primary": "lr",
        "threads": 2,
        "targeting": {"fraction": 0.1},
        "final_test": {"models": ["lr", "gb"]},
    }
    models = {
        name: {
            "path": f"{name}.joblib",
            "numeric_features": ["number_inpatient"],
            "categorical_features": [],
            "exact_estimator_parameters": {"fixed": 42},
        }
        for name in ["lr", "gb"]
    }
    for name in models:
        (tmp_path / f"{name}.joblib").write_bytes(b"original-model-bytes")
    lock = {
        "policy": policy,
        "pretest_code_commit": "code",
        "model_sha256": {name: sha256(tmp_path / f"{name}.joblib") for name in models},
        "frozen_files": {"test.csv.gz": sha256(directory / "test.csv.gz")},
    }
    write_json(tmp_path / freeze.LOCK, lock)
    spec = {"models": models}
    monkeypatch.setattr(final, "verify_freeze", lambda *args, **kwargs: (lock, spec, "freeze"))

    class Pipeline:
        def __init__(self):
            self.calls = 0
            self.params = {"fixed": 42}

        def __getitem__(self, key):
            assert key == -1
            return self

        def get_params(self, deep=False):
            return self.params

        def predict_proba(self, X):
            assert list(X) == ["number_inpatient"]
            self.calls += 1
            p = 0.05 + 0.1 * X.number_inpatient.to_numpy()
            return np.column_stack([1 - p, p])

    estimators = {name: Pipeline() for name in models}
    monkeypatch.setattr(final.joblib, "load", lambda path: estimators[path.stem])
    return tmp_path, lock, spec, estimators


def test_real_scorer_loads_synthetic_test_once_and_reuses_predictions(scoring_fixture, monkeypatch):
    root, _, _, models = scoring_fixture
    original = pd.read_csv
    reads = []

    def read(path, **kwargs):
        reads.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    first, manifest, *_ = final.score_once(root)
    second, again, *_ = final.score_once(root)
    assert reads.count("test.csv.gz") == 1
    assert manifest == again and manifest["test_table_loads"] == 1
    assert all(model.calls == 1 for model in models.values())
    np.testing.assert_array_equal(first.lr, second.lr)
    assert manifest["prediction_calls"] == {"lr": 1, "gb": 1}


@pytest.mark.parametrize("fault", ["test_bytes", "model_configuration", "incomplete", "published"])
def test_invalid_final_request_cannot_score(scoring_fixture, monkeypatch, fault):
    root, lock, _, models = scoring_fixture
    if fault == "test_bytes":
        (root / "data/processed/phase3/test.csv.gz").write_bytes(b"changed")
    elif fault == "model_configuration":
        models["lr"].params = {"fixed": 99}
    elif fault == "incomplete":
        write_json(root / lock["policy"]["artifact_dir"] / "test_evaluation_started.json", {})
    else:
        write_json(root / lock["policy"]["report_dir"] / "test/summary.json", {})
    monkeypatch.setattr(pd, "read_csv", fail_io)
    with pytest.raises(ValueError):
        final.score_once(root)
    assert all(model.calls == 0 for model in models.values())


@pytest.mark.parametrize("fault", ["prediction_bytes", "frozen_policy"])
def test_cache_tampering_cannot_trigger_rescoring(scoring_fixture, monkeypatch, fault):
    root, lock, _, models = scoring_fixture
    final.score_once(root)
    if fault == "prediction_bytes":
        (root / lock["policy"]["artifact_dir"] / "test_predictions.csv.gz").write_bytes(b"changed")
    else:
        (root / freeze.LOCK).write_text("changed freeze")
    monkeypatch.setattr(pd, "read_csv", fail_io)
    with pytest.raises(ValueError):
        final.score_once(root)
    assert all(model.calls == 1 for model in models.values())


@pytest.mark.parametrize("fault", [None, "metrics_bytes", "targeting", "missing_challenger"])
def test_ci_verifier_uses_only_archived_results_and_checks_integrity(tmp_path, monkeypatch, fault):
    policy = {
        "report_dir": "reports/modeling/phase5",
        "targeting": {"fraction": 0.1},
        "final_test": {"models": ["lr", "gb"]},
    }
    lock = {"policy": policy}
    write_json(tmp_path / freeze.LOCK, lock)
    out = tmp_path / policy["report_dir"] / "test"
    summary = {
        "targeting": policy["targeting"],
        "model_refitted": False,
        "decision_changes_after_test": False,
        "scoring": {"test_table_loads": 1, "prediction_calls": {"lr": 1, "gb": 1}},
    }
    if fault == "missing_challenger":
        summary["scoring"]["prediction_calls"].pop("gb")
    write_json(out / "summary.json", summary)
    (out / "metrics.csv").write_text("model,average_precision\nlr,0.2\n")
    metadata = tmp_path / "reports/modeling/final_model_metadata.json"
    write_json(metadata, {})
    publication = {
        "freeze_sha256": sha256(tmp_path / freeze.LOCK),
        "freeze_commit": "freeze",
        "targeting_sha256": digest(policy["targeting"]),
        "metadata_sha256": sha256(metadata),
        "files": {str(p.relative_to(tmp_path)): sha256(p) for p in out.iterdir()},
    }
    if fault == "targeting":
        publication["targeting_sha256"] = "changed"
    if fault == "metrics_bytes":
        (out / "metrics.csv").write_text("changed metric")
    write_json(out / "publication_manifest.json", publication)
    monkeypatch.setattr(final, "verify_freeze", lambda *args, **kwargs: (lock, {}, "freeze"))
    monkeypatch.setattr(pd, "read_csv", fail_io)
    monkeypatch.setattr(final.joblib, "load", fail_io)
    if fault:
        with pytest.raises(ValueError):
            final.verify_published_results(tmp_path)
    else:
        assert final.verify_published_results(tmp_path) == summary
