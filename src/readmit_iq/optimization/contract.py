"""Phase 4 provenance, strict frozen-data checks and purpose-limited development loading."""

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from readmit_iq.data.download import sha256, verify_file
from readmit_iq.modeling.settings import settings as phase3_settings
from readmit_iq.modeling.splitting import STRING_COLUMNS

BLOCKED = ["encounter_id", "patient_nbr", "readmitted", "readmitted_lt30"]


def settings() -> tuple[Path, dict]:
    root, _ = phase3_settings()
    policy = yaml.safe_load((root / "configs/phase4.yaml").read_text())
    if policy["phase"] != 4 or policy["random_seed"] != 42:
        raise ValueError("Phase 4 and seed 42 are required")
    return root, policy


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def verify_contract(root: Path) -> dict:
    """Hash existing bytes; never reconstruct partitions or parse the test table."""
    report = root / "reports/modeling/split_manifest.json"
    directory = root / "data/processed/phase3"
    if not (directory / "lock.json").exists():
        raise FileNotFoundError("Phase 4 requires the existing verified frozen split; no resplit")
    frozen = json.loads(report.read_text())
    if json.loads((directory / "lock.json").read_text()) != frozen:
        raise ValueError("Frozen split manifest/lock mismatch")
    if set(frozen["files"]) != {
        "train.csv.gz",
        "validation.csv.gz",
        "test.csv.gz",
        "assignments.csv.gz",
    }:
        raise ValueError("Unexpected frozen artifact inventory")
    for name, expected in frozen["files"].items():
        verify_file(directory / name, expected)
    assignment = pd.read_csv(
        directory / "assignments.csv.gz", dtype={"encounter_id": "string", "patient_nbr": "string"}
    )
    if (
        len(assignment) != frozen["eligible_encounters"]
        or not assignment.encounter_id.is_unique
        or assignment.patient_nbr.isna().any()
        or assignment.patient_nbr.nunique() != frozen["eligible_patients"]
        or not assignment.groupby("patient_nbr").partition.nunique().eq(1).all()
        or set(assignment.partition) != {"train", "validation", "test"}
        or any(frozen["patient_overlap"].values())
    ):
        raise ValueError("Frozen patient isolation/coverage failed")
    previous = json.loads((root / "reports/modeling/phase3_summary.json").read_text())
    if previous["test_evaluated"] or previous["evaluated_partition"] != "validation":
        raise ValueError("Prior evaluation record violates the development contract")
    return frozen


def load_partition(root: Path, partition: str, purpose: str) -> pd.DataFrame:
    """Optimization/calibration may load train only; validation needs sealed fitted models."""
    allowed = {"tuning": "train", "calibration": "train", "evaluation": "validation"}
    if allowed.get(purpose) != partition:
        raise ValueError("Phase 4 test access or partition/purpose mismatch is forbidden")
    if (
        purpose == "evaluation"
        and not (root / "models/development/phase4/fit_manifest.json").exists()
    ):
        raise ValueError("Fit and seal all candidates before opening validation")
    frozen = json.loads((root / "reports/modeling/split_manifest.json").read_text())
    path = root / "data/processed/phase3" / f"{partition}.csv.gz"
    verify_file(path, frozen["files"][path.name])
    return pd.read_csv(
        path, keep_default_na=False, na_values=[""], dtype=dict.fromkeys(STRING_COLUMNS, "string")
    )


def training_fingerprint(root: Path, policy: dict, frozen: dict) -> dict:
    """A finished training stage cannot be silently reused after changing its inputs or code."""
    import importlib.metadata

    modules = ["contract.py", "cv.py", "pipelines.py", "tracking.py", "tune.py", "fit.py"]
    files = {f"optimization/{p}": sha256(root / "src/readmit_iq/optimization" / p) for p in modules}
    for name in ["features.py", "preprocessing.py", "diagnoses.py", "metrics.py"]:
        files[f"modeling/{name}"] = sha256(root / "src/readmit_iq/modeling" / name)
    return {
        "policy": policy,
        "split_manifest_sha256": sha256(root / "reports/modeling/split_manifest.json"),
        "frozen_files": frozen["files"],
        "implementation": files,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "optuna", "mlflow-skinny"]
        },
    }
