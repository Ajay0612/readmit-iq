"""Reuse sealed Phase 4 models and validation predictions; never retrain or open test here."""

import json

import joblib
import numpy as np
import yaml

from readmit_iq.optimization.contract import load_partition
from readmit_iq.optimization.contract import settings as phase4_settings
from readmit_iq.optimization.evaluate import score_validation_once


def settings():
    root, _ = phase4_settings()
    policy = yaml.safe_load((root / "configs/phase5.yaml").read_text())
    if policy["primary"] != "logistic__uncalibrated" or policy["feature_configuration"] != "raw":
        raise ValueError("Phase 5 must preserve the chosen primary and raw feature configuration")
    if policy["random_seed"] != 42 or policy["phase"] != 5:
        raise ValueError("Unexpected Phase 5 seed or phase")
    return root, policy


def development_data():
    root, policy = settings()
    directory = root / "models/development/phase4"
    for filename in ["training_selection.json", "fit_manifest.json", "validation_scores.json"]:
        if not (directory / filename).exists():
            raise FileNotFoundError(
                "Phase 5 requires completed Phase 4 caches; no automatic fitting"
            )
    scored, scoring, fitted = score_validation_once()
    frame = load_partition(root, "validation", "evaluation")
    for key in ["patient_nbr", "encounter_id"]:
        if not np.array_equal(frame[key].to_numpy(), scored[key].to_numpy()):
            raise ValueError("Validation feature and prediction alignment changed")
    if not np.array_equal(frame.readmitted_lt30.to_numpy(), scored.y.to_numpy()):
        raise ValueError("Validation outcome alignment changed")
    frame = frame.assign(y=scored.y.to_numpy())
    names = [policy["primary"], policy["challenger"]]
    models = {name: joblib.load(directory / f"{name}.joblib") for name in names}
    for name in names:
        frame[name] = scored[name].to_numpy()
        if fitted["models"][name]["configuration"] != "raw":
            raise ValueError("Primary/challenger feature configuration changed")
    training = json.loads((directory / "training_selection.json").read_text())
    return frame, models, fitted, scoring, training
