"""One deterministic, validated set of patient-grouped training folds for all comparisons."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(
    frame: pd.DataFrame, folds: int = 5, seed: int = 42, tolerance: float = 0.01
) -> tuple[list, pd.DataFrame, str]:
    """Return positional folds, aggregate diagnostics and stable encounter-to-fold digest."""
    if frame.encounter_id.isna().any() or not frame.encounter_id.is_unique:
        raise ValueError("Complete unique encounter keys are required")
    if frame.patient_nbr.isna().any() or set(frame.readmitted_lt30.unique()) != {0, 1}:
        raise ValueError("Complete patients and both binary target classes are required")
    # Canonical order makes fold membership invariant to input row order.
    order = np.argsort(frame.encounter_id.to_numpy(dtype=str), kind="stable")
    ordered = frame.iloc[order]
    y = ordered.readmitted_lt30
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [(order[a], order[b]) for a, b in splitter.split(ordered, y, ordered.patient_nbr)]
    assigned = np.full(len(frame), -1, dtype=int)
    rows = []
    for fold, (a, b) in enumerate(splits):
        train, holdout = frame.iloc[a], frame.iloc[b]
        if set(train.patient_nbr) & set(holdout.patient_nbr):
            raise ValueError("Patient leakage across training CV folds")
        if (assigned[b] != -1).any() or len(a) + len(b) != len(frame):
            raise ValueError("CV holdouts must partition the training encounters exactly once")
        assigned[b] = fold
        for label, data in [("fit", train), ("holdout", holdout)]:
            if set(data.readmitted_lt30.unique()) != {0, 1}:
                raise ValueError("Every CV partition must contain both classes")
            if abs(data.readmitted_lt30.mean() - y.mean()) > tolerance:
                raise ValueError("CV target prevalence tolerance exceeded")
            rows.append(
                dict(
                    fold=fold,
                    role=label,
                    encounters=len(data),
                    patients=data.patient_nbr.nunique(),
                    positives=int(data.readmitted_lt30.sum()),
                    prevalence=float(data.readmitted_lt30.mean()),
                    patient_overlap=0,
                )
            )
    if (assigned < 0).any():
        raise ValueError("Incomplete training CV coverage")
    membership = (
        frame[["encounter_id", "patient_nbr"]].assign(fold=assigned).sort_values("encounter_id")
    )
    if not membership.groupby("patient_nbr").fold.nunique().eq(1).all():
        raise ValueError("A patient's encounters span CV holdouts")
    fingerprint = hashlib.sha256(
        membership.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    return splits, pd.DataFrame(rows), fingerprint
