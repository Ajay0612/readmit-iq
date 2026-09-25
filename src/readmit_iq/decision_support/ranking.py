"""Outcome-independent ranking and capacity-limited discharge targeting."""

import hashlib
import math

import numpy as np
import pandas as pd


def ranked_indices(probability, encounter_ids, seed=42) -> np.ndarray:
    p, ids = np.asarray(probability, dtype=float), pd.Series(encounter_ids, dtype="string")
    if p.ndim != 1 or len(p) != len(ids) or not len(p):
        raise ValueError("Require aligned nonempty probabilities and encounter keys")
    if ids.isna().any() or not ids.is_unique:
        raise ValueError("Ranking requires unique encounter keys")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    tie = ids.map(lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    return np.lexsort((tie.to_numpy(), -p))


def capacity_count(n: int, fraction: float) -> int:
    if n < 1 or not 0 < fraction <= 1:
        raise ValueError("Capacity requires nonempty data and a fraction in (0, 1]")
    return math.floor(n * fraction + 1e-12)


def target_mask(probability, encounter_ids, fraction, seed=42) -> np.ndarray:
    order = ranked_indices(probability, encounter_ids, seed)
    mask = np.zeros(len(order), dtype=bool)
    mask[order[: capacity_count(len(order), fraction)]] = True
    return mask


def policy_metrics(y, probability, patients, selected) -> dict:
    y, p, selected = np.asarray(y), np.asarray(probability), np.asarray(selected, dtype=bool)
    if set(np.unique(y)) != {0, 1} or y.shape != p.shape or y.shape != selected.shape:
        raise ValueError("Policy metrics require aligned outcomes, scores and selections")
    patients = pd.Series(patients).reset_index(drop=True)
    if len(patients) != len(y) or patients.isna().any():
        raise ValueError("Patient audit keys must align")
    tp, fp = int(y[selected].sum()), int((1 - y[selected]).sum())
    positives, negatives, k = int(y.sum()), int((1 - y).sum()), int(selected.sum())
    precision = tp / k if k else 0.0
    return dict(
        encounters=len(y),
        patients=patients.nunique(),
        positives=positives,
        prevalence=float(y.mean()),
        mean_probability=float(p.mean()),
        targeted_encounters=k,
        targeted_patients=patients[selected].nunique(),
        fraction_targeted=k / len(y),
        recall=tp / positives,
        precision=precision,
        specificity=(negatives - fp) / negatives,
        tp=tp,
        fp=fp,
        fn=positives - tp,
        tn=negatives - fp,
        lift=precision / y.mean(),
        number_needed_to_contact=k / tp if tp else np.nan,
        random_expected_captured=k * y.mean(),
        additional_captured=tp - k * y.mean(),
    )


def capacity_table(frame, probability, fractions, seed=42) -> pd.DataFrame:
    rows = []
    p = np.asarray(probability)
    for fraction in fractions:
        selected = target_mask(p, frame.encounter_id, fraction, seed)
        rows.append(
            dict(
                capacity=fraction,
                cutoff=float(p[selected].min()) if selected.any() else np.nan,
                **policy_metrics(frame.y, p, frame.patient_nbr, selected),
            )
        )
    return pd.DataFrame(rows)


def risk_groups(frame, probability, groups=10, seed=42) -> pd.DataFrame:
    p, y = np.asarray(probability), frame.y.to_numpy()
    order = ranked_indices(p, frame.encounter_id, seed)
    rows, captured, targeted = [], 0, 0
    boundaries = np.floor(np.arange(groups + 1) * len(order) / groups + 1e-12).astype(int)
    for number, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), 1):
        indices = order[left:right]
        if not len(indices):
            continue
        events = int(y[indices].sum())
        captured += events
        targeted += len(indices)
        rows.append(
            dict(
                risk_group=number,
                encounters=len(indices),
                patients=frame.patient_nbr.iloc[indices].nunique(),
                positives=events,
                observed_rate=float(y[indices].mean()),
                mean_probability=float(p[indices].mean()),
                cumulative_targeted=targeted,
                population_fraction=targeted / len(frame),
                cumulative_captured=captured,
                cumulative_recall=captured / y.sum(),
                group_lift=y[indices].mean() / y.mean(),
                cumulative_lift=(captured / targeted) / y.mean(),
            )
        )
    return pd.DataFrame(rows)


def gains_curve(frame, probability, seed=42) -> pd.DataFrame:
    order = ranked_indices(probability, frame.encounter_id, seed)
    y = frame.y.to_numpy()[order]
    # Export aggregate 1%-capacity points, not encounter-level ordered outcomes.
    indices = np.unique(np.floor(np.linspace(0, len(y), 101) + 1e-12).astype(int))
    return pd.DataFrame(
        {
            "population_fraction": indices / len(y),
            "readmissions_captured_fraction": np.r_[0, np.cumsum(y) / y.sum()][indices],
        }
    )


def threshold_table(frame, probability, thresholds) -> pd.DataFrame:
    p = np.asarray(probability)
    return pd.DataFrame(
        [
            dict(
                threshold=threshold, **policy_metrics(frame.y, p, frame.patient_nbr, p >= threshold)
            )
            for threshold in thresholds
        ]
    )


def business_scenario(policy_row: dict, discharges: int) -> dict:
    """Scale encounter-level observed yields, never prevented events or unique people."""
    targeted = discharges * policy_row["fraction_targeted"]
    captured = discharges * policy_row["prevalence"] * policy_row["recall"]
    random = targeted * policy_row["prevalence"]
    return dict(
        scenario="Illustrative operational scenario; capacity is a portfolio assumption",
        eligible_discharges=discharges,
        expected_observed_readmissions=discharges * policy_row["prevalence"],
        outreach_encounters=targeted,
        model_surfaced_readmissions=captured,
        random_surfaced_readmissions=random,
        additional_surfaced_readmissions=captured - random,
        number_needed_to_contact=policy_row["number_needed_to_contact"],
        prevention_effect_estimated=False,
        unique_people_scaled=False,
    )
