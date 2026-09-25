"""Paired whole-patient uncertainty, including the frozen capacity-ranking rule."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from readmit_iq.decision_support.ranking import capacity_count, ranked_indices


def weighted_selection(order, weights, fraction):
    """Equivalent to duplicating each patient's encounters and selecting the top floor(q*N)."""
    weight = np.asarray(weights, dtype=int)[order]
    k = capacity_count(int(weight.sum()), fraction)
    selected = np.clip(k - np.cumsum(weight) + weight, 0, weight)
    result = np.zeros(len(weights), dtype=int)
    result[order] = selected
    return result


def bootstrap_metrics(frame, predictions, fraction, replicates=1000, seed=42):
    """No model fitting; paired resamples retain every encounter of a sampled patient."""
    codes, patients = pd.factorize(frame.patient_nbr, sort=True)
    y = frame.y.to_numpy()
    orders = {name: ranked_indices(p, frame.encounter_id, seed) for name, p in predictions.items()}
    rng, rows = np.random.default_rng(seed), []
    for replicate in range(replicates):
        counts = np.bincount(
            rng.integers(len(patients), size=len(patients)), minlength=len(patients)
        )
        weights = counts[codes]
        total, positives = int(weights.sum()), int(np.dot(weights, y))
        if positives in {0, total}:
            continue
        for name, values in predictions.items():
            p = np.asarray(values)
            selected = weighted_selection(orders[name], weights, fraction)
            k, tp = int(selected.sum()), int(np.dot(selected, y))
            precision = tp / k if k else np.nan
            rows.append(
                dict(
                    replicate=replicate,
                    model=name,
                    average_precision=average_precision_score(y, p, sample_weight=weights),
                    roc_auc=roc_auc_score(y, p, sample_weight=weights),
                    brier=np.average((p - y) ** 2, weights=weights),
                    recall=tp / positives,
                    precision=precision,
                    lift=precision / (positives / total),
                    fraction_targeted=k / total,
                )
            )
    samples = pd.DataFrame(rows)
    intervals = []
    for name, group in samples.groupby("model"):
        for metric in ["average_precision", "roc_auc", "brier", "recall", "precision", "lift"]:
            intervals.append(
                dict(
                    model=name,
                    metric=metric,
                    ci_low=group[metric].quantile(0.025),
                    ci_high=group[metric].quantile(0.975),
                    valid_replicates=group[metric].notna().sum(),
                )
            )
    return pd.DataFrame(intervals), samples


def clustered_ratio(numerator, denominator, patients, bounds=(0, 1)):
    """Cluster-sandwich ratio interval; patients spanning encounters share an influence term."""
    numerator, denominator = (
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    den = denominator.sum()
    if den == 0:
        return np.nan, np.nan, np.nan
    estimate = numerator.sum() / den
    influence = pd.DataFrame(
        {"patient": np.asarray(patients), "residual": numerator - estimate * denominator}
    )
    sums = influence.groupby("patient").residual.sum().to_numpy()
    g = len(sums)
    if g < 2:
        return estimate, np.nan, np.nan
    se = np.sqrt(g / (g - 1) * np.dot(sums, sums)) / den
    return (
        estimate,
        max(bounds[0], estimate - 1.95996398454 * se),
        min(bounds[1], estimate + 1.95996398454 * se),
    )
