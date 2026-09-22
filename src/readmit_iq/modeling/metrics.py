"""Validation-only discrimination, probability quality and patient-cluster uncertainty."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    """Average precision is non-interpolated AP, not trapezoidal PR area."""
    y, probability = np.asarray(y), np.asarray(probability)
    if set(np.unique(y)) != {0, 1} or y.shape != probability.shape:
        raise ValueError("Metrics require aligned binary outcomes with both classes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predictions must be finite probabilities")
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(
        average_precision=float(average_precision_score(y, probability)),
        roc_auc=float(roc_auc_score(y, probability)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        precision=float(precision_score(y, predicted, zero_division=0)),
        f1=float(f1_score(y, predicted, zero_division=0)),
        specificity=float(tn / (tn + fp)),
        brier=float(brier_score_loss(y, probability)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        threshold=threshold,
        mean_probability=float(probability.mean()),
        prevalence=float(y.mean()),
        predicted_positive_fraction=float(predicted.mean()),
    )


def calibration_bins(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Descriptive quantile reliability bins; no probability calibration is fitted."""
    frame = pd.DataFrame({"outcome": y, "probability": probability})
    frame["bin"] = pd.qcut(frame.probability, q=bins, duplicates="drop")
    return (
        frame.groupby("bin", observed=True)
        .agg(
            encounters=("outcome", "size"),
            mean_probability=("probability", "mean"),
            observed_rate=("outcome", "mean"),
            positives=("outcome", "sum"),
        )
        .reset_index()
        .assign(bin=lambda f: f.bin.astype(str))
    )


def bootstrap_ap(
    y: np.ndarray,
    patients: pd.Series,
    predictions: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired patient bootstrap: all encounters from each resampled patient share a weight."""
    patient_index, unique = pd.factorize(patients, sort=True)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weight = multiplicity[patient_index]
        samples.append(
            {
                name: average_precision_score(y, p, sample_weight=weight)
                for name, p in predictions.items()
            }
        )
    return pd.DataFrame(samples)
