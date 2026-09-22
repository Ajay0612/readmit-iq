"""Patient-cluster uncertainty and hypothetical overlap without creating data splits."""

from statistics import NormalDist

import numpy as np
import pandas as pd

from readmit_iq.data.cohort import TARGET


def grouped_rates(
    frame: pd.DataFrame,
    groups: pd.Series,
    *,
    min_patients: int = 30,
    min_events: int = 10,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Encounter rates with cluster-robust normal intervals, clustering by patient.

    For group rate p, SE² = G/(G-1) * sum_g (y_g - p*n_g)² / N².
    Sparse groups retain their observed rates but receive no normal-approximation interval.
    Intervals describe sampling uncertainty, not confounding or out-of-system outcome loss.
    """
    if not frame.index.equals(groups.index):
        raise ValueError("Group labels must align exactly with encounter rows")
    if not frame[TARGET].isin([0, 1]).all() or frame.patient_nbr.isna().any():
        raise ValueError("Rates require binary outcomes and complete patient keys")
    data = pd.DataFrame(
        {
            "patient": frame.patient_nbr,
            "level": groups.astype("string").fillna("Unknown"),
            "y": frame[TARGET],
        }
    )
    clusters = data.groupby(["level", "patient"]).y.agg(n="size", y="sum")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    rows = []
    for level, sample in clusters.groupby(level="level", sort=False):
        n, positives = int(sample.n.sum()), int(sample.y.sum())
        patients = len(sample)
        rate = positives / n
        supported = patients >= min_patients and min(positives, n - positives) >= min_events
        se = np.nan
        if supported and patients > 1:
            residual = sample.y - rate * sample.n
            se = np.sqrt(patients / (patients - 1) * residual.pow(2).sum()) / n
        rows.append(
            {
                "level": str(level),
                "encounters": n,
                "patients": patients,
                "positives": positives,
                "negatives": n - positives,
                "rate": rate,
                "cluster_se": se,
                "ci_low": max(0, rate - z * se) if np.isfinite(se) else np.nan,
                "ci_high": min(1, rate + z * se) if np.isfinite(se) else np.nan,
                "sparse": not supported,
            }
        )
    return pd.DataFrame(rows)


def rate_contrast(frame: pd.DataFrame, groups: pd.Series, high: str, low: str) -> dict:
    """Difference in encounter rates; patient covariance is retained across categories."""
    if not frame.index.equals(groups.index):
        raise ValueError("Group labels must align exactly with encounter rows")
    terms = []
    rates = []
    counts = []
    for value in [high, low]:
        subset = frame.loc[groups.astype("string").eq(value)]
        if subset.empty:
            raise ValueError(f"No encounters in contrast category {value}")
        by_patient = subset.groupby("patient_nbr")[TARGET].agg(["size", "sum"])
        rate = float(subset[TARGET].mean())
        terms.append((by_patient["sum"] - rate * by_patient["size"]) / len(subset))
        rates.append(rate)
        counts.append(len(subset))
    influence = pd.concat(terms, axis=1).fillna(0)
    g = len(influence)
    se = np.sqrt(g / (g - 1) * (influence.iloc[:, 0] - influence.iloc[:, 1]).pow(2).sum())
    difference = rates[0] - rates[1]
    z = NormalDist().inv_cdf(0.975)
    return {
        "high": high,
        "low": low,
        "high_n": counts[0],
        "low_n": counts[1],
        "high_rate": rates[0],
        "low_rate": rates[1],
        "difference_pp": 100 * difference,
        "difference_ci_low_pp": 100 * (difference - z * se),
        "difference_ci_high_pp": 100 * (difference + z * se),
        "rate_ratio": rates[0] / rates[1] if rates[1] else None,
    }


def encounter_split_overlap(patient_counts: pd.Series, fractions: list[float]) -> dict:
    """Exact expectations for hypothetical iid encounter allocation, not actual partitions."""
    weights = np.asarray(fractions, dtype=float)
    if len(weights) != 3 or not np.isclose(weights.sum(), 1) or (weights <= 0).any():
        raise ValueError("Require three positive fractions summing to one")
    counts = np.asarray(patient_counts, dtype=int)
    if (counts < 1).any() or not len(counts):
        raise ValueError("Patient encounter counts must be positive")
    across = 1 - (weights[:, None] ** counts).sum(axis=0)
    test_shared_with_train = np.sum(counts * (1 - (1 - weights[0]) ** (counts - 1)))
    return {
        "expected_patients_in_multiple_partitions": float(across.sum()),
        "test_encounter_probability_patient_also_in_train": float(
            test_shared_with_train / counts.sum()
        ),
        "assumption": "Independent 70/15/15 encounter assignment; no partitions created",
        "limitation": "Estimates identity overlap, not metric inflation",
    }
