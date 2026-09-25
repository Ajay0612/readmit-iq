"""Predeclared clinical/error and demographic diagnostics with cluster-aware uncertainty."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from readmit_iq.decision_support.uncertainty import clustered_ratio
from readmit_iq.modeling.features import normalize_category


def group_labels(frame):
    return {
        "prior_inpatient": pd.Series(
            np.where(frame.number_inpatient.eq(0), "None", "Any"), index=frame.index
        ),
        "prior_inpatient_high": pd.Series(
            np.where(frame.number_inpatient.ge(3), "3+", "0–2"), index=frame.index
        ),
        "prior_emergency": pd.Series(
            np.where(frame.number_emergency.eq(0), "None", "Any"), index=frame.index
        ),
        "age": normalize_category(frame.age, "age"),
        "gender": normalize_category(frame.gender, "gender"),
        "race": normalize_category(frame.race, "race"),
        "length_of_stay": pd.cut(
            frame.time_in_hospital, [0, 2, 4, 7, 14], labels=["1–2", "3–4", "5–7", "8–14"]
        ).astype(str),
        "discharge_destination": normalize_category(
            frame.discharge_disposition_id, "discharge_disposition_id"
        ),
        "admission_type": normalize_category(frame.admission_type_id, "admission_type_id"),
        "admission_source": normalize_category(frame.admission_source_id, "admission_source_id"),
        "specialty_missing": normalize_category(frame.medical_specialty, "medical_specialty")
        .eq("Unknown")
        .map({True: "Unknown", False: "Recorded"}),
        "payer_missing": normalize_category(frame.payer_code, "payer_code")
        .eq("Unknown")
        .map({True: "Unknown", False: "Recorded"}),
    }


def subgroup_metrics(frame, probability, selected, rules):
    probability, selected = np.asarray(probability), np.asarray(selected, dtype=bool)
    rows = []
    labels = group_labels(frame)
    if set(labels) != set(rules["groups"]):
        raise ValueError("Subgroup definitions differ from the declared policy")
    for feature, groups in labels.items():
        for level in sorted(groups.unique()):
            mask = groups.eq(level).to_numpy()
            f, p, flag = frame.loc[mask], probability[mask], selected[mask]
            y = f.y.to_numpy()
            n, events, patients = len(y), int(y.sum()), f.patient_nbr.nunique()
            suppressed = (
                n < rules["minimum_encounters"]
                or patients < rules["minimum_patients"]
                or events < rules["minimum_positives"]
                or n - events < rules["minimum_negatives"]
            )
            row = dict(
                group=feature,
                level=str(level),
                encounters=n,
                patients=patients,
                positives=events,
                negatives=n - events,
                suppressed=suppressed,
            )
            if not suppressed:
                tp, fp = int(y[flag].sum()), int((1 - y[flag]).sum())
                row.update(
                    average_precision=average_precision_score(y, p),
                    brier=float(((p - y) ** 2).mean()),
                    mean_probability=float(p.mean()),
                    targeted_encounters=int(flag.sum()),
                    tp=tp,
                    fp=fp,
                    fn=events - tp,
                    tn=n - events - fp,
                )
                numerators = {
                    "prevalence": (y, np.ones(n)),
                    "recall": (y * flag, y),
                    "precision": (y * flag, flag),
                    "probability_bias": (p - y, np.ones(n)),
                }
                for metric, (num, den) in numerators.items():
                    estimate, low, high = clustered_ratio(
                        num, den, f.patient_nbr, (-1, 1) if metric == "probability_bias" else (0, 1)
                    )
                    row.update(
                        {metric: estimate, f"{metric}_ci_low": low, f"{metric}_ci_high": high}
                    )
                row["false_negative_rate"] = 1 - row["recall"]
                row["material_probability_bias"] = abs(row["probability_bias"]) >= rules[
                    "material_probability_bias"
                ] and (row["probability_bias_ci_low"] > 0 or row["probability_bias_ci_high"] < 0)
            rows.append(row)
    return pd.DataFrame(rows)


def error_profiles(frame, probability, selected):
    """Counts/profiles of errors; never use these comparisons to refit the model."""
    y, p, flag = frame.y.to_numpy(), np.asarray(probability), np.asarray(selected, dtype=bool)
    low = frame.number_inpatient.eq(0).to_numpy()
    cohorts = {
        "all_readmissions": y == 1,
        "false_negative": (y == 1) & ~flag,
        "false_positive": (y == 0) & flag,
        "low_utilization_false_negative": (y == 1) & ~flag & low,
        "low_utilization_true_positive": (y == 1) & flag & low,
    }
    rows = []
    for cohort, mask in cohorts.items():
        subset = frame.loc[mask]
        n = len(subset)
        row = dict(
            cohort=cohort,
            encounters=n,
            patients=subset.patient_nbr.nunique(),
            mean_probability=float(p[mask].mean()) if n else np.nan,
            median_stay=float(subset.time_in_hospital.median()) if n else np.nan,
            no_prior_inpatient_fraction=float(subset.number_inpatient.eq(0).mean())
            if n
            else np.nan,
            no_prior_emergency_fraction=float(subset.number_emergency.eq(0).mean())
            if n
            else np.nan,
        )
        rows.append(row)
    detail = []
    for name, labels in group_labels(frame).items():
        if name in {"race", "gender"}:
            continue  # demographic diagnostics have their separate controlled table
        for cohort in cohorts:
            mask = cohorts[cohort]
            counts = labels.loc[mask].value_counts()
            for level, count in counts.items():
                detail.append(
                    dict(
                        cohort=cohort,
                        group=name,
                        level=str(level),
                        encounters=int(count),
                        fraction=float(count / mask.sum()),
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(detail)
