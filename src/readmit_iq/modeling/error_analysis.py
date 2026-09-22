"""Validation-only error summaries at a fixed threshold."""

import numpy as np
import pandas as pd

from readmit_iq.modeling.features import normalize_category


def error_groups(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    """Count misses and false alarms within prespecified operational groups."""
    y = frame.readmitted_lt30.to_numpy()
    predicted = probability >= threshold
    groups = {
        "prior_inpatient": frame.number_inpatient.clip(upper=3).astype(str).replace("3", "3+"),
        "prior_emergency": frame.number_emergency.clip(upper=3).astype(str).replace("3", "3+"),
        "discharge_disposition": frame.discharge_disposition_id.astype(str),
        "specialty_coverage": frame.medical_specialty.isna().map(
            {True: "Unknown", False: "Observed"}
        ),
        "payer_coverage": frame.payer_code.isna().map({True: "Unknown", False: "Observed"}),
        "race_audit_only": normalize_category(frame.race, "race"),
    }
    result = []
    for feature, labels in groups.items():
        for level in sorted(labels.unique()):
            mask = labels.eq(level).to_numpy()
            actual, estimated = y[mask], predicted[mask]
            positives, negatives = int(actual.sum()), int((1 - actual).sum())
            tp = int(((actual == 1) & estimated).sum())
            fp = int(((actual == 0) & estimated).sum())
            result.append(
                dict(
                    group=feature,
                    level=str(level),
                    encounters=int(mask.sum()),
                    positives=positives,
                    negatives=negatives,
                    tp=tp,
                    fp=fp,
                    fn=positives - tp,
                    tn=negatives - fp,
                    recall=tp / positives if positives else np.nan,
                    false_positive_rate=fp / negatives if negatives else np.nan,
                    mean_probability=float(probability[mask].mean()),
                    predicted_positives=int(estimated.sum()),
                )
            )
    return pd.DataFrame(result)
