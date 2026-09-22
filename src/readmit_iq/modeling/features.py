"""Explicit feature allowlists and fixed, encounter-local transformations."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from readmit_iq.modeling.diagnoses import diagnosis_group

UTILIZATION = ["number_inpatient", "number_emergency", "number_outpatient"]
CORE_CATEGORICAL = [
    "age",
    "gender",
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "medical_specialty",
]
DERIVED_UTILIZATION = [
    "total_prior_utilization",
    "any_prior_inpatient",
    "any_prior_emergency",
    "any_prior_outpatient",
]
DIAGNOSES = ["diag_1", "diag_2", "diag_3"]
MEDICATIONS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
SNAPSHOT_NUMERIC = ["num_lab_procedures", "num_procedures", "num_medications"]
SNAPSHOT_CATEGORICAL = [
    "change",
    "diabetesMed",
    "insulin",
    "A1Cresult",
    "max_glu_serum",
    "payer_code",
]
# Historical source IDS_mapping.csv, not current administrative code assumptions.
ADMIN_UNKNOWN = {
    "admission_type_id": {"5", "6", "8"},
    "admission_source_id": {"9", "15", "17", "20", "21"},
}
VARIANTS = {
    "primary",
    "no_utilization",
    "raw_utilization",
    "no_disposition",
    "diagnosis_groups",
    "diagnosis_raw",
    "encounter_summaries",
}


def feature_columns(variant: str) -> tuple[list[str], list[str]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    numeric = ["time_in_hospital"]
    categorical = CORE_CATEGORICAL.copy()
    if variant != "no_utilization":
        numeric += UTILIZATION
        if variant != "raw_utilization":
            numeric += DERIVED_UTILIZATION
    if variant == "no_disposition":
        categorical.remove("discharge_disposition_id")
    if variant.startswith("diagnosis_"):
        numeric += ["number_diagnoses"]
        categorical += (
            [f"{c}_group" for c in DIAGNOSES] if variant == "diagnosis_groups" else DIAGNOSES
        )
    if variant == "encounter_summaries":
        numeric += SNAPSHOT_NUMERIC + [
            "active_diabetes_drug_fields",
            "changed_diabetes_drug_fields",
        ]
        categorical += SNAPSHOT_CATEGORICAL
    return numeric, categorical


def normalize_category(values: pd.Series, name: str) -> pd.Series:
    normalized = values.astype("string").fillna("Unknown").replace({"?": "Unknown", "": "Unknown"})
    if name in ADMIN_UNKNOWN:
        normalized = normalized.replace(dict.fromkeys(ADMIN_UNKNOWN[name], "Unknown"))
    if name == "gender":
        normalized = normalized.replace({"Unknown/Invalid": "Unknown"})
    if name in {"A1Cresult", "max_glu_serum"}:
        normalized = normalized.replace({"None": "Not measured"})
    return normalized.astype(object)


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Select only permitted columns; extra IDs, targets and future aggregates are ignored."""

    def __init__(self, variant: str = "primary"):
        self.variant = variant

    def fit(self, X: pd.DataFrame, y=None):
        output = self.transform(X)
        self.feature_names_out_ = np.asarray(output.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric, categorical = feature_columns(self.variant)
        output = pd.DataFrame(index=X.index)
        for name in numeric:
            if name in DERIVED_UTILIZATION or name.endswith("_diabetes_drug_fields"):
                continue
            output[name] = pd.to_numeric(X[name], errors="raise").astype(float)
        if "total_prior_utilization" in numeric:
            output["total_prior_utilization"] = X[UTILIZATION].sum(axis=1, skipna=False)
            for name in UTILIZATION:
                output["any_prior_" + name.removeprefix("number_")] = X[name].gt(0).astype(float)
        if self.variant == "encounter_summaries":
            values = X[MEDICATIONS]
            if not values.isin(["No", "Steady", "Up", "Down"]).all().all():
                raise ValueError("Medication counts require complete, documented status categories")
            output["active_diabetes_drug_fields"] = values.isin(["Steady", "Up", "Down"]).sum(
                axis=1
            )
            output["changed_diabetes_drug_fields"] = values.isin(["Up", "Down"]).sum(axis=1)
        for name in categorical:
            if name.endswith("_group"):
                output[name] = X[name.removesuffix("_group")].map(diagnosis_group)
            else:
                output[name] = normalize_category(X[name], name)
        output = output[numeric + categorical]
        numbers = output[numeric].to_numpy(dtype=float)
        if not np.isfinite(numbers).all() or (numbers < 0).any():
            raise ValueError(
                "Numeric source contract violated: missing/nonfinite/negative counts; "
                "no silent imputation"
            )
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_
