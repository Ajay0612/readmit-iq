"""Per-variable semantics, coverage and discharge-time availability review."""

import pandas as pd

from readmit_iq.data.cohort import TARGET

DEMOGRAPHICS = {"age", "race", "gender"}
UTILIZATION = {"number_inpatient", "number_emergency", "number_outpatient"}
DIAGNOSES = {"diag_1", "diag_2", "diag_3", "number_diagnoses"}
LABS = {"A1Cresult", "max_glu_serum"}
ADMIN = {"payer_code", "medical_specialty"}
ENCOUNTER = {
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
}


def unknown_codes(mappings: dict, column: str) -> list[int]:
    labels = {"NULL", "Not Available", "Not Mapped", "Unknown/Invalid"}
    return [code for code, label in mappings.get(column, {}).items() if label in labels]


def missing_mask(frame: pd.DataFrame, column: str, mappings: dict) -> pd.Series:
    """Unknown values for analysis; not-measured lab categories are explicitly excluded."""
    values = frame[column]
    mask = values.isna() | values.eq("?").fillna(False)
    if column in mappings:
        mask |= values.isin(unknown_codes(mappings, column))
    if column == "gender":
        mask |= values.eq("Unknown/Invalid")
    return mask


def feature_policy(name: str) -> dict:
    """Semantic availability is not proof of timestamp availability in a live EHR."""
    if name in {"encounter_id", "patient_nbr"}:
        return dict(
            group="identifiers",
            leakage="excluded: identity leakage",
            available="Yes, but never a predictor",
            treatment="Keep for audit/grouping only; no ID-order chronology.",
        )
    if name in {"readmitted", TARGET}:
        return dict(
            group="outcome",
            leakage="excluded: direct target leakage",
            available="No: requires post-discharge follow-up",
            treatment="Outcome only; exclude from every predictor matrix.",
        )
    if name in DEMOGRAPHICS:
        return dict(
            group="demographics",
            leakage="definitely safe by source semantics",
            available="Expected before discharge; source has no timestamps",
            treatment="Keep source categories/Unknown; subgroup audits; no causal inference.",
        )
    if name in UTILIZATION:
        return dict(
            group="utilization history",
            leakage="definitely safe by source semantics",
            available="Defined as the year before this encounter",
            treatment="Keep separate counts; evaluate nonlinear effects; no clipping yet.",
        )
    if name in DIAGNOSES:
        return dict(
            group="diagnoses",
            leakage="questionable: retrospective coding",
            available="Clinical diagnoses may be known; finalized billing codes may be late",
            treatment=(
                "Keep numeric diagnosis count; verify coding cutoff, then test nonlinear effects."
                if name == "number_diagnoses"
                else "Preserve strings/Unknown; verify cutoff, then test broad grouping."
            ),
        )
    if name in LABS:
        return dict(
            group="lab indicators",
            leakage="questionable: result availability",
            available="Only usable if the result exists by discharge; timing unverified",
            treatment="Preserve None as not measured; categorical, no numeric imputation.",
        )
    if name == "weight":
        return dict(
            group="encounter information",
            leakage="questionable: sparse coverage",
            available="If measured during stay; operational timing unverified",
            treatment="Exclude from first modeling pass; retain for sensitivity, no imputation.",
        )
    if name in ADMIN:
        return dict(
            group="administrative variables",
            leakage=(
                "questionable: billing finalization"
                if name == "payer_code"
                else "definitely safe by source semantics"
            ),
            available=(
                "Coverage information may be known; final billing payer unverified"
                if name == "payer_code"
                else "Admitting physician specialty, if recorded"
            ),
            treatment="Explicit Unknown; retain for ablation; monitor care-access proxies.",
        )
    if name in ENCOUNTER:
        safe = name in {"admission_type_id", "admission_source_id", "time_in_hospital"}
        return dict(
            group="encounter information",
            leakage=(
                "definitely safe by source semantics"
                if safe
                else "questionable: discharge snapshot"
            ),
            available=(
                "Admission information or elapsed stay is known at completed discharge"
                if safe
                else "Usable only if this final encounter value exists at scoring"
            ),
            treatment=(
                "Retain for cohort; compare models with/without disposition in Phase 3."
                if name == "discharge_disposition_id"
                else "Keep source categories/counts; verify runtime cutoff, no scaling yet."
            ),
        )
    return dict(
        group="medications",
        leakage="questionable: full-stay reconstruction",
        available="In-stay treatment may be known; source aggregation cutoff unverified",
        treatment="Keep categorical; separate No from missing; no treatment-effect claims.",
    )


def feature_audit(
    raw: pd.DataFrame, cohort: pd.DataFrame, variables: list, mappings: dict, exclusions: dict
) -> pd.DataFrame:
    metadata = {item["name"]: item for item in variables}
    if set(metadata) != set(raw.columns):
        raise ValueError("Feature audit requires metadata for every original variable")
    rows = []
    for name in raw.columns:
        source = metadata[name]
        values = cohort[name]
        policy = feature_policy(name)
        if name in {"examide", "citoglipton"}:
            policy.update(
                leakage="excluded: constant, not leakage",
                treatment="Exclude from modeling; keep original field for source audit.",
            )
        rows.append(
            {
                "variable": name,
                "group": policy["group"],
                "source_role": source["role"],
                "meaning": source.get("description", ""),
                "raw_dtype": str(raw[name].dtype),
                "cohort_dtype": str(values.dtype),
                "semantic_type": "identifier" if source["role"] == "ID" else source["type"],
                "raw_cardinality": int(raw[name].nunique(dropna=False)),
                "cohort_cardinality": int(values.nunique(dropna=False)),
                "raw_question_marks": int(raw[name].eq("?").sum()),
                "raw_unknown_pct": 100 * float(missing_mask(raw, name, mappings).mean()),
                "cohort_unknown_count": int(missing_mask(cohort, name, mappings).sum()),
                "cohort_unknown_pct": 100 * float(missing_mask(cohort, name, mappings).mean()),
                "cohort_not_measured_count": int(values.eq("None").sum()) if name in LABS else 0,
                "cohort_observed_values": ", ".join(sorted(values.dropna().astype(str).unique()))
                if values.nunique() <= 14
                else f"{values.nunique()} distinct nonmissing values; see tables",
                "identifier": source["role"] == "ID",
                "leakage_review": policy["leakage"],
                "availability_at_discharge": policy["available"],
                "preliminary_model_exclusion": name in exclusions,
                "treatment": policy["treatment"],
            }
        )
    return pd.DataFrame(rows)
