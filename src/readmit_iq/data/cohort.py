"""Fixed, outcome-independent cohort policy and lossless missing-marker normalization."""

from pathlib import Path

import pandas as pd
import yaml

from readmit_iq.data.validate import binary_target

TARGET = "readmitted_lt30"


def load_phase2_config(path: Path) -> dict:
    """Read the separate Phase 2 policy without changing the Phase 1 source contract."""
    config = yaml.safe_load(path.read_text())
    rules = config["cohort"]
    codes = list(rules["included_codes"])
    for rule in rules["exclusions"].values():
        codes.extend(rule["codes"])
    if len(codes) != len(set(codes)):
        raise ValueError("Cohort code policies overlap")
    return config


def assign_cohort(frame: pd.DataFrame, rules: dict) -> pd.Series:
    """Assign one reason to each row; fail closed on undocumented disposition codes."""
    if frame.encounter_id.duplicated().any():
        raise ValueError("Encounter identifiers must be unique before cohort assignment")
    lookup = dict.fromkeys(rules["included_codes"], "eligible")
    for reason, rule in rules["exclusions"].items():
        for code in rule["codes"]:
            if code in lookup:
                raise ValueError("Cohort code policies overlap")
            lookup[code] = reason
    reasons = frame.discharge_disposition_id.map(lookup)
    if reasons.isna().any():
        raise ValueError("Unknown discharge disposition: review eligibility policy first")
    return reasons.rename("cohort_reason")


def normalize_missing_markers(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert only literal '?' to NA on a copy. 'None', 'No', and 'NO' remain meaningful."""
    result = frame.copy(deep=True)
    for column in result.select_dtypes(include=["object", "string", "str"]).columns:
        result[column] = result[column].mask(result[column].eq("?"), pd.NA)
    return result


def build_cohort(frame: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Keep all encounters for eligible rows and retain both source and binary outcomes."""
    target = binary_target(frame.readmitted)  # Validate even the subsequently excluded rows.
    reasons = assign_cohort(frame, rules)
    result = normalize_missing_markers(frame.loc[reasons.eq("eligible")])
    result[TARGET] = target.loc[result.index]
    return result, reasons
