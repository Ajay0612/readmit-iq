"""Source-contract validation and review flags; no cleaning or cohort filtering."""

import pandas as pd

from readmit_iq.config import ProjectConfig
from readmit_iq.data.load_data import ID_COLUMNS

TARGET_LABELS = {"<30", ">30", "NO"}


def binary_target(labels: pd.Series) -> pd.Series:
    """Encode only the documented labels; never silently turn unknown labels into zero."""
    if labels.isna().any() or not set(labels.unique()).issubset(TARGET_LABELS):
        raise ValueError("readmitted contains missing or undocumented labels")
    return labels.eq("<30").astype("int8").rename("readmitted_lt30")


def validate_raw(
    frame: pd.DataFrame, config: ProjectConfig, mappings: dict[str, dict[int, str]]
) -> dict:
    """Return reproducible contract checks plus anomalies requiring later decisions."""
    errors = []
    if list(frame.columns) != config.dataset.expected_columns:
        return {"passed": False, "errors": ["Column names/order differ from source contract"]}
    if len(frame) != config.dataset.expected_rows:
        errors.append(f"Expected {config.dataset.expected_rows} rows, received {len(frame)}")
    for column in ID_COLUMNS:
        if not frame[column].astype("string").str.fullmatch(r"[0-9]+").fillna(False).all():
            errors.append(f"Missing or non-digit identifier in {column}")
    duplicate_ids = int(frame.encounter_id.duplicated().sum())
    duplicates = int(frame.duplicated().sum())
    if duplicate_ids:
        errors.append(f"{duplicate_ids} duplicate encounter IDs")
    if duplicates:
        errors.append(f"{duplicates} exact duplicate rows")
    try:
        binary_target(frame.readmitted)
    except ValueError as exc:
        errors.append(str(exc))
    invalid_numeric = {}
    for column, (lower, upper) in config.numeric_bounds.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | values.lt(lower) | values.mod(1).ne(0)
        if upper is not None:
            invalid |= values.gt(upper)
        invalid_numeric[column] = int(invalid.sum())
        if invalid.any():
            errors.append(f"{column}: {invalid.sum()} values violate source/count bounds")
    ages = {f"[{start}-{start + 10})" for start in range(0, 100, 10)}
    invalid_age = int((~frame.age.isin(ages)).sum())
    if invalid_age:
        errors.append(f"{invalid_age} undocumented age bands")
    unmapped = {}
    coded_unknowns = {}
    for column, table in mappings.items():
        unmapped[column] = sorted(set(frame[column].unique()) - set(table))
        if unmapped[column]:
            errors.append(f"Unmapped {column}: {unmapped[column]}")
        unknown_codes = {
            code: label
            for code, label in table.items()
            if label in {"NULL", "Not Available", "Not Mapped", "Unknown/Invalid"}
        }
        coded_unknowns[column] = {
            str(code): {"description": label, "count": int(frame[column].eq(code).sum())}
            for code, label in unknown_codes.items()
        }
    terminal_codes = {
        code: label
        for code, label in mappings["discharge_disposition_id"].items()
        if "expired" in label.lower() or "hospice" in label.lower()
    }
    return {
        "passed": not errors,
        "errors": errors,
        "duplicate_rows": duplicates,
        "duplicate_encounter_ids": duplicate_ids,
        "invalid_numeric_counts": invalid_numeric,
        "invalid_age_bands": invalid_age,
        "unmapped_codes": unmapped,
        "coded_unknowns": coded_unknowns,
        "unknown_invalid_gender": int(frame.gender.eq("Unknown/Invalid").sum()),
        "death_or_hospice_dispositions": {
            str(code): {
                "description": label,
                "count": int(frame.discharge_disposition_id.eq(code).sum()),
            }
            for code, label in terminal_codes.items()
        },
        "scope": "Structural/source bounds only; no clinical plausibility certification",
    }
