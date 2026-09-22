"""Business-question summaries and discovery-only diagnosis/feature checks."""

import pandas as pd

from readmit_iq.analysis.audit import missing_mask
from readmit_iq.analysis.statistics import encounter_split_overlap, grouped_rates, rate_contrast
from readmit_iq.data.cohort import TARGET


def analysis_groups(frame: pd.DataFrame, policy: dict) -> dict[str, pd.Series]:
    """Fixed display bins for EDA only, never a fitted preprocessing transformation."""
    groups = {}
    for name in ["number_inpatient", "number_emergency", "number_outpatient"]:
        groups[name] = frame[name].clip(upper=3).astype(str).replace("3", "3+")
    for name, bins in policy["eda"]["numeric_bins"].items():
        group = pd.cut(frame[name], bins["edges"], labels=bins["labels"])
        if group.isna().any():
            raise ValueError(f"EDA bins do not cover all {name} observations")
        groups[name] = group.astype("string")
    for name in [
        "age",
        "gender",
        "race",
        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id",
        "change",
        "diabetesMed",
        "insulin",
        "weight",
        "medical_specialty",
        "payer_code",
        "A1Cresult",
        "max_glu_serum",
    ]:
        groups[name] = frame[name].astype("string").fillna("Unknown")
    return groups


def make_rates(frame: pd.DataFrame, groups: dict, policy: dict) -> pd.DataFrame:
    tables = []
    settings = policy["eda"]
    for feature, labels in groups.items():
        table = grouped_rates(
            frame,
            labels,
            min_patients=settings["ci_min_patients"],
            min_events=settings["ci_min_events"],
            confidence=settings["confidence_level"],
        )
        table.insert(0, "feature", feature)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def exact_utilization_rates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Check observed shapes beyond coarse plot bins, without fitting constraints."""
    tables = []
    shapes = {}
    for column in ["number_inpatient", "number_emergency", "number_outpatient"]:
        table = grouped_rates(frame, frame[column].astype("string"))
        table["count"] = table.level.astype(int)
        table = table.sort_values("count")
        table.insert(0, "feature", column)
        supported = table.loc[table.encounters.ge(100)]
        shapes[column] = {
            "support_rule": "Exact-count categories with at least 100 encounters",
            "monotonic_increasing_observed_rate": bool(supported.rate.is_monotonic_increasing),
            "maximum_supported_count": int(supported["count"].max()),
            "interpretation": "Descriptive shape only; no fitted monotonic constraint",
        }
        tables.append(table)
    return pd.concat(tables, ignore_index=True), shapes


def missingness_tables(frame: pd.DataFrame, audit: pd.DataFrame, mappings: dict) -> tuple:
    ranked = audit.loc[
        audit.cohort_unknown_count.gt(0),
        [
            "variable",
            "raw_question_marks",
            "raw_unknown_pct",
            "cohort_unknown_count",
            "cohort_unknown_pct",
            "treatment",
        ],
    ].sort_values("cohort_unknown_pct", ascending=False)
    tables = []
    contrasts = {}
    for column in ranked.variable:
        mask = missing_mask(frame, column, mappings)
        groups = mask.map({True: "Unknown", False: "Observed"})
        table = grouped_rates(frame, groups)
        table.insert(0, "feature", column)
        tables.append(table)
        # Contrast intervals are used only for well-supported missingness groups.
        if not table["sparse"].any():
            contrasts[column] = rate_contrast(frame, groups, "Unknown", "Observed")
    return ranked, pd.concat(tables, ignore_index=True), contrasts


def repeated_patients(frame: pd.DataFrame, fractions: list[float]) -> tuple:
    counts = frame.patient_nbr.value_counts()
    bins = counts.clip(upper=4).astype(str).replace("4", "4+")
    categories = (
        pd.DataFrame({"encounters": counts, "bucket": bins})
        .groupby("bucket")
        .agg(patients=("encounters", "size"), encounters=("encounters", "sum"))
        .reset_index()
    )
    status = frame.patient_nbr.map(counts).gt(1).map({True: "Repeated", False: "Single"})
    rates = grouped_rates(frame, status)
    summary = {
        "patients": len(counts),
        "repeat_patients": int(counts.gt(1).sum()),
        "repeat_encounters": int(counts[counts.gt(1)].sum()),
        "repeat_encounter_fraction": float(counts[counts.gt(1)].sum() / len(frame)),
        "maximum_encounters": int(counts.max()),
        "contrast": rate_contrast(frame, status, "Repeated", "Single"),
        "hypothetical_encounter_split": encounter_split_overlap(counts, fractions),
    }
    return categories, rates, summary


def diagnosis_inventory(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect code syntax and prefixes; do not introduce a clinical grouping transformer."""
    inventory = []
    prefixes = []
    for name in ["diag_1", "diag_2", "diag_3"]:
        values = frame[name].astype("string")
        category = pd.Series("Other format", index=frame.index)
        for expression, label in [
            (r"\d{1,2}(?:\.\d+)?", "Numeric with short integer component"),
            (r"\d{3}(?:\.\d+)?", "Numeric with three-digit component"),
            (r"V\d{2}(?:\.\d+)?", "V supplementary code"),
            (r"E\d{3}(?:\.\d+)?", "E external-cause code"),
        ]:
            category.loc[values.str.fullmatch(expression).fillna(False)] = label
        category.loc[values.isna()] = "Unknown"
        counts = category.value_counts()
        prefix = values.str.split(".", regex=False).str[0].fillna("Unknown")
        rate_table = grouped_rates(frame, prefix).sort_values("encounters", ascending=False)
        rate_table.insert(0, "feature", name)
        prefixes.append(rate_table)
        observed = values.dropna().value_counts()
        for label, count in counts.items():
            inventory.append(
                {
                    "feature": name,
                    "format": label,
                    "encounters": int(count),
                    "raw_observed_cardinality": int(values.nunique()),
                    "integer_prefix_cardinality": int(prefix[prefix.ne("Unknown")].nunique()),
                    "codes_with_under_20_encounters": int(observed.lt(20).sum()),
                    "encounters_in_codes_under_20": int(observed[observed.lt(20)].sum()),
                }
            )
    return pd.DataFrame(inventory), pd.concat(prefixes, ignore_index=True)


def encounter_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
    ]
    output = []
    for column in columns:
        for target, values in frame.groupby(TARGET)[column]:
            output.append(
                {
                    "feature": column,
                    "target": target,
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "p25": values.quantile(0.25),
                    "p75": values.quantile(0.75),
                    "p95": values.quantile(0.95),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(output)
