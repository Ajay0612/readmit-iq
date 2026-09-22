"""Reproduce the eligible cohort, audit, descriptive tables and seven EDA figures."""

import json
import logging
import os

import pandas as pd

from readmit_iq.analysis.audit import feature_audit
from readmit_iq.analysis.reporting import write_json
from readmit_iq.analysis.reports import audit_report, cohort_report, missingness_report
from readmit_iq.analysis.statistics import rate_contrast
from readmit_iq.analysis.tables import (
    analysis_groups,
    diagnosis_inventory,
    encounter_summaries,
    exact_utilization_rates,
    make_rates,
    missingness_tables,
    repeated_patients,
)
from readmit_iq.config import configure_logging, load_config
from readmit_iq.data.cohort import TARGET, build_cohort, load_phase2_config
from readmit_iq.data.download import sha256, verify_file
from readmit_iq.data.inspect import column_profile, summarize
from readmit_iq.data.load_data import load_id_mapping, load_raw
from readmit_iq.data.validate import binary_target, validate_raw

LOGGER = logging.getLogger(__name__)


def population_summary(frame: pd.DataFrame) -> dict:
    positive = int(frame.readmitted.eq("<30").sum())
    return {
        "encounters": len(frame),
        "patients": int(frame.patient_nbr.nunique()),
        "positives": positive,
        "negatives": len(frame) - positive,
        "prevalence_pct": 100 * positive / len(frame),
    }


def main() -> None:
    configure_logging()
    source = load_config()
    root = source.raw_dir.parent.parent
    if (root / "data/processed/phase3/lock.json").exists():
        raise RuntimeError(
            "Phase 3 test is frozen. Use archived Phase 2 outputs; do not rerun full-cohort EDA."
        )
    policy = load_phase2_config(root / "configs/phase2.yaml")
    output = root / policy["eda"]["report_dir"]
    figures = root / policy["eda"]["figure_dir"]
    interim = root / policy["eda"]["interim_dir"]
    for directory in [output, figures, interim]:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    from readmit_iq.analysis.plots import create_figures

    for name, checksum in source.dataset.file_sha256.items():
        verify_file(source.raw_dir / name, checksum)
    raw = load_raw(source.raw_dir / "diabetic_data.csv")
    mappings = load_id_mapping(source.raw_dir / "IDS_mapping.csv")
    validation = validate_raw(raw, source, mappings)
    if not validation["passed"]:
        raise ValueError(validation["errors"])
    prior = summarize(raw, column_profile(raw), validation)
    if prior != json.loads((source.report_dir / "summary.json").read_text()):
        raise ValueError("Phase 1 summary does not reconcile with actual source")
    cohort, reasons = build_cohort(raw, policy["cohort"])
    membership = raw[["encounter_id", "patient_nbr"]].assign(cohort_reason=reasons)
    membership.to_csv(interim / "cohort_membership.csv", index=False)
    cohort.to_csv(interim / "eligible_encounters.csv", index=False, na_rep="")
    target = binary_target(raw.readmitted)
    flow = [{"stage": "Raw source", "removed_at_stage": 0, **population_summary(raw)}]
    retained = pd.Series(True, index=raw.index)
    excluded_rows = []
    for reason, rule in policy["cohort"]["exclusions"].items():
        excluded = reasons.eq(reason)
        retained &= ~excluded
        flow.append(
            {
                "stage": f"After {reason}",
                "removed_at_stage": int(excluded.sum()),
                **population_summary(raw.loc[retained]),
            }
        )
        excluded_rows.append(
            {
                "reason": reason,
                "codes": ", ".join(map(str, rule["codes"])),
                "encounters": int(excluded.sum()),
                "positives": int(target.loc[excluded].sum()),
                "rationale": rule["reason"],
            }
        )
    dispositions = []
    for code, label in mappings["discharge_disposition_id"].items():
        part = raw.loc[raw.discharge_disposition_id.eq(code)]
        reason = (
            reasons.loc[part.index].iloc[0]
            if len(part)
            else next(
                (
                    key
                    for key, rule in policy["cohort"]["exclusions"].items()
                    if code in rule["codes"]
                ),
                "eligible",
            )
        )
        dispositions.append(
            {
                "code": code,
                "description": label,
                "encounters": len(part),
                "positives": int(part.readmitted.eq("<30").sum()),
                "decision": reason,
            }
        )
    dispositions = pd.DataFrame(dispositions).sort_values("code")
    sensitivity = policy["cohort"]["sensitivity"]
    population_masks = {
        "Raw source": pd.Series(True, index=raw.index),
        "Exclude death/hospice only": ~reasons.isin(["death", "hospice"]),
        "Primary cohort": reasons.eq("eligible"),
        "Primary + unknown destinations": reasons.eq("eligible")
        | raw.discharge_disposition_id.isin(sensitivity["retain_unknown_codes"]),
        "Primary without mixed rehabilitation": reasons.eq("eligible")
        & ~raw.discharge_disposition_id.isin(sensitivity["remove_mixed_rehab_codes"]),
        "Community destinations only": raw.discharge_disposition_id.isin(
            sensitivity["community_only_codes"]
        ),
    }
    sensitivity_table = pd.DataFrame(
        [
            {"cohort": name, **population_summary(raw.loc[mask])}
            for name, mask in population_masks.items()
        ]
    )
    excluded_table = pd.DataFrame(excluded_rows)
    cohort_report(
        source.report_dir / "cohort_report.md",
        pd.DataFrame(flow),
        dispositions,
        sensitivity_table,
        excluded_table,
    )
    variables = json.loads((source.raw_dir / "uci_metadata.json").read_text())["data"]["variables"]
    audit = feature_audit(raw, cohort, variables, mappings, policy["preliminary_model_exclusions"])
    audit.to_csv(source.report_dir / "feature_audit.csv", index=False)
    audit_report(source.report_dir / "feature_audit.md", audit)
    ranked, missing_rates, missing_contrasts = missingness_tables(cohort, audit, mappings)
    missingness_report(
        source.report_dir / "missingness_strategy.md", ranked, missing_rates, missing_contrasts
    )
    groups = analysis_groups(cohort, policy)
    rates = make_rates(cohort, groups, policy)
    exact_rates, utilization_shapes = exact_utilization_rates(cohort)
    LOGGER.info("Computed feature rates with patient-cluster uncertainty")
    fractions = [
        policy["split_design"][f"{part}_fraction"] for part in ["train", "validation", "test"]
    ]
    repeat_bins, repeat_rates, repeat_summary = repeated_patients(cohort, fractions)
    raw_repeat_bins, raw_repeat_rates, raw_repeat_summary = repeated_patients(
        raw.assign(**{TARGET: target}), fractions
    )
    diagnoses, prefixes = diagnosis_inventory(cohort)
    contrasts = {
        "prior_inpatient_3plus_vs_0": rate_contrast(cohort, groups["number_inpatient"], "3+", "0"),
        "prior_emergency_3plus_vs_0": rate_contrast(cohort, groups["number_emergency"], "3+", "0"),
        "outpatient_3plus_vs_1": rate_contrast(cohort, groups["number_outpatient"], "3+", "1"),
        "stay_8to14_vs_1to2": rate_contrast(
            cohort, groups["time_in_hospital"], "8–14 days", "1–2 days"
        ),
        "medications_20to29_vs_1to9": rate_contrast(
            cohort, groups["num_medications"], "20–29", "1–9"
        ),
        "medications_30plus_vs_20to29": rate_contrast(
            cohort, groups["num_medications"], "30+", "20–29"
        ),
        "rehab_vs_home": rate_contrast(cohort, groups["discharge_disposition_id"], "22", "1"),
    }
    for name, table in {
        "cohort_flow": pd.DataFrame(flow),
        "exclusions": excluded_table,
        "dispositions": dispositions,
        "cohort_sensitivity": sensitivity_table,
        "grouped_rates": rates,
        "utilization_exact_rates": exact_rates,
        "ranked_missingness": ranked,
        "missingness_rates": missing_rates,
        "repeat_patient_counts": repeat_bins,
        "repeat_patient_rates": repeat_rates,
        "raw_repeat_patient_counts": raw_repeat_bins,
        "raw_repeat_patient_rates": raw_repeat_rates,
        "diagnosis_inventory": diagnoses,
        "diagnosis_prefixes": prefixes,
        "numeric_by_target": encounter_summaries(cohort),
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    cohort[["number_inpatient", "number_emergency", "number_outpatient"]].corr(
        method="spearman"
    ).to_csv(output / "utilization_spearman.csv", index_label="feature")
    names = create_figures(cohort, rates, ranked, policy, figures)
    summary = {
        "phase1_reconciled": True,
        "raw": population_summary(raw),
        "primary": population_summary(cohort),
        "excluded_encounters": len(raw) - len(cohort),
        "repeat_patients": repeat_summary,
        "raw_repeat_patients": raw_repeat_summary,
        "contrasts": contrasts,
        "utilization_shapes": utilization_shapes,
        "missingness_contrasts": missing_contrasts,
        "figure_files": names,
        "cohort_file_sha256": sha256(interim / "eligible_encounters.csv"),
        "policy_sha256": sha256(root / "configs/phase2.yaml"),
        "models_trained": 0,
        "partitions_created": False,
        "uncertainty": "Encounter rates; normal sandwich intervals with patient-cluster variance",
    }
    write_json(output / "summary.json", summary)
    LOGGER.info(
        "Phase 2 outputs: %s eligible encounters, %.4f%% positive, %s figures",
        len(cohort),
        summary["primary"]["prevalence_pct"],
        len(names),
    )


if __name__ == "__main__":
    main()
