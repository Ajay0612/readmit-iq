"""Prevent outcome-driven cohort selection, lost categories and encounter-level uncertainty."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from readmit_iq.analysis.audit import feature_audit, missing_mask
from readmit_iq.analysis.statistics import encounter_split_overlap, grouped_rates, rate_contrast
from readmit_iq.data.cohort import (
    TARGET,
    assign_cohort,
    build_cohort,
    load_phase2_config,
    normalize_missing_markers,
)


@pytest.fixture
def policy():
    return load_phase2_config(Path("configs/phase2.yaml"))


def sample(codes, outcomes=None):
    return pd.DataFrame(
        {
            "encounter_id": [str(n) for n in range(len(codes))],
            "patient_nbr": ["same_patient"] * len(codes),
            "discharge_disposition_id": codes,
            "readmitted": outcomes or ["NO"] * len(codes),
            "A1Cresult": ["None"] * len(codes),
            "race": ["?"] * len(codes),
        }
    )


@pytest.mark.parametrize(
    "code,reason",
    [
        (11, "death"),
        (19, "death"),
        (13, "hospice"),
        (2, "inpatient_transfer"),
        (9, "discharge_not_confirmed"),
        (12, "discharge_not_confirmed"),
        (15, "discharge_not_confirmed"),
        (18, "unknown_destination"),
        (25, "unknown_destination"),
        (30, "unknown_destination"),
    ],
)
def test_disposition_semantics_have_explicit_policy(policy, code, reason):
    assert assign_cohort(sample([code]), policy["cohort"]).iloc[0] == reason


def test_postacute_outreach_and_repeated_patients_are_retained(policy):
    raw = sample([1, 3, 4, 6, 7, 22, 24], ["<30", "NO", ">30", "NO", "<30", "NO", "NO"])
    cohort, reasons = build_cohort(raw, policy["cohort"])
    assert len(cohort) == len(raw)
    assert cohort.patient_nbr.nunique() == 1
    assert cohort[TARGET].tolist() == [1, 0, 0, 0, 1, 0, 0]
    assert reasons.eq("eligible").all()


def test_cohort_does_not_depend_on_outcome_or_mutate_source(policy):
    raw = sample([1, 11, 13, 2, 18, 22])
    before = raw.copy(deep=True)
    cohort, original_reasons = build_cohort(raw, policy["cohort"])
    changed = raw.assign(readmitted="<30")
    pd.testing.assert_series_equal(original_reasons, assign_cohort(changed, policy["cohort"]))
    pd.testing.assert_frame_equal(raw, before)
    assert cohort.race.isna().all()
    assert cohort.A1Cresult.eq("None").all()


def test_bad_label_in_excluded_row_still_fails(policy):
    with pytest.raises(ValueError, match="undocumented"):
        build_cohort(sample([1, 11], ["NO", "unknown"]), policy["cohort"])


def test_unknown_disposition_fails_closed(policy):
    with pytest.raises(ValueError, match="Unknown discharge"):
        assign_cohort(sample([999]), policy["cohort"])


def test_overlapping_cohort_rules_rejected(policy):
    bad = deepcopy(policy["cohort"])
    bad["exclusions"]["death"]["codes"].append(1)
    with pytest.raises(ValueError, match="overlap"):
        assign_cohort(sample([1]), bad)


def test_all_historical_disposition_codes_are_classified_once(policy):
    assignments = assign_cohort(sample(list(range(1, 31))), policy["cohort"])
    assert assignments.notna().all()


def test_normalization_keeps_not_measured_and_nonprescription_categories():
    raw = pd.DataFrame({"value": ["?", "None", "No", "NO", "Norm", "0"]})
    normalized = normalize_missing_markers(raw)
    assert pd.isna(normalized.iloc[0, 0])
    assert normalized.iloc[1:, 0].tolist() == raw.iloc[1:, 0].tolist()
    assert raw.iloc[0, 0] == "?"


def test_unknown_audit_does_not_count_unmeasured_lab_as_missing():
    frame = pd.DataFrame({"A1Cresult": ["None", "?", pd.NA, "Norm"]})
    assert missing_mask(frame, "A1Cresult", {}).tolist() == [False, True, True, False]


def test_grouped_rate_interval_uses_patient_clusters():
    frame = pd.DataFrame({"patient_nbr": ["a"] * 3 + ["b"] * 3, TARGET: [1, 1, 0, 0, 0, 0]})
    output = grouped_rates(frame, pd.Series(["All"] * 6), min_patients=2, min_events=0)
    row = output.iloc[0]
    assert row.encounters == 6 and row.patients == 2
    assert row.rate == pytest.approx(1 / 3)
    assert row.cluster_se == pytest.approx(1 / 3)


def test_sparse_groups_do_not_receive_false_precision():
    frame = pd.DataFrame({"patient_nbr": ["a", "b"], TARGET: [1, 0]})
    result = grouped_rates(frame, pd.Series(["All", "All"]))
    assert result.iloc[0]["sparse"] and np.isnan(result.iloc[0].ci_low)


def test_group_alignment_is_not_silently_changed():
    frame = pd.DataFrame({"patient_nbr": ["a", "b"], TARGET: [1, 0]})
    with pytest.raises(ValueError, match="align"):
        grouped_rates(frame, pd.Series(["a", "b"], index=[1, 0]))


def test_rate_contrast_accounts_for_shared_patients():
    frame = pd.DataFrame({"patient_nbr": ["a", "a", "b", "b"], TARGET: [1, 0, 1, 0]})
    comparison = rate_contrast(frame, pd.Series(["High", "Low", "High", "Low"]), "High", "Low")
    assert comparison["difference_pp"] == 100
    assert comparison["difference_ci_low_pp"] == 100
    assert comparison["rate_ratio"] is None  # A zero baseline must not become a finite ratio.


def test_hypothetical_overlap_without_creating_partitions():
    single = encounter_split_overlap(pd.Series([1, 1]), [0.7, 0.15, 0.15])
    assert single["expected_patients_in_multiple_partitions"] == pytest.approx(0)
    assert single["test_encounter_probability_patient_also_in_train"] == 0
    repeated = encounter_split_overlap(pd.Series([2]), [0.7, 0.15, 0.15])
    assert repeated["expected_patients_in_multiple_partitions"] == pytest.approx(0.465)
    assert repeated["test_encounter_probability_patient_also_in_train"] == pytest.approx(0.7)


def test_feature_audit_covers_all_variables_and_target_leakage(policy):
    raw = sample([1, 3])
    cohort, _ = build_cohort(raw, policy["cohort"])
    metadata = [
        {
            "name": name,
            "role": "ID" if name.endswith(("id", "nbr")) else "Feature",
            "type": "Categorical",
            "description": "Test field",
        }
        for name in raw.columns
    ]
    audit = feature_audit(raw, cohort, metadata, {}, policy["preliminary_model_exclusions"])
    assert set(audit.variable) == set(raw.columns)
    by_name = audit.set_index("variable")
    assert by_name.loc["A1Cresult", "cohort_not_measured_count"] == 2
    assert by_name.loc["A1Cresult", "cohort_unknown_count"] == 0
    assert by_name.loc["readmitted", "preliminary_model_exclusion"]
    assert "direct target leakage" in by_name.loc["readmitted", "leakage_review"]
