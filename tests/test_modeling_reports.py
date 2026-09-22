"""Reviewed reproductions may render reports; changed evidence must stop publication."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from readmit_iq.modeling.reports import verify_reviewed_results


@pytest.fixture
def evidence():
    directory = Path(__file__).resolve().parents[1] / "reports/modeling"
    return (
        pd.read_csv(directory / "baseline_model_results.csv"),
        pd.read_csv(directory / "feature_ablation_results.csv"),
        pd.read_csv(directory / "validation_error_groups.csv", keep_default_na=False),
    )


def test_original_reviewed_evidence_is_accepted(evidence):
    verify_reviewed_results(*evidence)


def test_reviewed_linux_forest_comparison_is_accepted(evidence):
    results, differences, errors = evidence
    results.loc[results.name.eq("random_forest"), "average_precision"] = 0.19771271712840022
    differences.loc[
        differences.model.eq("random_forest"), ["ap_difference", "ci_low", "ci_high"]
    ] = [0.0034299437929913024, -0.003170304683245369, 0.011525933373296202]
    verify_reviewed_results(results, differences, errors)


@pytest.mark.parametrize(
    "name,ap",
    [
        ("random_forest", 0.19760),
        ("random_forest", np.nan),
        ("logistic", 0.196),
    ],
)
def test_unreviewed_or_nonfinite_ap_is_rejected(evidence, name, ap):
    results, differences, errors = evidence
    results.loc[results.name.eq(name), "average_precision"] = ap
    with pytest.raises(ValueError, match="Validation results changed"):
        verify_reviewed_results(results, differences, errors)


@pytest.mark.parametrize("change", ["significant_interval", "inconsistent_delta", "confusion"])
def test_changed_forest_conclusions_are_rejected(evidence, change):
    results, differences, errors = evidence
    forest = differences.model.eq("random_forest")
    if change == "significant_interval":
        differences.loc[forest, "ci_low"] = 0.001
    elif change == "inconsistent_delta":
        differences.loc[forest, "ap_difference"] = 0.01
    else:
        results.loc[results.name.eq("random_forest"), "fp"] = 1
    with pytest.raises(ValueError, match="Forest comparison changed"):
        verify_reviewed_results(results, differences, errors)


def test_changed_high_utilization_errors_are_rejected(evidence):
    results, differences, errors = evidence
    errors.loc[errors.group.eq("prior_inpatient") & errors.level.eq("3+"), "fn"] = 218
    with pytest.raises(ValueError, match="Error evidence changed"):
        verify_reviewed_results(results, differences, errors)


def test_changed_experiment_registry_is_rejected(evidence):
    results, differences, errors = evidence
    with pytest.raises(ValueError, match="Experiment registry changed"):
        verify_reviewed_results(results.iloc[1:], differences, errors)


def test_changed_evidence_stops_before_reading_predictions_or_writing_reports(
    evidence, tmp_path, monkeypatch
):
    from readmit_iq.modeling import reports

    results, differences, errors = evidence
    results.loc[results.name.eq("random_forest"), "average_precision"] = 0.19760
    out = tmp_path / "reports"
    out.mkdir()
    results.to_csv(out / "baseline_model_results.csv", index=False)
    differences.to_csv(out / "feature_ablation_results.csv", index=False)
    errors.to_csv(out / "validation_error_groups.csv", index=False)
    (out / "phase3_summary.json").write_text("{}")
    (out / "calibration_bins.csv").write_text("model,mean_probability\n")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(
        reports, "settings", lambda: (tmp_path, {"modeling": {"report_dir": "reports"}})
    )
    with pytest.raises(ValueError, match="Validation results changed"):
        reports.build_reports()
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["reports"]
