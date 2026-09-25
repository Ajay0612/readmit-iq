"""Capacity policies, clustered uncertainty, fair comparisons and explanation correctness."""

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from readmit_iq.decision_support.context import settings
from readmit_iq.decision_support.explainability import coefficient_table, group_linear_values
from readmit_iq.decision_support.ranking import (
    business_scenario,
    capacity_count,
    capacity_table,
    gains_curve,
    policy_metrics,
    ranked_indices,
    risk_groups,
    target_mask,
)
from readmit_iq.decision_support.subgroups import error_profiles, subgroup_metrics
from readmit_iq.decision_support.uncertainty import (
    bootstrap_metrics,
    clustered_ratio,
    weighted_selection,
)
from readmit_iq.optimization.pipelines import build_pipeline


@pytest.fixture
def cohort():
    i = np.arange(200)
    return pd.DataFrame(
        dict(
            encounter_id=[f"e{j:04}" for j in i],
            patient_nbr=[f"p{j // 2}" for j in i],
            y=(i % 5 == 0).astype(int),
            time_in_hospital=1 + i % 14,
            number_inpatient=i % 4,
            number_emergency=i % 3,
            number_outpatient=i % 2,
            age="[50-60)",
            gender=np.where(i % 2, "Male", "Female"),
            race="Caucasian",
            admission_type_id=1,
            admission_source_id=7,
            discharge_disposition_id=1,
            medical_specialty="InternalMedicine",
            payer_code="?",
        )
    )


def test_ranking_is_order_invariant_and_never_uses_outcomes(cohort):
    p = np.repeat([0.1, 0.2, 0.3, 0.4], 50)
    a = target_mask(p, cohort.encounter_id, 0.15)
    order = np.random.default_rng(1).permutation(len(cohort))
    b = target_mask(p[order], cohort.encounter_id.iloc[order], 0.15)
    assert set(cohort.encounter_id[a]) == set(cohort.encounter_id.iloc[order][b])
    cohort.y = 1 - cohort.y
    np.testing.assert_array_equal(a, target_mask(p, cohort.encounter_id, 0.15))
    assert a.sum() == 30


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, np.nan])
def test_invalid_capacity_fails(fraction):
    with pytest.raises(ValueError):
        capacity_count(10, fraction)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "bad_probability", "empty"])
def test_ranking_rejects_invalid_inputs(fault):
    p, keys = [0.1, 0.2], ["a", "b"]
    if fault == "duplicate":
        keys = ["a", "a"]
    elif fault == "missing":
        keys[0] = None
    elif fault == "bad_probability":
        p[0] = np.inf
    else:
        p, keys = [], []
    with pytest.raises(ValueError):
        ranked_indices(p, keys)


def test_capacity_rounding_and_deciles_reconcile(cohort):
    f = cohort.iloc[:197]
    p = np.linspace(0.01, 0.8, len(f))
    deciles = risk_groups(f, p)
    capacities = capacity_table(f, p, [0.1, 0.2])
    assert deciles.encounters.sum() == len(f)
    assert deciles.positives.sum() == f.y.sum()
    assert deciles.iloc[0].cumulative_captured == capacities.iloc[0].tp
    assert deciles.iloc[1].cumulative_targeted == capacities.iloc[1].targeted_encounters
    assert capacities.iloc[0].targeted_encounters == 19
    curve = gains_curve(f, p)
    assert tuple(curve.iloc[0]) == (0, 0) and tuple(curve.iloc[-1]) == (1, 1)
    assert curve.readmissions_captured_fraction.is_monotonic_increasing


def test_policy_counts_unique_people_and_scenario_does_not_claim_prevention():
    row = policy_metrics(
        [1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], ["a", "a", "b", "c"], [True, True, False, False]
    )
    assert row["targeted_encounters"] == 2 and row["targeted_patients"] == 1
    assert row["recall"] == 0.5 and row["precision"] == 0.5 and row["lift"] == 1
    scenario = business_scenario(row, 10000)
    assert scenario["additional_surfaced_readmissions"] == 0
    assert scenario["prevention_effect_estimated"] is False
    assert scenario["unique_people_scaled"] is False


def test_weighted_capacity_matches_explicit_cluster_replication():
    p, ids, w = np.array([0.4, 0.9, 0.3, 0.7]), ["a", "b", "c", "d"], np.array([2, 2, 0, 3])
    order = ranked_indices(p, ids)
    selected = weighted_selection(order, w, 0.5)
    explicit = np.repeat(order, w[order])[:3]
    np.testing.assert_array_equal(selected, np.bincount(explicit, minlength=4))
    assert selected.sum() == 3


def test_patient_bootstrap_reproduces_weighted_reference_and_pairing(cohort):
    p = np.linspace(0.1, 0.9, len(cohort))
    intervals, samples = bootstrap_metrics(cohort, {"a": p, "b": p}, 0.1, 20, 42)
    again, second = bootstrap_metrics(cohort, {"a": p, "b": p}, 0.1, 20, 42)
    pd.testing.assert_frame_equal(intervals, again)
    pd.testing.assert_frame_equal(samples, second)
    pd.testing.assert_frame_equal(
        samples.loc[samples.model.eq("a")].drop(columns="model").reset_index(drop=True),
        samples.loc[samples.model.eq("b")].drop(columns="model").reset_index(drop=True),
    )
    codes, patients = pd.factorize(cohort.patient_nbr, sort=True)
    counts = np.bincount(
        np.random.default_rng(42).integers(len(patients), size=len(patients)),
        minlength=len(patients),
    )
    weight = counts[codes]
    first = samples.iloc[0]
    assert first.average_precision == pytest.approx(
        average_precision_score(cohort.y, p, sample_weight=weight)
    )
    assert first.roc_auc == pytest.approx(roc_auc_score(cohort.y, p, sample_weight=weight))


def test_ratio_uncertainty_respects_patient_clustering():
    numerator = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    denominator = np.ones(8)
    clustered = clustered_ratio(numerator, denominator, ["a", "a", "b", "b", "c", "c", "d", "d"])
    independent = clustered_ratio(numerator, denominator, np.arange(8))
    assert clustered[0] == independent[0] == 0.5
    assert clustered[2] - clustered[1] > independent[2] - independent[1]


def test_small_groups_keep_counts_but_suppress_performance(cohort):
    _, policy = settings()
    cohort.loc[:9, "race"] = "?"
    result = subgroup_metrics(
        cohort, np.full(len(cohort), 0.2), np.arange(len(cohort)) < 20, policy["subgroups"]
    )
    unknown = result.loc[result.group.eq("race") & result.level.eq("Unknown")].iloc[0]
    assert unknown.encounters == 10 and unknown.suppressed
    assert pd.isna(unknown.average_precision)
    assert result.loc[result.group.eq("race"), "encounters"].sum() == len(cohort)


def test_error_characteristics_reconcile_false_positive_and_negative_cohorts(cohort):
    p = np.linspace(0.01, 0.8, len(cohort))
    selected = target_mask(p, cohort.encounter_id, 0.1)
    summary, detail = error_profiles(cohort, p, selected)
    for name in ["false_positive", "false_negative", "low_utilization_false_negative"]:
        expected = summary.set_index("cohort").loc[name, "encounters"]
        by_group = detail.loc[detail.cohort.eq(name)].groupby("group")
        assert by_group.encounters.sum().eq(expected).all()
        np.testing.assert_allclose(by_group.fraction.sum(), 1)
    assert {"age", "admission_source", "admission_type", "discharge_destination"} <= set(
        detail.group
    )


def test_coefficients_use_original_units_and_within_field_contrasts(cohort):
    _, policy = settings()
    pipeline = build_pipeline("logistic", "raw", {"C": 0.1}, rare_min_count=2)
    with threadpool_limits(limits=2):
        pipeline.fit(cohort, cohort.y)
    table = coefficient_table(pipeline, policy["explanations"]["categorical_references"])
    base = cohort.iloc[[0]].copy()
    changed = base.copy()
    changed.number_inpatient += 1
    difference = pipeline.decision_function(changed) - pipeline.decision_function(base)
    beta = table.loc[table.feature.eq("number_inpatient"), "coefficient"].iloc[0]
    assert difference[0] == pytest.approx(beta)
    base.gender = "Female"
    changed = base.copy()
    changed.gender = "Male"
    beta = table.loc[table.feature.eq("gender") & table.level.eq("Male"), "coefficient"].iloc[0]
    assert (pipeline.decision_function(changed) - pipeline.decision_function(base))[
        0
    ] == pytest.approx(beta)


def test_grouped_linear_shap_reconstructs_actual_model(cohort):
    pipeline = build_pipeline("logistic", "raw", {"C": 0.1}, rare_min_count=2)
    with threadpool_limits(limits=2):
        pipeline.fit(cohort, cohort.y)
    x = pipeline[:-1].transform(cohort).toarray()
    explanation = shap.LinearExplainer(
        pipeline[-1], shap.maskers.Independent(x, max_samples=len(x))
    )(x[:10])
    grouped = group_linear_values(pipeline, explanation.values)
    assert grouped.shape == (10, 10)
    np.testing.assert_allclose(
        explanation.base_values + grouped.sum(axis=1),
        pipeline.decision_function(cohort.iloc[:10]),
        atol=1e-10,
    )
