"""Common prespecified metrics for validation and the later frozen test comparison."""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from readmit_iq.analysis.reporting import write_json
from readmit_iq.decision_support.ranking import (
    business_scenario,
    capacity_table,
    gains_curve,
    policy_metrics,
    risk_groups,
    target_mask,
    threshold_table,
)
from readmit_iq.decision_support.subgroups import error_profiles, subgroup_metrics
from readmit_iq.decision_support.uncertainty import bootstrap_metrics
from readmit_iq.modeling.metrics import calibration_bins, classification_metrics


def evaluate_predictions(frame, predictions, policy, out, partition):
    """Analyze already-scored records; no estimator fit or inference happens in this function."""
    if partition not in {"validation", "test"}:
        raise ValueError("Only prespecified evaluation partitions are allowed")
    out.mkdir(parents=True, exist_ok=True)
    metrics, capacities, deciles, gains, calibration, groups, errors, details = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for name, probability in predictions.items():
        p = np.asarray(probability)
        selected = target_mask(
            p, frame.encounter_id, policy["targeting"]["fraction"], policy["random_seed"]
        )
        row = policy_metrics(frame.y, p, frame.patient_nbr, selected)
        discrimination = classification_metrics(frame.y, p)
        row.update({key: discrimination[key] for key in ["average_precision", "roc_auc", "brier"]})
        row.update(
            model=name,
            partition=partition,
            capacity=policy["targeting"]["fraction"],
            realized_cutoff=float(p[selected].min()),
        )
        metrics.append(row)
        for accumulator, table in [
            (capacities, capacity_table(frame, p, policy["targeting"]["capacities"])),
            (deciles, risk_groups(frame, p, policy["targeting"]["risk_groups"])),
            (gains, gains_curve(frame, p)),
            (calibration, calibration_bins(frame.y, p)),
            (groups, subgroup_metrics(frame, p, selected, policy["subgroups"])),
        ]:
            accumulator.append(table.assign(model=name, partition=partition))
        error, detail = error_profiles(frame, p, selected)
        errors.append(error.assign(model=name, partition=partition))
        details.append(detail.assign(model=name, partition=partition))
        if partition == "validation":
            threshold_table(frame, p, policy["targeting"]["illustrative_thresholds"]).assign(
                model=name
            ).to_csv(out / f"{name}_thresholds.csv", index=False)
        precision, recall, _ = precision_recall_curve(frame.y, p)
        # Fixed recall grid exports aggregate curve points, not per-encounter thresholds/outcomes.
        grid = np.linspace(0, 1, 201)
        pd.DataFrame(
            {"recall": grid, "precision": np.interp(grid, recall[::-1], precision[::-1])}
        ).to_csv(out / f"{name}_pr_curve.csv", index=False)
        fpr, tpr, _ = roc_curve(frame.y, p)
        pd.DataFrame(
            {"false_positive_rate": grid, "true_positive_rate": np.interp(grid, fpr, tpr)}
        ).to_csv(out / f"{name}_roc_curve.csv", index=False)
    results = pd.DataFrame(metrics)
    results.to_csv(out / "metrics.csv", index=False)
    for name, tables in [
        ("capacity", capacities),
        ("risk_deciles", deciles),
        ("gains", gains),
        ("calibration", calibration),
        ("subgroups", groups),
        ("error_profiles", errors),
        ("error_characteristics", details),
    ]:
        pd.concat(tables, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    intervals, samples = bootstrap_metrics(
        frame,
        predictions,
        policy["targeting"]["fraction"],
        policy["uncertainty"]["bootstrap_replicates"],
        policy["random_seed"],
    )
    intervals.to_csv(out / "metric_intervals.csv", index=False)
    paired = []
    for metric in ["average_precision", "roc_auc", "brier", "recall", "precision", "lift"]:
        values = samples.pivot(index="replicate", columns="model", values=metric)
        delta = values[policy["challenger"]] - values[policy["primary"]]
        observed = results.set_index("model")[metric]
        paired.append(
            dict(
                metric=metric,
                challenger_minus_primary=observed[policy["challenger"]]
                - observed[policy["primary"]],
                ci_low=delta.quantile(0.025),
                ci_high=delta.quantile(0.975),
            )
        )
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)
    scenario = business_scenario(
        results.set_index("model").loc[policy["primary"]].to_dict(),
        policy["targeting"]["scenario_encounters"],
    )
    write_json(out / "business_scenario.json", scenario)
    summary = dict(
        partition=partition,
        primary=policy["primary"],
        challenger=policy["challenger"],
        targeting=policy["targeting"],
        metrics=metrics,
        scenario=scenario,
        bootstrap_replicates=policy["uncertainty"]["bootstrap_replicates"],
        model_refitted=False,
        calibration_changed=False,
        prevention_effect_estimated=False,
    )
    write_json(out / "summary.json", summary)
    return summary
