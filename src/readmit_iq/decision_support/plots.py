"""Eight aggregate-only Phase 5 figures, with explicit validation/final-test boundaries."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PRIMARY = "logistic__uncalibrated"
CHALLENGER = "boosting__uncalibrated"
COLORS = {"validation": "#287C8E", "test": "#BD5636"}
LABELS = {
    "number_inpatient": "Prior inpatient visits",
    "number_emergency": "Prior emergency visits",
    "number_outpatient": "Prior outpatient visits",
    "time_in_hospital": "Length of stay",
    "discharge_disposition_id": "Discharge destination",
    "admission_source_id": "Admission source",
    "admission_type_id": "Admission type",
    "medical_specialty": "Admitting specialty",
    "age": "Age band",
    "gender": "Recorded gender",
}


def read_table(root, partition, name, primary=False):
    # Literal "None" is the no-utilization group, not a missing CSV value.
    table = pd.read_csv(
        root / f"reports/modeling/phase5/{partition}/{name}.csv",
        keep_default_na=False,
        na_values=[""],
    )
    return table.loc[table.model.eq(PRIMARY)].copy() if primary else table


def generate_figures(root, destination, include_test=False):
    """No individual records, models or predictions are read by this presentation layer."""
    if include_test:
        from readmit_iq.decision_support.final_evaluation import verify_published_results

        verify_published_results(root)
    partitions = ["validation", "test"] if include_test else ["validation"]
    last = partitions[-1]
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120}
    )
    outputs = []

    def save(fig, filename, footnote):
        fig.text(0.02, 0.018, footnote, fontsize=9, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        path = destination / filename
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        outputs.append(path)

    metrics = {p: read_table(root, p, "metrics", True).iloc[0] for p in partitions}
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for p in partitions:
        curve = read_table(root, p, f"{PRIMARY}_pr_curve")
        ax.plot(
            curve.recall,
            curve.precision,
            color=COLORS[p],
            lw=2,
            label=f"{p.title()} · AP {metrics[p].average_precision:.3f}",
        )
        ax.scatter(metrics[p].recall, metrics[p].precision, color=COLORS[p], s=60, zorder=4)
    ax.axhline(metrics[last].prevalence, color="#777777", ls="--", label="Prevalence reference")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 0.65),
        xlabel="Readmissions captured (recall)",
        ylabel="Readmission yield (precision)",
        title="Risk ranking raises yield, but most readmissions remain outside the top 10%",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    save(
        fig,
        "01_precision_recall.png",
        "Logistic · dots: top 10% · aggregate PR grid; precision 100% endpoint outside view",
    )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for p in partitions:
        table = read_table(root, p, "gains", True)
        ax.plot(
            table.population_fraction,
            table.readmissions_captured_fraction,
            color=COLORS[p],
            lw=2.5,
            label=p.title(),
        )
        ax.scatter(metrics[p].fraction_targeted, metrics[p].recall, color=COLORS[p], s=65, zorder=4)
    ax.plot([0, 1], [0, 1], color="#777777", ls="--", label="Random outreach")
    ax.axvline(0.1, color="#BBBBBB", lw=1)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Eligible discharge encounters prioritized",
        ylabel="Observed readmissions captured",
        title=f"Top 10% captures {metrics[last].recall:.1%} of {last} readmissions",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="lower right")
    save(
        fig,
        "02_cumulative_gains.png",
        "Encounter ranking · historical people counted separately · events are not prevented",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    x = np.arange(5)
    for i, p in enumerate(partitions):
        table = read_table(root, p, "capacity", True)
        positions = x + (i - (len(partitions) - 1) / 2) * 0.34
        bars = ax.bar(positions, table.lift, width=0.32, color=COLORS[p], label=p.title())
        ax.bar_label(bars, labels=[f"{v:.2f}×" for v in table.lift], padding=4, fontsize=10)
    ax.axhline(1, color="#777777", ls="--", label="Random = 1×")
    ax.set(
        xticks=x,
        xticklabels=["5%", "10%\nselected", "15%", "20%", "25%"],
        ylim=(0, 3.6),
        xlabel="Share of eligible discharge encounters prioritized",
        ylabel="Readmission yield / random yield",
        title="More outreach increases coverage while reducing lift",
    )
    ax.legend(frameon=False)
    save(
        fig,
        "03_lift_by_capacity.png",
        "The capacity grid is descriptive; 10% was selected on validation before test scoring",
    )

    scenario = json.loads(
        (root / f"reports/modeling/phase5/{last}/business_scenario.json").read_text()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    values = [scenario["random_surfaced_readmissions"], scenario["model_surfaced_readmissions"]]
    bars = ax.barh(
        ["Random outreach", "Frozen logistic"], values, color=["#BBBBBB", COLORS[last]], height=0.5
    )
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], padding=6, fontsize=15)
    ax.set(
        xlim=(0, max(values) * 1.3),
        xlabel="Observed readmissions surfaced per 10,000 eligible discharges",
        title=(
            f"Same capacity surfaces about {scenario['additional_surfaced_readmissions']:.0f} "
            "additional readmissions"
        ),
    )
    save(
        fig,
        "04_operational_scenario.png",
        f"{last.title()} scenario · about 1,000 outreach encounters · no prevention estimate",
    )

    table = read_table(root, "validation", "global_feature_influence").sort_values(
        "logistic_mean_absolute_log_odds", ascending=True
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    y = np.arange(len(table))
    axes[0].barh(y, table.logistic_mean_absolute_log_odds, color=COLORS["validation"])
    axes[0].set(
        yticks=y,
        yticklabels=[LABELS[v] for v in table.feature],
        title="Primary logistic · all 13,590 validation rows",
        xlabel="Mean |SHAP| (log odds)",
    )
    axes[1].barh(
        y - 0.18,
        table.logistic_matched_mean_absolute_log_odds,
        height=0.34,
        color=COLORS["validation"],
        label="Logistic",
    )
    axes[1].barh(
        y + 0.18,
        table.boosting_matched_mean_absolute_log_odds,
        height=0.34,
        color="#A68A48",
        label="Boosting",
    )
    axes[1].set(title="Same 256 validation encounters", xlabel="Mean |SHAP| (log odds)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Prior inpatient use and discharge destination dominate both explanations", fontsize=14
    )
    save(
        fig,
        "05_global_feature_influence.png",
        "Linear / permutation SHAP · 128 training background rows · associations, not causes",
    )

    table = read_table(root, "validation", "individual_explanations")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    names = {
        "targeted_positive": "Targeted · readmitted",
        "targeted_negative": "Targeted · not readmitted",
        "missed_positive_no_prior_inpatient": "Missed · no prior inpatient use",
    }
    for ax, (_, case) in zip(axes, table.groupby("case", sort=True), strict=True):
        row = case.iloc[0]
        ranked = case.loc[case.contribution_log_odds.abs().sort_values(ascending=False).index]
        top = ranked.head(5)
        contributions = np.r_[
            top.contribution_log_odds, ranked.iloc[5:].contribution_log_odds.sum()
        ]
        labels = [LABELS[v] for v in top.feature] + ["Other fields (sum)"]
        ax.barh(
            np.arange(6),
            contributions,
            color=["#BD5636" if v > 0 else "#287C8E" for v in contributions],
        )
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set(
            yticks=np.arange(6),
            yticklabels=labels,
            xlabel="Contribution to log odds",
            title=f"{names[row.case_type]}\nPredicted risk {row.probability:.1%}",
        )
        ax.invert_yaxis()
        total = case.contribution_log_odds.sum()
        ax.text(
            0.02,
            -0.18,
            f"Base {row.base_log_odds:.2f} + contributions {total:+.2f}",
            transform=ax.transAxes,
            fontsize=9,
        )
    fig.suptitle(
        "Different histories can raise risk; a low score can still precede readmission",
        fontsize=14,
    )
    save(
        fig,
        "06_individual_explanations.png",
        "Purposive validation cases near case-type median risk · red raises, teal lowers log odds",
    )

    wanted = [
        ("prior_inpatient", "None", "No prior inpatient use"),
        ("prior_inpatient", "Any", "Any prior inpatient use"),
        ("prior_inpatient_high", "3+", "3+ prior inpatient visits"),
        ("gender", "Female", "Recorded female"),
        ("gender", "Male", "Recorded male"),
        ("race", "AfricanAmerican", "African American"),
        ("race", "Caucasian", "Caucasian"),
        ("race", "Hispanic", "Hispanic"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, p in enumerate(partitions):
        groups = read_table(root, p, "subgroups", True)
        for i, (group, level, _) in enumerate(wanted):
            row = groups.loc[groups.group.eq(group) & groups.level.eq(level)].iloc[0]
            offset = (j - (len(partitions) - 1) / 2) * 0.2
            if row.suppressed:
                ax.text(0.02, i + offset, f"{p}: suppressed", fontsize=8)
                continue
            ax.errorbar(
                row.recall,
                i + offset,
                xerr=[[row.recall - row.recall_ci_low], [row.recall_ci_high - row.recall]],
                fmt="o",
                color=COLORS[p],
                capsize=3,
                label=p.title() if i == 0 else None,
            )
    ax.set(
        yticks=np.arange(len(wanted)),
        yticklabels=[v[2] for v in wanted],
        xlim=(0, 1),
        xlabel="Recall at the frozen top 10% policy (patient-cluster 95% interval)",
        title="Prior-utilization differences exceed the observed gender differences",
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="center right")
    save(
        fig,
        "07_subgroup_recall.png",
        "Overlapping groups · size rules apply · no fairness or discrimination conclusion",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), sharex=True, sharey=True)
    for ax, model, name in zip(
        axes, [PRIMARY, CHALLENGER], ["Primary logistic", "Boosting challenger"], strict=True
    ):
        for p in partitions:
            table = read_table(root, p, "calibration")
            table = table.loc[table.model.eq(model)]
            ax.plot(
                table.mean_probability,
                table.observed_rate,
                marker="o",
                color=COLORS[p],
                label=p.title(),
            )
        ax.plot([0, 0.35], [0, 0.35], color="#888888", ls="--")
        ax.set(xlim=(0, 0.35), ylim=(0, 0.35), xlabel="Mean predicted risk", title=name)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Observed readmission rate")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Average risk alignment does not establish calibration for every patient", fontsize=14
    )
    save(
        fig,
        "08_calibration.png",
        "Ten quantile bins · frozen uncalibrated pipelines · no post-test recalibration",
    )
    return outputs
