"""Four focused validation figures; no test curves or optimized thresholds."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve

from readmit_iq.analysis.plots import ACCENT, BLUE, GRAY, save, style

LABELS = {
    "naive_prevalence": "Prevalence baseline",
    "logistic": "Logistic regression",
    "logistic_balanced": "Logistic · balanced",
    "random_forest": "Random forest",
    "histogram_boosting": "Histogram boosting",
}


def create_figures(
    results: pd.DataFrame,
    differences: pd.DataFrame,
    calibration: pd.DataFrame,
    predictions: pd.DataFrame,
    winner: str,
    directory: Path,
) -> list[str]:
    style()
    primary = results.loc[results.role.isin([("primary"), ("baseline")])].sort_values(
        "average_precision"
    )
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.29, right=0.90, top=0.75, bottom=0.18)
    fig.suptitle(
        f"{LABELS[winner]} leads the fixed validation comparison",
        x=0.035,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.865,
        ("Primary scoring-time features only; this is a development ranking, not final selection."),
    )
    for i, row in enumerate(primary.itertuples()):
        color = ACCENT if row.name == winner else BLUE if row.name != "naive_prevalence" else GRAY
        ax.plot([row.ap_ci_low, row.ap_ci_high], [i, i], color=color, lw=2)
        ax.scatter(row.average_precision, i, color=color, s=60, zorder=3)
        ax.text(row.ap_ci_high + 0.005, i, f"{row.average_precision:.3f}", va="center", fontsize=10)
    ax.set_yticks(range(len(primary)), [LABELS[n] for n in primary.name])
    ax.set_xlim(0, float(primary.ap_ci_high.max()) + 0.055)
    ax.set_ylim(-0.6, len(primary) - 0.4)
    ax.set_xlabel("Average precision (higher is better)")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#E7ECEF", lw=0.7)
    fig.text(
        0.035,
        0.035,
        ("Bars: 95% patient-bootstrap intervals on validation. Test remains locked."),
        fontsize=10,
    )
    save(fig, directory, "01_model_comparison")

    names = list(dict.fromkeys([winner, "logistic", "random_forest"]))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.12, right=0.96)
    fig.suptitle(
        "Ranking quality improves, but high recall still costs precision",
        x=0.035,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.865,
        ("Validation precision–recall curves; no operating threshold has been selected."),
    )
    for name, color in zip(names, [ACCENT, BLUE, "#667C69"], strict=False):
        precision, recall, _ = precision_recall_curve(predictions.y, predictions[name])
        ax.plot(recall, precision, color=color, lw=1.7, label=LABELS[name])
    ax.axhline(
        predictions.y.mean(),
        color=GRAY,
        ls=("--"),
        lw=1.2,
        label=f"Prevalence {predictions.y.mean():.1%}",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.text(
        0.035,
        0.035,
        ("Unadjusted validation curves; event coverage and cohort selection remain limitations."),
        fontsize=9,
    )
    save(fig, directory, "02_precision_recall")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.13, right=0.96)
    balanced = results.set_index("name").loc["logistic_balanced"]
    title = (
        ("Balanced weights shift probabilities above observed event rates")
        if balanced.mean_probability > balanced.prevalence + 0.05
        else ("Probability quality differs across development models")
    )
    fig.suptitle(title, x=0.035, y=0.97, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.035,
        0.865,
        ("Ten validation quantile bins per model; these curves are diagnostic, not recalibrated."),
    )
    maximum = max(
        0.5,
        float(calibration.mean_probability.max()) + 0.04,
        float(calibration.observed_rate.max()) + 0.04,
    )
    ax.plot([0, maximum], [0, maximum], color=GRAY, ls="--", lw=1.2, label="Ideal agreement")
    for name, color in zip(
        calibration.model.unique(), [ACCENT, BLUE, "#8A9BA6", "#667C69"], strict=True
    ):
        part = calibration.loc[calibration.model.eq(name)]
        ax.plot(
            part.mean_probability,
            part.observed_rate,
            ("o-"),
            color=color,
            lw=1.6,
            ms=5,
            label=LABELS[name],
        )
    ax.set(
        xlim=(0, maximum),
        ylim=(0, maximum),
        xlabel=("Mean predicted probability"),
        ylabel=("Observed readmission rate"),
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.text(
        0.035,
        0.035,
        ("Point estimates without uncertainty bands; Brier score also reflects discrimination."),
        fontsize=9,
    )
    save(fig, directory, "03_initial_calibration")

    comparisons = [
        "All prior utilization",
        "Add total and any-use indicators",
        "Discharge destination",
    ]
    data = differences.set_index("comparison").loc[comparisons]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(top=0.75, bottom=0.20, left=0.34, right=0.91)
    util = data.loc["All prior utilization", "ap_difference"]
    fig.suptitle(
        f"Prior utilization adds {util:.3f} average precision to logistic regression",
        x=0.035,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.86,
        ("Paired comparisons use the same patients, estimator settings and non-ablated features."),
    )
    ax.axvline(0, color=GRAY, lw=1)
    for i, row in enumerate(data.itertuples()):
        color = ACCENT if i == 0 else BLUE
        ax.plot([row.ci_low, row.ci_high], [i, i], color=color, lw=2)
        ax.scatter(row.ap_difference, i, color=color, s=60, zorder=3)
        ax.text(row.ci_high + 0.002, i, f"{row.ap_difference:+.3f}", va="center", fontsize=10)
    ax.set_yticks(
        range(len(data)),
        [
            ("Provided counts + derived terms"),
            ("Derived terms beyond raw counts"),
            ("Confirmed discharge destination"),
        ],
    )
    ax.set_ylim(len(data) - 0.4, -0.6)
    ax.set_xlim(min(-0.01, float(data.ci_low.min()) - 0.005), float(data.ci_high.max()) + 0.025)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Validation average-precision difference")
    fig.text(
        0.035,
        0.035,
        (
            "Bars: paired 95% patient-bootstrap intervals. These are fixed "
            "ablations, not a feature search."
        ),
        fontsize=9,
    )
    save(fig, directory, "04_utilization_ablation")
    return [p.name for p in sorted(directory.glob("*.png"))]
