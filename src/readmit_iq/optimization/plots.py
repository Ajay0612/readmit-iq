"""Six focused Phase 4 figures drawn only from training CV and cached validation outputs."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from readmit_iq.optimization.contract import settings

LABELS = {
    "logistic": "Logistic regression",
    "boosting": "Histogram boosting",
    "forest": "Forest reference",
}
COLORS = {"logistic": "#087e8b", "boosting": "#3157a4", "forest": "#92989f"}


def create_figures() -> list[str]:
    root, policy = settings()
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    out, figures = root / policy["report_dir"], root / policy["figure_dir"]
    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text())
    metrics = pd.read_csv(out / "validation_metrics.csv").set_index("model")
    features = pd.read_csv(out / "feature_comparison.csv")
    fold_metrics = pd.read_csv(out / "tuning_fold_metrics.csv")
    reliability = pd.read_csv(out / "calibration_bins.csv")
    thresholds = pd.read_csv(out / "thresholds.csv")
    predictions = pd.read_csv(root / policy["artifact_dir"] / "validation_predictions.csv.gz")
    names = [summary["primary"], summary["challenger"]]
    saved = []
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "savefig.facecolor": "white",
        }
    )

    def save(fig, name):
        fig.savefig(figures / name, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved.append(str((figures / name).relative_to(root)))

    # 1. Dots preserve honest comparisons without truncated bar lengths.
    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    for i, family in enumerate(["logistic", "forest", "boosting"]):
        result = summary["training"][family]
        mean, sd = result["cv_average_precision_mean"], result["cv_average_precision_sd"]
        ax.errorbar(mean, i, xerr=sd, fmt="o", markersize=8, capsize=5, color=COLORS[family])
        ax.annotate(
            f"{mean:.3f} ± {sd:.3f}",
            (mean + sd, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    ax.set(
        yticks=range(3),
        yticklabels=[LABELS[x] for x in ["logistic", "forest", "boosting"]],
        xlabel="Training CV average precision — dot: mean; line: ±1 fold SD",
        title="Boosting leads training CV; independent validation decides its value",
    )
    low = min(s["cv_ap_min"] for s in summary["training"].values()) - 0.004
    high = max(s["cv_ap_max"] for s in summary["training"].values()) + 0.015
    ax.set_xlim(low, high)
    ax.set_ylim(-0.6, 2.6)
    ax.grid(axis="x", alpha=0.15)
    save(fig, "01_cv_comparison.png")

    # 2. PR curves for the selected development candidates, not the test.
    fig, ax = plt.subplots(figsize=(8.8, 5), layout="constrained")
    for name in names:
        family = name.split("__")[0]
        precision, recall, _ = precision_recall_curve(predictions.y, predictions[name])
        ax.plot(
            recall,
            precision,
            color=COLORS[family],
            lw=1.8,
            label=f"{LABELS[family]} · AP {metrics.loc[name, 'average_precision']:.3f}",
        )
    ax.axhline(
        predictions.y.mean(),
        color="#787878",
        ls="--",
        lw=1.2,
        label=f"Prevalence baseline · {predictions.y.mean():.3f}",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precision",
        title="Validation ranking remains modest for both development candidates",
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.12)
    save(fig, "02_validation_precision_recall.png")

    # 3. One panel per family prevents six crossing calibration curves.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    method_colors = {"uncalibrated": "#263445", "sigmoid": "#087e8b", "isotonic": "#bc751c"}
    for ax, family in zip(axes, ["logistic", "boosting"], strict=True):
        ax.plot([0, 0.5], [0, 0.5], ls="--", color="#aaaaaa", lw=1)
        for method, color in method_colors.items():
            name = f"{family}__{method}"
            bins = reliability.loc[reliability.model.eq(name)]
            ax.plot(
                bins.mean_probability,
                bins.observed_rate,
                "o-",
                markersize=4,
                color=color,
                label=f"{method.capitalize()} · {metrics.loc[name, 'brier']:.5f}",
            )
        ax.set(
            xlim=(0, 0.4),
            ylim=(0, 0.4),
            xlabel="Mean predicted probability",
            ylabel="Observed readmission rate",
            title=LABELS[family],
        )
        ax.legend(title="Method · validation Brier", frameon=False, fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.12)
    fig.suptitle(
        "Calibration is a tradeoff: lower Brier need not preserve ranking",
        x=0.01,
        ha="left",
        fontsize=14,
    )
    save(fig, "03_calibration_comparison.png")

    # 4. Fixed illustrative thresholds; no optimizer or chosen operating point.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    primary = thresholds.loc[thresholds.model.eq(summary["primary"]) & thresholds.threshold.le(0.4)]
    axes[0].plot(primary.threshold, primary.recall, label="Recall", color="#087e8b", lw=2)
    axes[0].plot(primary.threshold, primary.precision, label="Precision", color="#bc751c", lw=2)
    axes[0].set(
        title=f"Primary: {LABELS[summary['primary'].split('__')[0]]}",
        xlabel="Illustrative risk threshold",
        ylabel="Validation rate",
        ylim=(0, 1),
    )
    for name in names:
        part = thresholds.loc[thresholds.model.eq(name) & thresholds.threshold.le(0.4)]
        family = name.split("__")[0]
        axes[1].plot(
            part.threshold,
            part.encounter_fraction_flagged,
            color=COLORS[family],
            lw=2,
            label=LABELS[family],
        )
    axes[1].set(
        title="Historical encounters flagged",
        xlabel="Illustrative risk threshold",
        ylabel="Share of validation encounters",
        ylim=(0, 1),
    )
    for ax in axes:
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.12)
        ax.set_xlim(0, 0.4)
    fig.suptitle(
        "More recall requires more flags; no operational threshold is selected",
        x=0.01,
        ha="left",
        fontsize=14,
    )
    save(fig, "04_threshold_behavior.png")

    # 5. Mark uncertainty in feature eligibility separately from sampling variability.
    order = ["raw", "indicators", "raw_engineered", "diagnosis_grouped", "diagnosis_raw"]
    labels = [
        "Raw utilization",
        "Any-use indicators only",
        "Raw + totals / indicators",
        "Raw + grouped diagnoses*",
        "Raw + raw diagnoses*",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained", sharex=True)
    for ax, family in zip(axes, ["logistic", "boosting"], strict=True):
        table = features.loc[features.family.eq(family)].set_index("configuration")
        for i, configuration in enumerate(order):
            row = table.loc[configuration]
            color = "#bc751c" if configuration.startswith("diagnosis") else COLORS[family]
            ax.errorbar(
                row.cv_average_precision_mean,
                i,
                xerr=row.cv_average_precision_sd,
                fmt="o",
                capsize=4,
                color=color,
                markersize=6,
            )
        ax.set(
            yticks=range(len(order)),
            yticklabels=labels,
            xlabel="CV AP mean ±1 fold SD",
            title=LABELS[family],
            ylim=(4.6, -0.6),
        )
        ax.grid(axis="x", alpha=0.12)
    fig.suptitle(
        "Raw counts retain information; diagnosis gains remain timing-restricted",
        x=0.01,
        ha="left",
        fontsize=14,
    )
    fig.supxlabel(
        "*Sensitivity only; ineligible for primary selection. Fixed baseline parameters.",
        fontsize=9,
    )
    save(fig, "05_feature_configuration.png")

    # 6. Common folds make variability directly inspectable; the x axis is not time.
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for family in ["logistic", "boosting", "forest"]:
        best = summary["training"][family]["best_trial"]
        selected = fold_metrics.loc[
            fold_metrics.family.eq(family) & fold_metrics.trial.eq(best)
        ].sort_values("fold")
        ax.plot(
            selected.fold + 1,
            selected.average_precision,
            "o-",
            lw=1.7,
            color=COLORS[family],
            label=LABELS[family],
        )
    ax.set(
        xticks=np.arange(1, 6),
        xlabel="Patient-grouped training holdout fold (shared across models)",
        ylabel="Average precision",
        title="Fold variation is larger than small model-score differences",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.12)
    save(fig, "06_fold_stability.png")
    return saved
