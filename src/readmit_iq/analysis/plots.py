"""Seven purposeful EDA figures with patient-cluster uncertainty where supported."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

BLUE = "#216B87"
ACCENT = "#BF5B30"
GRAY = "#9CA6AD"
INK = "#263742"
NOTE = "Observed associations, not causal effects. Bars show 95% patient-cluster intervals."


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#D8DFE3",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def rate_dots(
    ax: plt.Axes,
    table: pd.DataFrame,
    labels: dict | None = None,
    highlight: str | None = None,
    xmax: float | None = None,
) -> None:
    labels = labels or {}
    table = table.reset_index(drop=True)
    for index, row in table.iterrows():
        color = ACCENT if row.level == highlight else BLUE
        if np.isfinite(row.ci_low):
            ax.plot([row.ci_low, row.ci_high], [index, index], color=color, lw=2)
        ax.scatter(row.rate, index, color=color, s=48, zorder=3)
        ax.annotate(
            f"{row.rate:.1%}",
            (row.rate, index),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=10,
            color=color,
        )
    ax.set_yticks(
        range(len(table)),
        [
            f"{labels.get(row.level, row.level)}  ·  n={row.encounters:,}"
            for row in table.itertuples()
        ],
    )
    ax.invert_yaxis()
    ax.set_ylim(len(table) - 0.5, -0.7)
    maximum = max(table.rate.max(), table.ci_high.max())
    ax.set_xlim(0, xmax or min(1, maximum + 0.065))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("Encounters with a <30 readmission label")
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.xaxis.grid(True, color="#E7ECEF", linewidth=0.7)
    ax.set_axisbelow(True)


def single_rates(
    directory: Path,
    name: str,
    table: pd.DataFrame,
    title: str,
    subtitle: str,
    labels: dict | None = None,
    highlight: str | None = None,
    footnote: str = NOTE,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 2.3 + len(table) * 0.46)))
    fig.subplots_adjust(left=0.29, right=0.94, bottom=0.18, top=0.77)
    fig.suptitle(title, x=0.035, y=0.975, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.885, subtitle, fontsize=11)
    rate_dots(ax, table, labels, highlight)
    fig.text(0.035, 0.025, footnote, fontsize=9, color="#546773")
    save(fig, directory, name)


def create_figures(
    frame: pd.DataFrame, rates: pd.DataFrame, missing: pd.DataFrame, policy: dict, directory: Path
) -> list[str]:
    style()
    n = len(frame)
    prevalence = float(frame.readmitted_lt30.mean())

    def select(feature: str, order: list[str]) -> pd.DataFrame:
        return rates.loc[rates.feature.eq(feature)].set_index("level").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.22, top=0.70)
    fig.suptitle(
        f"{prevalence:.1%} of eligible encounters carry a <30 readmission label",
        x=0.04,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(0.04, 0.83, f"{n:,} eligible encounters; outcome counts before any model is trained.")
    shares = [prevalence, 1 - prevalence]
    ax.barh([0, 1], shares, color=[ACCENT, GRAY], height=0.48)
    ax.set_yticks([0, 1], ["<30 readmission", ">30 or NO"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Share of eligible encounters")
    ax.tick_params(axis="y", length=0)
    for index, value in enumerate(shares):
        ax.text(value + 0.015, index, f"{value:.1%}", va="center", fontweight="bold")
    fig.text(
        0.04,
        0.035,
        "NO means no recorded readmission; complete follow-up is not established.",
        fontsize=10,
        color="#546773",
    )
    save(fig, directory, "01_target_prevalence")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.22, top=0.67, wspace=0.80)
    fig.suptitle(
        "Prior inpatient and emergency use show a stronger gradient than outpatient use",
        x=0.025,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.025, 0.855, "Prior-year visits are separate signals; a single total may hide differences."
    )
    for ax, feature, title in zip(
        axes,
        ["number_inpatient", "number_emergency", "number_outpatient"],
        ["Inpatient stays", "Emergency visits", "Outpatient visits"],
        strict=True,
    ):
        rate_dots(ax, select(feature, ["0", "1", "2", "3+"]), highlight="3+", xmax=0.34)
        ax.set_title(title, loc="left", fontsize=13, pad=24)
        ax.set_xlabel("<30 readmission rate", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
    fig.text(0.025, 0.035, NOTE + " n denotes encounters.", fontsize=10, color="#546773")
    save(fig, directory, "02_prior_utilization")

    bins = policy["eda"]["numeric_bins"]
    single_rates(
        directory,
        "03_length_of_stay",
        select("time_in_hospital", bins["time_in_hospital"]["labels"]),
        "Longer stays are associated with higher observed readmission rates",
        (
            "Length of stay is known at completed discharge; illness "
            "severity may confound the association."
        ),
        highlight="8–14 days",
    )
    single_rates(
        directory,
        "04_medication_count",
        select("num_medications", bins["num_medications"]["labels"]),
        "Readmission rates rise with medication count, then level off",
        (
            "Counts are distinct medications during the stay, not a "
            "verified discharge medication list."
        ),
        highlight="20–29",
    )
    ages = [f"[{a}-{a + 10})" for a in range(0, 100, 10)]
    single_rates(
        directory,
        "05_age",
        select("age", ages),
        "Age does not show a simple increasing readmission gradient",
        "These are unadjusted encounter rates; small pediatric groups require caution.",
        labels={f"[{a}-{a + 10})": f"{a}–{a + 9} years" for a in range(0, 100, 10)},
        footnote=NOTE + "\nNo interval for ages 0–9: only 3 positive encounters.",
    )
    labels = {
        "1": "Home",
        "6": "Home health",
        "3": "Skilled nursing",
        "4": "Intermediate care",
        "22": "Rehabilitation",
        "7": "Left against medical advice",
    }
    single_rates(
        directory,
        "06_discharge_destination",
        select("discharge_disposition_id", list(labels)),
        "Rehabilitation discharges have the highest observed rate",
        (
            "Among destinations with ≥500 encounters. Mixed rehab "
            "settings remain a cohort sensitivity."
        ),
        labels=labels,
        highlight="22",
        footnote=NOTE
        + "\n181 encounters in smaller destination groups are retained in the tables.",
    )

    missing = missing.sort_values("cohort_unknown_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=0.29, right=0.91, bottom=0.18, top=0.76)
    fig.suptitle(
        "Weight has little coverage; specialty and payer need explicit unknown categories",
        x=0.035,
        y=0.975,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.875,
        (
            "Unknown or missing values in the eligible cohort, with "
            "source markers preserved for audit."
        ),
    )
    bars = missing.cohort_unknown_pct.to_numpy() / 100
    ax.barh(range(len(missing)), bars, color=[ACCENT] + [BLUE] * (len(missing) - 1), height=0.60)
    ax.set_yticks(range(len(missing)), [s.replace("_", " ") for s in missing.variable])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.10)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Unknown / missing share of eligible encounters")
    ax.tick_params(axis="y", length=0)
    for index, value in enumerate(bars):
        label = (
            "<0.01%" if 0 < value < 0.0001 else (f"{value:.2%}" if value < 0.01 else f"{value:.1%}")
        )
        ax.text(
            value + 0.012,
            index,
            label,
            va="center",
            fontsize=10,
        )
    fig.text(
        0.035,
        0.035,
        "Includes ? and administrative unknown codes. Lab None means not measured and is separate.",
        fontsize=9,
        color="#546773",
    )
    save(fig, directory, "07_missingness")
    return [p.name for p in sorted(directory.glob("*.png"))]
