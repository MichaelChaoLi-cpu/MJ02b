#!/usr/bin/env python3
"""Timing of Humid Heat and Work Participation.

Plan: Compare current, preceding, two-month cumulative, and following-month
associations under threshold and continuous wet-bulb definitions on one common
person-wave sample.

Framework: AnaSOP Section 5 treats the following month as a placebo and the
two-month window as the frozen primary timing estimand. Section 6 estimates
separate survey-weighted absorbed fixed-effect models with District-clustered
standard errors. Section 7 Step 3 uses this figure to assess whether the
association is cumulative or lagged rather than exclusively contemporaneous.
All PostgreSQL inputs are queried under a forced read-only session.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mj02b-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from audit_exposure_definition_comparison import residualize_exposure
from run_heat_labor_credibility_audit import fit_hdfe
from run_heat_labor_extensions import BASE_CONTROLS, connection_args, load_data


DEFAULT_OUTPUT = (
    ROOT
    / "data/results/figures"
    / "Figure_timing_of_humid_heat_and_work_participation.png"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_timing_of_humid_heat_and_work_participation.png"
)
DEFAULT_ESTIMATES_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_timing_of_humid_heat_and_work_participation_estimates.csv"
)

NAVY = "#17324D"
TEAL = "#3C7F7B"
GOLD = "#D39A35"
SLATE = "#7B8991"
GRID = "#D7E0E4"
TEXT = "#23313D"
MUTED = "#5C6B73"

WINDOWS = [
    ("current", "Current month"),
    ("previous", "Previous month"),
    ("cumulative", "Current + previous"),
    ("future", "Future month (placebo)"),
]

EXPOSURES = {
    "threshold": {
        "label": "WB26 threshold days",
        "terms": {
            "current": "wb26_5days",
            "previous": "lag_wb26_5days",
            "cumulative": "wb26_current_lag_total_5days",
            "future": "lead_wb26_5days",
        },
        "x_label": "Work participation (percentage points per within-FE SD of WB26 days)",
    },
    "continuous": {
        "label": "Mean daily maximum wet-bulb temperature",
        "terms": {
            "current": "wbmean_1c",
            "previous": "lag_wbmean_1c",
            "cumulative": "wbmean_current_lag_average_1c",
            "future": "lead_wbmean_1c",
        },
        "x_label": "Work participation (percentage points per within-FE SD of mean WBmax)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument(
        "--estimates-output", type=Path, default=DEFAULT_ESTIMATES_OUTPUT
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["wb26_current_lag_total_5days"] = (
        frame["wb26_5days"] + frame["lag_wb26_5days"]
    )
    frame["wbmean_current_lag_average_1c"] = (
        frame["wbmean_1c"] + frame["lag_wbmean_1c"]
    ) / 2.0
    return frame


def common_sample_mask(frame: pd.DataFrame) -> pd.Series:
    exposure_terms = [
        term
        for metadata in EXPOSURES.values()
        for term in metadata["terms"].values()
    ]
    required = [
        "worked_past_week",
        *exposure_terms,
        *BASE_CONTROLS,
        "analysis_weight",
        "admin2_calendar_month",
        "survey_year",
        "admin2_code",
    ]
    return frame[required].notna().all(axis=1) & frame["analysis_weight"].gt(0)


def common_cell_support(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    exposure_terms = [
        term
        for metadata in EXPOSURES.values()
        for term in metadata["terms"].values()
    ]
    cell_columns = [
        "survey_year",
        "survey_month",
        "admin2_code",
        "admin2_calendar_month",
        *exposure_terms,
    ]
    selected = frame.loc[mask, cell_columns].copy()
    key = ["survey_year", "survey_month", "admin2_code"]
    for term in exposure_terms:
        within_cell_unique = selected.groupby(key, observed=True)[term].nunique(dropna=False)
        if int(within_cell_unique.max()) != 1:
            raise AssertionError(f"Exposure varies within district-year-month: {term}")
    cells = selected.drop_duplicates(key).reset_index(drop=True)
    residual_sd = {}
    for term in exposure_terms:
        residual = residualize_exposure(cells, term).dropna()
        residual_sd[term] = float(residual.std(ddof=0))
    return pd.Series(residual_sd, name="within_fe_sd")


def estimate_models(frame: pd.DataFrame) -> pd.DataFrame:
    mask = common_sample_mask(frame)
    support = common_cell_support(frame, mask)
    rows: list[dict[str, object]] = []
    for family, metadata in EXPOSURES.items():
        for window, window_label in WINDOWS:
            term = metadata["terms"][window]
            records, model = fit_hdfe(
                frame,
                f"timing_common_{family}_{window}",
                "worked_past_week",
                [term],
                BASE_CONTROLS,
                sample_label="common timing sample",
                mask=mask,
            )
            record = records[0]
            if int(model.nobs) != int(mask.sum()):
                raise AssertionError("Timing models do not use the frozen common sample")
            within_fe_sd = float(support[term])
            rows.append(
                {
                    "exposure_family": family,
                    "exposure_label": metadata["label"],
                    "window": window,
                    "window_label": window_label,
                    "term": term,
                    "raw_effect_pp": 100.0 * float(record["coefficient"]),
                    "raw_ci_lower_pp": 100.0 * float(record["ci_lower_95"]),
                    "raw_ci_upper_pp": 100.0 * float(record["ci_upper_95"]),
                    "within_fe_sd": within_fe_sd,
                    "standardized_effect_pp": (
                        100.0 * float(record["coefficient"]) * within_fe_sd
                    ),
                    "standardized_ci_lower_pp": (
                        100.0 * float(record["ci_lower_95"]) * within_fe_sd
                    ),
                    "standardized_ci_upper_pp": (
                        100.0 * float(record["ci_upper_95"]) * within_fe_sd
                    ),
                    "p_value": float(record["p_value"]),
                    "n_obs": int(record["n_obs"]),
                    "n_admin2": int(record["n_admin2"]),
                    "sample": record["sample"],
                    "controls": record["controls"],
                    "fixed_effects": record["fixed_effects"],
                    "cluster_level": record["cluster_level"],
                }
            )
    estimates = pd.DataFrame(rows)
    if estimates["n_obs"].nunique() != 1:
        raise AssertionError("Timing estimates do not share one observation count")
    if estimates["n_admin2"].nunique() != 1:
        raise AssertionError("Timing estimates do not share one district count")
    return estimates


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="x", color=GRID, linewidth=0.75, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=TEXT, labelsize=9)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    sns.despine(ax=axis, top=True, right=True)


def build_figure(estimates: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "text.color": TEXT,
            "axes.edgecolor": MUTED,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.4),
        gridspec_kw={"width_ratios": [1.62, 1.0]},
    )
    forest_axis, evidence_axis = axes
    window_order = [item[0] for item in WINDOWS]
    labels = [item[1] for item in WINDOWS]
    y_positions = np.arange(len(window_order))[::-1]
    family_style = {
        "threshold": {
            "color": TEAL,
            "marker": "o",
            "label": "Cumulative WB26-day definition",
            "offset": 0.105,
        },
        "continuous": {
            "color": NAVY,
            "marker": "D",
            "label": "Mean WBmax definition",
            "offset": -0.105,
        },
    }
    all_bounds = estimates[
        ["standardized_ci_lower_pp", "standardized_ci_upper_pp"]
    ].to_numpy()
    extent = max(0.25, float(np.nanmax(np.abs(all_bounds))))
    x_limit = 1.13 * extent

    forest_axis.axhspan(
        y_positions[window_order.index("cumulative")] - 0.42,
        y_positions[window_order.index("cumulative")] + 0.42,
        color="#FFF1C9",
        alpha=0.78,
        zorder=0,
    )
    forest_axis.axhspan(
        y_positions[window_order.index("future")] - 0.42,
        y_positions[window_order.index("future")] + 0.42,
        color="#F1F4F5",
        alpha=0.95,
        zorder=0,
    )
    for family, metadata in EXPOSURES.items():
        subset = estimates.loc[estimates["exposure_family"].eq(family)].set_index(
            "window"
        ).loc[window_order]
        style = family_style[family]
        for y, window in zip(y_positions, window_order):
            row = subset.loc[window]
            estimate = float(row["standardized_effect_pp"])
            lower = float(row["standardized_ci_lower_pp"])
            upper = float(row["standardized_ci_upper_pp"])
            marker_face = "white" if window == "future" else style["color"]
            forest_axis.errorbar(
                estimate,
                y + style["offset"],
                xerr=np.array([[estimate - lower], [upper - estimate]]),
                fmt=style["marker"],
                color=style["color"],
                markerfacecolor=marker_face,
                markeredgecolor=style["color"],
                markersize=6.2,
                capsize=2.8,
                linewidth=1.45,
                zorder=3,
            )
    forest_axis.axvline(0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    forest_axis.set_xlim(-x_limit, x_limit)
    forest_axis.set_ylim(-0.55, 3.55)
    forest_axis.set_yticks(y_positions, labels)
    forest_axis.set_xlabel(
        "Work participation difference (percentage points per within-FE SD)"
    )
    forest_axis.text(
        0.98,
        0.965,
        (
            f"Common sample: N = {int(estimates['n_obs'].iloc[0]):,}\n"
            f"Districts = {int(estimates['n_admin2'].iloc[0])}"
        ),
        transform=forest_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=MUTED,
        bbox={
            "facecolor": "white",
            "edgecolor": GRID,
            "boxstyle": "round,pad=0.32",
            "alpha": 0.95,
        },
    )
    forest_axis.text(
        0.0,
        1.04,
        "a",
        transform=forest_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    forest_axis.text(
        0.06,
        1.04,
        "Standardized timing estimates and 95% CIs",
        transform=forest_axis.transAxes,
        fontsize=9.4,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    style_axis(forest_axis)

    for tick in forest_axis.get_yticklabels():
        if tick.get_text() == "Current + previous":
            tick.set_fontweight("bold")
        if tick.get_text() == "Future month (placebo)":
            tick.set_color(SLATE)

    evidence_axis.set_xlim(0, 1)
    evidence_axis.set_ylim(-0.78, 4.15)
    evidence_axis.axis("off")
    evidence_axis.text(
        0.0,
        1.04,
        "b",
        transform=evidence_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    evidence_axis.text(
        0.09,
        1.04,
        "Across-definition evidence matrix",
        transform=evidence_axis.transAxes,
        fontsize=9.4,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    column_centers = {
        "window": 0.015,
        "threshold": 0.49,
        "continuous": 0.69,
        "interval": 0.91,
    }
    evidence_axis.text(
        column_centers["threshold"],
        3.68,
        "WB26\ndays",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=MUTED,
        fontweight="bold",
    )
    evidence_axis.text(
        column_centers["continuous"],
        3.68,
        "Mean\nWBmax",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=MUTED,
        fontweight="bold",
    )
    evidence_axis.text(
        column_centers["interval"],
        3.68,
        "95% CIs\nbelow zero",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
        fontweight="bold",
        linespacing=1.05,
    )
    matrix = estimates.set_index(["window", "exposure_family"])
    row_y = {"current": 3.05, "previous": 2.17, "cumulative": 1.29, "future": 0.41}
    for window, window_label in WINDOWS:
        y = row_y[window]
        if window == "cumulative":
            background = "#FFF1C9"
        elif window == "future":
            background = "#F1F4F5"
        else:
            background = "#F8FAFA"
        evidence_axis.add_patch(
            FancyBboxPatch(
                (0.0, y - 0.34),
                0.995,
                0.68,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=0.55,
                edgecolor=GRID,
                facecolor=background,
                zorder=0,
            )
        )
        evidence_axis.text(
            column_centers["window"],
            y,
            window_label.replace("Current + previous", "Current + previous\n(primary)"),
            ha="left",
            va="center",
            fontsize=8.0,
            color=SLATE if window == "future" else TEXT,
            fontweight="bold" if window == "cumulative" else "normal",
            linespacing=1.05,
        )
        ci_count = 0
        for family in ["threshold", "continuous"]:
            row = matrix.loc[(window, family)]
            estimate = float(row["standardized_effect_pp"])
            if float(row["standardized_ci_upper_pp"]) < 0:
                ci_count += 1
            evidence_axis.text(
                column_centers[family],
                y,
                f"{estimate:.2f}",
                ha="center",
                va="center",
                fontsize=8.6,
                color=family_style[family]["color"],
                fontweight="bold" if window == "cumulative" else "normal",
            )
        support_color = NAVY if ci_count == 2 else (SLATE if ci_count == 0 else GOLD)
        evidence_axis.text(
            column_centers["interval"],
            y,
            f"{ci_count}/2",
            ha="center",
            va="center",
            fontsize=8.7,
            color=support_color,
            fontweight="bold" if ci_count == 2 else "normal",
        )
    evidence_axis.text(
        0.5,
        -0.16,
        "Only the cumulative window has both 95% CIs below zero.",
        ha="center",
        va="center",
        fontsize=7.9,
        color=NAVY,
        fontweight="bold",
        bbox={
            "facecolor": "#FFF8E4",
            "edgecolor": "#E4C36F",
            "boxstyle": "round,pad=0.35",
        },
    )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=family_style["threshold"]["marker"],
                color=family_style["threshold"]["color"],
                markerfacecolor=family_style["threshold"]["color"],
                markersize=6.2,
                linewidth=1.4,
                label="WB26 threshold-day definition",
            ),
            Line2D(
                [0],
                [0],
                marker=family_style["continuous"]["marker"],
                color=family_style["continuous"]["color"],
                markerfacecolor=family_style["continuous"]["color"],
                markersize=6.0,
                linewidth=1.4,
                label="Mean maximum wet-bulb definition",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=SLATE,
                markerfacecolor="white",
                markeredgecolor=SLATE,
                markersize=6.0,
                linewidth=0,
                label="Future exposure placebo",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        columnspacing=1.8,
        handlelength=1.8,
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.985,
        bottom=0.20,
        top=0.92,
        wspace=0.14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    estimates_output = resolve_path(args.estimates_output)
    with psycopg.connect(**connection_args(args)) as connection:
        frame = load_data(connection, args.schema)
        connection.rollback()
    frame = prepare_frame(frame)
    estimates = estimate_models(frame)
    build_figure(estimates, output)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    if review_output.resolve() != output.resolve():
        shutil.copy2(output, review_output)
    estimates_output.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(estimates_output, index=False)
    print(f"common_sample_n={int(estimates['n_obs'].iloc[0])}")
    print(f"districts={int(estimates['n_admin2'].iloc[0])}")
    for row in estimates.itertuples(index=False):
        print(
            f"{row.exposure_family}_{row.window}: "
            f"standardized_effect_pp={row.standardized_effect_pp:.6f} "
            f"p={row.p_value:.6f}"
        )
    print(f"saved_figure={output.relative_to(ROOT)}")
    print(f"saved_review_copy={review_output.relative_to(ROOT)}")
    print(f"saved_estimates={estimates_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
