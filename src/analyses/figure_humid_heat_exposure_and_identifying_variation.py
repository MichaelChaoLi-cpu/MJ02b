#!/usr/bin/env python3
"""Humid-Heat Exposure and Identifying Variation.

Plan: Show current and two-month cumulative district-month exposure,
wave-specific continuous humid-heat support, and fixed-effect-residual
cumulative humid-heat variation in three publication-ready panels.

Framework: AnaSOP Section 5 identifies effects from within-District-calendar-
month deviations after Survey-Year fixed effects; Section 6 defines the main
exposure as current-plus-previous-month threshold days; Section 7 Step 2
documents the identifying support.
All PostgreSQL inputs are queried under a forced read-only session.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mj02b-matplotlib-cache")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
import seaborn as sns
from linearmodels.iv import AbsorbingLS
from psycopg import sql


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "data/results/figures"
    / "Figure_humid_heat_exposure_and_identifying_variation.png"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_humid_heat_exposure_and_identifying_variation.png"
)

NAVY = "#17324D"
TEAL = "#3C7F7B"
GOLD = "#D39A35"
PALE_TEAL = "#DCEBEA"
GRID = "#D7E0E4"
TEXT = "#23313D"
MUTED = "#5C6B73"


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
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def connection_args(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "options": "-c default_transaction_read_only=on",
    }
    if args.user:
        values["user"] = args.user
    if args.password:
        values["password"] = args.password
    return values


def load_district_month_cells(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    query = sql.SQL(
        """
        SELECT DISTINCT
            a.survey_wave,
            a.survey_year,
            a.survey_month,
            a.admin2_code,
            a.days_wbmax_ge_26c,
            a.lag_days_wbmax_ge_26c,
            a.wet_bulb_max_mean_c,
            a.lag_wet_bulb_max_mean_c,
            climate.days_observed
        FROM {}.{} AS a
        LEFT JOIN {}.{} AS climate
          ON climate.admin2_code = a.admin2_code
         AND climate.reference_year = a.survey_year
         AND climate.calendar_month = a.survey_month
        WHERE a.analysis_weight > 0
          AND a.admin2_code IS NOT NULL
          AND a.survey_year IS NOT NULL
          AND a.survey_month IS NOT NULL
        ORDER BY a.survey_year, a.survey_month, a.admin2_code
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
        sql.Identifier(schema),
        sql.Identifier("final_CLIMATE_MONTHLY_ADMIN2"),
    )
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description]
        frame = pd.DataFrame(cursor.fetchall(), columns=columns)
    numeric_columns = [
        "survey_year",
        "survey_month",
        "days_wbmax_ge_26c",
        "lag_days_wbmax_ge_26c",
        "wet_bulb_max_mean_c",
        "lag_wet_bulb_max_mean_c",
        "days_observed",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["monthly_humid_heat_share"] = (
        frame["days_wbmax_ge_26c"] / frame["days_observed"]
    ).where(frame["days_observed"].gt(0))
    frame["cumulative_humid_heat_days"] = (
        frame["days_wbmax_ge_26c"] + frame["lag_days_wbmax_ge_26c"]
    )
    frame["two_month_average_max_wet_bulb_c"] = (
        frame["wet_bulb_max_mean_c"] + frame["lag_wet_bulb_max_mean_c"]
    ) / 2.0
    return frame


def residualize_heat(cells: pd.DataFrame) -> pd.Series:
    working = cells.dropna(subset=["cumulative_humid_heat_days"]).copy()
    working["admin2_calendar_month"] = (
        working["admin2_code"].astype("string")
        + "-m"
        + working["survey_month"].astype("Int64").astype("string").str.zfill(2)
    )
    absorb = pd.DataFrame(
        {
            "admin2_calendar_month": working["admin2_calendar_month"].astype(
                "category"
            ),
            "survey_year": working["survey_year"].astype("Int64").astype("category"),
        },
        index=working.index,
    )
    model = AbsorbingLS(
        working["cumulative_humid_heat_days"].astype(float),
        pd.DataFrame({"constant": 1.0}, index=working.index),
        absorb=absorb,
        drop_absorbed=True,
    ).fit()
    return pd.Series(model.resids, index=working.index).reindex(cells.index)


def validate_cells(cells: pd.DataFrame) -> None:
    key = ["admin2_code", "survey_year", "survey_month"]
    if cells.duplicated(key).any():
        raise AssertionError("District-year-month cells are not unique")
    if len(cells) != 3_325:
        raise AssertionError(f"Expected 3,325 cells, observed {len(cells):,}")
    if cells["survey_wave"].nunique() != 9:
        raise AssertionError("Expected nine survey waves")
    if cells["days_wbmax_ge_26c"].isna().any():
        raise AssertionError("Extreme humid-heat exposure is unexpectedly missing")
    cumulative_valid = cells["cumulative_humid_heat_days"].notna()
    continuous_valid = cells["two_month_average_max_wet_bulb_c"].notna()
    if int(cumulative_valid.sum()) < 2_500:
        raise AssertionError("Too few district-month cells have cumulative exposure")
    if not cumulative_valid.equals(continuous_valid):
        raise AssertionError("Threshold and continuous cumulative coverage differ")
    residual_sd = float(cells["residual_cumulative_humid_heat_days"].std(ddof=0))
    if not np.isfinite(residual_sd) or residual_sd <= 0:
        raise AssertionError("Cumulative residual exposure has invalid variation")


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=TEXT, labelsize=9)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    sns.despine(ax=axis, top=True, right=True)


def build_figure(cells: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "axes.labelweight": "medium",
            "text.color": TEXT,
            "axes.edgecolor": MUTED,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]},
    )

    current = cells["days_wbmax_ge_26c"].astype(float)
    cumulative = cells["cumulative_humid_heat_days"].dropna().astype(float)
    bins = np.arange(-0.5, 63.5, 2.0)
    sns.histplot(
        current,
        bins=bins,
        stat="density",
        color=TEAL,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.45,
        label="Current month",
        ax=axes[0],
    )
    sns.histplot(
        cumulative,
        bins=bins,
        stat="density",
        color=NAVY,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.35,
        label="Current + previous month",
        ax=axes[0],
    )
    cumulative_mean = float(cumulative.mean())
    cumulative_median = float(cumulative.median())
    axes[0].axvline(cumulative_mean, color=GOLD, linewidth=1.8, linestyle="--")
    axes[0].set_xlabel("Days with daily maximum wet-bulb temperature >= 26 C")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(-0.5, 62.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].text(
        0.97,
        0.67,
        f"Cumulative mean {cumulative_mean:.1f}\nCumulative median {cumulative_median:.0f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=TEXT,
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )

    wave_order = (
        cells[["survey_wave", "survey_year"]]
        .drop_duplicates()
        .sort_values(["survey_year", "survey_wave"])["survey_wave"]
        .tolist()
    )
    plot_cells = cells.dropna(subset=["two_month_average_max_wet_bulb_c"]).copy()
    sns.boxplot(
        data=plot_cells,
        x="survey_wave",
        y="two_month_average_max_wet_bulb_c",
        order=wave_order,
        color=PALE_TEAL,
        linecolor=TEAL,
        linewidth=1.0,
        fliersize=0,
        width=0.65,
        ax=axes[1],
    )
    wave_labels = [
        str(label).replace("CSES ", "").replace("CSES-", "") for label in wave_order
    ]
    axes[1].set_xticks(range(len(wave_order)), wave_labels, rotation=35, ha="right")
    axes[1].set_xlabel("Survey wave")
    axes[1].set_ylabel("Two-month mean maximum wet-bulb temperature (C)")

    residual = cells["residual_cumulative_humid_heat_days"].dropna().astype(float)
    sns.histplot(
        residual,
        bins=32,
        color=NAVY,
        edgecolor="white",
        linewidth=0.35,
        ax=axes[2],
    )
    axes[2].axvline(0, color=GOLD, linewidth=1.8, linestyle="--")
    residual_sd = float(residual.std(ddof=0))
    q05, q95 = residual.quantile([0.05, 0.95])
    axes[2].set_xlabel("Residual two-month cumulative humid-heat days")
    axes[2].set_ylabel("District–month cells")
    axes[2].text(
        0.97,
        0.94,
        f"SD {residual_sd:.2f} days\nP05–P95 {q05:.1f} to {q95:.1f}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=TEXT,
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )

    for label, axis in zip("abc", axes):
        style_axis(axis)
        axis.text(
            0.00,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            color=NAVY,
            va="top",
        )

    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.22, top=0.94, wspace=0.32)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    with psycopg.connect(**connection_args(args)) as connection:
        cells = load_district_month_cells(connection, args.schema)
        connection.rollback()
    cells["residual_cumulative_humid_heat_days"] = residualize_heat(cells)
    validate_cells(cells)
    build_figure(cells, output)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    if review_output.resolve() != output.resolve():
        shutil.copy2(output, review_output)
    residual_sd = float(cells["residual_cumulative_humid_heat_days"].std(ddof=0))
    print(f"district_month_cells={len(cells)}")
    print(
        "cumulative_exposure_cells="
        f"{cells['cumulative_humid_heat_days'].notna().sum()}"
    )
    print(f"survey_waves={cells['survey_wave'].nunique()}")
    print(
        "cumulative_heat_mean_days="
        f"{cells['cumulative_humid_heat_days'].mean():.6f}"
    )
    print(f"residual_cumulative_heat_sd_days={residual_sd:.6f}")
    print(f"saved_figure={output.relative_to(ROOT)}")
    print(f"saved_review_copy={review_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
