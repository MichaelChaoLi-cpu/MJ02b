#!/usr/bin/env python3
"""Geography of Cumulative Humid Heat and Identifying Support.

Plan: Map district-average cumulative humid heat, contributing survey-wave
coverage, and fixed-effect-residual exposure variation in one spatial triptych.

Framework: AnaSOP Sections 5-7 define District Calendar Month and Survey Year
fixed effects as the source of identifying variation. The maps summarize where
the analytical exposure and residual variation are observed; they do not show
household exposure, regression coefficients, or district-specific causal effects.
All PostgreSQL inputs are queried under a forced read-only session.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mj02b-matplotlib-cache")
)

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from matplotlib.ticker import FuncFormatter
from psycopg import sql
from shapely.geometry import shape

from audit_exposure_definition_comparison import residualize_exposure
from run_heat_labor_extensions import connection_args


DEFAULT_OUTPUT = (
    ROOT
    / "data/results/figures"
    / "Figure_geography_of_cumulative_humid_heat_and_identifying_support.png"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_geography_of_cumulative_humid_heat_and_identifying_support.png"
)
DEFAULT_SUMMARY_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_geography_of_cumulative_humid_heat_and_identifying_support.csv"
)
DEFAULT_CONTEXT = (
    ROOT / "data/exp/map-context/ne_10m_admin_0_countries.zip"
)
DEFAULT_LAKES = (
    ROOT / "data/exp/map-context/ne_10m_lakes.zip"
)

NAVY = "#17324D"
TEXT = "#23313D"
MUTED = "#5C6B73"
MISSING = "#ECEFF1"


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
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--lakes", type=Path, default=DEFAULT_LAKES)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def fetch_frame(connection: psycopg.Connection, query: object) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def load_inputs(
    connection: psycopg.Connection, schema: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    geography_query = sql.SQL(
        """
        SELECT
            admin2_code,
            admin2_name,
            admin1_code,
            admin1_name,
            geometry_geojson::text AS geometry_geojson
        FROM {}.{}
        ORDER BY admin2_code
        """
    ).format(sql.Identifier(schema), sql.Identifier("dim_admin2_cambodia"))
    cells_query = sql.SQL(
        """
        SELECT DISTINCT
            survey_wave,
            survey_year,
            survey_month,
            admin2_code,
            days_wbmax_ge_26c,
            lag_days_wbmax_ge_26c
        FROM {}.{}
        WHERE analysis_weight > 0
        ORDER BY survey_year, survey_month, admin2_code
        """
    ).format(sql.Identifier(schema), sql.Identifier("final_HEAT_LABOR_ANALYTIC"))
    geography = fetch_frame(connection, geography_query)
    cells = fetch_frame(connection, cells_query)
    geography["geometry"] = geography["geometry_geojson"].map(
        lambda value: shape(json.loads(value))
    )
    spatial = gpd.GeoDataFrame(
        geography.drop(columns="geometry_geojson"), geometry="geometry", crs="EPSG:4326"
    )
    for column in [
        "survey_year",
        "survey_month",
        "days_wbmax_ge_26c",
        "lag_days_wbmax_ge_26c",
    ]:
        cells[column] = pd.to_numeric(cells[column], errors="coerce")
    return spatial, cells


def build_district_summary(
    spatial: gpd.GeoDataFrame, cells: pd.DataFrame
) -> gpd.GeoDataFrame:
    key = ["survey_year", "survey_month", "admin2_code"]
    if cells.duplicated(key).any():
        raise AssertionError("District-year-month climate cells are not unique")
    cells = cells.copy()
    cells["cumulative_humid_heat_days"] = (
        cells["days_wbmax_ge_26c"] + cells["lag_days_wbmax_ge_26c"]
    )
    cells["admin2_calendar_month"] = (
        cells["admin2_code"].astype("string")
        + "-m"
        + cells["survey_month"].astype("Int64").astype("string").str.zfill(2)
    )
    cells["residual_cumulative_humid_heat_days"] = residualize_exposure(
        cells, "cumulative_humid_heat_days"
    )
    summary = (
        cells.groupby("admin2_code", observed=True)
        .agg(
            mean_cumulative_humid_heat_days=("cumulative_humid_heat_days", "mean"),
            contributing_survey_waves=("survey_wave", "nunique"),
            contributing_district_month_cells=("cumulative_humid_heat_days", "count"),
            residual_exposure_sd_days=(
                "residual_cumulative_humid_heat_days",
                lambda values: float(values.std(ddof=0)),
            ),
        )
        .reset_index()
    )
    result = spatial.merge(summary, on="admin2_code", how="left", validate="one_to_one")
    return gpd.GeoDataFrame(result, geometry="geometry", crs=spatial.crs)


def load_country_context(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Neighbor-country context is missing: {path}. "
            "Download the Natural Earth 1:10m Admin 0 Countries archive."
        )
    countries = gpd.read_file(f"zip://{path}").to_crs("EPSG:4326")
    keep = ["Cambodia", "Thailand", "Laos", "Vietnam"]
    context = countries.loc[countries["ADMIN"].isin(keep)].copy()
    if set(context["ADMIN"]) != set(keep):
        raise AssertionError("Natural Earth context does not contain all required countries")
    return context


def load_lake_context(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Lake context is missing: {path}. "
            "Download the Natural Earth 1:10m Lakes archive."
        )
    lakes = gpd.read_file(f"zip://{path}").to_crs("EPSG:4326")
    return lakes.cx[101.25:108.65, 9.55:15.45].copy()


def add_stat_box(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.03,
        0.04,
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=TEXT,
        bbox={
            "facecolor": "white",
            "edgecolor": "#D7E0E4",
            "alpha": 0.94,
            "boxstyle": "round,pad=0.35",
        },
        zorder=5,
    )


def draw_panel(
    axis: plt.Axes,
    frame: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    column: str,
    cmap: str,
    legend_label: str,
    statistic: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    axis.set_facecolor("#DCEBF2")
    context.plot(
        ax=axis,
        color="#EEECE6",
        edgecolor="#7D898F",
        linewidth=0.55,
        zorder=0,
    )
    if not lakes.empty:
        lakes.plot(
            ax=axis,
            color="#C3DDE8",
            edgecolor="#92BCCB",
            linewidth=0.35,
            zorder=1,
        )
    frame.plot(
        column=column,
        cmap=cmap,
        linewidth=0.22,
        edgecolor="white",
        missing_kwds={"color": MISSING, "edgecolor": "white"},
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={
            "label": legend_label,
            "orientation": "horizontal",
            "shrink": 0.78,
            "pad": 0.105,
            "aspect": 28,
        },
        ax=axis,
        zorder=2,
    )
    frame.boundary.plot(
        ax=axis, linewidth=0.28, color="#FFFFFF", alpha=0.95, zorder=3
    )
    min_x, min_y, max_x, max_y = 101.25, 9.55, 108.65, 15.45
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_xticks(np.arange(102, 109, 1.0))
    axis.set_yticks(np.arange(10, 16, 1.0))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}°E"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}°N"))
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=7.5,
        colors=MUTED,
        direction="out",
        length=3.0,
        width=0.7,
        pad=2.5,
    )
    axis.grid(
        True,
        which="major",
        color="#FFFFFF",
        linestyle=(0, (2.2, 2.2)),
        linewidth=0.65,
        alpha=0.72,
        zorder=4,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(MUTED)
        spine.set_linewidth(0.8)
    axis.set_aspect("equal")
    country_labels = {
        "Thailand": (101.72, 13.65),
        "Laos": (105.55, 15.02),
        "Vietnam": (108.18, 12.65),
    }
    for country, (longitude, latitude) in country_labels.items():
        axis.text(
            longitude,
            latitude,
            country,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#68767E",
            fontstyle="italic",
            zorder=5,
        )
    add_stat_box(axis, statistic)


def build_figure(
    frame: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 6.0))
    mean_heat = frame["mean_cumulative_humid_heat_days"]
    wave_count = frame["contributing_survey_waves"]
    residual_sd = frame["residual_exposure_sd_days"]
    draw_panel(
        axes[0],
        frame,
        context,
        lakes,
        "mean_cumulative_humid_heat_days",
        "YlOrRd",
        "Mean cumulative WB26 days across matched survey months",
        f"Median district: {mean_heat.median():.1f} days\nRange: {mean_heat.min():.1f}-{mean_heat.max():.1f}",
        vmin=float(mean_heat.quantile(0.02)),
        vmax=float(mean_heat.quantile(0.98)),
    )
    draw_panel(
        axes[1],
        frame,
        context,
        lakes,
        "contributing_survey_waves",
        "Blues",
        "Survey waves contributing cumulative exposure",
        f"All 9 waves: {int(wave_count.eq(9).sum())} districts\nMedian: {wave_count.median():.0f} waves",
        vmin=1,
        vmax=9,
    )
    draw_panel(
        axes[2],
        frame,
        context,
        lakes,
        "residual_exposure_sd_days",
        "viridis",
        "Within-FE SD of cumulative WB26 days",
        f"Median district SD: {residual_sd.median():.1f} days\nPositive support: {int(residual_sd.gt(0).sum())} districts",
        vmin=float(residual_sd.quantile(0.02)),
        vmax=float(residual_sd.quantile(0.98)),
    )
    for label, axis in zip("abc", axes, strict=True):
        axis.text(
            0.00,
            1.01,
            label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            color=NAVY,
            ha="left",
            va="bottom",
        )
    figure.subplots_adjust(left=0.04, right=0.995, bottom=0.12, top=0.96, wspace=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate(frame: gpd.GeoDataFrame, output: Path) -> None:
    if len(frame) != 197 or frame["admin2_code"].nunique() != 197:
        raise AssertionError(f"Expected 197 districts, observed {len(frame)}")
    required = [
        "mean_cumulative_humid_heat_days",
        "contributing_survey_waves",
        "residual_exposure_sd_days",
    ]
    if frame[required].isna().any().any():
        raise AssertionError("At least one district lacks map support statistics")
    if not output.exists() or output.stat().st_size == 0:
        raise AssertionError("Map output is missing or empty")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    summary_output = resolve_path(args.summary_output)
    context_path = resolve_path(args.context)
    lakes_path = resolve_path(args.lakes)
    with psycopg.connect(**connection_args(args)) as connection:
        spatial, cells = load_inputs(connection, args.schema)
        connection.rollback()
    frame = build_district_summary(spatial, cells)
    context = load_country_context(context_path)
    lakes = load_lake_context(lakes_path)
    build_figure(frame, context, lakes, output)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    if review_output.resolve() != output.resolve():
        shutil.copy2(output, review_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="geometry").to_csv(summary_output, index=False)
    validate(frame, output)
    print(f"districts={len(frame)}")
    print(f"district_month_cells={int(frame['contributing_district_month_cells'].sum())}")
    print(f"saved_figure={output.relative_to(ROOT)}")
    print(f"saved_review_copy={review_output.relative_to(ROOT)}")
    print(f"saved_summary={summary_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
