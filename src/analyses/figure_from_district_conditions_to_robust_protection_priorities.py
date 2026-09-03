#!/usr/bin/env python3
"""Generate district conditions and robust protection-priority maps.

Plan: Map threshold humid-heat burden, working-age scale, vulnerable-worker
scale, the equal-weight combined priority score, structural inclusion
frequency at the central 40-District capacity, and Data Support Grade.

Framework: AnaSOP Sections 5-7 separate substantive priority components from
evidence support, define the equal-percentile-rank score, and evaluate selection
stability across 420 deterministic scenarios. Section 7 Steps 5 and 8 require
the component-to-priority pathway and uncertain districts to remain spatially
auditable. PostgreSQL is queried under default_transaction_read_only=on; this
script performs no database writes.
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from psycopg import sql
from shapely.geometry import shape

from build_impact_assessment_indicators import (
    connection_args,
    fetch_frame,
    load_inputs,
    verify_read_only,
)
from table_district_priority_rankings_and_decision_robustness import (
    deterministic_robustness,
    load_coastal_flags,
)


TITLE = "From District Conditions to Robust Protection Priorities"
DEFAULT_OUTPUT = (
    ROOT
    / "data/results/figures"
    / "Figure_from_district_conditions_to_robust_protection_priorities.png"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_from_district_conditions_to_robust_protection_priorities.png"
)
DEFAULT_DISTRICT_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_from_district_conditions_to_robust_protection_priorities_districts.csv"
)
DEFAULT_CONTEXT = ROOT / "data/exp/map-context/ne_10m_admin_0_countries.zip"
DEFAULT_LAKES = ROOT / "data/exp/map-context/ne_10m_lakes.zip"

PRIMARY_CAPACITY = 40
NAVY = "#17324D"
TEXT = "#23313D"
MUTED = "#596A73"
GRID = "#FFFFFF"
WATER = "#DCEBF2"
LAND = "#EEECE6"
MISSING = "#ECEFF1"
STABLE = "#D7A92E"
SUPPORT_COLORS = {
    "High": "#3C7F7B",
    "Medium": "#D6A43A",
    "Limited": "#B45B5B",
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
    parser.add_argument("--district-output", type=Path, default=DEFAULT_DISTRICT_OUTPUT)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--lakes", type=Path, default=DEFAULT_LAKES)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_spatial_geography(
    connection: psycopg.Connection, schema: str
) -> gpd.GeoDataFrame:
    query = sql.SQL(
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
    frame = fetch_frame(connection, query)
    frame["admin2_code"] = frame["admin2_code"].astype(str)
    frame["geometry"] = frame["geometry_geojson"].map(
        lambda value: shape(json.loads(value))
    )
    return gpd.GeoDataFrame(
        frame.drop(columns="geometry_geojson"),
        geometry="geometry",
        crs="EPSG:4326",
    )


def load_country_context(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Neighbor-country context is missing: {path}")
    countries = gpd.read_file(f"zip://{path}").to_crs("EPSG:4326")
    keep = ["Cambodia", "Thailand", "Laos", "Vietnam"]
    context = countries.loc[countries["ADMIN"].isin(keep)].copy()
    if set(context["ADMIN"]) != set(keep):
        raise AssertionError("Country context lacks Cambodia or a required neighbor")
    return context


def load_lakes(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Lake context is missing: {path}")
    lakes = gpd.read_file(f"zip://{path}").to_crs("EPSG:4326")
    return lakes.cx[101.25:108.65, 9.55:15.45].copy()


def construct_map_frame(
    spatial: gpd.GeoDataFrame,
    district: pd.DataFrame,
    structural: pd.DataFrame,
) -> gpd.GeoDataFrame:
    district = district.copy()
    district["admin2_code"] = district["admin2_code"].astype(str)
    structural = structural.copy()
    structural["admin2_code"] = structural["admin2_code"].astype(str)
    district["combined_priority_score"] = (
        district["threshold_burden_percentile"]
        + district["working_age_percentile"]
        + district["vulnerable_worker_percentile"]
    ) / 3.0
    district["threshold_heat_burden_percent"] = (
        district["district_threshold_humid_heat_burden"]
        / district["district_threshold_humid_heat_burden"].sum()
        * 100.0
    )
    district["weighted_working_age_percent"] = (
        district["weighted_working_age_scale"] * 100.0
    )
    district["vulnerable_worker_percent"] = (
        district["district_vulnerable_worker_scale"]
        / district["district_vulnerable_worker_scale"].sum()
        * 100.0
    )
    frequency = f"inclusion_frequency_{PRIMARY_CAPACITY}"
    data = district.merge(
        structural[["admin2_code", frequency, "eligible_scenarios"]],
        on="admin2_code",
        how="left",
        validate="one_to_one",
    )
    data["stable_core_40"] = (
        data[frequency].ge(0.80) & data["data_support_grade"].ne("Limited")
    )
    mapped = spatial.merge(
        data.drop(columns=["admin2_name", "admin1_code", "admin1_name"]),
        on="admin2_code",
        how="left",
        validate="one_to_one",
    )
    return gpd.GeoDataFrame(mapped, geometry="geometry", crs="EPSG:4326")


def format_map_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(WATER)
    axis.set_xlim(101.25, 108.65)
    axis.set_ylim(9.55, 15.45)
    axis.set_xticks(np.arange(102, 109, 1.0))
    axis.set_yticks(np.arange(10, 16, 1.0))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}°E"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}°N"))
    axis.tick_params(
        axis="both",
        labelsize=7.0,
        colors=MUTED,
        direction="out",
        length=2.8,
        width=0.7,
        pad=2.0,
    )
    axis.grid(
        True,
        color=GRID,
        linestyle=(0, (2.2, 2.2)),
        linewidth=0.62,
        alpha=0.78,
        zorder=4,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(MUTED)
        spine.set_linewidth(0.8)
    axis.set_aspect("equal")


def draw_context(
    axis: plt.Axes,
    context: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
) -> None:
    context.plot(
        ax=axis,
        color=LAND,
        edgecolor="#7D898F",
        linewidth=0.50,
        zorder=0,
    )
    if not lakes.empty:
        lakes.plot(
            ax=axis,
            color="#C3DDE8",
            edgecolor="#92BCCB",
            linewidth=0.32,
            zorder=1,
        )
    labels = {
        "Thailand": (101.72, 13.65),
        "Laos": (105.55, 15.02),
        "Vietnam": (108.18, 12.65),
    }
    for country, (longitude, latitude) in labels.items():
        axis.text(
            longitude,
            latitude,
            country,
            ha="center",
            va="center",
            fontsize=6.6,
            color="#69777E",
            fontstyle="italic",
            zorder=5,
        )


def add_panel_heading(axis: plt.Axes, label: str, heading: str) -> None:
    axis.text(
        0.00,
        1.015,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        0.055,
        1.015,
        heading,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color=TEXT,
    )


def add_stat_badge(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.03,
        0.035,
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=TEXT,
        bbox={
            "facecolor": "white",
            "edgecolor": "#D7E0E4",
            "alpha": 0.93,
            "boxstyle": "round,pad=0.28",
        },
        zorder=6,
    )


def draw_continuous_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    legend_axis: plt.Axes,
    frame: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    *,
    column: str,
    cmap: str,
    norm: Normalize,
    legend_label: str,
    panel_label: str,
    panel_heading: str,
    statistic: str,
    stable_outline: bool = False,
) -> None:
    draw_context(axis, context, lakes)
    values = frame[column].astype(float)
    colors = plt.get_cmap(cmap)(norm(values.to_numpy()))
    frame.plot(
        ax=axis,
        color=colors,
        edgecolor="white",
        linewidth=0.24,
        missing_kwds={"color": MISSING, "edgecolor": "white"},
        zorder=2,
    )
    frame.boundary.plot(ax=axis, color="white", linewidth=0.25, zorder=3)
    if stable_outline:
        stable = frame.loc[frame["stable_core_40"]]
        if not stable.empty:
            stable.boundary.plot(
                ax=axis,
                color=STABLE,
                linewidth=1.15,
                zorder=5,
            )
    format_map_axis(axis)
    add_panel_heading(axis, panel_label, panel_heading)
    add_stat_badge(axis, statistic)
    legend_axis.set_axis_off()
    color_axis = legend_axis.inset_axes([0.10, 0.24, 0.80, 0.24])
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        cax=color_axis,
        orientation="horizontal",
        extend="max" if float(values.max()) > float(norm.vmax) else "neither",
    )
    colorbar.ax.tick_params(labelsize=6.8, colors=MUTED, length=2.3, pad=1.5)
    colorbar.outline.set_edgecolor("#8FA0A8")
    colorbar.outline.set_linewidth(0.55)
    legend_axis.text(
        0.50,
        0.94,
        legend_label,
        transform=legend_axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=TEXT,
    )


def draw_support_panel(
    axis: plt.Axes,
    legend_axis: plt.Axes,
    frame: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
) -> None:
    draw_context(axis, context, lakes)
    colors = frame["data_support_grade"].map(SUPPORT_COLORS)
    frame.plot(
        ax=axis,
        color=colors,
        edgecolor="white",
        linewidth=0.28,
        zorder=2,
    )
    frame.boundary.plot(ax=axis, color="white", linewidth=0.25, zorder=3)
    format_map_axis(axis)
    add_panel_heading(axis, "f", "Data support grade")
    counts = frame["data_support_grade"].value_counts()
    add_stat_badge(
        axis,
        " · ".join(
            [
                f"High {int(counts.get('High', 0))}",
                f"Medium {int(counts.get('Medium', 0))}",
                f"Limited {int(counts.get('Limited', 0))}",
            ]
        ),
    )
    legend_axis.set_axis_off()
    handles = [
        Patch(
            facecolor=SUPPORT_COLORS[grade],
            edgecolor="white",
            label=grade,
        )
        for grade in ["High", "Medium", "Limited"]
    ]
    legend_axis.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        handlelength=1.4,
        columnspacing=1.5,
        title="Evidence-support flag (excluded from priority score)",
        title_fontsize=7.2,
    )


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
    figure = plt.figure(figsize=(14.8, 10.2), facecolor="white")
    outer = figure.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.995,
        bottom=0.045,
        top=0.965,
        wspace=0.075,
        hspace=0.15,
    )
    map_axes: list[plt.Axes] = []
    legend_axes: list[plt.Axes] = []
    for row in range(2):
        for column in range(3):
            cell = outer[row, column].subgridspec(
                2, 1, height_ratios=[15.5, 1.15], hspace=0.02
            )
            map_axes.append(figure.add_subplot(cell[0]))
            legend_axes.append(figure.add_subplot(cell[1]))

    burden = frame["threshold_heat_burden_percent"].astype(float)
    draw_continuous_panel(
        figure,
        map_axes[0],
        legend_axes[0],
        frame,
        context,
        lakes,
        column="threshold_heat_burden_percent",
        cmap="YlOrRd",
        norm=Normalize(vmin=0.0, vmax=float(burden.quantile(0.98))),
        legend_label="Wave-balanced share of national working-age-weighted threshold heat burden (%)",
        panel_label="a",
        panel_heading="Humid-heat burden",
        statistic=f"Median {burden.median():.2f}% · top district {burden.max():.2f}%",
    )

    scale = frame["weighted_working_age_percent"].astype(float)
    draw_continuous_panel(
        figure,
        map_axes[1],
        legend_axes[1],
        frame,
        context,
        lakes,
        column="weighted_working_age_percent",
        cmap="YlGnBu",
        norm=Normalize(vmin=0.0, vmax=float(scale.quantile(0.98))),
        legend_label="Wave-balanced share of national working-age scale (%)",
        panel_label="b",
        panel_heading="Working-age scale",
        statistic=f"Median {scale.median():.2f}% · top district {scale.max():.2f}%",
    )

    vulnerable = frame["vulnerable_worker_percent"].astype(float)
    draw_continuous_panel(
        figure,
        map_axes[2],
        legend_axes[2],
        frame,
        context,
        lakes,
        column="vulnerable_worker_percent",
        cmap="PuRd",
        norm=Normalize(
            vmin=0.0,
            vmax=float(vulnerable.quantile(0.98)),
        ),
        legend_label="Share of national low-education working-age scale (%)",
        panel_label="c",
        panel_heading="Vulnerable-worker scale",
        statistic=f"Median {vulnerable.median():.2f}% · top district {vulnerable.max():.2f}%",
    )

    priority = frame["combined_priority_score"].astype(float)
    draw_continuous_panel(
        figure,
        map_axes[3],
        legend_axes[3],
        frame,
        context,
        lakes,
        column="combined_priority_score",
        cmap="YlOrBr",
        norm=Normalize(vmin=0.0, vmax=1.0),
        legend_label="Equal-weight combined priority score (0-1)",
        panel_label="d",
        panel_heading="Combined priority",
        statistic=f"Top 40 threshold ≥ {priority.nlargest(PRIMARY_CAPACITY).min():.3f}",
    )

    frequency_column = f"inclusion_frequency_{PRIMARY_CAPACITY}"
    frequency = frame[frequency_column].astype(float)
    stable_count = int(frame["stable_core_40"].sum())
    draw_continuous_panel(
        figure,
        map_axes[4],
        legend_axes[4],
        frame,
        context,
        lakes,
        column=frequency_column,
        cmap="cividis",
        norm=Normalize(vmin=0.0, vmax=1.0),
        legend_label="Selection frequency across 420 scenarios (40 districts)",
        panel_label="e",
        panel_heading="Priority inclusion frequency",
        statistic=(
            f"Stable core (≥ 0.80; non-Limited): {stable_count} districts"
        ),
        stable_outline=True,
    )

    draw_support_panel(
        map_axes[5],
        legend_axes[5],
        frame,
        context,
        lakes,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate_outputs(
    frame: gpd.GeoDataFrame,
    scenarios: pd.DataFrame,
    output: Path,
    review_output: Path,
    district_output: Path,
) -> None:
    if len(frame) != 197 or frame["admin2_code"].nunique() != 197:
        raise AssertionError(f"Expected 197 districts, observed {len(frame)}")
    required = [
        "district_threshold_humid_heat_burden",
        "weighted_working_age_scale",
        "district_vulnerable_worker_scale",
        "combined_priority_score",
        f"inclusion_frequency_{PRIMARY_CAPACITY}",
        "data_support_grade",
    ]
    if frame[required].isna().any().any():
        raise AssertionError("At least one district lacks a planned map variable")
    if len(scenarios) != 420:
        raise AssertionError(f"Expected 420 structural scenarios, observed {len(scenarios)}")
    if not np.isclose(frame["weighted_working_age_scale"].sum(), 1.0):
        raise AssertionError("Working-age scale does not sum to one")
    for path in [output, review_output, district_output]:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    district_output = resolve_path(args.district_output)
    context_path = resolve_path(args.context)
    lakes_path = resolve_path(args.lakes)

    with psycopg.connect(**connection_args(args)) as connection:
        verify_read_only(connection)
        geography, climate, people = load_inputs(connection, args.schema)
        coastal_flags = load_coastal_flags(connection, args.schema)
        spatial = load_spatial_geography(connection, args.schema)
        connection.rollback()

    district, structural, scenarios = deterministic_robustness(
        geography,
        climate,
        people,
        coastal_flags,
    )
    frame = construct_map_frame(spatial, district, structural)
    context = load_country_context(context_path)
    lakes = load_lakes(lakes_path)
    build_figure(frame, context, lakes, output)

    review_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, review_output)
    district_output.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="geometry").to_csv(district_output, index=False)
    validate_outputs(frame, scenarios, output, review_output, district_output)

    print(f"figure={output.relative_to(ROOT)}")
    print(f"review={review_output.relative_to(ROOT)}")
    print(f"district_audit={district_output.relative_to(ROOT)}")
    print(f"districts={len(frame)} structural_scenarios={len(scenarios)}")
    print(
        f"stable_core_k={PRIMARY_CAPACITY} "
        f"districts={int(frame['stable_core_40'].sum())}"
    )
    print("database_read_only=true database_writes=none")


if __name__ == "__main__":
    main()
