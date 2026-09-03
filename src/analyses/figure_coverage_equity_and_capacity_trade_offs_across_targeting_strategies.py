#!/usr/bin/env python3
"""Generate coverage, equity, and capacity trade-offs across strategies.

Plan: Compare targeted strategies at identical 20-, 40-, and 60-District
capacities using humid-heat coverage, vulnerable-worker coverage, and
working-age coverage, and display the within-capacity coverage-equity frontier.

Framework: AnaSOP Sections 5-6 define A0-A5, the three district-count capacity
levels, the coverage formulas, and the nondominated frontier. Section 7 Step 7
requires A2-A5 to be compared at identical capacity while A0 and A1 remain
fixed lower and upper references. PostgreSQL is queried under
default_transaction_read_only=on; this script performs no database writes.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from build_impact_assessment_indicators import (
    build_district_indicators,
    connection_args,
    load_inputs,
    verify_read_only,
)
from table_targeting_strategy_performance_across_capacity_levels import (
    CAPACITIES,
    WORKER_SHARE_TARGETS,
    build_performance_table,
)


TITLE = "Coverage, Equity, and Capacity Trade-offs across Targeting Strategies"
DEFAULT_OUTPUT = (
    ROOT
    / "data/results/figures"
    / "Figure_coverage_equity_and_capacity_trade_offs_across_targeting_strategies.png"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_coverage_equity_and_capacity_trade_offs_across_targeting_strategies.png"
)
DEFAULT_AUDIT_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_coverage_equity_and_capacity_trade_offs_across_targeting_strategies.csv"
)

TARGETED = ["A2", "A3", "A4", "A5"]
STRATEGY_LABELS = {
    "A2": "Heat burden",
    "A3": "Vulnerable workers",
    "A4": "Worker scale",
    "A5": "Combined compromise",
}
COLORS = {
    "A2": "#D97936",
    "A3": "#A44A73",
    "A4": "#2B7A78",
    "A5": "#D5A832",
}
MARKERS = {20: "o", 40: "s", 60: "D"}
FRONTIER_STYLES = {20: ":", 40: "--", 60: "-"}
NAVY = "#17324D"
TEXT = "#253642"
MUTED = "#60717D"
GRID = "#D8E0E4"
REFERENCE = "#9AA7AE"
PALE_GOLD = "#FFF3CC"


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
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def mark_frontier(audit: pd.DataFrame) -> pd.DataFrame:
    output = audit.copy()
    output["coverage_equity_frontier"] = False
    targeted = output[
        output["strategy"].isin(TARGETED)
        & output["capacity_definition"].eq("district_count")
    ]
    for capacity, group in targeted.groupby("capacity_target", observed=True):
        for index, row in group.iterrows():
            other = group.drop(index=index)
            weakly_better = (
                other["humid_heat_coverage"].ge(row["humid_heat_coverage"])
                & other["vulnerable_worker_coverage"].ge(
                    row["vulnerable_worker_coverage"]
                )
            )
            strictly_better = (
                other["humid_heat_coverage"].gt(row["humid_heat_coverage"])
                | other["vulnerable_worker_coverage"].gt(
                    row["vulnerable_worker_coverage"]
                )
            )
            output.loc[index, "coverage_equity_frontier"] = not bool(
                (weakly_better & strictly_better).any()
            )
    return output


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FBFCFC")
    axis.grid(
        True,
        axis="y",
        color=GRID,
        linewidth=0.75,
        linestyle=(0, (2.0, 2.0)),
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#83939B")
    axis.spines["bottom"].set_color("#83939B")
    axis.tick_params(axis="both", colors=MUTED, labelsize=8.2)


def add_panel_heading(axis: plt.Axes, label: str, heading: str) -> None:
    axis.text(
        -0.02,
        1.035,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        0.055,
        1.035,
        heading,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.4,
        fontweight="bold",
        color=TEXT,
    )


def place_end_labels(
    axis: plt.Axes,
    endpoints: list[tuple[str, float]],
    x: float,
) -> None:
    ordered = sorted(endpoints, key=lambda item: item[1])
    minimum_gap = 0.045
    adjusted: list[list[object]] = []
    for strategy, value in ordered:
        position = value
        if adjusted:
            position = max(position, float(adjusted[-1][1]) + minimum_gap)
        adjusted.append([strategy, position, value])
    if adjusted and float(adjusted[-1][1]) > 0.94:
        shift = float(adjusted[-1][1]) - 0.94
        for item in adjusted:
            item[1] = float(item[1]) - shift
    for strategy, position, value in adjusted:
        axis.plot(
            [60.5, x - 0.4],
            [float(value), float(position)],
            color=COLORS[str(strategy)],
            linewidth=0.75,
            alpha=0.8,
            clip_on=False,
        )
        axis.text(
            x,
            float(position),
            str(strategy),
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="normal",
            color=COLORS[str(strategy)],
            clip_on=False,
        )


def draw_capacity_lines(
    axis: plt.Axes,
    audit: pd.DataFrame,
    column: str,
    panel_label: str,
    heading: str,
) -> None:
    style_axis(axis)
    axis.axhline(1.0, color=REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    axis.axhline(0.0, color=REFERENCE, linewidth=0.8, zorder=1)
    axis.text(
        19.0,
        0.982,
        "A1 universal reference",
        ha="left",
        va="top",
        fontsize=7.3,
        color=MUTED,
    )
    axis.text(
        19.0,
        0.018,
        "A0 no-targeting reference",
        ha="left",
        va="bottom",
        fontsize=7.3,
        color=MUTED,
    )
    endpoints: list[tuple[str, float]] = []
    for strategy in TARGETED:
        data = audit.loc[
            audit["strategy"].eq(strategy)
            & audit["capacity_definition"].eq("district_count")
        ].sort_values("capacity_target")
        axis.plot(
            data["capacity_target"],
            data[column],
            color=COLORS[strategy],
            linewidth=2.1,
            marker="o",
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        endpoints.append((strategy, float(data[column].iloc[-1])))
    place_end_labels(axis, endpoints, 64.3)
    axis.set_xlim(18, 68)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xticks(CAPACITIES)
    axis.set_xticklabels(["20\n(10%)", "40\n(20%)", "60\n(30%)"])
    axis.set_xlabel("Implementation capacity: selected districts", fontsize=8.8, color=TEXT)
    axis.set_ylabel("National coverage", fontsize=8.8, color=TEXT)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    add_panel_heading(axis, panel_label, heading)


def bubble_size(value: float) -> float:
    return 45.0 + 520.0 * value


def draw_frontier(axis: plt.Axes, audit: pd.DataFrame) -> None:
    style_axis(axis)
    axis.grid(
        True,
        axis="both",
        color=GRID,
        linewidth=0.75,
        linestyle=(0, (2.0, 2.0)),
        zorder=0,
    )
    axis.plot(
        [0, 1],
        [0, 1],
        color="#B5C0C5",
        linewidth=1.0,
        linestyle=(0, (2, 3)),
        zorder=1,
    )
    axis.text(
        0.67,
        0.70,
        "Equal heat and equity coverage",
        rotation=39,
        fontsize=7.0,
        color="#8A989F",
        ha="center",
        va="center",
    )
    targeted = audit[
        audit["strategy"].isin(TARGETED)
        & audit["capacity_definition"].eq("district_count")
    ].copy()
    for capacity in CAPACITIES:
        frontier = targeted.loc[
            targeted["capacity_target"].eq(float(capacity))
            & targeted["coverage_equity_frontier"]
        ].sort_values("humid_heat_coverage")
        if len(frontier) > 1:
            axis.plot(
                frontier["humid_heat_coverage"],
                frontier["vulnerable_worker_coverage"],
                color="#6E7D84",
                linewidth=1.5,
                linestyle=FRONTIER_STYLES[capacity],
                alpha=0.80,
                zorder=2,
            )
    for _, row in targeted.iterrows():
        strategy = str(row["strategy"])
        capacity = int(row["capacity_target"])
        on_frontier = bool(row["coverage_equity_frontier"])
        axis.scatter(
            row["humid_heat_coverage"],
            row["vulnerable_worker_coverage"],
            s=bubble_size(float(row["working_age_coverage"])),
            marker=MARKERS[capacity],
            facecolor=COLORS[strategy],
            edgecolor="white",
            linewidth=0.9,
            alpha=1.0 if on_frontier else 0.68,
            zorder=5 if strategy == "A5" else 4,
        )
        if on_frontier:
            axis.scatter(
                row["humid_heat_coverage"],
                row["vulnerable_worker_coverage"],
                s=bubble_size(float(row["working_age_coverage"])) * 1.13,
                marker=MARKERS[capacity],
                facecolor="none",
                edgecolor=NAVY,
                linewidth=1.8,
                zorder=6,
            )
    axis.scatter(0, 0, s=48, marker="x", color=REFERENCE, linewidth=1.4, zorder=3)
    axis.scatter(1, 1, s=48, marker="x", color=REFERENCE, linewidth=1.4, zorder=3)
    axis.text(0.015, 0.015, "A0", fontsize=7.2, color=MUTED, ha="left", va="bottom")
    axis.text(0.985, 0.985, "A1", fontsize=7.2, color=MUTED, ha="right", va="top")
    axis.set_xlim(-0.02, 1.03)
    axis.set_ylim(-0.02, 1.03)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Humid-heat coverage", fontsize=8.8, color=TEXT)
    axis.set_ylabel("Vulnerable-worker coverage", fontsize=8.8, color=TEXT)
    add_panel_heading(axis, "c", "Coverage-equity frontier")

    capacity_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[capacity],
            color="#6E7D84",
            linestyle=FRONTIER_STYLES[capacity],
            markerfacecolor="white",
            markeredgecolor="#6E7D84",
            markersize=6.5,
            label=f"{capacity} districts",
        )
        for capacity in CAPACITIES
    ]
    capacity_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=NAVY,
            markeredgewidth=1.8,
            markersize=7.0,
            label="Outlined: frontier",
        )
    )
    capacity_legend = axis.legend(
        handles=capacity_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D4DDE1",
        fontsize=7.2,
        title="Capacity marker",
        title_fontsize=7.3,
    )
    axis.add_artist(capacity_legend)
    size_values = [0.20, 0.40, 0.60]
    size_handles = [
        axis.scatter(
            [],
            [],
            s=bubble_size(value),
            facecolor="none",
            edgecolor="#77868D",
            linewidth=0.9,
            label=f"{value:.0%}",
        )
        for value in size_values
    ]
    axis.legend(
        handles=size_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D4DDE1",
        fontsize=7.0,
        title="Working-age coverage",
        title_fontsize=7.3,
        labelspacing=1.0,
    )


def draw_worker_share_capacity(axis: plt.Axes, audit: pd.DataFrame) -> None:
    style_axis(axis)
    worker = audit[
        audit["strategy"].isin(TARGETED)
        & audit["capacity_definition"].eq("worker_share")
    ].copy()
    for strategy in TARGETED:
        data = worker.loc[worker["strategy"].eq(strategy)].sort_values(
            "capacity_target"
        )
        axis.plot(
            data["capacity_target"],
            data["selected_districts"],
            color=COLORS[strategy],
            linewidth=2.1,
            marker="o",
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
    minimum_label_gap = 1.75
    for target, target_rows in worker.groupby("capacity_target", sort=True):
        groups = [
            (float(district_count), rows["strategy"].tolist())
            for district_count, rows in target_rows.groupby(
                "selected_districts", sort=True
            )
        ]
        label_positions: list[float] = []
        for district_count, _ in groups:
            candidate = district_count + 0.70
            if label_positions:
                candidate = max(candidate, label_positions[-1] + minimum_label_gap)
            label_positions.append(candidate)
        for index, ((district_count, strategies), label_y) in enumerate(
            zip(groups, label_positions)
        ):
            shared = len(strategies) > 1
            x_offset = -0.0045 if index % 2 == 0 else 0.0045
            axis.annotate(
                str(int(district_count)),
                xy=(float(target), district_count),
                xytext=(float(target) + x_offset, label_y),
                ha="center",
                va="bottom",
                fontsize=6.9,
                fontweight="bold",
                color=MUTED if shared else COLORS[strategies[0]],
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": MUTED if shared else COLORS[strategies[0]],
                    "linewidth": 0.65,
                    "shrinkA": 1.5,
                    "shrinkB": 2.5,
                },
                zorder=5,
            )
    axis.set_xticks(WORKER_SHARE_TARGETS)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Target working-age coverage", fontsize=8.8, color=TEXT)
    axis.set_ylabel("Districts required", fontsize=8.8, color=TEXT)
    upper = max(10.0, float(worker["selected_districts"].max()) + 6.0)
    axis.set_ylim(0.0, upper)
    axis.text(
        0.02,
        0.95,
        "Matched worker reach\nLower count = more concentrated deployment",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": GRID},
    )
    add_panel_heading(axis, "d", "Worker-share capacity sensitivity")


def build_figure(audit: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    draw_capacity_lines(
        axes[0, 0],
        audit,
        "humid_heat_coverage",
        "a",
        "Humid-heat coverage",
    )
    draw_capacity_lines(
        axes[0, 1],
        audit,
        "vulnerable_worker_coverage",
        "b",
        "Vulnerable-worker coverage",
    )
    draw_frontier(axes[1, 0], audit)
    draw_worker_share_capacity(axes[1, 1], audit)

    strategy_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[strategy],
            linewidth=2.2,
            marker="o",
            markerfacecolor=COLORS[strategy],
            markeredgecolor="white",
            label=f"{strategy}  {STRATEGY_LABELS[strategy]}",
        )
        for strategy in TARGETED
    ]
    figure.legend(
        handles=strategy_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.008),
        ncol=4,
        frameon=False,
        fontsize=8.3,
        columnspacing=1.8,
        handlelength=2.4,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.95,
        bottom=0.10,
        wspace=0.24,
        hspace=0.33,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate_outputs(
    audit: pd.DataFrame,
    district_count: int,
    output: Path,
    review_output: Path,
    audit_output: Path,
) -> None:
    if district_count != 197:
        raise AssertionError(f"Expected 197 districts, observed {district_count}")
    if len(audit) != 36:
        raise AssertionError(f"Expected 36 strategy-capacity rows, observed {len(audit)}")
    targeted = audit[
        audit["strategy"].isin(TARGETED)
        & audit["capacity_definition"].eq("district_count")
    ]
    worker = audit[
        audit["strategy"].isin(TARGETED)
        & audit["capacity_definition"].eq("worker_share")
    ]
    if len(targeted) != 12 or len(worker) != 12:
        raise AssertionError("Expected 12 targeted rows for each capacity definition")
    if targeted.groupby("capacity_target")["coverage_equity_frontier"].sum().lt(1).any():
        raise AssertionError("Every capacity must have at least one frontier scenario")
    required = [
        "humid_heat_coverage",
        "vulnerable_worker_coverage",
        "working_age_coverage",
    ]
    if not targeted[required].apply(lambda values: values.between(0, 1).all()).all():
        raise AssertionError("Targeted coverage values must remain in [0, 1]")
    for path in [output, review_output, audit_output]:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    audit_output = resolve_path(args.audit_output)

    with psycopg.connect(**connection_args(args)) as connection:
        verify_read_only(connection)
        geography, climate, people = load_inputs(connection, args.schema)
        connection.rollback()
    district, _ = build_district_indicators(geography, climate, people)
    _, audit = build_performance_table(district)
    audit = mark_frontier(audit)
    build_figure(audit, output)

    review_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, review_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_output, index=False)
    validate_outputs(audit, len(district), output, review_output, audit_output)

    print(f"figure={output.relative_to(ROOT)}")
    print(f"review={review_output.relative_to(ROOT)}")
    print(f"scenario_audit={audit_output.relative_to(ROOT)}")
    print(f"districts={len(district)} scenarios={len(audit)}")
    for capacity in CAPACITIES:
        combined = audit.loc[
            audit["strategy"].eq("A5")
            & audit["capacity_definition"].eq("district_count")
            & audit["capacity_target"].eq(float(capacity))
        ].iloc[0]
        print(
            f"A5_k={capacity} "
            f"heat={combined['humid_heat_coverage']:.4f} "
            f"vulnerable={combined['vulnerable_worker_coverage']:.4f} "
            f"working_age={combined['working_age_coverage']:.4f} "
            f"frontier={bool(combined['coverage_equity_frontier'])}"
        )
    for worker_share in WORKER_SHARE_TARGETS:
        combined = audit.loc[
            audit["strategy"].eq("A5")
            & audit["capacity_definition"].eq("worker_share")
            & audit["capacity_target"].eq(float(worker_share))
        ].iloc[0]
        print(
            f"A5_worker_share={worker_share:.0%} "
            f"districts_required={int(combined['selected_districts'])} "
            f"working_age={combined['working_age_coverage']:.4f}"
        )
    print("database_read_only=true database_writes=none")


if __name__ == "__main__":
    main()
