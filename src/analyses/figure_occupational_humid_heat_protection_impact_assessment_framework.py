#!/usr/bin/env python3
"""Generate the occupational humid-heat protection impact-assessment framework.

Plan: Connect the empirical humid-heat and labor pathway to district decision
inputs, priority construction, action alternatives, capacity constraints,
coverage and equity consequences, and explicit uncertainty feedback.

Framework: AnaSOP Section 5 separates the exploratory person-level pathway from
the ex ante district allocation layer and keeps Data Support Grade outside the
priority score. Section 6 defines the score, alternatives, capacities, coverage
metrics, and Priority Inclusion Frequency. Section 7 Steps 5-9 integrate these
elements without estimating causal program benefits. The figure is generated as
a vector SVG with a PNG review companion and requires no database access.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SLUG = "occupational_humid_heat_protection_impact_assessment_framework"
DEFAULT_SVG = ROOT / "data/results/figures" / f"Figure_{SLUG}.svg"
DEFAULT_PNG = ROOT / "data/results/figures" / f"Figure_{SLUG}.png"
DEFAULT_REVIEW = ROOT / "data/exp/figure-table-review" / f"Figure_{SLUG}.png"

NAVY = "#17324D"
INK = "#243746"
MUTED = "#526570"
LINE = "#AFC1C9"
PALE = "#F6F8F9"
WHITE = "#FFFFFF"
HAZARD = "#D97936"
HAZARD_PALE = "#FCE9DB"
SCALE = "#2B7A78"
SCALE_PALE = "#DDEDEC"
EQUITY = "#A44A73"
EQUITY_PALE = "#F4E2EA"
SUPPORT = "#74838D"
SUPPORT_PALE = "#E9EEF0"
GOLD = "#D5A832"
GOLD_PALE = "#FFF3CC"
BLUE_PALE = "#E8F0F5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-output", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    heading: str,
    body: str,
    heading_color: str = INK,
    body_color: str = MUTED,
    heading_size: float = 7.8,
    body_size: float = 6.7,
    body_offset: float = 0.52,
    linewidth: float = 1.2,
    dashed: bool = False,
    zorder: int = 3,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    axis.add_patch(box)
    axis.text(
        x + 0.16,
        y + height - 0.18,
        heading,
        ha="left",
        va="top",
        fontsize=heading_size,
        fontweight="bold",
        color=heading_color,
        zorder=zorder + 1,
    )
    axis.text(
        x + 0.16,
        y + height - body_offset,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=body_color,
        linespacing=1.28,
        zorder=zorder + 1,
    )


def add_stage_frame(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    accent: str,
) -> None:
    """Draw a quiet visual container for one decision-framework stage."""
    frame = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        facecolor="#FCFDFD",
        edgecolor=LINE,
        linewidth=1.15,
        zorder=0,
    )
    axis.add_patch(frame)
    axis.text(
        x + 0.34,
        y + height - 0.23,
        text,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=accent,
        zorder=1,
    )
    axis.plot(
        [x + 0.16, x + width - 0.16],
        [y + height - 0.55, y + height - 0.55],
        color=LINE,
        linewidth=0.9,
        zorder=1,
    )
    axis.plot(
        [x + 0.16, x + 0.16],
        [y + height - 0.42, y + height - 0.12],
        color=accent,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=1,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    width: float = 1.8,
    connection: str = "arc3,rad=0",
    dashed: bool = False,
    zorder: int = 2,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=width,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    axis.add_patch(arrow)


def add_pill(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    label: str,
    text: str,
    color: str,
    face: str = WHITE,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        0.60,
        boxstyle="round,pad=0.03,rounding_size=0.22",
        facecolor=face,
        edgecolor=color,
        linewidth=1.2,
        zorder=4,
    )
    axis.add_patch(box)
    axis.text(
        x + 0.16,
        y + 0.30,
        label,
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=color,
        zorder=5,
    )
    axis.text(
        x + 0.57,
        y + 0.30,
        text,
        ha="left",
        va="center",
        fontsize=6.2,
        color=INK,
        zorder=5,
    )


def draw_framework() -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    # Use a compact vector canvas that preserves legibility when reduced for
    # full-width manuscript placement without forcing text into stage gutters.
    figure, axis = plt.subplots(figsize=(10.5, 5.83))
    figure.patch.set_facecolor(WHITE)
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 10)
    axis.axis("off")

    # A 0.25-unit white gutter remains visible between every stage after
    # manuscript-scale reduction, so the containers read as separate phases.
    add_stage_frame(axis, 0.20, 2.67, 3.00, 6.91, "Empirical basis", NAVY)
    add_stage_frame(axis, 3.45, 1.68, 3.55, 7.90, "District decision inputs", SCALE)
    add_stage_frame(axis, 7.25, 4.75, 2.62, 4.83, "Prioritization", GOLD)
    add_stage_frame(axis, 10.12, 3.03, 3.00, 6.55, "Action alternatives", NAVY)
    add_stage_frame(axis, 13.37, 3.92, 1.85, 5.66, "Capacity", NAVY)
    add_stage_frame(axis, 15.47, 2.18, 2.40, 7.40, "Consequences", SUPPORT)

    add_box(
        axis,
        0.40,
        7.15,
        2.60,
        1.70,
        face=HAZARD_PALE,
        edge=HAZARD,
        heading="Historical humid heat",
        body="9 waves · 197 districts\nThreshold + continuous measures",
        heading_color=HAZARD,
    )
    add_box(
        axis,
        0.40,
        5.05,
        2.60,
        1.70,
        face=BLUE_PALE,
        edge=NAVY,
        heading="Labor-risk pathway",
        body="Cumulative association with\nwork participation",
        heading_color=NAVY,
    )
    add_box(
        axis,
        0.40,
        2.95,
        2.60,
        1.70,
        face=EQUITY_PALE,
        edge=EQUITY,
        heading="Distributional context",
        body="Stronger low-education pattern\nExploratory · noncausal",
        heading_color=EQUITY,
    )

    add_arrow(axis, (3.00, 8.00), (3.60, 8.00), color=HAZARD)
    add_arrow(axis, (3.00, 5.90), (3.60, 6.05), color=NAVY)
    add_arrow(axis, (3.00, 3.80), (3.60, 4.10), color=EQUITY)

    add_box(
        axis,
        3.62,
        7.25,
        3.20,
        1.55,
        face=HAZARD_PALE,
        edge=HAZARD,
        heading="B  Humid-heat burden",
        body="Worker scale × cumulative heat\nThreshold days · 26 C excess check",
        heading_color=HAZARD,
    )
    add_box(
        axis,
        3.62,
        5.30,
        3.20,
        1.55,
        face=SCALE_PALE,
        edge=SCALE,
        heading="E  Working-age scale",
        body="Relative working-age reach\nWave-balanced population share",
        heading_color=SCALE,
    )
    add_box(
        axis,
        3.62,
        3.20,
        3.35,
        1.55,
        face=EQUITY_PALE,
        edge=EQUITY,
        heading="L  Vulnerable-worker scale",
        body="Worker scale × low-education share\nEquity objective",
        heading_color=EQUITY,
    )
    add_box(
        axis,
        3.62,
        1.95,
        3.20,
        0.95,
        face=SUPPORT_PALE,
        edge=SUPPORT,
        heading="S  Evidence support",
        body="High · Medium · Limited  |  flag only",
        heading_color=SUPPORT,
        body_size=6.3,
        dashed=True,
    )

    add_arrow(axis, (6.82, 8.00), (7.36, 7.35), color=HAZARD)
    add_arrow(axis, (6.82, 6.05), (7.36, 6.35), color=SCALE)
    add_arrow(axis, (6.82, 4.10), (7.36, 5.35), color=EQUITY)

    score_box = FancyBboxPatch(
        (7.38, 5.12),
        2.36,
        3.10,
        boxstyle="round,pad=0.06,rounding_size=0.28",
        facecolor=GOLD_PALE,
        edgecolor=GOLD,
        linewidth=2.0,
        zorder=4,
    )
    axis.add_patch(score_box)
    axis.text(
        8.58,
        7.80,
        "Allocation scores",
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=GOLD,
        zorder=5,
    )
    axis.text(
        8.58,
        7.10,
        "A2–A5",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
        zorder=5,
    )
    axis.text(
        8.58,
        6.45,
        "A2: B  ·  A3: L  ·  A4: E\nA5: mean(B, E, L)",
        ha="center",
        va="center",
        fontsize=6.7,
        color=INK,
        zorder=5,
    )
    axis.text(
        8.58,
        5.95,
        "Exact percentiles\nPure / emphasis checks",
        ha="center",
        va="center",
        fontsize=6.4,
        linespacing=1.30,
        color=MUTED,
        zorder=5,
    )
    axis.text(
        8.58,
        5.38,
        "Support excluded from score",
        ha="center",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=SUPPORT,
        zorder=5,
    )
    add_arrow(
        axis,
        (6.82, 2.42),
        (7.35, 1.24),
        color=SUPPORT,
        dashed=True,
        connection="arc3,rad=0.12",
    )

    add_arrow(axis, (9.76, 6.25), (10.25, 6.25), color=NAVY)
    alternatives = [
        ("A0", "No targeting", SUPPORT, SUPPORT_PALE),
        ("A1", "Uniform national", NAVY, PALE),
        ("A2", "Heat burden", HAZARD, WHITE),
        ("A3", "Vulnerable workers", EQUITY, WHITE),
        ("A4", "Worker scale", SCALE, WHITE),
        ("A5", "Combined", GOLD, WHITE),
    ]
    y_positions = [8.02, 7.18, 6.34, 5.50, 4.66, 3.82]
    for (label, text, color, face), y in zip(alternatives, y_positions):
        add_pill(axis, 10.27, y, 2.70, label, text, color, face)
    axis.text(
        11.62,
        3.32,
        "A0/A1 references · A2-A5 capacity-limited",
        ha="center",
        va="top",
        fontsize=5.9,
        color=MUTED,
        linespacing=1.25,
        zorder=5,
    )

    add_arrow(axis, (12.99, 6.25), (13.49, 6.25), color=NAVY)
    add_box(
        axis,
        13.51,
        4.20,
        1.56,
        4.10,
        face=PALE,
        edge=NAVY,
        heading="District count",
        body="",
        heading_color=NAVY,
        heading_size=7.4,
    )
    for y, count in [(7.25, "20"), (6.20, "40"), (5.15, "60")]:
        circle = plt.Circle((14.29, y), 0.40, facecolor=WHITE, edgecolor=NAVY, linewidth=1.5, zorder=5)
        axis.add_patch(circle)
        axis.text(14.29, y, count, ha="center", va="center", fontsize=9.0, fontweight="bold", color=NAVY, zorder=6)
    axis.text(
        14.29,
        4.48,
        "OR worker share\n10% · 20% · 30%",
        ha="center",
        va="center",
        fontsize=5.8,
        color=MUTED,
        linespacing=1.22,
        zorder=5,
    )

    add_arrow(axis, (15.09, 6.25), (15.60, 6.25), color=NAVY)
    add_box(
        axis,
        15.62,
        7.38,
        2.08,
        1.35,
        face=HAZARD_PALE,
        edge=HAZARD,
        heading="Humid-heat\ncoverage",
        body="Historical burden",
        heading_color=HAZARD,
        heading_size=6.8,
        body_size=6.0,
        body_offset=0.78,
    )
    add_box(
        axis,
        15.62,
        5.74,
        2.08,
        1.35,
        face=EQUITY_PALE,
        edge=EQUITY,
        heading="Vulnerable-worker\ncoverage",
        body="Low-education workers",
        heading_color=EQUITY,
        heading_size=6.8,
        body_size=6.0,
        body_offset=0.78,
    )
    add_box(
        axis,
        15.62,
        4.10,
        2.08,
        1.35,
        face=SCALE_PALE,
        edge=SCALE,
        heading="Working-age\ncoverage",
        body="Relative worker reach",
        heading_color=SCALE,
        heading_size=6.8,
        body_size=6.0,
        body_offset=0.78,
    )
    add_box(
        axis,
        15.62,
        2.46,
        2.08,
        1.35,
        face=SUPPORT_PALE,
        edge=SUPPORT,
        heading="Priority inclusion\nfrequency",
        body="Stable · borderline\nEvidence-limited",
        heading_color=SUPPORT,
        heading_size=6.8,
        body_size=5.8,
        body_offset=0.78,
        dashed=True,
    )

    robustness = FancyBboxPatch(
        (3.65, 0.55),
        13.90,
        0.67,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        facecolor=BLUE_PALE,
        edgecolor=NAVY,
        linewidth=1.2,
        zorder=3,
    )
    axis.add_patch(robustness)
    axis.text(
        3.88,
        0.885,
        "Decision robustness",
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=NAVY,
        zorder=4,
    )
    axis.plot(
        [5.82, 5.82],
        [0.67, 1.10],
        color=LINE,
        linewidth=0.9,
        zorder=4,
    )
    axis.text(
        6.10,
        0.885,
        "Stable core with explicit uncertainty; no single unqualified ranking",
        ha="left",
        va="center",
        fontsize=6.2,
        color=MUTED,
        zorder=4,
    )
    add_arrow(
        axis,
        (16.42, 2.43),
        (15.90, 1.24),
        color=SUPPORT,
        connection="arc3,rad=0.20",
        dashed=True,
    )
    add_arrow(
        axis,
        (8.40, 1.24),
        (8.35, 5.08),
        color=NAVY,
        connection="arc3,rad=0.10",
        dashed=True,
    )

    boundary = FancyBboxPatch(
        (0.45, 0.55),
        2.85,
        1.55,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=PALE,
        edgecolor=LINE,
        linewidth=1.0,
        zorder=3,
    )
    axis.add_patch(boundary)
    axis.text(
        0.68,
        1.82,
        "Evidence boundary",
        ha="left",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color=SUPPORT,
        zorder=4,
    )
    axis.text(
        0.68,
        1.45,
        "No causal heat effect\nNo avoided-loss estimate\nNo program-effect estimate\nNo monetized benefit",
        ha="left",
        va="top",
        fontsize=6.0,
        color=MUTED,
        linespacing=1.35,
        zorder=4,
    )

    figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return figure


def validate_outputs(svg_output: Path, png_output: Path, review_output: Path) -> None:
    for path in [svg_output, png_output, review_output]:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")
    ET.parse(svg_output)
    svg_text = svg_output.read_text(encoding="utf-8")
    required = [
        "Support excluded from score",
        "Historical burden",
        "Evidence-limited",
        "No causal heat effect",
    ]
    missing = [text for text in required if text not in svg_text]
    if missing:
        raise AssertionError(f"SVG is missing required framework text: {missing}")
    if "w_S" in svg_text or "avoided jobs" in svg_text:
        raise AssertionError("SVG contains a disallowed stale framework element")
    with Image.open(png_output) as image:
        if image.width < 2000 or image.height < 1000:
            raise AssertionError("PNG output is below the planned review resolution")
    with Image.open(review_output) as review_image, Image.open(png_output) as result_image:
        if review_image.size != result_image.size:
            raise AssertionError("Review PNG does not match the result PNG dimensions")


def main() -> None:
    args = parse_args()
    svg_output = resolve_path(args.svg_output)
    png_output = resolve_path(args.png_output)
    review_output = resolve_path(args.review_output)
    for path in [svg_output, png_output, review_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    figure = draw_framework()
    figure.savefig(
        svg_output,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Title": "Occupational Humid-Heat Protection Impact-Assessment Framework"},
    )
    figure.savefig(
        png_output,
        format="png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)
    shutil.copy2(png_output, review_output)
    validate_outputs(svg_output, png_output, review_output)
    print(f"svg={svg_output.relative_to(ROOT)}")
    print(f"png={png_output.relative_to(ROOT)}")
    print(f"review={review_output.relative_to(ROOT)}")
    print("database_access=none database_writes=none")


if __name__ == "__main__":
    main()
