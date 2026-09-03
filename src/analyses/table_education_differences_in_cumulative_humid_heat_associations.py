#!/usr/bin/env python3
"""Generate the education-difference cumulative humid-heat table.

Plan: Report higher-education and low-education cumulative heat slopes together
with the low-minus-higher interaction under both frozen exposure definitions.

Framework: AnaSOP Section 6 defines the education interaction model, group
slopes, and interaction estimand. Section 7 Step 5 requires both groups and the
interaction under both cumulative humid-heat definitions. The exploratory-
family and global BH values apply only to the two registered interaction terms.
All person-level analytical data are queried from PostgreSQL under a forced
read-only session; the database is never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Border, Side
from openpyxl.utils import get_column_letter

from figure_cumulative_humid_heat_by_education import (
    EXPOSURES,
    common_sample_mask,
    connection_args,
    estimate_models,
    load_data,
    prepare_frame,
)
from table_main_cumulative_humid_heat_and_work_participation_estimates import (
    LIGHT_BORDER,
    MEDIUM_BORDER,
    MUTED,
    NAVY,
    PALE_BLUE,
    PALE_GOLD,
    PALE_TEAL,
    TEXT,
    WHITE,
    render_review_png,
    resolve_path,
    set_cell_style,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "data/results/tables"
    / "Table_education_differences_in_cumulative_humid_heat_associations.xlsx"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Table_education_differences_in_cumulative_humid_heat_associations.png"
)
DEFAULT_MULTIPLICITY_AUDIT = ROOT / "data/exp/storyline-search/target_tests.csv"

DISPLAY = {
    "threshold": {
        "exposure": "Cumulative WB26 threshold days",
        "unit": "5 additional days across 2 months",
    },
    "continuous": {
        "exposure": "Two-month mean maximum wet-bulb temperature",
        "unit": "1 C higher 2-month average",
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
        "--multiplicity-audit", type=Path, default=DEFAULT_MULTIPLICITY_AUDIT
    )
    return parser.parse_args()


def load_interaction_adjustments(path: Path) -> pd.DataFrame:
    audit = pd.read_csv(path)
    terms = {
        "threshold": "wb26_current_lag_total_5days_x_low_education",
        "continuous": "wbmean_current_lag_average_1c_x_low_education",
    }
    selected = audit.loc[
        audit["analysis_domain"].eq("cumulative_heterogeneity")
        & audit["term"].isin(terms.values())
        & audit["is_target_test"].eq(True)
    ].copy()
    if len(selected) != 2 or selected["term"].nunique() != 2:
        raise AssertionError("Expected two registered cumulative education interactions")
    inverse = {term: family for family, term in terms.items()}
    selected["exposure_family"] = selected["term"].map(inverse)
    return selected.set_index("exposure_family")


def weighted_share(mask: pd.Series, values: pd.Series, weights: pd.Series) -> float:
    selected = mask & values.notna() & weights.notna() & weights.gt(0)
    return float(np.average(values.loc[selected].astype(float), weights=weights.loc[selected]))


def enrich_estimates(
    frame: pd.DataFrame,
    estimates: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> pd.DataFrame:
    mask = common_sample_mask(frame)
    common_n = int(mask.sum())
    low_n = int((mask & frame["low_education"].eq(1)).sum())
    high_n = int((mask & frame["low_education"].eq(0)).sum())
    low_share = weighted_share(mask, frame["low_education"], frame["analysis_weight"])
    rows: list[dict[str, object]] = []
    for row in estimates.itertuples(index=False):
        family = str(row.exposure_family)
        is_interaction = row.row_type == "interaction"
        family_q = np.nan
        global_q = np.nan
        if is_interaction:
            registered = adjustments.loc[family]
            if not np.isclose(float(row.p_value), float(registered["p_value"]), atol=2e-6):
                raise AssertionError(f"Registered interaction p-value mismatch: {family}")
            if not np.isclose(
                float(row.raw_effect_pp), float(registered["effect_display"]), atol=2e-6
            ):
                raise AssertionError(f"Registered interaction estimate mismatch: {family}")
            family_q = float(registered["family_bh_q_value"])
            global_q = float(registered["global_bh_q_value"])

        if row.row_type == "higher_education":
            coverage = f"Higher education: {high_n:,}"
            judgment = "Adverse slope; CI excludes zero" if row.raw_ci_upper_pp < 0 else "Adverse slope; imprecise"
        elif row.row_type == "low_education":
            coverage = f"Low education: {low_n:,}"
            judgment = "Adverse slope; CI excludes zero" if row.raw_ci_upper_pp < 0 else "Adverse slope; imprecise"
        else:
            coverage = f"Both groups; low weighted share {low_share:.1%}"
            judgment = (
                "Low-education slope more adverse; CI excludes zero"
                if row.raw_effect_pp < 0 and row.raw_ci_upper_pp < 0
                else "Education difference is imprecise"
            )
        rows.append(
            {
                **row._asdict(),
                "exposure_display": DISPLAY[family]["exposure"],
                "effect_unit": DISPLAY[family]["unit"],
                "exploratory_family_bh_q": family_q,
                "global_bh_q": global_q,
                "group_coverage": coverage,
                "evidence_judgment": judgment,
                "common_sample_n": common_n,
                "low_education_n": low_n,
                "higher_education_n": high_n,
                "weighted_low_education_share": low_share,
            }
        )
    enriched = pd.DataFrame(rows)
    if enriched["n_obs"].nunique() != 1 or int(enriched["n_obs"].iloc[0]) != common_n:
        raise AssertionError("Education estimates do not use the reconstructed common sample")
    return enriched


def format_ci(lower: float, upper: float) -> str:
    return f"[{lower:.3f}, {upper:.3f}]"


def format_q(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def format_p(value: float) -> str:
    """Return the strongest manuscript-approved p-value threshold."""
    if pd.isna(value) or value >= 0.10:
        return "—"
    if value < 0.01:
        return "p < 0.01"
    if value < 0.05:
        return "p < 0.05"
    return "p < 0.10"


def build_workbook(estimates: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Education Differences"
    sheet.sheet_view.showGridLines = False

    headers = [
        "Exposure",
        "Estimand",
        "Effect Unit",
        "Effect (pp)",
        "95% CI",
        "p threshold",
        "Family BH q",
        "Global BH q",
        "Standardized Effect (pp)",
        "Within-FE SD",
        "N",
        "Group Coverage",
        "Evidence Judgment",
    ]
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    title = sheet.cell(1, 1, "Education Differences in Cumulative Humid-Heat Associations")
    set_cell_style(title, fill=NAVY, color=WHITE, bold=True, size=15, align="left")
    sheet.row_dimensions[1].height = 31
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(2, column, header)
        set_cell_style(cell, fill=PALE_TEAL, bold=True, size=8, wrap=True)
    sheet.row_dimensions[2].height = 42

    thin = Side(style="thin", color=LIGHT_BORDER)
    group_top = Side(style="medium", color=MEDIUM_BORDER)
    for offset, row in enumerate(estimates.itertuples(index=False), start=0):
        row_number = 3 + offset
        values = [
            row.exposure_display,
            row.row_label,
            row.effect_unit,
            float(row.raw_effect_pp),
            format_ci(float(row.raw_ci_lower_pp), float(row.raw_ci_upper_pp)),
            format_p(float(row.p_value)),
            format_q(float(row.exploratory_family_bh_q)),
            format_q(float(row.global_bh_q)),
            float(row.standardized_effect_pp),
            float(row.within_fe_sd),
            int(row.n_obs),
            row.group_coverage,
            row.evidence_judgment,
        ]
        is_interaction = row.row_type == "interaction"
        starts_group = offset == 3
        fill = PALE_GOLD if is_interaction else (PALE_BLUE if offset % 2 else WHITE)
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row_number, column, value)
            set_cell_style(
                cell,
                fill=fill,
                bold=is_interaction and column in [2, 13],
                size=8,
                align="left" if column in [1, 2, 3, 12, 13] else "right",
                wrap=column in [1, 2, 3, 12, 13],
            )
            cell.border = Border(top=group_top if starts_group else Side(), bottom=thin)
        for column in [4, 9, 10]:
            sheet.cell(row_number, column).number_format = "0.000"
        sheet.cell(row_number, 11).number_format = "#,##0"
        sheet.row_dimensions[row_number].height = 34

    last_data_row = 2 + len(estimates)
    notes = [
        "Notes: Higher education means lower-secondary education or above; low education means none, preschool, or primary. Effects are percentage-point differences in working in the past week.",
        "Each exposure is estimated in one interaction model. The higher-education slope is the reference coefficient; the low-education slope is the reference plus interaction; the difference is low minus higher.",
        "Models include age, age squared, female, rural, monthly rainfall, District Calendar Month and Survey Year fixed effects, CSES analysis weights, and District-clustered standard errors.",
        "BH q-values apply only to registered interaction tests from the preserved exploratory universe. Both interactions are adverse, but neither survives the global 10% false-discovery threshold.",
        "The interaction is an associational vulnerability contrast, not the causal effect of schooling. Database access: read-only SELECT from mda.public; database writes: none. Generated "
        + datetime.now().astimezone().isoformat(timespec="seconds"),
    ]
    notes_start = last_data_row + 2
    for offset, note in enumerate(notes):
        row_number = notes_start + offset
        sheet.merge_cells(
            start_row=row_number, start_column=1, end_row=row_number, end_column=len(headers)
        )
        cell = sheet.cell(row_number, 1, note)
        set_cell_style(cell, color=MUTED, size=8, align="left", wrap=True)
        sheet.row_dimensions[row_number].height = 25

    widths = [28, 27, 25, 12, 19, 11, 13, 13, 20, 14, 12, 29, 35]
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A3"
    sheet.auto_filter.ref = f"A2:M{last_data_row}"
    sheet.print_area = f"A1:M{notes_start + len(notes) - 1}"
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 1
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_margins.left = 0.18
    sheet.page_margins.right = 0.18
    sheet.page_margins.top = 0.22
    sheet.page_margins.bottom = 0.22

    audit = workbook.create_sheet("Estimate Audit")
    audit.sheet_view.showGridLines = False
    for column, header in enumerate(estimates.columns, start=1):
        display_header = "p_threshold" if header == "p_value" else header
        cell = audit.cell(1, column, display_header)
        set_cell_style(cell, fill=NAVY, color=WHITE, bold=True, size=8, wrap=True)
    for row_number, row in enumerate(estimates.itertuples(index=False, name=None), start=2):
        for column, value in enumerate(row, start=1):
            if estimates.columns[column - 1] == "p_value":
                value = format_p(float(value))
            if pd.isna(value):
                value = None
            cell = audit.cell(row_number, column, value)
            set_cell_style(
                cell,
                size=8,
                align="left" if isinstance(value, str) else "right",
                wrap=isinstance(value, str),
            )
        audit.row_dimensions[row_number].height = 26
    audit.freeze_panes = "A2"
    audit.auto_filter.ref = audit.dimensions
    audit.print_area = audit.dimensions
    audit.page_setup.orientation = "landscape"
    audit.page_setup.paperSize = audit.PAPERSIZE_A3
    audit.page_setup.fitToWidth = 1
    audit.page_setup.fitToHeight = 0
    audit.sheet_properties.pageSetUpPr.fitToPage = True
    for column in range(1, audit.max_column + 1):
        audit.column_dimensions[get_column_letter(column)].width = 18

    metadata = workbook.create_sheet("Metadata")
    metadata.sheet_view.showGridLines = False
    metadata_rows = [
        ("Field", "Value"),
        ("Table", "Education Differences in Cumulative Humid-Heat Associations"),
        ("Outcome", "Worked in the past week"),
        ("Common sample", f"{int(estimates['n_obs'].iloc[0]):,} person-wave records"),
        ("Districts", f"{int(estimates['n_admin2'].iloc[0])}"),
        ("Higher-education observations", f"{int(estimates['higher_education_n'].iloc[0]):,}"),
        ("Low-education observations", f"{int(estimates['low_education_n'].iloc[0]):,}"),
        ("Weighted low-education share", f"{float(estimates['weighted_low_education_share'].iloc[0]):.3%}"),
        ("Fixed effects", "District Calendar Month and Survey Year"),
        ("Controls", "Age, age squared, female, rural, and monthly rainfall"),
        ("Weights", "CSES analysis weights"),
        ("Inference", "District-clustered, debiased standard errors"),
        ("Database session", "PostgreSQL default_transaction_read_only=on"),
        ("Database writes", "None"),
    ]
    for row_number, values in enumerate(metadata_rows, start=1):
        for column, value in enumerate(values, start=1):
            cell = metadata.cell(row_number, column, value)
            set_cell_style(
                cell,
                fill=NAVY if row_number == 1 else None,
                color=WHITE if row_number == 1 else TEXT,
                bold=row_number == 1 or column == 1,
                align="left",
                wrap=True,
            )
    metadata.column_dimensions["A"].width = 30
    metadata.column_dimensions["B"].width = 72
    metadata.freeze_panes = "A2"
    metadata.auto_filter.ref = metadata.dimensions
    metadata.print_area = metadata.dimensions
    workbook.save(output)


def validate_outputs(estimates: pd.DataFrame, workbook_path: Path, png_path: Path) -> None:
    if len(estimates) != 6:
        raise AssertionError(f"Expected six education estimates, observed {len(estimates)}")
    if estimates["n_obs"].nunique() != 1 or estimates["n_admin2"].nunique() != 1:
        raise AssertionError("Education estimates do not share one common sample")
    interactions = estimates.loc[estimates["row_type"].eq("interaction")]
    if len(interactions) != 2:
        raise AssertionError("Expected two education interaction rows")
    if not (interactions["raw_effect_pp"] < 0).all():
        raise AssertionError("Both frozen education interactions should be adverse")
    if interactions[["exploratory_family_bh_q", "global_bh_q"]].isna().any().any():
        raise AssertionError("Interaction multiplicity values are missing")
    non_interactions = estimates.loc[~estimates["row_type"].eq("interaction")]
    if non_interactions[["exploratory_family_bh_q", "global_bh_q"]].notna().any().any():
        raise AssertionError("Derived group slopes must not receive registered q-values")
    workbook = load_workbook(workbook_path, data_only=False, read_only=False)
    if workbook.sheetnames != ["Education Differences", "Estimate Audit", "Metadata"]:
        raise AssertionError(f"Unexpected workbook sheets: {workbook.sheetnames}")
    sheet = workbook["Education Differences"]
    if sheet["A1"].value != "Education Differences in Cumulative Humid-Heat Associations":
        raise AssertionError("Workbook title is missing")
    if sheet["A2"].value != "Exposure" or sheet.freeze_panes != "A3":
        raise AssertionError("Title must be followed immediately by column headers")
    if any(item.print_area is None for item in workbook.worksheets):
        raise AssertionError("Every worksheet must have a print area")
    workbook.close()
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise AssertionError("PNG review copy is missing")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    multiplicity_audit = resolve_path(args.multiplicity_audit)
    adjustments = load_interaction_adjustments(multiplicity_audit)
    with psycopg.connect(**connection_args(args)) as connection:
        frame = prepare_frame(load_data(connection, args.schema))
    estimates = estimate_models(frame)
    estimates = enrich_estimates(frame, estimates, adjustments)
    build_workbook(estimates, output)
    render_review_png(output, review_output)
    validate_outputs(estimates, output, review_output)
    print(f"table={output}")
    print(f"review={review_output}")
    print(
        f"rows={len(estimates)} n={int(estimates['n_obs'].iloc[0])} "
        f"districts={int(estimates['n_admin2'].iloc[0])}"
    )
    for row in estimates.loc[estimates["row_type"].eq("interaction")].itertuples(index=False):
        print(
            f"interaction={row.exposure_family} effect_pp={row.raw_effect_pp:.4f} "
            f"p={row.p_value:.4f} family_q={row.exploratory_family_bh_q:.4f} "
            f"global_q={row.global_bh_q:.4f}"
        )


if __name__ == "__main__":
    main()
