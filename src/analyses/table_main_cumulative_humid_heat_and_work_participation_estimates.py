#!/usr/bin/env python3
"""Generate the main cumulative humid-heat work-participation table.

Plan: Report threshold and continuous two-month cumulative humid-heat
estimates across four nested control sets on one common analytical sample.

Framework: AnaSOP Sections 5-7 define survey-weighted absorbed models with
District Calendar Month and Survey Year fixed effects and District-clustered
standard errors. The complete-control specification is the frozen primary
model. Its family and global BH values are read from the persisted exploratory
audit and checked against the newly estimated coefficient and p-value. All
person-level analytical inputs are queried from PostgreSQL using a forced
read-only session; this script performs no database writes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from audit_exposure_definition_comparison import residualize_exposure
from run_heat_labor_credibility_audit import fit_hdfe
from run_heat_labor_extensions import connection_args, load_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "data/results/tables"
    / "Table_main_cumulative_humid_heat_and_work_participation_estimates.xlsx"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Table_main_cumulative_humid_heat_and_work_participation_estimates.png"
)
DEFAULT_AUDIT = ROOT / "data/exp/storyline-search/target_tests.csv"

NAVY = "17324D"
TEAL = "3C7F7B"
PALE_TEAL = "DCEBEA"
PALE_BLUE = "EEF4F7"
PALE_GOLD = "FFF2CC"
WHITE = "FFFFFF"
TEXT = "23313D"
MUTED = "5C6B73"
LIGHT_BORDER = "B8C7CE"
MEDIUM_BORDER = "6F8791"

EXPOSURES = {
    "wb26_current_lag_total_5days": {
        "label": "Extreme humid-heat days (WBmax >= 26 C)",
        "short_label": "WB26 threshold days",
        "unit": "5 additional days across 2 months",
    },
    "wbmean_current_lag_average_1c": {
        "label": "Mean daily maximum wet-bulb temperature",
        "short_label": "Mean maximum wet-bulb temperature",
        "unit": "1 C higher 2-month average",
    },
}

SPECIFICATIONS = [
    ("fixed_effects_only", "Fixed effects only", []),
    ("rainfall_only", "+ Monthly rainfall", ["precipitation_100mm"]),
    (
        "demographics_only",
        "+ Demographic controls",
        ["age", "age_squared", "female", "rural"],
    ),
    (
        "frozen_primary",
        "Frozen primary: full controls",
        ["age", "age_squared", "female", "rural", "precipitation_100mm"],
    ),
]


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
    parser.add_argument("--multiplicity-audit", type=Path, default=DEFAULT_AUDIT)
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
    required = [
        "worked_past_week",
        *EXPOSURES,
        "age",
        "age_squared",
        "female",
        "rural",
        "precipitation_100mm",
        "analysis_weight",
        "admin2_calendar_month",
        "survey_year",
        "admin2_code",
        "admin1_code",
        "survey_month",
    ]
    return frame[required].notna().all(axis=1) & frame["analysis_weight"].gt(0)


def within_fe_support(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    terms = list(EXPOSURES)
    columns = [
        "survey_year",
        "survey_month",
        "admin2_code",
        "admin2_calendar_month",
        *terms,
    ]
    selected = frame.loc[mask, columns].copy()
    key = ["survey_year", "survey_month", "admin2_code"]
    for term in terms:
        unique_counts = selected.groupby(key, observed=True)[term].nunique(dropna=False)
        if int(unique_counts.max()) != 1:
            raise AssertionError(f"Exposure varies within district-year-month: {term}")
    cells = selected.drop_duplicates(key).reset_index(drop=True)
    support = {
        term: float(residualize_exposure(cells, term).dropna().std(ddof=0))
        for term in terms
    }
    return pd.Series(support, name="within_fe_sd")


def load_primary_adjustments(path: Path) -> pd.DataFrame:
    audit = pd.read_csv(path)
    selected = audit.loc[
        audit["outcome"].eq("worked_past_week")
        & audit["term"].isin(EXPOSURES)
        & audit["is_target_test"].eq(True)
        & audit["specification"].isin(
            ["timing_work_wb26_cumulative", "timing_work_wbmean_cumulative"]
        )
    ].copy()
    if len(selected) != 2 or selected["term"].nunique() != 2:
        raise AssertionError("Expected exactly two frozen cumulative target tests")
    return selected.set_index("term")


def estimate_models(
    frame: pd.DataFrame,
    mask: pd.Series,
    support: pd.Series,
    adjustments: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term, exposure in EXPOSURES.items():
        for specification, specification_label, controls in SPECIFICATIONS:
            results, _model = fit_hdfe(
                frame,
                f"main_{term}_{specification}",
                "worked_past_week",
                [term],
                controls,
                sample_label="Common complete-case sample",
                mask=mask,
            )
            result = results[0]
            scale = 100.0
            within_sd = float(support[term])
            is_primary = specification == "frozen_primary"
            family_q = np.nan
            global_q = np.nan
            if is_primary:
                frozen = adjustments.loc[term]
                if not np.isclose(
                    float(result["coefficient"]), float(frozen["coefficient"]), atol=1e-10
                ):
                    raise AssertionError(f"Frozen coefficient mismatch for {term}")
                if not np.isclose(
                    float(result["p_value"]), float(frozen["p_value"]), atol=1e-10
                ):
                    raise AssertionError(f"Frozen p-value mismatch for {term}")
                family_q = float(frozen["family_bh_q_value"])
                global_q = float(frozen["global_bh_q_value"])

            estimate = scale * float(result["coefficient"])
            lower = scale * float(result["ci_lower_95"])
            upper = scale * float(result["ci_upper_95"])
            rows.append(
                {
                    "exposure_term": term,
                    "exposure": exposure["label"],
                    "exposure_short": exposure["short_label"],
                    "specification": specification,
                    "specification_label": specification_label,
                    "effect_unit": exposure["unit"],
                    "effect_pp": estimate,
                    "ci_lower_pp": lower,
                    "ci_upper_pp": upper,
                    "p_value": float(result["p_value"]),
                    "family_bh_q_value": family_q,
                    "global_bh_q_value": global_q,
                    "within_fe_sd": within_sd,
                    "standardized_effect_pp": estimate * within_sd,
                    "standardized_ci_lower_pp": lower * within_sd,
                    "standardized_ci_upper_pp": upper * within_sd,
                    "n_obs": int(result["n_obs"]),
                    "n_admin2": int(result["n_admin2"]),
                    "controls": ", ".join(controls) if controls else "None",
                    "fixed_effects": str(result["fixed_effects"]),
                    "weights": "CSES analysis weights",
                    "cluster_level": "District",
                    "weighted_outcome_mean": float(result["weighted_outcome_mean"]),
                }
            )
    estimates = pd.DataFrame(rows)
    expected_n = estimates["n_obs"].iloc[0]
    if estimates["n_obs"].nunique() != 1 or expected_n != int(mask.sum()):
        raise AssertionError("Nested specifications do not share one common sample")
    return estimates


def set_cell_style(
    cell: object,
    *,
    fill: str | None = None,
    color: str = TEXT,
    bold: bool = False,
    size: int = 9,
    align: str = "center",
    wrap: bool = False,
) -> None:
    if fill:
        cell.fill = PatternFill("solid", fgColor=fill)
    cell.font = Font(name="Aptos", size=size, bold=bold, color=color)
    cell.alignment = Alignment(horizontal=align, vertical="center", wrap_text=wrap)


def formatted_ci(lower: float, upper: float) -> str:
    return f"[{lower:.3f}, {upper:.3f}]"


def formatted_q(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def formatted_p(value: float) -> str:
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
    sheet.title = "Main Estimates"
    sheet.sheet_view.showGridLines = False

    headers = [
        "Exposure",
        "Specification",
        "Effect Unit",
        "Effect (pp)",
        "95% CI",
        "p threshold",
        "Family BH q",
        "Global BH q",
        "Standardized Effect (pp)",
        "Within-FE SD",
        "N",
        "Districts",
    ]
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    title = sheet.cell(1, 1, "Main Cumulative Humid-Heat and Work Participation Estimates")
    set_cell_style(title, fill=NAVY, color=WHITE, bold=True, size=15, align="left")
    sheet.row_dimensions[1].height = 31
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(2, column, header)
        set_cell_style(cell, fill=PALE_TEAL, bold=True, size=8, wrap=True)
    sheet.row_dimensions[2].height = 42

    thin = Side(style="thin", color=LIGHT_BORDER)
    group_top = Side(style="medium", color=MEDIUM_BORDER)
    for offset, row in enumerate(estimates.itertuples(index=False), start=0):
        target_row = 3 + offset
        values = [
            row.exposure_short,
            row.specification_label,
            row.effect_unit,
            float(row.effect_pp),
            formatted_ci(float(row.ci_lower_pp), float(row.ci_upper_pp)),
            formatted_p(float(row.p_value)),
            formatted_q(float(row.family_bh_q_value)),
            formatted_q(float(row.global_bh_q_value)),
            float(row.standardized_effect_pp),
            float(row.within_fe_sd),
            int(row.n_obs),
            int(row.n_admin2),
        ]
        primary = row.specification == "frozen_primary"
        fill = PALE_GOLD if primary else (PALE_BLUE if offset % 2 else WHITE)
        starts_group = offset == 4
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(target_row, column, value)
            set_cell_style(
                cell,
                fill=fill,
                bold=primary and column == 2,
                size=8,
                align="left" if column <= 3 else "right",
                wrap=column <= 3,
            )
            cell.border = Border(top=group_top if starts_group else Side(), bottom=thin)
        for column in [4, 9, 10]:
            sheet.cell(target_row, column).number_format = "0.000"
        for column in [11, 12]:
            sheet.cell(target_row, column).number_format = "#,##0"
        sheet.row_dimensions[target_row].height = 31

    last_data_row = 2 + len(estimates)
    notes = [
        "Notes: Outcome is the percentage-point probability of working in the past week. All eight estimates use the same complete-case person-wave sample.",
        "All models include District Calendar Month and Survey Year fixed effects, CSES analysis weights, and District-clustered standard errors. Demographic controls are age, age squared, female, and rural.",
        "The standardized effect multiplies the raw effect by one fixed-effect-residual standard deviation of the exposure, calculated across unique district-year-month cells in the common sample.",
        "BH q-values are shown only for the frozen primary specifications. Neither primary estimate survives the global exploratory-universe adjustment (both global q = 0.210).",
        "Database access: read-only SELECT from mda.public; database writes: none. Generated "
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
        sheet.row_dimensions[row_number].height = 24

    widths = [25, 25, 27, 12, 19, 11, 13, 13, 20, 14, 12, 11]
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A3"
    sheet.auto_filter.ref = f"A2:L{last_data_row}"
    sheet.print_area = f"A1:L{notes_start + len(notes) - 1}"
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
                value = formatted_p(float(value))
            if pd.isna(value):
                value = None
            cell = audit.cell(row_number, column, value)
            set_cell_style(cell, size=8, align="left" if column <= 6 else "right", wrap=column <= 6)
        audit.row_dimensions[row_number].height = 24
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
        ("Table", "Main Cumulative Humid-Heat and Work Participation Estimates"),
        ("Outcome", "Worked in the past week"),
        ("Common sample", f"{int(estimates['n_obs'].iloc[0]):,} person-wave records"),
        ("Clusters", f"{int(estimates['n_admin2'].iloc[0])} Districts"),
        ("Fixed effects", "District Calendar Month and Survey Year"),
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
    metadata.column_dimensions["A"].width = 25
    metadata.column_dimensions["B"].width = 70
    metadata.freeze_panes = "A2"
    metadata.auto_filter.ref = metadata.dimensions
    metadata.print_area = metadata.dimensions

    workbook.save(output)


def render_review_png(workbook_path: Path, png_path: Path) -> None:
    soffice = Path("/opt/homebrew/bin/soffice")
    if not soffice.exists():
        located = shutil.which("soffice")
        if not located:
            raise RuntimeError("LibreOffice/soffice is required for workbook rendering")
        soffice = Path(located)
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required for PNG rendering")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mj02b-main-table-review-") as temporary:
        temporary_path = Path(temporary)
        profile_uri = (temporary_path / "lo-profile").as_uri()
        subprocess.run(
            [
                str(soffice),
                f"-env:UserInstallation={profile_uri}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(temporary_path),
                str(workbook_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pdf_path = temporary_path / f"{workbook_path.stem}.pdf"
        if not pdf_path.exists():
            raise RuntimeError("LibreOffice did not create the expected PDF")
        prefix = png_path.with_suffix("")
        subprocess.run(
            ["pdftoppm", "-png", "-r", "180", "-f", "1", "-singlefile", str(pdf_path), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise RuntimeError("PNG review render is missing or empty")


def validate_outputs(estimates: pd.DataFrame, workbook_path: Path, png_path: Path) -> None:
    if len(estimates) != 8:
        raise AssertionError(f"Expected 8 estimate rows, observed {len(estimates)}")
    if estimates["n_obs"].nunique() != 1 or estimates["n_admin2"].nunique() != 1:
        raise AssertionError("All rows must use one common sample and cluster set")
    primary = estimates.loc[estimates["specification"].eq("frozen_primary")]
    if len(primary) != 2 or primary["global_bh_q_value"].isna().any():
        raise AssertionError("Frozen primary rows are missing adjusted p-values")
    non_primary = estimates.loc[~estimates["specification"].eq("frozen_primary")]
    if non_primary[["family_bh_q_value", "global_bh_q_value"]].notna().any().any():
        raise AssertionError("Adjusted p-values must not be assigned to nested descriptive models")
    workbook = load_workbook(workbook_path, data_only=False, read_only=False)
    if workbook.sheetnames != ["Main Estimates", "Estimate Audit", "Metadata"]:
        raise AssertionError(f"Unexpected workbook sheets: {workbook.sheetnames}")
    sheet = workbook["Main Estimates"]
    if sheet["A1"].value != "Main Cumulative Humid-Heat and Work Participation Estimates":
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
    adjustments = load_primary_adjustments(multiplicity_audit)
    with psycopg.connect(**connection_args(args)) as connection:
        frame = prepare_frame(load_data(connection, args.schema))
    mask = common_sample_mask(frame)
    support = within_fe_support(frame, mask)
    estimates = estimate_models(frame, mask, support, adjustments)
    build_workbook(estimates, output)
    render_review_png(output, review_output)
    validate_outputs(estimates, output, review_output)
    print(f"table={output}")
    print(f"review={review_output}")
    print(f"rows={len(estimates)} n={int(estimates['n_obs'].iloc[0])}")
    for row in estimates.loc[estimates["specification"].eq("frozen_primary")].itertuples(index=False):
        print(
            f"primary={row.exposure_term} effect_pp={row.effect_pp:.4f} "
            f"p={row.p_value:.4f} family_q={row.family_bh_q_value:.4f} "
            f"global_q={row.global_bh_q_value:.4f}"
        )


if __name__ == "__main__":
    main()
