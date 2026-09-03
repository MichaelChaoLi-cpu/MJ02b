#!/usr/bin/env python3
"""Generate district priority rankings and decision robustness.

Plan: Report all 197 Districts with both humid-heat hazards, working-age scale,
low-education share, primary combined priority score and rank, capacity-specific
Priority Inclusion Frequency, and Data Support Grade.

Framework: AnaSOP Section 5 keeps Data Support Grade outside the score. Section
6 defines the equal-weight primary score, 20/40/60 District capacities, the
420-scenario structural robustness universe, and a separate 1,000-replicate
Survey Wave block bootstrap that fixes threshold burden and equal A5 weights.
Section 7 Step 8 requires stable, borderline, and evidence-limited Districts to
remain auditable. Inputs are reconstructed from PostgreSQL under
default_transaction_read_only=on; this script performs no database writes.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.pagebreak import Break
from psycopg import sql

from build_impact_assessment_indicators import (
    WEIGHT_SCENARIOS,
    build_district_indicators,
    connection_args,
    fetch_frame,
    load_inputs,
    percentile_rank,
    verify_read_only,
)
from table_proposed_occupational_humid_heat_protection_alternatives_and_assessment_criteria import (
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
TITLE = "District Priority Rankings and Decision Robustness"
DEFAULT_OUTPUT = (
    ROOT
    / "data/results/tables"
    / "Table_district_priority_rankings_and_decision_robustness.xlsx"
)
DEFAULT_REVIEW_DIR = ROOT / "data/exp/figure-table-review"
REVIEW_STEM = "Table_district_priority_rankings_and_decision_robustness"
CAPACITIES = [20, 40, 60]
WORKER_SHARE_TARGETS = [0.10, 0.20, 0.30]
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20260824
PAGE_ROWS = 40
PALE_RED = "FDE9E7"

HEADERS = [
    "District",
    "Threshold Heat-Burden Share",
    "Continuous Heat-Burden Share",
    "Working-Age Scale",
    "Vulnerable-Worker Scale",
    "Combined Priority Score",
    "Primary Rank",
    "Inclusion Frequency: 20",
    "Inclusion Frequency: 40",
    "Inclusion Frequency: 60",
    "Data Support Grade",
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
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def load_coastal_flags(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    query = sql.SQL(
        """
        SELECT admin2_code,
               MAX(
                   CASE
                       WHEN LOWER(COALESCE(temperature_location_fallback_used::text, ''))
                            IN ('1', 'true', 't', 'yes')
                       THEN 1 ELSE 0
                   END
               ) AS coastal_climate_fallback
        FROM {}.{}
        WHERE admin2_code IS NOT NULL
        GROUP BY admin2_code
        ORDER BY admin2_code
        """
    ).format(
        sql.Identifier(schema), sql.Identifier("final_HEAT_LABOR_ANALYTIC")
    )
    flags = fetch_frame(connection, query)
    flags["coastal_climate_fallback"] = pd.to_numeric(
        flags["coastal_climate_fallback"], errors="raise"
    ).astype(int)
    return flags


def ranked_order(
    district: pd.DataFrame,
    burden_column: str,
    weights: tuple[float, float, float],
    eligible_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    eligible = district.loc[eligible_mask].copy()
    burden_rank = percentile_rank(eligible[burden_column])
    scale_rank = percentile_rank(eligible["weighted_working_age_scale"])
    vulnerable_rank = percentile_rank(eligible["district_vulnerable_worker_scale"])
    score = (
        weights[0] * burden_rank
        + weights[1] * scale_rank
        + weights[2] * vulnerable_rank
    )
    ordered = (
        eligible.assign(_score=score)
        .sort_values(["_score", "admin2_code"], ascending=[False, True], kind="mergesort")
    )
    return ordered, score


def ranked_selection(
    district: pd.DataFrame,
    burden_column: str,
    weights: tuple[float, float, float],
    eligible_mask: pd.Series,
    capacity: int,
) -> tuple[pd.Index, pd.Series]:
    ordered, score = ranked_order(district, burden_column, weights, eligible_mask)
    ordered = ordered.head(capacity)
    return ordered.index, score


def ranked_worker_share_selection(
    district: pd.DataFrame,
    burden_column: str,
    weights: tuple[float, float, float],
    eligible_mask: pd.Series,
    target: float,
) -> pd.Index:
    ordered, _ = ranked_order(district, burden_column, weights, eligible_mask)
    cumulative = ordered["weighted_working_age_scale"].cumsum()
    reached = np.flatnonzero(cumulative.to_numpy() >= target)
    if len(reached) == 0:
        return ordered.index
    return ordered.iloc[: int(reached[0]) + 1].index


def deterministic_robustness(
    geography: pd.DataFrame,
    climate: pd.DataFrame,
    people: pd.DataFrame,
    coastal_flags: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    district_codes = geography["admin2_code"].astype(str).tolist()
    counts = pd.DataFrame(
        0,
        index=pd.Index(district_codes, name="admin2_code"),
        columns=[f"selected_{capacity}" for capacity in CAPACITIES],
        dtype=int,
    )
    denominators = pd.Series(0, index=counts.index, dtype=int, name="eligible_scenarios")
    scenario_rows: list[dict[str, object]] = []
    waves = sorted(people["survey_wave"].dropna().astype(str).unique())
    sample_variants: list[tuple[str, str | None]] = [("full", None)] + [
        (f"leave_out_{wave}", wave) for wave in waves
    ]
    flags = coastal_flags.set_index("admin2_code")["coastal_climate_fallback"]
    baseline_district: pd.DataFrame | None = None

    for sample_variant, omitted_wave in sample_variants:
        if omitted_wave is None:
            climate_part = climate
            people_part = people
        else:
            climate_part = climate.loc[
                climate["survey_wave"].astype(str).ne(omitted_wave)
            ]
            people_part = people.loc[
                people["survey_wave"].astype(str).ne(omitted_wave)
            ]
        district, _ = build_district_indicators(
            geography, climate_part, people_part
        )
        district["admin2_code"] = district["admin2_code"].astype(str)
        district["coastal_climate_fallback"] = (
            district["admin2_code"].map(flags).fillna(0).astype(int)
        )
        if omitted_wave is None:
            baseline_district = district.copy()

        eligibility_rules = {
            "all_districts": pd.Series(True, index=district.index),
            "exclude_limited_support": district["data_support_grade"].ne("Limited"),
            "exclude_coastal_fallback": district["coastal_climate_fallback"].eq(0),
        }
        for exposure_definition, burden_column in [
            ("threshold", "district_threshold_humid_heat_burden"),
            ("continuous", "district_continuous_humid_heat_burden"),
        ]:
            for weight_name, weights in WEIGHT_SCENARIOS.items():
                for eligibility_name, eligible_mask in eligibility_rules.items():
                    eligible_codes = district.loc[eligible_mask, "admin2_code"]
                    denominators.loc[eligible_codes] += 1
                    scenario: dict[str, object] = {
                        "sample_variant": sample_variant,
                        "omitted_wave": omitted_wave or "none",
                        "exposure_definition": exposure_definition,
                        "weight_scenario": weight_name,
                        "eligibility_rule": eligibility_name,
                        "eligible_districts": int(eligible_mask.sum()),
                    }
                    for capacity in CAPACITIES:
                        selected_index, _ = ranked_selection(
                            district, burden_column, weights, eligible_mask, capacity
                        )
                        selected_codes = district.loc[selected_index, "admin2_code"]
                        counts.loc[selected_codes, f"selected_{capacity}"] += 1
                        scenario[f"selected_codes_{capacity}"] = "|".join(selected_codes)
                    for target in WORKER_SHARE_TARGETS:
                        selected_index = ranked_worker_share_selection(
                            district, burden_column, weights, eligible_mask, target
                        )
                        selected_codes = district.loc[selected_index, "admin2_code"]
                        suffix = int(round(target * 100))
                        scenario[f"worker_share_selected_count_{suffix}"] = len(selected_codes)
                        scenario[f"worker_share_selected_codes_{suffix}"] = "|".join(selected_codes)
                    scenario_rows.append(scenario)

    if baseline_district is None:
        raise AssertionError("The full-sample district indicators were not constructed")
    frequencies = counts.div(denominators, axis=0)
    frequencies.columns = [
        f"inclusion_frequency_{capacity}" for capacity in CAPACITIES
    ]
    frequencies["eligible_scenarios"] = denominators
    frequencies = frequencies.reset_index()
    scenarios = pd.DataFrame(scenario_rows)
    expected_scenarios = 10 * 2 * len(WEIGHT_SCENARIOS) * 3
    if len(scenarios) != expected_scenarios:
        raise AssertionError(
            f"Expected {expected_scenarios} deterministic scenarios, observed {len(scenarios)}"
        )
    return baseline_district, frequencies, scenarios


def wave_component_matrices(
    climate: pd.DataFrame, people: pd.DataFrame, district_codes: list[str]
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    climate_frame = climate.copy()
    for column in [
        "days_wbmax_ge_26c",
        "lag_days_wbmax_ge_26c",
        "wet_bulb_max_mean_c",
        "lag_wet_bulb_max_mean_c",
    ]:
        climate_frame[column] = pd.to_numeric(climate_frame[column], errors="coerce")
    climate_frame["cumulative_wb26_days"] = (
        climate_frame["days_wbmax_ge_26c"]
        + climate_frame["lag_days_wbmax_ge_26c"]
    )
    climate_frame["two_month_wbmax_mean_c"] = (
        climate_frame["wet_bulb_max_mean_c"]
        + climate_frame["lag_wet_bulb_max_mean_c"]
    ) / 2.0
    climate_frame["two_month_wbmax_excess_above_26c"] = (
        climate_frame["two_month_wbmax_mean_c"] - 26.0
    ).clip(lower=0.0)
    wave_heat = (
        climate_frame.groupby(["admin2_code", "survey_wave"], observed=True)
        .agg(
            threshold=("cumulative_wb26_days", "mean"),
            continuous=("two_month_wbmax_excess_above_26c", "mean"),
        )
        .reset_index()
    )

    people_frame = people.copy()
    people_frame["analysis_weight"] = pd.to_numeric(
        people_frame["analysis_weight"], errors="coerce"
    )
    wave_scale = (
        people_frame.groupby(["survey_wave", "admin2_code"], observed=True)
        .agg(district_weight=("analysis_weight", "sum"))
        .reset_index()
    )
    wave_scale["wave_total"] = wave_scale.groupby("survey_wave", observed=True)[
        "district_weight"
    ].transform("sum")
    wave_scale["working_age"] = wave_scale["district_weight"] / wave_scale["wave_total"]

    education = people_frame.loc[
        pd.to_numeric(
            people_frame["education_level_harmonized"], errors="coerce"
        ).between(0, 7)
    ].copy()
    education["low_education"] = pd.to_numeric(
        education["education_level_harmonized"], errors="coerce"
    ).le(2)
    education["weighted_low"] = (
        education["analysis_weight"] * education["low_education"].astype(float)
    )
    wave_education = (
        education.groupby(["survey_wave", "admin2_code"], observed=True)
        .agg(
            education_weight=("analysis_weight", "sum"),
            weighted_low=("weighted_low", "sum"),
        )
        .reset_index()
    )
    wave_education["low_education"] = (
        wave_education["weighted_low"] / wave_education["education_weight"]
    )

    waves = sorted(people_frame["survey_wave"].dropna().astype(str).unique())

    def pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
        matrix = frame.assign(
            admin2_code=frame["admin2_code"].astype(str),
            survey_wave=frame["survey_wave"].astype(str),
        ).pivot(index="admin2_code", columns="survey_wave", values=value)
        return matrix.reindex(index=district_codes, columns=waves)

    matrices = {
        "threshold": pivot(wave_heat, "threshold"),
        "continuous": pivot(wave_heat, "continuous"),
        "working_age": pivot(wave_scale, "working_age"),
        "low_education": pivot(wave_education, "low_education"),
    }
    return waves, matrices


def bootstrap_robustness(
    climate: pd.DataFrame,
    people: pd.DataFrame,
    district_codes: list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    waves, matrices = wave_component_matrices(climate, people, district_codes)
    rng = np.random.default_rng(seed)
    counts = {
        capacity: pd.Series(0, index=district_codes, dtype=int)
        for capacity in CAPACITIES
    }
    eligible = pd.Series(0, index=district_codes, dtype=int)
    equal_weights = WEIGHT_SCENARIOS["equal_weight"]

    for _ in range(replicates):
        sampled_positions = rng.integers(0, len(waves), size=len(waves))
        heat_intensity = matrices["threshold"].iloc[:, sampled_positions].mean(
            axis=1, skipna=True
        )
        working_age = matrices["working_age"].iloc[:, sampled_positions].mean(
            axis=1, skipna=True
        )
        working_age = working_age / working_age.sum(skipna=True)
        low_education = matrices["low_education"].iloc[:, sampled_positions].mean(
            axis=1, skipna=True
        )
        heat_burden = working_age * heat_intensity
        vulnerable_worker = working_age * low_education
        complete = heat_burden.notna() & working_age.notna() & vulnerable_worker.notna()
        eligible.loc[complete.index[complete]] += 1
        score = (
            equal_weights[0] * percentile_rank(heat_burden.loc[complete])
            + equal_weights[1] * percentile_rank(working_age.loc[complete])
            + equal_weights[2] * percentile_rank(vulnerable_worker.loc[complete])
        )
        ordered = (
            pd.DataFrame(
                {
                    "admin2_code": score.index,
                    "score": score.to_numpy(),
                }
            )
            .sort_values(
                ["score", "admin2_code"],
                ascending=[False, True],
                kind="mergesort",
            )
        )
        for capacity in CAPACITIES:
            selected_codes = ordered.head(capacity)["admin2_code"]
            counts[capacity].loc[selected_codes] += 1

    output = pd.DataFrame({"admin2_code": district_codes})
    output["bootstrap_burden_definition"] = "threshold"
    output["bootstrap_weight_rule"] = "equal_weight_a5"
    output["eligible_bootstrap_replicates"] = eligible.to_numpy()
    for capacity in CAPACITIES:
        output[f"bootstrap_inclusion_frequency_{capacity}"] = (
            counts[capacity] / eligible.replace(0, np.nan)
        ).to_numpy()
    return output


def build_ranking_table(
    district: pd.DataFrame,
    structural: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = district.merge(
        structural, on="admin2_code", how="left", validate="one_to_one"
    ).merge(bootstrap, on="admin2_code", how="left", validate="one_to_one")
    audit["combined_priority_score"] = (
        audit["threshold_burden_percentile"]
        + audit["working_age_percentile"]
        + audit["vulnerable_worker_percentile"]
    ) / 3.0
    audit = audit.sort_values(
        ["combined_priority_score", "admin2_code"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    audit["primary_rank"] = np.arange(1, len(audit) + 1)
    audit["district_label"] = (
        audit["admin2_name"].astype(str)
        + " ("
        + audit["admin1_name"].astype(str)
        + ")"
    )
    audit["threshold_heat_burden_share"] = (
        audit["district_threshold_humid_heat_burden"]
        / audit["district_threshold_humid_heat_burden"].sum()
    )
    audit["continuous_heat_burden_share"] = (
        audit["district_continuous_humid_heat_burden"]
        / audit["district_continuous_humid_heat_burden"].sum()
    )
    audit["vulnerable_worker_scale_share"] = (
        audit["district_vulnerable_worker_scale"]
        / audit["district_vulnerable_worker_scale"].sum()
    )

    display = pd.DataFrame(
        {
            "District": audit["district_label"],
            "Threshold Heat-Burden Share": audit["threshold_heat_burden_share"],
            "Continuous Heat-Burden Share": audit["continuous_heat_burden_share"],
            "Working-Age Scale": audit["weighted_working_age_scale"],
            "Vulnerable-Worker Scale": audit["vulnerable_worker_scale_share"],
            "Combined Priority Score": audit["combined_priority_score"],
            "Primary Rank": audit["primary_rank"],
            "Inclusion Frequency: 20": audit["inclusion_frequency_20"],
            "Inclusion Frequency: 40": audit["inclusion_frequency_40"],
            "Inclusion Frequency: 60": audit["inclusion_frequency_60"],
            "Data Support Grade": audit["data_support_grade"],
        },
        columns=HEADERS,
    )
    validate_ranking(display, audit)
    return display, audit


def validate_ranking(display: pd.DataFrame, audit: pd.DataFrame) -> None:
    if display.shape != (197, 11):
        raise AssertionError(f"Expected a 197 x 11 table, observed {display.shape}")
    if display["Primary Rank"].tolist() != list(range(1, 198)):
        raise AssertionError("Primary Rank must run from 1 through 197")
    if display.isna().any().any():
        raise AssertionError("The main district ranking table must not contain missing cells")
    frequency_columns = [f"Inclusion Frequency: {capacity}" for capacity in CAPACITIES]
    if not display[frequency_columns].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise AssertionError("Priority Inclusion Frequency must remain within [0, 1]")
    if audit["eligible_scenarios"].min() <= 0:
        raise AssertionError("Every District must be eligible in at least one scenario")
    if audit["eligible_bootstrap_replicates"].min() <= 0:
        raise AssertionError("Every District must be eligible in at least one bootstrap replicate")


def row_fill(row: pd.Series, row_number: int) -> str:
    if row["Data Support Grade"] == "Limited":
        return PALE_RED
    if row["Inclusion Frequency: 20"] >= 0.80:
        return PALE_GOLD
    if row["Inclusion Frequency: 40"] >= 0.80:
        return PALE_TEAL
    return PALE_BLUE if row_number % 2 == 0 else WHITE


def write_rankings_sheet(
    sheet: object,
    display: pd.DataFrame,
    *,
    include_notes: bool,
) -> None:
    sheet.sheet_view.showGridLines = False
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(HEADERS))
    title_cell = sheet.cell(1, 1, TITLE)
    set_cell_style(title_cell, fill=NAVY, color=WHITE, bold=True, size=15, align="left")
    sheet.row_dimensions[1].height = 31
    for column, header in enumerate(HEADERS, start=1):
        cell = sheet.cell(2, column, header)
        set_cell_style(cell, fill=PALE_TEAL, bold=True, size=8, wrap=True)
    sheet.row_dimensions[2].height = 44

    thin = Side(style="thin", color=LIGHT_BORDER)
    group_top = Side(style="medium", color=MEDIUM_BORDER)
    for offset, (_, row) in enumerate(display.iterrows()):
        row_number = 3 + offset
        fill = row_fill(row, row_number)
        for column, value in enumerate(row.tolist(), start=1):
            cell = sheet.cell(row_number, column, value)
            set_cell_style(
                cell,
                fill=fill,
                bold=column in {1, 7} and int(row["Primary Rank"]) <= 20,
                size=8,
                align="left" if column in {1, 11} else "right",
                wrap=True,
            )
            cell.border = Border(
                top=group_top if int(row["Primary Rank"]) in {1, 21, 41, 61} else Side(),
                bottom=thin,
                left=thin if column == 11 else Side(),
            )
        sheet.cell(row_number, 6).number_format = "0.000"
        for column in [2, 3, 4, 5, 8, 9, 10]:
            sheet.cell(row_number, column).number_format = "0.0%"
        sheet.row_dimensions[row_number].height = 23

    last_data_row = 2 + len(display)
    notes = []
    if include_notes:
        notes = [
            "Notes: Threshold burden first averages months within each District and Survey Wave and then averages observed Survey Waves equally. Primary Rank uses equal exact-percentile weights for threshold burden, working-age scale, and vulnerable-worker scale; Data Support Grade is excluded.",
            "Inclusion frequencies use 420 deterministic scenarios: two burden definitions, seven weight rules, the full plus nine leave-one-wave-out samples, and three eligibility rules. Each District's denominator includes only scenarios in which it is eligible.",
            "Gold indicates inclusion frequency of at least 0.80 at 20-District capacity with non-Limited support; teal indicates at least 0.80 at 40-District capacity; red flags Limited Data Support Grade.",
            "The Structural Scenario Audit also reports the selected District count and codes required to reach 10%, 20%, and 30% worker-share capacity targets.",
            "A separate workbook sheet reports 1,000 seeded Survey Wave block-bootstrap inclusion frequencies with threshold burden and equal-weight A5 fixed. Only Survey Wave composition varies; these are not design-based survey confidence intervals.",
            "All inputs were reconstructed through read-only PostgreSQL SELECT statements; database writes: none. Rankings support screening, not causal program-effect claims.",
        ]
        notes_start = last_data_row + 2
        for offset, note in enumerate(notes):
            row_number = notes_start + offset
            sheet.merge_cells(
                start_row=row_number,
                start_column=1,
                end_row=row_number,
                end_column=len(HEADERS),
            )
            cell = sheet.cell(row_number, 1, note)
            set_cell_style(cell, color=MUTED, size=8, align="left", wrap=True)
            sheet.row_dimensions[row_number].height = 25

    widths = [36, 16, 17, 17, 18, 19, 12, 19, 19, 19, 18]
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A3"
    sheet.auto_filter.ref = f"A2:K{last_data_row}"
    sheet.print_title_rows = "1:2"
    end_row = last_data_row + (len(notes) + 1 if notes else 0)
    sheet.print_area = f"A1:K{end_row}"
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_margins.left = 0.15
    sheet.page_margins.right = 0.15
    sheet.page_margins.top = 0.20
    sheet.page_margins.bottom = 0.20
    if len(display) > PAGE_ROWS:
        for row_id in range(2 + PAGE_ROWS, last_data_row, PAGE_ROWS):
            sheet.row_breaks.append(Break(id=row_id))


def build_workbook(
    display: pd.DataFrame,
    audit: pd.DataFrame,
    scenarios: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    rankings = workbook.active
    rankings.title = "District Rankings"
    write_rankings_sheet(rankings, display, include_notes=True)

    bootstrap_columns = [
        "admin2_code",
        "admin2_name",
        "admin1_name",
        "bootstrap_burden_definition",
        "bootstrap_weight_rule",
        "eligible_bootstrap_replicates",
        "bootstrap_inclusion_frequency_20",
        "bootstrap_inclusion_frequency_40",
        "bootstrap_inclusion_frequency_60",
    ]
    bootstrap_audit = audit[bootstrap_columns]
    bootstrap_sheet = workbook.create_sheet("Bootstrap Robustness")
    bootstrap_sheet.sheet_view.showGridLines = False
    for column, header in enumerate(bootstrap_audit.columns, start=1):
        cell = bootstrap_sheet.cell(1, column, header)
        set_cell_style(cell, fill=NAVY, color=WHITE, bold=True, size=8, wrap=True)
    for row_number, row in enumerate(
        bootstrap_audit.itertuples(index=False, name=None), start=2
    ):
        for column, value in enumerate(row, start=1):
            cell = bootstrap_sheet.cell(row_number, column, value)
            set_cell_style(
                cell, size=8, align="left" if column <= 5 else "right", wrap=True
            )
            if column >= 7:
                cell.number_format = "0.0%"
    bootstrap_sheet.freeze_panes = "A2"
    bootstrap_sheet.auto_filter.ref = bootstrap_sheet.dimensions
    bootstrap_sheet.print_area = bootstrap_sheet.dimensions
    bootstrap_sheet.page_setup.orientation = "landscape"
    bootstrap_sheet.page_setup.fitToWidth = 1
    bootstrap_sheet.page_setup.fitToHeight = 0
    bootstrap_sheet.sheet_properties.pageSetUpPr.fitToPage = True
    for column in range(1, bootstrap_sheet.max_column + 1):
        bootstrap_sheet.column_dimensions[get_column_letter(column)].width = 24

    scenario_sheet = workbook.create_sheet("Structural Scenario Audit")
    scenario_sheet.sheet_view.showGridLines = False
    for column, header in enumerate(scenarios.columns, start=1):
        cell = scenario_sheet.cell(1, column, header)
        set_cell_style(cell, fill=NAVY, color=WHITE, bold=True, size=8, wrap=True)
    for row_number, row in enumerate(scenarios.itertuples(index=False, name=None), start=2):
        for column, value in enumerate(row, start=1):
            cell = scenario_sheet.cell(row_number, column, value)
            set_cell_style(cell, size=7, align="left", wrap=True)
    scenario_sheet.freeze_panes = "A2"
    scenario_sheet.auto_filter.ref = scenario_sheet.dimensions
    scenario_sheet.print_area = scenario_sheet.dimensions
    scenario_sheet.page_setup.orientation = "landscape"
    scenario_sheet.page_setup.fitToWidth = 1
    scenario_sheet.page_setup.fitToHeight = 0
    scenario_sheet.sheet_properties.pageSetUpPr.fitToPage = True
    for column in range(1, scenario_sheet.max_column + 1):
        scenario_sheet.column_dimensions[get_column_letter(column)].width = (
            65 if column > 6 else 22
        )

    metadata = workbook.create_sheet("Metadata")
    metadata.sheet_view.showGridLines = False
    metadata_rows = [
        ("Field", "Value"),
        ("Table", TITLE),
        ("Districts", "197 current Districts or Municipalities"),
        ("Primary score", "Equal exact-percentile weights for threshold humid-heat burden, working-age scale, and vulnerable-worker scale"),
        ("Structural scenarios", "420: 2 burden definitions x 7 weights x 10 sample variants x 3 eligibility rules; fixed-count and worker-share selections retained"),
        ("Bootstrap", f"{BOOTSTRAP_REPLICATES} Survey Wave block replicates; seed {BOOTSTRAP_SEED}; threshold burden and equal-weight A5 fixed; only sampled Survey Waves vary"),
        ("Stable core", "Structural inclusion frequency at least 0.80 and Data Support Grade not Limited"),
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
    metadata.column_dimensions["A"].width = 26
    metadata.column_dimensions["B"].width = 100
    metadata.freeze_panes = "A2"
    metadata.auto_filter.ref = metadata.dimensions
    metadata.print_area = metadata.dimensions
    metadata.page_setup.fitToWidth = 1
    metadata.page_setup.fitToHeight = 1
    metadata.sheet_properties.pageSetUpPr.fitToPage = True

    workbook.save(output)


def render_review_pages(display: pd.DataFrame, review_dir: Path) -> list[Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    page_count = int(np.ceil(len(display) / PAGE_ROWS))
    with tempfile.TemporaryDirectory(
        prefix="mj02b-district-ranking-review-", dir="/private/tmp"
    ) as temporary:
        temporary_path = Path(temporary)
        for page_index in range(page_count):
            start = page_index * PAGE_ROWS
            stop = min((page_index + 1) * PAGE_ROWS, len(display))
            page = display.iloc[start:stop].copy()
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = f"Districts {start + 1}-{stop}"
            write_rankings_sheet(sheet, page, include_notes=False)
            note_row = 3 + len(page) + 1
            sheet.merge_cells(
                start_row=note_row,
                start_column=1,
                end_row=note_row,
                end_column=len(HEADERS),
            )
            note = sheet.cell(
                note_row,
                1,
                f"Review page {page_index + 1} of {page_count}; Primary Rank {start + 1}-{stop}. Full definitions and robustness audit are retained in the authoritative workbook.",
            )
            set_cell_style(note, color=MUTED, size=8, align="left", wrap=True)
            sheet.row_dimensions[note_row].height = 24
            sheet.print_area = f"A1:K{note_row}"
            sheet.page_setup.fitToWidth = 1
            sheet.page_setup.fitToHeight = 1
            temporary_workbook = temporary_path / f"review_page_{page_index + 1}.xlsx"
            workbook.save(temporary_workbook)
            output = review_dir / f"{REVIEW_STEM}_page_{page_index + 1}.png"
            render_review_png(temporary_workbook, output)
            pages.append(output)
    return pages


def validate_outputs(
    display: pd.DataFrame,
    audit: pd.DataFrame,
    output: Path,
    review_pages: list[Path],
) -> None:
    validate_ranking(display, audit)
    workbook = load_workbook(output, data_only=False, read_only=False)
    expected = [
        "District Rankings",
        "Bootstrap Robustness",
        "Structural Scenario Audit",
        "Metadata",
    ]
    if workbook.sheetnames != expected:
        raise AssertionError(f"Unexpected workbook sheets: {workbook.sheetnames}")
    sheet = workbook["District Rankings"]
    if sheet["A1"].value != TITLE:
        raise AssertionError("Workbook title is missing")
    if sheet["A2"].value != "District" or sheet.freeze_panes != "A3":
        raise AssertionError("The title must be followed immediately by column headers")
    if any(item.print_area is None for item in workbook.worksheets):
        raise AssertionError("Every worksheet must have a print area")
    workbook.close()
    expected_pages = int(np.ceil(len(display) / PAGE_ROWS))
    if len(review_pages) != expected_pages:
        raise AssertionError("Unexpected number of review pages")
    if any(not page.exists() or page.stat().st_size == 0 for page in review_pages):
        raise AssertionError("At least one PNG review page is missing")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_dir = resolve_path(args.review_dir)
    with psycopg.connect(**connection_args(args)) as connection:
        verify_read_only(connection)
        geography, climate, people = load_inputs(connection, args.schema)
        coastal_flags = load_coastal_flags(connection, args.schema)
    district, structural, scenarios = deterministic_robustness(
        geography, climate, people, coastal_flags
    )
    district_codes = district["admin2_code"].astype(str).tolist()
    bootstrap = bootstrap_robustness(
        climate,
        people,
        district_codes,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    display, audit = build_ranking_table(district, structural, bootstrap)
    build_workbook(display, audit, scenarios, bootstrap, output)
    review_pages = render_review_pages(display, review_dir)
    validate_outputs(display, audit, output, review_pages)

    print(f"table={output.relative_to(ROOT)}")
    for page in review_pages:
        print(f"review={page.relative_to(ROOT)}")
    print(
        f"rows={len(display)} columns={len(display.columns)} "
        f"structural_scenarios={len(scenarios)} bootstrap={args.bootstrap_replicates}"
    )
    print("database_read_only=true database_writes=none")
    for capacity in CAPACITIES:
        stable = int(
            (
                audit[f"inclusion_frequency_{capacity}"].ge(0.80)
                & audit["data_support_grade"].ne("Limited")
            ).sum()
        )
        print(f"stable_core_k={capacity} districts={stable}")


if __name__ == "__main__":
    main()
