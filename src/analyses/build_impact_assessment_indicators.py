#!/usr/bin/env python3
"""Construct district indicators for the humid-heat social impact assessment.

The authoritative inputs remain in PostgreSQL. This script enforces a read-only
database session and writes reproducible exploratory audit snapshots only under
data/exp/. It does not create a processed-data cache or modify the database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/exp/impact-assessment-indicators"
DEFAULT_DECISION_OUTPUT = ROOT / "data/exp/data-preprocessing"

WEIGHT_SCENARIOS = {
    "heat_burden_only": (1.00, 0.00, 0.00),
    "worker_scale_only": (0.00, 1.00, 0.00),
    "vulnerable_worker_only": (0.00, 0.00, 1.00),
    "equal_weight": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "heat_burden_emphasis": (0.50, 0.25, 0.25),
    "worker_scale_emphasis": (0.25, 0.50, 0.25),
    "vulnerable_worker_emphasis": (0.25, 0.25, 0.50),
}


VARIABLE_DECISIONS = [
    {
        "readable_name": "District Threshold Humid-Heat Hazard",
        "full_name": "Wave-Month-Standardized District Cumulative Threshold Humid-Heat Hazard",
        "role": "impact-assessment hazard",
        "source_fields": ["days_wbmax_ge_26c", "lag_days_wbmax_ge_26c"],
        "construction": "Sum current and preceding-month threshold days, standardize across districts within Survey Wave and Survey Month, average within each District and Survey Wave, then average Survey Waves equally by District.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "District Continuous Humid-Heat Hazard",
        "full_name": "Wave-Month-Standardized District Continuous Cumulative Humid-Heat Hazard",
        "role": "replication hazard",
        "source_fields": ["wet_bulb_max_mean_c", "lag_wet_bulb_max_mean_c"],
        "construction": "Average current and preceding-month maximum wet-bulb means, standardize across districts within Survey Wave and Survey Month, average within each District and Survey Wave, then average Survey Waves equally by District.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Weighted Working-Age Scale",
        "full_name": "Wave-Balanced Relative Working-Age Population Scale",
        "role": "potential exposure scale",
        "source_fields": ["analysis_weight", "survey_wave", "admin2_code"],
        "construction": "Calculate each District share of positive working-age analysis weight within Survey Wave, average over observed waves, and renormalize across Districts.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "District Low-Education Share",
        "full_name": "Wave-Balanced Survey-Weighted District Low-Education Share",
        "role": "social vulnerability",
        "source_fields": ["education_level_harmonized", "analysis_weight", "survey_wave", "admin2_code"],
        "construction": "Calculate the weighted share with education levels 0-2 within each District and Survey Wave, then average over observed education waves.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "District Threshold Humid-Heat Burden",
        "full_name": "Working-Age-Weighted District Threshold Humid-Heat Burden",
        "role": "policy objective",
        "source_fields": ["weighted_working_age_scale", "mean_cumulative_wb26_days"],
        "construction": "Average Two-Month Cumulative Extreme Humid-Heat Days within each District and Survey Wave, average Survey Waves equally by District, and multiply by Weighted Working-Age Scale.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "District Continuous Humid-Heat Burden",
        "full_name": "Working-Age-Weighted District Continuous Excess Humid-Heat Burden",
        "role": "replication policy objective",
        "source_fields": [
            "weighted_working_age_scale",
            "mean_two_month_wbmax_excess_above_26c",
        ],
        "construction": "Calculate nonnegative Two-Month Average Maximum Wet-Bulb Temperature excess above 26 C in each District-wave-month cell, average within each District and Survey Wave, average Survey Waves equally by District, and multiply by Weighted Working-Age Scale.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "District Vulnerable-Worker Scale",
        "full_name": "Relative Low-Education Working-Age Scale",
        "role": "equity objective",
        "source_fields": ["weighted_working_age_scale", "district_low_education_share"],
        "construction": "Multiply Weighted Working-Age Scale by District Low-Education Share.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Education Effective Sample Size",
        "full_name": "Kish Effective Sample Size for District Education Share",
        "role": "uncertainty diagnostic",
        "source_fields": ["analysis_weight", "education_level_harmonized", "admin2_code"],
        "construction": "Square the sum of valid education weights and divide by the sum of squared weights within District.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Data Support Grade",
        "full_name": "District Impact-Assessment Data Support Grade",
        "role": "uncertainty flag",
        "source_fields": ["heat_waves", "education_effective_n"],
        "construction": "High when heat waves are at least 4 and education effective n is at least 100; Limited when heat waves are below 3 or education effective n is below 50; Medium otherwise.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "District Priority Score",
        "full_name": "Scenario-Specific District Occupational Humid-Heat Protection Priority Score",
        "role": "decision score",
        "source_fields": ["heat_burden_percentile", "working_age_percentile", "vulnerable_worker_percentile"],
        "construction": "Weighted sum of humid-heat burden, working-age scale, and vulnerable-worker scale percentile ranks under a prespecified nonnegative weight vector that sums to one; Data Support Grade is excluded.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Priority Inclusion Frequency",
        "full_name": "Structural District Priority-List Inclusion Frequency",
        "role": "decision robustness",
        "source_fields": [
            "district_priority_score",
            "capacity_level",
            "structural_scenario",
        ],
        "construction": "Share of 420 prespecified structural scenarios in which an eligible District enters the selected capacity tier; varies burden definition, component weights, full versus leave-one-wave-out construction, and eligibility rule.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Wave-Bootstrap Inclusion Frequency",
        "full_name": "Survey-Wave-Composition District Priority-List Inclusion Frequency",
        "role": "decision robustness",
        "source_fields": [
            "district_priority_score",
            "capacity_level",
            "survey_wave",
        ],
        "construction": "Share of 1,000 seeded Survey Wave block-bootstrap replicates in which a District enters the selected capacity tier when threshold burden and equal A5 weights are fixed and only Survey Waves are resampled.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Humid-Heat Coverage",
        "full_name": "Selected-District Weighted Cumulative Humid-Heat Coverage",
        "role": "policy consequence",
        "source_fields": ["weighted_working_age_scale", "mean_cumulative_wb26_days"],
        "construction": "Share of national working-age-weighted cumulative humid-heat burden located in selected Districts.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Vulnerable-Worker Coverage",
        "full_name": "Selected-District Low-Education Working-Age Coverage",
        "role": "distributional consequence",
        "source_fields": ["weighted_working_age_scale", "district_low_education_share"],
        "construction": "Share of the national relative low-education working-age scale located in selected Districts.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Working-Age Coverage",
        "full_name": "Selected-District Relative Working-Age Coverage",
        "role": "population consequence",
        "source_fields": ["weighted_working_age_scale"],
        "construction": "Sum of relative working-age scale in selected Districts.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Targeting Strategy",
        "full_name": "Occupational Humid-Heat Protection Targeting Strategy",
        "role": "policy alternative",
        "source_fields": [],
        "construction": "Prespecified allocation category for one common protection package: no additional targeting, uniform national provision, humid-heat burden first, vulnerable worker first, worker scale first, or combined targeting.",
        "is_final_variable": "yes",
    },
    {
        "readable_name": "Capacity Level",
        "full_name": "Occupational Humid-Heat Protection Implementation Capacity Level",
        "role": "policy constraint",
        "source_fields": [],
        "construction": "Primary capacity selects 20, 40, or 60 of 197 current Districts; required sensitivity selects the shortest ranked prefix reaching 10%, 20%, or 30% of Weighted Working-Age Scale.",
        "is_final_variable": "yes",
    },
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
    parser.add_argument(
        "--decision-output", type=Path, default=DEFAULT_DECISION_OUTPUT
    )
    return parser.parse_args()


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


def verify_read_only(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SHOW default_transaction_read_only")
        value = str(cursor.fetchone()[0]).lower()
    if value not in {"on", "true"}:
        raise RuntimeError("PostgreSQL session is not read-only")


def fetch_frame(connection: psycopg.Connection, query: object) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def load_inputs(
    connection: psycopg.Connection, schema: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    geography_query = sql.SQL(
        """
        SELECT admin2_code, admin2_name, admin1_code, admin1_name,
               (geometry_geojson IS NOT NULL)::integer AS geometry_available
        FROM {}.{}
        ORDER BY admin2_code
        """
    ).format(sql.Identifier(schema), sql.Identifier("dim_admin2_cambodia"))
    climate_query = sql.SQL(
        """
        SELECT DISTINCT
            survey_wave, survey_year, survey_month, admin2_code,
            days_wbmax_ge_26c, lag_days_wbmax_ge_26c,
            wet_bulb_max_mean_c, lag_wet_bulb_max_mean_c
        FROM {}.{}
        WHERE analysis_weight > 0
          AND admin2_code IS NOT NULL
        ORDER BY survey_year, survey_month, admin2_code
        """
    ).format(sql.Identifier(schema), sql.Identifier("final_HEAT_LABOR_ANALYTIC"))
    people_query = sql.SQL(
        """
        SELECT
            a.survey_wave,
            a.person_id,
            a.admin2_code,
            a.analysis_weight,
            ed.education_level_harmonized
        FROM {}.{} AS a
        LEFT JOIN {}.{} AS ed
          ON ed.survey_wave = a.survey_wave
         AND ed.person_id = a.person_id
        WHERE a.analysis_weight > 0
          AND a.admin2_code IS NOT NULL
        ORDER BY a.survey_wave, a.person_id
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
        sql.Identifier(schema),
        sql.Identifier("final_ED_CSES"),
    )
    geography = fetch_frame(connection, geography_query)
    climate = fetch_frame(connection, climate_query)
    people = fetch_frame(connection, people_query)
    return geography, climate, people


def standardize_within_group(
    frame: pd.DataFrame, column: str, groups: list[str]
) -> pd.Series:
    means = frame.groupby(groups, observed=True)[column].transform("mean")
    sds = frame.groupby(groups, observed=True)[column].transform(
        lambda values: float(values.std(ddof=0))
    )
    return ((frame[column] - means) / sds).where(sds.gt(0))


def build_hazard(climate: pd.DataFrame) -> pd.DataFrame:
    climate = climate.copy()
    numeric = [
        "survey_year",
        "survey_month",
        "days_wbmax_ge_26c",
        "lag_days_wbmax_ge_26c",
        "wet_bulb_max_mean_c",
        "lag_wet_bulb_max_mean_c",
    ]
    for column in numeric:
        climate[column] = pd.to_numeric(climate[column], errors="coerce")
    cell_key = ["survey_wave", "survey_year", "survey_month", "admin2_code"]
    if climate.duplicated(cell_key).any():
        raise AssertionError("Climate records are not unique by district wave-month")
    climate["cumulative_wb26_days"] = (
        climate["days_wbmax_ge_26c"] + climate["lag_days_wbmax_ge_26c"]
    ).where(
        climate[["days_wbmax_ge_26c", "lag_days_wbmax_ge_26c"]]
        .notna()
        .all(axis=1)
    )
    climate["two_month_wbmax_mean_c"] = (
        climate["wet_bulb_max_mean_c"] + climate["lag_wet_bulb_max_mean_c"]
    ).div(2.0).where(
        climate[["wet_bulb_max_mean_c", "lag_wet_bulb_max_mean_c"]]
        .notna()
        .all(axis=1)
    )
    climate["two_month_wbmax_excess_above_26c"] = (
        climate["two_month_wbmax_mean_c"] - 26.0
    ).clip(lower=0.0)
    standardization_cell = ["survey_wave", "survey_year", "survey_month"]
    climate["threshold_hazard_z"] = standardize_within_group(
        climate, "cumulative_wb26_days", standardization_cell
    )
    climate["continuous_hazard_z"] = standardize_within_group(
        climate, "two_month_wbmax_mean_c", standardization_cell
    )
    district_wave = (
        climate.groupby(["admin2_code", "survey_wave"], observed=True)
        .agg(
            wave_threshold_hazard=("threshold_hazard_z", "mean"),
            wave_continuous_hazard=("continuous_hazard_z", "mean"),
            wave_mean_cumulative_wb26_days=("cumulative_wb26_days", "mean"),
            wave_mean_two_month_wbmax_c=("two_month_wbmax_mean_c", "mean"),
            wave_mean_two_month_wbmax_excess_above_26c=(
                "two_month_wbmax_excess_above_26c",
                "mean",
            ),
            wave_heat_cells=("cumulative_wb26_days", "count"),
        )
        .reset_index()
    )
    return (
        district_wave.groupby("admin2_code", observed=True)
        .agg(
            district_threshold_humid_heat_hazard=("wave_threshold_hazard", "mean"),
            district_continuous_humid_heat_hazard=("wave_continuous_hazard", "mean"),
            mean_cumulative_wb26_days=("wave_mean_cumulative_wb26_days", "mean"),
            mean_two_month_wbmax_c=("wave_mean_two_month_wbmax_c", "mean"),
            mean_two_month_wbmax_excess_above_26c=(
                "wave_mean_two_month_wbmax_excess_above_26c",
                "mean",
            ),
            heat_cells=("wave_heat_cells", "sum"),
            heat_waves=("survey_wave", "nunique"),
        )
        .reset_index()
    )


def build_people_indicators(people: pd.DataFrame) -> pd.DataFrame:
    people = people.copy()
    people["analysis_weight"] = pd.to_numeric(
        people["analysis_weight"], errors="coerce"
    )
    people["education_level_harmonized"] = pd.to_numeric(
        people["education_level_harmonized"], errors="coerce"
    )
    person_key = ["survey_wave", "person_id"]
    if people.duplicated(person_key).any():
        raise AssertionError("Person records are not unique within Survey Wave")

    wave_scale = (
        people.groupby(["survey_wave", "admin2_code"], observed=True)
        .agg(district_weight=("analysis_weight", "sum"))
        .reset_index()
    )
    wave_scale["wave_total_weight"] = wave_scale.groupby(
        "survey_wave", observed=True
    )["district_weight"].transform("sum")
    wave_scale["working_age_share"] = (
        wave_scale["district_weight"] / wave_scale["wave_total_weight"]
    )
    scale = (
        wave_scale.groupby("admin2_code", observed=True)
        .agg(
            weighted_working_age_scale_raw=("working_age_share", "mean"),
            working_age_scale_waves=("survey_wave", "nunique"),
        )
        .reset_index()
    )
    scale["weighted_working_age_scale"] = (
        scale["weighted_working_age_scale_raw"]
        / scale["weighted_working_age_scale_raw"].sum()
    )

    valid = people["education_level_harmonized"].between(0, 7)
    education = people.loc[valid].copy()
    education["low_education"] = education["education_level_harmonized"].le(2)
    education["weighted_low_education"] = (
        education["analysis_weight"] * education["low_education"].astype(float)
    )
    wave_education = (
        education.groupby(["survey_wave", "admin2_code"], observed=True)
        .agg(
            education_weight=("analysis_weight", "sum"),
            weighted_low_education=("weighted_low_education", "sum"),
            education_rows=("person_id", "size"),
        )
        .reset_index()
    )
    wave_education["wave_low_education_share"] = (
        wave_education["weighted_low_education"]
        / wave_education["education_weight"]
    )
    education_summary = (
        wave_education.groupby("admin2_code", observed=True)
        .agg(
            district_low_education_share=("wave_low_education_share", "mean"),
            education_waves=("survey_wave", "nunique"),
            education_rows=("education_rows", "sum"),
        )
        .reset_index()
    )
    pooled = (
        education.groupby("admin2_code", observed=True)
        .agg(
            pooled_weighted_low_education=("weighted_low_education", "sum"),
            pooled_education_weight=("analysis_weight", "sum"),
            sum_squared_education_weight=(
                "analysis_weight",
                lambda values: float(np.square(values.astype(float)).sum()),
            ),
        )
        .reset_index()
    )
    pooled["pooled_low_education_share"] = (
        pooled["pooled_weighted_low_education"]
        / pooled["pooled_education_weight"]
    )
    pooled["education_effective_sample_size"] = (
        pooled["pooled_education_weight"].pow(2)
        / pooled["sum_squared_education_weight"]
    )
    pooled = pooled.drop(
        columns=[
            "pooled_weighted_low_education",
            "pooled_education_weight",
            "sum_squared_education_weight",
        ]
    )
    return scale.merge(
        education_summary, on="admin2_code", how="outer", validate="one_to_one"
    ).merge(pooled, on="admin2_code", how="outer", validate="one_to_one")


def support_grade(row: pd.Series) -> str:
    if row["education_effective_sample_size"] < 50 or row["heat_waves"] < 3:
        return "Limited"
    if row["education_effective_sample_size"] >= 100 and row["heat_waves"] >= 4:
        return "High"
    return "Medium"


def percentile_rank(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="average")
    denominator = int(series.notna().sum()) - 1
    if denominator <= 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (ranks - 1.0) / denominator


def build_district_indicators(
    geography: pd.DataFrame, climate: pd.DataFrame, people: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hazard = build_hazard(climate)
    people_indicators = build_people_indicators(people)
    district = geography.merge(
        hazard, on="admin2_code", how="left", validate="one_to_one"
    ).merge(
        people_indicators, on="admin2_code", how="left", validate="one_to_one"
    )
    if len(district) != 197:
        raise AssertionError(f"Expected 197 districts, found {len(district)}")
    required = [
        "district_threshold_humid_heat_hazard",
        "district_continuous_humid_heat_hazard",
        "mean_two_month_wbmax_excess_above_26c",
        "weighted_working_age_scale",
        "district_low_education_share",
        "education_effective_sample_size",
        "heat_waves",
        "geometry_available",
    ]
    missing = district[required].isna().sum()
    if int(missing.sum()) != 0:
        raise AssertionError(f"Core indicator missingness: {missing.to_dict()}")
    if not np.isclose(district["weighted_working_age_scale"].sum(), 1.0):
        raise AssertionError("Weighted Working-Age Scale does not sum to one")
    if not district["district_low_education_share"].between(0, 1).all():
        raise AssertionError("District Low-Education Share is outside [0, 1]")

    district["district_threshold_humid_heat_burden"] = (
        district["weighted_working_age_scale"]
        * district["mean_cumulative_wb26_days"]
    )
    district["district_continuous_humid_heat_burden"] = (
        district["weighted_working_age_scale"]
        * district["mean_two_month_wbmax_excess_above_26c"]
    )
    if not district["district_continuous_humid_heat_burden"].ge(0).all():
        raise AssertionError("Continuous humid-heat burden must be nonnegative")
    district["district_vulnerable_worker_scale"] = (
        district["weighted_working_age_scale"]
        * district["district_low_education_share"]
    )
    district["data_support_grade"] = district.apply(support_grade, axis=1)
    district["threshold_hazard_percentile"] = percentile_rank(
        district["district_threshold_humid_heat_hazard"]
    )
    district["continuous_hazard_percentile"] = percentile_rank(
        district["district_continuous_humid_heat_hazard"]
    )
    district["working_age_percentile"] = percentile_rank(
        district["weighted_working_age_scale"]
    )
    district["low_education_percentile"] = percentile_rank(
        district["district_low_education_share"]
    )
    district["threshold_burden_percentile"] = percentile_rank(
        district["district_threshold_humid_heat_burden"]
    )
    district["continuous_burden_percentile"] = percentile_rank(
        district["district_continuous_humid_heat_burden"]
    )
    district["vulnerable_worker_percentile"] = percentile_rank(
        district["district_vulnerable_worker_scale"]
    )

    scenario_rows: list[pd.DataFrame] = []
    for exposure_definition, burden_column in [
        ("threshold", "threshold_burden_percentile"),
        ("continuous", "continuous_burden_percentile"),
    ]:
        for scenario, (weight_burden, weight_scale, weight_vulnerable) in WEIGHT_SCENARIOS.items():
            score = (
                weight_burden * district[burden_column]
                + weight_scale * district["working_age_percentile"]
                + weight_vulnerable * district["vulnerable_worker_percentile"]
            )
            part = district[
                [
                    "admin2_code",
                    "admin2_name",
                    "admin1_code",
                    "admin1_name",
                    "data_support_grade",
                ]
            ].copy()
            part["exposure_definition"] = exposure_definition
            part["weight_scenario"] = scenario
            part["weight_heat_burden"] = weight_burden
            part["weight_working_age"] = weight_scale
            part["weight_vulnerable_worker"] = weight_vulnerable
            part["district_priority_score"] = score
            part["priority_rank"] = score.rank(
                method="min", ascending=False
            ).astype(int)
            scenario_rows.append(part)
    rankings = pd.concat(scenario_rows, ignore_index=True)
    return district.sort_values("admin2_code").reset_index(drop=True), rankings


def summarize_indicators(district: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "district_threshold_humid_heat_hazard",
        "district_continuous_humid_heat_hazard",
        "district_threshold_humid_heat_burden",
        "district_continuous_humid_heat_burden",
        "district_vulnerable_worker_scale",
        "mean_cumulative_wb26_days",
        "mean_two_month_wbmax_c",
        "mean_two_month_wbmax_excess_above_26c",
        "weighted_working_age_scale",
        "district_low_education_share",
        "pooled_low_education_share",
        "education_effective_sample_size",
        "heat_cells",
        "heat_waves",
        "education_rows",
        "education_waves",
    ]
    rows: list[dict[str, object]] = []
    for field in fields:
        values = pd.to_numeric(district[field], errors="coerce")
        rows.append(
            {
                "indicator": field,
                "districts": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "minimum": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def variable_list_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in VARIABLE_DECISIONS:
        rows.append(
            {
                "authoritative_dataset": "PostgreSQL mda.public plus scenario definition",
                "source_fields": "; ".join(item["source_fields"]) or "scenario-defined",
                "readable_name": item["readable_name"],
                "full_name": item["full_name"],
                "role": item["role"],
                "feasibility_status": "partly-testable",
                "construction": item["construction"],
                "is_final_variable": item["is_final_variable"],
            }
        )
    return pd.DataFrame(rows)


def write_decision_outputs(decision_output: Path) -> None:
    decision_output.mkdir(parents=True, exist_ok=True)
    variable_list_frame().to_csv(decision_output / "variable_list.csv", index=False)
    decisions = {
        "architecture": "database-first read-only",
        "authoritative_input": "PostgreSQL mda.public",
        "processed_parquet": None,
        "persistent_script": "src/analyses/build_impact_assessment_indicators.py",
        "audit_output": "data/exp/impact-assessment-indicators/district_indicators.csv",
        "variables": VARIABLE_DECISIONS,
        "support_grade": {
            "High": "heat waves >= 4 and education effective sample size >= 100",
            "Limited": "heat waves < 3 or education effective sample size < 50",
            "Medium": "all remaining districts",
            "priority_score_use": "excluded from the substantive priority score; used only for reliability display and sensitivity restrictions",
        },
        "weight_scenarios": {
            key: {
                "heat_burden": values[0],
                "working_age_scale": values[1],
                "vulnerable_worker_scale": values[2],
            }
            for key, values in WEIGHT_SCENARIOS.items()
        },
        "missing_value_rule": "Do not impute a missing core policy indicator; stop construction and report the affected District.",
        "outlier_rule": "Use percentile ranks without winsorizing the substantive indicators.",
    }
    (decision_output / "decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = """# Database-First Variable Construction

The impact-assessment variables are constructed from the authoritative PostgreSQL
analytical relations under `default_transaction_read_only=on`.

This project intentionally does not create `data/processed/*.parquet`. The reusable
construction code is `src/analyses/build_impact_assessment_indicators.py`; its CSV
outputs under `data/exp/impact-assessment-indicators/` are audit snapshots rather
than alternative authoritative inputs.

Seventeen readable final variables are specified in `variable_list.csv` and
`decisions.json`. Data Support Grade is kept separate from substantive policy
priority so that sparse measurement does not automatically deprioritize a district.
"""
    (decision_output / "README.md").write_text(readme, encoding="utf-8")


def write_audit_outputs(
    output: Path, district: pd.DataFrame, rankings: pd.DataFrame
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    district.to_csv(output / "district_indicators.csv", index=False)
    rankings.to_csv(output / "priority_scenario_rankings.csv", index=False)
    summary = summarize_indicators(district)
    summary.to_csv(output / "indicator_summary.csv", index=False)
    correlation_fields = [
        "district_threshold_humid_heat_burden",
        "district_continuous_humid_heat_burden",
        "weighted_working_age_scale",
        "district_vulnerable_worker_scale",
    ]
    correlations = district[correlation_fields].corr(method="spearman")
    correlations.to_csv(output / "indicator_spearman_correlations.csv")
    support_counts = (
        district["data_support_grade"].value_counts().reindex(
            ["High", "Medium", "Limited"], fill_value=0
        )
    )
    threshold_continuous_rho = float(
        correlations.loc[
            "district_threshold_humid_heat_burden",
            "district_continuous_humid_heat_burden",
        ]
    )
    manifest = {
        "database_read_only": True,
        "districts": len(district),
        "core_missing_values": int(
            district[
                [
                    "district_threshold_humid_heat_hazard",
                    "district_continuous_humid_heat_hazard",
                    "district_threshold_humid_heat_burden",
                    "district_continuous_humid_heat_burden",
                    "district_vulnerable_worker_scale",
                    "weighted_working_age_scale",
                    "district_low_education_share",
                    "education_effective_sample_size",
                    "data_support_grade",
                ]
            ].isna().sum().sum()
        ),
        "working_age_scale_sum": float(district["weighted_working_age_scale"].sum()),
        "support_grade_counts": {key: int(value) for key, value in support_counts.items()},
        "threshold_continuous_burden_spearman": threshold_continuous_rho,
        "weight_scenarios": len(WEIGHT_SCENARIOS),
        "ranking_rows": len(rankings),
        "data_support_in_priority_score": False,
        "generated_outputs": [
            "district_indicators.csv",
            "priority_scenario_rankings.csv",
            "indicator_summary.csv",
            "indicator_spearman_correlations.csv",
            "construction_manifest.json",
            "README.md",
        ],
    }
    (output / "construction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = f"""# District Impact-Assessment Indicator Audit

## Construction Status

- PostgreSQL access: read-only.
- Districts constructed: {len(district)}.
- Core missing values: {manifest['core_missing_values']}.
- Working-Age Scale sum: {manifest['working_age_scale_sum']:.6f}.
- Support grades: High {support_counts['High']}, Medium {support_counts['Medium']}, Limited {support_counts['Limited']}.
- Threshold-continuous District Humid-Heat Burden Spearman correlation: {threshold_continuous_rho:.3f}.
- Prespecified ranking scenarios: {len(WEIGHT_SCENARIOS)} per exposure definition.
- Data Support Grade enters the substantive priority score: no.

## Interpretation

The files in this folder are reproducible construction and review snapshots. They
are not a database replacement and are not final figures or tables. Priority ranks
remain scenario-specific and must be evaluated through capacity, coverage, equity,
and uncertainty analyses before any district recommendation is made. Continuous
burden uses only nonnegative Two-Month Average Maximum Wet-Bulb Temperature excess
above 26 C and never multiplies population scale by absolute Celsius temperature.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    with psycopg.connect(**connection_args(args)) as connection:
        verify_read_only(connection)
        geography, climate, people = load_inputs(connection, args.schema)
    district, rankings = build_district_indicators(geography, climate, people)
    write_decision_outputs(args.decision_output.resolve())
    manifest = write_audit_outputs(args.output.resolve(), district, rankings)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
