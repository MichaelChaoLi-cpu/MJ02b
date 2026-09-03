#!/usr/bin/env python3
"""Audit data feasibility for the occupational humid-heat impact assessment.

This project is database-first. The script therefore treats PostgreSQL as the
authoritative analytical source, forces every session to read-only, and writes
only exploratory feasibility artifacts under data/exp/feasibility-check/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/exp/feasibility-check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def fetch_frame(connection: psycopg.Connection, query: object) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def verify_read_only(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SHOW default_transaction_read_only")
        value = str(cursor.fetchone()[0]).lower()
    if value not in {"on", "true"}:
        raise RuntimeError("PostgreSQL session is not read-only")


def table_columns(
    connection: psycopg.Connection, schema: str, table: str
) -> list[str]:
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (schema, table))
        return [str(row[0]) for row in cursor.fetchall()]


def scalar_count(
    connection: psycopg.Connection,
    schema: str,
    table: str,
    distinct_column: str | None = None,
) -> tuple[int, int | None]:
    if distinct_column:
        query = sql.SQL("SELECT count(*)::bigint, count(DISTINCT {})::bigint FROM {}.{}").format(
            sql.Identifier(distinct_column), sql.Identifier(schema), sql.Identifier(table)
        )
    else:
        query = sql.SQL("SELECT count(*)::bigint, NULL::bigint FROM {}.{}").format(
            sql.Identifier(schema), sql.Identifier(table)
        )
    with connection.cursor() as cursor:
        cursor.execute(query)
        row = cursor.fetchone()
    return int(row[0]), None if row[1] is None else int(row[1])


def build_dataset_availability(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    specifications = [
        (
            "CSES-climate analytical relation",
            "final_HEAT_LABOR_ANALYTIC",
            "person-wave outcomes, survey weights, district linkage, current/lag climate",
            "admin2_code",
        ),
        (
            "Harmonized education",
            "final_ED_CSES",
            "education vulnerability",
            None,
        ),
        (
            "Monthly district climate",
            "final_CLIMATE_MONTHLY_ADMIN2",
            "historical humid-heat hazard",
            "admin2_code",
        ),
        (
            "Current district geography",
            "dim_admin2_cambodia",
            "district names, provinces, and map geometry",
            "admin2_code",
        ),
    ]
    rows: list[dict[str, object]] = []
    for source_name, table, role, distinct_column in specifications:
        columns = table_columns(connection, schema, table)
        available = bool(columns)
        record_count: int | None = None
        district_count: int | None = None
        if available:
            use_distinct = distinct_column if distinct_column in columns else None
            record_count, district_count = scalar_count(
                connection, schema, table, use_distinct
            )
        rows.append(
            {
                "dataset": source_name,
                "authoritative_source": f"PostgreSQL {schema}.{table}",
                "role": role,
                "available": available,
                "record_count": record_count,
                "district_count": district_count,
                "column_count": len(columns),
                "access_contract": "read-only SELECT",
            }
        )
    return pd.DataFrame(rows)


def load_district_support(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    query = sql.SQL(
        """
        WITH climate_cells AS (
            SELECT DISTINCT
                survey_wave,
                survey_year,
                survey_month,
                admin2_code,
                days_wbmax_ge_26c,
                lag_days_wbmax_ge_26c,
                wet_bulb_max_mean_c,
                lag_wet_bulb_max_mean_c
            FROM {}.{}
            WHERE analysis_weight > 0
              AND admin2_code IS NOT NULL
        ),
        heat AS (
            SELECT
                admin2_code,
                count(*) FILTER (
                    WHERE days_wbmax_ge_26c IS NOT NULL
                      AND lag_days_wbmax_ge_26c IS NOT NULL
                )::bigint AS heat_cells,
                count(DISTINCT survey_wave) FILTER (
                    WHERE days_wbmax_ge_26c IS NOT NULL
                      AND lag_days_wbmax_ge_26c IS NOT NULL
                )::bigint AS heat_waves,
                avg(days_wbmax_ge_26c + lag_days_wbmax_ge_26c) FILTER (
                    WHERE days_wbmax_ge_26c IS NOT NULL
                      AND lag_days_wbmax_ge_26c IS NOT NULL
                )::double precision AS mean_cumulative_wb26_days,
                stddev_pop(days_wbmax_ge_26c + lag_days_wbmax_ge_26c) FILTER (
                    WHERE days_wbmax_ge_26c IS NOT NULL
                      AND lag_days_wbmax_ge_26c IS NOT NULL
                )::double precision AS sd_cumulative_wb26_days,
                avg((wet_bulb_max_mean_c + lag_wet_bulb_max_mean_c) / 2.0) FILTER (
                    WHERE wet_bulb_max_mean_c IS NOT NULL
                      AND lag_wet_bulb_max_mean_c IS NOT NULL
                )::double precision AS mean_two_month_wbmax_c
            FROM climate_cells
            GROUP BY admin2_code
        ),
        people AS (
            SELECT
                a.survey_wave,
                a.person_id,
                a.admin2_code,
                a.analysis_weight::double precision AS analysis_weight,
                a.worked_past_week,
                ed.education_level_harmonized
            FROM {}.{} AS a
            LEFT JOIN {}.{} AS ed
              ON ed.survey_wave = a.survey_wave
             AND ed.person_id = a.person_id
            WHERE a.analysis_weight > 0
              AND a.admin2_code IS NOT NULL
        ),
        person_support AS (
            SELECT
                admin2_code,
                count(*)::bigint AS person_rows,
                count(DISTINCT survey_wave)::bigint AS person_waves,
                count(worked_past_week)::bigint AS work_rows,
                count(education_level_harmonized) FILTER (
                    WHERE education_level_harmonized BETWEEN 0 AND 7
                )::bigint AS education_rows,
                count(DISTINCT survey_wave) FILTER (
                    WHERE education_level_harmonized BETWEEN 0 AND 7
                )::bigint AS education_waves,
                sum(analysis_weight) FILTER (
                    WHERE education_level_harmonized BETWEEN 0 AND 7
                )::double precision AS education_weight,
                power(sum(analysis_weight) FILTER (
                    WHERE education_level_harmonized BETWEEN 0 AND 7
                ), 2) /
                nullif(sum(power(analysis_weight, 2)) FILTER (
                    WHERE education_level_harmonized BETWEEN 0 AND 7
                ), 0)::double precision AS education_effective_n,
                sum(
                    analysis_weight *
                    (education_level_harmonized BETWEEN 0 AND 2)::integer
                ) FILTER (
                    WHERE education_level_harmonized BETWEEN 0 AND 7
                ) /
                nullif(sum(analysis_weight) FILTER (
                    WHERE education_level_harmonized BETWEEN 0 AND 7
                ), 0)::double precision AS weighted_low_education_share
            FROM people
            GROUP BY admin2_code
        ),
        wave_district_scale AS (
            SELECT
                survey_wave,
                admin2_code,
                sum(analysis_weight)::double precision AS district_weight
            FROM people
            GROUP BY survey_wave, admin2_code
        ),
        wave_relative_scale AS (
            SELECT
                survey_wave,
                admin2_code,
                district_weight /
                nullif(sum(district_weight) OVER (PARTITION BY survey_wave), 0)
                    AS working_age_weight_share
            FROM wave_district_scale
        ),
        scale AS (
            SELECT
                admin2_code,
                avg(working_age_weight_share)::double precision
                    AS mean_wave_working_age_share,
                count(*)::bigint AS scale_waves
            FROM wave_relative_scale
            GROUP BY admin2_code
        )
        SELECT
            g.admin2_code,
            g.admin2_name,
            g.admin1_code,
            g.admin1_name,
            (g.geometry_geojson IS NOT NULL)::integer AS geometry_available,
            h.heat_cells,
            h.heat_waves,
            h.mean_cumulative_wb26_days,
            h.sd_cumulative_wb26_days,
            h.mean_two_month_wbmax_c,
            p.person_rows,
            p.person_waves,
            p.work_rows,
            p.education_rows,
            p.education_waves,
            p.education_weight,
            p.education_effective_n,
            p.weighted_low_education_share,
            s.mean_wave_working_age_share,
            s.scale_waves
        FROM {}.{} AS g
        LEFT JOIN heat AS h USING (admin2_code)
        LEFT JOIN person_support AS p USING (admin2_code)
        LEFT JOIN scale AS s USING (admin2_code)
        ORDER BY g.admin2_code
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
        sql.Identifier(schema),
        sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
        sql.Identifier(schema),
        sql.Identifier("final_ED_CSES"),
        sql.Identifier(schema),
        sql.Identifier("dim_admin2_cambodia"),
    )
    frame = fetch_frame(connection, query)
    numeric = [column for column in frame.columns if column not in {
        "admin2_code", "admin2_name", "admin1_code", "admin1_name"
    }]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["core_indicator_complete"] = frame[
        [
            "mean_cumulative_wb26_days",
            "mean_two_month_wbmax_c",
            "mean_wave_working_age_share",
            "weighted_low_education_share",
            "geometry_available",
        ]
    ].notna().all(axis=1)
    for threshold in (25, 50, 100, 200):
        frame[f"education_effective_n_ge_{threshold}"] = (
            frame["education_effective_n"] >= threshold
        )
    return frame


def extract_questions(anasop: Path) -> list[str]:
    text = anasop.read_text(encoding="utf-8")
    section = text.split("## 1. Research Objective", 1)[1].split(
        "## 2. Theoretical Background", 1
    )[0]
    return [
        match.strip()
        for match in re.findall(r"^- Research question:\s*(.+)$", section, re.MULTILINE)
    ]


def build_question_feasibility(
    questions: list[str], district_support: pd.DataFrame
) -> pd.DataFrame:
    complete = int(district_support["core_indicator_complete"].sum())
    supported_50 = int(district_support["education_effective_n_ge_50"].sum())
    evidence = [
        f"All four screening dimensions are derivable; {complete} districts have complete core indicators and {supported_50} have education effective n >= 50.",
        "Two cumulative humid-heat definitions, work participation, fixed effects, weights, and frozen robustness outputs already exist.",
        f"District heat, working-age scale, and weighted Low Education shares are derivable for {complete} districts; education precision can be screened explicitly.",
        "Six targeting alternatives and 10%, 20%, and 30% district-capacity sets are computable from relative indicators; uniform coverage is a full-national benchmark.",
        "Two exposure definitions, wave coverage, geographic restrictions, coefficient intervals, and resampling inputs support ranking-stability analysis.",
    ]
    limitations = [
        "Actual program costs and intervention effectiveness are unobserved; prospective benefits require transparent external assumptions.",
        "The labor association is exploratory, month-matched, non-causal, and does not survive global 10% false-discovery control.",
        "Global heat-education co-location is unsupported; local priority cannot be interpreted as a district-specific causal effect.",
        "Equal-district capacity is an illustrative resource unit, not an observed budget; monetized cost-benefit analysis is unavailable.",
        "Only nine survey waves are available and some districts have limited early-wave coverage; borderline ranks must be reported as uncertain.",
    ]
    next_checks = [
        "Freeze indicator transformations, minimum-support rules, policy alternatives, and externally specified effectiveness ranges before ranking.",
        "Reuse the accepted baseline models without adding outcome or subgroup searches.",
        "Select a minimum effective-sample rule and a prespecified uncertainty treatment for district education shares.",
        "Compare district-count and worker-share capacity definitions and report coverage-equity frontiers.",
        "Prespecify the admissible weight simplex, clustered resampling unit, top-list overlap, and inclusion-frequency thresholds.",
    ]
    if len(questions) != 5:
        raise AssertionError(f"Expected five approved questions, found {len(questions)}")
    return pd.DataFrame(
        {
            "question_id": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "research_question": questions,
            "status": ["partly-testable"] * 5,
            "current_data_support": evidence,
            "main_limitation": limitations,
            "knowledge_status": [
                "Can compare targeting consequences; cannot estimate realized program effects.",
                "Can preserve and summarize the accepted exploratory baseline.",
                "Can assess distributional coverage; cannot identify district causal impacts.",
                "Can compare relative allocation scenarios; cannot perform observed-cost benefit analysis.",
                "Can quantify ranking uncertainty and a stable core if one exists.",
            ],
            "required_next_check": next_checks,
        }
    )


def build_variable_inventory(district_support: pd.DataFrame) -> pd.DataFrame:
    district_n = len(district_support)
    complete_n = int(district_support["core_indicator_complete"].sum())
    rows = [
        ("District Humid-Heat Hazard", "current plus lag threshold days; current plus lag mean wet-bulb temperature", "district", "derivable", f"{complete_n}/{district_n} districts", "Primary hazard ranking under two definitions", "Historical survey-linked climate, not a future projection"),
        ("Weighted Working-Age Scale", "analysis sampling weight normalized within wave", "district", "derivable", f"{int(district_support['mean_wave_working_age_share'].notna().sum())}/{district_n} districts", "Relative population reach", "Repeated-wave relative scale, not a census population forecast"),
        ("District Low-Education Share", "education level and analysis sampling weight", "district", "derivable", f"{int(district_support['weighted_low_education_share'].notna().sum())}/{district_n} districts", "Equity and vulnerable-worker coverage", "Precision varies by district and requires a support rule"),
        ("Labor-Participation Risk Signal", "accepted cumulative heat coefficients and confidence intervals", "national coefficient applied to district scenarios", "available with assumptions", "Two exposure definitions", "Association-based screening range", "Exploratory and non-causal; not a district effect"),
        ("Data Support", "heat cells, survey waves, education effective sample size, geometry", "district", "derivable", f"{complete_n}/{district_n} complete core records", "Reliability screen and uncertainty flag", "Thresholds must be prespecified"),
        ("District Priority Score", "ranked hazard, exposure scale, vulnerability, and support", "district-by-weight scenario", "scenario-defined", f"{district_n} candidate districts", "Multi-criteria decision analysis", "No single objectively correct weight vector"),
        ("Capacity Level", "selected district count or covered worker share", "policy scenario", "scenario-defined", "10%, 20%, and 30% district tiers feasible", "Resource constraint", "Actual program budget is unobserved"),
        ("Policy Strategy", "no targeting, uniform, hazard, equity, worker scale, combined", "policy scenario", "scenario-defined", "Six alternatives", "Alternative comparison", "Uniform coverage is a benchmark rather than a constrained allocation"),
        ("Priority Inclusion Frequency", "selection across admissible weights and perturbations", "district", "derivable after prespecification", f"{district_n} candidate districts", "Ranking robustness", "Depends on the approved uncertainty set"),
        ("Program Cost", "not observed", "program", "not observed", "0 observed cost records", "Would support monetized efficiency", "Only relative capacity scenarios are currently defensible"),
        ("Program Effectiveness", "not observed", "program", "not observed", "0 intervention evaluations", "Would support avoided-impact estimation", "May enter only as an external sensitivity parameter"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "readable_construct",
            "source_or_definition",
            "analysis_level",
            "availability",
            "coverage",
            "planned_use",
            "interpretation_limit",
        ],
    )


def build_policy_scenarios(district_count: int) -> pd.DataFrame:
    strategies = [
        ("A0", "No additional targeting", "baseline", "No new district selection", False),
        ("A1", "Uniform national provision", "benchmark", "All districts", True),
        ("A2", "Hazard-first targeting", "ranked", "District Humid-Heat Hazard", True),
        ("A3", "Education-equity-first targeting", "ranked", "District Low-Education Share", True),
        ("A4", "Worker-scale-first targeting", "ranked", "Weighted Working-Age Scale", True),
        ("A5", "Combined targeting", "multi-criteria", "Hazard, working-age scale, vulnerability, and support", True),
    ]
    rows: list[dict[str, object]] = []
    for code, name, kind, inputs, feasible in strategies:
        for share in (0.10, 0.20, 0.30):
            if code == "A0":
                selected = 0
                capacity_note = "Reference without additional coverage"
            elif code == "A1":
                selected = district_count
                capacity_note = "Full-coverage benchmark; not constrained to the tier"
            else:
                selected = int(math.ceil(district_count * share))
                capacity_note = "Equal-district illustrative capacity"
            rows.append(
                {
                    "strategy_code": code,
                    "strategy": name,
                    "strategy_type": kind,
                    "capacity_share": share,
                    "selected_districts": selected,
                    "required_inputs": inputs,
                    "computable_now": feasible or code == "A0",
                    "capacity_interpretation": capacity_note,
                    "realized_program_effect_estimable": False,
                }
            )
    return pd.DataFrame(rows)


def describe_series(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": int(clean.size),
        "min": float(clean.min()),
        "median": float(clean.median()),
        "max": float(clean.max()),
    }


def write_readme(
    output: Path,
    datasets: pd.DataFrame,
    district_support: pd.DataFrame,
    questions: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> None:
    district_n = len(district_support)
    complete_n = int(district_support["core_indicator_complete"].sum())
    min_heat_waves = int(district_support["heat_waves"].min())
    median_heat_waves = float(district_support["heat_waves"].median())
    min_education_rows = int(district_support["education_rows"].min())
    median_education_rows = float(district_support["education_rows"].median())
    n_eff_50 = int(district_support["education_effective_n_ge_50"].sum())
    n_eff_100 = int(district_support["education_effective_n_ge_100"].sum())
    capacity = sorted(scenarios.loc[
        scenarios["strategy_code"] == "A2", "selected_districts"
    ].unique())
    lines = [
        "# Impact-Assessment Data Feasibility",
        "",
        "## Disposition",
        "",
        "The approved redesign is **partly testable and feasible as an ex ante targeting and decision-uncertainty analysis**. It is not feasible as a causal evaluation of realized program benefits or a monetized cost-benefit analysis with the current data.",
        "",
        "The generic feasibility scanner found zero CSV/TSV inputs because this project intentionally uses PostgreSQL as the authoritative analytical source. This database-aware audit replaces that file-based diagnostic while preserving its required output names.",
        "",
        "## Read-Only Data Contract",
        "",
        f"- PostgreSQL datasets verified: {int(datasets['available'].sum())}/{len(datasets)}.",
        "- Every database session used `default_transaction_read_only=on`.",
        "- No database object or research input was modified.",
        "",
        "## District Support",
        "",
        f"- Current districts in the geography: {district_n}.",
        f"- Districts with complete hazard, working-age scale, education vulnerability, support, and geometry fields: {complete_n}.",
        f"- Heat survey-wave support: minimum {min_heat_waves}, median {median_heat_waves:.1f}.",
        f"- Education rows per district: minimum {min_education_rows}, median {median_education_rows:.0f}.",
        f"- Districts with education effective sample size at least 50: {n_eff_50}; at least 100: {n_eff_100}.",
        f"- The 10%, 20%, and 30% equal-district scenarios select {capacity[0]}, {capacity[1]}, and {capacity[2]} districts.",
        "",
        "## Research-Question Status",
        "",
        f"- Questions assessed: {len(questions)}.",
        f"- Partly testable: {int((questions['status'] == 'partly-testable').sum())}.",
        "- Not-yet-testable: 0 after applying the approved database-first contract.",
        "",
        "## What Is Feasible",
        "",
        "- Preserve the accepted cumulative humid-heat and work-participation association as exploratory baseline evidence.",
        "- Construct district hazard under threshold and continuous wet-bulb definitions.",
        "- Construct relative working-age scale, weighted Low Education share, and data-support indicators.",
        "- Compare no-action, uniform, hazard-first, equity-first, worker-scale-first, and combined strategies.",
        "- Evaluate 10%, 20%, and 30% district-capacity tiers and alternative worker-share capacity definitions.",
        "- Quantify rank correlation, top-list overlap, inclusion frequency, and coverage-equity trade-offs across prespecified uncertainty scenarios.",
        "",
        "## What Is Not Directly Feasible",
        "",
        "- Estimating the causal effect of the proposed protection program.",
        "- Estimating actual implementation costs or monetized net benefits.",
        "- Producing district-specific causal heat effects from the repeated cross-sections.",
        "- Treating wet-bulb temperature as WBGT or as a regulatory safety cutoff.",
        "- Producing future climate projections without adding a separately approved climate-scenario dataset.",
        "",
        "## Required Design Decisions Before Final Construction",
        "",
        "1. Prespecify minimum district support for education shares and the treatment of borderline districts.",
        "2. Freeze indicator transformations and the admissible multi-criteria weight set.",
        "3. Define district-count capacity as the primary illustrative budget and worker-share capacity as sensitivity, or revise this choice.",
        "4. Decide whether association-based labor-risk screening adds decision value beyond hazard and population coverage; if retained, label it non-causal and propagate both exposure definitions and confidence intervals.",
        "5. Prespecify ranking-stability thresholds before inspecting final priority lists.",
        "",
        "## Recommended Workflow",
        "",
        "Proceed to data-preprocessing only for the newly approved district indicators, then revise the figure/table and estimation plans. All processing should pull from PostgreSQL in read-only mode and persist construction code under `src/analyses/`.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(**connection_args(args)) as connection:
        verify_read_only(connection)
        datasets = build_dataset_availability(connection, args.schema)
        if not datasets["available"].all():
            missing = datasets.loc[~datasets["available"], "authoritative_source"].tolist()
            raise RuntimeError(f"Required database sources are missing: {missing}")
        district_support = load_district_support(connection, args.schema)

    questions = extract_questions(ROOT / "docs/AnaSOP.md")
    question_feasibility = build_question_feasibility(questions, district_support)
    variable_inventory = build_variable_inventory(district_support)
    policy_scenarios = build_policy_scenarios(len(district_support))

    datasets.to_csv(output / "dataset_availability.csv", index=False)
    variable_inventory.to_csv(output / "variable_inventory.csv", index=False)
    question_feasibility.to_csv(output / "question_feasibility.csv", index=False)
    district_support.to_csv(output / "district_indicator_support.csv", index=False)
    policy_scenarios.to_csv(output / "policy_scenario_feasibility.csv", index=False)
    write_readme(output, datasets, district_support, question_feasibility, policy_scenarios)

    summary = {
        "database_read_only": True,
        "research_questions": len(question_feasibility),
        "status_counts": question_feasibility["status"].value_counts().to_dict(),
        "districts": len(district_support),
        "districts_complete_core": int(district_support["core_indicator_complete"].sum()),
        "education_effective_n": describe_series(district_support["education_effective_n"]),
        "education_support_thresholds": {
            str(threshold): int(
                district_support[f"education_effective_n_ge_{threshold}"].sum()
            )
            for threshold in (25, 50, 100, 200)
        },
        "heat_waves": describe_series(district_support["heat_waves"]),
        "person_rows": describe_series(district_support["person_rows"]),
        "generated_outputs": [
            "README.md",
            "dataset_availability.csv",
            "variable_inventory.csv",
            "question_feasibility.csv",
            "district_indicator_support.csv",
            "policy_scenario_feasibility.csv",
            "database_feasibility_summary.json",
        ],
        "disposition": "feasible for ex ante targeting and uncertainty analysis; not feasible for causal program-effect or monetized cost-benefit claims",
    }
    (output / "database_feasibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
