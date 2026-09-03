#!/usr/bin/env python3
"""Audit feasibility of the current cumulative humid-heat labor study.

The diagnostic reads the current research questions from docs/AnaSOP.md and
checks the database objects supporting the frozen district-year-month design.
Every PostgreSQL operation is a SELECT executed in a server-enforced read-only
session. Outputs are exploratory diagnostics under data/exp/feasibility-check.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg import sql


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/exp/feasibility-check"

TABLES = [
    {
        "name": "final_HEAT_LABOR_ANALYTIC",
        "grain": "person-wave",
        "key": "survey_wave, person_id",
        "role": "month-matched labor outcomes, weights, demographics, and climate exposure",
        "required": [
            "survey_wave", "survey_year", "survey_month", "person_id",
            "admin2_code", "analysis_weight", "worked_past_week",
            "days_wbmax_ge_26c", "lag_days_wbmax_ge_26c",
            "wet_bulb_max_mean_c", "lag_wet_bulb_max_mean_c",
            "lead_days_wbmax_ge_26c", "lead_wet_bulb_max_mean_c",
            "days_tmax_ge_35c", "temperature_2m_max_mean_c",
            "precipitation_month_sum_mm",
        ],
        "limitation": "District-month exposure; exact interview day is unavailable.",
    },
    {
        "name": "final_CLIMATE_MONTHLY_ADMIN2",
        "grain": "district-calendar-month",
        "key": "admin2_code, year, month",
        "role": "monthly humid heat, dry heat, and rainfall linkage",
        "required": [],
        "limitation": "Short climate history cannot support a 25-year local climatology.",
    },
    {
        "name": "final_CLIMATE_DAILY_ADMIN2",
        "grain": "district-calendar-day",
        "key": "admin2_code, date",
        "role": "daily source for monthly threshold and continuous exposure measures",
        "required": [],
        "limitation": "Gridded district location rather than household-level exposure.",
    },
    {
        "name": "final_ED_CSES",
        "grain": "person-wave",
        "key": "survey_wave, person_id",
        "role": "harmonized education used for the vulnerability contrast",
        "required": ["survey_wave", "person_id", "education_level_harmonized"],
        "limitation": "Education is a vulnerability marker, not an identified mechanism.",
    },
    {
        "name": "final_EC_CSES",
        "grain": "person-wave",
        "key": "survey_wave, person_id",
        "role": "employment, hours, wage, occupation, and industry fields",
        "required": ["survey_wave", "person_id"],
        "limitation": "Hours and wages have incomplete wave coverage.",
    },
    {
        "name": "dim_admin2_cambodia",
        "grain": "district",
        "key": "admin2_code",
        "role": "district/province crosswalk and mapped geographic support",
        "required": ["admin2_code"],
        "limitation": "Current-boundary crosswalk rather than historical polygons.",
    },
]

ROLE_VARIABLES = {
    "survey_wave": "time",
    "survey_year": "time",
    "survey_month": "time",
    "admin2_code": "geography and fixed effect",
    "analysis_weight": "survey design",
    "worked_past_week": "primary outcome",
    "weekly_hours_including_zero": "secondary outcome",
    "log_monthly_salary_wages": "secondary outcome",
    "days_wbmax_ge_26c": "primary current humid-heat exposure",
    "lag_days_wbmax_ge_26c": "previous-month humid-heat exposure",
    "lead_days_wbmax_ge_26c": "future-month placebo exposure",
    "wet_bulb_max_mean_c": "continuous current humid-heat exposure",
    "lag_wet_bulb_max_mean_c": "continuous previous-month exposure",
    "lead_wet_bulb_max_mean_c": "continuous future-month placebo",
    "days_tmax_ge_35c": "dry-heat comparison",
    "temperature_2m_max_mean_c": "continuous dry-heat comparison",
    "precipitation_month_sum_mm": "weather control",
    "education_level_harmonized": "education vulnerability modifier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--anasop", type=Path, default=ROOT / "docs/AnaSOP.md")
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


def extract_research_questions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    section = text.split("## 1. Research Objective", 1)[1].split(
        "## 2. Theoretical Background", 1
    )[0]
    questions = re.findall(r"^- Research question:\s*(.+)$", section, flags=re.MULTILINE)
    if len(questions) != 5:
        raise RuntimeError(
            f"Expected five current AnaSOP research questions, observed {len(questions)}"
        )
    return questions


def table_columns(
    cursor: psycopg.Cursor, schema: str, table: str
) -> list[tuple[str, str, int]]:
    cursor.execute(
        """
        SELECT column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return [(str(a), str(b), int(c)) for a, b, c in cursor.fetchall()]


def table_row_and_non_null_counts(
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
    columns: list[tuple[str, str, int]],
) -> tuple[int, list[int]]:
    expressions = sql.SQL(", ").join(
        sql.SQL("count({})").format(sql.Identifier(column))
        for column, _, _ in columns
    )
    cursor.execute(
        sql.SQL("SELECT count(*), {} FROM {}.{}").format(
            expressions, sql.Identifier(schema), sql.Identifier(table)
        )
    )
    values = cursor.fetchone()
    if values is None:
        raise RuntimeError(f"Could not count {schema}.{table}")
    return int(values[0]), [int(value) for value in values[1:]]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feasibility_rows(
    questions: list[str], coverage: dict[str, int]
) -> list[dict[str, object]]:
    evidence = [
        (
            f"{coverage['cumulative_n']:,} weighted person-wave rows have both current "
            f"and previous-month humid-heat exposure; {coverage['education_n']:,} also "
            "have harmonized education."
        ),
        (
            "Current, previous, cumulative, and future exposure fields exist under both "
            "threshold and continuous wet-bulb definitions."
        ),
        (
            "Two-month humid-heat measures can be compared with matched Tmax>=35 C days "
            "and continuous maximum air temperature."
        ),
        (
            f"Harmonized education is available for {coverage['education_n']:,} "
            "cumulative-exposure observations and supports group slopes and an interaction."
        ),
        (
            "Hours, wages, demographics, occupation, crowding, and electricity fields "
            "exist, with incomplete coverage treated as an evidence boundary."
        ),
    ]
    gaps = [
        "Exact interview day is unavailable; inference is month-level and associational.",
        "The windows cannot be interpreted as exact exposure during the reported past seven days.",
        "Specificity comparisons do not isolate a physiological humidity mechanism.",
        "Education can proxy several resources and occupations; it is not a causal mechanism.",
        "Secondary outcomes and modifiers have incomplete or inconsistent cross-wave coverage.",
    ]
    roles = [
        "central cumulative association and education vulnerability",
        "timing and accumulation",
        "humid heat versus dry heat",
        "education-related vulnerability",
        "labor margins and unsupported mechanisms",
    ]
    return [
        {
            "question_id": "central" if index == 0 else f"support_{index}",
            "research_role": roles[index],
            "question": question,
            "status": "partly-testable",
            "data_evidence": evidence[index],
            "remaining_limit": gaps[index],
            "knowledge_scope": "Exploratory repeated-cross-section association; no causal claim.",
        }
        for index, question in enumerate(questions)
    ]


def main() -> int:
    args = parse_args()
    questions = extract_research_questions(args.anasop)
    availability: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    with psycopg.connect(**connection_args(args)) as connection:
        with connection.cursor() as cursor:
            for contract in TABLES:
                table = str(contract["name"])
                columns = table_columns(cursor, args.schema, table)
                if not columns:
                    raise RuntimeError(f"Required database table is missing: {args.schema}.{table}")
                names = {column for column, _, _ in columns}
                missing = sorted(set(contract["required"]) - names)
                if missing:
                    raise RuntimeError(f"{table} is missing required columns: {missing}")
                row_count, non_null_counts = table_row_and_non_null_counts(
                    cursor, args.schema, table, columns
                )
                availability.append(
                    {
                        "table_name": table,
                        "grain": contract["grain"],
                        "rows": row_count,
                        "columns": len(columns),
                        "unique_key": contract["key"],
                        "study_role": contract["role"],
                        "availability_status": "available",
                        "limitation": contract["limitation"],
                    }
                )
                for (column, data_type, ordinal), non_null in zip(
                    columns, non_null_counts, strict=True
                ):
                    inventory.append(
                        {
                            "table_name": table,
                            "grain": contract["grain"],
                            "variable_name": column,
                            "data_type": data_type,
                            "ordinal_position": ordinal,
                            "non_null_rows": non_null,
                            "missing_rows": row_count - non_null,
                            "missing_pct": round(100 * (row_count - non_null) / row_count, 4)
                            if row_count else 0.0,
                            "feasibility_role": ROLE_VARIABLES.get(
                                column, "supporting or provenance variable"
                            ),
                        }
                    )

            cursor.execute(
                sql.SQL(
                    """
                    SELECT
                        count(*) FILTER (WHERE a.analysis_weight > 0),
                        count(*) FILTER (
                            WHERE a.analysis_weight > 0
                              AND a.worked_past_week IS NOT NULL
                        ),
                        count(*) FILTER (
                            WHERE a.analysis_weight > 0
                              AND a.worked_past_week IS NOT NULL
                              AND a.days_wbmax_ge_26c IS NOT NULL
                              AND a.lag_days_wbmax_ge_26c IS NOT NULL
                              AND a.wet_bulb_max_mean_c IS NOT NULL
                              AND a.lag_wet_bulb_max_mean_c IS NOT NULL
                        ),
                        count(*) FILTER (
                            WHERE a.analysis_weight > 0
                              AND a.worked_past_week IS NOT NULL
                              AND a.days_wbmax_ge_26c IS NOT NULL
                              AND a.lag_days_wbmax_ge_26c IS NOT NULL
                              AND a.wet_bulb_max_mean_c IS NOT NULL
                              AND a.lag_wet_bulb_max_mean_c IS NOT NULL
                              AND ed.education_level_harmonized BETWEEN 0 AND 7
                        ),
                        count(DISTINCT a.survey_wave),
                        count(DISTINCT a.admin2_code)
                    FROM {}.{} a
                    LEFT JOIN {}.{} ed
                      ON ed.survey_wave = a.survey_wave
                     AND ed.person_id = a.person_id
                    """
                ).format(
                    sql.Identifier(args.schema),
                    sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
                    sql.Identifier(args.schema),
                    sql.Identifier("final_ED_CSES"),
                )
            )
            values = cursor.fetchone()
            if values is None:
                raise RuntimeError("Analytical coverage query returned no row")
            coverage = {
                "weighted_n": int(values[0]),
                "work_n": int(values[1]),
                "cumulative_n": int(values[2]),
                "education_n": int(values[3]),
                "waves": int(values[4]),
                "districts": int(values[5]),
            }
        connection.rollback()

    if coverage["cumulative_n"] <= 0 or coverage["education_n"] <= 0:
        raise RuntimeError("Current cumulative or education analytical sample is empty")
    question_rows = feasibility_rows(questions, coverage)
    statuses = Counter(str(row["status"]) for row in question_rows)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dataset_availability.csv", availability)
    write_csv(output / "variable_inventory.csv", inventory)
    write_csv(output / "question_feasibility.csv", question_rows)

    readme = f"""# Cumulative Humid-Heat Labor Study Feasibility Check

Generated: {datetime.now().astimezone().isoformat(timespec="seconds")}

## Evidence boundary

This diagnostic reads PostgreSQL {args.dbname}.{args.schema} in a
server-enforced read-only transaction. It evaluates the five current research
questions extracted from docs/AnaSOP.md; it does not redefine them.

## Current data support

- Required database tables available: {len(availability)} of {len(TABLES)}
- Database variables inventoried: {len(inventory)}
- Positive-weight person-wave rows: {coverage['weighted_n']:,}
- Work-participation observations: {coverage['work_n']:,}
- Complete cumulative-exposure observations: {coverage['cumulative_n']:,}
- Complete cumulative-exposure and education observations: {coverage['education_n']:,}
- Survey waves: {coverage['waves']}
- Districts: {coverage['districts']}
- Research questions assessed: {len(question_rows)}
- Feasibility statuses: {dict(statuses)}

## Feasibility conclusion

The current PostgreSQL data support estimation of the frozen cumulative
humid-heat, timing, dry-heat specificity, education-interaction, and evidence-boundary
analyses. This is data availability, not proof of any hypothesis. Exact interview
days remain unavailable, so the design is a district-month repeated-cross-section
association using nominal survey year and released survey month.

The project does not need to return to research-question planning or build a new
climate database before analysis. Formal results must continue to use read-only
database access and preserve the exploratory, non-causal, multiplicity, temporal,
and spatial limitations in AnaSOP.

## Workflow decision

The current questions are feasible for exploratory analysis under the frozen
month-level design. Continue with the established analytical and reporting workflow;
do not reinterpret the data as an interview-day heat-shock design.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    print(f"database_tables={len(availability)}")
    print(f"variables={len(inventory)}")
    print(f"questions={len(question_rows)}")
    print(f"statuses={dict(statuses)}")
    print(f"cumulative_n={coverage['cumulative_n']}")
    print(f"education_n={coverage['education_n']}")
    print(f"output={output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
