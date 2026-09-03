#!/usr/bin/env python3
"""Validate the database-only CSES heat/labor analytic table."""

from __future__ import annotations

import argparse

import psycopg
from psycopg import sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    return parser.parse_args()


def scalar(cursor: psycopg.Cursor, query: object) -> object:
    cursor.execute(query)
    return cursor.fetchone()[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    table = "final_HEAT_LABOR_ANALYTIC"
    with psycopg.connect(host=args.host, port=args.port, dbname=args.dbname) as connection:
        with connection.cursor() as cursor:
            rows = int(scalar(
                cursor,
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(args.schema), sql.Identifier(table)
                ),
            ))
            duplicates = int(scalar(
                cursor,
                sql.SQL(
                    """
                    SELECT count(*) FROM (
                        SELECT survey_wave, person_id, count(*)
                        FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1
                    ) x
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(table)),
            ))
            invalid = int(scalar(
                cursor,
                sql.SQL(
                    """
                    SELECT count(*) FROM {}.{}
                    WHERE age NOT BETWEEN 15 AND 64
                       OR survey_month NOT BETWEEN 1 AND 12
                       OR admin2_code IS NULL
                       OR days_wbmax_ge_26c IS NULL
                       OR wet_bulb_max_mean_c IS NULL
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(table)),
            ))
            require(rows > 0, "Analytic table is empty")
            require(duplicates == 0, f"Duplicate person-wave keys={duplicates}")
            require(invalid == 0, f"Invalid core analytic rows={invalid}")

            cursor.execute(
                sql.SQL(
                    """
                    SELECT survey_wave, count(*), count(DISTINCT admin2_code),
                           round(avg((analysis_weight IS NOT NULL AND analysis_weight > 0)::int)::numeric, 4),
                           round(avg((worked_past_week IS NOT NULL)::int)::numeric, 4),
                           round(avg((weekly_hours_including_zero IS NOT NULL)::int)::numeric, 4),
                           round(avg((log_monthly_salary_wages IS NOT NULL)::int)::numeric, 4)
                    FROM {}.{} GROUP BY survey_wave ORDER BY min(survey_year)
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(table))
            )
            for row in cursor.fetchall():
                print("wave_validation=" + ",".join(map(str, row)))
            fallback_rows = int(scalar(
                cursor,
                sql.SQL(
                    "SELECT count(*) FROM {}.{} WHERE temperature_location_fallback_used"
                ).format(sql.Identifier(args.schema), sql.Identifier(table)),
            ))
            print(f"validated={args.schema}.{table} rows={rows}")
            print(f"coastal_fallback_person_rows={fallback_rows}")
            print("validation=passed")


if __name__ == "__main__":
    main()
