#!/usr/bin/env python3
"""Validate published daily and monthly district heat exposures."""

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
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--start-date", default="1991-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--full-calendar", action="store_true")
    parser.add_argument("--full-survey-years", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scalar(cursor: psycopg.Cursor, query: object, parameters: tuple = ()) -> object:
    cursor.execute(query, parameters)
    return cursor.fetchone()[0]


def main() -> None:
    args = parse_args()
    connect: dict[str, object] = {
        "host": args.host, "port": args.port, "dbname": args.dbname
    }
    if args.user:
        connect["user"] = args.user
    if args.password:
        connect["password"] = args.password

    daily = "final_CLIMATE_DAILY_ADMIN2"
    monthly = "final_CLIMATE_MONTHLY_ADMIN2"
    with psycopg.connect(**connect) as connection:
        with connection.cursor() as cursor:
            n_admin2 = int(scalar(
                cursor,
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(args.schema), sql.Identifier("dim_admin2_cambodia")
                ),
            ))
            full_year_mode = args.full_survey_years
            if not args.full_calendar and not args.full_survey_years:
                full_year_mode = bool(scalar(
                    cursor,
                    sql.SQL(
                        """
                        WITH coverage AS (
                            SELECT admin2_code, reference_year,
                                   min(observation_date) AS first_date,
                                   max(observation_date) AS last_date,
                                   count(*) AS day_count
                            FROM {}.{}
                            GROUP BY admin2_code, reference_year
                        )
                        SELECT bool_and(
                            first_date = make_date(reference_year, 1, 1)
                            AND last_date = make_date(reference_year, 12, 31)
                            AND day_count = (
                                make_date(reference_year + 1, 1, 1)
                                - make_date(reference_year, 1, 1)
                            )
                        )
                        FROM coverage
                        """
                    ).format(sql.Identifier(args.schema), sql.Identifier(daily)),
                ))
            if args.full_calendar:
                expected_days = int(scalar(
                    cursor,
                    "SELECT (%s::date - %s::date) + 1",
                    (args.end_date, args.start_date),
                ))
                expected_rows = n_admin2 * expected_days
            elif full_year_mode:
                expected_rows = int(scalar(
                    cursor,
                    sql.SQL(
                        """
                        WITH published_years AS (
                            SELECT DISTINCT reference_year AS survey_year
                            FROM {}.{}
                        ),
                        combos AS (
                            SELECT DISTINCT g.admin2_code, h.survey_year
                            FROM {}.{} h
                            JOIN {}.{} g USING (survey_wave, psu)
                            JOIN published_years y ON y.survey_year = h.survey_year
                            WHERE g.admin2_current_match
                              AND h.survey_month BETWEEN 1 AND 12
                        )
                        SELECT sum(
                            make_date(survey_year, 12, 31)
                            - make_date(survey_year, 1, 1) + 1
                        )::bigint
                        FROM combos
                        """
                    ).format(
                        sql.Identifier(args.schema), sql.Identifier(daily),
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                        sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES"),
                    ),
                ))
            else:
                expected_rows = int(scalar(
                    cursor,
                    sql.SQL(
                        """
                        WITH combos AS (
                            SELECT DISTINCT g.admin2_code, h.survey_year, h.survey_month
                            FROM {}.{} h
                            JOIN {}.{} g USING (survey_wave, psu)
                            WHERE g.admin2_current_match
                              AND h.survey_month BETWEEN 1 AND 12
                              AND make_date(h.survey_year, h.survey_month, 1)
                                  BETWEEN date_trunc('month', %s::date)
                                      AND date_trunc('month', %s::date)
                        )
                        SELECT sum(extract(day FROM
                            (make_date(survey_year, survey_month, 1)
                             + interval '1 month - 1 day')))::bigint
                        FROM combos
                        """
                    ).format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                        sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES"),
                    ),
                    (args.start_date, args.end_date),
                ))
            daily_rows = int(scalar(
                cursor,
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(args.schema), sql.Identifier(daily)
                ),
            ))
            require(daily_rows == expected_rows,
                    f"Daily rows={daily_rows}, expected={expected_rows}")

            duplicates = int(scalar(
                cursor,
                sql.SQL(
                    """
                    SELECT count(*) FROM (
                        SELECT admin2_code, observation_date, count(*)
                        FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1
                    ) x
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(daily)),
            ))
            require(duplicates == 0, f"Duplicate daily keys={duplicates}")

            missing_core = int(scalar(
                cursor,
                sql.SQL(
                    """
                    SELECT count(*) FROM {}.{}
                    WHERE temperature_2m_max_c IS NULL
                       OR wet_bulb_temperature_2m_max_c IS NULL
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(daily)),
            ))
            require(missing_core == 0, f"Daily rows missing core exposures={missing_core}")

            cursor.execute(
                sql.SQL(
                    """
                    SELECT count(temperature_2m_mean_c),
                           count(dew_point_2m_max_c),
                           count(relative_humidity_2m_max_pct),
                           count(apparent_temperature_max_c),
                           count(precipitation_sum_mm)
                    FROM {}.{}
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(daily))
            )
            optional_counts = tuple(int(value) for value in cursor.fetchone())
            require(
                all(value in (0, daily_rows) for value in optional_counts),
                f"Partially populated optional daily variables={optional_counts}",
            )
            require(
                len(set(optional_counts[:3])) == 1,
                f"Inconsistent ERA5-Land auxiliary coverage={optional_counts[:3]}",
            )
            require(
                len(set(optional_counts[3:])) == 1,
                f"Inconsistent ERA5 control coverage={optional_counts[3:]}",
            )

            invalid_values = int(scalar(
                cursor,
                sql.SQL(
                    """
                    SELECT count(*) FROM {}.{}
                    WHERE temperature_2m_max_c NOT BETWEEN 5 AND 50
                       OR wet_bulb_temperature_2m_max_c NOT BETWEEN 0 AND 40
                       OR relative_humidity_2m_max_pct NOT BETWEEN 0 AND 100
                       OR precipitation_sum_mm < 0
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(daily)),
            ))
            require(invalid_values == 0, f"Out-of-range daily values={invalid_values}")

            incomplete_months = int(scalar(
                cursor,
                sql.SQL("SELECT count(*) FROM {}.{} WHERE NOT complete_month").format(
                    sql.Identifier(args.schema), sql.Identifier(monthly)
                ),
            ))
            require(incomplete_months == 0, f"Incomplete months={incomplete_months}")

            cursor.execute(
                sql.SQL(
                    """
                    SELECT min(temperature_2m_max_c), max(temperature_2m_max_c),
                           min(wet_bulb_temperature_2m_max_c),
                           max(wet_bulb_temperature_2m_max_c),
                           min(observation_date), max(observation_date)
                    FROM {}.{}
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(daily))
            )
            ranges = cursor.fetchone()
            cursor.execute(
                sql.SQL(
                    """
                    SELECT count(*), count(DISTINCT admin2_code),
                           min(month_start), max(month_start),
                           min(days_wbmax_above_local_month_p95),
                           max(days_wbmax_above_local_month_p95)
                    FROM {}.{}
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(monthly))
            )
            monthly_summary = cursor.fetchone()
            control_coverage = scalar(
                cursor,
                sql.SQL(
                    "SELECT round(avg((precipitation_sum_mm IS NOT NULL)::int)::numeric, 4) "
                    "FROM {}.{}"
                ).format(sql.Identifier(args.schema), sql.Identifier(daily)),
            )
            coverage_mode = (
                "full_calendar" if args.full_calendar
                else "full_survey_years" if full_year_mode
                else "survey_months"
            )
            print(f"coverage_mode={coverage_mode}")
            print(f"validated={daily} rows={daily_rows} admin2={n_admin2}")
            print(f"optional_daily_non_null_counts={optional_counts}")
            print(f"daily_ranges={ranges}")
            print(f"precipitation_non_null_rate={control_coverage}")
            print(f"validated={monthly} summary={monthly_summary}")
            print("validation=passed")


if __name__ == "__main__":
    main()
