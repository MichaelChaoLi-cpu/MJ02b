#!/usr/bin/env python3
"""Read-only validation of the published CSES household survey-date layer."""

from __future__ import annotations

import argparse

import psycopg
from psycopg import sql

from cses_survey_date_contract import (
    ACTUAL_DATE_COLUMNS,
    EXPECTED_EXACT_COVERAGE,
    EXPECTED_ROWS,
    FINAL_SURVEY_DATE_TYPES,
    IND_QUE_TYPES,
    SURVEY_DATE_SUMMARY_TYPES,
)


FINAL_TABLE = "final_SURVEY_DATE_CSES"
DICTIONARY_TABLE = "ind_que_SURVEY_DATE_CSES"
SUMMARY_TABLE = "align_summary_SURVEY_DATE_CSES"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scalar(
    cursor: psycopg.Cursor, statement: object, parameters: tuple[object, ...] = ()
) -> object:
    cursor.execute(statement, parameters)
    return cursor.fetchone()[0]


def assert_table_contract(
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
    expected_rows: int,
    expected_columns: list[tuple[str, str]],
) -> None:
    rows = int(
        scalar(
            cursor,
            sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(schema), sql.Identifier(table)
            ),
        )
    )
    require(rows == expected_rows, f"{table} rows={rows}, expected={expected_rows}")
    cursor.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    actual_columns = cursor.fetchall()
    require(
        actual_columns == expected_columns,
        f"{table} database columns/types differ from the contract",
    )


def validate(args: argparse.Namespace) -> None:
    exact_total = sum(EXPECTED_EXACT_COVERAGE.values())
    with psycopg.connect(
        host=args.host, port=args.port, dbname=args.dbname
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            assert_table_contract(
                cursor, args.schema, FINAL_TABLE, EXPECTED_ROWS,
                FINAL_SURVEY_DATE_TYPES,
            )
            assert_table_contract(
                cursor, args.schema, DICTIONARY_TABLE,
                len(FINAL_SURVEY_DATE_TYPES), IND_QUE_TYPES,
            )
            assert_table_contract(
                cursor, args.schema, SUMMARY_TABLE, 10,
                SURVEY_DATE_SUMMARY_TYPES,
            )

            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema=%s AND table_name=%s
                  AND column_name = ANY(%s)
                ORDER BY column_name
                """,
                (args.schema, "final_HH_CSES", ACTUAL_DATE_COLUMNS),
            )
            hh_components = cursor.fetchall()
            require(
                hh_components
                == sorted((column, "smallint") for column in ACTUAL_DATE_COLUMNS),
                "final_HH_CSES actual-date columns are missing or mistyped",
            )

            duplicate_keys = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM (SELECT survey_wave, household_id "
                        "FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1) d"
                    ).format(sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE)),
                )
            )
            require(duplicate_keys == 0, f"Survey-date duplicate keys={duplicate_keys}")

            cursor.execute(
                sql.SQL(
                    "SELECT survey_wave, count(*) FROM {}.{} "
                    "WHERE survey_actual_year IS NOT NULL GROUP BY 1 ORDER BY 1"
                ).format(sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE))
            )
            coverage = {str(wave): int(count) for wave, count in cursor.fetchall()}
            require(
                coverage == EXPECTED_EXACT_COVERAGE,
                f"Published exact-date coverage differs: {coverage}",
            )

            invalid_components = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} WHERE "
                        "(survey_actual_year IS NULL) <> (survey_actual_month IS NULL) OR "
                        "(survey_actual_year IS NULL) <> (survey_actual_day IS NULL) OR "
                        "(survey_actual_year IS NOT NULL AND "
                        " make_date(survey_actual_year, survey_actual_month, survey_actual_day) "
                        " IS DISTINCT FROM candidate_reference_date)"
                    ).format(sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE)),
                )
            )
            require(
                invalid_components == 0,
                f"Published survey-date component violations={invalid_components}",
            )

            unsupported_dates = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} WHERE survey_wave <> ALL(%s) "
                        "AND survey_actual_year IS NOT NULL"
                    ).format(sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE)),
                    (list(EXPECTED_EXACT_COVERAGE),),
                )
            )
            require(unsupported_dates == 0, f"Unsupported-wave dates={unsupported_dates}")

            hh_rows = int(
                scalar(
                    cursor,
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES")
                    ),
                )
            )
            require(hh_rows == EXPECTED_ROWS, f"HH rows={hh_rows}")
            hh_actual = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} WHERE survey_actual_year IS NOT NULL"
                    ).format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES")
                    ),
                )
            )
            require(hh_actual == exact_total, f"HH actual-date rows={hh_actual}")
            hh_mismatches = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} hh FULL JOIN {}.{} d "
                        "USING (survey_wave, household_id) WHERE "
                        "hh.household_id IS NULL OR d.household_id IS NULL OR "
                        "hh.survey_actual_year IS DISTINCT FROM d.survey_actual_year OR "
                        "hh.survey_actual_month IS DISTINCT FROM d.survey_actual_month OR "
                        "hh.survey_actual_day IS DISTINCT FROM d.survey_actual_day"
                    ).format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                        sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE),
                    ),
                )
            )
            require(hh_mismatches == 0, f"HH/date table mismatches={hh_mismatches}")

            hh_dictionary_rows = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} WHERE canonical_varname = ANY(%s)"
                    ).format(
                        sql.Identifier(args.schema), sql.Identifier("ind_que_HH_CSES")
                    ),
                    (ACTUAL_DATE_COLUMNS,),
                )
            )
            require(hh_dictionary_rows == 9, f"HH date dictionary rows={hh_dictionary_rows}")
            hh_summary_rows = int(
                scalar(
                    cursor,
                    sql.SQL("SELECT count(*) FROM {}.{} WHERE varname = ANY(%s)").format(
                        sql.Identifier(args.schema), sql.Identifier("align_summary_HH_CSES")
                    ),
                    (ACTUAL_DATE_COLUMNS,),
                )
            )
            require(hh_summary_rows == 3, f"HH date summary rows={hh_summary_rows}")

            catalog_rows = int(
                scalar(
                    cursor,
                    sql.SQL("SELECT count(*) FROM {}.{} WHERE table_name=%s").format(
                        sql.Identifier(args.schema), sql.Identifier("_catalog")
                    ),
                    (FINAL_TABLE,),
                )
            )
            require(catalog_rows == 1, f"Survey-date catalog rows={catalog_rows}")
            outside_2004 = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} WHERE survey_wave='2004' "
                        "AND candidate_date_within_documented_period=0"
                    ).format(sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE)),
                )
            )
            require(outside_2004 == 8, f"2004 documented-period flags={outside_2004}")

        connection.rollback()

    print(f"validated={args.schema}.{FINAL_TABLE} rows={EXPECTED_ROWS}")
    print(f"hh_actual_date_rows={exact_total}")
    print("exact_date_coverage=" + ", ".join(
        f"{wave}:{count}" for wave, count in EXPECTED_EXACT_COVERAGE.items()
    ))
    print("database_transaction=read_only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
