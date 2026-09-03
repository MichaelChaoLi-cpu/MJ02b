#!/usr/bin/env python3
"""Read-only PostgreSQL inventory of CSES date-like fields.

The connection forces ``default_transaction_read_only=on`` before querying any
catalog or data table.  Only aggregate inventories are written locally under
``data/exp/database``; PostgreSQL is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql


DATE_WORDS = (
    "date", "day", "month", "year", "visit", "interview",
    "survey", "change", "signout", "timestamp",
)


def read_only_connection(args: argparse.Namespace) -> psycopg.Connection:
    connection = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        options="-c default_transaction_read_only=on",
    )
    with connection.cursor() as cursor:
        cursor.execute("SHOW transaction_read_only")
        if cursor.fetchone()[0] != "on":
            connection.close()
            raise RuntimeError("PostgreSQL transaction is not read-only")
    return connection


def table_exists(
    connection: psycopg.Connection, schema: str, table: str
) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema=%s AND table_name=%s
            )
            """,
            (schema, table),
        )
        return bool(cursor.fetchone()[0])


def field_inventory(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    query = """
        SELECT table_name, ordinal_position, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema=%s AND upper(table_name) LIKE '%%CSES%%'
        ORDER BY table_name, ordinal_position
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (schema,))
        rows = cursor.fetchall()
    frame = pd.DataFrame(
        rows,
        columns=["table_name", "ordinal_position", "column_name", "data_type", "is_nullable"],
    )
    if frame.empty:
        return frame
    keep = frame["column_name"].str.lower().map(
        lambda name: any(word in name for word in DATE_WORDS)
    )
    result = frame.loc[keep].copy()
    result.insert(0, "table_schema", schema)
    result["audit_classification"] = result["column_name"].map(classify_column)
    return result.reset_index(drop=True)


def classify_column(name: str) -> str:
    low = name.lower()
    if low.startswith("survey_actual_") or low in {
        "interview_date", "first_visit_date", "last_visit_date", "reinterview_date"
    }:
        return "explicit_or_selected_household_visit_date"
    if low.startswith("village_reference_"):
        return "village_demographic_reference_not_household_visit"
    if "survey_month" in low or "survey_year" in low:
        return "survey_calendar_component"
    if "change" in low or "signout" in low or "timestamp" in low:
        return "operational_timestamp"
    return "other_date_like_field"


def exact_date_availability(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    table = "final_SURVEY_DATE_CSES"
    if not table_exists(connection, schema, table):
        return pd.DataFrame()
    statement = sql.SQL(
        """
        SELECT survey_wave,
               count(*) AS rows,
               count(*) FILTER (WHERE survey_actual_year IS NOT NULL) AS exact_date_rows,
               count(*) FILTER (WHERE interview_date IS NOT NULL) AS interview_date_rows,
               count(*) FILTER (WHERE first_visit_date IS NOT NULL) AS first_visit_date_rows,
               count(*) FILTER (WHERE last_visit_date IS NOT NULL) AS last_visit_date_rows
        FROM {}.{}
        GROUP BY survey_wave
        ORDER BY survey_wave
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))
    with connection.cursor() as cursor:
        cursor.execute(statement)
        rows = cursor.fetchall()
    return pd.DataFrame(
        rows,
        columns=[
            "survey_wave", "rows", "exact_date_rows", "interview_date_rows",
            "first_visit_date_rows", "last_visit_date_rows",
        ],
    )


def village_reference_summary(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    table = "final_VL_CSES"
    if not table_exists(connection, schema, table):
        return pd.DataFrame()
    statement = sql.SQL(
        """
        SELECT survey_wave,
               village_reference_year,
               count(*) AS village_rows,
               count(*) FILTER (
                   WHERE village_reference_month=survey_month
               ) AS reference_month_matches_survey_month
        FROM {}.{}
        WHERE village_reference_year IS NOT NULL
        GROUP BY survey_wave, village_reference_year
        ORDER BY survey_wave, village_reference_year
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))
    with connection.cursor() as cursor:
        cursor.execute(statement)
        rows = cursor.fetchall()
    frame = pd.DataFrame(
        rows,
        columns=[
            "survey_wave", "village_reference_year", "village_rows",
            "reference_month_matches_survey_month",
        ],
    )
    if not frame.empty:
        frame["audit_interpretation"] = (
            "Village demographic as-of date; historical years prove it is not a "
            "household interview date."
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    output = args.root.resolve() / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)

    with read_only_connection(args) as connection:
        fields = field_inventory(connection, args.schema)
        exact = exact_date_availability(connection, args.schema)
        village = village_reference_summary(connection, args.schema)

    fields.to_csv(output / "cses_postgres_date_field_inventory.csv", index=False)
    exact.to_csv(output / "cses_postgres_exact_date_availability.csv", index=False)
    village.to_csv(
        output / "cses_village_reference_date_postgres_summary.csv", index=False
    )
    print(f"date_like_cses_columns={len(fields)}")
    print(f"exact_date_wave_rows={len(exact)}")
    print(f"village_reference_summary_rows={len(village)}")
    print("postgresql_transaction_read_only=on")
    print("postgresql_writes=0")


if __name__ == "__main__":
    main()
