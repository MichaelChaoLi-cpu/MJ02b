#!/usr/bin/env python3
"""Publish persisted heat-labor baseline result CSVs to PostgreSQL.

This write-capable infrastructure script is intentionally isolated from
``src/analyses``. It is never called by analysis, figure, or table scripts and
requires an explicit confirmation token before opening a write transaction.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/exp/experiments/heat-labor-baseline"
CONFIRMATION_TOKEN = "WRITE_MDA_PUBLIC"
ESTIMATES_TABLE = "final_HEAT_LABOR_ESTIMATES"
SUMMARY_TABLE = "final_HEAT_LABOR_SAMPLE_SUMMARY"

ESTIMATE_TYPES = [
    ("experiment_status", "text"),
    ("specification", "text"),
    ("outcome", "text"),
    ("exposure", "text"),
    ("term", "text"),
    ("sample", "text"),
    ("coefficient", "double precision"),
    ("standard_error", "double precision"),
    ("ci_lower_95", "double precision"),
    ("ci_upper_95", "double precision"),
    ("p_value", "double precision"),
    ("n_obs", "integer"),
    ("n_clusters", "integer"),
    ("weighted_outcome_mean", "double precision"),
    ("exposure_unit", "text"),
    ("fixed_effects", "text"),
    ("covariates", "text"),
    ("variance", "text"),
    ("interpretation_limit", "text"),
]
SUMMARY_TYPES = [
    ("survey_wave", "text"),
    ("survey_year", "smallint"),
    ("person_rows", "integer"),
    ("admin2_clusters", "integer"),
    ("worked_observed", "integer"),
    ("inclusive_hours_observed", "integer"),
    ("wage_observed", "integer"),
    ("weighted_work_rate", "double precision"),
    ("weighted_weekly_hours", "double precision"),
    ("mean_wb26_days", "double precision"),
    ("mean_precipitation_mm", "double precision"),
    ("coastal_fallback_rows", "integer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--confirm-write", required=True)
    return parser.parse_args()


def load_inputs(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates = pd.read_csv(directory / "provisional_estimates.csv")
    summary = pd.read_csv(directory / "sample_summary.csv")
    expected_estimates = [name for name, _type in ESTIMATE_TYPES]
    expected_summary = [name for name, _type in SUMMARY_TYPES]
    if estimates.columns.tolist() != expected_estimates:
        raise AssertionError("Provisional-estimate columns do not match publish contract")
    if summary.columns.tolist() != expected_summary:
        raise AssertionError("Sample-summary columns do not match publish contract")
    if estimates.empty or summary.empty:
        raise AssertionError("Refusing to publish empty result artifacts")
    return estimates, summary


def create_and_copy(
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
) -> None:
    definitions = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(data_type))
        for name, data_type in columns
    )
    cursor.execute(
        sql.SQL("CREATE TABLE {}.{} ({})").format(
            sql.Identifier(schema), sql.Identifier(table), definitions
        )
    )
    names = [name for name, _type in columns]
    statement = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, names)),
    )
    with cursor.copy(statement) as copy:
        for row in frame[names].itertuples(index=False, name=None):
            copy.write_row(
                tuple(
                    None
                    if pd.isna(value)
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                    for value in row
                )
            )


def publish(
    connection: psycopg.Connection,
    schema: str,
    estimates: pd.DataFrame,
    summary: pd.DataFrame,
    replace: bool,
) -> None:
    targets = [ESTIMATES_TABLE, SUMMARY_TABLE]
    frames = {ESTIMATES_TABLE: estimates, SUMMARY_TABLE: summary}
    types = {ESTIMATES_TABLE: ESTIMATE_TYPES, SUMMARY_TABLE: SUMMARY_TYPES}
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=%s AND table_name=ANY(%s)",
            (schema, targets),
        )
        existing = [row[0] for row in cursor.fetchall()]
        if existing and not replace:
            raise RuntimeError(f"Refusing to overwrite without --replace: {existing}")
        stages = {
            target: f"_staging_{target}_{uuid.uuid4().hex[:8]}" for target in targets
        }
        for target in targets:
            create_and_copy(cursor, schema, stages[target], frames[target], types[target])
        if replace:
            for target in reversed(targets):
                cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                        sql.Identifier(schema), sql.Identifier(target)
                    )
                )
        for target in targets:
            cursor.execute(
                sql.SQL("ALTER TABLE {}.{} RENAME TO {}").format(
                    sql.Identifier(schema),
                    sql.Identifier(stages[target]),
                    sql.Identifier(target),
                )
            )
    connection.commit()


def main() -> None:
    args = parse_args()
    if args.confirm_write != CONFIRMATION_TOKEN:
        raise RuntimeError(
            f"Write confirmation rejected; pass --confirm-write {CONFIRMATION_TOKEN}"
        )
    input_directory = args.input if args.input.is_absolute() else ROOT / args.input
    estimates, summary = load_inputs(input_directory)
    connection_values: dict[str, object] = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
    }
    if args.user:
        connection_values["user"] = args.user
    if args.password:
        connection_values["password"] = args.password
    with psycopg.connect(**connection_values) as connection:
        publish(connection, args.schema, estimates, summary, args.replace)
    print(f"published={args.schema}.{ESTIMATES_TABLE} rows={len(estimates)}")
    print(f"published={args.schema}.{SUMMARY_TABLE} rows={len(summary)}")


if __name__ == "__main__":
    main()
