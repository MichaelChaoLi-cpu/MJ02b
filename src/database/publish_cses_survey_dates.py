#!/usr/bin/env python3
"""Atomically publish the CSES survey-date layer and enrich ``final_HH_CSES``.

This publisher is deliberately scoped. It creates/replaces only the dedicated
survey-date ``final_*``, ``ind_que_*``, and ``align_summary_*`` tables, and it
adds/refreshes three actual-date components in the existing HH table and its
dictionary/summary. It does not touch climate or research-analysis tables.
"""

from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from build_cses_survey_dates import VARIABLES
from cses_hh_hl_common import snake_case
from cses_survey_date_contract import (
    ACTUAL_DATE_COLUMNS,
    EXPECTED_EXACT_COVERAGE,
    EXPECTED_ROWS,
    FINAL_SURVEY_DATE_TYPES,
    IND_QUE_TYPES,
    SURVEY_DATE_SUMMARY_TYPES,
    hh_dictionary_extension,
    hh_summary_extension,
)


FINAL_TABLE = "final_SURVEY_DATE_CSES"
DICTIONARY_TABLE = "ind_que_SURVEY_DATE_CSES"
SUMMARY_TABLE = "align_summary_SURVEY_DATE_CSES"
TARGET_TABLES = [FINAL_TABLE, DICTIONARY_TABLE, SUMMARY_TABLE]


@dataclass(frozen=True)
class TableSpec:
    name: str
    frame: pd.DataFrame
    columns: list[tuple[str, str]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def database_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [snake_case(str(column)) for column in result.columns]
    return result


def python_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_and_validate(root: Path) -> list[TableSpec]:
    output = root / "data" / "exp" / "database"
    final = database_frame(pd.read_parquet(output / f"{FINAL_TABLE}.parquet"))
    expected_final_columns = [name for name, _ in FINAL_SURVEY_DATE_TYPES]
    require(
        list(final.columns) == expected_final_columns,
        f"{FINAL_TABLE} columns differ from the publication contract",
    )
    require(len(final) == EXPECTED_ROWS, f"{FINAL_TABLE} rows={len(final)}")
    require(
        not final.duplicated(["survey_wave", "household_id"]).any(),
        f"{FINAL_TABLE} contains duplicate household keys",
    )

    component_nonnull = final[ACTUAL_DATE_COLUMNS].notna()
    require(
        (component_nonnull.nunique(axis=1) == 1).all(),
        "Actual survey-date components are partially populated",
    )
    constructed = pd.Series(pd.NaT, index=final.index, dtype="datetime64[ns]")
    complete = component_nonnull.all(axis=1)
    parts = final.loc[complete, ACTUAL_DATE_COLUMNS].rename(
        columns={
            "survey_actual_year": "year",
            "survey_actual_month": "month",
            "survey_actual_day": "day",
        }
    )
    constructed.loc[complete] = pd.to_datetime(parts.astype("int64"), errors="raise")
    candidate = pd.to_datetime(final["candidate_reference_date"])
    require(
        (constructed.eq(candidate) | (constructed.isna() & candidate.isna())).all(),
        "Actual survey-date components do not reproduce Candidate Reference Date",
    )
    coverage = (
        final.loc[candidate.notna()]
        .groupby("survey_wave", observed=True)
        .size()
        .astype(int)
        .to_dict()
    )
    require(
        coverage == EXPECTED_EXACT_COVERAGE,
        f"Unexpected exact-date coverage: {coverage}",
    )
    unsupported = ~final["survey_wave"].isin(EXPECTED_EXACT_COVERAGE)
    require(
        final.loc[unsupported, ACTUAL_DATE_COLUMNS].isna().all().all(),
        "An unsupported wave has an assigned actual date",
    )

    hh_keys = database_frame(
        pd.read_parquet(
            output / "final_HH_CSES.parquet",
            columns=["Survey Wave", "Household ID"],
        )
    )
    key_check = hh_keys.merge(
        final[["survey_wave", "household_id"]],
        on=["survey_wave", "household_id"],
        how="outer",
        indicator=True,
        validate="1:1",
    )
    require(
        key_check["_merge"].eq("both").all(),
        "Survey-date and HH household keys differ",
    )

    raw_dictionary = pd.read_csv(
        output / f"{DICTIONARY_TABLE}.csv", dtype="string"
    )
    dictionary = pd.DataFrame(
        {
            "canonical_varname": raw_dictionary["varname"].map(snake_case),
            "dataset_name": raw_dictionary["dataset_name"],
            "column_in_raw_sav": raw_dictionary["column_in_raw_sav"],
            "column_label_in_english": raw_dictionary["readable_name"],
            "source_kind": raw_dictionary["source_kind"],
            "measure_type": raw_dictionary["measure_type"],
            "canonical_text": raw_dictionary["canonical_text"],
        }
    )
    require(
        set(dictionary["canonical_varname"]) == set(expected_final_columns),
        "Survey-date dictionary does not cover every final-table column",
    )

    summary = database_frame(pd.read_csv(output / f"{SUMMARY_TABLE}.csv"))
    for date_column in ["candidate_minimum_date", "candidate_maximum_date"]:
        summary[date_column] = pd.to_datetime(summary[date_column], errors="coerce")
    require(len(summary) == 10, f"{SUMMARY_TABLE} must contain ten wave rows")
    require(
        set(summary["survey_wave"].astype(str))
        == {"2004", "2007", "2009", "2011-12", "2013", "2014", "2016", "2017", "2019", "2021"},
        "Survey-date summary wave coverage differs",
    )

    return [
        TableSpec(FINAL_TABLE, final, FINAL_SURVEY_DATE_TYPES),
        TableSpec(DICTIONARY_TABLE, dictionary, IND_QUE_TYPES),
        TableSpec(SUMMARY_TABLE, summary, SURVEY_DATE_SUMMARY_TYPES),
    ]


def table_exists(cursor: psycopg.Cursor, schema: str, table: str) -> bool:
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname=%s AND tablename=%s)",
        (schema, table),
    )
    return bool(cursor.fetchone()[0])


def create_table(
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
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


def copy_frame(
    cursor: psycopg.Cursor, schema: str, table: str, spec: TableSpec
) -> None:
    expected = [name for name, _ in spec.columns]
    require(
        list(spec.frame.columns) == expected,
        f"{spec.name} columns do not match COPY order",
    )
    statement = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, expected)),
    )
    with cursor.copy(statement) as copy:
        for row in spec.frame.itertuples(index=False, name=None):
            copy.write_row(tuple(python_value(value) for value in row))


def scalar(
    cursor: psycopg.Cursor, statement: object, parameters: tuple[object, ...] = ()
) -> object:
    cursor.execute(statement, parameters)
    return cursor.fetchone()[0]


def add_comments(cursor: psycopg.Cursor, schema: str) -> None:
    cursor.execute(
        sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
            sql.Identifier(schema),
            sql.Identifier(FINAL_TABLE),
            sql.Literal(
                "Household-grain CSES survey timing audit. The selected actual date "
                "is the explicit interview date in 2004 and explicit last-visit date "
                "in 2019/2021; unsupported waves remain null."
            ),
        )
    )
    for readable_name, (label, _measure, description) in VARIABLES.items():
        cursor.execute(
            sql.SQL("COMMENT ON COLUMN {}.{}.{} IS {}").format(
                sql.Identifier(schema),
                sql.Identifier(FINAL_TABLE),
                sql.Identifier(snake_case(readable_name)),
                sql.Literal(f"{label}. {description}"),
            )
        )
    actual_comments = {
        "survey_actual_year": "Calendar year of the selected explicit household survey date; null when unavailable.",
        "survey_actual_month": "Calendar month of the selected explicit household survey date; null when unavailable.",
        "survey_actual_day": "Calendar day of the selected explicit household survey date; null when unavailable.",
    }
    for column, comment in actual_comments.items():
        cursor.execute(
            sql.SQL("COMMENT ON COLUMN {}.{}.{} IS {}").format(
                sql.Identifier(schema),
                sql.Identifier("final_HH_CSES"),
                sql.Identifier(column),
                sql.Literal(comment),
            )
        )


def validate_staged_final(cursor: psycopg.Cursor, schema: str, table: str) -> None:
    rows = int(
        scalar(
            cursor,
            sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(schema), sql.Identifier(table)
            ),
        )
    )
    require(rows == EXPECTED_ROWS, f"Staged survey-date rows={rows}")
    duplicates = int(
        scalar(
            cursor,
            sql.SQL(
                "SELECT count(*) FROM (SELECT survey_wave, household_id FROM {}.{} "
                "GROUP BY 1,2 HAVING count(*) > 1) duplicate_keys"
            ).format(sql.Identifier(schema), sql.Identifier(table)),
        )
    )
    require(duplicates == 0, f"Staged survey-date duplicate keys={duplicates}")
    invalid = int(
        scalar(
            cursor,
            sql.SQL(
                "SELECT count(*) FROM {}.{} WHERE "
                "(survey_actual_year IS NULL) <> (survey_actual_month IS NULL) OR "
                "(survey_actual_year IS NULL) <> (survey_actual_day IS NULL) OR "
                "(survey_actual_month IS NOT NULL AND survey_actual_month NOT BETWEEN 1 AND 12) OR "
                "(survey_actual_day IS NOT NULL AND survey_actual_day NOT BETWEEN 1 AND 31) OR "
                "(survey_actual_year IS NOT NULL AND "
                " make_date(survey_actual_year, survey_actual_month, survey_actual_day) "
                " IS DISTINCT FROM candidate_reference_date)"
            ).format(sql.Identifier(schema), sql.Identifier(table)),
        )
    )
    require(invalid == 0, f"Staged survey-date component violations={invalid}")


def publish(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    specs = load_and_validate(root)
    exact_total = sum(EXPECTED_EXACT_COVERAGE.values())

    with psycopg.connect(
        host=args.host, port=args.port, dbname=args.dbname
    ) as connection:
        with connection.cursor() as cursor:
            for required in [
                "final_HH_CSES", "ind_que_HH_CSES",
                "align_summary_HH_CSES", "_catalog",
            ]:
                require(
                    table_exists(cursor, args.schema, required),
                    f"Required table is missing: {args.schema}.{required}",
                )

            existing = [
                table for table in TARGET_TABLES
                if table_exists(cursor, args.schema, table)
            ]
            if existing and not args.replace:
                raise RuntimeError(
                    f"Refusing to overwrite existing tables without --replace: {existing}"
                )

            hh_rows_before = int(
                scalar(
                    cursor,
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES")
                    ),
                )
            )
            require(hh_rows_before == EXPECTED_ROWS, f"Unexpected HH rows={hh_rows_before}")

            staging: dict[str, str] = {}
            for spec in specs:
                stage = f"_staging_{spec.name}_{uuid.uuid4().hex[:8]}"
                staging[spec.name] = stage
                create_table(cursor, args.schema, stage, spec.columns)
                copy_frame(cursor, args.schema, stage, spec)
                copied = int(
                    scalar(
                        cursor,
                        sql.SQL("SELECT count(*) FROM {}.{}").format(
                            sql.Identifier(args.schema), sql.Identifier(stage)
                        ),
                    )
                )
                require(copied == len(spec.frame), f"COPY mismatch for {spec.name}")
            validate_staged_final(cursor, args.schema, staging[FINAL_TABLE])

            if args.replace:
                cursor.execute(
                    sql.SQL("DELETE FROM {}.{} WHERE table_name = ANY(%s)").format(
                        sql.Identifier(args.schema), sql.Identifier("_catalog")
                    ),
                    (TARGET_TABLES,),
                )
                for table in reversed(TARGET_TABLES):
                    cursor.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                            sql.Identifier(args.schema), sql.Identifier(table)
                        )
                    )

            for spec in specs:
                cursor.execute(
                    sql.SQL("ALTER TABLE {}.{} RENAME TO {}").format(
                        sql.Identifier(args.schema),
                        sql.Identifier(staging[spec.name]),
                        sql.Identifier(spec.name),
                    )
                )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, household_id)").format(
                    sql.Identifier("uq_final_SURVEY_DATE_CSES_wave_household"),
                    sql.Identifier(args.schema),
                    sql.Identifier(FINAL_TABLE),
                )
            )

            for column in ACTUAL_DATE_COLUMNS:
                cursor.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS {} smallint").format(
                        sql.Identifier(args.schema),
                        sql.Identifier("final_HH_CSES"),
                        sql.Identifier(column),
                    )
                )
            cursor.execute(
                sql.SQL(
                    "UPDATE {}.{} SET survey_actual_year=NULL, "
                    "survey_actual_month=NULL, survey_actual_day=NULL"
                ).format(
                    sql.Identifier(args.schema), sql.Identifier("final_HH_CSES")
                )
            )
            cursor.execute(
                sql.SQL(
                    "UPDATE {}.{} hh SET "
                    "survey_actual_year=d.survey_actual_year, "
                    "survey_actual_month=d.survey_actual_month, "
                    "survey_actual_day=d.survey_actual_day "
                    "FROM {}.{} d WHERE hh.survey_wave=d.survey_wave "
                    "AND hh.household_id=d.household_id"
                ).format(
                    sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                    sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE),
                )
            )
            require(
                cursor.rowcount == EXPECTED_ROWS,
                f"HH date join updated {cursor.rowcount} rows",
            )

            hh_dictionary_rows = hh_dictionary_extension()
            cursor.execute(
                sql.SQL("DELETE FROM {}.{} WHERE canonical_varname = ANY(%s)").format(
                    sql.Identifier(args.schema), sql.Identifier("ind_que_HH_CSES")
                ),
                (ACTUAL_DATE_COLUMNS,),
            )
            cursor.executemany(
                sql.SQL(
                    "INSERT INTO {}.{} (canonical_varname, dataset_name, "
                    "column_in_raw_sav, column_label_in_english, source_kind, "
                    "measure_type, canonical_text) VALUES (%s,%s,%s,%s,%s,%s,%s)"
                ).format(
                    sql.Identifier(args.schema), sql.Identifier("ind_que_HH_CSES")
                ),
                [
                    tuple(python_value(value) for value in row)
                    for row in hh_dictionary_rows.itertuples(index=False, name=None)
                ],
            )
            hh_summary_rows = hh_summary_extension()
            cursor.execute(
                sql.SQL("DELETE FROM {}.{} WHERE varname = ANY(%s)").format(
                    sql.Identifier(args.schema), sql.Identifier("align_summary_HH_CSES")
                ),
                (ACTUAL_DATE_COLUMNS,),
            )
            cursor.executemany(
                sql.SQL(
                    "INSERT INTO {}.{} (varname, dataset_count, source_count, "
                    "explicit_count, derived_count, measure_type, canonical_text) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)"
                ).format(
                    sql.Identifier(args.schema), sql.Identifier("align_summary_HH_CSES")
                ),
                [
                    tuple(python_value(value) for value in row)
                    for row in hh_summary_rows.itertuples(index=False, name=None)
                ],
            )

            add_comments(cursor, args.schema)
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {}.{} "
                    "(table_name, module, grain, join_keys, ind_que_table, "
                    "description, caveats, row_count, n_datasets, updated_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,CURRENT_DATE)"
                ).format(sql.Identifier(args.schema), sql.Identifier("_catalog")),
                (
                    FINAL_TABLE,
                    "SURVEY_DATE",
                    "one row per CSES household",
                    "survey_wave + household_id",
                    DICTIONARY_TABLE,
                    "CSES household survey-date fields, selected exact date, precision, source provenance, and timing quality flags.",
                    "Exact household dates are defensible only for 2004, 2019, and 2021. The selected role is interview date in 2004 and last-visit date in 2019/2021; other waves remain null. Eight 2004 interview dates are flagged outside the documented fieldwork period and are retained, not silently deleted.",
                    EXPECTED_ROWS,
                    10,
                ),
            )

            hh_rows_after = int(
                scalar(
                    cursor,
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES")
                    ),
                )
            )
            require(hh_rows_after == hh_rows_before, "HH row count changed")
            actual_count = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} WHERE survey_actual_year IS NOT NULL"
                    ).format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES")
                    ),
                )
            )
            require(actual_count == exact_total, f"HH actual-date rows={actual_count}")
            mismatches = int(
                scalar(
                    cursor,
                    sql.SQL(
                        "SELECT count(*) FROM {}.{} hh JOIN {}.{} d "
                        "USING (survey_wave, household_id) WHERE "
                        "hh.survey_actual_year IS DISTINCT FROM d.survey_actual_year OR "
                        "hh.survey_actual_month IS DISTINCT FROM d.survey_actual_month OR "
                        "hh.survey_actual_day IS DISTINCT FROM d.survey_actual_day"
                    ).format(
                        sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                        sql.Identifier(args.schema), sql.Identifier(FINAL_TABLE),
                    ),
                )
            )
            require(mismatches == 0, f"HH-to-date component mismatches={mismatches}")

        connection.commit()

    print(f"published={args.schema}.{FINAL_TABLE} rows={EXPECTED_ROWS}")
    print(f"published={args.schema}.{DICTIONARY_TABLE} rows={len(specs[1].frame)}")
    print(f"published={args.schema}.{SUMMARY_TABLE} rows={len(specs[2].frame)}")
    print(f"updated={args.schema}.final_HH_CSES actual_date_rows={exact_total}")
    print("exact_date_coverage=" + ", ".join(
        f"{wave}:{count}" for wave, count in EXPECTED_EXACT_COVERAGE.items()
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace existing survey-date target tables.",
    )
    publish(parser.parse_args())


if __name__ == "__main__":
    main()
