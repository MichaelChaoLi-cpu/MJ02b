#!/usr/bin/env python3
"""Publish the database-only CSES humid-heat/labor analytic layer."""

from __future__ import annotations

import argparse
import uuid

import pandas as pd
import psycopg
from psycopg import sql


TARGET = "final_HEAT_LABOR_ANALYTIC"
DICTIONARY = "ind_que_HEAT_LABOR_ANALYTIC"
SUMMARY = "align_summary_HEAT_LABOR_ANALYTIC"
TARGETS = [TARGET, DICTIONARY, SUMMARY]

IND_QUE_TYPES = [
    ("canonical_varname", "text"),
    ("dataset_name", "text"),
    ("column_in_raw_sav", "text"),
    ("column_label_in_english", "text"),
    ("source_kind", "text"),
    ("measure_type", "text"),
    ("canonical_text", "text"),
]
ALIGN_TYPES = [
    ("varname", "text"),
    ("dataset_count", "integer"),
    ("source_count", "integer"),
    ("explicit_count", "integer"),
    ("derived_count", "integer"),
    ("measure_type", "text"),
    ("canonical_text", "text"),
]

VARIABLES = [
    ("survey_wave", "CSES EC", "survey_wave", "identifier", "CSES survey-wave label."),
    ("survey_year", "CSES EC", "survey_year", "time", "CSES reference year."),
    ("survey_month", "CSES EC", "survey_month", "time", "Aligned CSES survey month; raw visit/interview fields are not represented in this analytic table."),
    ("person_id", "CSES EC", "person_id", "identifier", "Within-wave unique CSES person identifier."),
    ("household_id", "CSES EC", "household_id", "identifier", "Within-wave unique CSES household identifier."),
    ("admin2_code", "CSES geography", "admin2_code", "geography", "Current-boundary matched district/municipality code."),
    ("sex", "CSES EC", "sex", "categorical", "Released sex code inherited from the household roster."),
    ("age", "CSES EC", "age", "continuous", "Age in completed years."),
    ("urban_rural", "CSES EC", "urban_rural", "categorical", "Released urban-rural residence stratum."),
    ("analysis_weight", "CSES EC", "person_weight", "survey weight", "Positive person weight, falling back to household weight when person weight is unavailable."),
    ("worked_past_week", "CSES EC", "worked_at_least_one_hour_past_7_days", "binary outcome", "1 when the person worked at least one hour in the past seven days; 0 otherwise."),
    ("weekly_hours_worked", "CSES EC", "total_hours_worked_past_7_days", "continuous outcome", "Total hours worked in the past seven days."),
    ("weekly_hours_including_zero", "analytic", "derived", "continuous outcome", "Weekly hours set to zero for people reporting no work; retained as missing for workers whose hours are unavailable."),
    ("monthly_salary_wages_riel", "CSES EC", "monthly_salary_wages_riel", "continuous outcome", "Released nominal monthly salary or wages in Cambodian riel."),
    ("log_monthly_salary_wages", "analytic", "derived", "continuous outcome", "Natural log of one plus nonnegative monthly salary or wages."),
    ("days_wbmax_ge_26c", "ERA5-Land", "days_wbmax_ge_26c", "heat exposure", "Days in the survey month with daily maximum wet-bulb temperature at least 26 C."),
    ("days_wbmax_ge_28c", "ERA5-Land", "days_wbmax_ge_28c", "heat exposure", "Days in the survey month with daily maximum wet-bulb temperature at least 28 C."),
    ("wet_bulb_max_mean_c", "ERA5-Land", "wet_bulb_max_mean_c", "heat exposure", "Survey-month mean of daily maximum wet-bulb temperature in C."),
    ("wet_bulb_month_max_c", "ERA5-Land", "wet_bulb_month_max_c", "heat exposure", "Maximum daily wet-bulb temperature during the survey month in C."),
    ("temperature_2m_max_mean_c", "ERA5-Land", "temperature_2m_max_mean_c", "heat exposure", "Survey-month mean of daily maximum 2 m air temperature in C."),
    ("temperature_2m_mean_c", "ERA5-Land", "temperature_2m_mean_c", "weather control", "Survey-month mean 2 m air temperature in C."),
    ("dew_point_2m_max_mean_c", "ERA5-Land", "dew_point_2m_max_mean_c", "weather control", "Survey-month mean of daily maximum 2 m dew-point temperature in C."),
    ("relative_humidity_2m_max_mean_pct", "ERA5-Land", "relative_humidity_2m_max_mean_pct", "weather control", "Survey-month mean of daily maximum 2 m relative humidity in percent."),
    ("apparent_temperature_max_mean_c", "ERA5", "apparent_temperature_max_mean_c", "weather control", "Survey-month mean of daily maximum apparent temperature in C."),
    ("apparent_temperature_month_max_c", "ERA5", "apparent_temperature_month_max_c", "weather control", "Maximum daily apparent temperature in the survey month in C."),
    ("precipitation_month_sum_mm", "ERA5", "precipitation_month_sum_mm", "weather control", "Total precipitation in the survey month in millimeters."),
    ("lag_days_wbmax_ge_26c", "ERA5-Land", "derived", "lagged exposure", "Days with WBmax at least 26 C in the previous calendar month when available."),
    ("lead_days_wbmax_ge_26c", "ERA5-Land", "derived", "placebo exposure", "Days with WBmax at least 26 C in the next calendar month when available."),
    ("monthly_electricity_expense_riel", "CSES HO", "monthly_electricity_expense_riel", "adaptation candidate", "Released nominal monthly electricity expense; not interpreted as electricity access."),
    ("electricity_spending_positive", "analytic", "derived", "adaptation candidate", "1 when released monthly electricity expense is positive; a spending proxy, not an access measure."),
    ("roof_material_source_code", "CSES HO", "roof_material_source_code", "housing candidate", "Wave-specific roof-material source code; not harmonized across detailed categories."),
    ("wall_material_source_code", "CSES HO", "wall_material_source_code", "housing candidate", "Wave-specific wall-material source code; not harmonized across detailed categories."),
    ("floor_area_square_meters", "CSES HO", "floor_area_square_meters", "housing candidate", "Occupied dwelling floor area in square meters."),
    ("has_toilet_facility", "CSES HO", "has_toilet_facility", "housing candidate", "Harmonized indicator for a reported toilet facility."),
    ("treats_drinking_water", "CSES HO", "treats_drinking_water", "water candidate", "Harmonized indicator for always or sometimes treating drinking water."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def connection_args(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {
        "host": args.host, "port": args.port, "dbname": args.dbname
    }
    if args.user:
        values["user"] = args.user
    if args.password:
        values["password"] = args.password
    return values


def table_exists(cursor: psycopg.Cursor, schema: str, table: str) -> bool:
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname=%s AND tablename=%s)",
        (schema, table),
    )
    return bool(cursor.fetchone()[0])


def create_dictionary_frames(n_waves: int, n_sources: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dictionary = pd.DataFrame(
        [
            {
                "canonical_varname": name,
                "dataset_name": dataset,
                "column_in_raw_sav": source,
                "column_label_in_english": description,
                "source_kind": "derived" if source == "derived" else "explicit",
                "measure_type": measure,
                "canonical_text": description,
            }
            for name, dataset, source, measure, description in VARIABLES
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "varname": name,
                "dataset_count": n_waves,
                "source_count": n_sources,
                "explicit_count": 0 if source == "derived" else n_sources,
                "derived_count": n_sources if source == "derived" else 0,
                "measure_type": measure,
                "canonical_text": description,
            }
            for name, _dataset, source, measure, description in VARIABLES
        ]
    )
    return dictionary, summary


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
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
) -> None:
    names = [name for name, _ in columns]
    statement = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema), sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, names)),
    )
    with cursor.copy(statement) as copy:
        for row in frame[names].itertuples(index=False, name=None):
            copy.write_row(tuple(None if pd.isna(value) else value for value in row))


def create_analytic(cursor: psycopg.Cursor, schema: str, table: str) -> None:
    cursor.execute(
        sql.SQL(
            """
            CREATE TABLE {}.{} AS
            SELECT
                e.survey_wave,
                e.survey_year,
                e.survey_month,
                make_date(e.survey_year, e.survey_month, 1) AS interview_month_start,
                e.person_id,
                e.household_id,
                e.psu,
                g.admin2_code,
                g.admin3_code,
                g.geography_match_level,
                c.temperature_location_fallback_used,
                e.sex,
                e.age,
                (e.age::integer * e.age::integer) AS age_squared,
                e.urban_rural,
                hh.household_member_count,
                hh.household_head_education_level,
                e.person_weight,
                e.household_weight,
                coalesce(nullif(e.person_weight, 0), nullif(e.household_weight, 0))
                    AS analysis_weight,
                e.worked_at_least_one_hour_past_7_days AS worked_past_week,
                e.total_hours_worked_past_7_days::double precision AS weekly_hours_worked,
                CASE
                    WHEN e.worked_at_least_one_hour_past_7_days = 0 THEN 0::double precision
                    WHEN e.worked_at_least_one_hour_past_7_days = 1
                        THEN e.total_hours_worked_past_7_days::double precision
                END AS weekly_hours_including_zero,
                e.monthly_salary_wages_riel,
                CASE WHEN e.monthly_salary_wages_riel >= 0
                     THEN ln(1 + e.monthly_salary_wages_riel) END
                    AS log_monthly_salary_wages,
                c.days_wbmax_ge_26c,
                c.days_wbmax_ge_28c,
                c.days_wbmax_ge_30c,
                c.wet_bulb_max_mean_c,
                c.wet_bulb_month_max_c,
                c.temperature_2m_max_mean_c,
                c.temperature_2m_month_max_c,
                c.temperature_2m_mean_c,
                c.dew_point_2m_max_mean_c,
                c.relative_humidity_2m_max_mean_pct,
                c.apparent_temperature_max_mean_c,
                c.apparent_temperature_month_max_c,
                c.precipitation_month_sum_mm,
                c.days_tmax_ge_35c,
                c.days_tmax_ge_37c,
                lag_c.days_wbmax_ge_26c AS lag_days_wbmax_ge_26c,
                lag_c.wet_bulb_max_mean_c AS lag_wet_bulb_max_mean_c,
                lead_c.days_wbmax_ge_26c AS lead_days_wbmax_ge_26c,
                lead_c.wet_bulb_max_mean_c AS lead_wet_bulb_max_mean_c,
                ho.monthly_electricity_expense_riel,
                CASE WHEN ho.monthly_electricity_expense_riel IS NULL THEN NULL
                     WHEN ho.monthly_electricity_expense_riel > 0 THEN 1 ELSE 0 END::smallint
                    AS electricity_spending_positive,
                ho.monthly_water_charges_riel,
                ho.main_lighting_source_code,
                ho.main_drinking_water_source_code,
                ho.roof_material_source_code,
                ho.wall_material_source_code,
                ho.floor_material_source_code,
                ho.floor_area_square_meters,
                ho.rooms_used,
                ho.has_toilet_facility,
                ho.treats_drinking_water
            FROM {}.{} e
            JOIN {}.{} g USING (survey_wave, psu)
            JOIN {}.{} c
              ON c.admin2_code = g.admin2_code
             AND c.reference_year = e.survey_year
             AND c.calendar_month = e.survey_month
            LEFT JOIN {}.{} lag_c
              ON lag_c.admin2_code = g.admin2_code
             AND lag_c.month_start = make_date(e.survey_year, e.survey_month, 1) - interval '1 month'
            LEFT JOIN {}.{} lead_c
              ON lead_c.admin2_code = g.admin2_code
             AND lead_c.month_start = make_date(e.survey_year, e.survey_month, 1) + interval '1 month'
            LEFT JOIN {}.{} hh USING (survey_wave, household_id)
            LEFT JOIN {}.{} ho USING (survey_wave, household_id)
            WHERE e.age BETWEEN 15 AND 64
              AND e.survey_month BETWEEN 1 AND 12
              AND g.admin2_current_match
            """
        ).format(
            sql.Identifier(schema), sql.Identifier(table),
            sql.Identifier(schema), sql.Identifier("final_EC_CSES"),
            sql.Identifier(schema), sql.Identifier("dim_geo_CSES"),
            sql.Identifier(schema), sql.Identifier("final_CLIMATE_MONTHLY_ADMIN2"),
            sql.Identifier(schema), sql.Identifier("final_CLIMATE_MONTHLY_ADMIN2"),
            sql.Identifier(schema), sql.Identifier("final_CLIMATE_MONTHLY_ADMIN2"),
            sql.Identifier(schema), sql.Identifier("final_HH_CSES"),
            sql.Identifier(schema), sql.Identifier("final_HO_CSES"),
        )
    )


def main() -> None:
    args = parse_args()
    stages = {
        target: f"_staging_{target}_{uuid.uuid4().hex[:8]}" for target in TARGETS
    }
    with psycopg.connect(**connection_args(args)) as connection:
        with connection.cursor() as cursor:
            for required in [
                "final_EC_CSES", "final_HH_CSES", "final_HO_CSES",
                "dim_geo_CSES", "final_CLIMATE_MONTHLY_ADMIN2",
            ]:
                if not table_exists(cursor, args.schema, required):
                    raise RuntimeError(f"Required table is missing: {args.schema}.{required}")
            existing = [name for name in TARGETS if table_exists(cursor, args.schema, name)]
            if existing and not args.replace:
                raise RuntimeError(f"Refusing to overwrite without --replace: {existing}")

            create_analytic(cursor, args.schema, stages[TARGET])
            cursor.execute(
                sql.SQL("SELECT count(*), count(DISTINCT survey_wave) FROM {}.{}").format(
                    sql.Identifier(args.schema), sql.Identifier(stages[TARGET])
                )
            )
            row_count, n_waves = map(int, cursor.fetchone())
            if row_count == 0:
                raise RuntimeError("Analytic table is empty")
            dictionary, summary = create_dictionary_frames(n_waves, 5)
            create_table(cursor, args.schema, stages[DICTIONARY], IND_QUE_TYPES)
            create_table(cursor, args.schema, stages[SUMMARY], ALIGN_TYPES)
            copy_frame(cursor, args.schema, stages[DICTIONARY], dictionary, IND_QUE_TYPES)
            copy_frame(cursor, args.schema, stages[SUMMARY], summary, ALIGN_TYPES)

            if args.replace:
                cursor.execute(
                    sql.SQL("DELETE FROM {}.{} WHERE table_name = ANY(%s)").format(
                        sql.Identifier(args.schema), sql.Identifier("_catalog")
                    ),
                    (TARGETS,),
                )
                for target in reversed(TARGETS):
                    cursor.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                            sql.Identifier(args.schema), sql.Identifier(target)
                        )
                    )
            for target in TARGETS:
                cursor.execute(
                    sql.SQL("ALTER TABLE {}.{} RENAME TO {}").format(
                        sql.Identifier(args.schema), sql.Identifier(stages[target]),
                        sql.Identifier(target),
                    )
                )

            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, person_id)").format(
                    sql.Identifier("uq_final_HEAT_LABOR_ANALYTIC_wave_person"),
                    sql.Identifier(args.schema), sql.Identifier(TARGET)
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (admin2_code, survey_year, survey_month)").format(
                    sql.Identifier("ix_final_HEAT_LABOR_ANALYTIC_admin2_month"),
                    sql.Identifier(args.schema), sql.Identifier(TARGET)
                )
            )
            cursor.execute(
                sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                    sql.Identifier(args.schema), sql.Identifier(TARGET),
                    sql.Literal(
                        "Working-age CSES employment records linked entirely within PostgreSQL "
                        "to current admin2 geography, monthly ERA5-Land heat, household context, "
                        "ERA5 weather controls, and housing candidates. Climate coverage includes "
                        "nine complete survey years through 2021; exact interview day is unavailable."
                    ),
                )
            )

            catalog_rows = [
                (
                    TARGET, "HEAT_LABOR", "one row per working-age CSES EC person",
                    "survey_wave + person_id", DICTIONARY,
                    "Database-only working-age labor analytic layer with current, lagged, and placebo monthly humid-heat exposures.",
                    "Repeated cross-sections. Current published climate coverage includes 2007, 2009, 2011, 2013, 2014, 2016, 2017, 2019, and 2021. Exposure is district-month because exact interview day is unavailable; source-code housing categories are not yet harmonized.",
                    row_count, n_waves,
                ),
                (
                    DICTIONARY, "HEAT_LABOR", "one row per documented analytic variable",
                    "canonical_varname + dataset_name", None,
                    "Variable dictionary for the heat-labor analytic layer.",
                    "Original source details remain in the component CSES and climate dictionaries.",
                    len(dictionary), n_waves,
                ),
                (
                    SUMMARY, "HEAT_LABOR", "one row per aligned analytic variable",
                    "varname", DICTIONARY,
                    "Alignment summary for the heat-labor analytic layer.",
                    "Counts summarize source modules and current wave coverage.",
                    len(summary), n_waves,
                ),
            ]
            cursor.executemany(
                sql.SQL(
                    """
                    INSERT INTO {}.{}
                        (table_name, module, grain, join_keys, ind_que_table,
                         description, caveats, row_count, n_datasets, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier("_catalog")),
                catalog_rows,
            )
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT survey_wave, count(*), count(DISTINCT admin2_code),
                           count(worked_past_week), count(weekly_hours_worked),
                           count(log_monthly_salary_wages)
                    FROM {}.{} GROUP BY survey_wave ORDER BY min(survey_year)
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier(TARGET))
            )
            for row in cursor.fetchall():
                print("analytic_coverage=" + ",".join(map(str, row)))
            print(f"published={args.schema}.{TARGET} rows={row_count}")
            print(f"published={args.schema}.{DICTIONARY} rows={len(dictionary)}")
            print(f"published={args.schema}.{SUMMARY} rows={len(summary)}")


if __name__ == "__main__":
    main()
