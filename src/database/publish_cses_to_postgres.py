#!/usr/bin/env python3
"""Publish validated CSES HH/HL/ED/HO/EC/VL database tables atomically."""

from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from cses_education import ED_VARIABLE_INFO
from cses_employment import EC_VARIABLE_INFO
from cses_hh_hl_common import VARIABLE_INFO, snake_case
from cses_housing import HO_VARIABLE_INFO
from cses_survey_date_contract import (
    ACTUAL_DATE_TYPES,
    enrich_hh_frame,
    extend_hh_dictionary,
    extend_hh_summary,
)
from cses_village import VL_VARIABLE_INFO
from validate_cses_ed import main as validate_ed_local_release
from validate_cses_ec import main as validate_ec_local_release
from validate_cses_hh_hl import main as validate_hh_hl_local_release
from validate_cses_ho import main as validate_ho_local_release
from validate_cses_vl import main as validate_vl_local_release


@dataclass(frozen=True)
class TableSpec:
    name: str
    frame: pd.DataFrame
    columns: list[tuple[str, str]]


FINAL_HH_TYPES = [
    ("dataset_name", "text"), ("survey_wave", "text"), ("survey_year", "smallint"),
    ("psu", "text"), ("household_id", "text"),
    ("household_member_count", "smallint"), ("male_member_count", "smallint"),
    ("female_member_count", "smallint"), ("child_member_count_0_14", "smallint"),
    ("working_age_member_count_15_64", "smallint"),
    ("older_member_count_65_plus", "smallint"), ("unknown_age_member_count", "smallint"),
    ("household_head_person_id", "text"), ("household_head_sex", "smallint"),
    ("household_head_age", "smallint"), ("household_head_marital_status", "smallint"),
    ("household_head_ethnicity", "smallint"),
    ("household_head_education_level", "smallint"),
    ("household_head_years_attended_school", "smallint"),
    ("household_head_can_read", "smallint"), ("household_head_can_write", "smallint"),
    ("province_code", "text"),
    ("district_code", "text"), ("commune_code", "text"), ("village_code", "text"),
    ("urban_rural", "text"), ("survey_month", "smallint"), ("stratum", "text"),
    ("household_weight", "double precision"), ("source_archive", "text"),
    ("source_submodule", "text"), ("source_row_id", "text"),
]
FINAL_HH_TYPES += ACTUAL_DATE_TYPES
FINAL_HL_TYPES = [
    ("dataset_name", "text"), ("survey_wave", "text"), ("survey_year", "smallint"),
    ("psu", "text"), ("household_id", "text"), ("person_id", "text"),
    ("member_line_number", "smallint"), ("sex", "smallint"),
    ("birth_day", "smallint"), ("birth_month", "smallint"),
    ("birth_year", "smallint"), ("age", "smallint"),
    ("relationship_to_household_head", "smallint"),
    ("father_line_number", "smallint"), ("father_person_id", "text"),
    ("mother_line_number", "smallint"), ("mother_person_id", "text"),
    ("marital_status_source_code", "smallint"),
    ("marital_status_harmonized", "smallint"),
    ("spouse_line_number", "smallint"), ("spouse_person_id", "text"),
    ("ethnicity_source_code", "smallint"), ("ethnicity_harmonized", "smallint"),
    ("absent_from_household", "smallint"), ("presence_reference_period", "text"),
    ("province_code", "text"), ("district_code", "text"),
    ("commune_code", "text"), ("village_code", "text"), ("urban_rural", "text"),
    ("survey_month", "smallint"), ("stratum", "text"),
    ("household_weight", "double precision"), ("person_weight", "double precision"),
    ("source_archive", "text"), ("source_submodule", "text"),
    ("source_row_id", "text"),
]
FINAL_ED_TYPES = [
    ("dataset_name", "text"), ("survey_wave", "text"), ("survey_year", "smallint"),
    ("psu", "text"), ("household_id", "text"), ("person_id", "text"),
    ("hl_link_matched", "smallint"), ("sex", "smallint"), ("age", "smallint"),
    ("province_code", "text"), ("district_code", "text"),
    ("commune_code", "text"), ("village_code", "text"), ("urban_rural", "text"),
    ("survey_month", "smallint"), ("stratum", "text"),
    ("household_weight", "double precision"), ("person_weight", "double precision"),
    ("can_read", "smallint"), ("can_write", "smallint"),
    ("ever_attended_school", "smallint"), ("years_attended_school", "smallint"),
    ("highest_education_level_source_code", "smallint"),
    ("education_level_harmonized", "smallint"),
    ("currently_attending_school", "smallint"),
    ("current_education_level_source_code", "smallint"),
    ("current_education_level_harmonized", "smallint"),
    ("source_archive", "text"), ("source_submodule", "text"),
    ("source_row_id", "text"),
]
FINAL_HO_TYPES = [
    ("dataset_name", "text"), ("survey_wave", "text"), ("survey_year", "smallint"),
    ("psu", "text"), ("household_id", "text"), ("hh_link_matched", "smallint"),
    ("province_code", "text"), ("district_code", "text"),
    ("commune_code", "text"), ("village_code", "text"), ("urban_rural", "text"),
    ("survey_month", "smallint"), ("stratum", "text"),
    ("household_weight", "double precision"),
    ("households_in_housing_unit", "smallint"),
    ("floor_area_square_meters", "double precision"), ("rooms_used", "smallint"),
    ("wall_material_source_code", "smallint"), ("roof_material_source_code", "smallint"),
    ("floor_material_source_code", "smallint"), ("main_lighting_source_code", "smallint"),
    ("main_drinking_water_source_code", "smallint"),
    ("monthly_water_charges_riel", "double precision"),
    ("drinking_water_treatment_frequency_source_code", "smallint"),
    ("treats_drinking_water", "smallint"), ("boils_drinking_water", "smallint"),
    ("filters_drinking_water", "smallint"),
    ("uses_chemical_water_treatment", "smallint"),
    ("uses_alum_water_treatment", "smallint"),
    ("uses_other_water_treatment", "smallint"),
    ("toilet_facility_source_code", "smallint"), ("has_toilet_facility", "smallint"),
    ("monthly_sewage_disposal_expense_riel", "double precision"),
    ("monthly_garbage_collection_expense_riel", "double precision"),
    ("main_cooking_fuel_source_code", "smallint"),
    ("monthly_electricity_expense_riel", "double precision"),
    ("monthly_gas_expense_riel", "double precision"),
    ("monthly_kerosene_expense_riel", "double precision"),
    ("monthly_firewood_expense_riel", "double precision"),
    ("monthly_charcoal_expense_riel", "double precision"),
    ("monthly_battery_expense_riel", "double precision"),
    ("monthly_other_energy_expense_riel", "double precision"),
    ("dwelling_tenure_source_code", "smallint"),
    ("dwelling_tenure_harmonized", "smallint"),
    ("monthly_rent_paid_riel", "double precision"),
    ("monthly_imputed_rent_riel", "double precision"),
    ("dwelling_maintenance_expense_riel", "double precision"),
    ("source_archive", "text"), ("source_submodule", "text"),
    ("source_row_id", "text"),
]
FINAL_EC_TYPES = [
    ("dataset_name", "text"), ("survey_wave", "text"), ("survey_year", "smallint"),
    ("psu", "text"), ("household_id", "text"), ("person_id", "text"),
    ("hl_link_matched", "smallint"), ("sex", "smallint"), ("age", "smallint"),
    ("province_code", "text"), ("district_code", "text"),
    ("commune_code", "text"), ("village_code", "text"), ("urban_rural", "text"),
    ("survey_month", "smallint"), ("stratum", "text"),
    ("household_weight", "double precision"), ("person_weight", "double precision"),
    ("worked_at_least_one_hour_past_7_days", "smallint"),
    ("second_work_screening_source_code", "smallint"),
    ("total_hours_worked_past_7_days", "smallint"),
    ("main_occupation_source_code", "text"), ("main_industry_source_code", "text"),
    ("main_employer_type_source_code", "smallint"),
    ("main_employment_status_source_code", "smallint"),
    ("main_hours_worked_past_7_days", "smallint"),
    ("main_days_worked_last_month", "smallint"),
    ("main_job_works_whole_year", "smallint"),
    ("main_job_was_usual_past_7_days", "smallint"),
    ("main_job_was_abroad", "smallint"),
    ("additional_jobs_count", "smallint"), ("total_occupations_past_7_days", "smallint"),
    ("secondary_occupation_source_code", "text"),
    ("secondary_industry_source_code", "text"),
    ("secondary_employer_type_source_code", "smallint"),
    ("secondary_employment_status_source_code", "smallint"),
    ("secondary_hours_worked_past_7_days", "smallint"),
    ("secondary_days_worked_last_month", "smallint"),
    ("secondary_job_works_whole_year", "smallint"),
    ("secondary_job_was_usual_past_7_days", "smallint"),
    ("monthly_salary_wages_riel", "double precision"),
    ("preferred_hours_change_source_code", "smallint"),
    ("hours_less_preferred", "smallint"), ("hours_more_preferred", "smallint"),
    ("available_for_additional_work", "smallint"),
    ("reason_working_fewer_hours_source_code", "smallint"),
    ("months_working_fewer_hours", "smallint"), ("actively_seeking_work", "smallint"),
    ("job_search_method_1_source_code", "smallint"),
    ("job_search_method_2_source_code", "smallint"),
    ("job_search_method_3_source_code", "smallint"),
    ("available_for_work", "smallint"), ("desired_weekly_hours", "smallint"),
    ("months_actively_seeking_work", "smallint"),
    ("reason_not_actively_seeking_source_code", "smallint"),
    ("months_out_of_work", "smallint"), ("latest_work_seasonal", "smallint"),
    ("source_archive", "text"), ("source_submodule", "text"), ("source_row_id", "text"),
]
FINAL_VL_TYPES = [
    ("dataset_name", "text"), ("survey_wave", "text"), ("survey_year", "smallint"),
    ("psu", "text"), ("hh_psu_link_matched", "smallint"),
    ("province_code", "text"), ("district_code", "text"),
    ("commune_code", "text"), ("village_code", "text"),
    ("province_name", "text"), ("district_name", "text"),
    ("commune_name", "text"), ("village_name", "text"),
    ("urban_rural", "text"), ("survey_month", "smallint"), ("stratum", "text"),
    ("sample_household_count", "smallint"), ("sample_person_count", "smallint"),
    ("village_reference_day", "smallint"), ("village_reference_month", "smallint"),
    ("village_reference_year", "smallint"), ("village_household_count", "integer"),
    ("enumeration_area_count", "integer"),
    ("households_in_surveyed_enumeration_area", "integer"),
    ("village_person_count", "integer"), ("village_male_count", "integer"),
    ("village_female_count", "integer"), ("population_below_18_count", "integer"),
    ("boys_below_18_count", "integer"), ("girls_below_18_count", "integer"),
    ("population_18_plus_count", "integer"), ("men_18_plus_count", "integer"),
    ("women_18_plus_count", "integer"),
    ("village_land_area_square_kilometers", "double precision"),
    ("five_year_population_movement_source_code", "smallint"),
    ("village_household_count_five_years_ago", "integer"),
    ("village_person_count_five_years_ago", "integer"),
    ("source_archive", "text"), ("source_submodule", "text"), ("source_row_id", "text"),
]
IND_QUE_TYPES = [
    ("canonical_varname", "text"), ("dataset_name", "text"),
    ("column_in_raw_sav", "text"), ("column_label_in_english", "text"),
    ("source_kind", "text"), ("measure_type", "text"), ("canonical_text", "text"),
]
ALIGN_SUMMARY_TYPES = [
    ("varname", "text"), ("dataset_count", "integer"), ("source_count", "integer"),
    ("explicit_count", "integer"), ("derived_count", "integer"),
    ("measure_type", "text"), ("canonical_text", "text"),
]


def database_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [snake_case(str(column)) for column in result.columns]
    return result


def load_specs(root: Path) -> list[TableSpec]:
    output = root / "data" / "exp" / "database"
    hh = database_frame(pd.read_parquet(output / "final_HH_CSES.parquet"))
    survey_dates = database_frame(
        pd.read_parquet(
            output / "final_SURVEY_DATE_CSES.parquet",
            columns=[
                "Survey Wave", "Household ID", "Survey Actual Year",
                "Survey Actual Month", "Survey Actual Day",
            ],
        )
    )
    hh = enrich_hh_frame(hh, survey_dates)
    hh_dictionary = extend_hh_dictionary(
        pd.read_csv(output / "ind_que_HH_CSES.csv", dtype="string")
    )
    hh_summary = extend_hh_summary(
        pd.read_csv(output / "align_summary_HH_CSES.csv")
    )
    return [
        TableSpec("final_HH_CSES", hh, FINAL_HH_TYPES),
        TableSpec("final_HL_CSES", database_frame(pd.read_parquet(output / "final_HL_CSES.parquet")), FINAL_HL_TYPES),
        TableSpec("final_ED_CSES", database_frame(pd.read_parquet(output / "final_ED_CSES.parquet")), FINAL_ED_TYPES),
        TableSpec("final_HO_CSES", database_frame(pd.read_parquet(output / "final_HO_CSES.parquet")), FINAL_HO_TYPES),
        TableSpec("final_EC_CSES", database_frame(pd.read_parquet(output / "final_EC_CSES.parquet")), FINAL_EC_TYPES),
        TableSpec("final_VL_CSES", database_frame(pd.read_parquet(output / "final_VL_CSES.parquet")), FINAL_VL_TYPES),
        TableSpec("ind_que_HH_CSES", hh_dictionary, IND_QUE_TYPES),
        TableSpec("ind_que_HL_CSES", pd.read_csv(output / "ind_que_HL_CSES.csv", dtype="string"), IND_QUE_TYPES),
        TableSpec("ind_que_ED_CSES", pd.read_csv(output / "ind_que_ED_CSES.csv", dtype="string"), IND_QUE_TYPES),
        TableSpec("ind_que_HO_CSES", pd.read_csv(output / "ind_que_HO_CSES.csv", dtype="string"), IND_QUE_TYPES),
        TableSpec("ind_que_EC_CSES", pd.read_csv(output / "ind_que_EC_CSES.csv", dtype="string"), IND_QUE_TYPES),
        TableSpec("ind_que_VL_CSES", pd.read_csv(output / "ind_que_VL_CSES.csv", dtype="string"), IND_QUE_TYPES),
        TableSpec("align_summary_HH_CSES", hh_summary, ALIGN_SUMMARY_TYPES),
        TableSpec("align_summary_HL_CSES", pd.read_csv(output / "align_summary_HL_CSES.csv"), ALIGN_SUMMARY_TYPES),
        TableSpec("align_summary_ED_CSES", pd.read_csv(output / "align_summary_ED_CSES.csv"), ALIGN_SUMMARY_TYPES),
        TableSpec("align_summary_HO_CSES", pd.read_csv(output / "align_summary_HO_CSES.csv"), ALIGN_SUMMARY_TYPES),
        TableSpec("align_summary_EC_CSES", pd.read_csv(output / "align_summary_EC_CSES.csv"), ALIGN_SUMMARY_TYPES),
        TableSpec("align_summary_VL_CSES", pd.read_csv(output / "align_summary_VL_CSES.csv"), ALIGN_SUMMARY_TYPES),
    ]


def python_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def create_table(cursor: psycopg.Cursor, schema: str, table: str, columns: list[tuple[str, str]]) -> None:
    definitions = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(data_type))
        for name, data_type in columns
    )
    cursor.execute(
        sql.SQL("CREATE TABLE {}.{} ({})").format(
            sql.Identifier(schema), sql.Identifier(table), definitions
        )
    )


def copy_frame(cursor: psycopg.Cursor, schema: str, table: str, spec: TableSpec) -> None:
    expected = [name for name, _ in spec.columns]
    if list(spec.frame.columns) != expected:
        raise ValueError(f"{spec.name} columns differ: expected={expected}, actual={list(spec.frame.columns)}")
    statement = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, expected)),
    )
    with cursor.copy(statement) as copy:
        for row in spec.frame.itertuples(index=False, name=None):
            copy.write_row(tuple(python_value(value) for value in row))


def add_final_comments(cursor: psycopg.Cursor, schema: str, table: str, frame: pd.DataFrame) -> None:
    variable_info = {
        **VARIABLE_INFO, **ED_VARIABLE_INFO, **HO_VARIABLE_INFO,
        **EC_VARIABLE_INFO, **VL_VARIABLE_INFO,
    }
    for readable_name, (full_name, _measure_type, canonical_text) in variable_info.items():
        column = snake_case(readable_name)
        if column not in frame.columns:
            continue
        cursor.execute(
            sql.SQL("COMMENT ON COLUMN {}.{}.{} IS {}").format(
                sql.Identifier(schema), sql.Identifier(table), sql.Identifier(column),
                sql.Literal(f"{full_name}. {canonical_text}"),
            )
        )
    if table == "final_HH_CSES":
        for component in ["year", "month", "day"]:
            cursor.execute(
                sql.SQL("COMMENT ON COLUMN {}.{}.{} IS {}").format(
                    sql.Identifier(schema), sql.Identifier(table),
                    sql.Identifier(f"survey_actual_{component}"),
                    sql.Literal(
                        f"Calendar {component} of the selected explicit household survey date: "
                        "interview date in 2004 and last-visit date in 2019/2021; "
                        "null when unavailable."
                    ),
                )
            )


def table_exists(cursor: psycopg.Cursor, schema: str, table: str) -> bool:
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname=%s AND tablename=%s)",
        (schema, table),
    )
    return bool(cursor.fetchone()[0])


def publish(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    validate_hh_hl_local_release()
    validate_ed_local_release()
    validate_ho_local_release()
    validate_ec_local_release()
    validate_vl_local_release()
    specs = load_specs(root)
    target_names = [spec.name for spec in specs]

    with psycopg.connect(host=args.host, port=args.port, dbname=args.dbname) as connection:
        with connection.cursor() as cursor:
            existing = [name for name in target_names if table_exists(cursor, args.schema, name)]
            if existing and not args.replace:
                raise RuntimeError(f"Refusing to overwrite existing tables without --replace: {existing}")

            staging: dict[str, str] = {}
            for spec in specs:
                stage_name = f"_staging_{spec.name}_{uuid.uuid4().hex[:8]}"
                staging[spec.name] = stage_name
                create_table(cursor, args.schema, stage_name, spec.columns)
                copy_frame(cursor, args.schema, stage_name, spec)
                cursor.execute(
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier(stage_name)
                    )
                )
                copied = int(cursor.fetchone()[0])
                if copied != len(spec.frame):
                    raise RuntimeError(f"COPY row-count mismatch for {spec.name}: {copied} != {len(spec.frame)}")

            if args.replace:
                cursor.execute("DELETE FROM public._catalog WHERE table_name = ANY(%s)", (target_names,))
                for name in reversed(target_names):
                    cursor.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                            sql.Identifier(args.schema), sql.Identifier(name)
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
                    sql.Identifier("uq_final_HH_CSES_wave_household"),
                    sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                )
            )
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX {} ON {}.{} (survey_wave, household_head_person_id) "
                    "WHERE household_head_person_id IS NOT NULL"
                ).format(
                    sql.Identifier("ix_final_HH_CSES_head_person"),
                    sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, person_id)").format(
                    sql.Identifier("uq_final_HL_CSES_wave_person"),
                    sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (survey_wave, household_id)").format(
                    sql.Identifier("ix_final_HL_CSES_wave_household"),
                    sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, household_id)").format(
                    sql.Identifier("uq_final_HO_CSES_wave_household"),
                    sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (survey_wave, psu)").format(
                    sql.Identifier("ix_final_HO_CSES_wave_psu"),
                    sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, person_id)").format(
                    sql.Identifier("uq_final_EC_CSES_wave_person"),
                    sql.Identifier(args.schema), sql.Identifier("final_EC_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (survey_wave, household_id)").format(
                    sql.Identifier("ix_final_EC_CSES_wave_household"),
                    sql.Identifier(args.schema), sql.Identifier("final_EC_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, psu)").format(
                    sql.Identifier("uq_final_VL_CSES_wave_psu"),
                    sql.Identifier(args.schema), sql.Identifier("final_VL_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, household_id, member_line_number)").format(
                    sql.Identifier("uq_final_HL_CSES_wave_household_member_line"),
                    sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                )
            )
            for role in ["father", "mother", "spouse"]:
                cursor.execute(
                    sql.SQL("CREATE INDEX {} ON {}.{} (survey_wave, {}) WHERE {} IS NOT NULL").format(
                        sql.Identifier(f"ix_final_HL_CSES_{role}_person"),
                        sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                        sql.Identifier(f"{role}_person_id"),
                        sql.Identifier(f"{role}_person_id"),
                    )
                )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, person_id)").format(
                    sql.Identifier("uq_final_ED_CSES_wave_person"),
                    sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"),
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (survey_wave, household_id)").format(
                    sql.Identifier("ix_final_ED_CSES_wave_household"),
                    sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"),
                )
            )

            hh = next(spec for spec in specs if spec.name == "final_HH_CSES")
            hl = next(spec for spec in specs if spec.name == "final_HL_CSES")
            ed = next(spec for spec in specs if spec.name == "final_ED_CSES")
            ho = next(spec for spec in specs if spec.name == "final_HO_CSES")
            ec = next(spec for spec in specs if spec.name == "final_EC_CSES")
            vl = next(spec for spec in specs if spec.name == "final_VL_CSES")
            add_final_comments(cursor, args.schema, hh.name, hh.frame)
            add_final_comments(cursor, args.schema, hl.name, hl.frame)
            add_final_comments(cursor, args.schema, ed.name, ed.frame)
            add_final_comments(cursor, args.schema, ho.name, ho.frame)
            add_final_comments(cursor, args.schema, ec.name, ec.frame)
            add_final_comments(cursor, args.schema, vl.name, vl.frame)
            cursor.execute(
                sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                    sql.Identifier(args.schema), sql.Identifier(hh.name),
                    sql.Literal("CSES household linkage spine with roster-derived composition and head attributes, pooled across ten repeated cross-sectional waves."),
                )
            )
            cursor.execute(
                sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                    sql.Identifier(args.schema), sql.Identifier(ec.name),
                    sql.Literal("CSES person-level current-employment records aligned across ten repeated cross-sectional waves, with retained HL-link status and wave-specific occupation/industry codes."),
                )
            )
            cursor.execute(
                sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                    sql.Identifier(args.schema), sql.Identifier(vl.name),
                    sql.Literal("CSES PSU-level village demographic records aligned across the eight waves with released village questionnaires and linked to HH/HL sample coverage."),
                )
            )
            cursor.execute(
                sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                    sql.Identifier(args.schema), sql.Identifier(ho.name),
                    sql.Literal("CSES household-level housing, water, sanitation, energy, tenure, and dwelling-cost records aligned across ten repeated cross-sectional waves, with retained HH-link status."),
                )
            )
            cursor.execute(
                sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                    sql.Identifier(args.schema), sql.Identifier(hl.name),
                    sql.Literal("CSES household-member linkage spine pooled across ten repeated cross-sectional waves."),
                )
            )
            cursor.execute(
                sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                    sql.Identifier(args.schema), sql.Identifier(ed.name),
                    sql.Literal("CSES person-level education records aligned across ten repeated cross-sectional waves, with retained HL-link status."),
                )
            )

            catalog_rows = [
                (
                    "final_HH_CSES", "HH", "one row per CSES household",
                    "survey_wave + household_id", "ind_que_HH_CSES",
                    "CSES household alignment: identifiers, roster-derived composition, household-head demographics and education, geography, survey design, weights, and source provenance.",
                    "Repeated cross-sections, not a panel. Composition and head fields are derived from final_HL_CSES; head education fields are linked from final_ED_CSES. One 2019 household has no coded head, and one 2016 coded head has no released ED record. Completed years are unavailable in 2004; one documented 2004 age=99 sentinel is null. Selected 2004 core sources lack district/commune/village codes, survey month, and a general household weight; stratum is unavailable in 2007; 24 households in two 2014 PSUs have unresolved survey-month conflicts.",
                    len(hh.frame), 10,
                ),
                (
                    "final_HL_CSES", "HL", "one row per CSES household member",
                    "survey_wave + person_id; household via survey_wave + household_id", "ind_que_HL_CSES",
                    "CSES member alignment: linkage identifiers, birth components, core demographics, marital status, ethnicity, within-household parent/spouse links, recent household absence, inherited geography, survey design, weights, and source provenance.",
                    "Repeated cross-sections, not a panel. Kin source lines are retained, but derived parent/spouse person IDs are null when the line has no same-household target or points to self. Marital source codes differ in 2004 and are separately harmonized. Presence means current status in 2004 versus all days in the past week thereafter. The source has 129 missing 2004 absence statuses, 11 missing 2007 relationship codes, and one 2019 household without a coded head. Selected 2004 core sources lack district/commune/village codes, survey month, and general household/person weights; stratum is unavailable in 2007.",
                    len(hl.frame), 10,
                ),
                (
                    "final_ED_CSES", "ED", "one row per released CSES education record",
                    "survey_wave + person_id; household via survey_wave + household_id", "ind_que_ED_CSES",
                    "CSES education alignment: literacy, attendance, completed years, detailed released level codes, broad cross-wave levels, HL context, weights, and source provenance.",
                    "Education-source inclusion differs by wave, so HL members without an education record must not be interpreted as never attending school. Four released education records absent from HL are retained with hl_link_matched=0. Completed years are unavailable in 2004; one invalid 2021 value was set to null. Eighteen current-level codes lacking confirmed current attendance are retained as source codes while the harmonized current level is null.",
                    len(ed.frame), 10,
                ),
                (
                    "final_HO_CSES", "HO", "one row per released CSES housing record",
                    "survey_wave + household_id; household via survey_wave + household_id", "ind_que_HO_CSES",
                    "CSES housing alignment: dwelling size and materials, lighting, water treatment, sanitation, cooking fuel, energy costs, tenure, rent, maintenance, HH context, weights, and source provenance.",
                    "Repeated cross-sections, not a panel. Nineteen released housing records absent from final_HH_CSES are retained with hh_link_matched=0. Detailed material, lighting, water, toilet, and cooking-fuel categories remain wave-specific source codes. Monetary variables are nominal Cambodian riel without imputation or price adjustment. Monthly imputed rent is unavailable in 2004.",
                    len(ho.frame), 10,
                ),
                (
                    "final_EC_CSES", "EC", "one row per released CSES current-employment record",
                    "survey_wave + person_id; household via survey_wave + household_id", "ind_que_EC_CSES",
                    "CSES current-employment alignment: work screening, hours, main and secondary jobs, occupation and industry source codes, wages, underemployment, job search, HL context, weights, and provenance.",
                    "Repeated cross-sections, not a panel. Two released EC records absent from final_HL_CSES are retained with hl_link_matched=0. The 2004 release combines section 13A and 13B one-to-one by person; 2007 has no detailed occupation/job block. Occupation and industry classifications remain wave-specific source codes. Monetary wages are nominal Cambodian riel without imputation or price adjustment.",
                    len(ec.frame), 10,
                ),
                (
                    "final_VL_CSES", "VL", "one row per released CSES village-survey PSU record",
                    "survey_wave + psu; households via survey_wave + psu", "ind_que_VL_CSES",
                    "CSES village alignment: village population and households, age/sex components, land area, five-year population change, administrative context and linked HH/HL sample coverage.",
                    "Village questionnaires are available in eight waves (2004, 2007, 2009, 2011-12, 2014, 2016, 2019, 2021), not all household-survey waves. All 5,718 village PSUs link to final_HH_CSES. Released demographic components are retained when they do not sum to the released total; these discrepancies are documented in cses_vl_data_issues.csv. Detailed age and historical village fields are unavailable in the compact 2019 village file.",
                    len(vl.frame), 8,
                ),
            ]
            cursor.executemany(
                """
                INSERT INTO public._catalog
                    (table_name, module, grain, join_keys, ind_que_table, description,
                     caveats, row_count, n_datasets, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                """,
                catalog_rows,
            )

        connection.commit()

        with connection.cursor() as cursor:
            for spec in specs:
                cursor.execute(
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier(spec.name)
                    )
                )
                print(f"published={args.schema}.{spec.name} rows={int(cursor.fetchone()[0])}")
            cursor.execute(
                "SELECT count(*) FROM public._catalog WHERE table_name = ANY(%s)",
                ([spec.name for spec in specs if spec.name.startswith("final_")],),
            )
            print(f"catalog_rows={int(cursor.fetchone()[0])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--replace", action="store_true", help="Explicitly replace existing CSES target tables.")
    publish(parser.parse_args())


if __name__ == "__main__":
    main()
