#!/usr/bin/env python3
"""Read back and validate the published CSES database tables in PostgreSQL."""

from __future__ import annotations

import argparse

import psycopg
from psycopg import sql

from publish_cses_to_postgres import (
    ALIGN_SUMMARY_TYPES,
    FINAL_EC_TYPES,
    FINAL_ED_TYPES,
    FINAL_HH_TYPES,
    FINAL_HL_TYPES,
    FINAL_HO_TYPES,
    FINAL_VL_TYPES,
    IND_QUE_TYPES,
)


TABLE_CONTRACTS = {
    "final_HH_CSES": (77_904, FINAL_HH_TYPES),
    "final_HL_CSES": (358_920, FINAL_HL_TYPES),
    "final_ED_CSES": (343_204, FINAL_ED_TYPES),
    "final_HO_CSES": (77_922, FINAL_HO_TYPES),
    "final_EC_CSES": (332_903, FINAL_EC_TYPES),
    "final_VL_CSES": (5_718, FINAL_VL_TYPES),
    "ind_que_HH_CSES": (342, IND_QUE_TYPES),
    "ind_que_HL_CSES": (384, IND_QUE_TYPES),
    "ind_que_ED_CSES": (307, IND_QUE_TYPES),
    "ind_que_HO_CSES": (524, IND_QUE_TYPES),
    "ind_que_EC_CSES": (557, IND_QUE_TYPES),
    "ind_que_VL_CSES": (300, IND_QUE_TYPES),
    "align_summary_HH_CSES": (35, ALIGN_SUMMARY_TYPES),
    "align_summary_HL_CSES": (37, ALIGN_SUMMARY_TYPES),
    "align_summary_ED_CSES": (30, ALIGN_SUMMARY_TYPES),
    "align_summary_HO_CSES": (50, ALIGN_SUMMARY_TYPES),
    "align_summary_EC_CSES": (60, ALIGN_SUMMARY_TYPES),
    "align_summary_VL_CSES": (40, ALIGN_SUMMARY_TYPES),
}

EXPECTED_INDEXES = {
    "uq_final_HH_CSES_wave_household",
    "ix_final_HH_CSES_head_person",
    "uq_final_HL_CSES_wave_person",
    "ix_final_HL_CSES_wave_household",
    "uq_final_HL_CSES_wave_household_member_line",
    "ix_final_HL_CSES_father_person",
    "ix_final_HL_CSES_mother_person",
    "ix_final_HL_CSES_spouse_person",
    "uq_final_ED_CSES_wave_person",
    "ix_final_ED_CSES_wave_household",
    "uq_final_HO_CSES_wave_household",
    "ix_final_HO_CSES_wave_psu",
    "uq_final_EC_CSES_wave_person",
    "ix_final_EC_CSES_wave_household",
    "uq_final_VL_CSES_wave_psu",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scalar(cursor: psycopg.Cursor, statement: object, parameters: tuple[object, ...] = ()) -> object:
    cursor.execute(statement, parameters)
    return cursor.fetchone()[0]


def validate(args: argparse.Namespace) -> None:
    with psycopg.connect(host=args.host, port=args.port, dbname=args.dbname) as connection:
        with connection.cursor() as cursor:
            for table, (expected_rows, columns) in TABLE_CONTRACTS.items():
                actual_rows = int(scalar(
                    cursor,
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier(table)
                    ),
                ))
                require(actual_rows == expected_rows, f"{table} rows={actual_rows}, expected={expected_rows}")
                cursor.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema=%s AND table_name=%s
                    ORDER BY ordinal_position
                    """,
                    (args.schema, table),
                )
                actual_columns = cursor.fetchall()
                require(actual_columns == columns, f"{table} database columns/types differ from the contract")

            duplicate_hh = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM (SELECT survey_wave, household_id FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1) d"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"))))
            duplicate_hl = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM (SELECT survey_wave, person_id FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1) d"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"))))
            duplicate_ed = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM (SELECT survey_wave, person_id FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1) d"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"))))
            duplicate_ho = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM (SELECT survey_wave, household_id FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1) d"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"))))
            duplicate_ec = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM (SELECT survey_wave, person_id FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1) d"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_EC_CSES"))))
            duplicate_vl = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM (SELECT survey_wave, psu FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1) d"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_VL_CSES"))))
            require(
                duplicate_hh == duplicate_hl == duplicate_ed == duplicate_ho == duplicate_ec == duplicate_vl == 0,
                "A published final table contains duplicate keys",
            )

            hl_orphans = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} hl
                LEFT JOIN {}.{} hh USING (survey_wave, household_id)
                WHERE hh.household_id IS NULL
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
            )))
            require(hl_orphans == 0, f"Published HL-to-HH orphan rows={hl_orphans}")

            invalid_hh = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE household_member_count IS NULL OR household_member_count < 1
                   OR male_member_count IS NULL OR female_member_count IS NULL
                   OR child_member_count_0_14 IS NULL
                   OR working_age_member_count_15_64 IS NULL
                   OR older_member_count_65_plus IS NULL OR unknown_age_member_count IS NULL
                   OR male_member_count + female_member_count <> household_member_count
                   OR child_member_count_0_14 + working_age_member_count_15_64
                      + older_member_count_65_plus + unknown_age_member_count <> household_member_count
                   OR (household_head_person_id IS NULL AND (
                         household_head_sex IS NOT NULL OR household_head_age IS NOT NULL
                         OR household_head_marital_status IS NOT NULL OR household_head_ethnicity IS NOT NULL
                         OR household_head_education_level IS NOT NULL
                         OR household_head_years_attended_school IS NOT NULL
                         OR household_head_can_read IS NOT NULL OR household_head_can_write IS NOT NULL
                      ))
                   OR (household_head_sex IS NOT NULL AND household_head_sex NOT IN (1,2))
                   OR (household_head_age IS NOT NULL AND household_head_age NOT BETWEEN 0 AND 120)
                   OR (household_head_marital_status IS NOT NULL AND household_head_marital_status NOT BETWEEN 1 AND 4)
                   OR (household_head_ethnicity IS NOT NULL AND household_head_ethnicity NOT BETWEEN 1 AND 8)
                   OR (household_head_education_level IS NOT NULL AND household_head_education_level NOT BETWEEN 0 AND 7)
                   OR (household_head_years_attended_school IS NOT NULL AND household_head_years_attended_school NOT BETWEEN 0 AND 30)
                   OR (household_head_can_read IS NOT NULL AND household_head_can_read NOT IN (0,1))
                   OR (household_head_can_write IS NOT NULL AND household_head_can_write NOT IN (0,1))
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"))))
            require(invalid_hh == 0, f"Published HH composition/head domain violations={invalid_hh}")

            composition_mismatches = int(scalar(cursor, sql.SQL(
                """
                WITH roster AS (
                    SELECT survey_wave, household_id,
                           count(*) AS member_count,
                           count(*) FILTER (WHERE sex=1) AS male_count,
                           count(*) FILTER (WHERE sex=2) AS female_count,
                           count(*) FILTER (WHERE age BETWEEN 0 AND 14) AS child_count,
                           count(*) FILTER (WHERE age BETWEEN 15 AND 64) AS working_age_count,
                           count(*) FILTER (WHERE age >= 65) AS older_count,
                           count(*) FILTER (WHERE age IS NULL) AS unknown_age_count
                    FROM {}.{} GROUP BY 1,2
                )
                SELECT count(*) FROM {}.{} hh
                JOIN roster r USING (survey_wave, household_id)
                WHERE hh.household_member_count IS DISTINCT FROM r.member_count
                   OR hh.male_member_count IS DISTINCT FROM r.male_count
                   OR hh.female_member_count IS DISTINCT FROM r.female_count
                   OR hh.child_member_count_0_14 IS DISTINCT FROM r.child_count
                   OR hh.working_age_member_count_15_64 IS DISTINCT FROM r.working_age_count
                   OR hh.older_member_count_65_plus IS DISTINCT FROM r.older_count
                   OR hh.unknown_age_member_count IS DISTINCT FROM r.unknown_age_count
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
            )))
            require(composition_mismatches == 0, f"Published HH composition differs from HL rows={composition_mismatches}")

            head_mismatches = int(scalar(cursor, sql.SQL(
                """
                WITH heads AS (
                    SELECT survey_wave, household_id, person_id, sex, age,
                           marital_status_harmonized, ethnicity_harmonized
                    FROM {}.{} WHERE relationship_to_household_head=1
                )
                SELECT count(*) FROM {}.{} hh
                LEFT JOIN heads h USING (survey_wave, household_id)
                WHERE hh.household_head_person_id IS DISTINCT FROM h.person_id
                   OR hh.household_head_sex IS DISTINCT FROM h.sex
                   OR hh.household_head_age IS DISTINCT FROM h.age
                   OR hh.household_head_marital_status IS DISTINCT FROM h.marital_status_harmonized
                   OR hh.household_head_ethnicity IS DISTINCT FROM h.ethnicity_harmonized
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
            )))
            require(head_mismatches == 0, f"Published HH head attributes differ from HL rows={head_mismatches}")

            head_education_mismatches = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} hh
                LEFT JOIN {}.{} ed
                  ON ed.survey_wave=hh.survey_wave
                 AND ed.person_id=hh.household_head_person_id
                WHERE hh.household_head_education_level IS DISTINCT FROM ed.education_level_harmonized
                   OR hh.household_head_years_attended_school IS DISTINCT FROM ed.years_attended_school
                   OR hh.household_head_can_read IS DISTINCT FROM ed.can_read
                   OR hh.household_head_can_write IS DISTINCT FROM ed.can_write
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"),
            )))
            require(head_education_mismatches == 0, f"Published HH head education differs from ED rows={head_education_mismatches}")

            households_without_head = int(scalar(
                cursor,
                sql.SQL("SELECT count(*) FROM {}.{} WHERE household_head_person_id IS NULL").format(
                    sql.Identifier(args.schema), sql.Identifier("final_HH_CSES")
                ),
            ))
            heads_without_ed = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} hh
                LEFT JOIN {}.{} ed
                  ON ed.survey_wave=hh.survey_wave AND ed.person_id=hh.household_head_person_id
                WHERE hh.household_head_person_id IS NOT NULL AND ed.person_id IS NULL
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"),
            )))
            require(households_without_head == 1, f"Published households without a coded head={households_without_head}")
            require(heads_without_ed == 1, f"Published coded heads without an ED record={heads_without_ed}")

            invalid_hl_extended = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE (age IS NOT NULL AND age NOT BETWEEN 0 AND 120)
                   OR (survey_wave='2004' AND age IN (98,99))
                   OR member_line_number IS NULL OR member_line_number NOT BETWEEN 1 AND 98
                   OR member_line_number <> right(person_id, 2)::smallint
                   OR (birth_day IS NOT NULL AND birth_day NOT BETWEEN 1 AND 31)
                   OR (birth_month IS NOT NULL AND birth_month NOT BETWEEN 1 AND 12)
                   OR (birth_year IS NOT NULL AND birth_year NOT BETWEEN 1800 AND survey_year + 1)
                   OR marital_status_harmonized IS DISTINCT FROM
                      CASE
                        WHEN survey_wave='2004' THEN
                          CASE marital_status_source_code
                            WHEN 1 THEN 1 WHEN 2 THEN 2 WHEN 3 THEN 2
                            WHEN 4 THEN 3 WHEN 5 THEN 4 WHEN 6 THEN 4
                          END
                        ELSE
                          CASE marital_status_source_code
                            WHEN 1 THEN 2 WHEN 2 THEN 4 WHEN 3 THEN 3 WHEN 4 THEN 1
                          END
                      END
                   OR ethnicity_harmonized IS DISTINCT FROM ethnicity_source_code
                   OR (ethnicity_source_code IS NOT NULL AND ethnicity_source_code NOT BETWEEN 1 AND 8)
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"))))
            require(invalid_hl_extended == 0, f"Published extended HL domain/mapping violations={invalid_hl_extended}")
            missing_age_2004 = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM {}.{} WHERE survey_wave='2004' AND age IS NULL"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"))))
            require(missing_age_2004 == 1, f"Published missing 2004 age rows={missing_age_2004}")

            duplicate_member_lines = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM (
                    SELECT survey_wave, household_id, member_line_number
                    FROM {}.{}
                    GROUP BY 1,2,3 HAVING count(*) > 1
                ) d
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"))))
            require(duplicate_member_lines == 0, f"Published duplicate household member lines={duplicate_member_lines}")

            for role in ["father", "mother", "spouse"]:
                line_column = f"{role}_line_number"
                person_column = f"{role}_person_id"
                invalid_links = int(scalar(cursor, sql.SQL(
                    """
                    SELECT count(*)
                    FROM {}.{} source
                    LEFT JOIN {}.{} target
                      ON target.survey_wave=source.survey_wave
                     AND target.household_id=source.household_id
                     AND target.member_line_number=source.{}
                    WHERE (source.{} IS NULL AND source.{} IS NOT NULL)
                       OR (
                            source.{} IS NOT NULL
                            AND (
                                (target.person_id IS NOT NULL AND target.person_id<>source.person_id
                                 AND source.{} IS DISTINCT FROM target.person_id)
                                OR ((target.person_id IS NULL OR target.person_id=source.person_id)
                                    AND source.{} IS NOT NULL)
                            )
                       )
                    """
                ).format(
                    sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                    sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                    sql.Identifier(line_column), sql.Identifier(line_column),
                    sql.Identifier(person_column), sql.Identifier(line_column),
                    sql.Identifier(person_column), sql.Identifier(person_column),
                )))
                require(invalid_links == 0, f"Published {role} link violations={invalid_links}")

            matched_ed_orphans = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ed
                LEFT JOIN {}.{} hl USING (survey_wave, person_id)
                WHERE ed.hl_link_matched=1 AND hl.person_id IS NULL
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
            )))
            false_unmatched_ed = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ed
                JOIN {}.{} hl USING (survey_wave, person_id)
                WHERE ed.hl_link_matched=0
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
            )))
            require(matched_ed_orphans == 0 and false_unmatched_ed == 0, "Published ED link flags disagree with HL")

            cursor.execute(sql.SQL(
                "SELECT survey_wave, person_id FROM {}.{} WHERE hl_link_matched=0 ORDER BY survey_wave, person_id"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_ED_CSES")))
            unmatched = cursor.fetchall()
            require(unmatched == [
                ("2013", "121010900"),
                ("2014", "020240101"),
                ("2014", "020240102"),
                ("2014", "020240103"),
            ], f"Published unmatched ED records differ: {unmatched}")

            invalid_ed = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE hl_link_matched NOT IN (0,1) OR hl_link_matched IS NULL
                   OR (can_read IS NOT NULL AND can_read NOT IN (0,1))
                   OR (can_write IS NOT NULL AND can_write NOT IN (0,1))
                   OR (ever_attended_school IS NOT NULL AND ever_attended_school NOT IN (0,1))
                   OR (currently_attending_school IS NOT NULL AND currently_attending_school NOT IN (0,1))
                   OR (years_attended_school IS NOT NULL AND years_attended_school NOT BETWEEN 0 AND 30)
                   OR (education_level_harmonized IS NOT NULL AND education_level_harmonized NOT BETWEEN 0 AND 7)
                   OR (current_education_level_harmonized IS NOT NULL AND current_education_level_harmonized NOT BETWEEN 1 AND 7)
                   OR (current_education_level_harmonized IS NOT NULL AND currently_attending_school IS DISTINCT FROM 1)
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_ED_CSES"))))
            require(invalid_ed == 0, f"Published ED domain/condition violations={invalid_ed}")

            matched_ho_orphans = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ho
                LEFT JOIN {}.{} hh USING (survey_wave, household_id)
                WHERE ho.hh_link_matched=1 AND hh.household_id IS NULL
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
            )))
            false_unmatched_ho = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ho
                JOIN {}.{} hh USING (survey_wave, household_id)
                WHERE ho.hh_link_matched=0
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
            )))
            unmatched_ho = int(scalar(cursor, sql.SQL(
                "SELECT count(*) FROM {}.{} WHERE hh_link_matched=0"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"))))
            require(
                matched_ho_orphans == 0 and false_unmatched_ho == 0 and unmatched_ho == 19,
                "Published HO link flags disagree with HH or the documented unmatched count changed",
            )

            ho_context_mismatches = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ho
                JOIN {}.{} hh USING (survey_wave, household_id)
                WHERE ho.hh_link_matched=1 AND (
                       ho.psu IS DISTINCT FROM hh.psu
                    OR ho.province_code IS DISTINCT FROM hh.province_code
                    OR ho.district_code IS DISTINCT FROM hh.district_code
                    OR ho.commune_code IS DISTINCT FROM hh.commune_code
                    OR ho.village_code IS DISTINCT FROM hh.village_code
                    OR ho.urban_rural IS DISTINCT FROM hh.urban_rural
                    OR ho.survey_month IS DISTINCT FROM hh.survey_month
                    OR ho.stratum IS DISTINCT FROM hh.stratum
                    OR ho.household_weight IS DISTINCT FROM hh.household_weight
                )
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
            )))
            require(ho_context_mismatches == 0, f"Published HO context differs from HH rows={ho_context_mismatches}")

            invalid_ho = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE hh_link_matched NOT IN (0,1) OR hh_link_matched IS NULL
                   OR (households_in_housing_unit IS NOT NULL AND households_in_housing_unit NOT BETWEEN 1 AND 50)
                   OR (floor_area_square_meters IS NOT NULL AND floor_area_square_meters NOT BETWEEN 0.0001 AND 10000)
                   OR (rooms_used IS NOT NULL AND rooms_used NOT BETWEEN 1 AND 50)
                   OR (drinking_water_treatment_frequency_source_code IS NOT NULL
                       AND drinking_water_treatment_frequency_source_code NOT IN (1,2,3))
                   OR treats_drinking_water IS DISTINCT FROM
                      CASE drinking_water_treatment_frequency_source_code
                        WHEN 1 THEN 1 WHEN 2 THEN 1 WHEN 3 THEN 0
                      END
                   OR (boils_drinking_water IS NOT NULL AND boils_drinking_water NOT IN (0,1))
                   OR (filters_drinking_water IS NOT NULL AND filters_drinking_water NOT IN (0,1))
                   OR (uses_chemical_water_treatment IS NOT NULL AND uses_chemical_water_treatment NOT IN (0,1))
                   OR (uses_alum_water_treatment IS NOT NULL AND uses_alum_water_treatment NOT IN (0,1))
                   OR (uses_other_water_treatment IS NOT NULL AND uses_other_water_treatment NOT IN (0,1))
                   OR (has_toilet_facility IS NOT NULL AND has_toilet_facility NOT IN (0,1))
                   OR dwelling_tenure_harmonized IS DISTINCT FROM dwelling_tenure_source_code
                   OR (dwelling_tenure_harmonized IS NOT NULL AND dwelling_tenure_harmonized NOT BETWEEN 1 AND 4)
                   OR monthly_water_charges_riel < 0
                   OR monthly_sewage_disposal_expense_riel < 0
                   OR monthly_garbage_collection_expense_riel < 0
                   OR monthly_electricity_expense_riel < 0
                   OR monthly_gas_expense_riel < 0
                   OR monthly_kerosene_expense_riel < 0
                   OR monthly_firewood_expense_riel < 0
                   OR monthly_charcoal_expense_riel < 0
                   OR monthly_battery_expense_riel < 0
                   OR monthly_other_energy_expense_riel < 0
                   OR monthly_rent_paid_riel < 0
                   OR monthly_imputed_rent_riel < 0
                   OR dwelling_maintenance_expense_riel < 0
                   OR 9999999 IN (
                        monthly_water_charges_riel, monthly_sewage_disposal_expense_riel,
                        monthly_garbage_collection_expense_riel, monthly_electricity_expense_riel,
                        monthly_gas_expense_riel, monthly_kerosene_expense_riel,
                        monthly_firewood_expense_riel, monthly_charcoal_expense_riel,
                        monthly_battery_expense_riel, monthly_other_energy_expense_riel,
                        monthly_rent_paid_riel, monthly_imputed_rent_riel,
                        dwelling_maintenance_expense_riel)
                   OR 99999999 IN (
                        monthly_water_charges_riel, monthly_sewage_disposal_expense_riel,
                        monthly_garbage_collection_expense_riel, monthly_electricity_expense_riel,
                        monthly_gas_expense_riel, monthly_kerosene_expense_riel,
                        monthly_firewood_expense_riel, monthly_charcoal_expense_riel,
                        monthly_battery_expense_riel, monthly_other_energy_expense_riel,
                        monthly_rent_paid_riel, monthly_imputed_rent_riel,
                        dwelling_maintenance_expense_riel)
                   OR 999999999 IN (
                        monthly_water_charges_riel, monthly_sewage_disposal_expense_riel,
                        monthly_garbage_collection_expense_riel, monthly_electricity_expense_riel,
                        monthly_gas_expense_riel, monthly_kerosene_expense_riel,
                        monthly_firewood_expense_riel, monthly_charcoal_expense_riel,
                        monthly_battery_expense_riel, monthly_other_energy_expense_riel,
                        monthly_rent_paid_riel, monthly_imputed_rent_riel,
                        dwelling_maintenance_expense_riel)
                   OR (survey_wave='2004' AND monthly_imputed_rent_riel IS NOT NULL)
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_HO_CSES"))))
            require(invalid_ho == 0, f"Published HO domain/harmonization violations={invalid_ho}")

            matched_ec_orphans = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ec
                LEFT JOIN {}.{} hl USING (survey_wave, person_id)
                WHERE ec.hl_link_matched=1 AND hl.person_id IS NULL
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_EC_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
            )))
            false_unmatched_ec = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ec
                JOIN {}.{} hl USING (survey_wave, person_id)
                WHERE ec.hl_link_matched=0
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_EC_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
            )))
            require(matched_ec_orphans == 0 and false_unmatched_ec == 0, "Published EC link flags disagree with HL")
            cursor.execute(sql.SQL(
                "SELECT survey_wave, person_id FROM {}.{} WHERE hl_link_matched=0 ORDER BY survey_wave, person_id"
            ).format(sql.Identifier(args.schema), sql.Identifier("final_EC_CSES")))
            require(cursor.fetchall() == [("2014", "121641106"), ("2016", "050210206")], "Published unmatched EC records changed")

            ec_context_mismatches = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{} ec
                JOIN {}.{} hl USING (survey_wave, person_id)
                WHERE ec.hl_link_matched=1 AND (
                       ec.psu IS DISTINCT FROM hl.psu
                    OR ec.sex IS DISTINCT FROM hl.sex OR ec.age IS DISTINCT FROM hl.age
                    OR ec.province_code IS DISTINCT FROM hl.province_code
                    OR ec.district_code IS DISTINCT FROM hl.district_code
                    OR ec.commune_code IS DISTINCT FROM hl.commune_code
                    OR ec.village_code IS DISTINCT FROM hl.village_code
                    OR ec.urban_rural IS DISTINCT FROM hl.urban_rural
                    OR ec.survey_month IS DISTINCT FROM hl.survey_month
                    OR ec.stratum IS DISTINCT FROM hl.stratum
                    OR ec.household_weight IS DISTINCT FROM hl.household_weight
                    OR ec.person_weight IS DISTINCT FROM hl.person_weight
                )
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_EC_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
            )))
            require(ec_context_mismatches == 0, f"Published EC context differs from HL rows={ec_context_mismatches}")

            invalid_ec = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE hl_link_matched NOT IN (0,1) OR hl_link_matched IS NULL
                   OR (worked_at_least_one_hour_past_7_days IS NOT NULL AND worked_at_least_one_hour_past_7_days NOT IN (0,1))
                   OR (main_job_works_whole_year IS NOT NULL AND main_job_works_whole_year NOT IN (0,1))
                   OR (main_job_was_usual_past_7_days IS NOT NULL AND main_job_was_usual_past_7_days NOT IN (0,1))
                   OR (main_job_was_abroad IS NOT NULL AND main_job_was_abroad NOT IN (0,1))
                   OR (secondary_job_works_whole_year IS NOT NULL AND secondary_job_works_whole_year NOT IN (0,1))
                   OR (secondary_job_was_usual_past_7_days IS NOT NULL AND secondary_job_was_usual_past_7_days NOT IN (0,1))
                   OR (available_for_additional_work IS NOT NULL AND available_for_additional_work NOT IN (0,1))
                   OR (actively_seeking_work IS NOT NULL AND actively_seeking_work NOT IN (0,1))
                   OR (available_for_work IS NOT NULL AND available_for_work NOT IN (0,1))
                   OR (latest_work_seasonal IS NOT NULL AND latest_work_seasonal NOT IN (0,1))
                   OR (second_work_screening_source_code IS NOT NULL AND second_work_screening_source_code NOT IN (1,2))
                   OR (total_hours_worked_past_7_days IS NOT NULL AND (total_hours_worked_past_7_days NOT BETWEEN 0 AND 168 OR total_hours_worked_past_7_days IN (98,99)))
                   OR (main_hours_worked_past_7_days IS NOT NULL AND (main_hours_worked_past_7_days NOT BETWEEN 0 AND 168 OR main_hours_worked_past_7_days IN (98,99)))
                   OR (secondary_hours_worked_past_7_days IS NOT NULL AND (secondary_hours_worked_past_7_days NOT BETWEEN 0 AND 168 OR secondary_hours_worked_past_7_days IN (98,99)))
                   OR (desired_weekly_hours IS NOT NULL AND (desired_weekly_hours NOT BETWEEN 0 AND 168 OR desired_weekly_hours IN (98,99)))
                   OR (main_days_worked_last_month IS NOT NULL AND main_days_worked_last_month NOT BETWEEN 0 AND 31)
                   OR (secondary_days_worked_last_month IS NOT NULL AND secondary_days_worked_last_month NOT BETWEEN 0 AND 31)
                   OR (additional_jobs_count IS NOT NULL AND additional_jobs_count NOT BETWEEN 0 AND 10)
                   OR (total_occupations_past_7_days IS NOT NULL AND total_occupations_past_7_days NOT BETWEEN 0 AND 10)
                   OR monthly_salary_wages_riel < 0
                   OR monthly_salary_wages_riel IN (9999999,99999999,999999999)
                   OR (survey_wave IN ('2004','2007') AND monthly_salary_wages_riel IS NOT NULL)
                   OR (main_occupation_source_code IS NOT NULL AND main_occupation_source_code !~ '^[0-9]+$')
                   OR (secondary_occupation_source_code IS NOT NULL AND secondary_occupation_source_code !~ '^[0-9]+$')
                   OR (main_industry_source_code IS NOT NULL AND main_industry_source_code !~ '^[0-9]+$')
                   OR (secondary_industry_source_code IS NOT NULL AND secondary_industry_source_code !~ '^[0-9]+$')
                   OR (total_occupations_past_7_days < 2 AND (
                        secondary_occupation_source_code IS NOT NULL
                     OR secondary_industry_source_code IS NOT NULL
                     OR secondary_employer_type_source_code IS NOT NULL
                     OR secondary_employment_status_source_code IS NOT NULL
                     OR secondary_hours_worked_past_7_days IS NOT NULL
                     OR secondary_days_worked_last_month IS NOT NULL
                     OR secondary_job_works_whole_year IS NOT NULL
                     OR secondary_job_was_usual_past_7_days IS NOT NULL))
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_EC_CSES"))))
            require(invalid_ec == 0, f"Published EC domain/structural-null violations={invalid_ec}")

            invalid_vl = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE hh_psu_link_matched IS DISTINCT FROM 1
                   OR sample_household_count < 0 OR sample_person_count < 0
                   OR village_household_count < 0 OR enumeration_area_count < 0
                   OR households_in_surveyed_enumeration_area < 0 OR village_person_count < 0
                   OR village_male_count < 0 OR village_female_count < 0
                   OR population_below_18_count < 0 OR boys_below_18_count < 0
                   OR girls_below_18_count < 0 OR population_18_plus_count < 0
                   OR men_18_plus_count < 0 OR women_18_plus_count < 0
                   OR village_household_count_five_years_ago < 0
                   OR village_person_count_five_years_ago < 0
                   OR (village_land_area_square_kilometers IS NOT NULL AND village_land_area_square_kilometers NOT BETWEEN 0.0000001 AND 100000)
                   OR (five_year_population_movement_source_code IS NOT NULL AND five_year_population_movement_source_code NOT IN (1,2,3,4))
                   OR (village_reference_day IS NOT NULL AND village_reference_day NOT BETWEEN 1 AND 31)
                   OR (village_reference_month IS NOT NULL AND village_reference_month NOT BETWEEN 1 AND 12)
                   OR (village_reference_year IS NOT NULL AND village_reference_year NOT BETWEEN 2000 AND 2022)
                   OR (survey_wave='2019' AND (
                        village_reference_day IS NOT NULL OR village_reference_month IS NOT NULL
                     OR village_reference_year IS NOT NULL OR enumeration_area_count IS NOT NULL
                     OR households_in_surveyed_enumeration_area IS NOT NULL
                     OR population_below_18_count IS NOT NULL OR boys_below_18_count IS NOT NULL
                     OR girls_below_18_count IS NOT NULL OR population_18_plus_count IS NOT NULL
                     OR men_18_plus_count IS NOT NULL OR women_18_plus_count IS NOT NULL
                     OR village_land_area_square_kilometers IS NOT NULL
                     OR five_year_population_movement_source_code IS NOT NULL
                     OR village_household_count_five_years_ago IS NOT NULL
                     OR village_person_count_five_years_ago IS NOT NULL))
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_VL_CSES"))))
            require(invalid_vl == 0, f"Published VL domain/structural-null violations={invalid_vl}")

            vl_context_mismatches = int(scalar(cursor, sql.SQL(
                """
                WITH hh AS (
                    SELECT survey_wave, psu,
                           CASE WHEN count(DISTINCT province_code)=1 THEN min(province_code) END AS province_code,
                           CASE WHEN count(DISTINCT district_code)=1 THEN min(district_code) END AS district_code,
                           CASE WHEN count(DISTINCT commune_code)=1 THEN min(commune_code) END AS commune_code,
                           CASE WHEN count(DISTINCT village_code)=1 THEN min(village_code) END AS village_code,
                           CASE WHEN count(DISTINCT urban_rural)=1 THEN min(urban_rural) END AS urban_rural,
                           CASE WHEN count(DISTINCT survey_month)=1 THEN min(survey_month) END AS survey_month,
                           CASE WHEN count(DISTINCT stratum)=1 THEN min(stratum) END AS stratum,
                           count(*) AS sample_household_count
                    FROM {}.{} GROUP BY 1,2
                ), persons AS (
                    SELECT survey_wave, psu, count(*) AS sample_person_count
                    FROM {}.{} GROUP BY 1,2
                )
                SELECT count(*) FROM {}.{} vl
                LEFT JOIN hh USING (survey_wave, psu)
                LEFT JOIN persons USING (survey_wave, psu)
                WHERE hh.psu IS NULL
                   OR vl.province_code IS DISTINCT FROM hh.province_code
                   OR vl.district_code IS DISTINCT FROM hh.district_code
                   OR vl.commune_code IS DISTINCT FROM hh.commune_code
                   OR vl.village_code IS DISTINCT FROM hh.village_code
                   OR vl.urban_rural IS DISTINCT FROM hh.urban_rural
                   OR vl.survey_month IS DISTINCT FROM hh.survey_month
                   OR vl.stratum IS DISTINCT FROM hh.stratum
                   OR vl.sample_household_count IS DISTINCT FROM hh.sample_household_count
                   OR vl.sample_person_count IS DISTINCT FROM persons.sample_person_count
                """
            ).format(
                sql.Identifier(args.schema), sql.Identifier("final_HH_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_HL_CSES"),
                sql.Identifier(args.schema), sql.Identifier("final_VL_CSES"),
            )))
            require(vl_context_mismatches == 0, f"Published VL context differs from HH/HL rows={vl_context_mismatches}")
            vl_age_mismatches = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE population_below_18_count IS NOT NULL AND population_18_plus_count IS NOT NULL
                  AND village_person_count IS NOT NULL
                  AND population_below_18_count + population_18_plus_count <> village_person_count
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_VL_CSES"))))
            vl_sex_mismatches = int(scalar(cursor, sql.SQL(
                """
                SELECT count(*) FROM {}.{}
                WHERE village_male_count IS NOT NULL AND village_female_count IS NOT NULL
                  AND village_person_count IS NOT NULL
                  AND village_male_count + village_female_count <> village_person_count
                """
            ).format(sql.Identifier(args.schema), sql.Identifier("final_VL_CSES"))))
            require(vl_age_mismatches == 120 and vl_sex_mismatches == 116, "Published VL documented component mismatch counts changed")

            for table, (_rows, columns) in TABLE_CONTRACTS.items():
                if not table.startswith("final_"):
                    continue
                commented = int(scalar(
                    cursor,
                    """
                    SELECT count(*)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid=a.attrelid
                    JOIN pg_namespace n ON n.oid=c.relnamespace
                    JOIN pg_description d ON d.objoid=c.oid AND d.objsubid=a.attnum
                    WHERE n.nspname=%s AND c.relname=%s AND a.attnum > 0 AND NOT a.attisdropped
                    """,
                    (args.schema, table),
                ))
                require(commented == len(columns), f"{table} commented columns={commented}, expected={len(columns)}")
                table_comment = scalar(cursor, "SELECT obj_description(to_regclass(%s))", (f'{args.schema}."{table}"',))
                require(bool(table_comment), f"{table} lacks a table comment")

            cursor.execute("SELECT indexname FROM pg_indexes WHERE schemaname=%s", (args.schema,))
            indexes = {row[0] for row in cursor.fetchall()}
            require(EXPECTED_INDEXES.issubset(indexes), f"Missing indexes: {sorted(EXPECTED_INDEXES - indexes)}")

            cursor.execute(
                """
                SELECT table_name, module, row_count, n_datasets, ind_que_table
                FROM public._catalog
                WHERE table_name = ANY(%s)
                ORDER BY table_name
                """,
                (["final_HH_CSES", "final_HL_CSES", "final_ED_CSES", "final_HO_CSES", "final_EC_CSES", "final_VL_CSES"],),
            )
            catalog = cursor.fetchall()
            require(catalog == [
                ("final_EC_CSES", "EC", 332_903, 10, "ind_que_EC_CSES"),
                ("final_ED_CSES", "ED", 343_204, 10, "ind_que_ED_CSES"),
                ("final_HH_CSES", "HH", 77_904, 10, "ind_que_HH_CSES"),
                ("final_HL_CSES", "HL", 358_920, 10, "ind_que_HL_CSES"),
                ("final_HO_CSES", "HO", 77_922, 10, "ind_que_HO_CSES"),
                ("final_VL_CSES", "VL", 5_718, 8, "ind_que_VL_CSES"),
            ], f"Catalog entries differ: {catalog}")
            staging = int(scalar(
                cursor,
                "SELECT count(*) FROM pg_tables WHERE schemaname=%s AND tablename LIKE '_staging_%%'",
                (args.schema,),
            ))
            require(staging == 0, f"Staging tables remain after publish: {staging}")

    print("database_contracts=18/18")
    print("database_row_counts=valid")
    print("database_unique_keys=valid")
    print("database_hl_to_hh_integrity=valid")
    print("database_hh_composition_and_head_links=valid")
    print("database_hh_head_education_links=valid")
    print("database_hl_extended_domains_and_kinship_links=valid")
    print("database_ed_to_hl_flags=valid")
    print("database_ed_domains=valid")
    print("database_ho_to_hh_flags_and_context=valid")
    print("database_ho_domains_and_harmonization=valid")
    print("database_ec_to_hl_flags_context_and_domains=valid")
    print("database_vl_to_hh_context_and_domains=valid")
    print("database_comments_and_indexes=valid")
    print("database_catalog=6/6")
    print("database_staging_tables=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
