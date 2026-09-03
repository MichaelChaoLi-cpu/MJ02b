#!/usr/bin/env python3
"""Validate the CSES HH/HL database build before publication."""

from pathlib import Path

import pandas as pd

from cses_hh_hl_common import HH_COLUMNS, HL_COLUMNS, WAVE_YEAR, snake_case


WAVES = {"2004", "2007", "2009", "2011-12", "2013", "2014", "2016", "2017", "2019", "2021"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_nullable_equal(left: pd.Series, right: pd.Series, message: str) -> None:
    equal = left.eq(right) | (left.isna() & right.isna())
    require(bool(equal.fillna(False).all()), message)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data/exp/database"
    hh = pd.read_parquet(output / "final_HH_CSES.parquet")
    hl = pd.read_parquet(output / "final_HL_CSES.parquet")
    ed = pd.read_parquet(output / "final_ED_CSES.parquet")
    reporting = output
    hh_audit = pd.read_csv(reporting / "cses_hh_alignment_audit.csv", dtype={"survey_wave": "string"})
    hl_audit = pd.read_csv(reporting / "cses_hl_alignment_audit.csv", dtype={"survey_wave": "string"})
    hl_issues = pd.read_csv(reporting / "cses_hl_data_issues.csv", dtype={"survey_wave": "string"})
    hh_dictionary = pd.read_csv(reporting / "ind_que_HH_CSES.csv", dtype="string")
    hl_dictionary = pd.read_csv(reporting / "ind_que_HL_CSES.csv", dtype="string")
    hh_summary = pd.read_csv(reporting / "align_summary_HH_CSES.csv")
    hl_summary = pd.read_csv(reporting / "align_summary_HL_CSES.csv")

    require(list(hh.columns) == HH_COLUMNS, "HH columns or column order differ from the contract")
    require(list(hl.columns) == HL_COLUMNS, "HL columns or column order differ from the contract")
    require(set(hh["Survey Wave"].dropna()) == WAVES, "HH wave coverage is incomplete")
    require(set(hl["Survey Wave"].dropna()) == WAVES, "HL wave coverage is incomplete")
    require(not hh[["Survey Wave", "Household ID"]].isna().any().any(), "HH key contains nulls")
    require(not hl[["Survey Wave", "Household ID", "Person ID"]].isna().any().any(), "HL key contains nulls")
    require(not hh.duplicated(["Survey Wave", "Household ID"]).any(), "HH key is not unique")
    require(not hl.duplicated(["Survey Wave", "Person ID"]).any(), "HL key is not unique")

    hh_keys = pd.MultiIndex.from_frame(hh[["Survey Wave", "Household ID"]])
    hl_keys = pd.MultiIndex.from_frame(hl[["Survey Wave", "Household ID"]])
    require(hl_keys.isin(hh_keys).all(), "HL contains a household not present in HH")

    require(hl["Sex"].dropna().isin([1, 2]).all(), "Sex contains an invalid code")
    require(hl["Age"].dropna().between(0, 120).all(), "Age falls outside 0-120")
    age_2004 = hl.loc[hl["Survey Wave"].eq("2004"), "Age"]
    require(not age_2004.dropna().isin([98, 99]).any(), "Documented 2004 age sentinel remains numeric")
    require(int(age_2004.isna().sum()) == 1, "Unexpected number of missing 2004 ages")
    require(hl["Member Line Number"].dropna().between(1, 98).all(), "Member line falls outside 1-98")
    member_line_keys = hl[["Survey Wave", "Household ID", "Member Line Number"]]
    require(not member_line_keys.isna().any().any(), "Member line key contains nulls")
    require(not member_line_keys.duplicated().any(), "Wave-household-member-line is not unique")
    person_suffix = pd.to_numeric(hl["Person ID"].str[-2:], errors="coerce").astype("Int16")
    require(hl["Member Line Number"].eq(person_suffix).all(), "Member line differs from person-ID suffix")
    require(hl["Birth Day"].dropna().between(1, 31).all(), "Birth day falls outside 1-31")
    require(hl["Birth Month"].dropna().between(1, 12).all(), "Birth month falls outside 1-12")
    valid_birth_year = hl.apply(
        lambda row: pd.isna(row["Birth Year"]) or 1800 <= row["Birth Year"] <= WAVE_YEAR[row["Survey Wave"]] + 1,
        axis=1,
    )
    require(bool(valid_birth_year.all()), "Birth year falls outside its wave-specific range")
    require(hl["Relationship to Household Head"].dropna().isin(range(1, 16)).all(), "Relationship contains an invalid code")
    require(hl["Absent From Household"].dropna().isin([0, 1]).all(), "Absence indicator contains an invalid code")
    require(set(hl.loc[hl["Survey Wave"].eq("2004"), "Presence Reference Period"]) == {"Current status"}, "2004 presence reference period is wrong")
    require(set(hl.loc[hl["Survey Wave"].ne("2004"), "Presence Reference Period"]) == {"Past 7 days"}, "Later presence reference period is wrong")
    heads = hl.assign(_head=hl["Relationship to Household Head"].eq(1)).groupby(["Survey Wave", "Household ID"])["_head"].sum()
    require(not heads.gt(1).any(), "A household contains multiple coded heads")

    marital_2004 = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4}
    marital_later = {1: 2, 2: 4, 3: 3, 4: 1}
    expected_marital = pd.Series(pd.NA, index=hl.index, dtype="Int8")
    is_2004 = hl["Survey Wave"].eq("2004")
    expected_marital.loc[is_2004] = hl.loc[is_2004, "Marital Status Source Code"].map(marital_2004).astype("Int8")
    expected_marital.loc[~is_2004] = hl.loc[~is_2004, "Marital Status Source Code"].map(marital_later).astype("Int8")
    require(
        (hl["Marital Status Harmonized"].eq(expected_marital) | (hl["Marital Status Harmonized"].isna() & expected_marital.isna())).all(),
        "Harmonized marital status differs from the wave-specific mapping",
    )
    require(hl["Ethnicity Source Code"].dropna().isin(range(1, 9)).all(), "Ethnicity source code is invalid")
    require(
        (hl["Ethnicity Harmonized"].eq(hl["Ethnicity Source Code"]) | (hl["Ethnicity Harmonized"].isna() & hl["Ethnicity Source Code"].isna())).all(),
        "Harmonized ethnicity differs from the stable 1-8 source coding",
    )

    for role in ["Father", "Mother", "Spouse"]:
        line_column = f"{role} Line Number"
        person_column = f"{role} Person ID"
        require(hl.loc[hl[person_column].notna(), line_column].notna().all(), f"{role} person ID lacks a source line")
        require(not hl.loc[hl[person_column].notna(), person_column].eq(hl.loc[hl[person_column].notna(), "Person ID"]).any(), f"{role} link contains a self-reference")
        targets = hl[["Survey Wave", "Household ID", "Member Line Number", "Person ID"]].rename(columns={
            "Member Line Number": line_column,
            "Person ID": f"_{role.lower()}_expected_person_id",
        })
        check = hl[["Survey Wave", "Household ID", line_column, person_column]].merge(
            targets,
            on=["Survey Wave", "Household ID", line_column],
            how="left",
            validate="m:1",
        )
        linked = check[person_column].notna()
        require(
            check.loc[linked, person_column].eq(check.loc[linked, f"_{role.lower()}_expected_person_id"]).all(),
            f"{role} person ID does not match its household roster line",
        )

    hh_counts = hh.groupby("Survey Wave")["Household ID"].nunique()
    hl_hh_counts = hl.groupby("Survey Wave")["Household ID"].nunique()
    require(hh_counts.equals(hl_hh_counts), "HH counts do not match distinct HL household counts")

    household_keys = ["Survey Wave", "Household ID"]
    composition_source = hl.assign(
        _male=hl["Sex"].eq(1).fillna(False).astype("int8"),
        _female=hl["Sex"].eq(2).fillna(False).astype("int8"),
        _child=hl["Age"].between(0, 14).fillna(False).astype("int8"),
        _working_age=hl["Age"].between(15, 64).fillna(False).astype("int8"),
        _older=hl["Age"].ge(65).fillna(False).astype("int8"),
        _unknown_age=hl["Age"].isna().astype("int8"),
    )
    composition = composition_source.groupby(household_keys, as_index=False).agg(
        _member_count=("Person ID", "size"),
        _male_count=("_male", "sum"),
        _female_count=("_female", "sum"),
        _child_count=("_child", "sum"),
        _working_age_count=("_working_age", "sum"),
        _older_count=("_older", "sum"),
        _unknown_age_count=("_unknown_age", "sum"),
    )
    composition_check = hh.merge(composition, on=household_keys, how="left", validate="1:1")
    composition_pairs = {
        "Household Member Count": "_member_count",
        "Male Member Count": "_male_count",
        "Female Member Count": "_female_count",
        "Child Member Count 0-14": "_child_count",
        "Working Age Member Count 15-64": "_working_age_count",
        "Older Member Count 65 Plus": "_older_count",
        "Unknown Age Member Count": "_unknown_age_count",
    }
    for actual, expected in composition_pairs.items():
        require_nullable_equal(composition_check[actual], composition_check[expected], f"{actual} differs from an independent HL aggregation")
    require(
        hh["Male Member Count"].add(hh["Female Member Count"]).eq(hh["Household Member Count"]).all(),
        "Male and female member counts do not sum to household size",
    )
    require(
        hh[["Child Member Count 0-14", "Working Age Member Count 15-64", "Older Member Count 65 Plus", "Unknown Age Member Count"]]
        .sum(axis=1).eq(hh["Household Member Count"]).all(),
        "Age-group member counts do not sum to household size",
    )

    head_fields = {
        "Household Head Person ID": "Person ID",
        "Household Head Sex": "Sex",
        "Household Head Age": "Age",
        "Household Head Marital Status": "Marital Status Harmonized",
        "Household Head Ethnicity": "Ethnicity Harmonized",
    }
    expected_heads = hl.loc[
        hl["Relationship to Household Head"].eq(1),
        household_keys + list(head_fields.values()),
    ].rename(columns={source: f"_expected_{snake_case(target)}" for target, source in head_fields.items()})
    head_check = hh.merge(expected_heads, on=household_keys, how="left", validate="1:1")
    for actual in head_fields:
        require_nullable_equal(
            head_check[actual], head_check[f"_expected_{snake_case(actual)}"],
            f"{actual} differs from the uniquely coded HL head",
        )
    require(int(hh["Household Head Person ID"].isna().sum()) == 1, "Unexpected number of households without a unique coded head")

    education_fields = {
        "Household Head Education Level": "Education Level Harmonized",
        "Household Head Years Attended School": "Years Attended School",
        "Household Head Can Read": "Can Read",
        "Household Head Can Write": "Can Write",
    }
    expected_education = ed[["Survey Wave", "Person ID"] + list(education_fields.values())].rename(
        columns={source: f"_expected_{snake_case(target)}" for target, source in education_fields.items()}
    )
    education_check = hh.merge(
        expected_education,
        left_on=["Survey Wave", "Household Head Person ID"],
        right_on=["Survey Wave", "Person ID"],
        how="left",
        validate="m:1",
    )
    for actual in education_fields:
        require_nullable_equal(
            education_check[actual], education_check[f"_expected_{snake_case(actual)}"],
            f"{actual} differs from the head's ED record",
        )

    for frame, column in ((hh, "Household Weight"), (hl, "Household Weight"), (hl, "Person Weight")):
        non_2004 = frame.loc[frame["Survey Wave"].ne("2004"), column]
        require(non_2004.notna().all(), f"{column} is missing outside 2004")
        require((non_2004 > 0).all(), f"{column} contains a non-positive value")
        require(frame.loc[frame["Survey Wave"].eq("2004"), column].isna().all(), f"Unexpected 2004 {column}")

    require(int(hh_audit["rows"].sum()) == len(hh), "HH audit row total is wrong")
    require(int(hl_audit["rows"].sum()) == len(hl), "HL audit row total is wrong")
    require(len(hh_summary) == len(HH_COLUMNS), "HH alignment summary does not have one row per column")
    require(set(hh_summary["varname"]) == {snake_case(column) for column in HH_COLUMNS}, "HH alignment summary variable set is wrong")
    require(len(hl_summary) == len(HL_COLUMNS), "HL alignment summary does not have one row per column")
    require(set(hl_summary["varname"]) == {snake_case(column) for column in HL_COLUMNS}, "HL alignment summary variable set is wrong")
    require(len(hh_dictionary) == 333, f"Unexpected HH variable-dictionary rows={len(hh_dictionary)}")
    require(hh_dictionary["dataset_name"].nunique() == len(WAVES), "HH variable dictionary does not cover all waves")
    require(len(hl_dictionary) == 384, f"Unexpected HL variable-dictionary rows={len(hl_dictionary)}")
    require(hl_dictionary["dataset_name"].nunique() == len(WAVES), "HL variable dictionary does not cover all waves")
    expected_unresolved_links = {
        "Father Person ID": 184,
        "Mother Person ID": 784,
        "Spouse Person ID": 2181,
    }
    for variable, expected in expected_unresolved_links.items():
        actual = int(hl_issues.loc[
            hl_issues["variable"].eq(variable)
            & hl_issues["issue_type"].isin(["kinship_line_not_in_household", "kinship_self_reference"]),
            "affected_rows",
        ].sum())
        require(actual == expected, f"Unexpected unresolved {variable} count={actual}")

    print(f"validated_hh_rows={len(hh)}")
    print(f"validated_hl_rows={len(hl)}")
    print(f"waves={len(WAVES)}")
    print("hh_key_unique=yes")
    print("hl_key_unique=yes")
    print("hl_to_hh_referential_integrity=yes")
    print("hh_composition_recomputed_from_hl=yes")
    print("hh_head_attributes_recomputed_from_hl=yes")
    print("hh_head_education_recomputed_from_ed=yes")
    print("hl_member_line_key_and_person_suffix=consistent")
    print("hl_birth_components=valid")
    print("hl_marital_and_ethnicity_harmonization=valid")
    print("hl_kinship_person_links=valid_nonself_targets")
    print("released_weights_positive_and_complete_for_2007_2021=yes")
    print("2004_general_weights=not_released_in_selected_core_sources")
    print(f"missing_relationship_values={int(hl['Relationship to Household Head'].isna().sum())}")
    print(f"missing_absence_status_values={int(hl['Absent From Household'].isna().sum())}")
    print(f"households_without_one_coded_head={int(heads.ne(1).sum())}")
    print(f"hh_variable_dictionary_rows={len(hh_dictionary)}")
    print(f"hh_alignment_summary_rows={len(hh_summary)}")
    print(f"hl_variable_dictionary_rows={len(hl_dictionary)}")
    print(f"hl_alignment_summary_rows={len(hl_summary)}")


if __name__ == "__main__":
    main()
