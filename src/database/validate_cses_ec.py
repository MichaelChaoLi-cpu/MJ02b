#!/usr/bin/env python3
"""Validate the CSES current-employment database build."""

from pathlib import Path

import pandas as pd

from cses_employment import (
    DURATION_COLUMNS,
    EC_COLUMNS,
    HOUR_COLUMNS,
    MONEY_SENTINELS,
    SECONDARY_COLUMNS,
    YES_NO_COLUMNS,
)
from cses_hh_hl_common import WAVES, snake_case


EXPECTED_WAVE_ROWS = {
    "2004": 74_719, "2007": 15_766, "2009": 51_460, "2011-12": 14_829,
    "2013": 15_774, "2014": 49_252, "2016": 15_498, "2017": 15_482,
    "2019": 40_379, "2021": 39_744,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_equal(left: pd.Series, right: pd.Series, message: str) -> None:
    equal = left.eq(right) | (left.isna() & right.isna())
    require(bool(equal.fillna(False).all()), f"{message}: mismatches={int((~equal.fillna(False)).sum())}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data/exp/database"
    ec = pd.read_parquet(output / "final_EC_CSES.parquet")
    hl = pd.read_parquet(output / "final_HL_CSES.parquet")
    reporting = output
    audit = pd.read_csv(reporting / "cses_ec_alignment_audit.csv", dtype={"survey_wave": "string"})
    issues = pd.read_csv(reporting / "cses_ec_data_issues.csv", dtype={"survey_wave": "string"})
    dictionary = pd.read_csv(reporting / "ind_que_EC_CSES.csv", dtype="string")
    summary = pd.read_csv(reporting / "align_summary_EC_CSES.csv")

    require(list(ec.columns) == EC_COLUMNS, "EC columns or order differ from the contract")
    require(set(ec["Survey Wave"].dropna()) == set(WAVES), "EC wave coverage is incomplete")
    require(not ec[["Survey Wave", "Person ID"]].isna().any().any(), "EC key contains nulls")
    require(not ec.duplicated(["Survey Wave", "Person ID"]).any(), "EC wave-person key is not unique")
    require(ec.groupby("Survey Wave", observed=True).size().to_dict() == EXPECTED_WAVE_ROWS, "EC wave row counts changed")

    hl_check = hl[[
        "Survey Wave", "Person ID", "Household ID", "PSU", "Sex", "Age",
        "Province Code", "District Code", "Commune Code", "Village Code",
        "Urban Rural", "Survey Month", "Stratum", "Household Weight", "Person Weight",
    ]].rename(columns=lambda column: column if column in {"Survey Wave", "Person ID"} else f"{column}_hl")
    linked = ec.merge(hl_check, on=["Survey Wave", "Person ID"], how="left", validate="1:1", indicator=True)
    require(linked.loc[linked["HL Link Matched"].eq(1), "_merge"].eq("both").all(), "A matched EC row is absent from HL")
    require(linked.loc[linked["HL Link Matched"].eq(0), "_merge"].eq("left_only").all(), "An unmatched EC row is present in HL")
    require(int(ec["HL Link Matched"].eq(0).sum()) == 2, "EC/HL unmatched-record count changed")
    unmatched = ec.loc[ec["HL Link Matched"].eq(0), ["Survey Wave", "Person ID"]].sort_values(["Survey Wave", "Person ID"])
    require(unmatched.itertuples(index=False, name=None).__iter__() is not None, "Unable to enumerate unmatched EC records")
    require(list(unmatched.itertuples(index=False, name=None)) == [("2014", "121641106"), ("2016", "050210206")], "Unmatched EC records changed")
    matched = linked.loc[linked["HL Link Matched"].eq(1)]
    for column in [
        "PSU", "Sex", "Age", "Province Code", "District Code", "Commune Code",
        "Village Code", "Urban Rural", "Survey Month", "Stratum",
        "Household Weight", "Person Weight",
    ]:
        require_equal(matched[column], matched[f"{column}_hl"], f"EC/HL {column} differs")

    for column in YES_NO_COLUMNS:
        require(ec[column].dropna().isin([0, 1]).all(), f"{column} contains an invalid indicator")
    require(ec["Second Work Screening Source Code"].dropna().isin([1, 2]).all(), "Second work screen code is invalid")
    for column in HOUR_COLUMNS:
        require(ec[column].dropna().between(0, 168).all(), f"{column} is out of range")
        require(not ec[column].dropna().isin([98, 99]).any(), f"{column} retains a missing sentinel")
    for column in ["Main Days Worked Last Month", "Secondary Days Worked Last Month"]:
        require(ec[column].dropna().between(0, 31).all(), f"{column} is out of range")
    for column in DURATION_COLUMNS:
        require(ec[column].dropna().between(0, 97).all(), f"{column} is out of range")
        require(not ec[column].dropna().isin([98, 99]).any(), f"{column} retains a duration sentinel")
    require(ec["Additional Jobs Count"].dropna().between(0, 10).all(), "Additional jobs count is out of range")
    require(ec["Total Occupations Past 7 Days"].dropna().between(0, 10).all(), "Total occupations is out of range")
    later = ec.loc[~ec["Survey Wave"].isin(["2004", "2007"]) & ec["Main Occupation Source Code"].notna()]
    require_equal(later["Total Occupations Past 7 Days"], (later["Additional Jobs Count"] + 1).astype("Int16"), "Later-wave total occupations differs from additional jobs")
    fewer_than_two = ec["Total Occupations Past 7 Days"].notna() & ec["Total Occupations Past 7 Days"].lt(2)
    require(not ec.loc[fewer_than_two, list(SECONDARY_COLUMNS)].notna().any().any(), "Secondary-job values remain when fewer than two occupations were reported")
    require(ec["Monthly Salary Wages Riel"].dropna().ge(0).all(), "Monthly wages contain a negative value")
    require(not ec["Monthly Salary Wages Riel"].dropna().isin(MONEY_SENTINELS).any(), "Monthly wages retain a sentinel")
    require(ec.loc[ec["Survey Wave"].isin(["2004", "2007"]), "Monthly Salary Wages Riel"].isna().all(), "Unexpected 2004/2007 monthly wage values")
    for column in ["Main Occupation Source Code", "Secondary Occupation Source Code", "Main Industry Source Code", "Secondary Industry Source Code"]:
        require(ec[column].dropna().str.fullmatch(r"\d+").all(), f"{column} contains a nonnumeric string")

    require(int(audit["rows"].sum()) == len(ec), "EC audit row total is wrong")
    require(int(audit["duplicate_key_rows"].sum()) == 0, "EC audit reports duplicate keys")
    require(int(audit["missing_person_id"].sum()) == 0, "EC audit reports missing person IDs")
    require(int(audit["hl_link_unmatched"].sum()) == 2, "EC audit unmatched count is wrong")
    require(int(issues.loc[issues["issue_type"].eq("employment_record_not_in_hl"), "affected_rows"].sum()) == 2, "EC issues do not enumerate unmatched records")
    require(len(summary) == len(EC_COLUMNS), "EC summary does not have one row per column")
    require(set(summary["varname"]) == {snake_case(column) for column in EC_COLUMNS}, "EC summary variable set is wrong")
    require(set(dictionary["canonical_varname"]).issubset(set(summary["varname"])), "EC dictionary contains an unknown variable")
    require(dictionary["dataset_name"].nunique() == len(WAVES), "EC dictionary does not cover all waves")

    print(f"validated_ec_rows={len(ec)}")
    print(f"waves={len(WAVES)}")
    print("ec_key_unique=yes")
    print(f"hl_link_matched={int(ec['HL Link Matched'].eq(1).sum())}")
    print(f"hl_link_unmatched={int(ec['HL Link Matched'].eq(0).sum())}")
    print("matched_ec_to_hl_context_consistency=yes")
    print("employment_domains_and_structural_nulls=valid")
    print(f"variable_dictionary_rows={len(dictionary)}")
    print(f"alignment_summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
