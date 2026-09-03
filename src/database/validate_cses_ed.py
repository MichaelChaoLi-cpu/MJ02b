#!/usr/bin/env python3
"""Validate the CSES education database build before publication."""

from pathlib import Path

import pandas as pd

from cses_education import ED_COLUMNS
from cses_hh_hl_common import WAVES, snake_case


EXPECTED_WAVE_ROWS = {
    "2004": 74_719,
    "2007": 15_789,
    "2009": 53_647,
    "2011-12": 15_469,
    "2013": 16_389,
    "2014": 51_221,
    "2016": 16_093,
    "2017": 16_110,
    "2019": 42_308,
    "2021": 41_459,
}

EXPECTED_UNMATCHED_PERSONS = {
    ("2013", "121010900"),
    ("2014", "020240101"),
    ("2014", "020240102"),
    ("2014", "020240103"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_equal(left: pd.Series, right: pd.Series, message: str) -> None:
    equal = left.eq(right) | (left.isna() & right.isna())
    require(bool(equal.all()), f"{message}: mismatches={int((~equal).sum())}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data/exp/database"
    ed = pd.read_parquet(output / "final_ED_CSES.parquet")
    hl = pd.read_parquet(output / "final_HL_CSES.parquet")
    reporting = output
    audit = pd.read_csv(reporting / "cses_ed_alignment_audit.csv", dtype={"survey_wave": "string"})
    issues = pd.read_csv(reporting / "cses_ed_data_issues.csv", dtype={"survey_wave": "string"})
    dictionary = pd.read_csv(reporting / "ind_que_ED_CSES.csv", dtype="string")
    summary = pd.read_csv(reporting / "align_summary_ED_CSES.csv")

    require(list(ed.columns) == ED_COLUMNS, "ED columns or column order differ from the contract")
    require(set(ed["Survey Wave"].dropna()) == set(WAVES), "ED wave coverage is incomplete")
    require(not ed[["Survey Wave", "Household ID", "Person ID"]].isna().any().any(), "ED key contains nulls")
    require(not ed.duplicated(["Survey Wave", "Person ID"]).any(), "ED wave-person key is not unique")
    counts = ed.groupby("Survey Wave", observed=True).size().to_dict()
    require(counts == EXPECTED_WAVE_ROWS, f"ED wave row counts changed: {counts}")

    require(ed["HL Link Matched"].dropna().isin([0, 1]).all(), "HL link flag contains an invalid code")
    require(ed["HL Link Matched"].notna().all(), "HL link flag contains nulls")
    unmatched = set(
        ed.loc[ed["HL Link Matched"].eq(0), ["Survey Wave", "Person ID"]]
        .itertuples(index=False, name=None)
    )
    require(unmatched == EXPECTED_UNMATCHED_PERSONS, f"Unexpected unmatched education records: {unmatched}")

    hl_check = hl[[
        "Survey Wave", "Person ID", "Household ID", "PSU", "Sex", "Age",
        "Province Code", "District Code", "Commune Code", "Village Code",
        "Urban Rural", "Survey Month", "Stratum", "Household Weight", "Person Weight",
    ]].rename(columns=lambda column: column if column in {"Survey Wave", "Person ID"} else f"{column}_hl")
    linked = ed.merge(hl_check, on=["Survey Wave", "Person ID"], how="left", validate="1:1", indicator=True)
    require(linked.loc[linked["HL Link Matched"].eq(1), "_merge"].eq("both").all(), "A matched ED row is absent from HL")
    require(linked.loc[linked["HL Link Matched"].eq(0), "_merge"].eq("left_only").all(), "An unmatched ED row is actually present in HL")
    matched = linked[linked["HL Link Matched"].eq(1)]
    for column in [
        "Household ID", "PSU", "Sex", "Age", "Province Code", "District Code",
        "Commune Code", "Village Code", "Urban Rural", "Survey Month", "Stratum",
        "Household Weight", "Person Weight",
    ]:
        require_equal(matched[column], matched[f"{column}_hl"], f"ED/HL {column} differs")

    for column in ["Can Read", "Can Write", "Ever Attended School", "Currently Attending School"]:
        require(ed[column].dropna().isin([0, 1]).all(), f"{column} contains an invalid code")
    require(ed["Years Attended School"].dropna().between(0, 30).all(), "Years attended falls outside 0-30")
    require(ed["Education Level Harmonized"].dropna().isin(range(0, 8)).all(), "Highest education level contains an invalid code")
    require(ed["Current Education Level Harmonized"].dropna().isin(range(1, 8)).all(), "Current education level contains an invalid code")
    current_level_without_confirmed_attendance = (
        ed["Current Education Level Harmonized"].notna()
        & ~ed["Currently Attending School"].eq(1).fillna(False)
    )
    require(
        not current_level_without_confirmed_attendance.any(),
        "A harmonized current education level is present without confirmed current attendance",
    )
    require(ed.loc[ed["Survey Wave"].eq("2004"), "Years Attended School"].isna().all(), "Unexpected 2004 completed-years values")

    matched_after_2004 = ed[ed["HL Link Matched"].eq(1) & ed["Survey Wave"].ne("2004")]
    for column in ["Household Weight", "Person Weight"]:
        require(matched_after_2004[column].notna().all(), f"Matched post-2004 {column} contains nulls")
        require(matched_after_2004[column].gt(0).all(), f"Matched post-2004 {column} contains non-positive values")
        require(ed.loc[ed["Survey Wave"].eq("2004"), column].isna().all(), f"Unexpected 2004 {column}")

    require(int(audit["rows"].sum()) == len(ed), "Audit total does not match ED row count")
    require(int(audit["hl_link_unmatched"].sum()) == len(unmatched), "Audit unmatched count is wrong")
    require(int(audit["duplicate_key_rows"].sum()) == 0, "Audit reports duplicate keys")
    require(int(audit["missing_person_id"].sum()) == 0, "Audit reports missing person IDs")
    require(len(summary) == len(ED_COLUMNS), "Alignment summary does not have one row per ED column")
    require(set(summary["varname"]) == {snake_case(column) for column in ED_COLUMNS}, "Alignment summary variable set is wrong")
    require(set(dictionary["canonical_varname"]).issubset(set(summary["varname"])), "Variable dictionary contains an unknown ED variable")
    require(dictionary["dataset_name"].nunique() == len(WAVES), "Variable dictionary does not cover all waves")
    require(
        int(issues.loc[issues["issue_type"].eq("education_record_not_in_hl"), "affected_rows"].sum()) == 4,
        "Issue report does not identify the four unmatched education records",
    )
    require(
        int(issues.loc[issues["issue_type"].eq("current_level_reported_without_confirmed_attendance"), "affected_rows"].sum()) == 18,
        "Issue report does not identify all current-level records lacking confirmed attendance",
    )

    print(f"validated_ed_rows={len(ed)}")
    print(f"waves={len(WAVES)}")
    print("ed_key_unique=yes")
    print(f"hl_link_matched={int(ed['HL Link Matched'].eq(1).sum())}")
    print(f"hl_link_unmatched={len(unmatched)}")
    print("matched_ed_to_hl_context_consistency=yes")
    print("education_code_domains_valid=yes")
    print("current_level_condition_consistency=yes")
    print(f"variable_dictionary_rows={len(dictionary)}")
    print(f"alignment_summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
