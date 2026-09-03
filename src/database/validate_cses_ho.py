#!/usr/bin/env python3
"""Validate the CSES housing database build before publication."""

from pathlib import Path

import pandas as pd

from cses_hh_hl_common import WAVES, snake_case
from cses_housing import HO_COLUMNS, MONEY_COLUMNS


EXPECTED_WAVE_ROWS = {
    "2004": 15_000, "2007": 3_593, "2009": 11_971, "2011-12": 3_592,
    "2013": 3_840, "2014": 12_092, "2016": 3_839, "2017": 3_840,
    "2019": 10_075, "2021": 10_080,
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
    ho = pd.read_parquet(output / "final_HO_CSES.parquet")
    hh = pd.read_parquet(output / "final_HH_CSES.parquet")
    reporting = output
    audit = pd.read_csv(reporting / "cses_ho_alignment_audit.csv", dtype={"survey_wave": "string"})
    issues = pd.read_csv(reporting / "cses_ho_data_issues.csv", dtype={"survey_wave": "string"})
    dictionary = pd.read_csv(reporting / "ind_que_HO_CSES.csv", dtype="string")
    summary = pd.read_csv(reporting / "align_summary_HO_CSES.csv")

    require(list(ho.columns) == HO_COLUMNS, "HO columns or order differ from the contract")
    require(set(ho["Survey Wave"].dropna()) == set(WAVES), "HO wave coverage is incomplete")
    require(not ho[["Survey Wave", "Household ID"]].isna().any().any(), "HO key contains nulls")
    require(not ho.duplicated(["Survey Wave", "Household ID"]).any(), "HO wave-household key is not unique")
    counts = ho.groupby("Survey Wave", observed=True).size().to_dict()
    require(counts == EXPECTED_WAVE_ROWS, f"HO wave row counts changed: {counts}")

    hh_check = hh[[
        "Survey Wave", "Household ID", "PSU", "Province Code", "District Code",
        "Commune Code", "Village Code", "Urban Rural", "Survey Month", "Stratum",
        "Household Weight",
    ]].rename(columns=lambda column: column if column in {"Survey Wave", "Household ID"} else f"{column}_hh")
    linked = ho.merge(hh_check, on=["Survey Wave", "Household ID"], how="left", validate="1:1", indicator=True)
    require(linked.loc[linked["HH Link Matched"].eq(1), "_merge"].eq("both").all(), "A matched HO row is absent from HH")
    require(linked.loc[linked["HH Link Matched"].eq(0), "_merge"].eq("left_only").all(), "An unmatched HO row is present in HH")
    matched = linked.loc[linked["HH Link Matched"].eq(1)]
    for column in [
        "PSU", "Province Code", "District Code", "Commune Code", "Village Code",
        "Urban Rural", "Survey Month", "Stratum", "Household Weight",
    ]:
        require_equal(matched[column], matched[f"{column}_hh"], f"HO/HH {column} differs")

    require(ho["HH Link Matched"].isin([0, 1]).all(), "HO link flag contains an invalid value")
    require(int(ho["HH Link Matched"].eq(0).sum()) == 19, "HO/HH unmatched-record count changed")
    require(ho["Households in Housing Unit"].dropna().between(1, 50).all(), "Households in unit is out of range")
    require(ho["Floor Area Square Meters"].dropna().between(0.0001, 10_000).all(), "Floor area is out of range")
    require(ho["Rooms Used"].dropna().between(1, 50).all(), "Rooms used is out of range")
    require(ho["Drinking Water Treatment Frequency Source Code"].dropna().isin([1, 2, 3]).all(), "Water treatment frequency code is invalid")
    for column in [
        "Treats Drinking Water", "Boils Drinking Water", "Filters Drinking Water",
        "Uses Chemical Water Treatment", "Uses Alum Water Treatment",
        "Uses Other Water Treatment", "Has Toilet Facility",
    ]:
        require(ho[column].dropna().isin([0, 1]).all(), f"{column} contains an invalid indicator")
    expected_treatment = ho["Drinking Water Treatment Frequency Source Code"].map({1: 1, 2: 1, 3: 0}).astype("Int8")
    require_equal(ho["Treats Drinking Water"], expected_treatment, "Water treatment indicator differs from frequency")
    require_equal(ho["Dwelling Tenure Source Code"], ho["Dwelling Tenure Harmonized"], "Tenure harmonization differs from stable source codes")
    for column in MONEY_COLUMNS:
        require(ho[column].dropna().ge(0).all(), f"{column} contains a negative value")
        require(not ho[column].dropna().isin([9_999_999, 99_999_999, 999_999_999]).any(), f"{column} retains an exact missing sentinel")
    require(ho.loc[ho["Survey Wave"].eq("2004"), "Monthly Imputed Rent Riel"].isna().all(), "Unexpected 2004 imputed-rent values")

    require(int(audit["rows"].sum()) == len(ho), "HO audit row total is wrong")
    require(int(audit["duplicate_key_rows"].sum()) == 0, "HO audit reports duplicate keys")
    require(int(audit["missing_household_id"].sum()) == 0, "HO audit reports missing household IDs")
    require(int(audit["hh_link_unmatched"].sum()) == int(ho["HH Link Matched"].eq(0).sum()), "HO audit unmatched count is wrong")
    require(len(summary) == len(HO_COLUMNS), "HO summary does not have one row per column")
    require(set(summary["varname"]) == {snake_case(column) for column in HO_COLUMNS}, "HO summary variable set is wrong")
    require(set(dictionary["canonical_varname"]).issubset(set(summary["varname"])), "HO dictionary contains an unknown variable")
    require(dictionary["dataset_name"].nunique() == len(WAVES), "HO dictionary does not cover all waves")
    require(
        int(issues.loc[issues["issue_type"].eq("housing_record_not_in_hh"), "affected_rows"].sum()) == int(ho["HH Link Matched"].eq(0).sum()),
        "HO issues do not enumerate every unmatched housing record",
    )

    print(f"validated_ho_rows={len(ho)}")
    print(f"waves={len(WAVES)}")
    print("ho_key_unique=yes")
    print(f"hh_link_matched={int(ho['HH Link Matched'].eq(1).sum())}")
    print(f"hh_link_unmatched={int(ho['HH Link Matched'].eq(0).sum())}")
    print("matched_ho_to_hh_context_consistency=yes")
    print("housing_domains_and_harmonization=valid")
    print(f"variable_dictionary_rows={len(dictionary)}")
    print(f"alignment_summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
