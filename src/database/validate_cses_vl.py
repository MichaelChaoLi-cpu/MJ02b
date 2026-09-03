#!/usr/bin/env python3
"""Validate the CSES village-demographic database build."""

from pathlib import Path

import pandas as pd

from cses_hh_hl_common import snake_case
from cses_village import COUNT_COLUMNS, HH_CONTEXT_COLUMNS, VL_COLUMNS, VL_WAVES, make_hh_psu_context


EXPECTED_WAVE_ROWS = {
    "2004": 900, "2007": 357, "2009": 720, "2011-12": 355,
    "2014": 1_006, "2016": 364, "2019": 1_008, "2021": 1_008,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_equal(left: pd.Series, right: pd.Series, message: str) -> None:
    left_value = left.astype("string")
    right_value = right.astype("string")
    equal = left_value.eq(right_value) | (left_value.isna() & right_value.isna())
    require(bool(equal.fillna(False).all()), f"{message}: mismatches={int((~equal.fillna(False)).sum())}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data/exp/database"
    vl = pd.read_parquet(output / "final_VL_CSES.parquet")
    hh = pd.read_parquet(output / "final_HH_CSES.parquet")
    hl = pd.read_parquet(output / "final_HL_CSES.parquet")
    reporting = output
    audit = pd.read_csv(reporting / "cses_vl_alignment_audit.csv", dtype={"survey_wave": "string"})
    issues = pd.read_csv(reporting / "cses_vl_data_issues.csv", dtype={"survey_wave": "string"})
    dictionary = pd.read_csv(reporting / "ind_que_VL_CSES.csv", dtype="string")
    summary = pd.read_csv(reporting / "align_summary_VL_CSES.csv")

    require(list(vl.columns) == VL_COLUMNS, "VL columns or order differ from the contract")
    require(set(vl["Survey Wave"].dropna()) == set(VL_WAVES), "VL wave coverage differs")
    require(not vl[["Survey Wave", "PSU"]].isna().any().any(), "VL key contains nulls")
    require(not vl.duplicated(["Survey Wave", "PSU"]).any(), "VL wave-PSU key is not unique")
    require(vl.groupby("Survey Wave", observed=True).size().to_dict() == EXPECTED_WAVE_ROWS, "VL wave row counts changed")
    require(vl["HH PSU Link Matched"].eq(1).all(), "A village PSU does not link to HH")

    hh_psu = make_hh_psu_context(hh, hl).rename(
        columns={column: f"{column}_hh" for column in HH_CONTEXT_COLUMNS + ["Sample Household Count", "Sample Person Count"]}
    )
    linked = vl.merge(hh_psu, on=["Survey Wave", "PSU"], how="left", validate="1:1", indicator=True)
    require(linked["_merge"].eq("both").all(), "A VL record is absent from the HH PSU context")
    for column in HH_CONTEXT_COLUMNS + ["Sample Household Count", "Sample Person Count"]:
        require_equal(linked[column], linked[f"{column}_hh"], f"VL/HH PSU {column} differs")

    for column in COUNT_COLUMNS + ["Sample Household Count", "Sample Person Count"]:
        require(vl[column].dropna().ge(0).all(), f"{column} contains a negative count")
    require(vl["Village Land Area Square Kilometers"].dropna().between(0.0000001, 100_000).all(), "Village land area is out of range")
    require(vl["Five Year Population Movement Source Code"].dropna().isin([1, 2, 3, 4]).all(), "Movement code is invalid")
    require(vl["Village Reference Day"].dropna().between(1, 31).all(), "Reference day is invalid")
    require(vl["Village Reference Month"].dropna().between(1, 12).all(), "Reference month is invalid")
    require(vl["Village Reference Year"].dropna().between(2000, 2022).all(), "Reference year is invalid")
    detailed = vl.loc[~vl["Survey Wave"].eq("2019")]
    require_equal(detailed["Village Male Count"], (detailed["Boys Below 18 Count"] + detailed["Men 18 Plus Count"]).astype("Int32"), "Derived village male count differs")
    require_equal(detailed["Village Female Count"], (detailed["Girls Below 18 Count"] + detailed["Women 18 Plus Count"]).astype("Int32"), "Derived village female count differs")
    structural_2019 = [
        "Village Reference Day", "Village Reference Month", "Village Reference Year",
        "Enumeration Area Count", "Households in Surveyed Enumeration Area",
        "Population Below 18 Count", "Boys Below 18 Count", "Girls Below 18 Count",
        "Population 18 Plus Count", "Men 18 Plus Count", "Women 18 Plus Count",
        "Village Land Area Square Kilometers", "Five Year Population Movement Source Code",
        "Village Household Count Five Years Ago", "Village Person Count Five Years Ago",
    ]
    require(vl.loc[vl["Survey Wave"].eq("2019"), structural_2019].isna().all().all(), "A structurally unavailable 2019 variable is populated")
    require(vl.loc[~vl["Survey Wave"].isin(["2019", "2021"]), ["Province Name", "District Name", "Commune Name", "Village Name"]].isna().all().all(), "Unexpected early-wave administrative names")

    age_mismatch = vl["Population Below 18 Count"].notna() & vl["Population 18 Plus Count"].notna() & vl["Village Person Count"].notna() & (vl["Population Below 18 Count"] + vl["Population 18 Plus Count"]).ne(vl["Village Person Count"])
    sex_mismatch = vl["Village Male Count"].notna() & vl["Village Female Count"].notna() & vl["Village Person Count"].notna() & (vl["Village Male Count"] + vl["Village Female Count"]).ne(vl["Village Person Count"])
    require(int(audit["age_component_mismatches"].sum()) == int(age_mismatch.sum()), "VL audit age mismatch count is wrong")
    require(int(audit["sex_component_mismatches"].sum()) == int(sex_mismatch.sum()), "VL audit sex mismatch count is wrong")
    require(int(audit["rows"].sum()) == len(vl), "VL audit row total is wrong")
    require(int(audit["duplicate_key_rows"].sum()) == 0, "VL audit reports duplicate keys")
    require(int(audit["missing_psu"].sum()) == 0, "VL audit reports missing PSUs")
    require(len(summary) == len(VL_COLUMNS), "VL summary does not have one row per column")
    require(set(summary["varname"]) == {snake_case(column) for column in VL_COLUMNS}, "VL summary variable set is wrong")
    require(set(dictionary["canonical_varname"]).issubset(set(summary["varname"])), "VL dictionary contains an unknown variable")
    require(dictionary["dataset_name"].nunique() == len(VL_WAVES), "VL dictionary does not cover all released village waves")
    require(not issues["issue_type"].eq("village_psu_not_in_hh").any(), "VL issues report an unexpected HH PSU orphan")

    print(f"validated_vl_rows={len(vl)}")
    print(f"waves={len(VL_WAVES)}")
    print("vl_key_unique=yes")
    print("vl_to_hh_psu_linkage=complete")
    print("vl_to_hh_psu_context_consistency=yes")
    print(f"released_age_component_mismatches_retained={int(age_mismatch.sum())}")
    print(f"released_sex_component_mismatches_retained={int(sex_mismatch.sum())}")
    print(f"variable_dictionary_rows={len(dictionary)}")
    print(f"alignment_summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
