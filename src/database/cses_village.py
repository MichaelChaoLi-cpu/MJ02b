#!/usr/bin/env python3
"""Harmonize the available CSES village-demographic modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cses_hh_hl_common import (
    AlignmentContext,
    VARIABLE_INFO,
    WAVES,
    clean_string,
    find_column,
    initialize_context,
    snake_case,
    standardize_source,
)
from inventory_cses_archives import discover_sources, normalize_wave, source_leaf, token


VL_WAVES = ["2004", "2007", "2009", "2011-12", "2014", "2016", "2019", "2021"]
SOURCE_KEYS = {
    "2004": "2004vls01demographic",
    "2007": "01villdemograpinfo",
    "2009": "demograpinfo",
    "2011-12": "2011vls01demographic",
    "2014": "demographic",
    "2016": "demographic",
    "2019": "cses2019village",
    "2021": "s1demographic",
}

VL_ALIASES = {
    "Province Name": ["province_name"],
    "District Name": ["district_name"],
    "Commune Name": ["commune_name"],
    "Village Name": ["village_name"],
    "Village Reference Day": ["s1q1_asonday"],
    "Village Reference Month": ["s1q1_asonmonth"],
    "Village Reference Year": ["s1q1_asonyear"],
    "Village Household Count": ["s1q1_hhsinvillage", "v_households"],
    "Enumeration Area Count": ["s1q1a"],
    "Households in Surveyed Enumeration Area": ["s1q1b"],
    "Village Person Count": ["s1q2_personsinvillage", "v_persons"],
    "Village Male Count": ["v_males"],
    "Village Female Count": ["v_females"],
    "Population Below 18 Count": ["s1q3_below18total"],
    "Boys Below 18 Count": ["s1q3_below18boys"],
    "Girls Below 18 Count": ["s1q3_below18girls"],
    "Population 18 Plus Count": ["s1q4_over18total"],
    "Men 18 Plus Count": ["s1q4_over18m"],
    "Women 18 Plus Count": ["s1q4_over18w"],
    "Village Land Area Square Kilometers": ["s1q5_totlandarea"],
    "Five Year Population Movement Source Code": ["s1q6_peoplemove"],
    "Village Household Count Five Years Ago": ["s1q7_hhs5yearago"],
    "Village Person Count Five Years Ago": ["s1q8_people5yearago"],
}

HH_CONTEXT_COLUMNS = [
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Urban Rural", "Survey Month", "Stratum",
]

COUNT_COLUMNS = [
    "Village Household Count", "Enumeration Area Count",
    "Households in Surveyed Enumeration Area", "Village Person Count",
    "Village Male Count", "Village Female Count", "Population Below 18 Count",
    "Boys Below 18 Count", "Girls Below 18 Count", "Population 18 Plus Count",
    "Men 18 Plus Count", "Women 18 Plus Count",
    "Village Household Count Five Years Ago", "Village Person Count Five Years Ago",
]

VL_COLUMNS = [
    "Dataset Name", "Survey Wave", "Survey Year", "PSU", "HH PSU Link Matched",
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Province Name", "District Name", "Commune Name", "Village Name",
    "Urban Rural", "Survey Month", "Stratum", "Sample Household Count",
    "Sample Person Count", "Village Reference Day", "Village Reference Month",
    "Village Reference Year", "Village Household Count", "Enumeration Area Count",
    "Households in Surveyed Enumeration Area", "Village Person Count",
    "Village Male Count", "Village Female Count", "Population Below 18 Count",
    "Boys Below 18 Count", "Girls Below 18 Count", "Population 18 Plus Count",
    "Men 18 Plus Count", "Women 18 Plus Count",
    "Village Land Area Square Kilometers", "Five Year Population Movement Source Code",
    "Village Household Count Five Years Ago", "Village Person Count Five Years Ago",
    "Source Archive", "Source Submodule", "Source Row ID",
]

VL_VARIABLE_INFO = {
    "HH PSU Link Matched": ("Household PSU Link Matched", "provenance", "1 when the village record's survey-wave PSU occurs in final_HH_CSES; otherwise 0."),
    "Province Name": ("Province Name", "geography", "Released province name where available; no name lookup is imposed on earlier waves."),
    "District Name": ("District Name", "geography", "Released district name where available; no name lookup is imposed on earlier waves."),
    "Commune Name": ("Commune Name", "geography", "Released commune name where available; no name lookup is imposed on earlier waves."),
    "Village Name": ("Village Name", "geography", "Released village name where available; no name lookup is imposed on earlier waves."),
    "Sample Household Count": ("CSES Sample Household Count in PSU", "sample coverage", "Number of final_HH_CSES records linked to the survey-wave PSU."),
    "Sample Person Count": ("CSES Sample Person Count in PSU", "sample coverage", "Number of final_HL_CSES records linked to the survey-wave PSU."),
    "Village Reference Day": ("Village Population Reference Day", "time", "Released day for the village population reference date."),
    "Village Reference Month": ("Village Population Reference Month", "time", "Released month for the village population reference date."),
    "Village Reference Year": ("Village Population Reference Year", "time", "Released year for the village population reference date."),
    "Village Household Count": ("Households Living in the Village", "village demographics", "Released number of households living in the village."),
    "Enumeration Area Count": ("Enumeration Areas in the Village", "village demographics", "Released number of enumeration areas; available only in later detailed village modules."),
    "Households in Surveyed Enumeration Area": ("Households in the Surveyed Enumeration Area", "village demographics", "Released household count for the surveyed enumeration area where a village has multiple enumeration areas."),
    "Village Person Count": ("People Living in the Village", "village demographics", "Released total village population."),
    "Village Male Count": ("Male Village Population", "village demographics", "Released directly in 2019 and otherwise derived as boys below 18 plus men age 18 or older."),
    "Village Female Count": ("Female Village Population", "village demographics", "Released directly in 2019 and otherwise derived as girls below 18 plus women age 18 or older."),
    "Population Below 18 Count": ("Village Population Below Age 18", "village demographics", "Released count of village residents below age 18."),
    "Boys Below 18 Count": ("Boys Below Age 18", "village demographics", "Released count of boys below age 18."),
    "Girls Below 18 Count": ("Girls Below Age 18", "village demographics", "Released count of girls below age 18."),
    "Population 18 Plus Count": ("Village Population Age 18 or Older", "village demographics", "Released count of village residents age 18 or older."),
    "Men 18 Plus Count": ("Men Age 18 or Older", "village demographics", "Released count of men age 18 or older."),
    "Women 18 Plus Count": ("Women Age 18 or Older", "village demographics", "Released count of women age 18 or older."),
    "Village Land Area Square Kilometers": ("Village Land Area", "geography", "Released village land area in square kilometers; positive values are retained without imputation."),
    "Five Year Population Movement Source Code": ("Five-Year Population Movement Source Code", "village demographics", "Released code 1-4 describing whether more people moved in, moved out, both equally, or neither over five years."),
    "Village Household Count Five Years Ago": ("Village Household Count Five Years Ago", "village demographics", "Released retrospective number of village households five years earlier."),
    "Village Person Count Five Years Ago": ("Village Person Count Five Years Ago", "village demographics", "Released retrospective village population five years earlier; the exact 99999 missing sentinel is null."),
}


def add_issue(rows: list[dict[str, object]], wave: str, issue_type: str, variable: str, affected_rows: int, detail: str) -> None:
    if affected_rows:
        rows.append({
            "grain": "VL", "survey_wave": wave, "issue_type": issue_type,
            "variable": variable, "affected_rows": int(affected_rows), "detail": detail,
        })


def village_sources(root: Path) -> list[tuple[str, object]]:
    candidates = {wave: [] for wave in VL_WAVES}
    for source in discover_sources(root):
        wave = normalize_wave(source.display_name(root))
        if wave in candidates and token(Path(source_leaf(source)).stem) == SOURCE_KEYS[wave]:
            candidates[wave].append(source)
    selected = []
    for wave in VL_WAVES:
        wave_sources = sorted(candidates[wave], key=lambda source: (len(source.archive_members), source.display_name(root)))
        expected = 2 if wave == "2019" else 1
        if len(wave_sources) != expected:
            raise RuntimeError(f"Expected {expected} village-demographic source copies in {wave}; found {len(wave_sources)}")
        selected.append((wave, wave_sources[0]))
    return selected


def make_hh_psu_context(hh: pd.DataFrame, hl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (wave, psu), group in hh.groupby(["Survey Wave", "PSU"], observed=True, dropna=False):
        row: dict[str, object] = {"Survey Wave": wave, "PSU": psu, "Sample Household Count": len(group)}
        for variable in HH_CONTEXT_COLUMNS:
            values = group[variable].dropna().unique()
            row[variable] = values[0] if len(values) == 1 else pd.NA
        rows.append(row)
    result = pd.DataFrame(rows)
    persons = hl.groupby(["Survey Wave", "PSU"], observed=True).size().rename("Sample Person Count").reset_index()
    return result.merge(persons, on=["Survey Wave", "PSU"], how="left", validate="1:1")


def clean_count(context: AlignmentContext, raw: pd.DataFrame, wave: str, variable: str, issues: list[dict[str, object]]) -> pd.Series:
    column = find_column(raw, VL_ALIASES[variable])
    if column is None:
        add_issue(issues, wave, "source_variable_unavailable", variable, len(raw), f"{variable} is unavailable in the selected {wave} village source.")
        return pd.Series(pd.NA, index=raw.index, dtype="Int32")
    numeric = pd.to_numeric(raw[column], errors="coerce")
    valid = numeric.between(0, 1_000_000) & numeric.mod(1).eq(0)
    if variable == "Village Person Count Five Years Ago":
        valid &= numeric.ne(99_999)
    invalid = numeric.notna() & ~valid
    add_issue(issues, wave, "invalid_value_set_null", variable, int(invalid.sum()), f"Invalid counts and documented sentinels in {column} were set to null.")
    context.record(variable, wave, column)
    return numeric.where(valid).astype("Int32")


def clean_date_part(context: AlignmentContext, raw: pd.DataFrame, wave: str, variable: str, lower: int, upper: int, issues: list[dict[str, object]]) -> pd.Series:
    column = find_column(raw, VL_ALIASES[variable])
    if column is None:
        add_issue(issues, wave, "source_variable_unavailable", variable, len(raw), f"{variable} is unavailable in the selected {wave} village source.")
        return pd.Series(pd.NA, index=raw.index, dtype="Int16")
    numeric = pd.to_numeric(raw[column], errors="coerce")
    valid = numeric.between(lower, upper) & numeric.mod(1).eq(0)
    add_issue(issues, wave, "invalid_value_set_null", variable, int((numeric.notna() & ~valid).sum()), f"Values outside {lower}-{upper} in {column} were set to null.")
    context.record(variable, wave, column)
    return numeric.where(valid).astype("Int16")


def build_wave(context: AlignmentContext, hh_psu: pd.DataFrame, wave: str, source: object, issues: list[dict[str, object]]) -> pd.DataFrame:
    raw = context.load(source).reset_index(drop=True)
    standardized = standardize_source(context, source, wave, "VL")
    frame = standardized[[
        "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Province Code",
        "District Code", "Commune Code", "Village Code", "Urban Rural",
        "Survey Month", "Stratum", "Source Archive", "Source Submodule", "Source Row ID",
    ]].copy()

    context_part = hh_psu.loc[hh_psu["Survey Wave"].eq(wave)].rename(
        columns={column: f"{column}_hh" for column in HH_CONTEXT_COLUMNS + ["Sample Household Count", "Sample Person Count"]}
    )
    frame = frame.merge(context_part, on=["Survey Wave", "PSU"], how="left", validate="1:1", indicator="_hh_merge")
    frame["HH PSU Link Matched"] = frame["_hh_merge"].eq("both").astype("Int8")
    context.record("HH PSU Link Matched", wave, "", "derived")
    for variable in HH_CONTEXT_COLUMNS:
        source_value = frame[variable]
        hh_value = frame[f"{variable}_hh"]
        conflict = source_value.notna() & hh_value.notna() & source_value.astype("string").ne(hh_value.astype("string"))
        add_issue(issues, wave, "source_hh_psu_context_conflict", variable, int(conflict.sum()), "HH PSU context was used and the source disagreement was reported.")
        frame[variable] = hh_value.combine_first(source_value)
        context.record(variable, wave, "", "derived")
    frame["Sample Household Count"] = frame["Sample Household Count_hh"].astype("Int16")
    frame["Sample Person Count"] = frame["Sample Person Count_hh"].astype("Int16")
    context.record("Sample Household Count", wave, "", "derived")
    context.record("Sample Person Count", wave, "", "derived")

    for variable in ["Province Name", "District Name", "Commune Name", "Village Name"]:
        column = find_column(raw, VL_ALIASES[variable])
        if column is None:
            frame[variable] = pd.Series(pd.NA, index=raw.index, dtype="string")
            add_issue(issues, wave, "source_variable_unavailable", variable, len(raw), f"{variable} is unavailable in the selected {wave} village source.")
        else:
            frame[variable] = clean_string(raw[column])
            context.record(variable, wave, column)

    frame["Village Reference Day"] = clean_date_part(context, raw, wave, "Village Reference Day", 1, 31, issues)
    frame["Village Reference Month"] = clean_date_part(context, raw, wave, "Village Reference Month", 1, 12, issues)
    frame["Village Reference Year"] = clean_date_part(context, raw, wave, "Village Reference Year", 2000, 2022, issues)
    for variable in COUNT_COLUMNS:
        if variable in {"Village Male Count", "Village Female Count"} and wave != "2019":
            continue
        frame[variable] = clean_count(context, raw, wave, variable, issues)

    if wave != "2019":
        frame["Village Male Count"] = (frame["Boys Below 18 Count"] + frame["Men 18 Plus Count"]).astype("Int32")
        frame["Village Female Count"] = (frame["Girls Below 18 Count"] + frame["Women 18 Plus Count"]).astype("Int32")
        for target, components in {
            "Village Male Count": ["Boys Below 18 Count", "Men 18 Plus Count"],
            "Village Female Count": ["Girls Below 18 Count", "Women 18 Plus Count"],
        }.items():
            for component in components:
                context.record(target, wave, find_column(raw, VL_ALIASES[component]) or "", "derived")

    land_column = find_column(raw, VL_ALIASES["Village Land Area Square Kilometers"])
    if land_column is None:
        frame["Village Land Area Square Kilometers"] = pd.Series(pd.NA, index=raw.index, dtype="Float64")
        add_issue(issues, wave, "source_variable_unavailable", "Village Land Area Square Kilometers", len(raw), f"Village land area is unavailable in the selected {wave} village source.")
    else:
        area = pd.to_numeric(raw[land_column], errors="coerce")
        valid = area.gt(0) & area.le(100_000)
        add_issue(issues, wave, "invalid_value_set_null", "Village Land Area Square Kilometers", int((area.notna() & ~valid).sum()), f"Non-positive or implausibly large values in {land_column} were set to null.")
        frame["Village Land Area Square Kilometers"] = area.where(valid).astype("Float64")
        context.record("Village Land Area Square Kilometers", wave, land_column)

    movement_column = find_column(raw, VL_ALIASES["Five Year Population Movement Source Code"])
    if movement_column is None:
        frame["Five Year Population Movement Source Code"] = pd.Series(pd.NA, index=raw.index, dtype="Int8")
        add_issue(issues, wave, "source_variable_unavailable", "Five Year Population Movement Source Code", len(raw), f"Five-year movement is unavailable in the selected {wave} village source.")
    else:
        movement = pd.to_numeric(raw[movement_column], errors="coerce")
        valid = movement.isin([1, 2, 3, 4])
        add_issue(issues, wave, "unresolved_code_set_null", "Five Year Population Movement Source Code", int((movement.notna() & ~valid).sum()), f"Codes outside 1-4 in {movement_column} were set to null.")
        frame["Five Year Population Movement Source Code"] = movement.where(valid).astype("Int8")
        context.record("Five Year Population Movement Source Code", wave, movement_column)

    age_sum_mismatch = frame["Population Below 18 Count"].notna() & frame["Population 18 Plus Count"].notna() & frame["Village Person Count"].notna() & (frame["Population Below 18 Count"] + frame["Population 18 Plus Count"]).ne(frame["Village Person Count"])
    sex_sum_mismatch = frame["Village Male Count"].notna() & frame["Village Female Count"].notna() & frame["Village Person Count"].notna() & (frame["Village Male Count"] + frame["Village Female Count"]).ne(frame["Village Person Count"])
    add_issue(issues, wave, "released_population_components_disagree", "Village Person Count", int(age_sum_mismatch.sum()), "Released below-18 and 18-plus counts do not sum to the released village population; all source values were retained.")
    add_issue(issues, wave, "released_sex_components_disagree", "Village Person Count", int(sex_sum_mismatch.sum()), "Male and female counts do not sum to the released village population; all source values were retained.")

    unmatched = frame.loc[frame["HH PSU Link Matched"].eq(0), "PSU"]
    for psu in unmatched:
        add_issue(issues, wave, "village_psu_not_in_hh", "PSU", 1, f"Village record retained: psu={psu}.")
    frame = frame.drop(columns=[column for column in frame.columns if column.endswith("_hh")] + ["_hh_merge"])[VL_COLUMNS]
    enforce_dtypes(frame)
    return frame


def enforce_dtypes(frame: pd.DataFrame) -> None:
    int8 = {"HH PSU Link Matched", "Five Year Population Movement Source Code"}
    int16 = {"Survey Year", "Survey Month", "Sample Household Count", "Sample Person Count", "Village Reference Day", "Village Reference Month", "Village Reference Year"}
    int32 = set(COUNT_COLUMNS)
    for column in frame.columns:
        if column in int8:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int8")
        elif column in int16:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int16")
        elif column in int32:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int32")
        elif column == "Village Land Area Square Kilometers":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        else:
            frame[column] = frame[column].astype("string")


def variable_info(variable: str) -> tuple[str, str, str]:
    if variable in VL_VARIABLE_INFO:
        return VL_VARIABLE_INFO[variable]
    return VARIABLE_INFO[variable]


def write_metadata(context: AlignmentContext, output: Path) -> None:
    for wave in VL_WAVES:
        for variable in ["Dataset Name", "Survey Wave", "Survey Year", "Source Archive", "Source Submodule", "Source Row ID"]:
            context.record(variable, wave, "", "derived")
    rows = []
    for variable, wave, raw_name, source_kind in sorted(
        context.mappings,
        key=lambda row: (VL_COLUMNS.index(row[0]) if row[0] in VL_COLUMNS else 999, VL_WAVES.index(row[1]) if row[1] in VL_WAVES else 999, row[2], row[3]),
    ):
        if variable not in VL_COLUMNS or wave not in VL_WAVES:
            continue
        full_name, measure_type, canonical_text = variable_info(variable)
        rows.append({
            "canonical_varname": snake_case(variable), "dataset_name": f"CSES {wave}",
            "column_in_raw_sav": raw_name, "column_label_in_english": full_name,
            "source_kind": source_kind, "measure_type": measure_type,
            "canonical_text": canonical_text,
        })
    dictionary = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    dictionary.to_csv(output / "ind_que_VL_CSES.csv", index=False)
    summary = []
    for variable in VL_COLUMNS:
        selected = dictionary.loc[dictionary["canonical_varname"].eq(snake_case(variable))]
        _full_name, measure_type, canonical_text = variable_info(variable)
        summary.append({
            "varname": snake_case(variable), "dataset_count": int(selected["dataset_name"].nunique()),
            "source_count": int(len(selected)), "explicit_count": int(selected["source_kind"].eq("explicit").sum()),
            "derived_count": int(selected["source_kind"].eq("derived").sum()),
            "measure_type": measure_type, "canonical_text": canonical_text,
        })
    pd.DataFrame(summary).to_csv(output / "align_summary_VL_CSES.csv", index=False)


def make_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave in VL_WAVES:
        part = frame.loc[frame["Survey Wave"].eq(wave)]
        age_mismatch = part["Population Below 18 Count"].notna() & part["Population 18 Plus Count"].notna() & part["Village Person Count"].notna() & (part["Population Below 18 Count"] + part["Population 18 Plus Count"]).ne(part["Village Person Count"])
        sex_mismatch = part["Village Male Count"].notna() & part["Village Female Count"].notna() & part["Village Person Count"].notna() & (part["Village Male Count"] + part["Village Female Count"]).ne(part["Village Person Count"])
        rows.append({
            "grain": "VL", "survey_wave": wave, "rows": len(part),
            "unique_psus": part["PSU"].nunique(dropna=True),
            "duplicate_key_rows": int(part.duplicated(["Survey Wave", "PSU"], keep=False).sum()),
            "missing_psu": int(part["PSU"].isna().sum()),
            "hh_psu_link_matched": int(part["HH PSU Link Matched"].eq(1).sum()),
            "hh_psu_link_unmatched": int(part["HH PSU Link Matched"].eq(0).sum()),
            "village_households_available": int(part["Village Household Count"].notna().sum()),
            "village_population_available": int(part["Village Person Count"].notna().sum()),
            "age_component_mismatches": int(age_mismatch.sum()),
            "sex_component_mismatches": int(sex_mismatch.sum()),
        })
    return pd.DataFrame(rows)


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = initialize_context(root)
    hh = pd.read_parquet(root / "data/exp/database/final_HH_CSES.parquet")
    hl = pd.read_parquet(root / "data/exp/database/final_HL_CSES.parquet")
    hh_psu = make_hh_psu_context(hh, hl)
    issues: list[dict[str, object]] = []
    frames = [build_wave(context, hh_psu, wave, source, issues) for wave, source in village_sources(root)]
    final = pd.concat(frames, ignore_index=True).sort_values(["Survey Year", "PSU", "Source Row ID"]).reset_index(drop=True)
    audit = make_audit(final)
    issue_frame = pd.concat([context.issues.frame(), pd.DataFrame(issues)], ignore_index=True).sort_values(
        ["survey_wave", "issue_type", "variable"], ignore_index=True,
    )
    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output / "final_VL_CSES.parquet", index=False)
    audit.to_csv(output / "cses_vl_alignment_audit.csv", index=False)
    issue_frame.to_csv(output / "cses_vl_data_issues.csv", index=False)
    write_metadata(context, output)
    return final, audit, issue_frame
