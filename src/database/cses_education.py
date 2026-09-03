#!/usr/bin/env python3
"""Harmonize the CSES education module while preserving its source grain."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cses_hh_hl_common import (
    AlignmentContext,
    VARIABLE_INFO,
    WAVES,
    enforce_dtypes,
    find_column,
    initialize_context,
    snake_case,
    standardize_source,
)
from inventory_cses_archives import discover_sources, modules_for_source, normalize_wave


ED_ALIASES = {
    "Can Read": ["q02_02", "q02c02"],
    "Can Write": ["q02_03", "q02c03"],
    "Ever Attended School": ["q02_04", "q02c04"],
    "Years Attended School": ["q02c05"],
    "Highest Education Level Source Code": ["q02_05", "q02c06"],
    "Currently Attending School": ["q02_06", "q02c07"],
    "Current Education Level Source Code": ["q02_07", "q02c08"],
}

CONTEXT_COLUMNS = [
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Urban Rural", "Survey Month", "Stratum", "Household Weight", "Person Weight",
    "Sex", "Age",
]

ED_COLUMNS = [
    "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID", "Person ID",
    "HL Link Matched", "Sex", "Age", "Province Code", "District Code", "Commune Code",
    "Village Code", "Urban Rural", "Survey Month", "Stratum", "Household Weight",
    "Person Weight", "Can Read", "Can Write", "Ever Attended School",
    "Years Attended School", "Highest Education Level Source Code",
    "Education Level Harmonized", "Currently Attending School",
    "Current Education Level Source Code", "Current Education Level Harmonized",
    "Source Archive", "Source Submodule", "Source Row ID",
]

ED_VARIABLE_INFO = {
    "HL Link Matched": ("Household Listing Link Matched", "provenance", "1 when the education record matches final_HL_CSES on survey wave and person identifier; otherwise 0."),
    "Can Read": ("Can Read a Simple Message", "education", "Harmonized indicator: 1=Yes and 0=No."),
    "Can Write": ("Can Write a Simple Message", "education", "Harmonized indicator: 1=Yes and 0=No."),
    "Ever Attended School": ("Ever Attended Formal School", "education", "Harmonized indicator: 1=Yes and 0=No."),
    "Years Attended School": ("Completed Years Attended School", "education", "Released completed years attended, retained from 0 through 30 without imputation; unavailable in 2004."),
    "Highest Education Level Source Code": ("Highest Completed Education Source Code", "education", "Released wave-specific detailed code, retained for provenance."),
    "Education Level Harmonized": ("Highest Completed Education Level Harmonized", "education", "Cross-wave level: 0=None, 1=Preschool, 2=Primary, 3=Lower secondary, 4=Upper secondary, 5=Technical/vocational, 6=Higher education, 7=Other."),
    "Currently Attending School": ("Currently in Formal School System", "education", "Harmonized indicator: 1=Yes and 0=No."),
    "Current Education Level Source Code": ("Current Education Level Source Code", "education", "Released wave-specific current grade or level code, retained for provenance."),
    "Current Education Level Harmonized": ("Current Education Level Harmonized", "education", "Cross-wave level: 1=Preschool, 2=Primary, 3=Lower secondary, 4=Upper secondary, 5=Technical/vocational, 6=Higher education, 7=Other."),
}


def education_sources(root: Path) -> list[tuple[str, object]]:
    sources = [
        (normalize_wave(source.display_name(root)), source)
        for source in discover_sources(root)
        if "education" in modules_for_source(source, root)
    ]
    sources.sort(key=lambda item: (WAVES.index(item[0]), item[1].display_name(root)))
    counts = pd.Series([wave for wave, _ in sources]).value_counts().to_dict()
    bad = {wave: counts.get(wave, 0) for wave in WAVES if counts.get(wave, 0) != 1}
    if bad:
        raise RuntimeError(f"Expected exactly one education source per wave; found {bad}")
    return sources


def add_issue(
    rows: list[dict[str, object]],
    wave: str,
    issue_type: str,
    variable: str,
    affected_rows: int,
    detail: str,
) -> None:
    if affected_rows:
        rows.append({
            "grain": "ED", "survey_wave": wave, "issue_type": issue_type,
            "variable": variable, "affected_rows": int(affected_rows), "detail": detail,
        })


def yes_no_indicator(
    context: AlignmentContext,
    raw: pd.DataFrame,
    wave: str,
    variable: str,
    issues: list[dict[str, object]],
) -> pd.Series:
    column = find_column(raw, ED_ALIASES[variable])
    if column is None:
        raise RuntimeError(f"Missing {variable} source column in {wave}")
    numeric = pd.to_numeric(raw[column], errors="coerce")
    output = numeric.map({1: 1, 2: 0}).astype("Int8")
    invalid = int((numeric.notna() & ~numeric.isin([1, 2])).sum())
    add_issue(issues, wave, "unresolved_yes_no_code_set_null", variable, invalid, f"Codes outside 1/2 in {column} were set to null.")
    context.record(variable, wave, column, "derived")
    return output


def harmonize_highest_level(wave: str, value: object) -> object:
    if pd.isna(value):
        return pd.NA
    code = int(value)
    if wave == "2004":
        if code == 90:
            return 0
        if code == 0:
            return 1
        if 1 <= code <= 6:
            return 2
        if 7 <= code <= 9:
            return 3
        if 10 <= code <= 13:
            return 4
        if code in {14, 15}:
            return 5
        if code in {16, 17, 18}:
            return 6
        if code == 19:
            return 7
        return pd.NA
    if code == 88:
        return 0
    if code == 0:
        return 1
    if 1 <= code <= 6:
        return 2
    if code in {7, 8, 9, 13}:
        return 3
    if code in {10, 11, 12, 14}:
        return 4
    if code in {15, 16}:
        return 5
    if code in {17, 18, 19, 20}:
        return 6
    if code == 21:
        return 7
    return pd.NA


def harmonize_current_level(wave: str, value: object) -> object:
    if pd.isna(value):
        return pd.NA
    code = int(value)
    if code == 0:
        return 1
    if 1 <= code <= 6:
        return 2
    if 7 <= code <= 9:
        return 3
    if 10 <= code <= 12:
        return 4
    if wave == "2004":
        if code in {13}:
            return 4
        if code in {14, 15}:
            return 5
        if code in {16, 17, 18}:
            return 6
        if code == 19:
            return 7
        return pd.NA
    if wave in {"2007", "2009", "2011-12"}:
        if code in {13, 14}:
            return 5
        if code in {15, 16}:
            return 6
        return pd.NA
    if code in {13, 14, 15, 16}:
        return 5
    if code in {17, 18, 19}:
        return 6
    if code in {20, 21}:
        return 7
    return pd.NA


def build_wave(
    context: AlignmentContext,
    hl: pd.DataFrame,
    wave: str,
    source: object,
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    raw = context.load(source)
    standardized = standardize_source(context, source, wave, "ED")
    base_columns = [
        "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID", "Person ID",
        "Province Code", "District Code", "Commune Code", "Village Code", "Urban Rural",
        "Survey Month", "Stratum", "Source Archive", "Source Submodule", "Source Row ID",
    ]
    frame = standardized[base_columns].copy()

    hl_wave = hl.loc[hl["Survey Wave"].eq(wave), ["Survey Wave", "Person ID", "Household ID", "PSU"] + CONTEXT_COLUMNS].copy()
    hl_wave = hl_wave.rename(columns={column: f"{column}_hl" for column in hl_wave.columns if column not in {"Survey Wave", "Person ID"}})
    frame = frame.merge(hl_wave, on=["Survey Wave", "Person ID"], how="left", validate="m:1", indicator="_hl_merge")
    frame["HL Link Matched"] = frame["_hl_merge"].eq("both").astype("Int8")
    context.record("HL Link Matched", wave, "", "derived")

    hhid_conflict = frame["Household ID"].notna() & frame["Household ID_hl"].notna() & frame["Household ID"].ne(frame["Household ID_hl"])
    psu_conflict = frame["PSU"].notna() & frame["PSU_hl"].notna() & frame["PSU"].ne(frame["PSU_hl"])
    add_issue(issues, wave, "source_hl_key_conflict", "Household ID", int(hhid_conflict.sum()), "Source household identifier differs from the matched HL record; source value retained.")
    add_issue(issues, wave, "source_hl_key_conflict", "PSU", int(psu_conflict.sum()), "Source PSU differs from the matched HL record; source value retained.")
    frame["PSU"] = frame["PSU"].combine_first(frame["PSU_hl"])

    for variable in CONTEXT_COLUMNS:
        if variable in {"Household Weight", "Person Weight", "Sex", "Age"}:
            frame[variable] = frame[f"{variable}_hl"]
        else:
            source_value = frame[variable]
            hl_value = frame[f"{variable}_hl"]
            conflict = source_value.notna() & hl_value.notna() & source_value.astype("string").ne(hl_value.astype("string"))
            add_issue(issues, wave, "source_hl_context_conflict", variable, int(conflict.sum()), "Matched HL value was used and the disagreement was reported.")
            frame[variable] = hl_value.combine_first(source_value)
        if frame[variable].notna().any():
            context.record(variable, wave, "", "derived")

    for variable in ("Can Read", "Can Write", "Ever Attended School", "Currently Attending School"):
        frame[variable] = yes_no_indicator(context, raw, wave, variable, issues)

    years_column = find_column(raw, ED_ALIASES["Years Attended School"])
    if years_column is None:
        frame["Years Attended School"] = pd.Series(pd.NA, index=frame.index, dtype="Int16")
        add_issue(issues, wave, "source_variable_unavailable", "Years Attended School", len(frame), "The 2004 education release does not provide the later-wave completed-years item.")
    else:
        years = pd.to_numeric(raw[years_column], errors="coerce")
        valid_years = years.between(0, 30) & years.mod(1).eq(0)
        add_issue(issues, wave, "invalid_value_set_null", "Years Attended School", int((years.notna() & ~valid_years).sum()), f"Values outside integer years 0-30 in {years_column} were set to null.")
        frame["Years Attended School"] = years.where(valid_years).astype("Int16")
        context.record("Years Attended School", wave, years_column)

    highest_column = find_column(raw, ED_ALIASES["Highest Education Level Source Code"])
    current_column = find_column(raw, ED_ALIASES["Current Education Level Source Code"])
    if highest_column is None or current_column is None:
        raise RuntimeError(f"Missing education level source columns in {wave}")
    highest = pd.to_numeric(raw[highest_column], errors="coerce")
    current = pd.to_numeric(raw[current_column], errors="coerce")
    frame["Highest Education Level Source Code"] = highest.astype("Int16")
    frame["Current Education Level Source Code"] = current.astype("Int16")
    frame["Education Level Harmonized"] = highest.map(lambda value: harmonize_highest_level(wave, value)).astype("Int8")
    frame["Current Education Level Harmonized"] = current.map(lambda value: harmonize_current_level(wave, value)).astype("Int8")
    current_level_conflict = (
        ~frame["Currently Attending School"].eq(1).fillna(False)
        & current.notna()
    )
    add_issue(
        issues,
        wave,
        "current_level_reported_without_confirmed_attendance",
        "Current Education Level Harmonized",
        int(current_level_conflict.sum()),
        "The released current-level code was retained, but its harmonized value was set to null because current attendance is not confirmed as Yes.",
    )
    frame.loc[current_level_conflict, "Current Education Level Harmonized"] = pd.NA
    context.record("Highest Education Level Source Code", wave, highest_column)
    context.record("Education Level Harmonized", wave, highest_column, "derived")
    context.record("Current Education Level Source Code", wave, current_column)
    context.record("Current Education Level Harmonized", wave, current_column, "derived")

    unmatched = frame.loc[frame["HL Link Matched"].eq(0)]
    for person_id, household_id in unmatched[["Person ID", "Household ID"]].itertuples(index=False, name=None):
        add_issue(issues, wave, "education_record_not_in_hl", "Person ID", 1, f"Education record retained: person_id={person_id} household_id={household_id}.")

    drop_columns = [column for column in frame.columns if column.endswith("_hl")] + ["_hl_merge"]
    frame = frame.drop(columns=drop_columns)
    frame = frame[ED_COLUMNS]
    enforce_ed_dtypes(frame)
    return frame


def enforce_ed_dtypes(frame: pd.DataFrame) -> None:
    integer_types = {
        "Survey Year": "Int16", "Survey Month": "Int16", "HL Link Matched": "Int8",
        "Sex": "Int8", "Age": "Int16", "Can Read": "Int8", "Can Write": "Int8",
        "Ever Attended School": "Int8", "Years Attended School": "Int16",
        "Highest Education Level Source Code": "Int16", "Education Level Harmonized": "Int8",
        "Currently Attending School": "Int8", "Current Education Level Source Code": "Int16",
        "Current Education Level Harmonized": "Int8",
    }
    floats = {"Household Weight", "Person Weight"}
    for column in frame.columns:
        if column in integer_types:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(integer_types[column])
        elif column in floats:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        else:
            frame[column] = frame[column].astype("string")


def make_audit(frame: pd.DataFrame, hl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave in WAVES:
        part = frame.loc[frame["Survey Wave"].eq(wave)]
        hl_part = hl.loc[hl["Survey Wave"].eq(wave)]
        rows.append({
            "grain": "ED", "survey_wave": wave, "rows": len(part),
            "unique_persons": part["Person ID"].nunique(dropna=True),
            "duplicate_key_rows": int(part.duplicated(["Survey Wave", "Person ID"], keep=False).sum()),
            "missing_person_id": int(part["Person ID"].isna().sum()),
            "hl_link_matched": int(part["HL Link Matched"].eq(1).sum()),
            "hl_link_unmatched": int(part["HL Link Matched"].eq(0).sum()),
            "hl_rows_without_education_record": int((~hl_part["Person ID"].isin(part["Person ID"])).sum()),
            "missing_can_read": int(part["Can Read"].isna().sum()),
            "missing_can_write": int(part["Can Write"].isna().sum()),
            "missing_ever_attended": int(part["Ever Attended School"].isna().sum()),
            "years_attended_available": int(part["Years Attended School"].notna().sum()),
            "highest_level_harmonized_available": int(part["Education Level Harmonized"].notna().sum()),
            "currently_attending_available": int(part["Currently Attending School"].notna().sum()),
            "current_level_harmonized_available": int(part["Current Education Level Harmonized"].notna().sum()),
        })
    return pd.DataFrame(rows)


def variable_info(variable: str) -> tuple[str, str, str]:
    if variable in ED_VARIABLE_INFO:
        return ED_VARIABLE_INFO[variable]
    return VARIABLE_INFO[variable]


def write_metadata(context: AlignmentContext, frame: pd.DataFrame, output: Path) -> None:
    for wave in WAVES:
        for variable in ["Dataset Name", "Survey Wave", "Survey Year", "Source Archive", "Source Submodule", "Source Row ID"]:
            context.record(variable, wave, "", "derived")
    rows = []
    for variable, wave, raw_name, source_kind in sorted(
        context.mappings,
        key=lambda row: (ED_COLUMNS.index(row[0]) if row[0] in ED_COLUMNS else 999, WAVES.index(row[1]), row[2], row[3]),
    ):
        if variable not in ED_COLUMNS:
            continue
        full_name, measure_type, canonical_text = variable_info(variable)
        rows.append({
            "canonical_varname": snake_case(variable), "dataset_name": f"CSES {wave}",
            "column_in_raw_sav": raw_name, "column_label_in_english": full_name,
            "source_kind": source_kind, "measure_type": measure_type,
            "canonical_text": canonical_text,
        })
    ind_que = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    ind_que.to_csv(output / "ind_que_ED_CSES.csv", index=False)

    summary_rows = []
    for variable in ED_COLUMNS:
        selected = ind_que.loc[ind_que["canonical_varname"].eq(snake_case(variable))]
        _full_name, measure_type, canonical_text = variable_info(variable)
        summary_rows.append({
            "varname": snake_case(variable), "dataset_count": int(selected["dataset_name"].nunique()),
            "source_count": int(len(selected)), "explicit_count": int(selected["source_kind"].eq("explicit").sum()),
            "derived_count": int(selected["source_kind"].eq("derived").sum()),
            "measure_type": measure_type, "canonical_text": canonical_text,
        })
    pd.DataFrame(summary_rows).to_csv(output / "align_summary_ED_CSES.csv", index=False)


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = initialize_context(root)
    hl = pd.read_parquet(root / "data/exp/database/final_HL_CSES.parquet")
    issues: list[dict[str, object]] = []
    frames = [build_wave(context, hl, wave, source, issues) for wave, source in education_sources(root)]
    final = pd.concat(frames, ignore_index=True).sort_values(
        ["Survey Year", "Household ID", "Person ID", "Source Row ID"], na_position="last"
    ).reset_index(drop=True)
    audit = make_audit(final, hl)
    issue_frame = pd.concat(
        [context.issues.frame(), pd.DataFrame(issues)], ignore_index=True
    ).sort_values(["survey_wave", "issue_type", "variable"], ignore_index=True)

    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output / "final_ED_CSES.parquet", index=False)
    audit.to_csv(output / "cses_ed_alignment_audit.csv", index=False)
    issue_frame.to_csv(output / "cses_ed_data_issues.csv", index=False)
    write_metadata(context, final, output)
    return final, audit, issue_frame
