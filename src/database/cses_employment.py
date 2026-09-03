#!/usr/bin/env python3
"""Harmonize CSES current-employment records across ten survey waves."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cses_hh_hl_common import (
    AlignmentContext,
    VARIABLE_INFO,
    WAVES,
    clean_code,
    find_column,
    initialize_context,
    snake_case,
    standardize_source,
)
from inventory_cses_archives import discover_sources, normalize_wave, source_leaf, token


SOURCE_KEYS = {
    "2004": {"2004hhs13alabor7days", "2004hhs13blabor7days"},
    "2007": {"13aeconomicactivity"},
    "2009": {"personecocurrent"},
    "2011-12": {"11personecocurrent"},
    "2013": {"personecocurrent"},
    "2014": {"ecocurrent"},
    "2016": {"personecocurrent"},
    "2017": {"2017hhs15labor7days"},
    "2019": {"s15personecocurrent"},
    "2021": {"s15personecocurrent"},
}

ALIASES = {
    "Worked At Least One Hour Past 7 Days": ["q13a03", "q13ac03", "q15_c03"],
    "Second Work Screening Source Code": ["q13a04", "q13ac04", "q15_c04"],
    "Total Hours Worked Past 7 Days": ["q13a05", "q13ac05", "q15_c19"],
    "Main Occupation Source Code": ["q13b02b1", "q15_c05b"],
    "Main Industry Source Code": ["q13b03b1", "q15_c06b"],
    "Main Employer Type Source Code": ["q13b07_1", "q15_c07"],
    "Main Employment Status Source Code": ["q13b06_1", "q15_c08"],
    "Main Hours Worked Past 7 Days": ["q13b04_1", "q15_c09"],
    "Main Days Worked Last Month": ["q13b05_1", "q15_c10", "q15_c10a"],
    "Main Job Works Whole Year": ["q15_c10b"],
    "Main Job Was Usual Past 7 Days": ["q15_c10c"],
    "Main Job Was Abroad": ["q15_c10d"],
    "Additional Jobs Count": ["q15_c11"],
    "Total Occupations Past 7 Days": ["q13a12", "q13ac14"],
    "Secondary Occupation Source Code": ["q13b02b2", "q15_c12b"],
    "Secondary Industry Source Code": ["q13b03b2", "q15_c13b"],
    "Secondary Employer Type Source Code": ["q13b07_2", "q15_c14"],
    "Secondary Employment Status Source Code": ["q13b06_2", "q15_c15"],
    "Secondary Hours Worked Past 7 Days": ["q13b04_2", "q15_c16"],
    "Secondary Days Worked Last Month": ["q13b05_2", "q15_c17a"],
    "Secondary Job Works Whole Year": ["q15_c17b"],
    "Secondary Job Was Usual Past 7 Days": ["q15_c17c"],
    "Monthly Salary Wages Riel": ["q15_c20"],
    "Preferred Hours Change Source Code": ["q15_c21"],
    "Hours Less Preferred": ["q15_c22a"],
    "Hours More Preferred": ["q15_c22b"],
    "Available for Additional Work": ["q15_c23"],
    "Reason Working Fewer Hours Source Code": ["q15_c24"],
    "Months Working Fewer Hours": ["q15_c25"],
    "Actively Seeking Work": ["q13a08", "q13ac08", "q15_c26"],
    "Job Search Method 1 Source Code": ["q13a09a", "q13ac11a", "q15_c27a"],
    "Job Search Method 2 Source Code": ["q13a09b", "q13ac11b", "q15_c27b"],
    "Job Search Method 3 Source Code": ["q13a09c", "q13ac11c", "q15_c27c"],
    "Available for Work": ["q13a07", "q13ac07", "q15_c28"],
    "Desired Weekly Hours": ["q13a10", "q13ac12", "q15_c29"],
    "Months Actively Seeking Work": ["q15_c30"],
    "Reason Not Actively Seeking Source Code": ["q15_c31"],
    "Months Out of Work": ["q15_c32"],
    "Latest Work Seasonal": ["q15_c33"],
}

CONTEXT_COLUMNS = [
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Urban Rural", "Survey Month", "Stratum", "Household Weight", "Person Weight",
    "Sex", "Age",
]

EC_MODULE_COLUMNS = [
    "Worked At Least One Hour Past 7 Days", "Second Work Screening Source Code",
    "Total Hours Worked Past 7 Days", "Main Occupation Source Code",
    "Main Industry Source Code", "Main Employer Type Source Code",
    "Main Employment Status Source Code", "Main Hours Worked Past 7 Days",
    "Main Days Worked Last Month", "Main Job Works Whole Year",
    "Main Job Was Usual Past 7 Days", "Main Job Was Abroad",
    "Additional Jobs Count", "Total Occupations Past 7 Days",
    "Secondary Occupation Source Code", "Secondary Industry Source Code",
    "Secondary Employer Type Source Code", "Secondary Employment Status Source Code",
    "Secondary Hours Worked Past 7 Days", "Secondary Days Worked Last Month",
    "Secondary Job Works Whole Year", "Secondary Job Was Usual Past 7 Days",
    "Monthly Salary Wages Riel", "Preferred Hours Change Source Code",
    "Hours Less Preferred", "Hours More Preferred", "Available for Additional Work",
    "Reason Working Fewer Hours Source Code", "Months Working Fewer Hours",
    "Actively Seeking Work", "Job Search Method 1 Source Code",
    "Job Search Method 2 Source Code", "Job Search Method 3 Source Code",
    "Available for Work", "Desired Weekly Hours", "Months Actively Seeking Work",
    "Reason Not Actively Seeking Source Code", "Months Out of Work",
    "Latest Work Seasonal",
]

EC_COLUMNS = [
    "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID", "Person ID",
    "HL Link Matched", "Sex", "Age", "Province Code", "District Code", "Commune Code",
    "Village Code", "Urban Rural", "Survey Month", "Stratum", "Household Weight",
    "Person Weight", *EC_MODULE_COLUMNS,
    "Source Archive", "Source Submodule", "Source Row ID",
]

EC_VARIABLE_INFO = {
    "HL Link Matched": ("Household Listing Link Matched", "provenance", "1 when the current-employment record matches final_HL_CSES on survey wave and person identifier; otherwise 0."),
    "Worked At Least One Hour Past 7 Days": ("Worked At Least One Hour in the Past Seven Days", "employment", "Harmonized released response: 1=Yes and 0=No; wording emphasizes paid work in later waves."),
    "Second Work Screening Source Code": ("Second Current-Work Screening Source Code", "employment", "Released 1/2 response retained as a source code because the wording shifts between job absence and unpaid work across waves."),
    "Total Hours Worked Past 7 Days": ("Total Hours Worked in the Past Seven Days", "employment", "Released total weekly hours, with documented 98/99 sentinels set to null."),
    "Main Occupation Source Code": ("Main Occupation Source Code", "employment", "Released three-character occupation code; classification is not assumed stable across waves."),
    "Main Industry Source Code": ("Main Industry Source Code", "employment", "Released industry code retained at its wave-specific two- or four-character width."),
    "Main Employer Type Source Code": ("Main Job Employer Type Source Code", "employment", "Released wave-specific employer-type code."),
    "Main Employment Status Source Code": ("Main Job Employment Status Source Code", "employment", "Released wave-specific employment-status code."),
    "Main Hours Worked Past 7 Days": ("Main Job Hours Worked in the Past Seven Days", "employment", "Released main-job weekly hours; 98/99 sentinels are null."),
    "Main Days Worked Last Month": ("Main Job Days Worked Last Month", "employment", "Released main-job workdays in the last month, retained from 0 through 31."),
    "Main Job Works Whole Year": ("Main Job Works Whole Year", "employment", "Harmonized released response: 1=Yes and 0=No."),
    "Main Job Was Usual Past 7 Days": ("Main Job Work Was Usual in the Past Seven Days", "employment", "Harmonized released response: 1=Yes and 0=No."),
    "Main Job Was Abroad": ("Main Job Was Performed Abroad", "employment", "Harmonized released response: 1=Yes and 0=No."),
    "Additional Jobs Count": ("Additional Jobs Count", "employment", "Released number of jobs beyond the main job in later-wave current-employment modules."),
    "Total Occupations Past 7 Days": ("Total Occupations in the Past Seven Days", "employment", "Released total in 2004/2007 and derived as one plus additional jobs in later waves when a main job is reported."),
    "Secondary Occupation Source Code": ("Secondary Occupation Source Code", "employment", "Released three-character secondary occupation code; null unless at least two occupations are reported."),
    "Secondary Industry Source Code": ("Secondary Industry Source Code", "employment", "Released wave-specific secondary industry code; null unless at least two occupations are reported."),
    "Secondary Employer Type Source Code": ("Secondary Job Employer Type Source Code", "employment", "Released wave-specific secondary employer-type code."),
    "Secondary Employment Status Source Code": ("Secondary Job Employment Status Source Code", "employment", "Released wave-specific secondary employment-status code."),
    "Secondary Hours Worked Past 7 Days": ("Secondary Job Hours Worked in the Past Seven Days", "employment", "Released secondary-job weekly hours; 98/99 sentinels are null."),
    "Secondary Days Worked Last Month": ("Secondary Job Days Worked Last Month", "employment", "Released secondary-job workdays in the last month, retained from 0 through 31."),
    "Secondary Job Works Whole Year": ("Secondary Job Works Whole Year", "employment", "Harmonized released response: 1=Yes and 0=No."),
    "Secondary Job Was Usual Past 7 Days": ("Secondary Job Work Was Usual in the Past Seven Days", "employment", "Harmonized released response: 1=Yes and 0=No."),
    "Monthly Salary Wages Riel": ("Monthly Salary and Wages", "employment income", "Released total salary and wages in nominal Cambodian riel for 2009-2021; no imputation or price adjustment."),
    "Preferred Hours Change Source Code": ("Preferred Change in Work Hours Source Code", "underemployment", "Released preference code for fewer, more, or unchanged hours."),
    "Hours Less Preferred": ("Preferred Reduction in Weekly Hours", "underemployment", "Released number of fewer weekly hours preferred."),
    "Hours More Preferred": ("Preferred Increase in Weekly Hours", "underemployment", "Released number of additional weekly hours preferred."),
    "Available for Additional Work": ("Available for Additional Work", "underemployment", "Harmonized released response: 1=Yes and 0=No."),
    "Reason Working Fewer Hours Source Code": ("Reason Working Fewer Hours Source Code", "underemployment", "Released wave-specific reason code."),
    "Months Working Fewer Hours": ("Months Working Fewer Hours Than Preferred", "underemployment", "Released duration in months; 98/99 sentinels are null."),
    "Actively Seeking Work": ("Actively Seeking Work", "unemployment", "Harmonized released response: 1=Yes and 0=No; later waves specify the past four weeks."),
    "Job Search Method 1 Source Code": ("First Job Search Method Source Code", "unemployment", "Released wave-specific first job-search method code."),
    "Job Search Method 2 Source Code": ("Second Job Search Method Source Code", "unemployment", "Released wave-specific second job-search method code."),
    "Job Search Method 3 Source Code": ("Third Job Search Method Source Code", "unemployment", "Released wave-specific third job-search method code."),
    "Available for Work": ("Available for Work", "unemployment", "Harmonized released response: 1=Yes and 0=No."),
    "Desired Weekly Hours": ("Desired Weekly Work Hours", "unemployment", "Released desired weekly hours; 98/99 sentinels are null."),
    "Months Actively Seeking Work": ("Months Actively Seeking Work", "unemployment", "Released active-search duration in months; 98/99 sentinels are null."),
    "Reason Not Actively Seeking Source Code": ("Reason Not Actively Seeking Work Source Code", "unemployment", "Released wave-specific reason code."),
    "Months Out of Work": ("Total Months Out of Work", "unemployment", "Released duration out of work in months; 98/99 sentinels are null."),
    "Latest Work Seasonal": ("Latest Work Was Seasonal", "employment", "Harmonized released response: 1=Yes and 0=No."),
}

YES_NO_COLUMNS = {
    "Worked At Least One Hour Past 7 Days", "Main Job Works Whole Year",
    "Main Job Was Usual Past 7 Days", "Main Job Was Abroad",
    "Secondary Job Works Whole Year", "Secondary Job Was Usual Past 7 Days",
    "Available for Additional Work", "Actively Seeking Work", "Available for Work",
    "Latest Work Seasonal",
}
CODE_COLUMNS = {
    "Main Employer Type Source Code", "Main Employment Status Source Code",
    "Secondary Employer Type Source Code", "Secondary Employment Status Source Code",
    "Preferred Hours Change Source Code", "Reason Working Fewer Hours Source Code",
    "Job Search Method 1 Source Code", "Job Search Method 2 Source Code",
    "Job Search Method 3 Source Code", "Reason Not Actively Seeking Source Code",
}
HOUR_COLUMNS = {
    "Total Hours Worked Past 7 Days", "Main Hours Worked Past 7 Days",
    "Secondary Hours Worked Past 7 Days", "Hours Less Preferred",
    "Hours More Preferred", "Desired Weekly Hours",
}
DURATION_COLUMNS = {"Months Working Fewer Hours", "Months Actively Seeking Work", "Months Out of Work"}
SECONDARY_COLUMNS = {
    "Secondary Occupation Source Code", "Secondary Industry Source Code",
    "Secondary Employer Type Source Code", "Secondary Employment Status Source Code",
    "Secondary Hours Worked Past 7 Days", "Secondary Days Worked Last Month",
    "Secondary Job Works Whole Year", "Secondary Job Was Usual Past 7 Days",
}
MONEY_SENTINELS = {9_999_999, 99_999_999, 999_999_999}


def add_issue(rows: list[dict[str, object]], wave: str, issue_type: str, variable: str, affected_rows: int, detail: str) -> None:
    if affected_rows:
        rows.append({
            "grain": "EC", "survey_wave": wave, "issue_type": issue_type,
            "variable": variable, "affected_rows": int(affected_rows), "detail": detail,
        })


def employment_sources(root: Path) -> list[tuple[str, list[object]]]:
    grouped = {wave: [] for wave in WAVES}
    for source in discover_sources(root):
        wave = normalize_wave(source.display_name(root))
        if wave not in grouped:
            continue
        leaf_key = token(Path(source_leaf(source)).stem)
        if leaf_key in SOURCE_KEYS[wave]:
            grouped[wave].append(source)
    expected = {wave: (2 if wave == "2004" else 1) for wave in WAVES}
    bad = {wave: len(grouped[wave]) for wave in WAVES if len(grouped[wave]) != expected[wave]}
    if bad:
        raise RuntimeError(f"Current-employment source counts differ from the contract: {bad}")
    return [(wave, sorted(grouped[wave], key=lambda source: source.display_name(root))) for wave in WAVES]


def prepare_wave_sources(context: AlignmentContext, wave: str, sources: list[object]) -> tuple[pd.DataFrame, list[tuple[pd.DataFrame, pd.DataFrame]]]:
    standardized = [standardize_source(context, source, wave, "EC") for source in sources]
    raw = [context.load(source).reset_index(drop=True) for source in sources]
    base = standardized[0].copy()
    aligned = [(raw[0], standardized[0])]
    if wave == "2004":
        left_ids = set(standardized[0]["Person ID"])
        right_ids = set(standardized[1]["Person ID"])
        if left_ids != right_ids or standardized[1]["Person ID"].duplicated().any():
            raise RuntimeError("The two 2004 employment sources do not form a one-to-one person merge")
        lookup = raw[1].copy()
        lookup["__person_id"] = standardized[1]["Person ID"].to_numpy()
        lookup = lookup.set_index("__person_id", drop=True).reindex(base["Person ID"])
        reordered = lookup.reset_index(drop=True)
        aligned.append((reordered, standardized[1].set_index("Person ID").reindex(base["Person ID"]).reset_index()))
        base["Source Submodule"] = standardized[0]["Source Submodule"] + " | " + standardized[1]["Source Submodule"]
        base["Source Row ID"] = standardized[0]["Source Row ID"] + " | " + standardized[1].set_index("Person ID").reindex(base["Person ID"])["Source Row ID"].reset_index(drop=True)
    return base, aligned


def source_value(aligned: list[tuple[pd.DataFrame, pd.DataFrame]], aliases: list[str]) -> tuple[pd.Series | None, str | None]:
    for raw, _standardized in aligned:
        column = find_column(raw, aliases)
        if column is not None:
            return raw[column].reset_index(drop=True), column
    return None, None


def unavailable(index: pd.Index, wave: str, variable: str, issues: list[dict[str, object]], dtype: str) -> pd.Series:
    add_issue(issues, wave, "source_variable_unavailable", variable, len(index), f"{variable} is not released in the selected {wave} current-employment source.")
    return pd.Series(pd.NA, index=index, dtype=dtype)


def yes_no(context: AlignmentContext, aligned: list[tuple[pd.DataFrame, pd.DataFrame]], wave: str, variable: str, issues: list[dict[str, object]]) -> pd.Series:
    values, column = source_value(aligned, ALIASES[variable])
    if column is None:
        return unavailable(aligned[0][0].index, wave, variable, issues, "Int8")
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.notna() & ~numeric.isin([1, 2])
    add_issue(issues, wave, "unresolved_yes_no_code_set_null", variable, int(invalid.sum()), f"Codes outside 1/2 in {column} were set to null.")
    context.record(variable, wave, column, "derived")
    return numeric.map({1: 1, 2: 0}).astype("Int8")


def numeric_value(
    context: AlignmentContext, aligned: list[tuple[pd.DataFrame, pd.DataFrame]], wave: str,
    variable: str, issues: list[dict[str, object]], lower: int, upper: int,
    sentinels: set[int] | None = None, dtype: str = "Int16",
) -> pd.Series:
    values, column = source_value(aligned, ALIASES[variable])
    if column is None:
        return unavailable(aligned[0][0].index, wave, variable, issues, dtype)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.between(lower, upper) & numeric.mod(1).eq(0)
    if sentinels:
        valid &= ~numeric.isin(sentinels)
    invalid = numeric.notna() & ~valid
    add_issue(issues, wave, "invalid_value_set_null", variable, int(invalid.sum()), f"Values outside the retained domain in {column} were set to null.")
    context.record(variable, wave, column)
    return numeric.where(valid).astype(dtype)


def code_value(context: AlignmentContext, aligned: list[tuple[pd.DataFrame, pd.DataFrame]], wave: str, variable: str, issues: list[dict[str, object]]) -> pd.Series:
    return numeric_value(context, aligned, wave, variable, issues, 0, 9999, dtype="Int16")


def classification_code(
    context: AlignmentContext, aligned: list[tuple[pd.DataFrame, pd.DataFrame]], wave: str,
    variable: str, issues: list[dict[str, object]], width: int,
) -> pd.Series:
    values, column = source_value(aligned, ALIASES[variable])
    if column is None:
        return unavailable(aligned[0][0].index, wave, variable, issues, "string")
    cleaned, invalid = clean_code(values, width)
    add_issue(issues, wave, "invalid_classification_code_set_null", variable, invalid, f"Nonnumeric classification values in {column} were set to null.")
    context.record(variable, wave, column)
    return cleaned


def money_value(context: AlignmentContext, aligned: list[tuple[pd.DataFrame, pd.DataFrame]], wave: str, variable: str, issues: list[dict[str, object]]) -> pd.Series:
    values, column = source_value(aligned, ALIASES[variable])
    if column is None:
        return unavailable(aligned[0][0].index, wave, variable, issues, "Float64")
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.lt(0) | numeric.isin(MONEY_SENTINELS)
    add_issue(issues, wave, "invalid_value_set_null", variable, int((numeric.notna() & invalid).sum()), f"Negative values and exact missing sentinels in {column} were set to null.")
    context.record(variable, wave, column)
    return numeric.mask(invalid).astype("Float64")


def build_wave(context: AlignmentContext, hl: pd.DataFrame, wave: str, sources: list[object], issues: list[dict[str, object]]) -> pd.DataFrame:
    standardized, aligned = prepare_wave_sources(context, wave, sources)
    base_columns = [
        "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID", "Person ID",
        "Province Code", "District Code", "Commune Code", "Village Code", "Urban Rural",
        "Survey Month", "Stratum", "Source Archive", "Source Submodule", "Source Row ID",
    ]
    frame = standardized[base_columns].copy()
    for variable in CONTEXT_COLUMNS:
        if variable not in frame.columns:
            frame[variable] = pd.NA
    hl_wave = hl.loc[hl["Survey Wave"].eq(wave), ["Survey Wave", "Person ID", "Household ID", "PSU"] + CONTEXT_COLUMNS].copy()
    hl_wave = hl_wave.rename(columns={column: f"{column}_hl" for column in hl_wave.columns if column not in {"Survey Wave", "Person ID"}})
    frame = frame.merge(hl_wave, on=["Survey Wave", "Person ID"], how="left", validate="m:1", indicator="_hl_merge")
    frame["HL Link Matched"] = frame["_hl_merge"].eq("both").astype("Int8")
    context.record("HL Link Matched", wave, "", "derived")

    for key in ["Household ID", "PSU"]:
        conflict = frame[key].notna() & frame[f"{key}_hl"].notna() & frame[key].ne(frame[f"{key}_hl"])
        add_issue(issues, wave, "source_hl_key_conflict", key, int(conflict.sum()), "Source key differs from matched HL; the source key was retained.")
        frame[key] = frame[key].combine_first(frame[f"{key}_hl"])
    for variable in CONTEXT_COLUMNS:
        source = frame[variable]
        hl_value = frame[f"{variable}_hl"]
        conflict = source.notna() & hl_value.notna() & source.astype("string").ne(hl_value.astype("string"))
        add_issue(issues, wave, "source_hl_context_conflict", variable, int(conflict.sum()), "Matched HL value was used and the disagreement was reported.")
        frame[variable] = hl_value.combine_first(source)
        context.record(variable, wave, "", "derived")

    frame["Worked At Least One Hour Past 7 Days"] = yes_no(context, aligned, wave, "Worked At Least One Hour Past 7 Days", issues)
    frame["Second Work Screening Source Code"] = numeric_value(context, aligned, wave, "Second Work Screening Source Code", issues, 1, 2, dtype="Int8")
    for variable in HOUR_COLUMNS:
        frame[variable] = numeric_value(context, aligned, wave, variable, issues, 0, 168, {98, 99})
    frame["Main Days Worked Last Month"] = numeric_value(context, aligned, wave, "Main Days Worked Last Month", issues, 0, 31, {98, 99})
    frame["Secondary Days Worked Last Month"] = numeric_value(context, aligned, wave, "Secondary Days Worked Last Month", issues, 0, 31, {98, 99})
    frame["Main Occupation Source Code"] = classification_code(context, aligned, wave, "Main Occupation Source Code", issues, 3)
    frame["Secondary Occupation Source Code"] = classification_code(context, aligned, wave, "Secondary Occupation Source Code", issues, 3)
    industry_width = 2 if wave == "2004" else 4
    frame["Main Industry Source Code"] = classification_code(context, aligned, wave, "Main Industry Source Code", issues, industry_width)
    frame["Secondary Industry Source Code"] = classification_code(context, aligned, wave, "Secondary Industry Source Code", issues, industry_width)
    for variable in CODE_COLUMNS:
        frame[variable] = code_value(context, aligned, wave, variable, issues)
    for variable in YES_NO_COLUMNS - {"Worked At Least One Hour Past 7 Days"}:
        frame[variable] = yes_no(context, aligned, wave, variable, issues)
    frame["Additional Jobs Count"] = numeric_value(context, aligned, wave, "Additional Jobs Count", issues, 0, 10)
    released_total = numeric_value(context, aligned, wave, "Total Occupations Past 7 Days", issues, 0, 10)
    if wave in {"2004", "2007"}:
        frame["Total Occupations Past 7 Days"] = released_total
    else:
        has_main_job = frame["Main Occupation Source Code"].notna()
        frame["Total Occupations Past 7 Days"] = (frame["Additional Jobs Count"] + 1).where(has_main_job).astype("Int16")
        context.record("Total Occupations Past 7 Days", wave, find_column(aligned[0][0], ALIASES["Additional Jobs Count"]) or "", "derived")
    frame["Monthly Salary Wages Riel"] = money_value(context, aligned, wave, "Monthly Salary Wages Riel", issues)
    for variable in DURATION_COLUMNS:
        frame[variable] = numeric_value(context, aligned, wave, variable, issues, 0, 97, {98, 99})

    no_secondary = frame["Total Occupations Past 7 Days"].notna() & frame["Total Occupations Past 7 Days"].lt(2)
    for variable in SECONDARY_COLUMNS:
        inconsistent = no_secondary & frame[variable].notna()
        add_issue(issues, wave, "secondary_job_value_without_two_occupations_set_null", variable, int(inconsistent.sum()), "Secondary-job value was set to null when fewer than two occupations were reported.")
        frame.loc[no_secondary, variable] = pd.NA

    unmatched = frame.loc[frame["HL Link Matched"].eq(0), ["Person ID", "Household ID"]]
    for person_id, household_id in unmatched.itertuples(index=False, name=None):
        add_issue(issues, wave, "employment_record_not_in_hl", "Person ID", 1, f"Current-employment record retained: person_id={person_id} household_id={household_id}.")

    frame = frame.drop(columns=[column for column in frame.columns if column.endswith("_hl")] + ["_hl_merge"])[EC_COLUMNS]
    enforce_dtypes(frame)
    return frame


def enforce_dtypes(frame: pd.DataFrame) -> None:
    int8 = {"HL Link Matched", "Second Work Screening Source Code", *YES_NO_COLUMNS}
    int16 = {
        "Survey Year", "Survey Month", *CODE_COLUMNS, *HOUR_COLUMNS,
        "Main Days Worked Last Month", "Secondary Days Worked Last Month",
        "Additional Jobs Count", "Total Occupations Past 7 Days", *DURATION_COLUMNS,
    }
    floats = {"Household Weight", "Person Weight", "Monthly Salary Wages Riel"}
    for column in frame.columns:
        if column in int8:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int8")
        elif column in int16:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int16")
        elif column in {"Sex"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int8")
        elif column in {"Age"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int16")
        elif column in floats:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        else:
            frame[column] = frame[column].astype("string")


def variable_info(variable: str) -> tuple[str, str, str]:
    if variable in EC_VARIABLE_INFO:
        return EC_VARIABLE_INFO[variable]
    return VARIABLE_INFO[variable]


def write_metadata(context: AlignmentContext, output: Path) -> None:
    for wave in WAVES:
        for variable in ["Dataset Name", "Survey Wave", "Survey Year", "Source Archive", "Source Submodule", "Source Row ID"]:
            context.record(variable, wave, "", "derived")
    rows = []
    for variable, wave, raw_name, source_kind in sorted(
        context.mappings,
        key=lambda row: (EC_COLUMNS.index(row[0]) if row[0] in EC_COLUMNS else 999, WAVES.index(row[1]), row[2], row[3]),
    ):
        if variable not in EC_COLUMNS:
            continue
        full_name, measure_type, canonical_text = variable_info(variable)
        rows.append({
            "canonical_varname": snake_case(variable), "dataset_name": f"CSES {wave}",
            "column_in_raw_sav": raw_name, "column_label_in_english": full_name,
            "source_kind": source_kind, "measure_type": measure_type,
            "canonical_text": canonical_text,
        })
    dictionary = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    dictionary.to_csv(output / "ind_que_EC_CSES.csv", index=False)
    summary = []
    for variable in EC_COLUMNS:
        selected = dictionary.loc[dictionary["canonical_varname"].eq(snake_case(variable))]
        _full_name, measure_type, canonical_text = variable_info(variable)
        summary.append({
            "varname": snake_case(variable), "dataset_count": int(selected["dataset_name"].nunique()),
            "source_count": int(len(selected)), "explicit_count": int(selected["source_kind"].eq("explicit").sum()),
            "derived_count": int(selected["source_kind"].eq("derived").sum()),
            "measure_type": measure_type, "canonical_text": canonical_text,
        })
    pd.DataFrame(summary).to_csv(output / "align_summary_EC_CSES.csv", index=False)


def make_audit(frame: pd.DataFrame, hl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave in WAVES:
        part = frame.loc[frame["Survey Wave"].eq(wave)]
        hl_part = hl.loc[hl["Survey Wave"].eq(wave)]
        rows.append({
            "grain": "EC", "survey_wave": wave, "rows": len(part),
            "unique_persons": part["Person ID"].nunique(dropna=True),
            "duplicate_key_rows": int(part.duplicated(["Survey Wave", "Person ID"], keep=False).sum()),
            "missing_person_id": int(part["Person ID"].isna().sum()),
            "hl_link_matched": int(part["HL Link Matched"].eq(1).sum()),
            "hl_link_unmatched": int(part["HL Link Matched"].eq(0).sum()),
            "hl_rows_without_employment_record": int((~hl_part["Person ID"].isin(part["Person ID"])).sum()),
            "worked_screen_available": int(part["Worked At Least One Hour Past 7 Days"].notna().sum()),
            "main_occupation_available": int(part["Main Occupation Source Code"].notna().sum()),
            "total_hours_available": int(part["Total Hours Worked Past 7 Days"].notna().sum()),
            "monthly_wages_available": int(part["Monthly Salary Wages Riel"].notna().sum()),
            "actively_seeking_available": int(part["Actively Seeking Work"].notna().sum()),
        })
    return pd.DataFrame(rows)


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = initialize_context(root)
    hl = pd.read_parquet(root / "data/exp/database/final_HL_CSES.parquet")
    issues: list[dict[str, object]] = []
    frames = [build_wave(context, hl, wave, sources, issues) for wave, sources in employment_sources(root)]
    final = pd.concat(frames, ignore_index=True).sort_values(
        ["Survey Year", "Household ID", "Person ID", "Source Row ID"], na_position="last",
    ).reset_index(drop=True)
    audit = make_audit(final, hl)
    issue_frame = pd.concat([context.issues.frame(), pd.DataFrame(issues)], ignore_index=True).sort_values(
        ["survey_wave", "issue_type", "variable"], ignore_index=True,
    )
    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output / "final_EC_CSES.parquet", index=False)
    audit.to_csv(output / "cses_ec_alignment_audit.csv", index=False)
    issue_frame.to_csv(output / "cses_ec_data_issues.csv", index=False)
    write_metadata(context, output)
    return final, audit, issue_frame
