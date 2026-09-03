#!/usr/bin/env python3
"""Harmonize the household-grain CSES housing module across ten waves."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from cses_hh_hl_common import (
    AlignmentContext,
    HOUSEHOLD_WEIGHT_NAMES,
    VARIABLE_INFO,
    WAVES,
    find_column,
    initialize_context,
    snake_case,
    standardize_source,
)
from inventory_cses_archives import discover_sources, normalize_wave, source_leaf, token


HO_ALIASES = {
    "Households in Housing Unit": ["q03_01", "q04_01"],
    "Floor Area Square Meters": ["q03_02", "q04_02"],
    "Rooms Used": ["q03_03", "q04_03"],
    "Wall Material Source Code": ["q03_04fc", "q04_04"],
    "Roof Material Source Code": ["q03_06fc", "q04_05"],
    "Floor Material Source Code": ["q03_07fc", "q04_06"],
    "Main Lighting Source Code": ["q03_08", "q04_07"],
    "Main Drinking Water Source Code": ["q03_09", "q04_08"],
    "Monthly Water Charges Riel": ["q03_17", "q04_16"],
    "Drinking Water Treatment Frequency Source Code": ["q03_18", "q04_17"],
    "Boils Drinking Water": ["q03_19a", "q04_18a"],
    "Filters Drinking Water": ["q03_19b", "q04_18b"],
    "Uses Chemical Water Treatment": ["q03_19c", "q04_18c"],
    "Uses Alum Water Treatment": ["q03_19d", "q04_18d"],
    "Uses Other Water Treatment": ["q03_19e", "q04_18e"],
    "Toilet Facility Source Code": ["q03_20", "q04_19a"],
    "Monthly Sewage Disposal Expense Riel": ["q03_21", "q04_20"],
    "Monthly Garbage Collection Expense Riel": ["q03_22", "q04_21"],
}

HO_COLUMNS = [
    "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID",
    "HH Link Matched", "Province Code", "District Code", "Commune Code",
    "Village Code", "Urban Rural", "Survey Month", "Stratum", "Household Weight",
    "Households in Housing Unit", "Floor Area Square Meters", "Rooms Used",
    "Wall Material Source Code", "Roof Material Source Code", "Floor Material Source Code",
    "Main Lighting Source Code", "Main Drinking Water Source Code",
    "Monthly Water Charges Riel", "Drinking Water Treatment Frequency Source Code",
    "Treats Drinking Water", "Boils Drinking Water", "Filters Drinking Water",
    "Uses Chemical Water Treatment", "Uses Alum Water Treatment",
    "Uses Other Water Treatment", "Toilet Facility Source Code", "Has Toilet Facility",
    "Monthly Sewage Disposal Expense Riel", "Monthly Garbage Collection Expense Riel",
    "Main Cooking Fuel Source Code", "Monthly Electricity Expense Riel",
    "Monthly Gas Expense Riel", "Monthly Kerosene Expense Riel",
    "Monthly Firewood Expense Riel", "Monthly Charcoal Expense Riel",
    "Monthly Battery Expense Riel", "Monthly Other Energy Expense Riel",
    "Dwelling Tenure Source Code", "Dwelling Tenure Harmonized",
    "Monthly Rent Paid Riel", "Monthly Imputed Rent Riel",
    "Dwelling Maintenance Expense Riel",
    "Source Archive", "Source Submodule", "Source Row ID",
]

HO_VARIABLE_INFO = {
    "HH Link Matched": ("Household Link Matched", "provenance", "1 when the housing record matches final_HH_CSES on survey wave and household identifier; otherwise 0."),
    "Households in Housing Unit": ("Households Residing in the Housing Unit", "housing", "Released number of households sharing the housing unit, retained as a positive integer."),
    "Floor Area Square Meters": ("Occupied Dwelling Floor Area", "housing", "Released occupied floor area in square meters; positive values are retained without imputation."),
    "Rooms Used": ("Rooms Used by the Household", "housing", "Released number of rooms used by the household, excluding dedicated kitchen and bathroom spaces where specified."),
    "Wall Material Source Code": ("Primary Wall Material Source Code", "housing", "Released wave-specific primary wall-material code; consult the variable dictionary before comparing detailed categories across waves."),
    "Roof Material Source Code": ("Primary Roof Material Source Code", "housing", "Released wave-specific primary roof-material code; consult the variable dictionary before comparing detailed categories across waves."),
    "Floor Material Source Code": ("Primary Floor Material Source Code", "housing", "Released wave-specific primary floor-material code; consult the variable dictionary before comparing detailed categories across waves."),
    "Main Lighting Source Code": ("Main Lighting Source Code", "housing", "Released wave-specific main lighting-source code."),
    "Main Drinking Water Source Code": ("Main Wet-Season Drinking Water Source Code", "water and sanitation", "Released wave-specific main drinking-water source in the wet season."),
    "Monthly Water Charges Riel": ("Monthly Water Charges", "housing expenditure", "Released monthly water charges in Cambodian riel; exact missing sentinels and negative values are null, with no imputation or price adjustment."),
    "Drinking Water Treatment Frequency Source Code": ("Drinking Water Treatment Frequency Source Code", "water and sanitation", "Released frequency code: 1=always, 2=sometimes, 3=never."),
    "Treats Drinking Water": ("Treats Drinking Water", "water and sanitation", "Harmonized indicator: 1 when treatment is always or sometimes, 0 when never, and null otherwise."),
    "Boils Drinking Water": ("Boils Drinking Water", "water and sanitation", "Harmonized released response: 1=Yes and 0=No."),
    "Filters Drinking Water": ("Filters Drinking Water", "water and sanitation", "Harmonized released response: 1=Yes and 0=No."),
    "Uses Chemical Water Treatment": ("Uses Chemical Water Treatment", "water and sanitation", "Harmonized released response: 1=Yes and 0=No."),
    "Uses Alum Water Treatment": ("Uses Alum Water Treatment", "water and sanitation", "Harmonized released response: 1=Yes and 0=No."),
    "Uses Other Water Treatment": ("Uses Other Water Treatment", "water and sanitation", "Harmonized released response: 1=Yes and 0=No."),
    "Toilet Facility Source Code": ("Toilet Facility Source Code", "water and sanitation", "Released wave-specific toilet-facility code; detailed categories change in 2019."),
    "Has Toilet Facility": ("Has a Toilet Facility", "water and sanitation", "Wave-specific harmonized indicator distinguishing a reported toilet facility from no facility or open-land defecation."),
    "Monthly Sewage Disposal Expense Riel": ("Monthly Sewage or Wastewater Disposal Expense", "housing expenditure", "Released monthly sewage or wastewater-disposal expense in Cambodian riel; no imputation or price adjustment."),
    "Monthly Garbage Collection Expense Riel": ("Monthly Garbage Collection Expense", "housing expenditure", "Released monthly garbage-collection expense in Cambodian riel; no imputation or price adjustment."),
    "Main Cooking Fuel Source Code": ("Main Cooking Fuel Source Code", "housing", "Released wave-specific main cooking-fuel code."),
    "Monthly Electricity Expense Riel": ("Monthly Electricity Expense", "housing expenditure", "Released household electricity expense in Cambodian riel; no imputation or price adjustment."),
    "Monthly Gas Expense Riel": ("Monthly Gas Expense", "housing expenditure", "Released household gas or LPG expense in Cambodian riel; no imputation or price adjustment."),
    "Monthly Kerosene Expense Riel": ("Monthly Kerosene Expense", "housing expenditure", "Released household kerosene expense in Cambodian riel; no imputation or price adjustment."),
    "Monthly Firewood Expense Riel": ("Monthly Firewood Expense", "housing expenditure", "Released household firewood expense in Cambodian riel; no imputation or price adjustment."),
    "Monthly Charcoal Expense Riel": ("Monthly Charcoal Expense", "housing expenditure", "Released household charcoal expense in Cambodian riel; no imputation or price adjustment."),
    "Monthly Battery Expense Riel": ("Monthly Battery Expense", "housing expenditure", "Released household battery expense in Cambodian riel; no imputation or price adjustment."),
    "Monthly Other Energy Expense Riel": ("Monthly Other Energy Expense", "housing expenditure", "Released other household energy expense in Cambodian riel; no imputation or price adjustment."),
    "Dwelling Tenure Source Code": ("Dwelling Tenure Source Code", "housing", "Released dwelling legal-status or tenure code."),
    "Dwelling Tenure Harmonized": ("Dwelling Tenure Harmonized", "housing", "Cross-wave code: 1=owned, 2=occupied rent-free, 3=rented, 4=other."),
    "Monthly Rent Paid Riel": ("Monthly Rent Paid", "housing expenditure", "Released monthly dwelling rent paid in Cambodian riel; structurally null for non-renting households."),
    "Monthly Imputed Rent Riel": ("Monthly Imputed Rent", "housing expenditure", "Released estimated monthly rent for a similar dwelling in Cambodian riel; unavailable in 2004."),
    "Dwelling Maintenance Expense Riel": ("Dwelling Maintenance and Minor Repair Expense", "housing expenditure", "Released dwelling maintenance and minor-repair expense in Cambodian riel; retain source reference-period interpretation."),
}

CONTEXT_COLUMNS = [
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Urban Rural", "Survey Month", "Stratum", "Household Weight",
]
YES_NO_COLUMNS = [
    "Boils Drinking Water", "Filters Drinking Water",
    "Uses Chemical Water Treatment", "Uses Alum Water Treatment",
    "Uses Other Water Treatment",
]
MONEY_COLUMNS = [
    "Monthly Water Charges Riel", "Monthly Sewage Disposal Expense Riel",
    "Monthly Garbage Collection Expense Riel", "Monthly Electricity Expense Riel",
    "Monthly Gas Expense Riel", "Monthly Kerosene Expense Riel",
    "Monthly Firewood Expense Riel", "Monthly Charcoal Expense Riel",
    "Monthly Battery Expense Riel", "Monthly Other Energy Expense Riel",
    "Monthly Rent Paid Riel", "Monthly Imputed Rent Riel",
    "Dwelling Maintenance Expense Riel",
]
MONEY_SENTINELS = {9_999_999, 99_999_999, 999_999_999}


def housing_sources(root: Path) -> list[tuple[str, object]]:
    sources = []
    for source in discover_sources(root):
        wave = normalize_wave(source.display_name(root))
        leaf_key = token(Path(source_leaf(source)).stem)
        if wave in WAVES and re.search(r"housing$", leaf_key):
            sources.append((wave, source))
    sources.sort(key=lambda item: (WAVES.index(item[0]), item[1].display_name(root)))
    counts = pd.Series([wave for wave, _ in sources]).value_counts().to_dict()
    bad = {wave: counts.get(wave, 0) for wave in WAVES if counts.get(wave, 0) != 1}
    if bad:
        raise RuntimeError(f"Expected exactly one housing source per wave; found {bad}")
    return sources


def add_issue(
    rows: list[dict[str, object]], wave: str, issue_type: str,
    variable: str, affected_rows: int, detail: str,
) -> None:
    if affected_rows:
        rows.append({
            "grain": "HO", "survey_wave": wave, "issue_type": issue_type,
            "variable": variable, "affected_rows": int(affected_rows), "detail": detail,
        })


def wave_aliases(wave: str) -> dict[str, list[str]]:
    if wave == "2004":
        return {
            "Main Cooking Fuel Source Code": ["q03_23a"],
            "Monthly Electricity Expense Riel": ["q03_24a"],
            "Monthly Gas Expense Riel": ["q03_24b"],
            "Monthly Kerosene Expense Riel": ["q03_24c"],
            "Monthly Firewood Expense Riel": ["q03_24d"],
            "Monthly Charcoal Expense Riel": ["q03_24e"],
            "Monthly Battery Expense Riel": ["q03_24f"],
            "Monthly Other Energy Expense Riel": ["q03_24g"],
            "Dwelling Tenure Source Code": ["q03_27"],
            "Monthly Rent Paid Riel": ["q03_28"],
            "Monthly Imputed Rent Riel": [],
            "Dwelling Maintenance Expense Riel": ["q03_29"],
        }
    if wave in {"2007", "2009", "2011-12", "2013"}:
        return {
            "Main Cooking Fuel Source Code": ["q04_22a"],
            "Monthly Electricity Expense Riel": ["q04_23a"],
            "Monthly Gas Expense Riel": ["q04_23b"],
            "Monthly Kerosene Expense Riel": ["q04_23c"],
            "Monthly Firewood Expense Riel": ["q04_23d"],
            "Monthly Charcoal Expense Riel": ["q04_23e"],
            "Monthly Battery Expense Riel": ["q04_23f"],
            "Monthly Other Energy Expense Riel": ["q04_23g"],
            "Dwelling Tenure Source Code": ["q04_24"],
            "Monthly Rent Paid Riel": ["q04_25a"],
            "Monthly Imputed Rent Riel": ["q04_25b"],
            "Dwelling Maintenance Expense Riel": ["q04_26"],
        }
    return {
        "Main Cooking Fuel Source Code": ["q04_26a"] if wave in {"2014", "2016", "2017"} else ["q04_26"],
        "Monthly Electricity Expense Riel": ["q04_27a"],
        "Monthly Gas Expense Riel": ["q04_27b"],
        "Monthly Kerosene Expense Riel": ["q04_27c"],
        "Monthly Firewood Expense Riel": ["q04_27d"],
        "Monthly Charcoal Expense Riel": ["q04_27e"],
        "Monthly Battery Expense Riel": ["q04_27f"],
        "Monthly Other Energy Expense Riel": ["q04_27g"],
        "Dwelling Tenure Source Code": ["q04_28"],
        "Monthly Rent Paid Riel": ["q04_29a"],
        "Monthly Imputed Rent Riel": ["q04_29b"],
        "Dwelling Maintenance Expense Riel": ["q04_30"],
    }


def clean_integer(
    context: AlignmentContext, raw: pd.DataFrame, wave: str, variable: str,
    aliases: list[str], lower: int, upper: int, issues: list[dict[str, object]],
) -> pd.Series:
    column = find_column(raw, aliases)
    if column is None:
        raise RuntimeError(f"Missing {variable} source column in {wave}")
    numeric = pd.to_numeric(raw[column], errors="coerce")
    valid = numeric.between(lower, upper) & numeric.mod(1).eq(0)
    add_issue(issues, wave, "invalid_value_set_null", variable, int((numeric.notna() & ~valid).sum()), f"Values outside integer range {lower}-{upper} in {column} were set to null.")
    context.record(variable, wave, column)
    return numeric.where(valid).astype("Int16")


def clean_source_code(
    context: AlignmentContext, raw: pd.DataFrame, wave: str, variable: str,
    aliases: list[str], valid_codes: set[int], issues: list[dict[str, object]],
) -> pd.Series:
    column = find_column(raw, aliases)
    if column is None:
        raise RuntimeError(f"Missing {variable} source column in {wave}")
    numeric = pd.to_numeric(raw[column], errors="coerce")
    valid = numeric.isin(valid_codes)
    add_issue(issues, wave, "unresolved_code_set_null", variable, int((numeric.notna() & ~valid).sum()), f"Codes outside the documented set in {column} were set to null.")
    context.record(variable, wave, column)
    return numeric.where(valid).astype("Int16")


def clean_money(
    context: AlignmentContext, raw: pd.DataFrame, wave: str, variable: str,
    aliases: list[str], issues: list[dict[str, object]], required: bool = True,
) -> pd.Series:
    column = find_column(raw, aliases)
    if column is None:
        if required:
            raise RuntimeError(f"Missing {variable} source column in {wave}")
        add_issue(issues, wave, "source_variable_unavailable", variable, len(raw), f"{variable} is not released in {wave}.")
        return pd.Series(pd.NA, index=raw.index, dtype="Float64")
    numeric = pd.to_numeric(raw[column], errors="coerce")
    invalid = numeric.lt(0) | numeric.isin(MONEY_SENTINELS)
    add_issue(issues, wave, "invalid_value_set_null", variable, int((numeric.notna() & invalid).sum()), f"Negative values and exact missing sentinels in {column} were set to null.")
    context.record(variable, wave, column)
    return numeric.mask(invalid).astype("Float64")


def build_wave(
    context: AlignmentContext, hh: pd.DataFrame, wave: str, source: object,
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    raw = context.load(source)
    standardized = standardize_source(context, source, wave, "HO")
    base_columns = [
        "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID",
        "Province Code", "District Code", "Commune Code", "Village Code",
        "Urban Rural", "Survey Month", "Stratum", "Source Archive",
        "Source Submodule", "Source Row ID",
    ]
    frame = standardized[base_columns].copy()

    hh_wave = hh.loc[
        hh["Survey Wave"].eq(wave), ["Survey Wave", "Household ID", "PSU"] + CONTEXT_COLUMNS,
    ].rename(columns={
        column: f"{column}_hh"
        for column in ["PSU"] + CONTEXT_COLUMNS
    })
    frame = frame.merge(hh_wave, on=["Survey Wave", "Household ID"], how="left", validate="m:1", indicator="_hh_merge")
    frame["HH Link Matched"] = frame["_hh_merge"].eq("both").astype("Int8")
    context.record("HH Link Matched", wave, "", "derived")

    source_weight_column = find_column(raw, HOUSEHOLD_WEIGHT_NAMES.get(wave, []))
    source_weight = (
        pd.to_numeric(raw[source_weight_column], errors="coerce").astype("Float64")
        if source_weight_column else pd.Series(pd.NA, index=raw.index, dtype="Float64")
    )
    if source_weight_column:
        invalid_weight = source_weight.le(0)
        add_issue(issues, wave, "invalid_value_set_null", "Household Weight", int((source_weight.notna() & invalid_weight).sum()), f"Non-positive values in {source_weight_column} were set to null.")
        source_weight = source_weight.mask(invalid_weight)

    for variable in ["PSU"] + CONTEXT_COLUMNS:
        source_value = source_weight if variable == "Household Weight" else frame[variable]
        hh_value = frame[f"{variable}_hh"]
        conflict = source_value.notna() & hh_value.notna() & source_value.astype("string").ne(hh_value.astype("string"))
        add_issue(issues, wave, "source_hh_context_conflict", variable, int(conflict.sum()), "Matched HH value was used and the disagreement was reported.")
        frame[variable] = hh_value.combine_first(source_value)
        raw_name = (source_weight_column or "") if variable == "Household Weight" else ""
        context.record(variable, wave, raw_name, "derived")

    frame["Households in Housing Unit"] = clean_integer(context, raw, wave, "Households in Housing Unit", HO_ALIASES["Households in Housing Unit"], 1, 50, issues)
    area_column = find_column(raw, HO_ALIASES["Floor Area Square Meters"])
    if area_column is None:
        raise RuntimeError(f"Missing Floor Area Square Meters source column in {wave}")
    area = pd.to_numeric(raw[area_column], errors="coerce")
    valid_area = area.gt(0) & area.le(10_000)
    add_issue(issues, wave, "invalid_value_set_null", "Floor Area Square Meters", int((area.notna() & ~valid_area).sum()), f"Non-positive or implausibly large values in {area_column} were set to null.")
    frame["Floor Area Square Meters"] = area.where(valid_area).astype("Float64")
    context.record("Floor Area Square Meters", wave, area_column)
    frame["Rooms Used"] = clean_integer(context, raw, wave, "Rooms Used", HO_ALIASES["Rooms Used"], 1, 50, issues)

    code_sets = {
        "Wall Material Source Code": set(range(1, 9 if wave == "2004" else 10)),
        "Roof Material Source Code": set(range(1, 11)),
        "Floor Material Source Code": set(range(1, 9 if wave == "2004" else 10)),
        "Main Lighting Source Code": set(range(1, 11)),
        "Main Drinking Water Source Code": set(range(1, 21)),
        "Drinking Water Treatment Frequency Source Code": {1, 2, 3},
        "Toilet Facility Source Code": set(range(1, 10 if wave == "2004" else 9)),
    }
    for variable, valid_codes in code_sets.items():
        frame[variable] = clean_source_code(context, raw, wave, variable, HO_ALIASES[variable], valid_codes, issues)

    treatment_column = find_column(raw, HO_ALIASES["Drinking Water Treatment Frequency Source Code"])
    frame["Treats Drinking Water"] = pd.to_numeric(raw[treatment_column], errors="coerce").map({1: 1, 2: 1, 3: 0}).astype("Int8")
    context.record("Treats Drinking Water", wave, treatment_column, "derived")
    for variable in YES_NO_COLUMNS:
        column = find_column(raw, HO_ALIASES[variable])
        numeric = pd.to_numeric(raw[column], errors="coerce")
        frame[variable] = numeric.map({1: 1, 2: 0}).astype("Int8")
        add_issue(issues, wave, "unresolved_yes_no_code_set_null", variable, int((numeric.notna() & ~numeric.isin([1, 2, 9])).sum()), f"Codes outside 1/2 and the documented missing sentinel in {column} were set to null.")
        context.record(variable, wave, column, "derived")

    toilet = frame["Toilet Facility Source Code"]
    no_toilet_codes = {7, 8} if wave not in {"2019", "2021"} else {1}
    frame["Has Toilet Facility"] = toilet.map(lambda value: pd.NA if pd.isna(value) else int(int(value) not in no_toilet_codes)).astype("Int8")
    toilet_column = find_column(raw, HO_ALIASES["Toilet Facility Source Code"])
    context.record("Has Toilet Facility", wave, toilet_column, "derived")

    aliases = wave_aliases(wave)
    frame["Main Cooking Fuel Source Code"] = clean_source_code(
        context, raw, wave, "Main Cooking Fuel Source Code",
        aliases["Main Cooking Fuel Source Code"], set(range(1, 16)), issues,
    )
    tenure = clean_source_code(context, raw, wave, "Dwelling Tenure Source Code", aliases["Dwelling Tenure Source Code"], {1, 2, 3, 4}, issues)
    frame["Dwelling Tenure Source Code"] = tenure
    frame["Dwelling Tenure Harmonized"] = tenure.astype("Int8")
    tenure_column = find_column(raw, aliases["Dwelling Tenure Source Code"])
    context.record("Dwelling Tenure Harmonized", wave, tenure_column, "derived")

    common_money = {
        "Monthly Water Charges Riel": HO_ALIASES["Monthly Water Charges Riel"],
        "Monthly Sewage Disposal Expense Riel": HO_ALIASES["Monthly Sewage Disposal Expense Riel"],
        "Monthly Garbage Collection Expense Riel": HO_ALIASES["Monthly Garbage Collection Expense Riel"],
    }
    for variable, variable_aliases in {**common_money, **{k: v for k, v in aliases.items() if k in MONEY_COLUMNS}}.items():
        frame[variable] = clean_money(
            context, raw, wave, variable, variable_aliases, issues,
            required=not (wave == "2004" and variable == "Monthly Imputed Rent Riel"),
        )

    drop_columns = [column for column in frame.columns if column.endswith("_hh")] + ["_hh_merge"]
    frame = frame.drop(columns=drop_columns)[HO_COLUMNS]
    enforce_dtypes(frame)

    unmatched = frame.loc[frame["HH Link Matched"].eq(0), ["Household ID"]]
    for household_id in unmatched["Household ID"]:
        add_issue(issues, wave, "housing_record_not_in_hh", "Household ID", 1, f"Housing record retained: household_id={household_id}.")
    return frame


def enforce_dtypes(frame: pd.DataFrame) -> None:
    int16_columns = {
        "Survey Year", "Survey Month", "Households in Housing Unit", "Rooms Used",
        "Wall Material Source Code", "Roof Material Source Code", "Floor Material Source Code",
        "Main Lighting Source Code", "Main Drinking Water Source Code",
        "Toilet Facility Source Code", "Main Cooking Fuel Source Code",
        "Dwelling Tenure Source Code",
    }
    int8_columns = {
        "HH Link Matched", "Drinking Water Treatment Frequency Source Code",
        "Treats Drinking Water", *YES_NO_COLUMNS, "Has Toilet Facility",
        "Dwelling Tenure Harmonized",
    }
    float_columns = {"Floor Area Square Meters", "Household Weight", *MONEY_COLUMNS}
    for column in frame.columns:
        if column in int16_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int16")
        elif column in int8_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int8")
        elif column in float_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        else:
            frame[column] = frame[column].astype("string")


def variable_info(variable: str) -> tuple[str, str, str]:
    if variable in HO_VARIABLE_INFO:
        return HO_VARIABLE_INFO[variable]
    return VARIABLE_INFO[variable]


def write_metadata(context: AlignmentContext, output: Path) -> None:
    for wave in WAVES:
        for variable in ["Dataset Name", "Survey Wave", "Survey Year", "Source Archive", "Source Submodule", "Source Row ID"]:
            context.record(variable, wave, "", "derived")
    rows = []
    for variable, wave, raw_name, source_kind in sorted(
        context.mappings,
        key=lambda row: (HO_COLUMNS.index(row[0]) if row[0] in HO_COLUMNS else 999, WAVES.index(row[1]), row[2], row[3]),
    ):
        if variable not in HO_COLUMNS:
            continue
        full_name, measure_type, canonical_text = variable_info(variable)
        rows.append({
            "canonical_varname": snake_case(variable), "dataset_name": f"CSES {wave}",
            "column_in_raw_sav": raw_name, "column_label_in_english": full_name,
            "source_kind": source_kind, "measure_type": measure_type,
            "canonical_text": canonical_text,
        })
    ind_que = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    ind_que.to_csv(output / "ind_que_HO_CSES.csv", index=False)

    summary = []
    for variable in HO_COLUMNS:
        selected = ind_que.loc[ind_que["canonical_varname"].eq(snake_case(variable))]
        _full_name, measure_type, canonical_text = variable_info(variable)
        summary.append({
            "varname": snake_case(variable), "dataset_count": int(selected["dataset_name"].nunique()),
            "source_count": int(len(selected)), "explicit_count": int(selected["source_kind"].eq("explicit").sum()),
            "derived_count": int(selected["source_kind"].eq("derived").sum()),
            "measure_type": measure_type, "canonical_text": canonical_text,
        })
    pd.DataFrame(summary).to_csv(output / "align_summary_HO_CSES.csv", index=False)


def make_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave in WAVES:
        part = frame.loc[frame["Survey Wave"].eq(wave)]
        rows.append({
            "grain": "HO", "survey_wave": wave, "rows": len(part),
            "unique_households": part["Household ID"].nunique(dropna=True),
            "duplicate_key_rows": int(part.duplicated(["Survey Wave", "Household ID"], keep=False).sum()),
            "missing_household_id": int(part["Household ID"].isna().sum()),
            "hh_link_matched": int(part["HH Link Matched"].eq(1).sum()),
            "hh_link_unmatched": int(part["HH Link Matched"].eq(0).sum()),
            "floor_area_available": int(part["Floor Area Square Meters"].notna().sum()),
            "rooms_available": int(part["Rooms Used"].notna().sum()),
            "water_source_available": int(part["Main Drinking Water Source Code"].notna().sum()),
            "toilet_available": int(part["Has Toilet Facility"].notna().sum()),
            "tenure_available": int(part["Dwelling Tenure Harmonized"].notna().sum()),
        })
    return pd.DataFrame(rows)


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = initialize_context(root)
    hh = pd.read_parquet(root / "data/exp/database/final_HH_CSES.parquet")
    issues: list[dict[str, object]] = []
    frames = [build_wave(context, hh, wave, source, issues) for wave, source in housing_sources(root)]
    final = pd.concat(frames, ignore_index=True).sort_values(
        ["Survey Year", "Household ID", "Source Row ID"], na_position="last",
    ).reset_index(drop=True)
    audit = make_audit(final)
    issue_frame = pd.concat([context.issues.frame(), pd.DataFrame(issues)], ignore_index=True).sort_values(
        ["survey_wave", "issue_type", "variable"], ignore_index=True,
    )

    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output / "final_HO_CSES.parquet", index=False)
    audit.to_csv(output / "cses_ho_alignment_audit.csv", index=False)
    issue_frame.to_csv(output / "cses_ho_data_issues.csv", index=False)
    write_metadata(context, output)
    return final, audit, issue_frame
