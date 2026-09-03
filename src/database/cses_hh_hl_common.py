#!/usr/bin/env python3
"""Shared CSES HH/HL linkage-spine harmonization.

The source releases contain Stata files inside ZIP and nested-ZIP archives.
Everything is read in memory. Raw archives are never unpacked or modified.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from inventory_cses_archives import (
    DataSource,
    discover_sources,
    modules_for_source,
    normalize_wave,
)


WAVES = ["2004", "2007", "2009", "2011-12", "2013", "2014", "2016", "2017", "2019", "2021"]
WAVE_YEAR = {wave: int(wave[:4]) for wave in WAVES}
ID_WIDTHS = {
    wave: ({"PSU": 4, "Household ID": 6, "Person ID": 8} if wave == "2004" else {"PSU": 5, "Household ID": 7, "Person ID": 9})
    for wave in WAVES
}
GEO_WIDTHS = {"Province Code": 2, "District Code": 4, "Commune Code": 6, "Village Code": 8}

FIELD_ALIASES = {
    "PSU": ["psu", "psu11", "vid", "vid11"],
    "Household ID": ["hhid"],
    "Person ID": ["persid"],
    "Province Code": ["provincecode", "province"],
    "District Code": ["districtcode", "district"],
    "Commune Code": ["communecode", "commune"],
    "Village Code": ["villagecode", "village"],
    "Urban Rural": ["urbanrural", "urbrura", "urbrur", "urban"],
    "Survey Month": ["surveymonth", "surveymonths", "surveymonthcode", "monthsur", "surveymonth"],
    "Stratum": ["stratum", "strata"],
}

DEMOGRAPHIC_ALIASES = {
    "Sex": ["q01a03", "q01ac03"],
    "Age": ["q01a05", "q01ac05", "q01ac05a"],
    "Relationship to Household Head": ["q01a06", "q01ac06"],
    "Absent From Household": ["q01a13", "q01ac13"],
}

EXTENDED_HL_ALIASES = {
    "Member Line Number": ["q01a01", "q01ac01"],
    "Birth Day": ["q01a04a", "q01ac04a"],
    "Birth Month": ["q01a04b", "q01ac04b"],
    "Birth Year": ["q01a04c", "q01ac04c"],
    "Father Line Number": ["q01a07", "q01ac07"],
    "Mother Line Number": ["q01a08", "q01ac08"],
    "Marital Status Source Code": ["q01a09", "q01ac09"],
    "Spouse Line Number": ["q01a10", "q01ac10"],
    "Ethnicity Source Code": ["q01a11", "q01ac11a"],
}

HOUSEHOLD_WEIGHT_NAMES = {
    "2007": ["hhweightadjusted"],
    "2009": ["hw09a"],
    "2011-12": ["hw11a"],
    "2013": ["hw13a"],
    "2014": ["hw14a"],
    "2016": ["hw16a"],
    "2017": ["hw17a"],
    "2019": ["hw20a"],
    "2021": ["hw21a", "hw2021"],
}
PERSON_WEIGHT_NAMES = {
    "2007": ["personweightadjusted"],
    "2009": ["pw09a"],
    "2011-12": ["pw11a"],
    "2013": ["pw13a"],
    "2014": ["pw14a"],
    "2016": ["pw16a"],
    "2017": ["pw17a"],
    "2019": ["pw20a"],
    "2021": ["pw2021"],
}

HH_DERIVED_COLUMNS = [
    "Household Member Count", "Male Member Count", "Female Member Count",
    "Child Member Count 0-14", "Working Age Member Count 15-64",
    "Older Member Count 65 Plus", "Unknown Age Member Count",
    "Household Head Person ID", "Household Head Sex", "Household Head Age",
    "Household Head Marital Status", "Household Head Ethnicity",
    "Household Head Education Level", "Household Head Years Attended School",
    "Household Head Can Read", "Household Head Can Write",
]
HH_COLUMNS = [
    "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID",
    *HH_DERIVED_COLUMNS,
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Urban Rural", "Survey Month", "Stratum", "Household Weight",
    "Source Archive", "Source Submodule", "Source Row ID",
]
HL_COLUMNS = [
    "Dataset Name", "Survey Wave", "Survey Year", "PSU", "Household ID", "Person ID",
    "Member Line Number", "Sex", "Birth Day", "Birth Month", "Birth Year", "Age",
    "Relationship to Household Head", "Father Line Number", "Father Person ID",
    "Mother Line Number", "Mother Person ID", "Marital Status Source Code",
    "Marital Status Harmonized", "Spouse Line Number", "Spouse Person ID",
    "Ethnicity Source Code", "Ethnicity Harmonized", "Absent From Household",
    "Presence Reference Period",
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Urban Rural", "Survey Month", "Stratum", "Household Weight", "Person Weight",
    "Source Archive", "Source Submodule", "Source Row ID",
]
ALIGNMENT_FIELDS = [
    "PSU", "Household ID", "Person ID", "Province Code", "District Code",
    "Commune Code", "Village Code", "Urban Rural", "Survey Month", "Stratum",
]
HH_CONTEXT_FIELDS = [
    "Province Code", "District Code", "Commune Code", "Village Code",
    "Urban Rural", "Survey Month", "Stratum",
]

VARIABLE_INFO = {
    "Dataset Name": ("CSES Dataset Name", "identifier", "CSES plus the normalized survey wave."),
    "Survey Wave": ("CSES Survey Wave", "identifier", "Normalized CSES release wave."),
    "Survey Year": ("CSES Survey Start Year", "time", "First calendar year represented by the normalized wave."),
    "PSU": ("Primary Sampling Unit Identifier", "identifier", "Wave-normalized primary sampling unit identifier."),
    "Household ID": ("Household Identifier", "identifier", "Wave-normalized household identifier."),
    "Person ID": ("Person Identifier", "identifier", "Wave-normalized person identifier."),
    "Household Member Count": ("Household Member Count", "household composition", "Number of released household-roster members in the survey-wave household."),
    "Male Member Count": ("Male Household Member Count", "household composition", "Number of roster members with harmonized Sex=1."),
    "Female Member Count": ("Female Household Member Count", "household composition", "Number of roster members with harmonized Sex=2."),
    "Child Member Count 0-14": ("Household Child Member Count Ages 0-14", "household composition", "Number of roster members with valid completed age from 0 through 14."),
    "Working Age Member Count 15-64": ("Household Working-Age Member Count Ages 15-64", "household composition", "Number of roster members with valid completed age from 15 through 64."),
    "Older Member Count 65 Plus": ("Household Older Member Count Ages 65 Plus", "household composition", "Number of roster members with valid completed age of 65 or older."),
    "Unknown Age Member Count": ("Household Member Count with Unknown Age", "household composition", "Number of roster members whose completed age is null after sentinel and range cleaning."),
    "Household Head Person ID": ("Household Head Person Identifier", "household head", "Person identifier for the unique member coded as relationship-to-head=1; null when the household has no unique coded head."),
    "Household Head Sex": ("Sex of Household Head", "household head", "Harmonized sex code inherited from the unique coded household head: 1=Male and 2=Female."),
    "Household Head Age": ("Age of Household Head", "household head", "Cleaned completed age inherited from the unique coded household head."),
    "Household Head Marital Status": ("Marital Status of Household Head", "household head", "Four-category harmonized marital status inherited from the unique coded household head."),
    "Household Head Ethnicity": ("Ethnicity of Household Head", "household head", "Eight-category harmonized ethnicity inherited from the unique coded household head."),
    "Household Head Education Level": ("Highest Completed Education Level of Household Head", "household head education", "Broad harmonized completed education level inherited from the matched education record for the coded household head."),
    "Household Head Years Attended School": ("Completed Years Attended School of Household Head", "household head education", "Cleaned completed years attended inherited from the matched education record for the coded household head; unavailable in 2004."),
    "Household Head Can Read": ("Household Head Can Read a Simple Message", "household head education", "Harmonized literacy indicator inherited from the matched education record: 1=Yes and 0=No."),
    "Household Head Can Write": ("Household Head Can Write a Simple Message", "household head education", "Harmonized writing indicator inherited from the matched education record: 1=Yes and 0=No."),
    "Sex": ("Sex of Household Member", "demographics", "Released code: 1=Male and 2=Female."),
    "Age": ("Age in Completed Years", "demographics", "Completed age in years, retained from 0 through 120 without imputation."),
    "Member Line Number": ("Household Member Roster Line Number", "identifier", "Released within-household member line number, retained as an integer from 1 through 98."),
    "Birth Day": ("Day of Birth", "demographics", "Released day of birth retained from 1 through 31; don't-know and missing sentinels are null."),
    "Birth Month": ("Month of Birth", "demographics", "Released month of birth retained from 1 through 12; don't-know and missing sentinels are null."),
    "Birth Year": ("Year of Birth", "demographics", "Released year of birth retained from 1800 through the survey start year plus one; don't-know and missing sentinels are null."),
    "Relationship to Household Head": ("Relationship to Household Head", "demographics", "Released relationship code from 1 through 15; the code list is stable across labeled waves."),
    "Father Line Number": ("Father Roster Line Number", "kinship", "Released within-household line number for the member's father; no-parent-in-household and missing codes are null."),
    "Father Person ID": ("Father Person Identifier", "kinship", "Person identifier derived only when Father Line Number matches another member of the same survey-wave household and is not a self-reference."),
    "Mother Line Number": ("Mother Roster Line Number", "kinship", "Released within-household line number for the member's mother; no-parent-in-household and missing codes are null."),
    "Mother Person ID": ("Mother Person Identifier", "kinship", "Person identifier derived only when Mother Line Number matches another member of the same survey-wave household and is not a self-reference."),
    "Marital Status Source Code": ("Marital Status Source Code", "demographics", "Released wave-specific marital-status code retained for provenance."),
    "Marital Status Harmonized": ("Marital Status Harmonized", "demographics", "Cross-wave status: 1=Never married/never cohabited, 2=Married/cohabiting, 3=Widowed, 4=Divorced/separated."),
    "Spouse Line Number": ("Spouse Roster Line Number", "kinship", "Released within-household line number for the member's spouse; spouse-not-in-household and missing codes are null."),
    "Spouse Person ID": ("Spouse Person Identifier", "kinship", "Person identifier derived only when Spouse Line Number matches another member of the same survey-wave household and is not a self-reference."),
    "Ethnicity Source Code": ("Ethnicity Source Code", "demographics", "Released detailed ethnicity code retained for provenance; the 2004 missing sentinel is null."),
    "Ethnicity Harmonized": ("Ethnicity Harmonized", "demographics", "Cross-wave ethnicity: 1=Khmer, 2=Cham, 3=Other local/indigenous group, 4=Chinese, 5=Vietnamese, 6=Thai, 7=Lao, 8=Other."),
    "Absent From Household": ("Recent Absence from Household Indicator", "demographics", "Harmonized indicator: 1=absent and 0=present; reference period is current status in 2004 and the past 7 days in later waves."),
    "Presence Reference Period": ("Household Presence Reference Period", "provenance", "Reference period for the harmonized household-absence indicator."),
    "Province Code": ("Province Code", "geography", "Source province code normalized as a two-character string."),
    "District Code": ("District Code", "geography", "Source district code normalized as a four-character string."),
    "Commune Code": ("Commune Code", "geography", "Source commune code normalized as a six-character string."),
    "Village Code": ("Village Code", "geography", "Source village code normalized as an eight-character string."),
    "Urban Rural": ("Source Urban Rural Classification", "categorical", "Released urban/rural classification code; labels are not recoded across waves."),
    "Survey Month": ("Survey Interview Month", "time", "Released survey month coerced to integer when parseable."),
    "Stratum": ("Sampling Stratum Code", "identifier", "Released sampling stratum code retained as a string."),
    "Household Weight": ("Released Household Sampling Weight", "weight", "Released wave-specific household sampling weight; no imputation."),
    "Person Weight": ("Released Person Sampling Weight", "weight", "Released wave-specific person sampling weight; no imputation."),
    "Source Archive": ("Source Archive Path", "provenance", "Raw archive path relative to the project root."),
    "Source Submodule": ("Source Stata Submodule", "provenance", "Stata member path inside the raw archive."),
    "Source Row ID": ("Stable Source Row Identifier", "provenance", "Deterministic identifier for the row in the member-roster source."),
}


def token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def source_leaf(source: DataSource) -> str:
    target = source.archive_members[-1] if source.archive_members else source.root_file.name
    return PurePosixPath(target).name


def read_frame(source: DataSource) -> pd.DataFrame:
    input_obj: Path | io.BytesIO = (
        io.BytesIO(source.read_bytes()) if source.archive_members else source.root_file
    )
    return pd.read_stata(input_obj, convert_categoricals=False)


def find_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    columns = {token(column): str(column) for column in frame.columns}
    for alias in aliases:
        if token(alias) in columns:
            return columns[token(alias)]
    return None


def clean_string(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    return result.mask(result.str.lower().isin(["", "nan", "none", "<na>"]))


def clean_code(series: pd.Series, width: int | None = None) -> tuple[pd.Series, int]:
    result = clean_string(series)
    digits = result.str.fullmatch(r"\d+", na=False)
    if width is not None:
        result = result.where(~digits, result.str.zfill(width))
        # In 2004 the published identifiers omit a leading zero for one-digit
        # province codes, so 4/6/8 are minimum rather than exact lengths.
        valid = result.str.fullmatch(rf"\d{{{width},}}", na=False)
    else:
        valid = digits
    invalid = int((result.notna() & ~valid).sum())
    return result.where(valid), invalid


def coerce_month(series: pd.Series) -> tuple[pd.Series, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.isna() | ((numeric >= 1) & (numeric <= 12) & (numeric % 1 == 0))
    invalid = int((numeric.notna() & ~valid).sum() + (series.notna() & numeric.isna()).sum())
    return numeric.where(valid).astype("Int16"), invalid


def snake_case(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


@dataclass
class IssueCollector:
    rows: list[dict[str, object]] = field(default_factory=list)

    def add(self, grain: str, wave: str, issue_type: str, variable: str, affected_rows: int, detail: str) -> None:
        if affected_rows:
            self.rows.append({
                "grain": grain,
                "survey_wave": wave,
                "issue_type": issue_type,
                "variable": variable,
                "affected_rows": int(affected_rows),
                "detail": detail,
            })

    def frame(self) -> pd.DataFrame:
        columns = ["grain", "survey_wave", "issue_type", "variable", "affected_rows", "detail"]
        return pd.DataFrame(self.rows, columns=columns).sort_values(
            ["grain", "survey_wave", "issue_type", "variable"], ignore_index=True
        ) if self.rows else pd.DataFrame(columns=columns)


@dataclass
class AlignmentContext:
    root: Path
    issues: IssueCollector = field(default_factory=IssueCollector)
    mappings: set[tuple[str, str, str, str]] = field(default_factory=set)
    sources_by_module: dict[str, list[tuple[str, DataSource]]] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def record(self, variable: str, wave: str, raw_name: str, source_kind: str = "explicit") -> None:
        self.mappings.add((variable, wave, raw_name, source_kind))

    def load(self, source: DataSource) -> pd.DataFrame:
        key = source.display_name(self.root)
        if key not in self.frames:
            self.frames[key] = read_frame(source)
        return self.frames[key]


def initialize_context(root: Path) -> AlignmentContext:
    context = AlignmentContext(root=root)
    grouped: dict[str, list[tuple[str, DataSource]]] = {
        "household_members": [], "household_core": [], "household_weights": [], "person_weights": []
    }
    for source in discover_sources(root):
        wave = normalize_wave(source.display_name(root))
        if wave not in WAVES:
            continue
        for module in modules_for_source(source, root):
            if module in grouped:
                grouped[module].append((wave, source))
    for module in grouped:
        grouped[module].sort(key=lambda item: (WAVES.index(item[0]), item[1].display_name(root)))
    context.sources_by_module = grouped
    return context


def standardize_source(context: AlignmentContext, source: DataSource, wave: str, grain: str) -> pd.DataFrame:
    frame = context.load(source)
    output = pd.DataFrame(index=frame.index)
    output["Dataset Name"] = f"CSES {wave}"
    output["Survey Wave"] = wave
    output["Survey Year"] = WAVE_YEAR[wave]
    output["Source Archive"] = str(source.root_file.relative_to(context.root))
    output["Source Submodule"] = "::".join(source.archive_members) if source.archive_members else source.root_file.name
    output["Source Row ID"] = [f"{wave}:{source_leaf(source)}:{row + 1}" for row in range(len(frame))]

    for variable in ALIGNMENT_FIELDS:
        column = find_column(frame, FIELD_ALIASES[variable])
        if column is None:
            output[variable] = pd.Series(pd.NA, index=frame.index, dtype="string")
            continue
        context.record(variable, wave, column)
        if variable in ID_WIDTHS[wave]:
            output[variable], invalid = clean_code(frame[column], ID_WIDTHS[wave][variable])
        elif variable in GEO_WIDTHS:
            output[variable], invalid = clean_code(frame[column], GEO_WIDTHS[variable])
        elif variable == "Survey Month":
            output[variable], invalid = coerce_month(frame[column])
        else:
            output[variable] = clean_string(frame[column])
            invalid = 0
        context.issues.add(grain, wave, "invalid_value_set_null", variable, invalid, f"Invalid values in {column} were set to null.")

    if output["Household ID"].isna().any() and output["Person ID"].notna().any():
        derived = output["Person ID"].str.slice(stop=-2)
        mask = output["Household ID"].isna() & derived.notna()
        output.loc[mask, "Household ID"] = derived[mask]
        if mask.any():
            context.record("Household ID", wave, "persid", "derived")

    # Household identifiers end in the two-digit household sequence number.
    # Taking everything before that suffix works for both one- and two-digit
    # province encodings in the 2004 release.
    derived_psu = output["Household ID"].str.slice(stop=-2)
    mask = output["PSU"].isna() & derived_psu.notna()
    output.loc[mask, "PSU"] = derived_psu[mask]
    if mask.any():
        context.record("PSU", wave, "hhid", "derived")
    return output.reset_index(drop=True)


def roster_frames(context: AlignmentContext) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    counts = {wave: 0 for wave in WAVES}
    for wave, source in context.sources_by_module["household_members"]:
        standardized = standardize_source(context, source, wave, "HL")
        frames.append(add_demographics(context, source, wave, standardized))
        counts[wave] += 1
    bad = {wave: count for wave, count in counts.items() if count != 1}
    if bad:
        raise RuntimeError(f"Expected exactly one household-member source per wave; found {bad}")
    return frames


def add_demographics(
    context: AlignmentContext,
    source: DataSource,
    wave: str,
    standardized: pd.DataFrame,
) -> pd.DataFrame:
    raw = context.load(source)
    output = standardized.copy()

    sex_column = find_column(raw, DEMOGRAPHIC_ALIASES["Sex"])
    age_column = find_column(raw, DEMOGRAPHIC_ALIASES["Age"])
    relationship_column = find_column(raw, DEMOGRAPHIC_ALIASES["Relationship to Household Head"])
    presence_column = find_column(raw, DEMOGRAPHIC_ALIASES["Absent From Household"])
    required = {
        "Sex": sex_column,
        "Age": age_column,
        "Relationship to Household Head": relationship_column,
        "Absent From Household": presence_column,
    }
    missing_columns = [name for name, column in required.items() if column is None]
    if missing_columns:
        raise RuntimeError(f"Missing demographic columns in {wave}: {missing_columns}")

    sex = pd.to_numeric(raw[sex_column], errors="coerce")
    valid_sex = sex.isin([1, 2])
    context.issues.add("HL", wave, "invalid_value_set_null", "Sex", int((sex.notna() & ~valid_sex).sum()), f"Codes outside 1/2 in {sex_column} were set to null.")
    output["Sex"] = sex.where(valid_sex).astype("Int8")
    context.record("Sex", wave, sex_column)

    age = pd.to_numeric(raw[age_column], errors="coerce")
    valid_age = age.between(0, 120) & age.mod(1).eq(0)
    if wave == "2004":
        valid_age &= ~age.isin([98, 99])
    context.issues.add("HL", wave, "invalid_value_set_null", "Age", int((age.notna() & ~valid_age).sum()), f"Values outside integer ages 0-120, including documented 2004 missing sentinels 98/99, in {age_column} were set to null.")
    output["Age"] = age.where(valid_age).astype("Int16")
    context.record("Age", wave, age_column)

    extended_columns = {
        variable: find_column(raw, aliases)
        for variable, aliases in EXTENDED_HL_ALIASES.items()
    }
    missing_extended = [name for name, column in extended_columns.items() if column is None]
    if missing_extended:
        raise RuntimeError(f"Missing extended HL columns in {wave}: {missing_extended}")

    member_column = extended_columns["Member Line Number"]
    member_line = pd.to_numeric(raw[member_column], errors="coerce")
    valid_member_line = member_line.between(1, 98) & member_line.mod(1).eq(0)
    context.issues.add("HL", wave, "invalid_value_set_null", "Member Line Number", int((member_line.notna() & ~valid_member_line).sum()), f"Values outside integer roster lines 1-98 in {member_column} were set to null.")
    output["Member Line Number"] = member_line.where(valid_member_line).astype("Int16")
    context.record("Member Line Number", wave, member_column)
    person_suffix = pd.to_numeric(output["Person ID"].str[-2:], errors="coerce").astype("Int16")
    line_mismatch = output["Member Line Number"].notna() & person_suffix.notna() & output["Member Line Number"].ne(person_suffix)
    context.issues.add("HL", wave, "member_line_person_id_mismatch", "Member Line Number", int(line_mismatch.sum()), "Released member line differs from the two-digit person-identifier suffix; both values were retained.")

    birth_rules = {
        "Birth Day": (1, 31, "Int8"),
        "Birth Month": (1, 12, "Int8"),
        "Birth Year": (1800, WAVE_YEAR[wave] + 1, "Int16"),
    }
    for variable, (lower, upper, dtype) in birth_rules.items():
        column = extended_columns[variable]
        numeric = pd.to_numeric(raw[column], errors="coerce")
        valid = numeric.between(lower, upper) & numeric.mod(1).eq(0)
        context.issues.add("HL", wave, "invalid_value_set_null", variable, int((numeric.notna() & ~valid).sum()), f"Values outside integer range {lower}-{upper} in {column} were set to null.")
        output[variable] = numeric.where(valid).astype(dtype)
        context.record(variable, wave, column)

    for variable in ["Father Line Number", "Mother Line Number", "Spouse Line Number"]:
        column = extended_columns[variable]
        numeric = pd.to_numeric(raw[column], errors="coerce")
        valid = numeric.between(1, 98) & numeric.mod(1).eq(0)
        invalid = numeric.notna() & ~numeric.isin([0, 99]) & ~valid
        context.issues.add("HL", wave, "invalid_value_set_null", variable, int(invalid.sum()), f"Values other than no-link code 0, missing code 99, or integer roster lines 1-98 in {column} were set to null.")
        output[variable] = numeric.where(valid).astype("Int16")
        context.record(variable, wave, column)
        person_variable = variable.replace("Line Number", "Person ID")
        context.record(person_variable, wave, column, "derived")

    marital_column = extended_columns["Marital Status Source Code"]
    marital = pd.to_numeric(raw[marital_column], errors="coerce")
    valid_marital_codes = range(1, 7) if wave == "2004" else range(1, 5)
    valid_marital = marital.isin(valid_marital_codes)
    context.issues.add("HL", wave, "unresolved_code_set_null", "Marital Status Source Code", int((marital.notna() & ~valid_marital).sum()), f"Codes outside the documented wave-specific set in {marital_column} were set to null.")
    output["Marital Status Source Code"] = marital.where(valid_marital).astype("Int8")
    marital_mapping = (
        {1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4}
        if wave == "2004"
        else {1: 2, 2: 4, 3: 3, 4: 1}
    )
    output["Marital Status Harmonized"] = marital.map(marital_mapping).astype("Int8")
    context.record("Marital Status Source Code", wave, marital_column)
    context.record("Marital Status Harmonized", wave, marital_column, "derived")

    ethnicity_column = extended_columns["Ethnicity Source Code"]
    ethnicity = pd.to_numeric(raw[ethnicity_column], errors="coerce")
    valid_ethnicity = ethnicity.isin(range(1, 9))
    context.issues.add("HL", wave, "unresolved_code_set_null", "Ethnicity Source Code", int((ethnicity.notna() & ~valid_ethnicity).sum()), f"Codes outside 1-8 in {ethnicity_column} were set to null.")
    output["Ethnicity Source Code"] = ethnicity.where(valid_ethnicity).astype("Int8")
    output["Ethnicity Harmonized"] = ethnicity.where(valid_ethnicity).astype("Int8")
    context.record("Ethnicity Source Code", wave, ethnicity_column)
    context.record("Ethnicity Harmonized", wave, ethnicity_column, "derived")

    relationship = pd.to_numeric(raw[relationship_column], errors="coerce")
    valid_relationship = relationship.isin(range(1, 16))
    context.issues.add("HL", wave, "invalid_value_set_null", "Relationship to Household Head", int((relationship.notna() & ~valid_relationship).sum()), f"Codes outside 1-15 in {relationship_column} were set to null.")
    output["Relationship to Household Head"] = relationship.where(valid_relationship).astype("Int8")
    context.record("Relationship to Household Head", wave, relationship_column)

    presence = pd.to_numeric(raw[presence_column], errors="coerce")
    mapping = {1: 1, 2: 0} if wave == "2004" else {1: 0, 2: 1}
    output["Absent From Household"] = presence.map(mapping).astype("Int8")
    unresolved = int((presence.notna() & ~presence.isin(mapping)).sum())
    context.issues.add("HL", wave, "unresolved_presence_code_set_null", "Absent From Household", unresolved, f"Unresolved codes in {presence_column} were set to null.")
    output["Presence Reference Period"] = "Current status" if wave == "2004" else "Past 7 days"
    context.record("Absent From Household", wave, presence_column, "derived")
    return output


def add_kinship_person_ids(context: AlignmentContext, hl: pd.DataFrame) -> pd.DataFrame:
    """Resolve released within-household kin lines to valid, non-self person keys."""

    result = hl.copy()
    lookup = result[["Survey Wave", "Household ID", "Member Line Number", "Person ID"]].dropna(
        subset=["Household ID", "Member Line Number", "Person ID"]
    )
    duplicate_lookup = lookup.duplicated(
        ["Survey Wave", "Household ID", "Member Line Number"], keep=False
    )
    if duplicate_lookup.any():
        raise RuntimeError(
            "Cannot resolve kinship links because wave-household-member-line is not unique: "
            f"rows={int(duplicate_lookup.sum())}"
        )

    for role in ["Father", "Mother", "Spouse"]:
        line_variable = f"{role} Line Number"
        person_variable = f"{role} Person ID"
        target = lookup.rename(columns={
            "Member Line Number": line_variable,
            "Person ID": f"_{role.lower()}_target_person_id",
        })
        result = result.merge(
            target,
            on=["Survey Wave", "Household ID", line_variable],
            how="left",
            validate="m:1",
        )
        target_column = f"_{role.lower()}_target_person_id"
        supplied = result[line_variable].notna()
        missing_target = supplied & result[target_column].isna()
        self_reference = supplied & result[target_column].eq(result["Person ID"])
        for wave, count in result.loc[missing_target].groupby("Survey Wave").size().items():
            context.issues.add(
                "HL", str(wave), "kinship_line_not_in_household", person_variable,
                int(count), f"Released {line_variable} has no matching member in the same household; the source line was retained and {person_variable} was set to null.",
            )
        for wave, count in result.loc[self_reference].groupby("Survey Wave").size().items():
            context.issues.add(
                "HL", str(wave), "kinship_self_reference", person_variable,
                int(count), f"Released {line_variable} points to the same person; the source line was retained and {person_variable} was set to null.",
            )
        result[person_variable] = result[target_column].mask(self_reference).astype("string")
        result = result.drop(columns=target_column)
    return result


def raw_weight_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    return find_column(frame, names)


def collect_weight_candidates(
    context: AlignmentContext,
    rosters: list[pd.DataFrame],
    variable: str,
) -> pd.DataFrame:
    names_by_wave = HOUSEHOLD_WEIGHT_NAMES if variable == "Household Weight" else PERSON_WEIGHT_NAMES
    module = "household_weights" if variable == "Household Weight" else "person_weights"
    keys = ["Survey Wave", "Household ID"] if variable == "Household Weight" else ["Survey Wave", "Person ID"]
    pieces: list[pd.DataFrame] = []

    # Embedded weights are the authoritative released fields for 2019 and a fallback for 2021.
    for standardized, (wave, source) in zip(rosters, context.sources_by_module["household_members"], strict=True):
        raw = context.load(source)
        column = raw_weight_column(raw, names_by_wave.get(wave, []))
        if column is None:
            continue
        piece = standardized[keys].copy()
        piece[variable] = pd.to_numeric(raw[column], errors="coerce").astype("Float64")
        piece["_priority"] = 1
        pieces.append(piece)
        context.record(variable, wave, column)

    for wave, source in context.sources_by_module[module]:
        raw = context.load(source)
        column = raw_weight_column(raw, names_by_wave.get(wave, []))
        if column is None:
            continue
        standardized = standardize_source(context, source, wave, "weight")
        piece = standardized[keys].copy()
        piece[variable] = pd.to_numeric(raw[column], errors="coerce").astype("Float64")
        piece["_priority"] = 0
        pieces.append(piece)
        context.record(variable, wave, column)

    if not pieces:
        return pd.DataFrame(columns=keys + [variable])
    candidates = pd.concat(pieces, ignore_index=True)
    candidates = candidates.dropna(subset=[keys[-1], variable]).sort_values("_priority")
    rows: list[dict[str, object]] = []
    for group_key, group in candidates.groupby(keys, sort=False, dropna=False):
        values = group[variable].dropna().astype(float).to_numpy()
        if len(values) == 0:
            continue
        first = values[0]
        conflict = bool((~np.isclose(values, first, rtol=1e-7, atol=1e-10)).any())
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(keys, key_tuple, strict=True))
        row[variable] = pd.NA if conflict else first
        rows.append(row)
        if conflict:
            context.issues.add("weight", str(row["Survey Wave"]), "conflicting_sources_set_null", variable, 1, f"Conflicting released weights for {keys[-1]}={row[keys[-1]]}.")
    result = pd.DataFrame(rows, columns=keys + [variable])
    if not result.empty:
        result[variable] = pd.to_numeric(result[variable], errors="coerce").astype("Float64")
    return result


def collapse_context(
    context: AlignmentContext,
    candidates: pd.DataFrame,
    keys: list[str],
    suffix: str,
) -> pd.DataFrame:
    candidates = candidates.dropna(subset=[keys[-1]])
    rows: list[dict[str, object]] = []
    for group_key, group in candidates.groupby(keys, sort=False, dropna=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, object] = dict(zip(keys, key_tuple, strict=True))
        for variable in HH_CONTEXT_FIELDS:
            values = group[variable].dropna().astype("string").unique().tolist()
            if len(values) == 1:
                row[f"{variable}{suffix}"] = values[0]
            elif len(values) > 1:
                row[f"{variable}{suffix}"] = pd.NA
                context.issues.add("HH", str(row["Survey Wave"]), "conflicting_sources_set_null", variable, 1, f"Conflicting values for {keys[-1]}={row[keys[-1]]}.")
            else:
                row[f"{variable}{suffix}"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def add_household_derivations(
    context: AlignmentContext,
    hh: pd.DataFrame,
    roster: pd.DataFrame,
    include_head_education: bool,
) -> pd.DataFrame:
    """Aggregate household composition and unique-head attributes from HL and ED."""

    keys = ["Survey Wave", "Household ID"]
    valid_roster = roster.dropna(subset=["Household ID"]).copy()
    valid_roster["_male"] = valid_roster["Sex"].eq(1)
    valid_roster["_female"] = valid_roster["Sex"].eq(2)
    valid_roster["_child"] = valid_roster["Age"].between(0, 14)
    valid_roster["_working_age"] = valid_roster["Age"].between(15, 64)
    valid_roster["_older"] = valid_roster["Age"].ge(65)
    valid_roster["_unknown_age"] = valid_roster["Age"].isna()
    composition = valid_roster.groupby(keys, observed=True, sort=False).agg(
        **{
            "Household Member Count": ("Person ID", "size"),
            "Male Member Count": ("_male", "sum"),
            "Female Member Count": ("_female", "sum"),
            "Child Member Count 0-14": ("_child", "sum"),
            "Working Age Member Count 15-64": ("_working_age", "sum"),
            "Older Member Count 65 Plus": ("_older", "sum"),
            "Unknown Age Member Count": ("_unknown_age", "sum"),
        }
    ).reset_index()
    hh = hh.merge(composition, on=keys, how="left", validate="1:1")

    head_source = valid_roster.loc[
        valid_roster["Relationship to Household Head"].eq(1),
        keys + ["Person ID", "Sex", "Age", "Marital Status Harmonized", "Ethnicity Harmonized"],
    ].copy()
    head_counts = head_source.groupby(keys, observed=True).size()
    unique_head_keys = head_counts.loc[head_counts.eq(1)].index
    head_source = head_source.set_index(keys).loc[lambda frame: frame.index.isin(unique_head_keys)].reset_index()
    head_source = head_source.rename(columns={
        "Person ID": "Household Head Person ID",
        "Sex": "Household Head Sex",
        "Age": "Household Head Age",
        "Marital Status Harmonized": "Household Head Marital Status",
        "Ethnicity Harmonized": "Household Head Ethnicity",
    })
    hh = hh.merge(head_source, on=keys, how="left", validate="1:1")

    hh_key_index = pd.MultiIndex.from_frame(hh[keys])
    for wave in WAVES:
        wave_hh = hh.loc[hh["Survey Wave"].eq(wave), keys]
        wave_index = pd.MultiIndex.from_frame(wave_hh)
        wave_head_counts = head_counts.reindex(wave_index, fill_value=0)
        context.issues.add(
            "HH", wave, "household_head_count_not_one", "Household Head Person ID",
            int(wave_head_counts.ne(1).sum()),
            "Household retained; derived head attributes are null unless exactly one member is coded as head.",
        )
        for variable in HH_DERIVED_COLUMNS[:12]:
            context.record(variable, wave, "", "derived")

    education_variables = {
        "Education Level Harmonized": "Household Head Education Level",
        "Years Attended School": "Household Head Years Attended School",
        "Can Read": "Household Head Can Read",
        "Can Write": "Household Head Can Write",
    }
    if include_head_education:
        education_path = context.root / "data/exp/database/final_ED_CSES.parquet"
        if not education_path.exists():
            raise RuntimeError(
                "Household-head education requires data/exp/database/final_ED_CSES.parquet; "
                "run build_cses_hl.py and build_cses_ed.py before build_cses_hh.py."
            )
        education = pd.read_parquet(
            education_path,
            columns=["Survey Wave", "Person ID", *education_variables],
        )
        if education.duplicated(["Survey Wave", "Person ID"]).any():
            raise RuntimeError("Education wave-person key is not unique during household-head enrichment.")
        education = education.rename(columns={
            "Person ID": "Household Head Person ID",
            **education_variables,
        })
        hh = hh.merge(
            education,
            on=["Survey Wave", "Household Head Person ID"],
            how="left",
            validate="m:1",
            indicator="_head_ed_merge",
        )
        for wave, count in hh.loc[
            hh["Household Head Person ID"].notna() & hh["_head_ed_merge"].eq("left_only")
        ].groupby("Survey Wave").size().items():
            context.issues.add(
                "HH", str(wave), "household_head_not_in_education", "Household Head Education Level",
                int(count), "The coded household head has no released education record; education fields remain null.",
            )
        hh = hh.drop(columns="_head_ed_merge")
        for wave in WAVES:
            for variable in education_variables.values():
                context.record(variable, wave, "", "derived")
    else:
        for variable in education_variables.values():
            hh[variable] = pd.NA

    if len(hh_key_index) != len(hh):
        raise RuntimeError("Household derivation changed the HH row count.")
    return hh


def build_hh(
    context: AlignmentContext,
    rosters: list[pd.DataFrame],
    include_head_education: bool = False,
) -> pd.DataFrame:
    roster = pd.concat(rosters, ignore_index=True)
    candidates = [roster[["Survey Wave", "PSU", "Household ID"] + HH_CONTEXT_FIELDS]]
    for module in ("household_core", "household_weights", "person_weights"):
        for wave, source in context.sources_by_module[module]:
            standardized = standardize_source(context, source, wave, "HH context")
            candidates.append(standardized[["Survey Wave", "PSU", "Household ID"] + HH_CONTEXT_FIELDS])
    all_candidates = pd.concat(candidates, ignore_index=True)
    by_hh = collapse_context(context, all_candidates, ["Survey Wave", "Household ID"], "_hh")
    by_psu = collapse_context(context, all_candidates, ["Survey Wave", "PSU"], "_psu")

    roster = roster.copy()
    roster["_household_group"] = roster["Household ID"].fillna(roster["Source Row ID"])
    hh = roster.drop_duplicates(["Survey Wave", "_household_group"], keep="first").copy()
    hh = hh.drop(columns=HH_CONTEXT_FIELDS)
    hh = hh.merge(by_hh, on=["Survey Wave", "Household ID"], how="left", validate="m:1")
    hh = hh.merge(by_psu, on=["Survey Wave", "PSU"], how="left", validate="m:1")
    for variable in HH_CONTEXT_FIELDS:
        direct = hh.pop(f"{variable}_hh")
        psu = hh.pop(f"{variable}_psu")
        conflict = direct.notna() & psu.notna() & direct.astype("string").ne(psu.astype("string"))
        for wave, count in hh.loc[conflict].groupby("Survey Wave").size().items():
            context.issues.add("HH", str(wave), "hh_psu_conflict_set_null", variable, int(count), "Household- and PSU-level values disagree.")
        hh[variable] = direct.combine_first(psu).mask(conflict)

    weights = collect_weight_candidates(context, rosters, "Household Weight")
    hh = hh.merge(weights, on=["Survey Wave", "Household ID"], how="left", validate="m:1")
    hh = add_household_derivations(context, hh, roster, include_head_education)
    hh = hh[HH_COLUMNS].sort_values(["Survey Year", "Household ID", "Source Row ID"], na_position="last").reset_index(drop=True)
    enforce_dtypes(hh)
    return hh


def build_hl(context: AlignmentContext, rosters: list[pd.DataFrame], hh: pd.DataFrame) -> pd.DataFrame:
    hl = pd.concat(rosters, ignore_index=True)
    hl = add_kinship_person_ids(context, hl)
    inherited = hh[["Survey Wave", "Household ID", "PSU"] + HH_CONTEXT_FIELDS + ["Household Weight"]].copy()
    inherited = inherited.dropna(subset=["Household ID"]).drop_duplicates(["Survey Wave", "Household ID"])
    hl = hl.drop(columns=["PSU"] + HH_CONTEXT_FIELDS)
    hl = hl.merge(inherited, on=["Survey Wave", "Household ID"], how="left", validate="m:1")
    person_weights = collect_weight_candidates(context, rosters, "Person Weight")
    hl = hl.merge(person_weights, on=["Survey Wave", "Person ID"], how="left", validate="m:1")
    hl = hl[HL_COLUMNS].sort_values(["Survey Year", "Household ID", "Person ID", "Source Row ID"], na_position="last").reset_index(drop=True)
    enforce_dtypes(hl)
    return hl


def enforce_dtypes(frame: pd.DataFrame) -> None:
    numeric_columns = {
        "Survey Year", "Survey Month", "Household Weight", "Person Weight", "Sex", "Age",
        "Household Member Count", "Male Member Count", "Female Member Count",
        "Child Member Count 0-14", "Working Age Member Count 15-64",
        "Older Member Count 65 Plus", "Unknown Age Member Count",
        "Household Head Sex", "Household Head Age", "Household Head Marital Status",
        "Household Head Ethnicity", "Household Head Education Level",
        "Household Head Years Attended School", "Household Head Can Read",
        "Household Head Can Write",
        "Member Line Number", "Birth Day", "Birth Month", "Birth Year",
        "Relationship to Household Head", "Father Line Number", "Mother Line Number",
        "Marital Status Source Code", "Marital Status Harmonized", "Spouse Line Number",
        "Ethnicity Source Code", "Ethnicity Harmonized", "Absent From Household",
    }
    string_columns = [column for column in frame.columns if column not in numeric_columns]
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    frame["Survey Year"] = pd.to_numeric(frame["Survey Year"], errors="coerce").astype("Int16")
    frame["Survey Month"] = pd.to_numeric(frame["Survey Month"], errors="coerce").astype("Int16")
    for column, dtype in (
        ("Household Member Count", "Int16"), ("Male Member Count", "Int16"),
        ("Female Member Count", "Int16"), ("Child Member Count 0-14", "Int16"),
        ("Working Age Member Count 15-64", "Int16"),
        ("Older Member Count 65 Plus", "Int16"), ("Unknown Age Member Count", "Int16"),
        ("Household Head Sex", "Int8"), ("Household Head Age", "Int16"),
        ("Household Head Marital Status", "Int8"), ("Household Head Ethnicity", "Int8"),
        ("Household Head Education Level", "Int8"),
        ("Household Head Years Attended School", "Int16"),
        ("Household Head Can Read", "Int8"), ("Household Head Can Write", "Int8"),
        ("Member Line Number", "Int16"), ("Sex", "Int8"),
        ("Birth Day", "Int8"), ("Birth Month", "Int8"),
        ("Birth Year", "Int16"), ("Age", "Int16"),
        ("Relationship to Household Head", "Int8"),
        ("Father Line Number", "Int16"), ("Mother Line Number", "Int16"),
        ("Marital Status Source Code", "Int8"), ("Marital Status Harmonized", "Int8"),
        ("Spouse Line Number", "Int16"), ("Ethnicity Source Code", "Int8"),
        ("Ethnicity Harmonized", "Int8"),
        ("Absent From Household", "Int8"),
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(dtype)
    for column in ("Household Weight", "Person Weight"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")


def make_audit(frame: pd.DataFrame, grain: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    key = ["Survey Wave", "Household ID"] if grain == "HH" else ["Survey Wave", "Person ID"]
    for wave in WAVES:
        part = frame.loc[frame["Survey Wave"].eq(wave)]
        row: dict[str, object] = {
            "grain": grain,
            "survey_wave": wave,
            "rows": len(part),
            "unique_households": part["Household ID"].nunique(dropna=True),
            "missing_psu": int(part["PSU"].isna().sum()),
            "missing_household_id": int(part["Household ID"].isna().sum()),
            "missing_household_weight": int(part["Household Weight"].isna().sum()),
            "duplicate_key_rows": int(part.duplicated(key, keep=False).sum()),
        }
        if grain == "HH":
            row["minimum_household_member_count"] = part["Household Member Count"].min()
            row["maximum_household_member_count"] = part["Household Member Count"].max()
            row["households_missing_unique_head"] = int(part["Household Head Person ID"].isna().sum())
            row["head_education_level_available"] = int(part["Household Head Education Level"].notna().sum())
            row["head_years_attended_available"] = int(part["Household Head Years Attended School"].notna().sum())
            row["head_can_read_available"] = int(part["Household Head Can Read"].notna().sum())
            row["head_can_write_available"] = int(part["Household Head Can Write"].notna().sum())
        if grain == "HL":
            row["unique_persons"] = part["Person ID"].nunique(dropna=True)
            row["missing_person_id"] = int(part["Person ID"].isna().sum())
            row["missing_person_weight"] = int(part["Person Weight"].isna().sum())
            row["missing_sex"] = int(part["Sex"].isna().sum())
            row["missing_age"] = int(part["Age"].isna().sum())
            row["minimum_age"] = part["Age"].min()
            row["maximum_age"] = part["Age"].max()
            row["missing_member_line_number"] = int(part["Member Line Number"].isna().sum())
            row["birth_day_available"] = int(part["Birth Day"].notna().sum())
            row["birth_month_available"] = int(part["Birth Month"].notna().sum())
            row["birth_year_available"] = int(part["Birth Year"].notna().sum())
            row["missing_relationship"] = int(part["Relationship to Household Head"].isna().sum())
            row["father_lines_available"] = int(part["Father Line Number"].notna().sum())
            row["father_links_resolved"] = int(part["Father Person ID"].notna().sum())
            row["mother_lines_available"] = int(part["Mother Line Number"].notna().sum())
            row["mother_links_resolved"] = int(part["Mother Person ID"].notna().sum())
            row["marital_status_available"] = int(part["Marital Status Harmonized"].notna().sum())
            row["spouse_lines_available"] = int(part["Spouse Line Number"].notna().sum())
            row["spouse_links_resolved"] = int(part["Spouse Person ID"].notna().sum())
            row["ethnicity_available"] = int(part["Ethnicity Harmonized"].notna().sum())
            row["missing_absence_status"] = int(part["Absent From Household"].isna().sum())
            row["absent_members"] = int(part["Absent From Household"].eq(1).sum())
            heads = part.assign(_head=part["Relationship to Household Head"].eq(1)).groupby("Household ID")["_head"].sum()
            row["households_without_one_head"] = int(heads.ne(1).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def add_summary_issues(context: AlignmentContext, frame: pd.DataFrame, grain: str) -> None:
    for wave in WAVES:
        part = frame.loc[frame["Survey Wave"].eq(wave)]
        for variable in ["PSU", "Household ID"] + (["Person ID"] if grain == "HL" else []):
            context.issues.add(grain, wave, "missing_alignment_key", variable, int(part[variable].isna().sum()), "Row retained with a null alignment key.")
        for variable in ["Household Weight"] + (["Person Weight"] if grain == "HL" else []):
            context.issues.add(grain, wave, "missing_released_weight", variable, int(part[variable].isna().sum()), "No released/matched weight was imputed.")
        key = ["Survey Wave", "Household ID"] if grain == "HH" else ["Survey Wave", "Person ID"]
        context.issues.add(grain, wave, "duplicate_alignment_key", " + ".join(key), int(part.duplicated(key, keep=False).sum()), "Duplicate source rows were retained for audit.")
        if grain == "HL":
            for variable in ["Sex", "Age", "Relationship to Household Head", "Absent From Household"]:
                context.issues.add(grain, wave, "missing_demographic_value", variable, int(part[variable].isna().sum()), "Source row retained without imputation.")
            heads = part.assign(_head=part["Relationship to Household Head"].eq(1)).groupby("Household ID")["_head"].sum()
            context.issues.add(grain, wave, "household_head_count_not_one", "Relationship to Household Head", int(heads.ne(1).sum()), "Household retained; head-count inconsistency was not repaired.")


def write_metadata(context: AlignmentContext, output_dir: Path, grain: str, columns: list[str]) -> None:
    for wave in WAVES:
        for variable in ["Dataset Name", "Survey Wave", "Survey Year", "Source Archive", "Source Submodule", "Source Row ID"] + (["Presence Reference Period"] if grain == "HL" else []):
            context.record(variable, wave, "", "derived")
    mappings = []
    for variable, wave, raw_name, source_kind in sorted(context.mappings, key=lambda row: (columns.index(row[0]) if row[0] in columns else 999, WAVES.index(row[1]), row[2], row[3])):
        if variable not in columns:
            continue
        full_name, measure_type, canonical_text = VARIABLE_INFO[variable]
        mappings.append({
            "canonical_varname": snake_case(variable),
            "dataset_name": f"CSES {wave}",
            "column_in_raw_sav": raw_name,
            "column_label_in_english": full_name,
            "source_kind": source_kind,
            "measure_type": measure_type,
            "canonical_text": canonical_text,
        })
    ind_que = pd.DataFrame(mappings).drop_duplicates(ignore_index=True)
    ind_que.to_csv(output_dir / f"ind_que_{grain}_CSES.csv", index=False)

    make_alignment_summary(ind_que, columns).to_csv(
        output_dir / f"align_summary_{grain}_CSES.csv", index=False
    )


def make_alignment_summary(ind_que: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Match the existing mda summary semantics: source counts are mapping rows."""

    summaries = []
    for variable in columns:
        canonical = snake_case(variable)
        selected = ind_que.loc[ind_que["canonical_varname"].eq(canonical)]
        full_name, measure_type, canonical_text = VARIABLE_INFO[variable]
        summaries.append({
            "varname": canonical,
            "dataset_count": int(selected["dataset_name"].nunique()),
            "source_count": int(len(selected)),
            "explicit_count": int(selected["source_kind"].eq("explicit").sum()),
            "derived_count": int(selected["source_kind"].eq("derived").sum()),
            "measure_type": measure_type,
            "canonical_text": canonical_text,
        })
    return pd.DataFrame(summaries)


def run(root: Path, grain: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = initialize_context(root)
    rosters = roster_frames(context)
    hh = build_hh(context, rosters, include_head_education=grain == "HH")
    frame = hh if grain == "HH" else build_hl(context, rosters, hh)
    add_summary_issues(context, frame, grain)

    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / f"final_{grain}_CSES.parquet", index=False)
    audit = make_audit(frame, grain)
    issues = context.issues.frame()
    audit.to_csv(output / f"cses_{grain.lower()}_alignment_audit.csv", index=False)
    issues.to_csv(output / f"cses_{grain.lower()}_data_issues.csv", index=False)
    write_metadata(context, output, grain, HH_COLUMNS if grain == "HH" else HL_COLUMNS)
    return frame, audit, issues
