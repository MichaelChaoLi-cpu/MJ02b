#!/usr/bin/env python3
"""Audit CSES interview-date fields without modifying raw data or PostgreSQL.

The CSES releases are stored as Stata files inside ZIP and nested-ZIP archives.
This program reads those members in memory, identifies date-like fields, and
distinguishes explicit household visit dates from survey-month fields and
operational timestamps such as ``ChangeDate`` or ``SignOut``.

Only aggregate audit files are written. Row-level household dates are not
exported, and this program has no database connection or publication path.
"""

from __future__ import annotations

import io
import re
from pathlib import Path, PurePosixPath

import pandas as pd

from inventory_cses_archives import (
    DataSource,
    discover_sources,
    normalize_wave,
    source_leaf,
    token,
)


WAVES = [
    "2004", "2007", "2009", "2011-12", "2013",
    "2014", "2016", "2017", "2019", "2021",
]
NOMINAL_YEAR = {wave: int(wave[:4]) for wave in WAVES}

DIRECT_COMPONENT_RULES = {
    # A rule is used only when all three components exist in the same source.
    "first_visit": ("day_1st", "month_1st", "year_1st"),
    "last_visit": ("day_2nd", "month_2nd", "year_2nd"),
    "last_visit_unsuffixed": ("day", "month", "year"),
}
SPECIAL_COMPONENT_RULES = {
    # CSES 2004 stores four-digit calendar years as 3/4/5.
    "interview_date": ("q40_intd", "q40_intm", "q40_inty", 2000),
    "last_visit_2004": ("q40_lasd", "q40_lasm", "q40_lasy", 2000),
    "reinterview_date": ("q40_ared", "q40_arem", "q40_arey", 2000),
}

DATE_TERMS = re.compile(
    r"(?:interview|visit|survey|enumerat|fieldwork|change.?date|sign.?out|"
    r"date|\bday\b|\bmonth\b|\byear\b)",
    flags=re.IGNORECASE,
)
EXCLUDED_TERMS = re.compile(
    r"(?:birth|death|marri|crop|harvest|loan|repay|purchase|school|migration|"
    r"employment|illness|injury|consumption|recall)",
    flags=re.IGNORECASE,
)
OPERATIONAL_TERMS = re.compile(
    r"(?:change.?date|modified|updated|created|timestamp|sign.?out)",
    flags=re.IGNORECASE,
)
VISIT_TERMS = re.compile(r"(?:interview|visit)", flags=re.IGNORECASE)
SURVEY_MONTH_NAMES = {
    "monthsur", "surveymonth", "surveymonths", "surveymonthcode",
}
HOUSEHOLD_SOURCE_TERMS = re.compile(
    r"(?:hhother|otherinfo|households|headinghouseholds|areainfo|areainformation)",
    flags=re.IGNORECASE,
)

DOCUMENTED_FIELDWORK = [
    {
        "survey_wave": "2004",
        "source_archive": "data/raw/CSE/CSES 2004.zip",
        "source_member": "CSES 2004/CSES Report 03.04/CSES Report 2004.pdf",
        "documented_period": "November 2003 through January 2005",
        "implication": "The nominal 2004 wave deliberately spans three calendar years; exact household dates are required.",
    },
    {
        "survey_wave": "2007",
        "source_archive": "https://microdata.nis.gov.kh/index.php/catalog/13/study-description",
        "source_member": "Official NIS CSES 2007 study metadata",
        "documented_period": "Fieldwork October 2006 through December 2007; released estimates use calendar-year 2007 records",
        "implication": "The released 3,593-household analytical sample is CSES 2007, but the broader operation began in 2006.",
    },
    {
        "survey_wave": "2009",
        "source_archive": "data/raw/CSE/CSES 2009.zip",
        "source_member": "CSES 2009/Final Report CSES 2009.pdf",
        "documented_period": "January through December 2009",
        "implication": "Nominal year 2009 is supported for every released survey month.",
    },
    {
        "survey_wave": "2011-12",
        "source_archive": "https://microdata.nis.gov.kh/index.php/catalog/17",
        "source_member": "Official NIS CSES 2011 study metadata",
        "documented_period": "January 1 through December 29, 2011",
        "implication": "The microdata are CSES 2011; the 2012 field manual bundled in the local archive does not change the survey data year.",
    },
    {
        "survey_wave": "2013",
        "source_archive": "data/raw/CSE/CSES2013.zip",
        "source_member": "CSES2013/CSES2013/Final Report CSES 2013.pdf",
        "documented_period": "January through December 2013",
        "implication": "Nominal year 2013 is supported; ChangeDate values in 2014 are data-operation timestamps, not fieldwork dates.",
    },
    {
        "survey_wave": "2014",
        "source_archive": "data/raw/CSE/CSES 2014.zip",
        "source_member": "CSES 2014/CSES_2014_Report.pdf",
        "documented_period": "January through December 2014",
        "implication": "Nominal year 2014 is supported for every released survey month.",
    },
    {
        "survey_wave": "2016",
        "source_archive": "https://microdata.nis.gov.kh/index.php/catalog/42",
        "source_member": "Official NIS CSES 2016 study metadata",
        "documented_period": "January 1 through December 29, 2016",
        "implication": "Nominal year 2016 is supported; released ChangeDate fields are later data-operation timestamps.",
    },
    {
        "survey_wave": "2017",
        "source_archive": "https://www.nis.gov.kh/nis/CSES/Final%20Report%20CSES%202017.pdf",
        "source_member": "Official NIS CSES 2017 final report",
        "documented_period": "January through December 2017",
        "implication": "Nominal year 2017 is supported; no household interview day is released.",
    },
]


def read_stata(source: DataSource) -> tuple[pd.DataFrame, dict[str, str]]:
    input_obj: Path | io.BytesIO = (
        io.BytesIO(source.read_bytes()) if source.archive_members else source.root_file
    )
    reader = pd.io.stata.StataReader(input_obj, convert_categoricals=False)
    labels = reader.variable_labels()
    frame = reader.read()
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    return frame, labels


def display_source(source: DataSource, root: Path) -> str:
    return source.display_name(root)


def compact_values(series: pd.Series, limit: int = 5) -> str:
    values = series.dropna().drop_duplicates().head(limit)
    return " | ".join(str(value).replace("\n", " ")[:80] for value in values)


def is_household_candidate_source(source: DataSource) -> bool:
    leaf = PurePosixPath(source_leaf(source)).stem
    return bool(HOUSEHOLD_SOURCE_TERMS.search(token(leaf)))


def classify_field(
    source: DataSource,
    variable: str,
    label: str,
    dtype: object,
) -> tuple[str, str] | None:
    name_key = token(variable)
    combined = f"{variable} {label}".strip()
    leaf_key = token(PurePosixPath(source_leaf(source)).stem)

    if OPERATIONAL_TERMS.search(combined):
        return (
            "operational_timestamp_not_interview_date",
            "Field name/label indicates a data-operation or sign-out timestamp.",
        )
    if name_key in SURVEY_MONTH_NAMES:
        return (
            "survey_month_only",
            "Released survey month; useful for month-level alignment but not an exact date.",
        )
    if EXCLUDED_TERMS.search(combined):
        return None
    if VISIT_TERMS.search(combined):
        component_hint = bool(
            re.search(r"(?:date|\bday\b|\bmonth\b|\byear\b)", combined, re.IGNORECASE)
        )
        if not component_hint:
            return None
        return (
            "explicit_visit_field",
            "Variable label explicitly refers to a household visit or interview.",
        )
    if (
        is_household_candidate_source(source)
        and name_key in {"day", "month", "year", "day1st", "month1st", "year1st", "day2nd", "month2nd", "year2nd"}
    ):
        return (
            "candidate_date_component",
            "Generic date component in a household other-information/core source; component labels must confirm interpretation.",
        )
    if DATE_TERMS.search(combined) and (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or "survey" in combined.lower()
        or "enumerat" in combined.lower()
    ):
        return (
            "other_date_candidate",
            "Date-like survey field requiring documentation before use.",
        )
    return None


def clean_component(series: pd.Series, lower: int, upper: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.between(lower, upper) & values.mod(1).eq(0)
    return values.where(valid).astype("Int64")


def construct_date(
    frame: pd.DataFrame,
    components: tuple[str, str, str],
    year_offset: int = 0,
) -> pd.Series:
    day = clean_component(frame[components[0]], 1, 31)
    month = clean_component(frame[components[1]], 1, 12)
    raw_year = pd.to_numeric(frame[components[2]], errors="coerce") + year_offset
    year = clean_component(raw_year, 1990, 2035)
    # pandas' datetime assembler cannot always consume nullable Int64 columns
    # containing pd.NA. Float64 preserves missingness and integer-valued parts.
    parts = pd.DataFrame({
        "year": year.astype("float64"),
        "month": month.astype("float64"),
        "day": day.astype("float64"),
    })
    return pd.to_datetime(parts, errors="coerce")


def household_key_column(frame: pd.DataFrame) -> str | None:
    columns = {token(column): str(column) for column in frame.columns}
    for alias in ("hhid", "householdid"):
        if alias in columns:
            return columns[alias]
    return None


def audit_source(
    root: Path,
    source: DataSource,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    wave = normalize_wave(display_source(source, root))
    if wave not in WAVES:
        return [], []
    frame, labels = read_stata(source)
    source_name = display_source(source, root)
    field_rows: list[dict[str, object]] = []
    construction_rows: list[dict[str, object]] = []

    for variable in frame.columns:
        label = labels.get(str(variable), "") or ""
        classification = classify_field(source, str(variable), label, frame[variable].dtype)
        if classification is None:
            continue
        category, rationale = classification
        series = frame[variable]
        field_rows.append({
            "survey_wave": wave,
            "nominal_survey_year": NOMINAL_YEAR[wave],
            "source": source_name,
            "source_leaf": source_leaf(source),
            "variable": str(variable),
            "variable_label": label,
            "dtype": str(series.dtype),
            "classification": category,
            "classification_rationale": rationale,
            "rows": len(frame),
            "non_null": int(series.notna().sum()),
            "non_null_rate": float(series.notna().mean()) if len(frame) else 0.0,
            "distinct_non_null": int(series.nunique(dropna=True)),
            "minimum": str(series.dropna().min()) if series.notna().any() else "",
            "maximum": str(series.dropna().max()) if series.notna().any() else "",
            "sample_values": compact_values(series),
        })

    normalized_columns = {token(column): str(column) for column in frame.columns}
    key_column = household_key_column(frame)
    household_rows = int(frame[key_column].nunique(dropna=True)) if key_column else 0
    household_source = is_household_candidate_source(source)

    rules = [
        (role, aliases, 0) for role, aliases in DIRECT_COMPONENT_RULES.items()
    ] + [
        (role, rule[:3], rule[3]) for role, rule in SPECIAL_COMPONENT_RULES.items()
    ]
    for date_role, aliases, year_offset in rules:
        normalized_aliases = tuple(token(alias) for alias in aliases)
        if not all(alias in normalized_columns for alias in normalized_aliases):
            continue
        columns = tuple(normalized_columns[alias] for alias in normalized_aliases)
        # Unsuffixed day/month/year are accepted only in a household core source.
        if date_role == "last_visit_unsuffixed" and not household_source:
            continue
        labels_text = " ".join(labels.get(column, "") or "" for column in columns)
        label_confirms_visit = bool(VISIT_TERMS.search(labels_text))
        dates = construct_date(frame, columns, year_offset=year_offset)
        valid = dates.notna()
        date_year = dates.dt.year
        nominal_mismatch = valid & date_year.ne(NOMINAL_YEAR[wave])
        if key_column:
            keyed = pd.DataFrame({"key": frame[key_column], "date": dates}).dropna(subset=["key"])
            household_with_date = int(keyed.loc[keyed["date"].notna(), "key"].nunique())
            household_conflicting_dates = int(
                (keyed.groupby("key", dropna=True)["date"].nunique(dropna=True) > 1).sum()
            )
        else:
            household_with_date = 0
            household_conflicting_dates = 0
        confidence = (
            "direct_visit_date"
            if label_confirms_visit
            else "unconfirmed_component_date"
        )
        construction_rows.append({
            "survey_wave": wave,
            "nominal_survey_year": NOMINAL_YEAR[wave],
            "source": source_name,
            "source_leaf": source_leaf(source),
            "date_role": date_role,
            "day_variable": columns[0],
            "month_variable": columns[1],
            "year_variable": columns[2],
            "year_offset": year_offset,
            "component_labels": labels_text.strip(),
            "classification": confidence,
            "rows": len(frame),
            "valid_dates": int(valid.sum()),
            "valid_date_rate": float(valid.mean()) if len(frame) else 0.0,
            "minimum_date": dates.min().date().isoformat() if valid.any() else "",
            "maximum_date": dates.max().date().isoformat() if valid.any() else "",
            "distinct_dates": int(dates.nunique(dropna=True)),
            "actual_years": ", ".join(str(int(year)) for year in sorted(date_year.dropna().unique())),
            "nominal_year_mismatch_rows": int(nominal_mismatch.sum()),
            "nominal_year_mismatch_rate": float(nominal_mismatch.mean()) if len(frame) else 0.0,
            "household_key_variable": key_column or "",
            "unique_households": household_rows,
            "households_with_valid_date": household_with_date,
            "household_date_coverage": (
                household_with_date / household_rows if household_rows else 0.0
            ),
            "households_with_conflicting_dates": household_conflicting_dates,
        })

        # In 2019 the visit-year component is missing for a small number of
        # households, while the separately released survey-year field is
        # complete. Quantify (but do not silently apply) that recovery path.
        survey_year_column = normalized_columns.get("yearsur")
        if date_role == "last_visit_unsuffixed" and survey_year_column:
            direct_year = pd.to_numeric(frame[columns[2]], errors="coerce")
            survey_year = pd.to_numeric(frame[survey_year_column], errors="coerce")
            overlap = direct_year.notna() & survey_year.notna()
            agreement = direct_year.eq(survey_year)
            recoverable = direct_year.isna() & survey_year.notna()
            recovered_frame = frame.copy()
            recovered_frame[columns[2]] = direct_year.fillna(survey_year)
            recovered_dates = construct_date(recovered_frame, columns)
            recovery_valid = recovered_dates.notna()
            construction_rows.append({
                "survey_wave": wave,
                "nominal_survey_year": NOMINAL_YEAR[wave],
                "source": source_name,
                "source_leaf": source_leaf(source),
                "date_role": "last_visit_with_survey_year_fallback",
                "day_variable": columns[0],
                "month_variable": columns[1],
                "year_variable": f"{columns[2]} fallback {survey_year_column}",
                "year_offset": 0,
                "component_labels": f"{labels_text.strip()} / {labels.get(survey_year_column, '')}",
                "classification": (
                    "validated_component_fallback"
                    if overlap.any() and agreement.loc[overlap].all()
                    else "unvalidated_component_fallback"
                ),
                "rows": len(frame),
                "valid_dates": int(recovery_valid.sum()),
                "valid_date_rate": float(recovery_valid.mean()) if len(frame) else 0.0,
                "minimum_date": recovered_dates.min().date().isoformat() if recovery_valid.any() else "",
                "maximum_date": recovered_dates.max().date().isoformat() if recovery_valid.any() else "",
                "distinct_dates": int(recovered_dates.nunique(dropna=True)),
                "actual_years": ", ".join(str(int(year)) for year in sorted(recovered_dates.dt.year.dropna().unique())),
                "nominal_year_mismatch_rows": int((recovery_valid & recovered_dates.dt.year.ne(NOMINAL_YEAR[wave])).sum()),
                "nominal_year_mismatch_rate": float((recovery_valid & recovered_dates.dt.year.ne(NOMINAL_YEAR[wave])).mean()) if len(frame) else 0.0,
                "household_key_variable": key_column or "",
                "unique_households": household_rows,
                "households_with_valid_date": int(recovery_valid.sum()) if key_column else 0,
                "household_date_coverage": float(recovery_valid.mean()) if key_column and len(frame) else 0.0,
                "households_with_conflicting_dates": 0,
                "fallback_variable": survey_year_column,
                "fallback_recoverable_rows": int(recoverable.sum()),
                "fallback_overlap_rows": int(overlap.sum()),
                "fallback_overlap_agreement_rate": float(agreement.loc[overlap].mean()) if overlap.any() else 0.0,
            })
    return field_rows, construction_rows


def wave_summary(
    fields: pd.DataFrame,
    constructions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for wave in WAVES:
        wave_fields = fields.loc[fields["survey_wave"].eq(wave)] if not fields.empty else fields
        wave_dates = constructions.loc[
            constructions["survey_wave"].eq(wave)
            & constructions["classification"].eq("direct_visit_date")
        ] if not constructions.empty else constructions
        if not wave_dates.empty:
            role_priority = {
                "interview_date": 0,
                "last_visit": 1,
                "last_visit_unsuffixed": 1,
                "last_visit_2004": 1,
                "first_visit": 2,
                "reinterview_date": 3,
            }
            preferred = wave_dates.assign(
                _role_priority=wave_dates["date_role"].map(role_priority).fillna(9)
            ).sort_values(
                ["_role_priority", "valid_date_rate"], ascending=[True, False]
            ).iloc[0]
            status = "exact_visit_date_available"
            source = preferred["source"]
            role = preferred["date_role"]
            coverage = preferred["valid_date_rate"]
            min_date = preferred["minimum_date"]
            max_date = preferred["maximum_date"]
            years = preferred["actual_years"]
            mismatch = preferred["nominal_year_mismatch_rate"]
        elif not wave_fields.loc[wave_fields["classification"].eq("survey_month_only")].empty:
            month = wave_fields.loc[wave_fields["classification"].eq("survey_month_only")].sort_values(
                "non_null_rate", ascending=False
            ).iloc[0]
            status = "survey_month_only"
            source = month["source"]
            role = "survey_month"
            coverage = month["non_null_rate"]
            min_date = ""
            max_date = ""
            years = ""
            mismatch = 0.0
        else:
            status = "no_confirmed_interview_date"
            source = ""
            role = ""
            coverage = 0.0
            min_date = ""
            max_date = ""
            years = ""
            mismatch = 0.0
        rows.append({
            "survey_wave": wave,
            "nominal_survey_year": NOMINAL_YEAR[wave],
            "date_availability": status,
            "preferred_source": source,
            "preferred_date_role": role,
            "coverage_rate": coverage,
            "minimum_date": min_date,
            "maximum_date": max_date,
            "actual_years": years,
            "nominal_year_mismatch_rate": mismatch,
            "operational_timestamp_fields": int(
                wave_fields["classification"].eq("operational_timestamp_not_interview_date").sum()
            ) if not wave_fields.empty else 0,
        })
    return pd.DataFrame(rows)


def write_readme(
    path: Path,
    fields: pd.DataFrame,
    constructions: pd.DataFrame,
    summary: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    exact = summary.loc[summary["date_availability"].eq("exact_visit_date_available")]
    mismatch = exact.loc[exact["nominal_year_mismatch_rate"].gt(0)]
    lines = [
        "# CSES interview-date audit",
        "",
        "This is a read-only audit of the original CSES archives. ZIP members are",
        "read in memory; `data/raw/` and PostgreSQL are not modified.",
        "",
        "## Interpretation rules",
        "",
        "- `direct_visit_date` requires complete day/month/year components whose",
        "  Stata labels explicitly identify a visit or interview.",
        "- `survey_month_only` supports month-level matching but is not an exact date.",
        "- `ChangeDate` and `SignOut` are classified as operational timestamps and",
        "  must not be substituted for interview dates without documentation.",
        "- Outputs are aggregate audits; no household-level date records are exported.",
        "",
        "## Summary",
        "",
        f"- Waves scanned: {len(summary)}",
        f"- Waves with confirmed exact visit dates: {len(exact)}",
        f"- Confirmed exact-date constructions: {int(constructions['classification'].eq('direct_visit_date').sum()) if not constructions.empty else 0}",
        f"- Candidate fields audited: {len(fields)}",
        f"- Read errors: {len(errors)}",
    ]
    if len(mismatch):
        lines.extend([
            "",
            "## Timing warning",
            "",
            "At least one release wave contains confirmed visit dates outside its",
            "nominal wave year. Weather exposure must use the actual visit year for",
            "those records before date-sensitive estimation is interpreted.",
        ])
    lines.extend([
        "",
        "## Files",
        "",
        "- `cses_interview_date_wave_summary.csv`: one row per release wave.",
        "- `cses_interview_date_constructions.csv`: validated component-date audits.",
        "- `cses_interview_date_candidate_fields.csv`: field-level candidate inventory.",
        "- `cses_interview_date_read_errors.csv`: sources that could not be read.",
        "- `cses_fieldwork_document_evidence.csv`: fieldwork-period evidence from release documentation.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)

    field_rows: list[dict[str, object]] = []
    construction_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for source in discover_sources(root):
        try:
            fields, constructions = audit_source(root, source)
            field_rows.extend(fields)
            construction_rows.extend(constructions)
        except Exception as exc:  # Preserve a full audit trail and continue scanning.
            error_rows.append({
                "source": display_source(source, root),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    fields = pd.DataFrame(field_rows)
    constructions = pd.DataFrame(construction_rows)
    errors = pd.DataFrame(error_rows, columns=["source", "error_type", "error"])
    if not fields.empty:
        fields = fields.sort_values(
            ["survey_wave", "classification", "source", "variable"], ignore_index=True
        )
    if not constructions.empty:
        constructions = constructions.sort_values(
            ["survey_wave", "classification", "source", "date_role"], ignore_index=True
        )
    summary = wave_summary(fields, constructions)

    fields.to_csv(output / "cses_interview_date_candidate_fields.csv", index=False)
    constructions.to_csv(output / "cses_interview_date_constructions.csv", index=False)
    summary.to_csv(output / "cses_interview_date_wave_summary.csv", index=False)
    errors.to_csv(output / "cses_interview_date_read_errors.csv", index=False)
    pd.DataFrame(DOCUMENTED_FIELDWORK).to_csv(
        output / "cses_fieldwork_document_evidence.csv", index=False
    )
    write_readme(
        output / "README_interview_date_audit.md",
        fields,
        constructions,
        summary,
        errors,
    )

    print(summary.to_string(index=False))
    print(f"candidate_fields={len(fields)} constructions={len(constructions)} errors={len(errors)}")
    print("output=data/exp/database/cses_interview_date_wave_summary.csv")


if __name__ == "__main__":
    main()
