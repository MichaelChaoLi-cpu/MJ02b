"""Shared contracts for the household-level CSES survey-date extension."""

from __future__ import annotations

import pandas as pd


EXPECTED_ROWS = 77_904
EXPECTED_EXACT_COVERAGE = {"2004": 14_984, "2019": 10_041, "2021": 10_080}
ACTUAL_DATE_COLUMNS = [
    "survey_actual_year",
    "survey_actual_month",
    "survey_actual_day",
]
ACTUAL_DATE_TYPES = [(column, "smallint") for column in ACTUAL_DATE_COLUMNS]

FINAL_SURVEY_DATE_TYPES = [
    ("dataset_name", "text"),
    ("survey_wave", "text"),
    ("nominal_survey_year", "smallint"),
    ("household_id", "text"),
    ("survey_month", "smallint"),
    ("released_survey_year", "smallint"),
    ("released_survey_month", "smallint"),
    ("interview_date", "date"),
    ("first_visit_date", "date"),
    ("last_visit_date", "date"),
    ("reinterview_date", "date"),
    ("candidate_reference_date", "date"),
    ("candidate_reference_definition", "text"),
    ("survey_actual_year", "smallint"),
    ("survey_actual_month", "smallint"),
    ("survey_actual_day", "smallint"),
    ("date_precision", "text"),
    ("candidate_actual_year", "smallint"),
    ("candidate_actual_month", "smallint"),
    ("confirmed_survey_year", "smallint"),
    ("confirmed_survey_month", "smallint"),
    ("confirmed_survey_time_source", "text"),
    ("confirmed_year_differs", "smallint"),
    ("nominal_year_differs", "smallint"),
    ("survey_month_matches_candidate", "smallint"),
    ("candidate_date_within_documented_period", "smallint"),
    ("exact_date_source_archive", "text"),
    ("exact_date_source_submodule", "text"),
]

IND_QUE_TYPES = [
    ("canonical_varname", "text"),
    ("dataset_name", "text"),
    ("column_in_raw_sav", "text"),
    ("column_label_in_english", "text"),
    ("source_kind", "text"),
    ("measure_type", "text"),
    ("canonical_text", "text"),
]

SURVEY_DATE_SUMMARY_TYPES = [
    ("dataset_name", "text"),
    ("survey_wave", "text"),
    ("rows", "integer"),
    ("unique_households", "integer"),
    ("survey_month_available", "integer"),
    ("interview_date_available", "integer"),
    ("first_visit_date_available", "integer"),
    ("last_visit_date_available", "integer"),
    ("candidate_reference_date_available", "integer"),
    ("candidate_reference_date_coverage", "double precision"),
    ("candidate_minimum_date", "date"),
    ("candidate_maximum_date", "date"),
    ("nominal_year_differs", "integer"),
    ("nominal_year_differs_rate_among_exact", "double precision"),
    ("released_survey_year_available", "integer"),
    ("confirmed_survey_year_available", "integer"),
    ("confirmed_year_differs", "integer"),
    ("confirmed_year_differs_rate", "double precision"),
    ("survey_month_comparable", "integer"),
    ("survey_month_disagrees", "integer"),
    ("survey_month_disagrees_rate", "double precision"),
    ("candidate_dates_outside_documented_period", "integer"),
]

ALIGN_SUMMARY_TYPES = [
    ("varname", "text"),
    ("dataset_count", "integer"),
    ("source_count", "integer"),
    ("explicit_count", "integer"),
    ("derived_count", "integer"),
    ("measure_type", "text"),
    ("canonical_text", "text"),
]

_COMPONENT_LABELS = {
    "survey_actual_year": "Actual Household Survey Year",
    "survey_actual_month": "Actual Household Survey Month",
    "survey_actual_day": "Actual Household Survey Day",
}
_COMPONENT_TEXT = {
    component: (
        f"Calendar {component.removeprefix('survey_actual_')} of the selected explicit "
        "household survey date: interview date in 2004 and last-visit date in "
        "2019/2021; null in waves without a defensible household-level exact date."
    )
    for component in ACTUAL_DATE_COLUMNS
}
_RAW_COMPONENTS = {
    "2004": {
        "survey_actual_year": "q40_inty",
        "survey_actual_month": "q40_intm",
        "survey_actual_day": "q40_intd",
    },
    "2019": {
        "survey_actual_year": "year",
        "survey_actual_month": "month",
        "survey_actual_day": "day",
    },
    "2021": {
        "survey_actual_year": "year_2nd",
        "survey_actual_month": "month_2nd",
        "survey_actual_day": "day_2nd",
    },
}


def hh_dictionary_extension() -> pd.DataFrame:
    """Return the nine source-mapping rows added to ``ind_que_HH_CSES``."""
    rows = []
    for wave, raw_fields in _RAW_COMPONENTS.items():
        for component in ACTUAL_DATE_COLUMNS:
            rows.append(
                {
                    "canonical_varname": component,
                    "dataset_name": f"CSES {wave}",
                    "column_in_raw_sav": raw_fields[component],
                    "column_label_in_english": _COMPONENT_LABELS[component],
                    "source_kind": "derived",
                    "measure_type": "time",
                    "canonical_text": _COMPONENT_TEXT[component],
                }
            )
    return pd.DataFrame(rows, columns=[name for name, _ in IND_QUE_TYPES])


def hh_summary_extension() -> pd.DataFrame:
    """Return the three variable rows added to ``align_summary_HH_CSES``."""
    rows = [
        {
            "varname": component,
            "dataset_count": 3,
            "source_count": 3,
            "explicit_count": 0,
            "derived_count": 3,
            "measure_type": "time",
            "canonical_text": _COMPONENT_TEXT[component],
        }
        for component in ACTUAL_DATE_COLUMNS
    ]
    return pd.DataFrame(rows, columns=[name for name, _ in ALIGN_SUMMARY_TYPES])


def enrich_hh_frame(hh: pd.DataFrame, survey_dates: pd.DataFrame) -> pd.DataFrame:
    """Attach coherent actual-date components to the household table."""
    keys = ["survey_wave", "household_id"]
    extension = survey_dates[keys + ACTUAL_DATE_COLUMNS]
    if extension.duplicated(keys).any():
        raise ValueError("Survey-date extension has duplicate household keys")
    enriched = hh.merge(extension, on=keys, how="left", validate="1:1")
    return enriched


def extend_hh_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    """Idempotently attach the actual-date mappings to the HH dictionary."""
    base = dictionary.loc[
        ~dictionary["canonical_varname"].isin(ACTUAL_DATE_COLUMNS)
    ]
    return pd.concat([base, hh_dictionary_extension()], ignore_index=True)


def extend_hh_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Idempotently attach the actual-date rows to the HH alignment summary."""
    base = summary.loc[~summary["varname"].isin(ACTUAL_DATE_COLUMNS)]
    return pd.concat([base, hh_summary_extension()], ignore_index=True)
