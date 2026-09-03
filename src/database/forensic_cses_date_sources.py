#!/usr/bin/env python3
"""Forensically audit CSES day-level time sources without writing PostgreSQL.

The normal survey-date publisher accepts only explicitly labelled household
interview/visit dates.  This deeper audit documents why other day-level fields
must remain separate:

* household questionnaire cover fields that were collected on paper but were
  omitted from the released Stata tables;
* expenditure/income diary dates, which describe an observation window rather
  than a household interview day; and
* child anthropometric measurement dates, which are exact field dates for a
  selected household subset but are not a general interview date.

All row-level candidate artifacts stay in ``data/exp/database``.  The program
has no database imports, credentials, connections, or publication code.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import openpyxl
import pandas as pd
import xlrd

from audit_cses_interview_dates import read_stata
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

DIARY_CONFIG = {
    "2009": {
        "year": 2009,
        "sources": {
            "diaryexpenditure": "diaryexp_q2",
            "diaryincome": "diaryinc_q2",
        },
    },
    "2011-12": {
        # Official NIS metadata identifies this release as CSES 2011 and gives
        # fieldwork dates of 2011-01-01 through 2011-12-29.  The archive also
        # happens to contain a 2012 field manual, which is not the data year.
        "year": 2011,
        "sources": {
            "11dyexpenditure": "diaryexp_q2",
            "11dyincome": "diaryinc_q2",
        },
    },
    "2014": {
        "year": 2014,
        "sources": {
            "diaryexpenditure": "diaryexp_q2",
            "diaryincome": "diaryinc_q2",
        },
    },
}

MEASUREMENT_CONFIG = {
    "2007": {
        "leaf": "12_healthcheckchildren.dta",
        "day": "q12_c03a",
        "month": "q12_c03b",
        "year": "q12_c03c",
        "year_constant": None,
    },
    "2009": {
        "leaf": "personhealthu5.dta",
        "day": "q12_c10a",
        "month": "q12_c10b",
        "year": None,
        "year_constant": 2009,
    },
}

QUESTIONNAIRE_CONFIG = {
    "2009": (
        Path("data/raw/CSE/CSES 2009.zip"),
        ("CSES 2009/CSES2009 HH Questionnaire ENG.xls",),
    ),
    "2011-12": (
        Path("data/raw/CSE/CSES 2011-12.zip"),
        ("CSES 2011-12/CSES2011HH Questionnaire ENG.xls",),
    ),
    "2013": (
        Path("data/raw/CSE/CSES2013.zip"),
        (
            "CSES2013/CSES2013/CSES 2013.zip",
            "CSES2013 HH Questionnaire ENG.xls",
        ),
    ),
    "2014": (
        Path("data/raw/CSE/CSES 2014.zip"),
        ("CSES 2014/CSES2014 HH Quest ENG_draft_16_11_2013_WFP comments.xlsx",),
    ),
    "2016": (
        Path("data/raw/CSE/CSES2016.zip"),
        ("CSES2016/CSES2016/CSES2016 HH Quest ENG.xlsx",),
    ),
}

OFFICIAL_2007_QUESTIONNAIRE_URL = (
    "https://microdata.nis.gov.kh/index.php/catalog/13/download/86"
)
OFFICIAL_2007_QUESTIONNAIRE_SHA256 = (
    "5b6391dad86800ed6b83e6c5a85d3b96548f39ff0e82a57ead91d45cfaab9ad5"
)
OFFICIAL_2007_EVIDENCE = (
    (32, "first_household_visit", "Date of first visit to Household | Day: | Month: | Year:"),
    (34, "last_household_visit", "Date of last visit | Day: | Month: | Year:"),
    (40, "survey_month_and_year", "Month and Year of Survey | Month | Year:"),
)

QUESTIONNAIRE_PATTERNS = (
    (re.compile(r"date of first visit", re.I), "first_household_visit"),
    (re.compile(r"date of last visit", re.I), "last_household_visit"),
    (re.compile(r"month and year of survey", re.I), "survey_month_and_year"),
    (re.compile(r"date of re-?interview", re.I), "quality_control_reinterview"),
    (re.compile(r"date of interview", re.I), "interview_date"),
)


def clean_household_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    values = values.mask(values.str.lower().isin(["", "nan", "none", "<na>"]))
    digits = values.str.fullmatch(r"\d+", na=False)
    return values.where(~digits, values.str.zfill(7))


def read_member_chain(archive_path: Path, members: tuple[str, ...]) -> bytes:
    payload: bytes | None = None
    for member in members:
        archive_input: Path | io.BytesIO = (
            archive_path if payload is None else io.BytesIO(payload)
        )
        with zipfile.ZipFile(archive_input) as archive:
            payload = archive.read(member)
    if payload is None:
        raise ValueError(f"No members supplied for {archive_path}")
    return payload


def spreadsheet_rows(payload: bytes, suffix: str) -> Iterable[tuple[str, int, str]]:
    if suffix.lower() == ".xlsx":
        workbook = openpyxl.load_workbook(
            io.BytesIO(payload), read_only=True, data_only=True
        )
        for sheet in workbook.worksheets:
            for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                cells = [" ".join(str(value).split()) for value in row if value not in (None, "")]
                if cells:
                    yield sheet.title, row_number, " | ".join(cells)
        workbook.close()
        return

    workbook = xlrd.open_workbook(file_contents=payload, on_demand=True)
    for sheet_name in workbook.sheet_names():
        sheet = workbook.sheet_by_name(sheet_name)
        for row_number in range(sheet.nrows):
            cells = [
                " ".join(str(sheet.cell_value(row_number, column)).split())
                for column in range(sheet.ncols)
                if str(sheet.cell_value(row_number, column)).strip()
            ]
            if cells:
                yield sheet_name, row_number + 1, " | ".join(cells)
    workbook.release_resources()


def questionnaire_evidence(
    root: Path,
    official_2007_questionnaire: Path | None,
) -> pd.DataFrame:
    sources: list[tuple[str, str, str, bytes]] = []
    for wave, (relative_archive, members) in QUESTIONNAIRE_CONFIG.items():
        archive_path = root / relative_archive
        payload = read_member_chain(archive_path, members)
        reference = str(relative_archive) + "::" + "::".join(members)
        sources.append((wave, "local_raw_archive", reference, payload))

    if official_2007_questionnaire is not None:
        payload = official_2007_questionnaire.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != OFFICIAL_2007_QUESTIONNAIRE_SHA256:
            raise RuntimeError(
                "The supplied official CSES 2007 questionnaire does not match the "
                f"audited SHA-256: {checksum}"
            )
        sources.append(
            (
                "2007",
                "official_nis_download",
                OFFICIAL_2007_QUESTIONNAIRE_URL,
                payload,
            )
        )

    rows: list[dict[str, object]] = []
    for wave, source_kind, reference, payload in sources:
        suffix = ".xlsx" if payload[:2] == b"PK" else ".xls"
        checksum = hashlib.sha256(payload).hexdigest()
        for sheet, row_number, text in spreadsheet_rows(payload, suffix):
            for pattern, role in QUESTIONNAIRE_PATTERNS:
                if pattern.search(text):
                    rows.append(
                        {
                            "survey_wave": wave,
                            "source_kind": source_kind,
                            "source_reference": reference,
                            "source_sha256": checksum,
                            "sheet": sheet,
                            "row_number": row_number,
                            "date_field_role": role,
                            "evidence_text": text,
                            "released_in_current_stata": wave in {"2004", "2019", "2021"},
                            "audit_interpretation": (
                                "Collected on the questionnaire; omission from the released "
                                "Stata tables does not mean it was absent from fieldwork."
                            ),
                        }
                    )
                    break
    if official_2007_questionnaire is None:
        for row_number, role, text in OFFICIAL_2007_EVIDENCE:
            rows.append(
                {
                    "survey_wave": "2007",
                    "source_kind": "official_nis_download_preverified",
                    "source_reference": OFFICIAL_2007_QUESTIONNAIRE_URL,
                    "source_sha256": OFFICIAL_2007_QUESTIONNAIRE_SHA256,
                    "sheet": "Cover",
                    "row_number": row_number,
                    "date_field_role": role,
                    "evidence_text": text,
                    "released_in_current_stata": False,
                    "audit_interpretation": (
                        "Collected on the official questionnaire; the checked source hash "
                        "and exact cover rows are retained for offline reproducibility."
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["survey_wave", "row_number", "date_field_role"]
    ).reset_index(drop=True)


def source_index(root: Path) -> dict[tuple[str, str], list[DataSource]]:
    index: dict[tuple[str, str], list[DataSource]] = {}
    for source in discover_sources(root):
        wave = normalize_wave(source.display_name(root))
        leaf_key = token(Path(source_leaf(source)).stem)
        index.setdefault((wave, leaf_key), []).append(source)
    return index


def unique_source(
    index: dict[tuple[str, str], list[DataSource]], wave: str, leaf: str
) -> DataSource:
    matches = index.get((wave, token(Path(leaf).stem)), [])
    if len(matches) != 1:
        names = [source_leaf(source) for source in matches]
        raise RuntimeError(f"Expected one {wave} source for {leaf}; found {names}")
    return matches[0]


def parse_ddmm(series: pd.Series, year: int) -> pd.Series:
    raw = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.zfill(4)
    )
    day = pd.to_numeric(raw.str[:2], errors="coerce")
    month = pd.to_numeric(raw.str[2:4], errors="coerce")
    return pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": day}),
        errors="coerce",
    )


def diary_candidates(
    root: Path,
    spine: pd.DataFrame,
    index: dict[tuple[str, str], list[DataSource]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    household_outputs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for wave, config in DIARY_CONFIG.items():
        all_pairs: list[pd.DataFrame] = []
        source_rows = 0
        valid_source_rows = 0
        source_names: list[str] = []
        for leaf_key, date_variable in config["sources"].items():
            source = unique_source(index, wave, leaf_key)
            frame, _labels = read_stata(source)
            if not {"hhid", date_variable}.issubset(frame.columns):
                raise RuntimeError(
                    f"{source.display_name(root)} lacks hhid or {date_variable}"
                )
            dates = parse_ddmm(frame[date_variable], int(config["year"]))
            household_id = clean_household_id(frame["hhid"])
            valid = dates.notna() & household_id.notna()
            source_rows += len(frame)
            valid_source_rows += int(valid.sum())
            source_names.append(source.display_name(root))
            all_pairs.append(
                pd.DataFrame(
                    {"Household ID": household_id[valid], "Diary Date": dates[valid]}
                ).drop_duplicates()
            )

        pairs = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
        reference = spine.loc[
            spine["Survey Wave"].eq(wave),
            ["Survey Wave", "Household ID", "Survey Month"],
        ].copy()
        joined = reference.merge(
            pairs, on="Household ID", how="left", validate="one_to_many"
        )
        joined["Diary Date Matches Survey Month"] = (
            joined["Diary Date"].dt.month.eq(joined["Survey Month"])
            .where(joined["Diary Date"].notna())
        )

        observed = joined.loc[joined["Diary Date"].notna()].copy()
        month_counts = (
            observed.assign(Diary_Month=observed["Diary Date"].dt.month)
            .groupby(["Household ID", "Diary_Month"], as_index=False)
            .size()
            .sort_values(
                ["Household ID", "size", "Diary_Month"],
                ascending=[True, False, True],
            )
        )
        modal = month_counts.drop_duplicates("Household ID").rename(
            columns={"Diary_Month": "Diary Modal Month", "size": "Diary Modal Month Dates"}
        )

        all_dates = observed.groupby("Household ID", as_index=False).agg(
            **{
                "Diary Start Date": ("Diary Date", "min"),
                "Diary End Date": ("Diary Date", "max"),
                "Diary Distinct Dates": ("Diary Date", "nunique"),
                "Diary All Dates Match Survey Month": (
                    "Diary Date Matches Survey Month", "all"
                ),
            }
        )
        in_month = observed.loc[observed["Diary Date Matches Survey Month"].eq(True)]
        in_month_agg = in_month.groupby("Household ID", as_index=False).agg(
            **{
                "Survey-Month Diary Start Date": ("Diary Date", "min"),
                "Survey-Month Diary End Date": ("Diary Date", "max"),
                "Survey-Month Diary Distinct Dates": ("Diary Date", "nunique"),
            }
        )
        household = (
            reference.merge(all_dates, on="Household ID", how="left", validate="one_to_one")
            .merge(modal, on="Household ID", how="left", validate="one_to_one")
            .merge(in_month_agg, on="Household ID", how="left", validate="one_to_one")
        )
        household["Diary Modal Month Share"] = (
            household["Diary Modal Month Dates"] / household["Diary Distinct Dates"]
        )
        household["Diary Modal Month Matches Survey Month"] = (
            household["Diary Modal Month"].eq(household["Survey Month"])
            .where(household["Diary Modal Month"].notna())
        )
        household["Diary Date Role"] = "daily expenditure/income activity date"
        household["Eligible For survey_actual_day"] = False
        household["Date Use Decision"] = (
            "Retain as a diary observation window; do not label as a household interview day."
        )
        household_outputs.append(household)

        has_date = household["Diary Start Date"].notna()
        last_date = household.loc[has_date, "Diary End Date"]
        summary_rows.append(
            {
                "survey_wave": wave,
                "calendar_year_used": int(config["year"]),
                "final_households": len(reference),
                "source_transaction_rows": source_rows,
                "valid_source_transaction_rows": valid_source_rows,
                "unique_household_date_pairs": len(pairs),
                "households_with_diary_date": int(has_date.sum()),
                "household_coverage": float(has_date.mean()),
                "diary_date_min": pairs["Diary Date"].min(),
                "diary_date_max": pairs["Diary Date"].max(),
                "median_distinct_diary_dates": float(
                    household.loc[has_date, "Diary Distinct Dates"].median()
                ),
                "date_pair_matches_survey_month_rate": float(
                    observed["Diary Date Matches Survey Month"].mean()
                ),
                "households_all_dates_match_survey_month_rate": float(
                    household.loc[has_date, "Diary All Dates Match Survey Month"].mean()
                ),
                "households_modal_month_matches_survey_month_rate": float(
                    household.loc[has_date, "Diary Modal Month Matches Survey Month"].mean()
                ),
                "last_observed_date_is_calendar_month_end_rate": float(
                    last_date.dt.day.eq(last_date.dt.days_in_month).mean()
                ),
                "eligible_for_survey_actual_day": False,
                "source_modules": " | ".join(source_names),
                "interpretation": (
                    "Observed household diary activity dates, not first/last interview dates."
                ),
            }
        )

    return (
        pd.concat(household_outputs, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def measurement_candidates(
    root: Path,
    spine: pd.DataFrame,
    index: dict[tuple[str, str], list[DataSource]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    household_outputs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for wave, config in MEASUREMENT_CONFIG.items():
        source = unique_source(index, wave, str(config["leaf"]))
        frame, _labels = read_stata(source)
        household_id = clean_household_id(frame["hhid"])
        if config["year"] is None:
            year = pd.Series(config["year_constant"], index=frame.index)
            year_source = "official survey calendar year"
        else:
            year = pd.to_numeric(frame[str(config["year"])], errors="coerce")
            year_source = str(config["year"])
        dates = pd.to_datetime(
            pd.DataFrame(
                {
                    "year": year,
                    "month": pd.to_numeric(frame[str(config["month"])], errors="coerce"),
                    "day": pd.to_numeric(frame[str(config["day"])], errors="coerce"),
                }
            ),
            errors="coerce",
        )
        valid = dates.notna() & household_id.notna()
        pairs = pd.DataFrame(
            {"Household ID": household_id[valid], "Measurement Date": dates[valid]}
        ).drop_duplicates()
        reference = spine.loc[
            spine["Survey Wave"].eq(wave),
            ["Survey Wave", "Household ID", "Survey Month"],
        ].copy()
        joined = reference.merge(
            pairs, on="Household ID", how="left", validate="one_to_many"
        )
        joined["Measurement Date Matches Survey Month"] = (
            joined["Measurement Date"].dt.month.eq(joined["Survey Month"])
            .where(joined["Measurement Date"].notna())
        )
        observed = joined.loc[joined["Measurement Date"].notna()].copy()
        aggregate = observed.groupby("Household ID", as_index=False).agg(
            **{
                "First Measurement Date": ("Measurement Date", "min"),
                "Last Measurement Date": ("Measurement Date", "max"),
                "Distinct Measurement Dates": ("Measurement Date", "nunique"),
                "All Measurement Dates Match Survey Month": (
                    "Measurement Date Matches Survey Month", "all"
                ),
            }
        )
        household = reference.merge(
            aggregate, on="Household ID", how="left", validate="one_to_one"
        )
        household["Measurement Date Role"] = "child anthropometric field measurement"
        household["Measurement Year Source"] = year_source
        household["Eligible For survey_actual_day"] = False
        household["Date Use Decision"] = (
            "Subset fieldwork-date proxy only; do not label as the household interview day."
        )
        household_outputs.append(household)

        has_date = household["First Measurement Date"].notna()
        summary_rows.append(
            {
                "survey_wave": wave,
                "final_households": len(reference),
                "source_child_rows": len(frame),
                "valid_child_measurement_rows": int(valid.sum()),
                "unique_household_date_pairs": len(pairs),
                "households_with_measurement_date": int(has_date.sum()),
                "household_coverage": float(has_date.mean()),
                "measurement_date_min": pairs["Measurement Date"].min(),
                "measurement_date_max": pairs["Measurement Date"].max(),
                "date_pair_matches_survey_month_rate": float(
                    observed["Measurement Date Matches Survey Month"].mean()
                ),
                "households_all_dates_match_survey_month_rate": float(
                    household.loc[
                        has_date, "All Measurement Dates Match Survey Month"
                    ].mean()
                ),
                "households_with_multiple_measurement_dates": int(
                    household.loc[has_date, "Distinct Measurement Dates"].gt(1).sum()
                ),
                "eligible_for_survey_actual_day": False,
                "source_module": source.display_name(root),
                "interpretation": (
                    "Exact child measurement field date for a selected subset; not a general "
                    "household interview date."
                ),
            }
        )

    return (
        pd.concat(household_outputs, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def forensic_matrix(questionnaire: pd.DataFrame) -> pd.DataFrame:
    cover_waves = set(questionnaire["survey_wave"].unique())
    facts = {
        "2004": (True, True, True, True, True, False, "explicit household interview date"),
        "2007": (False, "2007" in cover_waves, False, False, True, True, "child measurement date for a subset"),
        "2009": (False, True, False, True, True, False, "household diary activity window"),
        "2011-12": (False, True, False, True, False, False, "household diary activity window"),
        "2013": (False, True, False, False, False, False, "none; operational timestamps rejected"),
        "2014": (False, True, False, True, False, False, "household diary activity window"),
        "2016": (False, True, False, False, False, True, "none; village reference date is not household visit"),
        "2017": (False, None, False, False, False, False, "none in released data"),
        "2019": (True, True, True, False, False, False, "explicit household last-visit date"),
        "2021": (True, True, True, False, False, False, "explicit household last-visit date"),
    }
    operational = {"2013", "2014", "2016"}
    rows = []
    for wave in WAVES:
        exact, collected, released, diary, measurement, village_ref, best = facts[wave]
        rows.append(
            {
                "survey_wave": wave,
                "exact_household_visit_date_in_released_data": exact,
                "household_cover_visit_date_collected": collected,
                "household_cover_visit_date_released": released,
                "daily_diary_date_in_released_data": diary,
                "child_measurement_date_in_released_data": measurement,
                "village_demographic_reference_date_in_released_data": village_ref,
                "operational_timestamp_present_but_rejected": wave in operational,
                "best_current_day_level_source": best,
                "eligible_for_survey_actual_day": exact,
                "decision": (
                    "May populate survey_actual_day from an explicit household visit date."
                    if exact
                    else "Keep survey_actual_day null; retain any proxy under its own semantic name."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    path: Path,
    diary_summary: pd.DataFrame,
    measurement_summary: pd.DataFrame,
    questionnaire: pd.DataFrame,
) -> None:
    lines = [
        "# CSES date-source forensic audit",
        "",
        "This audit is part of the database preparation layer. It does not connect to or write PostgreSQL.",
        "",
        "## Main finding",
        "",
        "The missing waves are not all devoid of day-level information. However, no newly found field",
        "is semantically equivalent to an explicit household first/last interview date. Therefore the",
        "current `survey_actual_day` rule remains unchanged.",
        "",
        "Questionnaire cover sheets confirm that first- and last-visit dates were collected for the",
        "audited 2007, 2009, 2011, 2013, 2014, and 2016 instruments, but those cover fields were omitted",
        "from the corresponding released Stata household tables. CSES 2017 has no questionnaire in the",
        "local archive, so paper-field collection remains unverified for that wave.",
        "",
        "## Diary dates",
        "",
        "| wave | final HH | HH with dates | coverage | median distinct dates | pair/month match | interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in diary_summary.itertuples(index=False):
        lines.append(
            f"| {row.survey_wave} | {row.final_households:,} | {row.households_with_diary_date:,} "
            f"| {row.household_coverage:.2%} | {row.median_distinct_diary_dates:.0f} "
            f"| {row.date_pair_matches_survey_month_rate:.2%} | activity window, not interview day |"
        )
    lines.extend(
        [
            "",
            "CSES 2009 and 2011 diaries normally cover the full month. CSES 2014 normally covers two",
            "executed weeks, which is consistent with the official survey report. These dates may support",
            "daily environmental exposure over an observed diary window, but cannot populate",
            "`survey_actual_day`.",
            "",
            "## Child measurement dates",
            "",
            "| wave | final HH | HH with dates | coverage | pair/month match | interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in measurement_summary.itertuples(index=False):
        lines.append(
            f"| {row.survey_wave} | {row.final_households:,} | {row.households_with_measurement_date:,} "
            f"| {row.household_coverage:.2%} | {row.date_pair_matches_survey_month_rate:.2%} "
            "| child-measurement field date; selected subset only |"
        )
    lines.extend(
        [
            "",
            "## Questionnaire evidence",
            "",
            f"Questionnaire evidence rows: {len(questionnaire):,} across "
            f"{questionnaire['survey_wave'].nunique()} waves.",
            "",
            "## Official sources used to resolve calendar years",
            "",
            "- CSES 2007: `https://microdata.nis.gov.kh/index.php/catalog/13/study-description`",
            "- CSES 2009: `https://microdata.nis.gov.kh/index.php/catalog/15`",
            "- CSES 2011: `https://microdata.nis.gov.kh/index.php/catalog/17`",
            "- CSES 2013: `https://microdata.nis.gov.kh/index.php/catalog/19`",
            "- CSES 2014: `https://microdata.nis.gov.kh/index.php/catalog/20`",
            "- CSES 2016: `https://microdata.nis.gov.kh/index.php/catalog/42`",
            "- CSES 2017 final report: `https://www.nis.gov.kh/nis/CSES/Final%20Report%20CSES%202017.pdf`",
            "",
            "## Outputs",
            "",
            "- `cses_date_source_forensic_matrix.csv`: one-row-per-wave decision matrix.",
            "- `cses_questionnaire_date_evidence.csv`: cover-sheet evidence and checksums.",
            "- `cses_diary_date_wave_summary.csv`: aggregate diary coverage and validation.",
            "- `cses_diary_date_household_candidates.parquet`: household diary windows, explicitly not interview dates.",
            "- `cses_child_measurement_date_wave_summary.csv`: subset field-date coverage.",
            "- `cses_child_measurement_date_household_candidates.parquet`: subset measurement-date candidates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--official-2007-questionnaire",
        type=Path,
        help=(
            "Optional official XLS downloaded from " + OFFICIAL_2007_QUESTIONNAIRE_URL
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)

    spine = pd.read_parquet(
        output / "final_HH_CSES.parquet",
        columns=["Survey Wave", "Household ID", "Survey Month"],
    )
    index = source_index(root)
    questionnaire = questionnaire_evidence(root, args.official_2007_questionnaire)
    diary_household, diary_summary = diary_candidates(root, spine, index)
    measurement_household, measurement_summary = measurement_candidates(
        root, spine, index
    )
    matrix = forensic_matrix(questionnaire)

    questionnaire.to_csv(output / "cses_questionnaire_date_evidence.csv", index=False)
    diary_summary.to_csv(output / "cses_diary_date_wave_summary.csv", index=False)
    diary_household.to_parquet(
        output / "cses_diary_date_household_candidates.parquet", index=False
    )
    measurement_summary.to_csv(
        output / "cses_child_measurement_date_wave_summary.csv", index=False
    )
    measurement_household.to_parquet(
        output / "cses_child_measurement_date_household_candidates.parquet", index=False
    )
    matrix.to_csv(output / "cses_date_source_forensic_matrix.csv", index=False)
    write_readme(
        output / "README_cses_date_forensics.md",
        diary_summary,
        measurement_summary,
        questionnaire,
    )

    print(f"questionnaire_evidence_rows={len(questionnaire)}")
    print(f"diary_household_rows={len(diary_household)}")
    print(f"measurement_household_rows={len(measurement_household)}")
    print("postgresql_writes=0")


if __name__ == "__main__":
    main()
