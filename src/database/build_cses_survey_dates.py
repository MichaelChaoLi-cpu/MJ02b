#!/usr/bin/env python3
"""Build a household-grain CSES survey-date staging dataset.

This is database preparation, not research preprocessing. It reads original
CSES archives and the locally staged household linkage spine, then writes the
``final_*``, ``ind_que_*``, and ``align_summary_*`` artifacts expected by the
project's database layer. It never connects to PostgreSQL.

Exact dates are retained only when the source explicitly labels the components
as interview/visit dates. Operational ``ChangeDate`` and ``SignOut`` values are
excluded. A candidate reference date is supplied for audit and future exposure
construction, but this script does not use it in an analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_cses_interview_dates import construct_date, read_stata
from inventory_cses_archives import discover_sources, normalize_wave, source_leaf, token


WAVES = [
    "2004", "2007", "2009", "2011-12", "2013",
    "2014", "2016", "2017", "2019", "2021",
]
ID_WIDTH = {wave: (6 if wave == "2004" else 7) for wave in WAVES}

SOURCE_LEAF_BY_WAVE = {
    "2004": "2004hhs99singlequestions",
    "2019": "s0117hhotherinfo",
    "2021": "s0117hhotherinfo",
}


VARIABLES = {
    "Dataset Name": (
        "CSES Survey-Date Dataset Name", "identifier",
        "CSES survey-date staging dataset plus normalized release wave.",
    ),
    "Survey Wave": (
        "CSES Survey Wave", "identifier", "Normalized CSES release wave.",
    ),
    "Nominal Survey Year": (
        "Nominal Survey Release Year", "time",
        "First year encoded by the release-wave name; not necessarily the actual interview year.",
    ),
    "Household ID": (
        "Household Identifier", "identifier", "Wave-normalized household identifier.",
    ),
    "Survey Month": (
        "Released Survey Month", "time",
        "Month retained from the household linkage spine; may be the only available timing field.",
    ),
    "Released Survey Year": (
        "Released Survey Year", "time",
        "Household-level year from a raw field explicitly labeled Year of Survey; released in 2019 and 2021.",
    ),
    "Released Survey Month": (
        "Released Survey Month from Exact-Date Source", "time",
        "Household-level month from the same raw other-information source as Year of Survey; retained for linkage validation.",
    ),
    "Interview Date": (
        "Explicit Interview Date", "time",
        "Date constructed only from day/month/year fields explicitly labeled as the interview date.",
    ),
    "First Visit Date": (
        "Explicit First Visit Date", "time",
        "Date constructed only from fields explicitly labeled as the first household visit.",
    ),
    "Last Visit Date": (
        "Explicit Last Visit Date", "time",
        "Date constructed only from fields explicitly labeled as the last household visit.",
    ),
    "Reinterview Date": (
        "Explicit Re-interview Date", "time",
        "Quality-control re-interview date where released; it is not the main survey date.",
    ),
    "Candidate Reference Date": (
        "Candidate Survey Reference Date", "time",
        "Transparent candidate for future exposure alignment: interview date in 2004 and last visit in 2019/2021. It has not been adopted as the final analytical anchor.",
    ),
    "Candidate Reference Definition": (
        "Candidate Reference-Date Definition", "provenance",
        "Source role used to construct Candidate Reference Date.",
    ),
    "Survey Actual Year": (
        "Actual Household Survey Year", "time",
        "Calendar year of the selected explicit household survey date: interview date in 2004 and last-visit date in 2019/2021; null otherwise.",
    ),
    "Survey Actual Month": (
        "Actual Household Survey Month", "time",
        "Calendar month of the selected explicit household survey date: interview date in 2004 and last-visit date in 2019/2021; null otherwise.",
    ),
    "Survey Actual Day": (
        "Actual Household Survey Day", "time",
        "Calendar day of the selected explicit household survey date: interview date in 2004 and last-visit date in 2019/2021; null otherwise.",
    ),
    "Date Precision": (
        "Survey-Date Precision", "time",
        "Exact day where a confirmed date is available, month where only Survey Month is available, otherwise unavailable.",
    ),
    "Candidate Actual Year": (
        "Candidate Actual Survey Year", "time",
        "Calendar year of Candidate Reference Date.",
    ),
    "Candidate Actual Month": (
        "Candidate Actual Survey Month", "time",
        "Calendar month of Candidate Reference Date.",
    ),
    "Confirmed Survey Year": (
        "Confirmed Household Survey Year", "time",
        "Raw Year of Survey in 2019/2021, or year of the explicitly labeled interview date in 2004; null when no household-level year is confirmed.",
    ),
    "Confirmed Survey Month": (
        "Confirmed Household Survey Month", "time",
        "Raw Month of Survey in 2019/2021, or month of the explicitly labeled interview date in 2004; null when unavailable.",
    ),
    "Confirmed Survey Time Source": (
        "Confirmed Survey-Time Source", "provenance",
        "Field definition supplying the confirmed household survey year/month.",
    ),
    "Confirmed Year Differs": (
        "Confirmed Survey Year Differs from Nominal Year", "quality flag",
        "1 when Confirmed Survey Year differs from Nominal Survey Year, 0 when equal, null without a confirmed year.",
    ),
    "Nominal Year Differs": (
        "Candidate Date Differs from Nominal Year", "quality flag",
        "1 when Candidate Actual Year differs from Nominal Survey Year, 0 when equal, null without an exact candidate date.",
    ),
    "Survey Month Matches Candidate": (
        "Survey Month Matches Candidate Date", "quality flag",
        "1 when released Survey Month equals Candidate Actual Month, 0 when they differ, null when either is unavailable.",
    ),
    "Candidate Date Within Documented Period": (
        "Candidate Date Within Documented Fieldwork Period", "quality flag",
        "For 2004, 1 when Candidate Reference Date is within November 2003 through January 2005 and 0 otherwise; null where no exact documented period check is encoded.",
    ),
    "Exact Date Source Archive": (
        "Exact-Date Source Archive", "provenance", "Original raw archive containing the exact date fields.",
    ),
    "Exact Date Source Submodule": (
        "Exact-Date Source Submodule", "provenance", "Stata member containing the exact date fields.",
    ),
}


def clean_household_id(series: pd.Series, wave: str) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    values = values.mask(values.str.lower().isin(["", "nan", "none", "<na>"]))
    digits = values.str.fullmatch(r"\d+", na=False)
    return values.where(~digits, values.str.zfill(ID_WIDTH[wave]))


def find_exact_sources(root: Path) -> dict[str, object]:
    found: dict[str, list[object]] = {wave: [] for wave in SOURCE_LEAF_BY_WAVE}
    for source in discover_sources(root):
        wave = normalize_wave(source.display_name(root))
        if wave not in found:
            continue
        if token(Path(source_leaf(source)).stem) == SOURCE_LEAF_BY_WAVE[wave]:
            found[wave].append(source)
    bad = {wave: len(sources) for wave, sources in found.items() if len(sources) != 1}
    if bad:
        raise RuntimeError(f"Expected one exact-date source per wave; found {bad}")
    return {wave: sources[0] for wave, sources in found.items()}


def exact_date_piece(root: Path, wave: str, source: object) -> pd.DataFrame:
    raw, _labels = read_stata(source)
    if "hhid" not in raw.columns:
        raise RuntimeError(f"Exact-date source for {wave} lacks hhid")
    piece = pd.DataFrame({"Household ID": clean_household_id(raw["hhid"], wave)})
    piece["Interview Date"] = pd.NaT
    piece["First Visit Date"] = pd.NaT
    piece["Last Visit Date"] = pd.NaT
    piece["Reinterview Date"] = pd.NaT
    piece["Released Survey Year"] = pd.Series(pd.NA, index=piece.index, dtype="Int16")
    piece["Released Survey Month"] = pd.Series(pd.NA, index=piece.index, dtype="Int8")

    if wave == "2004":
        piece["Interview Date"] = construct_date(
            raw, ("q40_intd", "q40_intm", "q40_inty"), year_offset=2000
        )
        piece["Last Visit Date"] = construct_date(
            raw, ("q40_lasd", "q40_lasm", "q40_lasy"), year_offset=2000
        )
        piece["Reinterview Date"] = construct_date(
            raw, ("q40_ared", "q40_arem", "q40_arey"), year_offset=2000
        )
        piece["Candidate Reference Date"] = piece["Interview Date"]
        piece["Candidate Reference Definition"] = "explicit interview date"
    elif wave == "2019":
        piece["Last Visit Date"] = construct_date(raw, ("day", "month", "year"))
        piece["Candidate Reference Date"] = piece["Last Visit Date"]
        piece["Candidate Reference Definition"] = "explicit last-visit date"
        piece["Released Survey Year"] = pd.to_numeric(raw["yearsur"], errors="coerce").astype("Int16")
        piece["Released Survey Month"] = pd.to_numeric(raw["monthsur"], errors="coerce").astype("Int8")
    elif wave == "2021":
        piece["First Visit Date"] = construct_date(
            raw, ("day_1st", "month_1st", "year_1st")
        )
        piece["Last Visit Date"] = construct_date(
            raw, ("day_2nd", "month_2nd", "year_2nd")
        )
        piece["Candidate Reference Date"] = piece["Last Visit Date"]
        piece["Candidate Reference Definition"] = "explicit last-visit date"
        piece["Released Survey Year"] = pd.to_numeric(raw["yearsur"], errors="coerce").astype("Int16")
        piece["Released Survey Month"] = pd.to_numeric(raw["monthsur"], errors="coerce").astype("Int8")
    else:  # Defensive guard for future edits.
        raise RuntimeError(f"No exact-date construction for {wave}")

    piece["Exact Date Source Archive"] = str(source.root_file.relative_to(root))
    piece["Exact Date Source Submodule"] = "::".join(source.archive_members)
    if piece["Household ID"].isna().any():
        raise RuntimeError(f"Exact-date source for {wave} has missing household IDs")
    if piece.duplicated("Household ID").any():
        raise RuntimeError(f"Exact-date source for {wave} has duplicate household IDs")
    return piece


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hh_path = root / "data" / "exp" / "database" / "final_HH_CSES.parquet"
    hh = pd.read_parquet(
        hh_path,
        columns=["Survey Wave", "Survey Year", "Household ID", "Survey Month"],
    ).rename(columns={"Survey Year": "Nominal Survey Year"})
    if hh.duplicated(["Survey Wave", "Household ID"]).any():
        raise RuntimeError("Household linkage spine is not unique by wave-household")
    hh["Dataset Name"] = "CSES " + hh["Survey Wave"].astype("string") + " survey dates"

    sources = find_exact_sources(root)
    pieces = []
    for wave, source in sources.items():
        piece = exact_date_piece(root, wave, source)
        piece["Survey Wave"] = wave
        pieces.append(piece)
    exact = pd.concat(pieces, ignore_index=True)
    result = hh.merge(
        exact,
        on=["Survey Wave", "Household ID"],
        how="left",
        validate="1:1",
    )

    result["Candidate Actual Year"] = result["Candidate Reference Date"].dt.year.astype("Int16")
    result["Candidate Actual Month"] = result["Candidate Reference Date"].dt.month.astype("Int8")
    result["Survey Actual Year"] = result["Candidate Reference Date"].dt.year.astype("Int16")
    result["Survey Actual Month"] = result["Candidate Reference Date"].dt.month.astype("Int8")
    result["Survey Actual Day"] = result["Candidate Reference Date"].dt.day.astype("Int8")
    result["Confirmed Survey Year"] = result["Released Survey Year"].copy()
    result["Confirmed Survey Month"] = result["Released Survey Month"].copy()
    result["Confirmed Survey Time Source"] = pd.Series(pd.NA, index=result.index, dtype="string")
    released_time = result["Released Survey Year"].notna() & result["Released Survey Month"].notna()
    result.loc[released_time, "Confirmed Survey Time Source"] = "raw Year of Survey + Month of Survey"
    interview_time = result["Survey Wave"].eq("2004") & result["Interview Date"].notna()
    result.loc[interview_time, "Confirmed Survey Year"] = result.loc[interview_time, "Interview Date"].dt.year.astype("Int16")
    result.loc[interview_time, "Confirmed Survey Month"] = result.loc[interview_time, "Interview Date"].dt.month.astype("Int8")
    result.loc[interview_time, "Confirmed Survey Time Source"] = "raw explicit interview date"
    exact_available = result["Candidate Reference Date"].notna()
    result["Nominal Year Differs"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[exact_available, "Nominal Year Differs"] = (
        result.loc[exact_available, "Candidate Actual Year"]
        .ne(result.loc[exact_available, "Nominal Survey Year"])
        .astype("int8")
    )
    month_comparable = result["Survey Month"].notna() & result["Candidate Actual Month"].notna()
    result["Survey Month Matches Candidate"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[month_comparable, "Survey Month Matches Candidate"] = (
        result.loc[month_comparable, "Survey Month"]
        .eq(result.loc[month_comparable, "Candidate Actual Month"])
        .astype("int8")
    )
    confirmed_year = result["Confirmed Survey Year"].notna()
    result["Confirmed Year Differs"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[confirmed_year, "Confirmed Year Differs"] = (
        result.loc[confirmed_year, "Confirmed Survey Year"]
        .ne(result.loc[confirmed_year, "Nominal Survey Year"])
        .astype("int8")
    )
    result["Date Precision"] = "unavailable"
    result.loc[result["Survey Month"].notna(), "Date Precision"] = "month"
    result.loc[exact_available, "Date Precision"] = "day"
    result["Candidate Date Within Documented Period"] = pd.Series(
        pd.NA, index=result.index, dtype="Int8"
    )
    documented_2004 = result["Survey Wave"].eq("2004") & exact_available
    result.loc[documented_2004, "Candidate Date Within Documented Period"] = (
        result.loc[documented_2004, "Candidate Reference Date"]
        .between(pd.Timestamp("2003-11-01"), pd.Timestamp("2005-01-31"))
        .astype("int8")
    )

    columns = list(VARIABLES)
    result = result[columns].sort_values(
        ["Nominal Survey Year", "Household ID"], ignore_index=True
    )

    dictionary_rows = []
    raw_by_variable = {
        "Interview Date": "q40_intd + q40_intm + q40_inty",
        "First Visit Date": "day_1st + month_1st + year_1st",
        "Last Visit Date": "q40_lasd/q40_lasm/q40_lasy; day/month/year; day_2nd/month_2nd/year_2nd",
        "Reinterview Date": "q40_ared + q40_arem + q40_arey",
        "Candidate Reference Date": "derived from retained explicit date fields",
        "Survey Actual Year": "derived from retained explicit date fields",
        "Survey Actual Month": "derived from retained explicit date fields",
        "Survey Actual Day": "derived from retained explicit date fields",
        "Candidate Actual Year": "derived",
        "Candidate Actual Month": "derived",
        "Confirmed Survey Year": "derived from explicitly labeled raw survey-time fields",
        "Confirmed Survey Month": "derived from explicitly labeled raw survey-time fields",
        "Confirmed Survey Time Source": "derived",
        "Confirmed Year Differs": "derived",
        "Nominal Year Differs": "derived",
        "Survey Month Matches Candidate": "derived",
        "Candidate Date Within Documented Period": "derived",
        "Date Precision": "derived",
    }
    for variable, (readable, measure, description) in VARIABLES.items():
        dictionary_rows.append({
            "varname": variable,
            "readable_name": readable,
            "dataset_name": "SURVEY_DATE_CSES",
            "column_in_raw_sav": raw_by_variable.get(variable, variable),
            "canonical_text": description,
            "source_kind": "derived" if variable in raw_by_variable and raw_by_variable[variable] == "derived" else "explicit_or_derived",
            "measure_type": measure,
        })
    dictionary = pd.DataFrame(dictionary_rows)

    summary_rows = []
    for wave in WAVES:
        part = result.loc[result["Survey Wave"].eq(wave)]
        exact_part = part["Candidate Reference Date"].notna()
        comparable = part["Survey Month Matches Candidate"].notna()
        summary_rows.append({
            "dataset_name": "SURVEY_DATE_CSES",
            "survey_wave": wave,
            "rows": len(part),
            "unique_households": int(part["Household ID"].nunique()),
            "survey_month_available": int(part["Survey Month"].notna().sum()),
            "interview_date_available": int(part["Interview Date"].notna().sum()),
            "first_visit_date_available": int(part["First Visit Date"].notna().sum()),
            "last_visit_date_available": int(part["Last Visit Date"].notna().sum()),
            "candidate_reference_date_available": int(exact_part.sum()),
            "candidate_reference_date_coverage": float(exact_part.mean()) if len(part) else 0.0,
            "candidate_minimum_date": part["Candidate Reference Date"].min(),
            "candidate_maximum_date": part["Candidate Reference Date"].max(),
            "nominal_year_differs": int(part["Nominal Year Differs"].eq(1).sum()),
            "nominal_year_differs_rate_among_exact": float(part.loc[exact_part, "Nominal Year Differs"].mean()) if exact_part.any() else 0.0,
            "released_survey_year_available": int(part["Released Survey Year"].notna().sum()),
            "confirmed_survey_year_available": int(part["Confirmed Survey Year"].notna().sum()),
            "confirmed_year_differs": int(part["Confirmed Year Differs"].eq(1).sum()),
            "confirmed_year_differs_rate": float(part.loc[part["Confirmed Year Differs"].notna(), "Confirmed Year Differs"].mean()) if part["Confirmed Year Differs"].notna().any() else 0.0,
            "survey_month_comparable": int(comparable.sum()),
            "survey_month_disagrees": int(part["Survey Month Matches Candidate"].eq(0).sum()),
            "survey_month_disagrees_rate": float(part.loc[comparable, "Survey Month Matches Candidate"].eq(0).mean()) if comparable.any() else 0.0,
            "candidate_dates_outside_documented_period": int(
                part["Candidate Date Within Documented Period"].eq(0).sum()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    return result, dictionary, summary


def write_readme(path: Path, final: pd.DataFrame, summary: pd.DataFrame) -> None:
    by_wave = summary.set_index("survey_wave")
    raw_years = (
        final.dropna(subset=["Released Survey Year"])
        .groupby(["Survey Wave", "Released Survey Year"])
        .size()
    )
    year_2019 = {
        int(year): int(count) for year, count in raw_years.loc["2019"].items()
    }
    year_2021 = {
        int(year): int(count) for year, count in raw_years.loc["2021"].items()
    }
    lines = [
        "# CSES household survey-date staging layer",
        "",
        "Status: reproducible database staging; publish with `src/database/publish_cses_survey_dates.py`.",
        "",
        "## Contract",
        "",
        "- Grain: one row per `Survey Wave` and `Household ID`.",
        "- Raw ZIP and nested-ZIP members are read in memory and never modified.",
        "- `ChangeDate` and `SignOut` operational timestamps are excluded.",
        "- `Candidate Reference Date` is retained for audit; it has not been",
        "  adopted as the final heat-exposure anchor.",
        "",
        "## Coverage and timing findings",
        "",
        f"- Total rows: {len(final):,}; duplicate household keys: {int(final.duplicated(['Survey Wave', 'Household ID']).sum())}.",
        f"- 2004: {int(by_wave.loc['2004', 'interview_date_available']):,} explicit interview dates; {int(by_wave.loc['2004', 'candidate_dates_outside_documented_period']):,} fall outside the report's November 2003-January 2005 fieldwork period.",
        f"- 2019: {int(by_wave.loc['2019', 'last_visit_date_available']):,} explicit last-visit dates; raw Year of Survey counts are {year_2019}.",
        f"- 2021: {int(by_wave.loc['2021', 'last_visit_date_available']):,} first/last-visit dates; raw Year of Survey counts are {year_2021}.",
        "- 2007, 2009, 2011-12, 2013, 2014, 2016, and 2017 have released",
        "  survey month but no confirmed household-level exact date in the scanned files.",
        "- Official NIS metadata identifies the 2011-12 archive's microdata as CSES 2011",
        "  with fieldwork from January 1 through December 29, 2011. The bundled 2012",
        "  field manual is documentation for a later round, not evidence of 2012 records.",
        "- Separate forensic outputs retain 2009/2011/2014 diary activity windows and",
        "  2007/2009 child measurement dates under their own semantic names; neither",
        "  source is relabeled as a household interview day.",
        "",
        "## Artifacts",
        "",
        "- `final_SURVEY_DATE_CSES.parquet`: household-grain staged dates and flags.",
        "- `ind_que_SURVEY_DATE_CSES.csv`: variable dictionary.",
        "- `align_summary_SURVEY_DATE_CSES.csv`: wave-level coverage and quality summary.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)
    final, dictionary, summary = build(root)
    final.to_parquet(output / "final_SURVEY_DATE_CSES.parquet", index=False)
    dictionary.to_csv(output / "ind_que_SURVEY_DATE_CSES.csv", index=False)
    summary.to_csv(output / "align_summary_SURVEY_DATE_CSES.csv", index=False)
    write_readme(output / "README_SURVEY_DATE_CSES.md", final, summary)
    print(summary.to_string(index=False))
    print(f"output=data/exp/database/final_SURVEY_DATE_CSES.parquet rows={len(final)}")


if __name__ == "__main__":
    main()
