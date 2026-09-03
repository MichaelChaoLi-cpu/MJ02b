# CSES Database Infrastructure

This directory contains the reusable CSES database-build pipeline. It is
infrastructure preparation and is intentionally separate from this project's
research preprocessing, AnaSOP, analysis, figures, and tables.

## Boundaries

- Read-only source: `data/raw/CSE/`
- Generated database artifacts: `data/exp/database/`
- PostgreSQL target: `mda.public`
- No dependency on `docs/AnaSOP.md` or research-specific processed data

## Downstream consumption contract

After publication, PostgreSQL is the only supported source for downstream
research preprocessing, analysis, validation, figures, and tables. Downstream
code must query `mda.public`; it must not read the Parquet or CSV build artifacts
under `data/exp/database/` directly.

- Aligned observations: `final_HH_CSES`, `final_HL_CSES`, `final_ED_CSES`,
  `final_HO_CSES`, `final_EC_CSES`, `final_VL_CSES`, and
  `final_SURVEY_DATE_CSES`
- Variable dictionaries: `ind_que_*_CSES`
- Alignment summaries: `align_summary_*_CSES`
- Dataset discovery and row counts: `_catalog`

The files under `data/exp/database/` are reproducible staging artifacts used only
by this database-build pipeline and its pre-publication validators.

## Build order

The household-member table must precede education; education must precede the
enriched household table. The remaining layers consume HH and/or HL.

```bash
uv run python src/database/build_cses_hl.py
uv run python src/database/build_cses_ed.py
uv run python src/database/build_cses_hh.py
uv run python src/database/build_cses_ho.py
uv run python src/database/build_cses_ec.py
uv run python src/database/build_cses_vl.py
uv run python src/database/build_cses_survey_dates.py
```

## Validate and publish

```bash
uv run python src/database/validate_cses_hh_hl.py
uv run python src/database/validate_cses_ed.py
uv run python src/database/validate_cses_ho.py
uv run python src/database/validate_cses_ec.py
uv run python src/database/validate_cses_vl.py
uv run python src/database/publish_cses_to_postgres.py --replace
uv run python src/database/validate_cses_postgres.py
uv run python src/database/publish_cses_survey_dates.py --replace
uv run python src/database/validate_cses_survey_dates_postgres.py
```

Publication is transactional. It writes six `final_*` tables, six
`ind_que_*` variable dictionaries, six `align_summary_*` alignment summaries,
and the corresponding catalog metadata, comments, keys, and indexes.

The core publisher consumes the staged survey-date artifact when building HH,
so a future full core rebuild retains `survey_actual_year`,
`survey_actual_month`, and `survey_actual_day`. The scoped date publisher then
publishes the full timing/audit layer without modifying climate or analytical
tables.

## Survey-date extension

The date builder scans original CSES files for fields explicitly labeled as an
interview or visit date. It selects the interview date in 2004 and the last
visit date in 2019/2021. These three waves provide 35,105 household-level exact
dates. Other waves remain null rather than inheriting a nominal release year or
an operational file timestamp.

- `final_SURVEY_DATE_CSES`: household-grain dates, date roles, precision,
  provenance, and quality flags
- `ind_que_SURVEY_DATE_CSES`: survey-date variable dictionary
- `align_summary_SURVEY_DATE_CSES`: wave-level timing coverage and consistency
- `final_HH_CSES.survey_actual_year/month/day`: selected coherent date
  components for direct household joins

## Geography extension

The geography publisher downloads the official Cambodia commune boundary,
creates current admin2/admin3 dimensions, composes nationally unique CSES
administrative keys, audits current-boundary matches, and publishes the PSU
bridge. The downloaded boundary is build input only; downstream code reads the
published tables.

```bash
uv run python src/database/publish_cses_geography.py --replace
uv run python src/database/validate_cses_geography.py
```

- `dim_admin2_cambodia`: current district/municipality geometry and centroid
- `dim_admin3_cambodia`: current commune/sangkat geometry and centroid
- `dim_geo_CSES`: wave-PSU bridge, composed admin keys, survey month, match flags

The bridge does not pretend that a current boundary is a historical crosswalk.
Historical codes without a defensible current match remain explicitly
unmatched.

## Climate extension

The climate publisher queries district centroids from PostgreSQL, downloads
ERA5-Land and ERA5 daily fields through Open-Meteo, and transactionally
publishes daily observations, prespecified monthly heat treatments, a variable
dictionary, and an alignment summary. Network acquisition is separated from
publication: download-only mode closes PostgreSQL before making requests,
verifies every batch, persists a status CSV, and can safely resume from cache.

```bash
# Stage the primary heat exposure without a long-lived database transaction.
uv run python src/database/publish_heat_exposure.py \
  --download-only --full-survey-years --start-date 2007-01-01 \
  --end-date 2021-12-31 --batch-size 100 --minimal-exposure --skip-controls \
  --download-status-file data/exp/database/climate_download_status_primary.csv

# Stage precipitation and apparent temperature, reusing the primary cache.
uv run python src/database/publish_heat_exposure.py \
  --download-only --full-survey-years --start-date 2007-01-01 \
  --end-date 2021-12-31 --batch-size 100 --minimal-exposure \
  --download-status-file data/exp/database/climate_download_status_controls.csv

# Publish only complete cached survey years in one transaction.
uv run python src/database/publish_heat_exposure.py \
  --replace --full-survey-years --start-date 2007-01-01 \
  --end-date 2021-12-31 --batch-size 100 --minimal-exposure \
  --cached-complete-periods-only
uv run python src/database/validate_heat_exposure.py
```

- `final_CLIMATE_DAILY_ADMIN2`: district-day source fields
- `final_CLIMATE_MONTHLY_ADMIN2`: monthly absolute and local-percentile heat metrics
- `ind_que_CLIMATE_ADMIN2`: climate variable dictionary
- `align_summary_CLIMATE_ADMIN2`: coverage and derivation summary

The current release covers nine complete CSES survey years (2007, 2009, 2011,
2013, 2014, 2016, 2017, 2019, and 2021) and includes complete Tmax, WBmax,
precipitation, and apparent-temperature fields. The daily-mean temperature,
dew-point, and relative-humidity download is resumable but not published until
every batch is complete. Local 1991-2020 percentile/anomaly fields remain null
until at least 25 baseline years are available.

## Heat-labor analytic extension

The analytic publisher performs every join inside PostgreSQL and publishes the
working-age person layer used by the first humid-heat experiment. It does not
read database-build Parquet or CSV artifacts.

```bash
uv run python src/database/publish_heat_labor_analytic.py --replace
uv run python src/database/validate_heat_labor_analytic.py
```

- `final_HEAT_LABOR_ANALYTIC`: CSES EC outcomes, geography, monthly heat and
  rainfall/apparent-temperature controls, lag/lead placebo exposure, survey
  weights, and housing candidates
- `ind_que_HEAT_LABOR_ANALYTIC`: analytic variable dictionary
- `align_summary_HEAT_LABOR_ANALYTIC`: analytic alignment summary

Research analysis scripts are read-only. If provisional baseline result CSVs
must later be published, the separate
`publish_heat_labor_baseline_results.py` infrastructure script requires both
`--replace` (when targets exist) and the explicit
`--confirm-write WRITE_MDA_PUBLIC` token. It must not be invoked during normal
analysis or figure/table generation.
