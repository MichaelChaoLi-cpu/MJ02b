#!/usr/bin/env python3
"""Publish district-day and district-month heat exposure tables for CSES.

ERA5-Land supplies the primary air-temperature and humid-heat variables. ERA5
supplies precipitation and apparent temperature. Both are requested through
the Open-Meteo Historical Weather API at current admin2 polygon centroids.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import hashlib
import json
import time
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg
import requests
from psycopg import sql
from shapely.geometry import shape


API_URL = "https://archive-api.open-meteo.com/v1/archive"
API_DOCS = "https://open-meteo.com/en/docs/historical-weather-api"
TEMPERATURE_MODEL = "era5_land"
CONTROL_MODEL = "era5"
TIMEZONE = "Asia/Phnom_Penh"
TARGETS = [
    "final_CLIMATE_DAILY_ADMIN2",
    "final_CLIMATE_MONTHLY_ADMIN2",
    "ind_que_CLIMATE_ADMIN2",
    "align_summary_CLIMATE_ADMIN2",
]

DAILY_TYPES = [
    ("admin2_code", "text"),
    ("observation_date", "date"),
    ("reference_year", "smallint"),
    ("calendar_month", "smallint"),
    ("centroid_lat", "double precision"),
    ("centroid_lon", "double precision"),
    ("temperature_request_lat", "double precision"),
    ("temperature_request_lon", "double precision"),
    ("temperature_location_fallback_used", "boolean"),
    ("temperature_grid_lat", "double precision"),
    ("temperature_grid_lon", "double precision"),
    ("temperature_grid_elevation_m", "double precision"),
    ("control_grid_lat", "double precision"),
    ("control_grid_lon", "double precision"),
    ("control_grid_elevation_m", "double precision"),
    ("temperature_2m_max_c", "double precision"),
    ("temperature_2m_mean_c", "double precision"),
    ("wet_bulb_temperature_2m_max_c", "double precision"),
    ("dew_point_2m_max_c", "double precision"),
    ("relative_humidity_2m_max_pct", "double precision"),
    ("apparent_temperature_max_c", "double precision"),
    ("precipitation_sum_mm", "double precision"),
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
ALIGN_TYPES = [
    ("varname", "text"),
    ("dataset_count", "integer"),
    ("source_count", "integer"),
    ("explicit_count", "integer"),
    ("derived_count", "integer"),
    ("measure_type", "text"),
    ("canonical_text", "text"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--start-date", default="1991-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--climatology-start", default="1991-01-01")
    parser.add_argument("--climatology-end", default="2020-12-31")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--full-calendar",
        action="store_true",
        help="Download every day/location in the date range. Default: only CSES survey district-months.",
    )
    parser.add_argument(
        "--full-survey-years",
        action="store_true",
        help="Download complete survey years for admin2 areas sampled in each year.",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip the secondary ERA5 precipitation/apparent-temperature request.",
    )
    parser.add_argument(
        "--minimal-exposure",
        action="store_true",
        help="Request only daily Tmax and maximum wet-bulb temperature.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait before each uncached API request to respect rate limits.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/exp/database/cache/open_meteo"),
        help="Build-only compressed response cache for resumable publication.",
    )
    parser.add_argument(
        "--cached-complete-periods-only",
        action="store_true",
        help="Publish only task periods whose every response is already cached; make no network requests.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help=(
            "Populate and verify the response cache without creating PostgreSQL "
            "staging tables. Database connections are closed before network requests."
        ),
    )
    parser.add_argument(
        "--download-status-file",
        type=Path,
        default=Path("data/exp/database/climate_download_status.csv"),
        help="Resumable per-batch cache status written during --download-only runs.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def connection_args(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {
        "host": args.host, "port": args.port, "dbname": args.dbname
    }
    if args.user:
        values["user"] = args.user
    if args.password:
        values["password"] = args.password
    return values


def table_exists(cursor: psycopg.Cursor, schema: str, table: str) -> bool:
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname=%s AND tablename=%s)",
        (schema, table),
    )
    return bool(cursor.fetchone()[0])


def load_locations(cursor: psycopg.Cursor, schema: str) -> list[dict[str, object]]:
    cursor.execute(
        sql.SQL(
            """
            SELECT admin2_code, centroid_lat, centroid_lon, geometry_geojson
            FROM {}.{}
            WHERE centroid_lat IS NOT NULL AND centroid_lon IS NOT NULL
            ORDER BY admin2_code
            """
        ).format(sql.Identifier(schema), sql.Identifier("dim_admin2_cambodia"))
    )
    rows = [
        {
            "admin2_code": code,
            "centroid_lat": float(lat),
            "centroid_lon": float(lon),
            "land_fallback_lat": float(shape(geometry).representative_point().y),
            "land_fallback_lon": float(shape(geometry).representative_point().x),
        }
        for code, lat, lon, geometry in cursor.fetchall()
    ]
    if not rows:
        raise RuntimeError(f"No locations found in {schema}.dim_admin2_cambodia")
    return rows


def load_survey_month_tasks(
    cursor: psycopg.Cursor,
    schema: str,
    start_date: str,
    end_date: str,
) -> list[tuple[str, str, list[dict[str, object]]]]:
    cursor.execute(
        sql.SQL(
            """
            SELECT DISTINCT
                d.admin2_code,
                d.centroid_lat,
                d.centroid_lon,
                d.geometry_geojson,
                h.survey_year,
                h.survey_month
            FROM {}.{} h
            JOIN {}.{} g USING (survey_wave, psu)
            JOIN {}.{} d USING (admin2_code)
            WHERE g.admin2_current_match
              AND h.survey_month BETWEEN 1 AND 12
              AND make_date(h.survey_year, h.survey_month, 1)
                    BETWEEN date_trunc('month', %s::date)
                        AND date_trunc('month', %s::date)
            ORDER BY h.survey_year, h.survey_month, d.admin2_code
            """
        ).format(
            sql.Identifier(schema), sql.Identifier("final_HH_CSES"),
            sql.Identifier(schema), sql.Identifier("dim_geo_CSES"),
            sql.Identifier(schema), sql.Identifier("dim_admin2_cambodia"),
        ),
        (start_date, end_date),
    )
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for code, lat, lon, geometry, year, month in cursor.fetchall():
        representative = shape(geometry).representative_point()
        grouped.setdefault((int(year), int(month)), []).append(
            {
                "admin2_code": code,
                "centroid_lat": float(lat),
                "centroid_lon": float(lon),
                "land_fallback_lat": float(representative.y),
                "land_fallback_lon": float(representative.x),
            }
        )
    tasks = []
    for (year, month), locations in grouped.items():
        last_day = calendar.monthrange(year, month)[1]
        tasks.append(
            (f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}", locations)
        )
    if not tasks:
        raise RuntimeError("No eligible CSES district-month tasks found in PostgreSQL")
    return tasks


def load_survey_year_tasks(
    cursor: psycopg.Cursor,
    schema: str,
    start_date: str,
    end_date: str,
) -> list[tuple[str, str, list[dict[str, object]]]]:
    cursor.execute(
        sql.SQL(
            """
            SELECT DISTINCT
                d.admin2_code, d.centroid_lat, d.centroid_lon,
                d.geometry_geojson, h.survey_year
            FROM {}.{} h
            JOIN {}.{} g USING (survey_wave, psu)
            JOIN {}.{} d USING (admin2_code)
            WHERE g.admin2_current_match
              AND h.survey_month BETWEEN 1 AND 12
              AND h.survey_year BETWEEN extract(year FROM %s::date)
                                    AND extract(year FROM %s::date)
            ORDER BY h.survey_year, d.admin2_code
            """
        ).format(
            sql.Identifier(schema), sql.Identifier("final_HH_CSES"),
            sql.Identifier(schema), sql.Identifier("dim_geo_CSES"),
            sql.Identifier(schema), sql.Identifier("dim_admin2_cambodia"),
        ),
        (start_date, end_date),
    )
    grouped: dict[int, list[dict[str, object]]] = {}
    for code, lat, lon, geometry, year in cursor.fetchall():
        representative = shape(geometry).representative_point()
        grouped.setdefault(int(year), []).append(
            {
                "admin2_code": code,
                "centroid_lat": float(lat),
                "centroid_lon": float(lon),
                "land_fallback_lat": float(representative.y),
                "land_fallback_lon": float(representative.x),
            }
        )
    tasks = [
        (f"{year:04d}-01-01", f"{year:04d}-12-31", locations)
        for year, locations in grouped.items()
    ]
    if not tasks:
        raise RuntimeError("No eligible CSES survey-year tasks found in PostgreSQL")
    return tasks


def cache_path(cache_dir: Path, params: dict[str, str]) -> Path:
    fingerprint = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    model = params["models"]
    return cache_dir / model / f"{params['start_date']}_{params['end_date']}_{fingerprint}.json.gz"


def split_task_groups(
    task_groups: list[tuple[str, str, list[dict[str, object]]]],
    batch_size: int,
) -> list[tuple[str, str, list[dict[str, object]]]]:
    """Split periods deterministically so cache fingerprints remain reproducible."""
    return [
        (task_start, task_end, task_locations[index:index + batch_size])
        for task_start, task_end, task_locations in task_groups
        for index in range(0, len(task_locations), batch_size)
    ]


def required_cache_paths(
    args: argparse.Namespace,
    task_start: str,
    task_end: str,
    batch: list[dict[str, object]],
) -> list[Path]:
    paths = [
        cache_path(
            args.cache_dir,
            temperature_params(batch, task_start, task_end, args.minimal_exposure),
        )
    ]
    if not args.skip_controls:
        paths.append(
            cache_path(
                args.cache_dir,
                control_params(batch, task_start, task_end),
            )
        )
    return paths


def request_json(params: dict[str, str], cache_dir: Path, request_delay: float) -> object:
    path = cache_path(cache_dir, params)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            if request_delay > 0:
                time.sleep(min(request_delay, 55.0))
            response = requests.get(API_URL, params=params, timeout=(30, 300))
            if response.status_code == 429:
                last_error = RuntimeError("HTTP 429 Too Many Requests")
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after) if retry_after else 45.0, 55.0)
                if attempt == 8:
                    print(
                        f"rate_limited model={params['models']} attempt={attempt}/8 "
                        "retry_limit_reached=true",
                        flush=True,
                    )
                    break
                print(
                    f"rate_limited model={params['models']} attempt={attempt}/8 "
                    f"waiting_seconds={delay:g}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            temporary.replace(path)
            return payload
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt == 8:
                break
            delay = min(2 ** attempt, 45)
            print(
                f"request_retry model={params['models']} attempt={attempt}/8 "
                f"waiting_seconds={delay}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Open-Meteo request failed after retries: {last_error}")


def normalized_responses(payload: object, expected: int) -> list[dict[str, object]]:
    rows = payload if isinstance(payload, list) else [payload]
    if len(rows) != expected:
        raise RuntimeError(f"API location count={len(rows)}, expected={expected}")
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("API returned an unexpected response structure")
    return rows


def temperature_params(
    locations: list[dict[str, object]],
    start_date: str,
    end_date: str,
    minimal_exposure: bool,
) -> dict[str, str]:
    temperature_variables = (
        "temperature_2m_max,wet_bulb_temperature_2m_max"
        if minimal_exposure
        else (
            "temperature_2m_max,temperature_2m_mean,"
            "wet_bulb_temperature_2m_max,dew_point_2m_max,"
            "relative_humidity_2m_max"
        )
    )
    return {
        "latitude": ",".join(str(row["centroid_lat"]) for row in locations),
        "longitude": ",".join(str(row["centroid_lon"]) for row in locations),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
        "models": TEMPERATURE_MODEL,
        "daily": temperature_variables,
    }


def control_params(
    locations: list[dict[str, object]], start_date: str, end_date: str
) -> dict[str, str]:
    return {
        "latitude": ",".join(str(row["centroid_lat"]) for row in locations),
        "longitude": ",".join(str(row["centroid_lon"]) for row in locations),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
        "models": CONTROL_MODEL,
        "daily": "apparent_temperature_max,precipitation_sum",
    }


def fetch_batch(
    locations: list[dict[str, object]],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    include_controls: bool,
    minimal_exposure: bool,
    request_delay: float,
) -> pd.DataFrame:
    temperature_request = temperature_params(
        locations, start_date, end_date, minimal_exposure
    )
    temperature_payload = request_json(
        temperature_request,
        cache_dir,
        request_delay,
    )
    temperature_rows = normalized_responses(temperature_payload, len(locations))
    for index, (location, temperature) in enumerate(
        zip(locations, temperature_rows, strict=True)
    ):
        tmax = temperature.get("daily", {}).get("temperature_2m_max", [])
        fallback_used = bool(tmax) and all(value is None for value in tmax)
        if fallback_used:
            fallback_location = {
                "admin2_code": location["admin2_code"],
                "centroid_lat": location["land_fallback_lat"],
                "centroid_lon": location["land_fallback_lon"],
            }
            fallback_payload = request_json(
                temperature_params(
                    [fallback_location], start_date, end_date, minimal_exposure
                ),
                cache_dir,
                request_delay,
            )
            temperature = normalized_responses(fallback_payload, 1)[0]
            fallback_tmax = temperature.get("daily", {}).get(
                "temperature_2m_max", []
            )
            if not fallback_tmax or all(value is None for value in fallback_tmax):
                raise RuntimeError(
                    f"No land-grid temperature after representative-point fallback "
                    f"for admin2={location['admin2_code']}"
                )
            temperature_rows[index] = temperature
            print(
                f"land_grid_fallback={location['admin2_code']} "
                f"lat={location['land_fallback_lat']:.6f} "
                f"lon={location['land_fallback_lon']:.6f}",
                flush=True,
            )
        temperature_rows[index]["_request_lat"] = (
            location["land_fallback_lat"] if fallback_used else location["centroid_lat"]
        )
        temperature_rows[index]["_request_lon"] = (
            location["land_fallback_lon"] if fallback_used else location["centroid_lon"]
        )
        temperature_rows[index]["_fallback_used"] = fallback_used
    if include_controls:
        control_payload = request_json(
            control_params(locations, start_date, end_date),
            cache_dir,
            request_delay,
        )
        control_rows = normalized_responses(control_payload, len(locations))
    else:
        control_rows = [None] * len(locations)

    frames: list[pd.DataFrame] = []
    for location, temperature, control in zip(
        locations, temperature_rows, control_rows, strict=True
    ):
        temperature_daily = temperature.get("daily", {})
        control_daily = control.get("daily", {}) if control else {}
        days = temperature_daily.get("time", [])
        if control and days != control_daily.get("time", []):
            raise RuntimeError(f"Model dates differ for admin2={location['admin2_code']}")
        if not days:
            raise RuntimeError(f"No daily data for admin2={location['admin2_code']}")
        dates = pd.to_datetime(days)
        frame = pd.DataFrame(
            {
                "admin2_code": location["admin2_code"],
                "observation_date": dates.date,
                "reference_year": dates.year.astype("int16"),
                "calendar_month": dates.month.astype("int8"),
                "centroid_lat": location["centroid_lat"],
                "centroid_lon": location["centroid_lon"],
                "temperature_request_lat": temperature.get("_request_lat"),
                "temperature_request_lon": temperature.get("_request_lon"),
                "temperature_location_fallback_used": temperature.get(
                    "_fallback_used", False
                ),
                "temperature_grid_lat": temperature.get("latitude"),
                "temperature_grid_lon": temperature.get("longitude"),
                "temperature_grid_elevation_m": temperature.get("elevation"),
                "control_grid_lat": control.get("latitude") if control else None,
                "control_grid_lon": control.get("longitude") if control else None,
                "control_grid_elevation_m": control.get("elevation") if control else None,
                "temperature_2m_max_c": temperature_daily.get("temperature_2m_max"),
                "temperature_2m_mean_c": (
                    temperature_daily.get("temperature_2m_mean")
                    if not minimal_exposure else [None] * len(days)
                ),
                "wet_bulb_temperature_2m_max_c": temperature_daily.get(
                    "wet_bulb_temperature_2m_max"
                ),
                "dew_point_2m_max_c": (
                    temperature_daily.get("dew_point_2m_max")
                    if not minimal_exposure else [None] * len(days)
                ),
                "relative_humidity_2m_max_pct": (
                    temperature_daily.get("relative_humidity_2m_max")
                    if not minimal_exposure else [None] * len(days)
                ),
                "apparent_temperature_max_c": (
                    control_daily.get("apparent_temperature_max")
                    if control else [None] * len(days)
                ),
                "precipitation_sum_mm": (
                    control_daily.get("precipitation_sum")
                    if control else [None] * len(days)
                ),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def create_table(
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
    columns: list[tuple[str, str]],
) -> None:
    definitions = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(data_type))
        for name, data_type in columns
    )
    cursor.execute(
        sql.SQL("CREATE TABLE {}.{} ({})").format(
            sql.Identifier(schema), sql.Identifier(table), definitions
        )
    )


def python_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def copy_frame(
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
) -> None:
    names = [name for name, _ in columns]
    if list(frame.columns) != names:
        raise RuntimeError(f"Column mismatch for {table}: {list(frame.columns)}")
    statement = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema), sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, names)),
    )
    with cursor.copy(statement) as copy:
        for row in frame.itertuples(index=False, name=None):
            copy.write_row(tuple(python_value(value) for value in row))


def create_monthly_table(
    cursor: psycopg.Cursor,
    schema: str,
    daily_table: str,
    monthly_table: str,
    climatology_start: str,
    climatology_end: str,
) -> None:
    cursor.execute(
        sql.SQL(
            """
            CREATE TABLE {}.{} AS
            WITH climatology AS (
                SELECT
                    admin2_code,
                    calendar_month,
                    avg(temperature_2m_max_c)::double precision AS baseline_tmax_mean_c,
                    avg(wet_bulb_temperature_2m_max_c)::double precision AS baseline_wbmax_mean_c,
                    count(DISTINCT reference_year)::smallint AS climatology_year_count,
                    percentile_cont(0.90) WITHIN GROUP
                        (ORDER BY temperature_2m_max_c)::double precision AS local_month_tmax_p90_c,
                    percentile_cont(0.95) WITHIN GROUP
                        (ORDER BY temperature_2m_max_c)::double precision AS local_month_tmax_p95_c,
                    percentile_cont(0.90) WITHIN GROUP
                        (ORDER BY wet_bulb_temperature_2m_max_c)::double precision AS local_month_wbmax_p90_c,
                    percentile_cont(0.95) WITHIN GROUP
                        (ORDER BY wet_bulb_temperature_2m_max_c)::double precision AS local_month_wbmax_p95_c
                FROM {}.{}
                WHERE observation_date BETWEEN %s::date AND %s::date
                GROUP BY admin2_code, calendar_month
            )
            SELECT
                d.admin2_code,
                date_trunc('month', d.observation_date)::date AS month_start,
                d.reference_year,
                d.calendar_month,
                count(*)::smallint AS days_observed,
                extract(day FROM
                    (date_trunc('month', d.observation_date)
                     + interval '1 month - 1 day'))::smallint AS days_expected,
                (count(*) = extract(day FROM
                    (date_trunc('month', d.observation_date)
                     + interval '1 month - 1 day'))) AS complete_month,
                bool_or(d.temperature_location_fallback_used)
                    AS temperature_location_fallback_used,
                avg(d.temperature_2m_max_c)::double precision AS temperature_2m_max_mean_c,
                max(d.temperature_2m_max_c)::double precision AS temperature_2m_month_max_c,
                avg(d.temperature_2m_mean_c)::double precision AS temperature_2m_mean_c,
                avg(d.wet_bulb_temperature_2m_max_c)::double precision AS wet_bulb_max_mean_c,
                max(d.wet_bulb_temperature_2m_max_c)::double precision AS wet_bulb_month_max_c,
                avg(d.dew_point_2m_max_c)::double precision AS dew_point_2m_max_mean_c,
                avg(d.relative_humidity_2m_max_pct)::double precision
                    AS relative_humidity_2m_max_mean_pct,
                avg(d.apparent_temperature_max_c)::double precision
                    AS apparent_temperature_max_mean_c,
                max(d.apparent_temperature_max_c)::double precision AS apparent_temperature_month_max_c,
                sum(d.precipitation_sum_mm)::double precision AS precipitation_month_sum_mm,
                count(*) FILTER (WHERE d.temperature_2m_max_c >= 35)::smallint AS days_tmax_ge_35c,
                count(*) FILTER (WHERE d.temperature_2m_max_c >= 37)::smallint AS days_tmax_ge_37c,
                count(*) FILTER (WHERE d.temperature_2m_max_c >= 40)::smallint AS days_tmax_ge_40c,
                count(*) FILTER (WHERE d.wet_bulb_temperature_2m_max_c >= 26)::smallint AS days_wbmax_ge_26c,
                count(*) FILTER (WHERE d.wet_bulb_temperature_2m_max_c >= 28)::smallint AS days_wbmax_ge_28c,
                count(*) FILTER (WHERE d.wet_bulb_temperature_2m_max_c >= 30)::smallint AS days_wbmax_ge_30c,
                CASE WHEN c.climatology_year_count >= 25 THEN
                    count(*) FILTER (WHERE d.temperature_2m_max_c > c.local_month_tmax_p90_c)::smallint
                END AS days_tmax_above_local_month_p90,
                CASE WHEN c.climatology_year_count >= 25 THEN
                    count(*) FILTER (WHERE d.temperature_2m_max_c > c.local_month_tmax_p95_c)::smallint
                END AS days_tmax_above_local_month_p95,
                CASE WHEN c.climatology_year_count >= 25 THEN
                    count(*) FILTER (WHERE d.wet_bulb_temperature_2m_max_c > c.local_month_wbmax_p90_c)::smallint
                END AS days_wbmax_above_local_month_p90,
                CASE WHEN c.climatology_year_count >= 25 THEN
                    count(*) FILTER (WHERE d.wet_bulb_temperature_2m_max_c > c.local_month_wbmax_p95_c)::smallint
                END AS days_wbmax_above_local_month_p95,
                c.climatology_year_count,
                (c.climatology_year_count >= 25) AS climatology_complete,
                CASE WHEN c.climatology_year_count >= 25 THEN c.baseline_tmax_mean_c END
                    AS baseline_tmax_mean_c,
                CASE WHEN c.climatology_year_count >= 25 THEN c.baseline_wbmax_mean_c END
                    AS baseline_wbmax_mean_c,
                CASE WHEN c.climatology_year_count >= 25 THEN c.local_month_tmax_p90_c END
                    AS local_month_tmax_p90_c,
                CASE WHEN c.climatology_year_count >= 25 THEN c.local_month_tmax_p95_c END
                    AS local_month_tmax_p95_c,
                CASE WHEN c.climatology_year_count >= 25 THEN c.local_month_wbmax_p90_c END
                    AS local_month_wbmax_p90_c,
                CASE WHEN c.climatology_year_count >= 25 THEN c.local_month_wbmax_p95_c END
                    AS local_month_wbmax_p95_c,
                CASE WHEN c.climatology_year_count >= 25 THEN
                    (avg(d.temperature_2m_max_c) - c.baseline_tmax_mean_c)::double precision
                END AS temperature_2m_max_anomaly_c,
                CASE WHEN c.climatology_year_count >= 25 THEN
                    (avg(d.wet_bulb_temperature_2m_max_c) - c.baseline_wbmax_mean_c)::double precision
                END AS wet_bulb_max_anomaly_c
            FROM {}.{} d
            JOIN climatology c USING (admin2_code, calendar_month)
            GROUP BY d.admin2_code, date_trunc('month', d.observation_date),
                     d.reference_year, d.calendar_month,
                     c.baseline_tmax_mean_c, c.baseline_wbmax_mean_c,
                     c.climatology_year_count,
                     c.local_month_tmax_p90_c, c.local_month_tmax_p95_c,
                     c.local_month_wbmax_p90_c, c.local_month_wbmax_p95_c
            ORDER BY d.admin2_code, month_start
            """
        ).format(
            sql.Identifier(schema), sql.Identifier(monthly_table),
            sql.Identifier(schema), sql.Identifier(daily_table),
            sql.Identifier(schema), sql.Identifier(daily_table),
        ),
        (climatology_start, climatology_end),
    )


VARIABLES = [
    ("admin2_code", "daily and monthly", "database key", "identifier", "Current four-digit district/municipality code."),
    ("observation_date", "daily", "API date", "time", "Local calendar date in Asia/Phnom_Penh."),
    ("temperature_2m_max_c", "daily", "temperature_2m_max", "continuous", "ERA5-Land daily maximum 2 m air temperature in degrees Celsius."),
    ("temperature_2m_mean_c", "daily", "temperature_2m_mean", "continuous", "ERA5-Land daily mean 2 m air temperature in degrees Celsius."),
    ("wet_bulb_temperature_2m_max_c", "daily", "wet_bulb_temperature_2m_max", "continuous", "ERA5-Land daily maximum 2 m wet-bulb temperature in degrees Celsius."),
    ("dew_point_2m_max_c", "daily", "dew_point_2m_max", "continuous", "ERA5-Land daily maximum 2 m dew-point temperature in degrees Celsius."),
    ("relative_humidity_2m_max_pct", "daily", "relative_humidity_2m_max", "continuous", "ERA5-Land daily maximum 2 m relative humidity in percent."),
    ("apparent_temperature_max_c", "daily", "apparent_temperature_max", "continuous", "ERA5 daily maximum apparent temperature in degrees Celsius."),
    ("precipitation_sum_mm", "daily", "precipitation_sum", "continuous", "ERA5 daily precipitation sum in millimeters."),
    ("dew_point_2m_max_mean_c", "monthly", "derived", "continuous", "Monthly mean of ERA5-Land daily maximum 2 m dew-point temperature in degrees Celsius."),
    ("relative_humidity_2m_max_mean_pct", "monthly", "derived", "continuous", "Monthly mean of ERA5-Land daily maximum 2 m relative humidity in percent."),
    ("apparent_temperature_max_mean_c", "monthly", "derived", "continuous", "Monthly mean of ERA5 daily maximum apparent temperature in degrees Celsius."),
    ("apparent_temperature_month_max_c", "monthly", "derived", "continuous", "Maximum ERA5 daily apparent temperature during the calendar month in degrees Celsius."),
    ("precipitation_month_sum_mm", "monthly", "derived", "continuous", "Sum of ERA5 daily precipitation during the calendar month in millimeters."),
    ("days_wbmax_ge_26c", "monthly", "derived", "count", "Days in the month with daily maximum wet-bulb temperature at least 26 C."),
    ("days_wbmax_ge_28c", "monthly", "derived", "count", "Days in the month with daily maximum wet-bulb temperature at least 28 C."),
    ("days_wbmax_ge_30c", "monthly", "derived", "count", "Days in the month with daily maximum wet-bulb temperature at least 30 C."),
    ("days_tmax_above_local_month_p95", "monthly", "derived", "count", "Days exceeding the admin2-calendar-month 1991-2020 daily Tmax P95 threshold."),
    ("days_wbmax_above_local_month_p95", "monthly", "derived", "count", "Days exceeding the admin2-calendar-month 1991-2020 daily WBmax P95 threshold."),
    ("temperature_2m_max_anomaly_c", "monthly", "derived", "continuous", "Monthly mean daily Tmax minus the admin2-calendar-month 1991-2020 mean."),
    ("wet_bulb_max_anomaly_c", "monthly", "derived", "continuous", "Monthly mean daily WBmax minus the admin2-calendar-month 1991-2020 mean."),
]


def dictionary_frames(n_admin2: int, n_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dictionary = pd.DataFrame(
        [
            {
                "canonical_varname": name,
                "dataset_name": dataset,
                "column_in_raw_sav": source,
                "column_label_in_english": description,
                "source_kind": "derived" if source == "derived" else "explicit",
                "measure_type": measure,
                "canonical_text": description,
            }
            for name, dataset, source, measure, description in VARIABLES
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "varname": name,
                "dataset_count": n_years,
                "source_count": n_admin2,
                "explicit_count": 0 if source == "derived" else n_admin2,
                "derived_count": n_admin2 if source == "derived" else 0,
                "measure_type": measure,
                "canonical_text": description,
            }
            for name, _dataset, source, measure, description in VARIABLES
        ]
    )
    return dictionary, summary


def write_build_artifacts(
    args: argparse.Namespace,
    n_admin2: int,
    daily_rows: int,
    monthly_rows: int,
    dictionary: pd.DataFrame,
    summary: pd.DataFrame,
    published_years: list[int],
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "exp" / "database"
    output.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(output / "ind_que_CLIMATE_ADMIN2.csv", index=False)
    summary.to_csv(output / "align_summary_CLIMATE_ADMIN2.csv", index=False)
    with (output / "climate_source_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "retrieved_on", "api_docs", "temperature_model", "control_model",
                "timezone", "start_date", "end_date", "climatology_start",
                "climatology_end", "admin2_rows", "daily_rows", "monthly_rows",
                "coverage_mode", "controls_included",
                "minimal_exposure",
                "published_years",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "retrieved_on": date.today().isoformat(),
                "api_docs": API_DOCS,
                "temperature_model": TEMPERATURE_MODEL,
                "control_model": CONTROL_MODEL,
                "timezone": TIMEZONE,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "climatology_start": args.climatology_start,
                "climatology_end": args.climatology_end,
                "admin2_rows": n_admin2,
                "daily_rows": daily_rows,
                "monthly_rows": monthly_rows,
                "coverage_mode": (
                    "full_calendar" if args.full_calendar
                    else "cses_full_survey_years" if args.full_survey_years
                    else "cses_survey_months"
                ),
                "controls_included": not args.skip_controls,
                "minimal_exposure": args.minimal_exposure,
                "published_years": ";".join(map(str, published_years)),
            }
        )


def load_task_definition(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[tuple[str, str, list[dict[str, object]]]]]:
    """Read locations/tasks in a short-lived transaction before network work."""
    with psycopg.connect(**connection_args(args)) as connection:
        with connection.cursor() as cursor:
            if not table_exists(cursor, args.schema, "dim_admin2_cambodia"):
                raise RuntimeError("Run publish_cses_geography.py first")
            locations = load_locations(cursor, args.schema)
            if args.full_calendar:
                task_groups = [(args.start_date, args.end_date, locations)]
            elif args.full_survey_years:
                task_groups = load_survey_year_tasks(
                    cursor, args.schema, args.start_date, args.end_date
                )
            else:
                task_groups = load_survey_month_tasks(
                    cursor, args.schema, args.start_date, args.end_date
                )
    return locations, task_groups


def write_download_status(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def download_to_cache(
    args: argparse.Namespace,
    task_groups: list[tuple[str, str, list[dict[str, object]]]],
) -> None:
    """Download batches without holding a PostgreSQL connection or write lock."""
    batches = split_task_groups(task_groups, args.batch_size)
    records: list[dict[str, object]] = []
    for batch_number, (task_start, task_end, batch) in enumerate(batches, start=1):
        paths = required_cache_paths(args, task_start, task_end, batch)
        records.append(
            {
                "run_date": date.today().isoformat(),
                "period_start": task_start,
                "period_end": task_end,
                "batch_number": batch_number,
                "batch_count": len(batches),
                "admin2_count": len(batch),
                "admin2_first": batch[0]["admin2_code"],
                "admin2_last": batch[-1]["admin2_code"],
                "temperature_cache": str(paths[0]),
                "control_cache": str(paths[1]) if len(paths) == 2 else "",
                "cached_before": all(path.exists() for path in paths),
                "status": "pending",
                "daily_rows_verified": None,
            }
        )
    write_download_status(args.download_status_file, records)

    # Deliberately process batches sequentially. This makes rate-limit behavior
    # predictable and guarantees a failed request cannot leave queued requests
    # consuming quota after the command exits.
    if args.workers != 1:
        print(
            "download_only_forces_sequential=true requested_workers="
            f"{args.workers}",
            flush=True,
        )
    for index, (task_start, task_end, batch) in enumerate(batches):
        record = records[index]
        try:
            frame = fetch_batch(
                batch,
                task_start,
                task_end,
                args.cache_dir,
                not args.skip_controls,
                args.minimal_exposure,
                args.request_delay,
            )
            expected_rows = len(batch) * len(
                pd.date_range(task_start, task_end, freq="D")
            )
            if len(frame) != expected_rows:
                raise RuntimeError(
                    f"Batch rows={len(frame)}, expected={expected_rows}"
                )
            paths = required_cache_paths(args, task_start, task_end, batch)
            if not all(path.exists() for path in paths):
                raise RuntimeError("A verified response was not persisted to cache")
            record["status"] = "cached" if record["cached_before"] else "downloaded"
            record["daily_rows_verified"] = len(frame)
            write_download_status(args.download_status_file, records)
            print(
                f"cached_batches={index + 1}/{len(batches)} "
                f"period={task_start[:4]} admin2={len(batch)} "
                f"rows_verified={len(frame)} status={record['status']}",
                flush=True,
            )
        except Exception:
            record["status"] = "failed"
            write_download_status(args.download_status_file, records)
            raise

    print(
        f"download_complete batches={len(batches)} "
        f"status_file={args.download_status_file}",
        flush=True,
    )


def publish(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.workers < 1 or args.request_delay < 0:
        raise ValueError("batch size/workers must be positive and request delay nonnegative")
    if args.full_calendar and args.full_survey_years:
        raise ValueError("Choose only one of --full-calendar and --full-survey-years")
    if args.download_only and args.cached_complete_periods_only:
        raise ValueError(
            "--download-only and --cached-complete-periods-only are mutually exclusive"
        )
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start > end:
        raise ValueError("start date must not exceed end date")
    if args.download_only:
        _locations, task_groups = load_task_definition(args)
        download_to_cache(args, task_groups)
        return
    with psycopg.connect(**connection_args(args)) as connection:
        with connection.cursor() as cursor:
            if not table_exists(cursor, args.schema, "dim_admin2_cambodia"):
                raise RuntimeError("Run publish_cses_geography.py first")
            locations = load_locations(cursor, args.schema)
            if args.full_calendar:
                task_groups = [(args.start_date, args.end_date, locations)]
            elif args.full_survey_years:
                task_groups = load_survey_year_tasks(
                    cursor, args.schema, args.start_date, args.end_date
                )
            else:
                task_groups = load_survey_month_tasks(
                    cursor, args.schema, args.start_date, args.end_date
                )
            existing = [name for name in TARGETS if table_exists(cursor, args.schema, name)]
            if existing and not args.replace:
                raise RuntimeError(
                    f"Refusing to overwrite existing tables without --replace: {existing}"
                )
            daily_stage = f"_staging_final_CLIMATE_DAILY_ADMIN2_{uuid.uuid4().hex[:8]}"
            monthly_stage = f"_staging_final_CLIMATE_MONTHLY_ADMIN2_{uuid.uuid4().hex[:8]}"
            dictionary_stage = f"_staging_ind_que_CLIMATE_ADMIN2_{uuid.uuid4().hex[:8]}"
            summary_stage = f"_staging_align_summary_CLIMATE_ADMIN2_{uuid.uuid4().hex[:8]}"
            create_table(cursor, args.schema, daily_stage, DAILY_TYPES)

            batches: list[tuple[str, str, list[dict[str, object]]]] = []
            selected_task_groups: list[
                tuple[str, str, list[dict[str, object]]]
            ] = []
            for task_start, task_end, task_locations in task_groups:
                period_batches = split_task_groups(
                    [(task_start, task_end, task_locations)], args.batch_size
                )
                if args.cached_complete_periods_only:
                    complete = all(
                        cache_path(
                            args.cache_dir,
                            temperature_params(
                                batch, period_start, period_end, args.minimal_exposure
                            ),
                        ).exists()
                        and (
                            args.skip_controls
                            or cache_path(
                                args.cache_dir,
                                control_params(batch, period_start, period_end),
                            ).exists()
                        )
                        for period_start, period_end, batch in period_batches
                    )
                    if not complete:
                        print(f"skipped_uncached_period={task_start}_{task_end}", flush=True)
                        continue
                selected_task_groups.append((task_start, task_end, task_locations))
                batches.extend(period_batches)
            if not batches:
                raise RuntimeError("No complete cached periods are available for publication")
            task_groups = selected_task_groups
            expected_rows = sum(
                len(batch) * len(pd.date_range(task_start, task_end, freq="D"))
                for task_start, task_end, batch in batches
            )
            copied_rows = 0
            if args.workers != 1:
                print(
                    "publication_forces_sequential=true requested_workers="
                    f"{args.workers}",
                    flush=True,
                )
            for completed, (task_start, task_end, batch) in enumerate(
                batches, start=1
            ):
                frame = fetch_batch(
                    batch,
                    task_start,
                    task_end,
                    args.cache_dir,
                    not args.skip_controls,
                    args.minimal_exposure,
                    args.request_delay,
                )
                expected_batch_rows = len(batch) * len(
                    pd.date_range(task_start, task_end, freq="D")
                )
                if len(frame) != expected_batch_rows:
                    raise RuntimeError(
                        f"Batch rows={len(frame)}, expected={expected_batch_rows}"
                    )
                copy_frame(cursor, args.schema, daily_stage, frame, DAILY_TYPES)
                copied_rows += len(frame)
                print(
                    f"processed_batches={completed}/{len(batches)} "
                    f"period={task_start[:7]} admin2={len(batch)} "
                    f"cumulative_rows={copied_rows}",
                    flush=True,
                )

            if copied_rows != expected_rows:
                raise RuntimeError(f"Daily rows={copied_rows}, expected={expected_rows}")
            create_monthly_table(
                cursor, args.schema, daily_stage, monthly_stage,
                args.climatology_start, args.climatology_end,
            )
            cursor.execute(
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(args.schema), sql.Identifier(monthly_stage)
                )
            )
            monthly_rows = int(cursor.fetchone()[0])

            published_years = sorted(
                set(pd.to_datetime([group[0] for group in task_groups]).year.tolist())
            )
            n_years = len(
                published_years
            )
            dictionary, summary = dictionary_frames(len(locations), n_years)
            create_table(cursor, args.schema, dictionary_stage, IND_QUE_TYPES)
            create_table(cursor, args.schema, summary_stage, ALIGN_TYPES)
            copy_frame(cursor, args.schema, dictionary_stage, dictionary, IND_QUE_TYPES)
            copy_frame(cursor, args.schema, summary_stage, summary, ALIGN_TYPES)

            if args.replace:
                cursor.execute(
                    sql.SQL("DELETE FROM {}.{} WHERE table_name = ANY(%s)").format(
                        sql.Identifier(args.schema), sql.Identifier("_catalog")
                    ),
                    (TARGETS,),
                )
                for target in reversed(TARGETS):
                    cursor.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                            sql.Identifier(args.schema), sql.Identifier(target)
                        )
                    )

            for stage, target in [
                (daily_stage, TARGETS[0]),
                (monthly_stage, TARGETS[1]),
                (dictionary_stage, TARGETS[2]),
                (summary_stage, TARGETS[3]),
            ]:
                cursor.execute(
                    sql.SQL("ALTER TABLE {}.{} RENAME TO {}").format(
                        sql.Identifier(args.schema), sql.Identifier(stage), sql.Identifier(target)
                    )
                )

            cursor.execute(
                sql.SQL("ALTER TABLE {}.{} ADD PRIMARY KEY (admin2_code, observation_date)").format(
                    sql.Identifier(args.schema), sql.Identifier(TARGETS[0])
                )
            )
            cursor.execute(
                sql.SQL("ALTER TABLE {}.{} ADD PRIMARY KEY (admin2_code, month_start)").format(
                    sql.Identifier(args.schema), sql.Identifier(TARGETS[1])
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (reference_year, calendar_month, admin2_code)").format(
                    sql.Identifier("ix_final_CLIMATE_MONTHLY_ADMIN2_year_month"),
                    sql.Identifier(args.schema), sql.Identifier(TARGETS[1])
                )
            )

            table_comments = {
                TARGETS[0]: (
                    "Daily admin2-centroid weather for CSES survey district-months: "
                    "primary humid heat and air temperature from ERA5-Land; optional "
                    "precipitation and apparent temperature from ERA5 via Open-Meteo."
                ),
                TARGETS[1]: (
                    "Monthly admin2 heat treatments and controls derived from the daily "
                    "table using prespecified absolute thresholds. Local calendar-month "
                    "climatology fields remain null until at least 25 baseline years exist."
                ),
            }
            for table, comment in table_comments.items():
                cursor.execute(
                    sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                        sql.Identifier(args.schema), sql.Identifier(table), sql.Literal(comment)
                    )
                )

            catalog_rows = [
                (
                    TARGETS[0], "CLIMATE", "one row per admin2 and local calendar day",
                    "admin2_code + observation_date", TARGETS[2],
                    "Daily ERA5-Land humid heat and air temperature with optional ERA5 weather controls.",
                    "Coverage is limited to CSES survey district-month combinations unless --full-calendar is used. Minimal mode leaves secondary temperature and weather-control columns null. Centroid exposure represents an admin2 area; gridded reanalysis is not station observation or satellite LST.",
                    copied_rows, n_years,
                ),
                (
                    TARGETS[1], "CLIMATE", "one row per admin2 and calendar month",
                    "admin2_code + reference_year + calendar_month", TARGETS[2],
                    "Monthly extreme-heat treatments and available weather controls for CSES survey district-months.",
                    "Local climatology-derived fields are null until 25+ baseline years are published. CSES has survey month but no aligned interview day, so exposure is temporally coarse for past-week labor outcomes.",
                    monthly_rows, n_years,
                ),
                (
                    TARGETS[2], "CLIMATE", "one row per documented climate variable",
                    "canonical_varname + dataset_name", None,
                    "Variable dictionary for the daily and monthly admin2 climate tables.",
                    "Source API labels are retained only as provenance; research code uses canonical variable names.",
                    len(dictionary), 2,
                ),
                (
                    TARGETS[3], "CLIMATE", "one row per aligned climate variable",
                    "varname", TARGETS[2],
                    "Coverage and derivation summary for aligned climate variables.",
                    "Counts describe source/derived admin2 coverage rather than CSES survey-wave alignment.",
                    len(summary), 2,
                ),
            ]
            cursor.executemany(
                sql.SQL(
                    """
                    INSERT INTO {}.{}
                        (table_name, module, grain, join_keys, ind_que_table,
                         description, caveats, row_count, n_datasets, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier("_catalog")),
                catalog_rows,
            )
        connection.commit()

        with connection.cursor() as cursor:
            for target in TARGETS:
                cursor.execute(
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier(target)
                    )
                )
                print(f"published={args.schema}.{target} rows={cursor.fetchone()[0]}")

    write_build_artifacts(
        args, len(locations), copied_rows, monthly_rows, dictionary, summary,
        published_years,
    )


if __name__ == "__main__":
    publish(parse_args())
