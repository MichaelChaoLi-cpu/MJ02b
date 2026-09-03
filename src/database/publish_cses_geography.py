#!/usr/bin/env python3
"""Publish Cambodia administrative dimensions and CSES geography keys.

The official boundary file is used only by this database-build program.  Once
published, downstream research code must obtain geography from PostgreSQL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
import uuid
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import psycopg
import requests
from psycopg import sql
from shapely.geometry import mapping


DEFAULT_BOUNDARY_URL = (
    "https://data.mef.gov.kh/api/v1/public-datasets/"
    "pd_66c2f89949eb6c00013c1385/file"
)
SOURCE_PAGE = "https://data.mef.gov.kh/datasets/pd_66c2f89949eb6c00013c1385"
TARGETS = ["dim_admin2_cambodia", "dim_admin3_cambodia", "dim_geo_CSES"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--boundary-url", default=DEFAULT_BOUNDARY_URL)
    parser.add_argument(
        "--boundary-file",
        type=Path,
        help="Optional already-downloaded GeoJSON; useful for an offline reproducible run.",
    )
    parser.add_argument(
        "--boundary-sha256",
        help="Optional expected SHA-256; publication stops if the source differs.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def connection_args(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
    }
    if args.user:
        values["user"] = args.user
    if args.password:
        values["password"] = args.password
    return values


def download_boundary(url: str) -> tuple[bytes, str]:
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        payload = response.content
    except requests.exceptions.SSLError:
        # Some macOS/Python CA bundles reject this government endpoint even
        # when the system TLS store validates it. curl still verifies TLS.
        result = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", url],
            check=True,
            capture_output=True,
        )
        payload = result.stdout
    if not payload:
        raise RuntimeError("Boundary download was empty")
    return payload, hashlib.sha256(payload).hexdigest()


def load_boundary(args: argparse.Namespace) -> tuple[bytes, str]:
    if args.boundary_file:
        payload = args.boundary_file.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
    else:
        payload, sha256 = download_boundary(args.boundary_url)
    if args.boundary_sha256 and sha256.lower() != args.boundary_sha256.lower():
        raise RuntimeError(
            f"Boundary SHA-256 mismatch: observed={sha256}, "
            f"expected={args.boundary_sha256.lower()}"
        )
    return payload, sha256


def clean_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.removeprefix("KH")


def geometry_json(geometry: object) -> str:
    return json.dumps(mapping(geometry), separators=(",", ":"))


def prepare_dimensions(payload: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    with tempfile.NamedTemporaryFile(suffix=".geojson") as handle:
        handle.write(payload)
        handle.flush()
        communes = gpd.read_file(handle.name)

    required = {
        "ADM3_EN", "ADM3_PCODE", "ADM2_EN", "ADM2_PCODE",
        "ADM1_EN", "ADM1_PCODE", "date", "validOn", "geometry",
    }
    missing = required.difference(communes.columns)
    if missing:
        raise RuntimeError(f"Boundary source lacks required fields: {sorted(missing)}")
    if communes.crs is None:
        raise RuntimeError("Boundary source has no coordinate reference system")

    communes = communes[list(required)].copy()
    communes = communes.rename(
        columns={
            "ADM3_EN": "admin3_name",
            "ADM3_PCODE": "admin3_code",
            "ADM2_EN": "admin2_name",
            "ADM2_PCODE": "admin2_code",
            "ADM1_EN": "admin1_name",
            "ADM1_PCODE": "admin1_code",
            "date": "boundary_date",
            "validOn": "valid_on",
        }
    )
    for column in ["admin1_code", "admin2_code", "admin3_code"]:
        communes[column] = clean_code(communes[column])
    communes["boundary_date"] = pd.to_datetime(
        communes["boundary_date"], errors="coerce"
    ).dt.date
    communes["valid_on"] = pd.to_datetime(
        communes["valid_on"], errors="coerce"
    ).dt.date
    communes = communes.to_crs(4326)
    if communes["admin3_code"].duplicated().any():
        duplicates = communes.loc[
            communes["admin3_code"].duplicated(keep=False), "admin3_code"
        ].tolist()
        raise RuntimeError(f"Duplicate admin3 codes in source: {duplicates[:10]}")

    projected = communes.to_crs(32648)
    commune_centroids = gpd.GeoSeries(projected.geometry.centroid, crs=32648).to_crs(4326)
    admin3 = pd.DataFrame(
        {
            "admin3_code": communes["admin3_code"],
            "admin3_name": communes["admin3_name"],
            "admin2_code": communes["admin2_code"],
            "admin2_name": communes["admin2_name"],
            "admin1_code": communes["admin1_code"],
            "admin1_name": communes["admin1_name"],
            "centroid_lat": commune_centroids.y,
            "centroid_lon": commune_centroids.x,
            "boundary_date": communes["boundary_date"],
            "valid_on": communes["valid_on"],
            "geometry_geojson": communes.geometry.map(geometry_json),
            "source_url": SOURCE_PAGE,
        }
    ).sort_values("admin3_code", ignore_index=True)

    districts = communes.dissolve(
        by="admin2_code",
        aggfunc={
            "admin2_name": "first",
            "admin1_code": "first",
            "admin1_name": "first",
            "boundary_date": "max",
            "valid_on": "max",
            "admin3_code": "count",
        },
    ).rename(columns={"admin3_code": "admin3_count"}).reset_index()
    district_projected = districts.to_crs(32648)
    district_centroids = gpd.GeoSeries(
        district_projected.geometry.centroid, crs=32648
    ).to_crs(4326)
    admin2 = pd.DataFrame(
        {
            "admin2_code": districts["admin2_code"],
            "admin2_name": districts["admin2_name"],
            "admin1_code": districts["admin1_code"],
            "admin1_name": districts["admin1_name"],
            "admin3_count": districts["admin3_count"].astype("int64"),
            "centroid_lat": district_centroids.y,
            "centroid_lon": district_centroids.x,
            "boundary_date": districts["boundary_date"],
            "valid_on": districts["valid_on"],
            "geometry_geojson": districts.geometry.map(geometry_json),
            "source_url": SOURCE_PAGE,
        }
    ).sort_values("admin2_code", ignore_index=True)
    return admin2, admin3


ADMIN2_TYPES = [
    ("admin2_code", "text"),
    ("admin2_name", "text"),
    ("admin1_code", "text"),
    ("admin1_name", "text"),
    ("admin3_count", "integer"),
    ("centroid_lat", "double precision"),
    ("centroid_lon", "double precision"),
    ("boundary_date", "date"),
    ("valid_on", "date"),
    ("geometry_geojson", "jsonb"),
    ("source_url", "text"),
]
ADMIN3_TYPES = [
    ("admin3_code", "text"),
    ("admin3_name", "text"),
    ("admin2_code", "text"),
    ("admin2_name", "text"),
    ("admin1_code", "text"),
    ("admin1_name", "text"),
    ("centroid_lat", "double precision"),
    ("centroid_lon", "double precision"),
    ("boundary_date", "date"),
    ("valid_on", "date"),
    ("geometry_geojson", "jsonb"),
    ("source_url", "text"),
]


def python_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


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


def copy_frame(
    cursor: psycopg.Cursor,
    schema: str,
    table: str,
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
) -> None:
    expected = [name for name, _ in columns]
    if list(frame.columns) != expected:
        raise RuntimeError(f"Column mismatch for {table}: {list(frame.columns)}")
    statement = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, expected)),
    )
    with cursor.copy(statement) as copy:
        for row in frame.itertuples(index=False, name=None):
            copy.write_row(tuple(python_value(value) for value in row))


def table_exists(cursor: psycopg.Cursor, schema: str, table: str) -> bool:
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname=%s AND tablename=%s)",
        (schema, table),
    )
    return bool(cursor.fetchone()[0])


def create_geo_table(cursor: psycopg.Cursor, schema: str, table: str) -> None:
    cursor.execute(
        sql.SQL(
            """
            CREATE TABLE {}.{} AS
            WITH psu AS (
                SELECT
                    survey_wave,
                    min(survey_year)::smallint AS survey_year,
                    psu,
                    min(province_code) AS province_code_source,
                    min(district_code) AS district_code_source,
                    min(commune_code) AS commune_code_source,
                    min(village_code) AS village_code_source,
                    CASE WHEN count(DISTINCT survey_month) = 1
                         THEN min(survey_month) END::smallint AS survey_month,
                    count(*)::integer AS sample_household_count,
                    (
                        count(DISTINCT province_code) > 1
                        OR count(DISTINCT district_code) > 1
                        OR count(DISTINCT commune_code) > 1
                        OR count(DISTINCT village_code) > 1
                    ) AS source_code_conflict,
                    count(DISTINCT survey_month) > 1 AS survey_month_conflict
                FROM {}.{}
                GROUP BY survey_wave, psu
            ), keys AS (
                SELECT *,
                    CASE WHEN province_code_source IS NOT NULL
                         THEN lpad(province_code_source, 2, '0') END AS admin1_code,
                    CASE WHEN province_code_source IS NOT NULL
                               AND district_code_source IS NOT NULL
                         THEN lpad(province_code_source, 2, '0')
                              || right(lpad(district_code_source, 4, '0'), 2) END AS admin2_code,
                    CASE WHEN province_code_source IS NOT NULL
                               AND district_code_source IS NOT NULL
                               AND commune_code_source IS NOT NULL
                         THEN lpad(province_code_source, 2, '0')
                              || right(lpad(district_code_source, 4, '0'), 2)
                              || right(lpad(commune_code_source, 6, '0'), 2) END AS admin3_code,
                    CASE WHEN province_code_source IS NOT NULL
                               AND district_code_source IS NOT NULL
                               AND commune_code_source IS NOT NULL
                               AND village_code_source IS NOT NULL
                         THEN lpad(province_code_source, 2, '0')
                              || right(lpad(district_code_source, 4, '0'), 2)
                              || right(lpad(commune_code_source, 6, '0'), 2)
                              || right(lpad(village_code_source, 8, '0'), 2) END AS admin4_code
                FROM psu
            )
            SELECT
                k.survey_wave,
                k.survey_year,
                k.psu,
                k.province_code_source,
                k.district_code_source,
                k.commune_code_source,
                k.village_code_source,
                k.admin1_code,
                k.admin2_code,
                k.admin3_code,
                k.admin4_code,
                k.survey_month,
                k.sample_household_count,
                k.source_code_conflict,
                k.survey_month_conflict,
                (a2.admin2_code IS NOT NULL) AS admin2_current_match,
                (a3.admin3_code IS NOT NULL) AS admin3_current_match,
                CASE
                    WHEN a3.admin3_code IS NOT NULL THEN 'admin3_current'
                    WHEN a2.admin2_code IS NOT NULL THEN 'admin2_current'
                    ELSE 'unmatched_current_boundary'
                END::text AS geography_match_level,
                a2.centroid_lat AS admin2_centroid_lat,
                a2.centroid_lon AS admin2_centroid_lon
            FROM keys k
            LEFT JOIN {}.{} a2 USING (admin2_code)
            LEFT JOIN {}.{} a3 USING (admin3_code)
            """
        ).format(
            sql.Identifier(schema), sql.Identifier(table),
            sql.Identifier(schema), sql.Identifier("final_HH_CSES"),
            sql.Identifier(schema), sql.Identifier("dim_admin2_cambodia"),
            sql.Identifier(schema), sql.Identifier("dim_admin3_cambodia"),
        )
    )


def write_manifest(args: argparse.Namespace, sha256: str, admin2: int, admin3: int) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "exp" / "database" / "geography_source_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "retrieved_on", "source_page", "download_url", "sha256",
                "admin2_rows", "admin3_rows", "crs",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "retrieved_on": date.today().isoformat(),
                "source_page": SOURCE_PAGE,
                "download_url": args.boundary_url,
                "sha256": sha256,
                "admin2_rows": admin2,
                "admin3_rows": admin3,
                "crs": "EPSG:4326",
            }
        )


def publish(args: argparse.Namespace) -> None:
    payload, sha256 = load_boundary(args)
    admin2, admin3 = prepare_dimensions(payload)
    frames = {
        "dim_admin2_cambodia": (admin2, ADMIN2_TYPES),
        "dim_admin3_cambodia": (admin3, ADMIN3_TYPES),
    }
    with psycopg.connect(**connection_args(args)) as connection:
        with connection.cursor() as cursor:
            required = table_exists(cursor, args.schema, "final_HH_CSES")
            if not required:
                raise RuntimeError(f"Required table is missing: {args.schema}.final_HH_CSES")
            existing = [name for name in TARGETS if table_exists(cursor, args.schema, name)]
            if existing and not args.replace:
                raise RuntimeError(
                    f"Refusing to overwrite existing tables without --replace: {existing}"
                )

            stages: dict[str, str] = {}
            for target, (frame, columns) in frames.items():
                stage = f"_staging_{target}_{uuid.uuid4().hex[:8]}"
                stages[target] = stage
                create_table(cursor, args.schema, stage, columns)
                copy_frame(cursor, args.schema, stage, frame, columns)

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

            for target in ["dim_admin2_cambodia", "dim_admin3_cambodia"]:
                cursor.execute(
                    sql.SQL("ALTER TABLE {}.{} RENAME TO {}").format(
                        sql.Identifier(args.schema), sql.Identifier(stages[target]),
                        sql.Identifier(target),
                    )
                )
            create_geo_table(cursor, args.schema, "dim_geo_CSES")

            cursor.execute(
                sql.SQL("ALTER TABLE {}.{} ADD PRIMARY KEY (admin2_code)").format(
                    sql.Identifier(args.schema), sql.Identifier("dim_admin2_cambodia")
                )
            )
            cursor.execute(
                sql.SQL("ALTER TABLE {}.{} ADD PRIMARY KEY (admin3_code)").format(
                    sql.Identifier(args.schema), sql.Identifier("dim_admin3_cambodia")
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (admin2_code)").format(
                    sql.Identifier("ix_dim_admin3_cambodia_admin2"),
                    sql.Identifier(args.schema), sql.Identifier("dim_admin3_cambodia")
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX {} ON {}.{} (survey_wave, psu)").format(
                    sql.Identifier("uq_dim_geo_CSES_wave_psu"),
                    sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES")
                )
            )
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {}.{} (admin2_code, survey_year, survey_month)").format(
                    sql.Identifier("ix_dim_geo_CSES_admin2_month"),
                    sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES")
                )
            )

            comments = {
                "dim_admin2_cambodia": (
                    "Current Cambodia district/municipality boundary dimension derived "
                    "from the official commune boundary release; centroids are polygon "
                    "centroids computed in UTM zone 48N and stored in EPSG:4326."
                ),
                "dim_admin3_cambodia": (
                    "Current Cambodia commune/sangkat boundary dimension from the "
                    "official public boundary release."
                ),
                "dim_geo_CSES": (
                    "CSES PSU geography bridge. Administrative keys are deterministically "
                    "composed from released code components; current-boundary matches and "
                    "month/code conflicts are explicit."
                ),
            }
            for table, comment in comments.items():
                cursor.execute(
                    sql.SQL("COMMENT ON TABLE {}.{} IS {}").format(
                        sql.Identifier(args.schema), sql.Identifier(table), sql.Literal(comment)
                    )
                )

            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.{}
                        (table_name, module, grain, join_keys, ind_que_table,
                         description, caveats, row_count, n_datasets, updated_at)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE),
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE),
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier("_catalog")),
                (
                    "dim_admin2_cambodia", "GEO", "one row per current admin2",
                    "admin2_code", None,
                    "Cambodia current admin2 names, parent keys, centroids, and geometry.",
                    "A current boundary is not a historical crosswalk; historical CSES codes can remain unmatched.",
                    len(admin2), 1,
                    "dim_admin3_cambodia", "GEO", "one row per current admin3",
                    "admin3_code; parent via admin2_code", None,
                    "Cambodia current admin3 names, parent keys, centroids, and geometry.",
                    "A current boundary is not a historical crosswalk; geometry validity follows the source metadata.",
                    len(admin3), 1,
                    "dim_geo_CSES", "GEO", "one row per CSES wave and PSU",
                    "survey_wave + psu; climate via admin2_code + survey_year + survey_month", None,
                    "CSES hierarchical administrative keys and current-boundary match audit.",
                    "2004 lacks detailed geography/month; historical administrative changes cause some unmatched codes.",
                    0, 10,
                ),
            )
            cursor.execute(
                sql.SQL(
                    "UPDATE {}.{} SET row_count=(SELECT count(*) FROM {}.{}) "
                    "WHERE table_name='dim_geo_CSES'"
                ).format(
                    sql.Identifier(args.schema), sql.Identifier("_catalog"),
                    sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES"),
                )
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
            cursor.execute(
                sql.SQL(
                    """
                    SELECT survey_wave, count(*) AS psus,
                           count(*) FILTER (WHERE admin2_current_match) AS admin2_matches,
                           round(avg(admin2_current_match::int)::numeric, 4) AS admin2_rate,
                           count(*) FILTER (WHERE admin3_current_match) AS admin3_matches,
                           round(avg(admin3_current_match::int)::numeric, 4) AS admin3_rate
                    FROM {}.{}
                    WHERE admin2_code IS NOT NULL
                    GROUP BY survey_wave
                    ORDER BY survey_wave
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES"))
            )
            for row in cursor.fetchall():
                print("match=" + ",".join(map(str, row)))

    write_manifest(args, sha256, len(admin2), len(admin3))
    print(f"boundary_sha256={sha256}")


if __name__ == "__main__":
    publish(parse_args())
