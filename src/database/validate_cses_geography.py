#!/usr/bin/env python3
"""Validate published Cambodia dimensions and the CSES geography bridge."""

from __future__ import annotations

import argparse

import psycopg
from psycopg import sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scalar(cursor: psycopg.Cursor, statement: object, parameters: tuple = ()) -> object:
    cursor.execute(statement, parameters)
    return cursor.fetchone()[0]


def main() -> None:
    args = parse_args()
    connection_args: dict[str, object] = {
        "host": args.host, "port": args.port, "dbname": args.dbname
    }
    if args.user:
        connection_args["user"] = args.user
    if args.password:
        connection_args["password"] = args.password

    with psycopg.connect(**connection_args) as connection:
        with connection.cursor() as cursor:
            counts = {}
            for table in ["dim_admin2_cambodia", "dim_admin3_cambodia", "dim_geo_CSES"]:
                counts[table] = int(scalar(
                    cursor,
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(args.schema), sql.Identifier(table)
                    ),
                ))
                require(counts[table] > 0, f"{table} is empty")
            require(counts["dim_admin3_cambodia"] == 1_633,
                    f"Expected 1,633 current communes, got {counts['dim_admin3_cambodia']}")

            invalid_dimensions = int(scalar(
                cursor,
                sql.SQL(
                    """
                    SELECT count(*) FROM {}.{}
                    WHERE admin2_code !~ '^[0-9]{{4}}$'
                       OR admin1_code !~ '^[0-9]{{2}}$'
                       OR centroid_lat NOT BETWEEN 9 AND 15
                       OR centroid_lon NOT BETWEEN 102 AND 108
                       OR geometry_geojson IS NULL
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier("dim_admin2_cambodia")),
            ))
            require(invalid_dimensions == 0, f"Invalid admin2 rows={invalid_dimensions}")

            duplicates = int(scalar(
                cursor,
                sql.SQL(
                    """
                    SELECT count(*) FROM (
                        SELECT survey_wave, psu, count(*)
                        FROM {}.{} GROUP BY 1,2 HAVING count(*) > 1
                    ) x
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES")),
            ))
            require(duplicates == 0, f"Duplicate geography bridge keys={duplicates}")

            source_conflicts = int(scalar(
                cursor,
                sql.SQL("SELECT count(*) FROM {}.{} WHERE source_code_conflict").format(
                    sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES")
                ),
            ))
            require(source_conflicts == 0, f"PSUs with conflicting source geography={source_conflicts}")

            cursor.execute(
                sql.SQL(
                    """
                    SELECT survey_wave,
                           count(*) FILTER (WHERE admin2_code IS NOT NULL) AS eligible,
                           count(*) FILTER (WHERE admin2_current_match) AS matched,
                           round(avg(admin2_current_match::int) FILTER
                               (WHERE admin2_code IS NOT NULL)::numeric, 4) AS rate
                    FROM {}.{}
                    GROUP BY survey_wave
                    ORDER BY survey_wave
                    """
                ).format(sql.Identifier(args.schema), sql.Identifier("dim_geo_CSES"))
            )
            rows = cursor.fetchall()
            rates = {str(wave): float(rate) for wave, eligible, matched, rate in rows if eligible}
            for wave in ["2007", "2009", "2011-12", "2013", "2014", "2016", "2017", "2019", "2021"]:
                require(rates.get(wave, 0) >= 0.94, f"Admin2 match rate too low for {wave}: {rates.get(wave)}")

            catalog_count = int(scalar(
                cursor,
                sql.SQL("SELECT count(*) FROM {}.{} WHERE table_name = ANY(%s)").format(
                    sql.Identifier(args.schema), sql.Identifier("_catalog")
                ),
                (["dim_admin2_cambodia", "dim_admin3_cambodia", "dim_geo_CSES"],),
            ))
            require(catalog_count == 3, f"Catalog geography rows={catalog_count}")

            print(f"validated=dim_admin2_cambodia rows={counts['dim_admin2_cambodia']}")
            print(f"validated=dim_admin3_cambodia rows={counts['dim_admin3_cambodia']}")
            print(f"validated=dim_geo_CSES rows={counts['dim_geo_CSES']}")
            print(f"survey_month_conflict_psus={scalar(cursor, sql.SQL('SELECT count(*) FROM {}.{} WHERE survey_month_conflict').format(sql.Identifier(args.schema), sql.Identifier('dim_geo_CSES')))}")
            for wave, eligible, matched, rate in rows:
                print(f"admin2_match={wave},eligible={eligible},matched={matched},rate={rate}")
            print("validation=passed")


if __name__ == "__main__":
    main()
