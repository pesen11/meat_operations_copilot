"""
Load data/generated/*.csv into Postgres.

    psql "$DATABASE_URL" -f db/schema.sql
    python -m db.load_csv              # load (upsert)
    python -m db.load_csv --truncate   # replace contents first
    python -m db.load_csv --check      # report row counts, change nothing

Idempotent: every table has a natural unique key and rows upsert on it, so
re-running after regenerating the CSVs updates rather than duplicates.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from db.postgres import OPS_TABLES, connect, dsn, reset_readiness_cache

GENERATED_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

# table -> (csv file, columns, conflict key)
_SPEC = {
    "historical_sales": (
        "historical_sales.csv",
        ("sku", "sale_date", "units_sold", "revenue"),
        ("sku", "sale_date")),
    "primal_stock": (
        "primal_stock.csv",
        ("source_primal", "as_of", "boxes_on_hand", "velocity_tier"),
        ("source_primal", "as_of")),
    "labor_availability": (
        "labor_availability.csv",
        ("shift_date", "role", "available_minutes", "headcount"),
        ("shift_date", "role")),
    "production_schedule": (
        "production_schedule.csv",
        ("schedule_date", "source_primal", "planned_primal_kg",
         "status", "priority", "notes"),
        ("schedule_date", "source_primal")),
    "purchase_orders": (
        "purchase_orders.csv",
        ("po_id", "source_primal", "order_date", "expected_arrival",
         "quantity_kg", "status", "supplier", "actual_arrival", "received_kg",
         "data_source"),
        ("po_id",)),
    "actual_production": (
        "actual_production.csv",
        ("production_date", "source_primal", "planned_primal_kg",
         "actual_primal_kg", "status", "shortfall_reason", "data_source"),
        ("production_date", "source_primal")),
}

BATCH = 1000


def load_table(conn, table: str, truncate: bool) -> int:
    filename, columns, conflict = _SPEC[table]
    path = GENERATED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python data/generate_history.py` first.")

    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in conflict)
    sql = (f"INSERT INTO {table} ({', '.join(columns)}) "
           f"VALUES ({', '.join(['%s'] * len(columns))}) "
           f"ON CONFLICT ({', '.join(conflict)}) DO UPDATE SET {updates}")

    with conn.cursor() as cur:
        if truncate:
            cur.execute(f"TRUNCATE {table} RESTART IDENTITY")
        with open(path, newline="", encoding="utf-8") as f:
            rows, batch = 0, []
            for record in csv.DictReader(f):
                # `or None` maps a blank CSV cell to SQL NULL. That matters
                # most for purchase_orders: an open order has no
                # actual_arrival and no received_kg, and writing "" into a
                # DATE column fails outright while writing it into a numeric
                # one would make "not yet known" indistinguishable from zero.
                batch.append(tuple(record[c] or None for c in columns))
                if len(batch) >= BATCH:
                    cur.executemany(sql, batch)
                    rows += len(batch)
                    batch = []
            if batch:
                cur.executemany(sql, batch)
                rows += len(batch)
    return rows


def check(conn) -> dict[str, int]:
    counts = {}
    with conn.cursor() as cur:
        for table in OPS_TABLES:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if cur.fetchone()[0] is None:
                counts[table] = -1
                continue
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Load generated CSVs into Postgres")
    parser.add_argument("--truncate", action="store_true",
                        help="empty each table before loading")
    parser.add_argument("--check", action="store_true",
                        help="report row counts and exit")
    args = parser.parse_args(argv)

    if not dsn():
        print("DATABASE_URL is not set. The project runs on the generated CSVs "
              "without it; set it only when you want Postgres.", file=sys.stderr)
        return 1

    with connect() as conn:
        if args.check:
            for table, count in check(conn).items():
                print(f"{table}: {'MISSING TABLE' if count < 0 else f'{count} rows'}")
            return 0

        total = 0
        for table in OPS_TABLES:
            rows = load_table(conn, table, args.truncate)
            print(f"{table}: {rows} rows")
            total += rows
        conn.commit()

    reset_readiness_cache()
    print(f"\nLoaded {total} rows. simulation/ops_data.py will now read from "
          f"Postgres instead of the CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
