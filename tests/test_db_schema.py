"""
Consistency between the CSVs, the loader, the read layer and the schema.

No database is needed for any of this, which is the point: these are the
checks that catch a column added in one place and forgotten in the other two,
and they have to run in the same credential-free CI as everything else.

The failure mode being guarded against is quiet. A table missing from
OPS_TABLES makes `tables_ready()` return True on an incomplete database, so
`ops_data` silently switches to Postgres and reads nothing for the tables
that were never created.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from db import load_csv, postgres
from simulation import ops_data

SCHEMA = (Path(__file__).resolve().parent.parent / "db" / "schema.sql").read_text(
    encoding="utf-8")
GENERATED = Path(__file__).resolve().parent.parent / "data" / "generated"


def _declared_tables() -> set[str]:
    return set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA))


# ---------------------------------------------------------------------------
# The four places a table name has to appear
# ---------------------------------------------------------------------------

def test_every_ops_table_has_a_schema():
    missing = set(postgres.OPS_TABLES) - _declared_tables()
    assert not missing, f"no CREATE TABLE for {sorted(missing)}"


def test_every_ops_table_has_a_loader_spec():
    assert set(postgres.OPS_TABLES) == set(load_csv._SPEC)


def test_every_ops_table_has_a_column_map():
    assert set(postgres.OPS_TABLES) == set(postgres._COLUMNS)


def test_every_ops_table_is_readable_from_csv():
    """ops_data must know about every table the loader can populate, or the
    Postgres and CSV backends return different worlds."""
    assert set(postgres.OPS_TABLES) <= set(ops_data._TABLES)


# ---------------------------------------------------------------------------
# Column agreement
# ---------------------------------------------------------------------------

def test_loader_and_read_layer_agree_on_column_order():
    """Positional indexing over either backend must give the same answer."""
    for table in postgres.OPS_TABLES:
        assert postgres._COLUMNS[table] == load_csv._SPEC[table][1], table


def test_csv_headers_match_the_loader_spec():
    for table, (filename, columns, _conflict) in load_csv._SPEC.items():
        path = GENERATED / filename
        assert path.exists(), f"{filename} missing - run data/generate_history.py"
        with open(path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        missing = set(columns) - set(header)
        assert not missing, f"{filename} has no column(s) {sorted(missing)}"


def test_schema_declares_every_loaded_column():
    for table, (_filename, columns, _conflict) in load_csv._SPEC.items():
        block = re.search(
            rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", SCHEMA, re.S)
        assert block, f"no schema block for {table}"
        body = block.group(1)
        for column in columns:
            assert re.search(rf"\b{column}\b", body), f"{table}.{column} not in schema"


def test_conflict_targets_are_real_columns():
    """ON CONFLICT names a constraint's columns; a typo there fails only once
    a real database is attached."""
    for table, (_filename, columns, conflict) in load_csv._SPEC.items():
        for column in conflict:
            assert column in columns, f"{table}: conflict column {column} not selected"


# ---------------------------------------------------------------------------
# The new supply tables specifically
# ---------------------------------------------------------------------------

def test_purchase_orders_allows_open_rows():
    """An order still in transit has no arrival date and no received
    quantity. A NOT NULL on either would make the table unable to represent
    incoming supply at all, which is the only reason it exists."""
    block = re.search(
        r"CREATE TABLE IF NOT EXISTS purchase_orders \((.*?)\n\);", SCHEMA, re.S).group(1)
    for column in ("actual_arrival", "received_kg"):
        line = next(l for l in block.splitlines() if re.search(rf"\b{column}\b", l))
        assert "NOT NULL" not in line, f"{column} may not be NOT NULL"


def test_open_orders_really_are_open_in_the_data():
    orders = ops_data.load_purchase_orders()
    open_rows = orders[orders["actual_arrival"].isna()]
    assert len(open_rows) > 0, "no open orders - the forward projection has nothing to use"
    assert open_rows["received_kg"].isna().all()
    assert set(open_rows["status"]) <= {"ordered", "in_transit", "delayed"}


def test_delivered_orders_carry_both_arrival_and_quantity():
    orders = ops_data.load_purchase_orders()
    closed = orders[orders["actual_arrival"].notna()]
    assert len(closed) > 0
    assert closed["received_kg"].notna().all(), (
        "an order cannot have arrived without a received quantity")


def test_shortfall_reasons_are_from_the_known_set():
    production = ops_data.load_actual_production()
    reasons = {r for r in production["shortfall_reason"] if isinstance(r, str) and r}
    assert reasons <= {"stock_short", "labor_short", "equipment", "quality_hold"}


def test_completed_runs_have_no_shortfall_reason():
    """A completed run that also carries a failure cause is incoherent, and
    would make the shortfall breakdown double-count."""
    production = ops_data.load_actual_production()
    completed = production[production["status"] == "completed"]
    assert completed["shortfall_reason"].isna().all()


def test_cancelled_runs_produced_nothing():
    production = ops_data.load_actual_production()
    cancelled = production[production["status"] == "cancelled"]
    assert len(cancelled) > 0
    assert (cancelled["actual_primal_kg"] == 0).all()


def test_schedule_status_matches_actual_production():
    """The schedule's status is backfilled from what actually happened. If the
    two drift, one of the files is describing a different day."""
    schedule = ops_data.load_production_schedule()
    production = ops_data.load_actual_production()
    merged = schedule.merge(
        production, left_on=["schedule_date", "source_primal"],
        right_on=["production_date", "source_primal"], suffixes=("_sched", "_actual"))
    assert len(merged) > 0
    mismatched = merged[merged["status_sched"] != merged["status_actual"]]
    assert mismatched.empty, f"{len(mismatched)} schedule rows disagree with production"


# ---------------------------------------------------------------------------
# Derived exports are derived
# ---------------------------------------------------------------------------

def test_product_catalog_export_matches_the_typed_catalog():
    """product_catalog.csv is a VIEW of data/catalog.py, not a second source
    of truth. If it can drift, someone will edit it and expect that to matter."""
    from data.catalog import products_by_sku, yield_profiles

    with open(GENERATED / "product_catalog.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(yield_profiles)
    for row in rows:
        product = products_by_sku[row["sku"]]
        assert row["product_name"] == product.name
        assert float(row["price_per_kg"]) == pytest.approx(product.price_per_kg, abs=0.01)


def test_derived_exports_are_not_loaded_back():
    """Nothing reads them. Loading them would create the second source of
    truth the export exists to avoid."""
    for name in ("product_catalog", "labor_requirements"):
        assert name not in load_csv._SPEC
        assert name not in ops_data._TABLES
