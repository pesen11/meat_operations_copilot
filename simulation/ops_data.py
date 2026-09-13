"""
Read-only access layer over the generated operational history.

Everything above this module (inventory_calc, history_analysis, the agent
tools, the API) talks to these four loaders and never touches a file path
or a SQL cursor directly. That indirection is the whole point: step 7
moves data/generated/*.csv into Postgres, and only `_load()` changes.

Backend selection:
  - If DATABASE_URL is set AND the ops tables exist, read from Postgres.
  - Otherwise read the CSVs in data/generated/.
Both paths return identically-shaped, identically-typed DataFrames, so
nothing downstream can tell which one it got.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

GENERATED_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

# table name -> (csv filename, date columns to parse)
_TABLES: dict[str, tuple[str, list[str]]] = {
    "historical_sales": ("historical_sales.csv", ["sale_date"]),
    "primal_stock": ("primal_stock.csv", ["as_of"]),
    "labor_availability": ("labor_availability.csv", ["shift_date"]),
    "production_schedule": ("production_schedule.csv", ["schedule_date"]),
    # Supply and execution. `purchase_orders` carries two nullable date
    # columns: an open order has no actual_arrival, and coercing those to a
    # real date would turn "has not arrived" into "arrived on some date",
    # which is precisely the error the projected-inventory equation exists
    # to avoid. _load() therefore parses them as nullable (NaT-preserving).
    "purchase_orders": ("purchase_orders.csv", ["order_date", "expected_arrival"]),
    "actual_production": ("actual_production.csv", ["production_date"]),
}

# Date columns that may legitimately be empty, parsed without forcing a value.
_NULLABLE_DATE_COLS: dict[str, list[str]] = {
    "purchase_orders": ["actual_arrival"],
}


def _postgres_available() -> bool:
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        from db.postgres import tables_ready  # noqa: PLC0415 - optional dependency
        return tables_ready()
    except Exception:
        return False


def _load(table: str) -> pd.DataFrame:
    filename, date_cols = _TABLES[table]
    if _postgres_available():
        from db.postgres import read_table  # noqa: PLC0415 - optional dependency
        df = read_table(table)
    else:
        path = GENERATED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python data/generate_history.py` first."
            )
        df = pd.read_csv(path)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col]).dt.date
    for col in _NULLABLE_DATE_COLS.get(table, []):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = [None if pd.isna(v) else v.date() for v in parsed]
    return df


@lru_cache(maxsize=None)
def _cached(table: str) -> pd.DataFrame:
    return _load(table)


def load_sales() -> pd.DataFrame:
    """sku, sale_date (date), units_sold (float), revenue (float)."""
    return _cached("historical_sales").copy()


def load_primal_stock() -> pd.DataFrame:
    """source_primal, as_of (date), boxes_on_hand (float), velocity_tier (str)."""
    return _cached("primal_stock").copy()


def load_labor() -> pd.DataFrame:
    """shift_date (date), role (str), available_minutes (float), headcount (int)."""
    return _cached("labor_availability").copy()


def load_production_schedule() -> pd.DataFrame:
    """schedule_date (date), source_primal, planned_primal_kg (float), notes."""
    return _cached("production_schedule").copy()


def load_purchase_orders() -> pd.DataFrame:
    """po_id, source_primal, order_date, expected_arrival (dates),
    quantity_kg, status, supplier, actual_arrival (date|None),
    received_kg (float|NaN), data_source.

    An OPEN order (status ordered/in_transit/delayed) has actual_arrival
    None and received_kg NaN. Read quantity_kg for those and received_kg
    for delivered ones - simulation/supply_calc.py is the only module
    allowed to make that distinction."""
    return _cached("purchase_orders").copy()


def load_actual_production() -> pd.DataFrame:
    """production_date (date), source_primal, planned_primal_kg,
    actual_primal_kg, status, shortfall_reason, data_source."""
    return _cached("actual_production").copy()


def clear_cache() -> None:
    """Drop the in-process cache — used by tests and after a data reload."""
    _cached.cache_clear()


def latest_date(df: pd.DataFrame, col: str):
    return max(df[col])
