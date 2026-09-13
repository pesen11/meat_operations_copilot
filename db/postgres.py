"""
Postgres access for the operational tables.

simulation/ops_data.py calls `tables_ready()` and `read_table()` here when
DATABASE_URL is set, and falls back to the generated CSVs otherwise. Both
paths must return identically-shaped DataFrames - that is what lets the
whole simulation, agent, and eval stack run unchanged on either.

Requires: pip install -e ".[postgres]"
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

OPS_TABLES = ("historical_sales", "primal_stock", "labor_availability",
              "production_schedule", "purchase_orders", "actual_production")

# Columns to select per table, in the same order the CSVs use, so a caller
# indexing positionally cannot get different answers from the two backends.
_COLUMNS = {
    "historical_sales": ("sku", "sale_date", "units_sold", "revenue"),
    "primal_stock": ("source_primal", "as_of", "boxes_on_hand", "velocity_tier"),
    "labor_availability": ("shift_date", "role", "available_minutes", "headcount"),
    "production_schedule": ("schedule_date", "source_primal", "planned_primal_kg",
                            "status", "priority", "notes"),
    "purchase_orders": ("po_id", "source_primal", "order_date", "expected_arrival",
                        "quantity_kg", "status", "supplier", "actual_arrival",
                        "received_kg", "data_source"),
    "actual_production": ("production_date", "source_primal", "planned_primal_kg",
                          "actual_primal_kg", "status", "shortfall_reason",
                          "data_source"),
}

# Numeric columns arrive from Postgres as Decimal, which does not behave like
# float in downstream arithmetic (Decimal * float raises). Cast on read.
_NUMERIC = {
    "historical_sales": ("units_sold", "revenue"),
    "primal_stock": ("boxes_on_hand",),
    "labor_availability": ("available_minutes",),
    "production_schedule": ("planned_primal_kg",),
}


def dsn() -> Optional[str]:
    return os.getenv("DATABASE_URL")


def connect():
    url = dsn()
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    import psycopg
    return psycopg.connect(url)


@lru_cache(maxsize=1)
def tables_ready() -> bool:
    """True when every operational table exists AND has rows.

    An empty table counts as not ready on purpose: a schema applied but never
    loaded should fall back to the CSVs rather than silently reporting that
    the shop sold nothing all year.
    """
    try:
        with connect() as conn, conn.cursor() as cur:
            for table in OPS_TABLES:
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    return False
                cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)")
                if not cur.fetchone()[0]:
                    return False
        return True
    except Exception:
        return False


def reset_readiness_cache() -> None:
    tables_ready.cache_clear()


def read_table(table: str) -> pd.DataFrame:
    if table not in _COLUMNS:
        raise ValueError(f"Unknown table '{table}'")
    columns = _COLUMNS[table]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(columns)} FROM {table}")
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=list(columns))
    for col in _NUMERIC.get(table, ()):
        df[col] = df[col].astype(float)
    if "headcount" in df.columns:
        df["headcount"] = df["headcount"].astype(int)
    return df


def log_decision(thread_id: str, question: str, plan: dict, projected_outcome: dict,
                 verdict: str, narration: str, narrated_by: str) -> Optional[int]:
    """Record a scenario answer. Returns the row id, or None if no DB."""
    import json
    if not dsn():
        return None
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO decision_log
                       (thread_id, question, plan, projected_outcome, verdict,
                        narration, narrated_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (thread_id, question, json.dumps(plan, default=str),
                 json.dumps(projected_outcome, default=str), verdict,
                 narration, narrated_by),
            )
            row_id = cur.fetchone()[0]
            conn.commit()
            return row_id
    except Exception:
        # An audit-log failure must not fail the operator's request.
        return None


def log_approval(thread_id: str, status: str, approved_by: Optional[str],
                 note: Optional[str]) -> None:
    if not dsn():
        return
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE decision_log
                      SET approval_status = %s, approved_by = %s,
                          approved_at = now(), note = %s
                    WHERE id = (SELECT id FROM decision_log
                                 WHERE thread_id = %s
                                 ORDER BY asked_at DESC LIMIT 1)""",
                (status, approved_by, note, thread_id),
            )
            conn.commit()
    except Exception:
        return
