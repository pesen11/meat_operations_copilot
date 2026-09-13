"""
Generates a full year (2025) of synthetic operational history and writes
it to CSV under data/generated/, for inspection and for loading into
Postgres in a later step.

Generation order is a dependency order, not a preference - each stage reads
the one above it, which is what keeps the files mutually consistent:

    sales            demand actually observed
      |
      +-> schedule   what we planned to cut to meet it
      |
      +-> stock      boxes in the cooler, drawn down by that plan
            |
            +-> purchase_orders   back-filled from the deliveries stock implies
            |
            +-> actual_production what the runs really produced, starved on
                                  the days stock says the cooler was empty
      |
      +-> schedule status backfilled from actual_production

Two of the outputs are DERIVED REFERENCE EXPORTS, not new sources of truth:
product_catalog.csv and labor_requirements.csv are written out of
data/catalog.py (the typed, validated Pydantic catalog) so the data model is
inspectable as a table and loadable into Postgres. Nothing in the codebase
reads them back - data/catalog.py remains the single source of truth, and
editing the CSV changes nothing. They are regenerated, never hand-edited.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

import pandas as pd

from data.catalog import (
    products_by_sku, yield_profiles, primal_reference_box_weight_kg,
    yield_profiles_by_primal,
)
from simulation.history_generator import (
    generate_sales_history, generate_production_schedule,
    generate_primal_stock_history, generate_labor_history,
)
from simulation.supply_generator import (
    generate_actual_production, generate_open_orders, generate_purchase_orders,
    schedule_priority, supplier_for,
)
from simulation.velocity_tiers import classify_velocity_tiers
from simulation.yield_calc import usable_product_kg

OUT_DIR = Path(__file__).parent / "generated"
START, END = date(2025, 1, 1), date(2025, 12, 31)


def _cell(value):
    """CSV-safe scalar: enums write their value, dates write ISO."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    return value


def _write_csv(path: Path, rows: list, fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _cell(getattr(r, k)) for k in fieldnames})


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _cell(r.get(k)) for k in fieldnames})


# ---------------------------------------------------------------------------
# Derived reference exports
# ---------------------------------------------------------------------------

def product_catalog_rows() -> list[dict]:
    """
    The SKU -> primal bridge, as a table.

    This mapping has always existed and has always been enforced
    (YieldProfile.source_primal, validated by data/catalog.py's
    check_cut_plan_shares and check_primal_costs). Exporting it makes the
    demand-side and supply-side join visible to anyone reading the CSVs
    alone, and gives db/load_csv.py something to load.

    Both cutting-minutes columns are present on purpose. "Minutes per kg" is
    ambiguous in a yield-bearing process - per kg of raw primal that goes in,
    or per kg of finished product that comes out? They differ by the yield,
    which for this catalog ranges 0.62-0.95, so the two figures are up to
    60% apart. Naming both stops a caller picking the wrong one silently.
    """
    box_kg = primal_reference_box_weight_kg()
    tiers = classify_velocity_tiers()
    rows = []
    for yp in yield_profiles:
        product = products_by_sku[yp.product_sku]
        unit_kg = yp.processing_unit_weight_kg
        finished_kg = usable_product_kg(yp)
        minutes = yp.labor_minutes_per_processing_unit
        supplier_key, supplier = supplier_for(yp.source_primal)
        tier = tiers.get(yp.source_primal)
        rows.append({
            "sku": product.sku,
            "product_name": product.name,
            "source_primal": yp.source_primal,
            "unit": product.unit,
            "price_per_kg": round(product.price_per_kg, 2),
            "avg_pack_weight_kg": round(product.avg_pack_weight_kg, 3),
            "yield_pct": round(yp.yield_pct, 4),
            "waste_pct": round(yp.trim_waste_pct, 4),
            "cut_plan_share": round(yp.cut_plan_share, 4),
            "labor_basis": yp.labor_basis,
            "processing_unit_weight_kg": round(unit_kg, 3),
            "reference_box_weight_kg": round(box_kg.get(yp.source_primal, unit_kg), 2),
            "cutting_minutes_per_kg_primal": round(minutes / unit_kg, 4),
            "cutting_minutes_per_kg_product": round(minutes / finished_kg, 4),
            "primal_cost_per_kg": yp.primal_cost_per_kg,
            "velocity_tier": getattr(tier, "value", tier),
            "supplier": supplier,
            "data_source": product.data_source,
        })
    return sorted(rows, key=lambda r: (r["source_primal"], r["sku"]))


def labor_requirement_rows() -> list[dict]:
    """
    Cutting minutes consumed per kg of each primal processed.

    This is the other half of the labor equation. labor_availability.csv
    says what capacity EXISTS; this says what a given production volume
    CONSUMES, so the two can finally be compared without re-deriving the
    conversion at every call site.

    A multi-path primal is averaged across its paths weighted by
    cut_plan_share - one kg of Blade Eyes is 65% thin-sliced and 35% chuck
    roast, and those two paths do not take the same time per kg.
    """
    rows = []
    for primal, paths in sorted(yield_profiles_by_primal().items()):
        total_share = sum(p.cut_plan_share for p in paths) or 1.0
        weighted = sum(
            (p.labor_minutes_per_processing_unit / p.processing_unit_weight_kg)
            * (p.cut_plan_share / total_share)
            for p in paths
        )
        rows.append({
            "source_primal": primal,
            "role": "cutter",
            "minutes_per_kg_primal": round(weighted, 4),
            "paths": "|".join(p.product_sku for p in paths),
            "data_source": "synthetic",
        })
    return rows


# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)

    sales = generate_sales_history(START, END)
    _write_csv(OUT_DIR / "historical_sales.csv", sales,
               ["sku", "sale_date", "units_sold", "revenue"])
    print(f"historical_sales.csv: {len(sales)} rows")

    schedule = generate_production_schedule(START, END)
    stock = generate_primal_stock_history(START, END)
    _write_csv(OUT_DIR / "primal_stock.csv", stock,
               ["source_primal", "as_of", "boxes_on_hand", "velocity_tier"])
    print(f"primal_stock.csv: {len(stock)} rows")

    labor = generate_labor_history(START, END)
    _write_csv(OUT_DIR / "labor_availability.csv", labor,
               ["shift_date", "role", "available_minutes", "headcount"])
    print(f"labor_availability.csv: {len(labor)} rows")

    # --- Supply side: derived from the stock series, so the two agree -----
    stock_df = pd.DataFrame([{
        "source_primal": s.source_primal, "as_of": s.as_of,
        "boxes_on_hand": s.boxes_on_hand,
    } for s in stock])

    delivered = generate_purchase_orders(stock_df)
    open_orders = generate_open_orders(stock_df, as_of=END)
    orders = delivered + open_orders
    _write_csv(OUT_DIR / "purchase_orders.csv", orders,
               ["po_id", "source_primal", "order_date", "expected_arrival",
                "quantity_kg", "status", "supplier", "actual_arrival",
                "received_kg", "data_source"])
    print(f"purchase_orders.csv: {len(orders)} rows "
          f"({len(delivered)} delivered, {len(open_orders)} open)")

    # --- Execution side ---------------------------------------------------
    schedule_df = pd.DataFrame([{
        "schedule_date": e.schedule_date, "source_primal": e.source_primal,
        "planned_primal_kg": e.planned_primal_kg,
    } for e in schedule])

    actuals = generate_actual_production(schedule_df, stock_df)
    _write_csv(OUT_DIR / "actual_production.csv", actuals,
               ["production_date", "source_primal", "planned_primal_kg",
                "actual_primal_kg", "status", "shortfall_reason", "data_source"])
    print(f"actual_production.csv: {len(actuals)} rows")

    # --- Backfill schedule status from what actually happened -------------
    tiers = classify_velocity_tiers()
    resolved = {(a.production_date, a.source_primal): a.status for a in actuals}
    for entry in schedule:
        entry.priority = schedule_priority(entry.source_primal, tiers)
        status = resolved.get((entry.schedule_date, entry.source_primal))
        if status is not None:
            entry.status = status
    _write_csv(OUT_DIR / "production_schedule.csv", schedule,
               ["schedule_date", "source_primal", "planned_primal_kg",
                "status", "priority", "notes"])
    print(f"production_schedule.csv: {len(schedule)} rows")

    # --- Derived reference exports ---------------------------------------
    catalog_rows = product_catalog_rows()
    _write_rows(OUT_DIR / "product_catalog.csv", catalog_rows,
                list(catalog_rows[0].keys()))
    print(f"product_catalog.csv: {len(catalog_rows)} rows (derived from data/catalog.py)")

    labor_rows = labor_requirement_rows()
    _write_rows(OUT_DIR / "labor_requirements.csv", labor_rows,
                list(labor_rows[0].keys()))
    print(f"labor_requirements.csv: {len(labor_rows)} rows (derived from data/catalog.py)")


if __name__ == "__main__":
    main()
