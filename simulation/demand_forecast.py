"""
Demand forecasting: turns a product's DemandProfile (weekday/weekend avg
kg) into an expected kg for a SPECIFIC calendar day, accounting for:

1. A global scale factor so the catalog's implied weekly revenue matches
   the user's confirmed real-world baseline.
2. The day's seasonal multiplier (see seasonality.py).

Baseline derivation (confirmed with user):
- Whole-store revenue in a regular winter week: ~$400,000.
- Cut meat products are ~30-40% of that (midpoint 35% used) — the rest is
  whole primals, pre-packed items, chicken, fish, which this project does
  not model.
- So CUT_PRODUCT_WEEKLY_BASELINE = $400,000 * 0.35 = $140,000/week.

This is compared against what the current catalog's DemandProfile rows
imply at face value, and a single global CATALOG_SCALE_FACTOR is derived
to reconcile the two — rather than scaling each product individually,
which would require guessing which specific products are over/under-
represented with no basis to do so.
"""

from __future__ import annotations

from datetime import date

from data.catalog import products_by_sku, demand_by_sku
from models.domain import DemandProfile, Product
from simulation.seasonality import week_multiplier, is_closed

WHOLE_STORE_WEEKLY_BASELINE = 400_000.0
CUT_PRODUCT_SHARE_OF_TOTAL = 0.35  # midpoint of user's confirmed 30-40% range
CUT_PRODUCT_WEEKLY_BASELINE = WHOLE_STORE_WEEKLY_BASELINE * CUT_PRODUCT_SHARE_OF_TOTAL


def _catalog_implied_weekly_revenue() -> float:
    total = 0.0
    for sku, dp in demand_by_sku.items():
        p = products_by_sku[sku]
        weekly_kg = dp.avg_sales_weekday_kg * 5 + dp.avg_sales_weekend_kg * 2
        total += weekly_kg * p.price_per_kg
    return total


# Computed once at import time from the current catalog. If the catalog
# changes (more products added/removed), this recalculates automatically
# rather than drifting out of sync with a hardcoded number.
CATALOG_IMPLIED_WEEKLY_REVENUE = _catalog_implied_weekly_revenue()
CATALOG_SCALE_FACTOR = CUT_PRODUCT_WEEKLY_BASELINE / CATALOG_IMPLIED_WEEKLY_REVENUE


def expected_kg_for_day(product_sku: str, d: date) -> float:
    """
    Expected kg sold of a product on a specific calendar day, incorporating:
    weekday/weekend base rate -> catalog scale factor -> seasonal multiplier.
    Returns 0.0 on days the shop is closed.
    """
    if is_closed(d):
        return 0.0
    dp: DemandProfile = demand_by_sku[product_sku]
    base = dp.avg_sales_weekend_kg if d.weekday() >= 5 else dp.avg_sales_weekday_kg
    return base * CATALOG_SCALE_FACTOR * week_multiplier(d)


def expected_revenue_for_day(product_sku: str, d: date) -> float:
    p: Product = products_by_sku[product_sku]
    return expected_kg_for_day(product_sku, d) * p.price_per_kg
