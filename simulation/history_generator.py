"""
Generates a full year (2025) of synthetic operational history:
- HistoricalSale: daily sales per product, seasonality-driven + noise
- ProductionScheduleEntry: daily primal cutting plan (today + tomorrow, per user's process)
- PrimalStockLevel: daily box-level raw primal inventory, with periodic vendor stockouts
- LaborAvailability: daily staffing by role, per user's confirmed shift pattern

FLAGGED ASSUMPTIONS not directly confirmed with the user (all noted inline
too — these are the knobs to correct first if the generated data looks off):
- SALES_NOISE_CV = 0.12 (12% day-to-day variance around the seasonal average)
- STOCKOUT_PROB = 0.04 per eligible delivery day per primal ("sometimes"
  the vendor runs out, no frequency given)
- Primal box "reference weight" for stock tracking uses each primal's
  first yield path's processing-unit weight (box or piece)
- Summer months = Jun/Jul/Aug, Christmas window = Dec 19-25 (see seasonality.py)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from data.catalog import (
    products_by_sku, yield_profiles, demand_by_sku, primal_reference_box_weight_kg,
)
from models.domain import (
    HistoricalSale, LaborAvailability, PrimalStockLevel, ProductionScheduleEntry, Unit,
)
from simulation.demand_forecast import expected_kg_for_day
from simulation.seasonality import is_closed, is_weekend
from simulation.velocity_tiers import classify_velocity_tiers, TIER_BOX_RANGES

SALES_NOISE_CV = 0.12
STOCKOUT_PROB = 0.04
RNG_SEED = 42

FULL_MINUTES_PER_SHIFT = 8.5 * 60  # 510 minutes


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------------
# Sales
# ---------------------------------------------------------------------------

def generate_sales_history(start: date, end: date, seed: int = RNG_SEED) -> list[HistoricalSale]:
    rng = np.random.default_rng(seed)
    sales: list[HistoricalSale] = []
    sigma = np.sqrt(np.log(1 + SALES_NOISE_CV ** 2))  # lognormal sigma for target CV, mean multiplier ~1

    for d in _date_range(start, end):
        if is_closed(d):
            continue
        for sku, product in products_by_sku.items():
            expected_kg = expected_kg_for_day(sku, d)
            if expected_kg <= 0:
                continue
            noise = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma)  # mean-1 lognormal
            actual_kg = max(0.0, expected_kg * noise)

            if product.unit == Unit.EACH:
                units = max(0.0, round(actual_kg / product.avg_pack_weight_kg))
                revenue = round(units * product.avg_pack_weight_kg * product.price_per_kg, 2)
            else:
                units = round(actual_kg, 2)
                revenue = round(units * product.price_per_kg, 2)

            if units <= 0:
                continue
            sales.append(HistoricalSale(sku=sku, sale_date=d, units_sold=units, revenue=revenue))

    return sales


# ---------------------------------------------------------------------------
# Production schedule (today + tomorrow, per user's confirmed process)
# ---------------------------------------------------------------------------

def required_primal_kg_for_day(d: date) -> dict[str, float]:
    """Kg of each primal needed to cover that day's expected product demand."""
    result: dict[str, float] = defaultdict(float)
    for yp in yield_profiles:
        expected = expected_kg_for_day(yp.product_sku, d)
        if expected <= 0:
            continue
        result[yp.source_primal] += expected / yp.yield_pct
    return dict(result)


def generate_production_schedule(start: date, end: date) -> list[ProductionScheduleEntry]:
    entries: list[ProductionScheduleEntry] = []
    for d in _date_range(start, end):
        if is_closed(d):
            continue
        need_today = required_primal_kg_for_day(d)
        need_tomorrow = required_primal_kg_for_day(d + timedelta(days=1))
        all_primals = set(need_today) | set(need_tomorrow)
        for primal in all_primals:
            planned_kg = need_today.get(primal, 0.0) + need_tomorrow.get(primal, 0.0)
            if planned_kg <= 0:
                continue
            entries.append(ProductionScheduleEntry(
                schedule_date=d, source_primal=primal, planned_primal_kg=round(planned_kg, 2),
                notes="today + next-day buffer",
            ))
    return entries


# ---------------------------------------------------------------------------
# Primal box stock
# ---------------------------------------------------------------------------

TIER_STOCK_OVERRIDES: dict[str, tuple[float, float]] = {
    # Tenderloin doesn't fit the generic tier box-count pattern (its real
    # delivery box is ~40kg vs. the ~4.5kg piece used for labor math), so
    # the tier-based 5-6 box range would mean ~200kg+ on hand — absurd for
    # a low-seller kept for freshness reasons. Confirmed with user: keep
    # ~1-2 REAL boxes on hand instead.
    "Tenderloin": (1.0, 2.0),
}


def _primal_reference_weight_kg() -> dict[str, float]:
    """Reference 'box' weight per primal, for converting kg <-> box counts.
    Delegates to data.catalog so the generator and inventory_calc cannot
    drift apart on what a 'box' of a given primal weighs."""
    return primal_reference_box_weight_kg()


def generate_primal_stock_history(start: date, end: date, seed: int = RNG_SEED + 1) -> list[PrimalStockLevel]:
    rng = np.random.default_rng(seed)
    tiers = classify_velocity_tiers()
    ref_weight = _primal_reference_weight_kg()
    primals = list(tiers.keys())

    stock: dict[str, float] = {}
    for p in primals:
        lo, hi = TIER_STOCK_OVERRIDES.get(p, TIER_BOX_RANGES[tiers[p]])
        stock[p] = (lo + hi) / 2
    cooldown: dict[str, int] = {p: 0 for p in primals}

    records: list[PrimalStockLevel] = []

    for d in _date_range(start, end):
        if is_closed(d):
            continue

        need_today_kg = required_primal_kg_for_day(d)
        need_tomorrow_kg = required_primal_kg_for_day(d + timedelta(days=1))

        for primal in primals:
            # cooldown decays every day the shop is open, regardless of delivery day
            if cooldown[primal] > 0:
                cooldown[primal] -= 1

            is_sunday = d.weekday() == 6
            if not is_sunday and cooldown[primal] == 0:
                if rng.random() < STOCKOUT_PROB:
                    cooldown[primal] = int(rng.integers(3, 5))  # 3 or 4 days out of stock
                else:
                    lo, hi = TIER_STOCK_OVERRIDES.get(primal, TIER_BOX_RANGES[tiers[primal]])
                    target = rng.uniform(lo, hi)
                    if stock[primal] < target:
                        stock[primal] += (target - stock[primal]) * rng.uniform(0.8, 1.0)

            records.append(PrimalStockLevel(
                source_primal=primal, as_of=d,
                boxes_on_hand=round(max(stock[primal], 0.0), 1),
                velocity_tier=tiers[primal],
            ))

            planned_kg = need_today_kg.get(primal, 0.0) + need_tomorrow_kg.get(primal, 0.0)
            consumed_boxes = planned_kg / ref_weight[primal]
            stock[primal] = max(0.0, stock[primal] - consumed_boxes)

    return records


# ---------------------------------------------------------------------------
# Labor
# ---------------------------------------------------------------------------

def generate_labor_history(start: date, end: date) -> list[LaborAvailability]:
    """
    Per user's confirmed schedule:
    - Morning (5am): 2 cutters, 1 wrapper, 1 fishmonger, 1 counter maintainer — all 8.5h
    - Mid-shift (9-10:30am start): 1 more counter/wrapper — 8.5h
    - Closing (1:30pm start): 2 closers — 8.5h
    - Weekends add: 1 more mid-shift person, 1 more closer

    Only the 'cutter' role feeds the yield/labor simulation in cost_calc.py —
    the rest are generated for a complete operational picture but aren't
    consumed by the current cutting-labor math.
    """
    labor: list[LaborAvailability] = []

    for d in _date_range(start, end):
        if is_closed(d):
            continue
        weekend = is_weekend(d)

        role_counts: dict[str, int] = {
            "cutter": 2,
            "wrapper": 1,
            "fishmonger": 1,
            "counter": 1,
            "mid_wrapper_counter": 1,
            "closing": 2,
        }
        if weekend:
            role_counts["mid_wrapper_counter"] += 1
            role_counts["closing"] += 1

        for role, headcount in role_counts.items():
            labor.append(LaborAvailability(
                shift_date=d, role=role,
                available_minutes=headcount * FULL_MINUTES_PER_SHIFT,
                headcount=headcount,
            ))

    return labor
