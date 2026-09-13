"""
Deterministic inventory & labor-capacity calculators.

This is the bridge between the catalog-level yield math (yield_calc,
cost_calc) and the actual operational history (ops_data). Same rule as
everywhere else in simulation/: plain Python, unit-tested, no LLM
anywhere near the arithmetic.

Two things worth knowing before touching this file:

- Daily primal draw is derived from ACTUAL SALES, not from
  production_schedule.planned_primal_kg. The schedule deliberately plans
  "today + tomorrow" (the user's real process), so every day's planned kg
  contains the next day's requirement too - summing it across a date range
  double-counts roughly 2x. Sales -> finished kg -> / yield_pct -> primal
  kg is the honest draw.
- Box counts use data.catalog.primal_reference_box_weight_kg(), the REAL
  delivery box weight, not the labor-math processing unit. Conflating
  those is the tenderloin bug the project already hit once.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Optional

from data.catalog import (
    products_by_sku, yield_profile_by_sku, primal_reference_box_weight_kg,
)
from models.domain import VelocityTier
from simulation import ops_data
from simulation.units import units_to_kg
from simulation.velocity_tiers import TIER_BOX_RANGES
from simulation.history_generator import TIER_STOCK_OVERRIDES
from simulation.yield_calc import labor_minutes_per_pack

# How far back to average when deriving "normal" daily consumption. 28 days
# = 4 full weeks, so the weekday/weekend mix is balanced and one holiday
# week cannot dominate. ASSUMPTION: not a user-confirmed number, just a
# reporting-window choice - widen it if the shop thinks in a longer cycle.
DEFAULT_LOOKBACK_DAYS = 28

# Status threshold: days_of_cover < 1 means the cooler cannot cover even a
# single normal day of cutting.
CRITICAL_DAYS_OF_COVER = 1.0

# Only cutters do the work the yield/labor math models. The other roles in
# labor_availability.csv (wrapper, fishmonger, counter, closing) exist for
# operational realism and are not cutting capacity.
CUTTING_ROLE = "cutter"


# ---------------------------------------------------------------------------
# Primal consumption
# ---------------------------------------------------------------------------

def primal_kg_consumed_by_day(start: date, end: date) -> dict[tuple[str, date], float]:
    """
    Kg of each primal actually drawn down per day, derived from that day's
    sales: finished kg / yield_pct, summed over every product the primal
    feeds. Keyed (source_primal, date).
    """
    sales = ops_data.load_sales()
    window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= end)]
    out: dict[tuple[str, date], float] = defaultdict(float)
    for row in window.itertuples(index=False):
        yp = yield_profile_by_sku.get(row.sku)
        if yp is None:
            continue
        product_kg = units_to_kg(row.sku, float(row.units_sold))
        out[(yp.source_primal, row.sale_date)] += product_kg / yp.yield_pct
    return dict(out)


def _latest_sales_date() -> date:
    return max(ops_data.load_sales()["sale_date"])


def avg_daily_primal_kg(as_of: Optional[date] = None,
                        lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> dict[str, float]:
    """
    Average kg/day of each primal consumed over the lookback window ending
    at `as_of`. Averaged over OPEN days only - dividing by calendar days
    would understate the draw by however many stat holidays fell in the
    window, which is exactly the kind of quiet bias that makes days-of-cover
    look safer than it is.
    """
    as_of = as_of or _latest_sales_date()
    start = as_of - timedelta(days=lookback_days - 1)
    per_day = primal_kg_consumed_by_day(start, as_of)

    totals: dict[str, float] = defaultdict(float)
    open_days: dict[str, set] = defaultdict(set)
    for (primal, d), kg in per_day.items():
        totals[primal] += kg
        open_days[primal].add(d)

    return {p: totals[p] / len(open_days[p]) for p in totals if open_days[p]}


# ---------------------------------------------------------------------------
# Stock positions
# ---------------------------------------------------------------------------

@dataclass
class PrimalStockPosition:
    source_primal: str
    as_of: date
    boxes_on_hand: float
    velocity_tier: str
    target_boxes_low: float
    target_boxes_high: float
    reference_box_weight_kg: float
    kg_on_hand: float
    avg_daily_primal_kg: float
    days_of_cover: Optional[float]
    status: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


def target_box_range(source_primal: str, tier: VelocityTier | str) -> tuple[float, float]:
    """Normal on-hand box range for a primal. Per-primal overrides
    (tenderloin) win over the generic velocity-tier range."""
    if source_primal in TIER_STOCK_OVERRIDES:
        return TIER_STOCK_OVERRIDES[source_primal]
    tier_enum = VelocityTier(tier) if isinstance(tier, str) else tier
    return TIER_BOX_RANGES[tier_enum]


def _classify(boxes: float, low: float, high: float, days_of_cover: Optional[float]) -> str:
    if boxes <= 0:
        return "stockout"
    if days_of_cover is not None and days_of_cover < CRITICAL_DAYS_OF_COVER:
        return "critical"
    if boxes < low:
        return "below_target"
    if boxes > high:
        return "over_target"
    return "ok"


def stock_position(source_primal: str, as_of: Optional[date] = None,
                   lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> PrimalStockPosition:
    stock = ops_data.load_primal_stock()
    rows = stock[stock["source_primal"] == source_primal]
    if rows.empty:
        raise ValueError(f"Unknown primal '{source_primal}'. "
                         f"Known primals: {sorted(set(stock['source_primal']))}")
    as_of = as_of or max(rows["as_of"])
    on_or_before = rows[rows["as_of"] <= as_of]
    if on_or_before.empty:
        raise ValueError(f"No stock record for '{source_primal}' on or before {as_of}")
    row = on_or_before.loc[on_or_before["as_of"].idxmax()]

    boxes = float(row["boxes_on_hand"])
    tier = str(row["velocity_tier"])
    low, high = target_box_range(source_primal, tier)
    box_kg = primal_reference_box_weight_kg()[source_primal]
    daily = avg_daily_primal_kg(as_of=as_of, lookback_days=lookback_days).get(source_primal, 0.0)
    kg_on_hand = boxes * box_kg
    cover = round(kg_on_hand / daily, 2) if daily > 0 else None

    return PrimalStockPosition(
        source_primal=source_primal,
        as_of=row["as_of"],
        boxes_on_hand=round(boxes, 2),
        velocity_tier=tier,
        target_boxes_low=low,
        target_boxes_high=high,
        reference_box_weight_kg=round(box_kg, 2),
        kg_on_hand=round(kg_on_hand, 2),
        avg_daily_primal_kg=round(daily, 2),
        days_of_cover=cover,
        status=_classify(boxes, low, high, cover),
    )


def all_stock_positions(as_of: Optional[date] = None,
                        lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> list[PrimalStockPosition]:
    stock = ops_data.load_primal_stock()
    as_of = as_of or max(stock["as_of"])
    primals = sorted(set(stock["source_primal"]))
    return [stock_position(p, as_of=as_of, lookback_days=lookback_days) for p in primals]


def stockout_day_count(source_primal: str, start: date, end: date) -> int:
    """How many days in the window the primal was fully out of stock."""
    stock = ops_data.load_primal_stock()
    rows = stock[(stock["source_primal"] == source_primal)
                 & (stock["as_of"] >= start) & (stock["as_of"] <= end)]
    return int((rows["boxes_on_hand"] <= 0).sum())


# ---------------------------------------------------------------------------
# Labor capacity
# ---------------------------------------------------------------------------

@dataclass
class LaborPosition:
    shift_date: date
    role: str
    headcount: int
    available_minutes: float
    required_cutting_minutes: float
    slack_minutes: float
    utilization_pct: float
    is_over_capacity: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shift_date"] = self.shift_date.isoformat()
        return d


def required_cutting_minutes_for_day(d: date) -> float:
    """
    Cutting minutes implied by what actually sold on day d: for each product,
    finished kg -> packs -> packs x that path's labor_minutes_per_pack.
    """
    sales = ops_data.load_sales()
    day = sales[sales["sale_date"] == d]
    total = 0.0
    for row in day.itertuples(index=False):
        yp = yield_profile_by_sku.get(row.sku)
        if yp is None:
            continue
        product = products_by_sku[row.sku]
        kg = units_to_kg(row.sku, float(row.units_sold))
        packs = kg / product.avg_pack_weight_kg
        total += packs * labor_minutes_per_pack(yp, product)
    return total


def cutter_capacity(d: Optional[date] = None) -> LaborPosition:
    """Cutter minutes available on day d vs. the cutting minutes that day's
    sales actually required."""
    labor = ops_data.load_labor()
    d = d or max(labor["shift_date"])
    rows = labor[(labor["shift_date"] == d) & (labor["role"] == CUTTING_ROLE)]
    if rows.empty:
        raise ValueError(f"No '{CUTTING_ROLE}' labor record for {d} "
                         f"(the shop is closed on Ontario stat holidays).")

    available = float(rows["available_minutes"].sum())
    headcount = int(rows["headcount"].sum())
    required = required_cutting_minutes_for_day(d)
    return LaborPosition(
        shift_date=d,
        role=CUTTING_ROLE,
        headcount=headcount,
        available_minutes=round(available, 1),
        required_cutting_minutes=round(required, 1),
        slack_minutes=round(available - required, 1),
        utilization_pct=round(required / available, 4) if available else 0.0,
        is_over_capacity=required > available,
    )


def avg_weekly_cutter_minutes(as_of: Optional[date] = None,
                              lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> float:
    """Average cutter minutes available per 7-day week over the lookback
    window. Divides by calendar weeks (window / 7) rather than open days,
    because a closed stat holiday genuinely removes that capacity from the
    week - it is not available time that simply went unused."""
    labor = ops_data.load_labor()
    as_of = as_of or max(labor["shift_date"])
    start = as_of - timedelta(days=lookback_days - 1)
    rows = labor[(labor["role"] == CUTTING_ROLE)
                 & (labor["shift_date"] >= start) & (labor["shift_date"] <= as_of)]
    if rows.empty:
        return 0.0
    return float(rows["available_minutes"].sum()) / (lookback_days / 7)


# ---------------------------------------------------------------------------
# Scenario impact - the "what if we make 15% more chuck roast" bridge
# ---------------------------------------------------------------------------

@dataclass
class ScenarioImpact:
    product_sku: str
    source_primal: str
    demand_multiplier: float

    baseline_weekly_product_kg: float
    scenario_weekly_product_kg: float
    extra_weekly_product_kg: float

    baseline_weekly_primal_kg: float
    scenario_weekly_primal_kg: float
    extra_weekly_primal_kg: float
    extra_weekly_boxes: float

    baseline_weekly_trim_waste_kg: float
    scenario_weekly_trim_waste_kg: float
    extra_weekly_trim_waste_kg: float

    baseline_weekly_cutting_minutes: float
    scenario_weekly_cutting_minutes: float
    extra_weekly_cutting_minutes: float
    weekly_cutter_minutes_available: float
    baseline_cutter_utilization_pct: float
    scenario_cutter_utilization_pct: float
    exceeds_cutter_capacity: bool

    boxes_on_hand: float
    baseline_days_of_cover: Optional[float]
    scenario_days_of_cover: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def scenario_impact(product_sku: str, demand_multiplier: float,
                    as_of: Optional[date] = None,
                    lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> ScenarioImpact:
    """
    Full inventory/labor/waste consequence of scaling one product's demand.

    Baselines come from observed history over the lookback window (not from
    the DemandProfile), so the answer reflects what the shop actually did;
    the scaling itself is a pure multiplication, so no forecasting model is
    being smuggled in here.

    Cutter utilization is reported against TOTAL cutting minutes for all
    products, not just this one: a 15% lift on chuck roast eats into the
    same shared cutter capacity every other cut needs, and measuring it
    against the single product's minutes would make capacity look far safer
    than it is.
    """
    if demand_multiplier <= 0:
        raise ValueError("demand_multiplier must be positive")
    if product_sku not in products_by_sku:
        raise ValueError(f"Unknown product_sku '{product_sku}'")

    product = products_by_sku[product_sku]
    yp = yield_profile_by_sku[product_sku]
    as_of = as_of or _latest_sales_date()
    start = as_of - timedelta(days=lookback_days - 1)
    weeks = lookback_days / 7

    sales = ops_data.load_sales()
    window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= as_of)]

    sku_rows = window[window["sku"] == product_sku]
    baseline_weekly_kg = sum(units_to_kg(product_sku, float(u))
                             for u in sku_rows["units_sold"]) / weeks
    extra_weekly_kg = baseline_weekly_kg * (demand_multiplier - 1)

    baseline_primal_kg = baseline_weekly_kg / yp.yield_pct
    extra_primal_kg = extra_weekly_kg / yp.yield_pct
    box_kg = primal_reference_box_weight_kg()[yp.source_primal]

    # Trim waste scales with PRIMAL throughput, not finished product.
    baseline_waste = baseline_primal_kg * yp.trim_waste_pct
    extra_waste = extra_primal_kg * yp.trim_waste_pct

    per_pack_min = labor_minutes_per_pack(yp, product)
    extra_minutes = (extra_weekly_kg / product.avg_pack_weight_kg) * per_pack_min

    # Shop-wide weekly cutting minutes across every product in the window.
    total_minutes = sum(required_cutting_minutes_for_day(d)
                        for d in sorted(set(window["sale_date"])))
    baseline_total_weekly_minutes = total_minutes / weeks
    scenario_total_weekly_minutes = baseline_total_weekly_minutes + extra_minutes

    available = avg_weekly_cutter_minutes(as_of=as_of, lookback_days=lookback_days)

    pos = stock_position(yp.source_primal, as_of=as_of, lookback_days=lookback_days)
    scenario_daily_primal = pos.avg_daily_primal_kg + extra_primal_kg / 7
    scenario_cover = (round(pos.kg_on_hand / scenario_daily_primal, 2)
                      if scenario_daily_primal > 0 else None)

    return ScenarioImpact(
        product_sku=product_sku,
        source_primal=yp.source_primal,
        demand_multiplier=demand_multiplier,
        baseline_weekly_product_kg=round(baseline_weekly_kg, 2),
        scenario_weekly_product_kg=round(baseline_weekly_kg * demand_multiplier, 2),
        extra_weekly_product_kg=round(extra_weekly_kg, 2),
        baseline_weekly_primal_kg=round(baseline_primal_kg, 2),
        scenario_weekly_primal_kg=round(baseline_primal_kg * demand_multiplier, 2),
        extra_weekly_primal_kg=round(extra_primal_kg, 2),
        extra_weekly_boxes=round(extra_primal_kg / box_kg, 2),
        baseline_weekly_trim_waste_kg=round(baseline_waste, 2),
        scenario_weekly_trim_waste_kg=round(baseline_waste * demand_multiplier, 2),
        extra_weekly_trim_waste_kg=round(extra_waste, 2),
        baseline_weekly_cutting_minutes=round(baseline_total_weekly_minutes, 1),
        scenario_weekly_cutting_minutes=round(scenario_total_weekly_minutes, 1),
        extra_weekly_cutting_minutes=round(extra_minutes, 1),
        weekly_cutter_minutes_available=round(available, 1),
        baseline_cutter_utilization_pct=(
            round(baseline_total_weekly_minutes / available, 4) if available else 0.0),
        scenario_cutter_utilization_pct=(
            round(scenario_total_weekly_minutes / available, 4) if available else 0.0),
        exceeds_cutter_capacity=scenario_total_weekly_minutes > available,
        boxes_on_hand=pos.boxes_on_hand,
        baseline_days_of_cover=pos.days_of_cover,
        scenario_days_of_cover=scenario_cover,
    )
