"""
Deterministic cost & margin calculators, built on top of yield_calc.py.

Two inputs here are ESTIMATES, not exact invoice numbers (this is a
synthetic-data project, not a real P&L):

- Primal cost comes from data/primal_costs.py — an estimated vendor price
  per kg of raw primal, carried on YieldProfile.primal_cost_per_kg. The
  real vendor price list is confidential, so those figures are calibrated
  to land the catalog's blended gross margin in the 38-40% the project
  owner confirmed, and are the first thing to replace with real invoices.

  This replaced an earlier model that derived primal cost as a discount
  off the finished product's RETAIL price. That model was wrong, and
  wrong in a quiet way: retail price is what a customer pays per kg of
  trimmed, cut, packaged product, so discounting it by 22.5% and then
  paying that rate across the whole primal — trim, bone and all — implies
  buying primals at roughly what they retail for. Cost of goods came out
  at 80-97% of revenue and the whole catalog cleared about 9.5% margin.
  The discount was never yield-aware, and no value of it could fix that:
  at an 88% yield, a 22.5% discount already puts COGS at 88% of revenue
  before labor. Don't reintroduce a retail-anchored cost estimate.

- Labor rate is a flat average ($27/hr) rather than per-employee, per the
  user's rate. Cutting labor is treated as part of cost of goods here,
  since it is direct production labor; margin_pct is therefore net of
  both primal and labor, and runs ~1 point below the gross margin on
  primal cost alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.domain import Product, YieldProfile, DemandProfile
from simulation.yield_calc import packs_per_processing_unit, usable_product_kg

# Fallback only — used when a YieldProfile carries no costed primal (test
# fixtures, a newly added cut not yet in data/primal_costs.py). The real
# numbers live in that table; this just keeps the model coherent instead
# of silently costing an uncosted primal at zero or at retail.
DEFAULT_TARGET_GROSS_MARGIN_PCT = 0.40
DEFAULT_LABOR_RATE_PER_HOUR = 27.0   # user's stated average


def primal_cost_per_kg(
    product: Product, yp: YieldProfile,
    target_gross_margin_pct: float = DEFAULT_TARGET_GROSS_MARGIN_PCT,
) -> float:
    """
    Estimated vendor cost per kg of the RAW primal (trim and bone included).

    Prefers the costed value on the profile (from data/primal_costs.py).
    Falls back to back-solving from a target gross margin, which is where
    the yield correction the old discount model lacked lives explicitly:
    one kg of finished product consumes 1/yield_pct kg of primal, so the
    primal can only cost yield_pct * price * (1 - margin) per kg.
    """
    if yp.primal_cost_per_kg is not None:
        return yp.primal_cost_per_kg
    return product.price_per_kg * yp.yield_pct * (1 - target_gross_margin_pct)


def primal_cost_per_processing_unit(
    yp: YieldProfile, product: Product,
    target_gross_margin_pct: float = DEFAULT_TARGET_GROSS_MARGIN_PCT,
) -> float:
    """Cost of one whole box/piece of primal (you pay for the trim/waste too, not just usable product)."""
    return yp.processing_unit_weight_kg * primal_cost_per_kg(product, yp, target_gross_margin_pct)


def labor_cost_per_processing_unit(yp: YieldProfile, labor_rate_per_hour: float = DEFAULT_LABOR_RATE_PER_HOUR) -> float:
    return (yp.labor_minutes_per_processing_unit / 60) * labor_rate_per_hour


@dataclass
class PackEconomics:
    product_sku: str
    revenue_per_pack: float
    primal_cost_per_pack: float
    labor_cost_per_pack: float
    margin_per_pack: float
    margin_pct: float


def pack_economics(
    yp: YieldProfile, product: Product,
    target_gross_margin_pct: float = DEFAULT_TARGET_GROSS_MARGIN_PCT,
    labor_rate_per_hour: float = DEFAULT_LABOR_RATE_PER_HOUR,
) -> PackEconomics:
    packs = packs_per_processing_unit(yp, product)
    primal_cost = primal_cost_per_processing_unit(yp, product, target_gross_margin_pct) / packs
    labor_cost = labor_cost_per_processing_unit(yp, labor_rate_per_hour) / packs
    revenue = product.price_per_kg * product.avg_pack_weight_kg
    margin = revenue - primal_cost - labor_cost
    return PackEconomics(
        product_sku=product.sku,
        revenue_per_pack=round(revenue, 3),
        primal_cost_per_pack=round(primal_cost, 3),
        labor_cost_per_pack=round(labor_cost, 3),
        margin_per_pack=round(margin, 3),
        margin_pct=round(margin / revenue, 4) if revenue else 0.0,
    )


@dataclass
class WeeklyProjection:
    product_sku: str
    weekly_kg_sold: float
    weekly_revenue: float
    weekly_primal_cost: float
    weekly_labor_minutes: float
    weekly_labor_cost: float
    weekly_margin: float


def weekly_projection(
    yp: YieldProfile, product: Product, demand: DemandProfile,
    target_gross_margin_pct: float = DEFAULT_TARGET_GROSS_MARGIN_PCT,
    labor_rate_per_hour: float = DEFAULT_LABOR_RATE_PER_HOUR,
    demand_multiplier: float = 1.0,
) -> WeeklyProjection:
    """
    Projects weekly revenue/cost/labor/margin for a product, from its
    weekday/weekend average demand. `demand_multiplier` lets scenario
    questions like "what if we increase production by 15%" scale demand
    without touching the underlying yield/labor model.
    """
    weekly_kg = (demand.avg_sales_weekday_kg * 5 + demand.avg_sales_weekend_kg * 2) * demand_multiplier
    weekly_revenue = weekly_kg * product.price_per_kg

    # kg of primal needed to yield weekly_kg of finished product
    weekly_primal_kg = weekly_kg / yp.yield_pct
    weekly_primal_cost = weekly_primal_kg * primal_cost_per_kg(product, yp, target_gross_margin_pct)

    weekly_packs = weekly_kg / product.avg_pack_weight_kg
    per_pack_labor_min = yp.labor_minutes_per_processing_unit / packs_per_processing_unit(yp, product)
    weekly_labor_minutes = weekly_packs * per_pack_labor_min
    weekly_labor_cost = (weekly_labor_minutes / 60) * labor_rate_per_hour

    weekly_margin = weekly_revenue - weekly_primal_cost - weekly_labor_cost

    return WeeklyProjection(
        product_sku=product.sku,
        weekly_kg_sold=round(weekly_kg, 2),
        weekly_revenue=round(weekly_revenue, 2),
        weekly_primal_cost=round(weekly_primal_cost, 2),
        weekly_labor_minutes=round(weekly_labor_minutes, 1),
        weekly_labor_cost=round(weekly_labor_cost, 2),
        weekly_margin=round(weekly_margin, 2),
    )
