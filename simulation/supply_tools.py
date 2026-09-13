"""
LangChain @tool wrappers for the Supply and Production agents.

Same contract as simulation/tools.py and simulation/inventory_tools.py:
these functions contain ZERO arithmetic. They parse arguments, call a tested
function in supply_calc / execution_rates / uncertainty / bottlenecks /
scenario_sweep, and shape the result. If you find yourself writing an
operator other than `=` in this file, the calculation belongs upstream with
a test.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from langchain_core.tools import tool

from simulation import bottlenecks as bn
from simulation import execution_rates as er
from simulation import scenario_sweep as ss
from simulation import supply_calc as sc
from simulation import uncertainty as un


def _parse_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


# ---------------------------------------------------------------------------
# Incoming supply
# ---------------------------------------------------------------------------

@tool
def get_open_purchase_orders(source_primal: Optional[str] = None,
                             as_of: Optional[str] = None,
                             horizon_days: int = 14) -> list[dict]:
    """List purchase orders still expected to arrive within the horizon:
    po_id, primal, supplier, order date, expected arrival, quantity in kg,
    and status (ordered/in_transit/delayed). Omit `source_primal` for every
    primal. Use this for 'what is on order' and 'what is arriving' questions."""
    return sc.open_orders(source_primal, as_of=_parse_date(as_of),
                          horizon_days=horizon_days)


@tool
def get_supply_projection(source_primal: str, as_of: Optional[str] = None,
                          horizon_days: int = 14,
                          demand_multiplier: float = 1.0) -> dict:
    """
    Project one primal's cooler forward day by day.

    Returns opening stock, incoming supply (both as scheduled and discounted
    for measured vendor fill rate), forecast primal draw, projected closing
    stock and days of cover, whether a stockout is expected and on which
    date, plus a day-by-day ledger. Use this for 'will we run out of X'
    rather than get_stock_position, which is only a snapshot of today.
    """
    return sc.project_supply(source_primal, as_of=_parse_date(as_of),
                             horizon_days=horizon_days,
                             demand_multiplier=demand_multiplier).to_dict()


@tool
def get_all_supply_projections(as_of: Optional[str] = None,
                               horizon_days: int = 14,
                               demand_multiplier: float = 1.0) -> list[dict]:
    """Project every primal forward, soonest-to-run-out first. Use this for
    'what do we need to order' and 'what is at risk' across the whole shop.
    Pass `demand_multiplier` to project under elevated demand (1.9 for
    Christmas) rather than at today's run rate."""
    return [p.to_dict() for p in
            sc.project_all_supply(as_of=_parse_date(as_of),
                                  horizon_days=horizon_days,
                                  demand_multiplier=demand_multiplier)]


@tool
def get_supplier_reliability(source_primal: Optional[str] = None,
                             supplier: Optional[str] = None,
                             as_of: Optional[str] = None) -> dict:
    """Measured vendor performance: how many closed orders, the on-time rate,
    the average fill rate (received divided by ordered), and how often orders
    arrive short. Scope it by primal or by supplier, or omit both for the
    whole book."""
    return sc.supplier_reliability(source_primal=source_primal, supplier=supplier,
                                   as_of=_parse_date(as_of)).to_dict()


# ---------------------------------------------------------------------------
# Production execution
# ---------------------------------------------------------------------------

@tool
def get_execution_rate(source_primal: Optional[str] = None,
                       end: Optional[str] = None,
                       window_days: int = 90) -> dict:
    """
    How reliably planned cutting actually gets done: planned kg vs actual kg,
    the resulting rate, run counts by status, and the shortfall broken down
    by cause (stock_short / labor_short / equipment / quality_hold).

    Omit `source_primal` for the whole shop. Use this before treating a
    production plan as supply that will exist.
    """
    return er.execution_rate(source_primal, end=_parse_date(end),
                             window_days=window_days).to_dict()


@tool
def get_all_execution_rates(end: Optional[str] = None,
                            window_days: int = 90) -> list[dict]:
    """Every primal's execution rate, worst first. Use this for 'which cuts
    do we keep failing to get through' questions."""
    return [r.to_dict() for r in er.all_execution_rates(
        end=_parse_date(end), window_days=window_days)]


@tool
def get_expected_actual_production(planned_kg: float, source_primal: str,
                                   end: Optional[str] = None) -> dict:
    """Discount a planned cutting quantity by how reliably that primal is
    actually cut, returning the expected actual kg, the expected shortfall,
    and the basis for the rate used. Use this to turn a plan into a forecast."""
    return er.expected_actual_kg(planned_kg, source_primal, end=_parse_date(end))


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

@tool
def get_demand_interval(product_sku: str, start: str, end: str,
                        confidence: float = 0.95,
                        demand_multiplier: float = 1.0) -> dict:
    """Expected kg of a product over an ISO date range, with a prediction
    interval (lower and upper bounds at the given confidence) and the
    measured volatility behind it. Use this when a point estimate alone
    would overstate how precisely demand is known."""
    return un.demand_interval(product_sku, date.fromisoformat(start),
                              date.fromisoformat(end), confidence=confidence,
                              demand_multiplier=demand_multiplier).to_dict()


@tool
def get_stockout_risk(source_primal: str, as_of: Optional[str] = None,
                      horizon_days: int = 14,
                      demand_multiplier: float = 1.0) -> dict:
    """
    Probability that a primal runs out at any point in the horizon, with the
    date the risk peaks and a day-by-day breakdown.

    Pass `demand_multiplier` to ask the question under elevated demand (1.9
    for Christmas). Use this instead of days-of-cover when the question is
    about risk rather than about today's position.
    """
    return un.stockout_probability(source_primal, as_of=_parse_date(as_of),
                                   horizon_days=horizon_days,
                                   demand_multiplier=demand_multiplier).to_dict()


@tool
def get_supply_for_service_level(source_primal: str,
                                 target_service_level: float = 0.95,
                                 as_of: Optional[str] = None,
                                 horizon_days: int = 14,
                                 demand_multiplier: float = 1.0) -> dict:
    """How much EXTRA primal must be ordered to hold the chance of running
    out below a target service level. Use this for 'how much should we order
    to be safe' questions."""
    return un.supply_needed_for_service_level(
        source_primal, target_service_level=target_service_level,
        as_of=_parse_date(as_of), horizon_days=horizon_days,
        demand_multiplier=demand_multiplier)


# ---------------------------------------------------------------------------
# Bottlenecks and scenario ladders
# ---------------------------------------------------------------------------

@tool
def get_catalog_impact(demand_multiplier: float, as_of: Optional[str] = None,
                       horizon_days: int = 14) -> dict:
    """
    Scale the WHOLE catalog and report what binds: total extra product,
    primal, boxes, waste and cutting minutes, plus a ranked list of
    bottlenecks (capacity / stock / supply / execution), each with the
    constraint named and a suggested remedy.

    Use this for whole-shop questions where a single averaged figure across
    seventeen primals would be meaningless.
    """
    return bn.catalog_impact(demand_multiplier, as_of=_parse_date(as_of),
                             horizon_days=horizon_days).to_dict()


@tool
def get_bottlenecks(demand_multiplier: float, limit: int = 3,
                    as_of: Optional[str] = None) -> list[dict]:
    """The few constraints that actually bind at a given demand level, worst
    first. Use this to answer 'what stops us' in one line rather than
    listing every primal."""
    return bn.top_bottlenecks(demand_multiplier, limit=limit, as_of=_parse_date(as_of))


@tool
def get_product_scenario_ladder(product_sku: str,
                                target_multiplier: Optional[float] = None,
                                as_of: Optional[str] = None) -> dict:
    """
    Evaluate one product at a ladder of production levels (-20% through +90%)
    and return every rung with its extra volume, cutter utilization, waste,
    margin change, stockout probability and whether it is feasible - plus the
    rung the stated selection policy picks and why.

    Pass `target_multiplier` when the operator named a size, so their number
    is evaluated and the nearest feasible rung is offered if it does not fit.
    """
    return ss.sweep_product(product_sku, as_of=_parse_date(as_of),
                            target_multiplier=target_multiplier).to_dict()


@tool
def get_catalog_scenario_ladder(target_multiplier: Optional[float] = None,
                                as_of: Optional[str] = None) -> dict:
    """Evaluate the WHOLE catalog at a ladder of production levels and return
    each rung plus the recommended one. Use this for 'how much more should we
    make' questions that name no single product."""
    return ss.sweep_catalog(as_of=_parse_date(as_of),
                            target_multiplier=target_multiplier).to_dict()


SUPPLY_TOOLS = [
    get_open_purchase_orders,
    get_supply_projection,
    get_all_supply_projections,
    get_supplier_reliability,
    get_execution_rate,
    get_all_execution_rates,
    get_expected_actual_production,
    get_demand_interval,
    get_stockout_risk,
    get_supply_for_service_level,
    get_catalog_impact,
    get_bottlenecks,
    get_product_scenario_ladder,
    get_catalog_scenario_ladder,
]
