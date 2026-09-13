"""
LangChain @tool wrappers for the Inventory agent.

Same contract as simulation/tools.py: these functions contain ZERO
arithmetic. They parse arguments, call a tested function in
simulation/inventory_calc.py, and shape the result into a dict. If you
find yourself writing an operator other than `=` in this file, the
calculation belongs in inventory_calc.py with a test.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from langchain_core.tools import tool

from data.catalog import primal_reference_box_weight_kg
from simulation import inventory_calc as ic


def _parse_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


@tool
def list_primals() -> list[str]:
    """List every source primal tracked in inventory. Use this to get exact
    primal-name spellings before calling get_stock_position."""
    return sorted(primal_reference_box_weight_kg())


@tool
def get_stock_position(source_primal: str, as_of: Optional[str] = None) -> dict:
    """Get the current raw-primal stock position for one primal: boxes on
    hand, the normal target box range for its velocity tier, kg on hand,
    average daily consumption, days of cover, and a status
    (stockout/critical/below_target/ok/over_target). `as_of` is an optional
    ISO date (YYYY-MM-DD); it defaults to the most recent record."""
    return ic.stock_position(source_primal, as_of=_parse_date(as_of)).to_dict()


@tool
def get_all_stock_positions(as_of: Optional[str] = None) -> list[dict]:
    """Get the stock position for every primal at once. Use this for
    shop-wide inventory questions ('what are we short on?') rather than
    calling get_stock_position repeatedly."""
    return [p.to_dict() for p in ic.all_stock_positions(as_of=_parse_date(as_of))]


@tool
def get_cutter_capacity(shift_date: Optional[str] = None) -> dict:
    """Get cutting-labor capacity for one day: cutter headcount, minutes
    available, minutes actually required by that day's output, slack, and
    utilization. `shift_date` is an optional ISO date; it defaults to the
    most recent day. Raises if the shop was closed that day (stat holiday)."""
    return ic.cutter_capacity(_parse_date(shift_date)).to_dict()


@tool
def get_scenario_impact(product_sku: str, demand_multiplier: float,
                        as_of: Optional[str] = None) -> dict:
    """
    THE scenario tool: compute what scaling one product's demand does to
    inventory, labor, and waste.

    Use demand_multiplier 1.15 for '15% more', 0.9 for '10% less'. Returns
    baseline vs. scenario weekly figures for finished product kg, primal kg,
    delivery boxes, trim waste kg, and cutting minutes, plus shop-wide cutter
    utilization before and after, whether cutter capacity is exceeded, and
    days of primal cover before and after.
    """
    return ic.scenario_impact(product_sku, demand_multiplier,
                              as_of=_parse_date(as_of)).to_dict()


@tool
def get_stockout_days(source_primal: str, start: str, end: str) -> dict:
    """Count how many days a primal was completely out of stock between two
    ISO dates (inclusive). Useful for vendor-reliability questions."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    return {
        "source_primal": source_primal,
        "start": start,
        "end": end,
        "stockout_days": ic.stockout_day_count(source_primal, s, e),
        "window_days": (e - s).days + 1,
    }


INVENTORY_TOOLS = [
    list_primals,
    get_stock_position,
    get_all_stock_positions,
    get_cutter_capacity,
    get_scenario_impact,
    get_stockout_days,
]
