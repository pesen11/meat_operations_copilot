"""
LangChain @tool wrappers for seasonal planning.

Same contract as the other tool belts: lookup, delegate to a tested function
in simulation/seasonal_planning.py, shape the result. No arithmetic here.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from langchain_core.tools import tool

from simulation import seasonal_planning as sp


def _parse_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


@tool
def get_seasonal_plan(period: str, as_of: Optional[str] = None,
                      product_sku: Optional[str] = None) -> dict:
    """
    Answer "how much more should we produce for <season>?".

    `period` is one of: christmas, long_weekend, summer, regular. The uplift
    applied is the shop's own CONFIRMED seasonal multiplier (Christmas 1.9x,
    long weekend 1.7x, summer 1.4x) — it is not estimated.

    Returns the per-product uplift, the extra primal kg and delivery boxes to
    order per primal, extra trim waste, and whether the extra cutting work
    fits inside available cutter minutes. Omit `product_sku` to plan the
    whole catalog.
    """
    return sp.seasonal_plan(period, as_of=_parse_date(as_of),
                            product_sku=product_sku).to_dict()


@tool
def get_seasonal_calendar(as_of: Optional[str] = None,
                          horizon_days: int = 120) -> list[dict]:
    """List the elevated trading periods coming up within `horizon_days`, each
    with its confirmed multiplier and how many days away it is. Use this for
    'what's coming up that we should plan for?'."""
    return sp.seasonal_calendar(as_of=_parse_date(as_of), horizon_days=horizon_days)


@tool
def get_day_multiplier(day: str) -> dict:
    """Get the seasonal demand multiplier that applies on one calendar date
    (ISO YYYY-MM-DD), and which seasonal rules caused it."""
    return sp.multiplier_for_day(date.fromisoformat(day))


@tool
def get_last_period_actuals(period: str, as_of: Optional[str] = None) -> Optional[dict]:
    """
    What the shop ACTUALLY sold during the most recent occurrence of a named
    period (christmas, long_weekend, summer) - total kg and revenue, average
    kg/day, the same figure for ordinary trading before it, the observed
    multiplier that implies, and the products that sold most.

    Use this alongside get_seasonal_plan for "how should we prepare for the
    holiday given what happened last time". Returns null when the history
    holds no completed occurrence, which must be reported as "no comparable
    period in the data" rather than filled in from the multiplier.
    """
    actuals = sp.period_actuals(period, as_of=_parse_date(as_of))
    return actuals.to_dict() if actuals else None


SEASONAL_TOOLS = [get_seasonal_plan, get_seasonal_calendar, get_day_multiplier,
                  get_last_period_actuals]
