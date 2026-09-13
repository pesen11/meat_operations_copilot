"""
LangChain @tool wrappers for the Historical Data agent.

Same contract as simulation/tools.py and inventory_tools.py: lookup,
delegate to a tested function in simulation/history_analysis.py, shape the
result. No arithmetic in this file.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from langchain_core.tools import tool

from simulation import history_analysis as ha


def _parse_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


@tool
def get_history_range() -> dict:
    """Get the first and last date for which sales history exists. Call this
    before asking for a specific window so you don't request dates outside
    the dataset."""
    return {
        "earliest": ha.earliest_sales_date().isoformat(),
        "latest": ha.latest_sales_date().isoformat(),
    }


@tool
def get_sales_summary(sku: str, start: Optional[str] = None,
                      end: Optional[str] = None) -> dict:
    """Summarise one product's sales over a window: total kg, total revenue,
    average kg per open day, weekday vs. weekend averages, and the single
    best day. Dates are ISO (YYYY-MM-DD); omitting them uses the most recent
    28 days. All weights are kilograms of finished product."""
    return ha.sales_summary(sku, _parse_date(start), _parse_date(end)).to_dict()


@tool
def get_catalog_sales_summary(start: Optional[str] = None,
                              end: Optional[str] = None) -> dict:
    """Total sales across the WHOLE catalog over a window: total kg, total
    revenue, how many products sold, average kg and revenue per open day,
    weekday vs. weekend averages, and the single best day. Use this for a
    question about the shop's overall sales that names no product ("what
    were our sales yesterday", "how much did we sell last week"). Dates are
    ISO (YYYY-MM-DD); omitting them uses the most recent 28 days."""
    return ha.catalog_sales_summary(_parse_date(start), _parse_date(end)).to_dict()


@tool
def get_sales_trend(sku: str, as_of: Optional[str] = None,
                    window_days: int = 28) -> dict:
    """
    Compare a product's recent sales window against the window immediately
    before it: average kg/day in each, percent change, and a direction
    (up/down/flat).

    Also returns each window's average seasonal multiplier and a
    `seasonality_explains_change` flag. When that flag is true the change is
    mostly the calendar (a holiday or summer window), NOT a real shift in
    demand - say so rather than reporting it as a trend.
    """
    return ha.sales_trend(sku, as_of=_parse_date(as_of), window_days=window_days).to_dict()


@tool
def get_top_movers(start: Optional[str] = None, end: Optional[str] = None,
                   limit: int = 5, by: str = "revenue") -> list[dict]:
    """Rank the best-selling products over a window. `by` is 'revenue' or
    'kg' - these give different orderings, since a cheap high-volume cut can
    top the kg ranking while a premium cut tops revenue."""
    rows = ha.top_movers(_parse_date(start), _parse_date(end), limit=limit, by=by)
    return [r.to_dict() for r in rows]


@tool
def get_weekly_revenue_series(start: Optional[str] = None,
                              end: Optional[str] = None) -> list[dict]:
    """Whole-catalog revenue bucketed into 7-day periods, for charting a
    trend line. Buckets count backwards from `end`, so the most recent
    bucket is always a full week."""
    return ha.weekly_revenue_series(_parse_date(start), _parse_date(end))


@tool
def get_catalog_performance(start: Optional[str] = None, end: Optional[str] = None,
                            weekends_only: bool = False) -> list[dict]:
    """
    Every product's trade over a window vs. the window before it, with a
    recommended action per product: 'focus' (growing and material),
    'reduce' (really declining - order less), 'watch' (moving but small, or
    explained by the calendar) or 'steady'.

    Use this for "which items are doing well", "what is selling poorly" and
    "what should we order less of". A top-seller ranking cannot answer those:
    the biggest seller can still be shrinking.
    """
    rows = ha.catalog_performance(_parse_date(start), _parse_date(end),
                                  weekends_only=weekends_only)
    return [r.to_dict() for r in rows]


@tool
def get_slow_movers(start: Optional[str] = None, end: Optional[str] = None,
                    limit: int = 5, by: str = "revenue") -> list[dict]:
    """The WORST-selling products over a window. Products that sold nothing at
    all are included at zero, which a reversed top-seller list would miss."""
    rows = ha.slow_movers(_parse_date(start), _parse_date(end), limit=limit, by=by)
    return [r.to_dict() for r in rows]


@tool
def get_weekend_uplift(sku: Optional[str] = None, start: Optional[str] = None,
                       end: Optional[str] = None) -> dict:
    """
    How much heavier a weekend day trades than a weekday, measured from
    history - catalog-wide, or for one product.

    This is the answer to "how much more should we produce for weekends".
    The confirmed seasonal multipliers cover Christmas, long weekends and
    summer; none of them describes an ordinary Saturday.
    """
    return ha.weekend_uplift(sku, _parse_date(start), _parse_date(end)).to_dict()


HISTORY_TOOLS = [
    get_history_range,
    get_catalog_performance,
    get_slow_movers,
    get_weekend_uplift,
    get_sales_summary,
    get_catalog_sales_summary,
    get_sales_trend,
    get_top_movers,
    get_weekly_revenue_series,
]
