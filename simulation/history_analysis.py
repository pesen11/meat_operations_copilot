"""
Deterministic analysis over the generated sales history.

Feeds the Historical Data agent. Like every other module in simulation/,
this is plain tested Python - the LLM reads these results, it never
derives them.

Everything is reported in KILOGRAMS of finished product, normalised via
simulation.units, so tomahawk steak (sold as EACH) is comparable with the
kg-denominated cuts rather than silently being counted as "1.0" per tray.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Optional

from data.catalog import products_by_sku
from simulation import ops_data
from simulation.seasonality import is_weekend, week_multiplier
from simulation.units import units_to_kg

# Default comparison window for trend calls: 28 days recent vs. the 28 days
# before it. Four whole weeks on each side keeps the weekday/weekend mix
# balanced. ASSUMPTION: a reporting convention, not a user-confirmed number.
DEFAULT_TREND_WINDOW_DAYS = 28


def _sales_window(start: date, end: date, sku: Optional[str] = None):
    sales = ops_data.load_sales()
    window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= end)]
    if sku is not None:
        window = window[window["sku"] == sku]
    return window


def latest_sales_date() -> date:
    return max(ops_data.load_sales()["sale_date"])


def earliest_sales_date() -> date:
    return min(ops_data.load_sales()["sale_date"])


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

@dataclass
class SalesSummary:
    sku: str
    product_name: str
    start: date
    end: date
    open_days: int
    total_kg: float
    total_revenue: float
    avg_kg_per_open_day: float
    avg_kg_weekday: float
    avg_kg_weekend: float
    best_day: Optional[date]
    best_day_kg: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        d["best_day"] = self.best_day.isoformat() if self.best_day else None
        return d


def sales_summary(sku: str, start: Optional[date] = None,
                  end: Optional[date] = None) -> SalesSummary:
    if sku not in products_by_sku:
        raise ValueError(f"Unknown sku '{sku}'")
    end = end or latest_sales_date()
    start = start or (end - timedelta(days=DEFAULT_TREND_WINDOW_DAYS - 1))
    window = _sales_window(start, end, sku)

    by_day: dict[date, float] = {}
    revenue = 0.0
    for row in window.itertuples(index=False):
        kg = units_to_kg(sku, float(row.units_sold))
        by_day[row.sale_date] = by_day.get(row.sale_date, 0.0) + kg
        revenue += float(row.revenue)

    total_kg = sum(by_day.values())
    weekday_vals = [kg for d, kg in by_day.items() if not is_weekend(d)]
    weekend_vals = [kg for d, kg in by_day.items() if is_weekend(d)]
    best = max(by_day.items(), key=lambda kv: kv[1]) if by_day else None

    return SalesSummary(
        sku=sku,
        product_name=products_by_sku[sku].name,
        start=start,
        end=end,
        open_days=len(by_day),
        total_kg=round(total_kg, 2),
        total_revenue=round(revenue, 2),
        avg_kg_per_open_day=round(total_kg / len(by_day), 2) if by_day else 0.0,
        avg_kg_weekday=round(sum(weekday_vals) / len(weekday_vals), 2) if weekday_vals else 0.0,
        avg_kg_weekend=round(sum(weekend_vals) / len(weekend_vals), 2) if weekend_vals else 0.0,
        best_day=best[0] if best else None,
        best_day_kg=round(best[1], 2) if best else 0.0,
    )


@dataclass
class CatalogSalesSummary:
    """Whole-catalog trade over a window - the shop's total, not one product's."""
    start: date
    end: date
    open_days: int
    products_sold: int
    total_kg: float
    total_revenue: float
    avg_kg_per_open_day: float
    avg_revenue_per_open_day: float
    avg_kg_weekday: float
    avg_kg_weekend: float
    best_day: Optional[date]
    best_day_kg: float
    best_day_revenue: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        d["best_day"] = self.best_day.isoformat() if self.best_day else None
        return d


def catalog_sales_summary(start: Optional[date] = None,
                          end: Optional[date] = None) -> CatalogSalesSummary:
    """
    What the whole shop sold over a window.

    `sales_summary` above requires a SKU, so a question that named no product
    ("what were our sales yesterday") had no total available and the narration
    could only rank products against each other - it reported which cuts led
    without ever saying how much the shop sold. This is the catalog-wide
    counterpart, aggregated per DAY across every SKU so the weekday/weekend
    split and best-day figures mean the same thing they do per product.

    Revenue is summed straight from the history rows; kg goes through
    `units_to_kg` per SKU because pack sizes differ, so units are not
    comparable across products and must never be added directly.
    """
    end = end or latest_sales_date()
    start = start or (end - timedelta(days=DEFAULT_TREND_WINDOW_DAYS - 1))

    kg_by_day: dict[date, float] = {}
    rev_by_day: dict[date, float] = {}
    skus: set[str] = set()
    for row in _sales_window(start, end).itertuples(index=False):
        kg_by_day[row.sale_date] = (kg_by_day.get(row.sale_date, 0.0)
                                    + units_to_kg(row.sku, float(row.units_sold)))
        rev_by_day[row.sale_date] = rev_by_day.get(row.sale_date, 0.0) + float(row.revenue)
        skus.add(row.sku)

    total_kg = sum(kg_by_day.values())
    total_revenue = sum(rev_by_day.values())
    open_days = len(kg_by_day)
    weekday_vals = [v for d, v in kg_by_day.items() if not is_weekend(d)]
    weekend_vals = [v for d, v in kg_by_day.items() if is_weekend(d)]
    best = max(kg_by_day.items(), key=lambda kv: kv[1]) if kg_by_day else None

    return CatalogSalesSummary(
        start=start,
        end=end,
        open_days=open_days,
        products_sold=len(skus),
        total_kg=round(total_kg, 2),
        total_revenue=round(total_revenue, 2),
        avg_kg_per_open_day=round(total_kg / open_days, 2) if open_days else 0.0,
        avg_revenue_per_open_day=round(total_revenue / open_days, 2) if open_days else 0.0,
        avg_kg_weekday=round(sum(weekday_vals) / len(weekday_vals), 2) if weekday_vals else 0.0,
        avg_kg_weekend=round(sum(weekend_vals) / len(weekend_vals), 2) if weekend_vals else 0.0,
        best_day=best[0] if best else None,
        best_day_kg=round(best[1], 2) if best else 0.0,
        best_day_revenue=round(rev_by_day[best[0]], 2) if best else 0.0,
    )


def trailing_avg_kg_per_day(sku: str, as_of: Optional[date] = None,
                            days: int = DEFAULT_TREND_WINDOW_DAYS) -> float:
    """Average kg/open-day over the `days`-long window ending at as_of."""
    as_of = as_of or latest_sales_date()
    return sales_summary(sku, start=as_of - timedelta(days=days - 1), end=as_of).avg_kg_per_open_day


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

@dataclass
class SalesTrend:
    sku: str
    recent_start: date
    recent_end: date
    prior_start: date
    prior_end: date
    recent_avg_kg_per_day: float
    prior_avg_kg_per_day: float
    change_pct: Optional[float]
    direction: str
    recent_avg_seasonal_multiplier: float
    prior_avg_seasonal_multiplier: float
    seasonality_explains_change: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("recent_start", "recent_end", "prior_start", "prior_end"):
            d[k] = getattr(self, k).isoformat()
        return d


# A trend inside +/-5% is noise, not a signal: generated sales carry a 12%
# day-to-day coefficient of variation (history_generator.SALES_NOISE_CV), so
# a 28-day mean still wobbles by a couple of percent on chance alone.
FLAT_TREND_BAND = 0.05


def sales_trend(sku: str, as_of: Optional[date] = None,
                window_days: int = DEFAULT_TREND_WINDOW_DAYS) -> SalesTrend:
    """
    Recent window vs. the window immediately before it.

    Also reports each window's average seasonal multiplier, and flags when
    the sales change is mostly just the calendar. Without that, the agent
    would happily narrate "chuck roast is up 40%!" for what is really the
    Christmas window doing its confirmed 1.9x thing - a real analytical
    error, not a cosmetic one.
    """
    as_of = as_of or latest_sales_date()
    recent_start = as_of - timedelta(days=window_days - 1)
    prior_end = recent_start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=window_days - 1)

    recent = sales_summary(sku, recent_start, as_of).avg_kg_per_open_day
    prior = sales_summary(sku, prior_start, prior_end).avg_kg_per_open_day

    change = (recent - prior) / prior if prior > 0 else None
    if change is None:
        direction = "unknown"
    elif change > FLAT_TREND_BAND:
        direction = "up"
    elif change < -FLAT_TREND_BAND:
        direction = "down"
    else:
        direction = "flat"

    recent_mult = _avg_multiplier(recent_start, as_of)
    prior_mult = _avg_multiplier(prior_start, prior_end)
    seasonal_change = (recent_mult - prior_mult) / prior_mult if prior_mult else 0.0

    # "Seasonality explains it" = the calendar moved at least as far as sales
    # did, in the same direction.
    explains = (
        change is not None
        and abs(change) > FLAT_TREND_BAND
        and (change * seasonal_change) > 0
        and abs(seasonal_change) >= abs(change) * 0.8
    )

    return SalesTrend(
        sku=sku,
        recent_start=recent_start,
        recent_end=as_of,
        prior_start=prior_start,
        prior_end=prior_end,
        recent_avg_kg_per_day=recent,
        prior_avg_kg_per_day=prior,
        change_pct=round(change, 4) if change is not None else None,
        direction=direction,
        recent_avg_seasonal_multiplier=round(recent_mult, 3),
        prior_avg_seasonal_multiplier=round(prior_mult, 3),
        seasonality_explains_change=explains,
    )


def _avg_multiplier(start: date, end: date) -> float:
    days, total = 0, 0.0
    d = start
    while d <= end:
        total += week_multiplier(d)
        days += 1
        d += timedelta(days=1)
    return total / days if days else 0.0


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

@dataclass
class MoverRow:
    sku: str
    product_name: str
    total_kg: float
    total_revenue: float
    share_of_revenue_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def top_movers(start: Optional[date] = None, end: Optional[date] = None,
               limit: int = 5, by: str = "revenue") -> list[MoverRow]:
    """Rank products over a window by total revenue or total kg."""
    if by not in ("revenue", "kg"):
        raise ValueError("by must be 'revenue' or 'kg'")
    end = end or latest_sales_date()
    start = start or (end - timedelta(days=DEFAULT_TREND_WINDOW_DAYS - 1))
    window = _sales_window(start, end)

    kg: dict[str, float] = {}
    rev: dict[str, float] = {}
    for row in window.itertuples(index=False):
        kg[row.sku] = kg.get(row.sku, 0.0) + units_to_kg(row.sku, float(row.units_sold))
        rev[row.sku] = rev.get(row.sku, 0.0) + float(row.revenue)

    total_rev = sum(rev.values())
    rows = [
        MoverRow(
            sku=sku,
            product_name=products_by_sku[sku].name,
            total_kg=round(kg[sku], 2),
            total_revenue=round(rev[sku], 2),
            share_of_revenue_pct=round(rev[sku] / total_rev, 4) if total_rev else 0.0,
        )
        for sku in rev
    ]
    rows.sort(key=lambda r: r.total_revenue if by == "revenue" else r.total_kg, reverse=True)
    return rows[:limit]


def weekly_revenue_series(start: Optional[date] = None,
                          end: Optional[date] = None) -> list[dict]:
    """
    Whole-catalog revenue bucketed into 7-day periods counting back from `end`.

    Buckets count backwards so the most recent one is always a full week.
    Partial buckets are DROPPED, not returned short: when the requested window
    is not a multiple of seven, the oldest bucket covers only a day or two, and
    plotting it beside full weeks draws a cliff at the left edge of the chart
    that looks like a collapse in trade. A week that isn't a week is worse than
    a missing point.
    """
    end = end or latest_sales_date()
    start = start or (end - timedelta(days=90))
    window = _sales_window(start, end)

    buckets: dict[date, float] = {}
    for row in window.itertuples(index=False):
        offset = (end - row.sale_date).days // 7
        bucket_end = end - timedelta(days=offset * 7)
        buckets[bucket_end] = buckets.get(bucket_end, 0.0) + float(row.revenue)

    return [
        {"week_ending": d.isoformat(), "revenue": round(v, 2)}
        for d, v in sorted(buckets.items())
        if d - timedelta(days=6) >= start  # full weeks only
    ]


# ---------------------------------------------------------------------------
# Whole-catalog performance
# ---------------------------------------------------------------------------

# A product needs to move more than this share of catalog revenue before
# "it is declining" is worth acting on. Below it, a large percentage swing
# is a rounding artefact on a tiny base.
# ASSUMPTION: a reporting convention, not a user-confirmed number.
MATERIAL_REVENUE_SHARE = 0.01

# Shortest window that can support a BUY/DON'T-BUY recommendation.
# Generated sales carry a 12% day-to-day coefficient of variation
# (history_generator.SALES_NOISE_CV), so a one- or two-day window swings
# tens of percent on chance alone. Asking "how did we do yesterday" is a
# perfectly good question; answering it with "so order less short ribs" is
# advice built on noise. Below this span the analysis still reports what
# sold, but every verdict is downgraded to "watch" and comparison_reliable
# is False. ASSUMPTION: two weeks is the shortest span that covers both
# weekend patterns twice; tune against real variance.
MIN_COMPARISON_DAYS = 14


@dataclass
class ProductPerformance:
    """One product's trade over a window, versus the window before it."""
    sku: str
    product_name: str
    total_kg: float
    total_revenue: float
    share_of_revenue_pct: float
    avg_kg_per_open_day: float
    prior_avg_kg_per_open_day: float
    change_pct: Optional[float]
    direction: str                    # up | down | flat | unknown
    seasonality_explains_change: bool
    verdict: str                      # focus | watch | reduce | steady
    comparison_reliable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _aggregate_by_sku(start: date, end: date, weekends_only: bool = False):
    """kg, revenue and the set of open days per SKU over a window, one pass."""
    kg: dict[str, float] = {}
    rev: dict[str, float] = {}
    days: dict[str, set] = {}
    for row in _sales_window(start, end).itertuples(index=False):
        if weekends_only and not is_weekend(row.sale_date):
            continue
        kg[row.sku] = kg.get(row.sku, 0.0) + units_to_kg(row.sku, float(row.units_sold))
        rev[row.sku] = rev.get(row.sku, 0.0) + float(row.revenue)
        days.setdefault(row.sku, set()).add(row.sale_date)
    return kg, rev, days


def catalog_performance(start: Optional[date] = None, end: Optional[date] = None,
                        weekends_only: bool = False) -> list[ProductPerformance]:
    """
    Every product's trade over a window, compared with the equal-length window
    before it, with a recommended action.

    This is what "which items are doing well, and which should we order less
    of" needs. A ranking alone cannot answer it: the biggest seller can still
    be shrinking, and the smallest can still be the one to push.

    The verdict is deterministic and decided here, not by the narration layer:
      - focus  : growing, on a share big enough for the growth to matter
      - reduce : shrinking on a material share and NOT explained by the
                 calendar - a real demand fall, so order less
      - watch  : moving, but either immaterial or explained by seasonality
      - steady : inside the flat band

    Seasonality is checked before recommending a cut, because telling an
    operator to pull back on ribeye when the real cause is that Christmas
    just ended is precisely the wrong call.
    """
    end = end or latest_sales_date()
    start = start or (end - timedelta(days=DEFAULT_TREND_WINDOW_DAYS - 1))
    span = (end - start).days + 1
    prior_end = start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=span - 1)

    reliable = span >= MIN_COMPARISON_DAYS

    kg, rev, days = _aggregate_by_sku(start, end, weekends_only)
    prior_kg, _, prior_days = _aggregate_by_sku(prior_start, prior_end, weekends_only)

    total_rev = sum(rev.values())
    recent_mult = _avg_multiplier(start, end)
    prior_mult = _avg_multiplier(prior_start, prior_end)
    seasonal_change = ((recent_mult - prior_mult) / prior_mult) if prior_mult else 0.0

    rows: list[ProductPerformance] = []
    for sku in sorted(rev, key=lambda s: -rev[s]):
        # Compare average daily rate, not totals: the prior window can hold a
        # different number of open days, and comparing raw totals across
        # unequal day counts invents a trend that is not there.
        recent_rate = kg[sku] / len(days[sku]) if days.get(sku) else 0.0
        prior_rate = (prior_kg.get(sku, 0.0) / len(prior_days[sku])
                      if prior_days.get(sku) else 0.0)
        change = (recent_rate - prior_rate) / prior_rate if prior_rate > 0 else None

        if change is None:
            direction = "unknown"
        elif change > FLAT_TREND_BAND:
            direction = "up"
        elif change < -FLAT_TREND_BAND:
            direction = "down"
        else:
            direction = "flat"

        explains = (
            change is not None
            and abs(change) > FLAT_TREND_BAND
            and (change * seasonal_change) > 0
            and abs(seasonal_change) >= abs(change) * 0.8
        )
        share = rev[sku] / total_rev if total_rev else 0.0
        material = share >= MATERIAL_REVENUE_SHARE

        if not reliable:
            # Too short a window to act on. Report the movement, withhold
            # the recommendation.
            verdict = "watch" if direction in ("up", "down") else "steady"
        elif direction == "up" and material and not explains:
            verdict = "focus"
        elif direction == "down" and material and not explains:
            verdict = "reduce"
        elif direction in ("up", "down"):
            verdict = "watch"
        else:
            verdict = "steady"

        rows.append(ProductPerformance(
            sku=sku,
            product_name=products_by_sku[sku].name,
            total_kg=round(kg[sku], 2),
            total_revenue=round(rev[sku], 2),
            share_of_revenue_pct=round(share, 4),
            avg_kg_per_open_day=round(recent_rate, 2),
            prior_avg_kg_per_open_day=round(prior_rate, 2),
            change_pct=round(change, 4) if change is not None else None,
            direction=direction,
            seasonality_explains_change=explains,
            verdict=verdict,
            comparison_reliable=reliable,
        ))
    return rows


def slow_movers(start: Optional[date] = None, end: Optional[date] = None,
                limit: int = 5, by: str = "revenue") -> list[MoverRow]:
    """
    The WORST-selling products over a window.

    Not simply top_movers reversed at the call site: a product that sold
    nothing at all has no rows in the sales table, so reversing a ranking
    built from those rows would hide exactly the items an operator most needs
    to see. Every catalog product is materialised here, at zero if absent.
    """
    if by not in ("revenue", "kg"):
        raise ValueError("by must be 'revenue' or 'kg'")
    end = end or latest_sales_date()
    start = start or (end - timedelta(days=DEFAULT_TREND_WINDOW_DAYS - 1))

    kg, rev, _ = _aggregate_by_sku(start, end)
    total_rev = sum(rev.values())
    rows = [
        MoverRow(
            sku=sku,
            product_name=products_by_sku[sku].name,
            total_kg=round(kg.get(sku, 0.0), 2),
            total_revenue=round(rev.get(sku, 0.0), 2),
            share_of_revenue_pct=round(rev.get(sku, 0.0) / total_rev, 4) if total_rev else 0.0,
        )
        for sku in products_by_sku
    ]
    rows.sort(key=lambda r: r.total_revenue if by == "revenue" else r.total_kg)
    return rows[:limit]


# ---------------------------------------------------------------------------
# Weekday / weekend split
# ---------------------------------------------------------------------------

@dataclass
class WeekendUplift:
    """How much heavier a weekend day trades than a weekday, from history."""
    sku: Optional[str]
    product_name: Optional[str]
    start: date
    end: date
    weekday_avg_kg: float
    weekend_avg_kg: float
    uplift_pct: Optional[float]
    implied_multiplier: Optional[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d


def weekend_uplift(sku: Optional[str] = None, start: Optional[date] = None,
                   end: Optional[date] = None) -> WeekendUplift:
    """
    Observed weekend-vs-weekday trading, catalog-wide or for one product.

    Answers "how much more should we produce for weekends" from what the shop
    actually sold, rather than from a seasonal multiplier: the confirmed
    multipliers cover Christmas, long weekends and summer, and none of them
    describes an ordinary Saturday.
    """
    end = end or latest_sales_date()
    start = start or (end - timedelta(days=89))
    window = _sales_window(start, end, sku)

    weekday_by_day: dict[date, float] = {}
    weekend_by_day: dict[date, float] = {}
    for row in window.itertuples(index=False):
        bucket = weekend_by_day if is_weekend(row.sale_date) else weekday_by_day
        bucket[row.sale_date] = bucket.get(row.sale_date, 0.0) + units_to_kg(
            row.sku, float(row.units_sold))

    wd = sum(weekday_by_day.values()) / len(weekday_by_day) if weekday_by_day else 0.0
    we = sum(weekend_by_day.values()) / len(weekend_by_day) if weekend_by_day else 0.0
    uplift = (we - wd) / wd if wd > 0 else None

    return WeekendUplift(
        sku=sku,
        product_name=products_by_sku[sku].name if sku else None,
        start=start,
        end=end,
        weekday_avg_kg=round(wd, 2),
        weekend_avg_kg=round(we, 2),
        uplift_pct=round(uplift, 4) if uplift is not None else None,
        implied_multiplier=round(we / wd, 3) if wd > 0 else None,
    )
