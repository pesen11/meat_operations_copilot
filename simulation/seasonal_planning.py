"""
Seasonal production planning - "how much more should we cut for Christmas?"

This closes a real gap. Until now the system could only evaluate a scenario
the operator proposed ("what if we increase chuck roast 15%"); it could not
answer the question a shop actually asks first, which is how much to increase
in the first place, and for which period.

**The uplift is not invented and not chosen by the model.** It is the shop's
own confirmed seasonal multiplier from simulation/seasonality.py - Christmas
1.9x, long weekend 1.7x, summer 1.4x, regular winter week 1.0x. Those numbers
came from the project owner and are already unit-tested. This module applies
them across the catalog and reports the consequences, exactly as the
what-if path does.

Everything here is deterministic. Same rule as the rest of simulation/.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from typing import Optional

from data.catalog import (
    products_by_sku, yield_profile_by_sku, primal_reference_box_weight_kg,
)
from simulation import ops_data
from simulation.inventory_calc import (
    DEFAULT_LOOKBACK_DAYS, avg_weekly_cutter_minutes, required_cutting_minutes_for_day,
    stock_position,
)
from simulation.seasonality import (
    CHRISTMAS_WEEK_MULTIPLIER, LONG_WEEKEND_MULTIPLIER, SUMMER_MULTIPLIER,
    WINTER_MULTIPLIER, is_christmas_period, is_long_weekend_period, is_summer,
    week_multiplier,
)
from simulation.units import units_to_kg
from simulation.yield_calc import labor_minutes_per_pack

# The named periods an operator can plan for, with the CONFIRMED multiplier
# each one carries. Keys are what the intent parser resolves phrasing onto.
PERIODS: dict[str, float] = {
    "christmas": CHRISTMAS_WEEK_MULTIPLIER,      # 1.9x - confirmed
    "long_weekend": LONG_WEEKEND_MULTIPLIER,     # 1.7x - confirmed
    "summer": SUMMER_MULTIPLIER,                 # 1.4x - confirmed
    "regular": WINTER_MULTIPLIER,                # 1.0x baseline
}

_DETECTORS = {
    "christmas": is_christmas_period,
    "long_weekend": is_long_weekend_period,
    "summer": is_summer,
}

# How far ahead to look for the next occurrence of a period. 400 days covers
# a full year plus slack, so "next Christmas" resolves from any date.
_SEARCH_HORIZON_DAYS = 400


def period_window(period: str, from_date: date) -> Optional[tuple[date, date]]:
    """
    The next date range matching `period`, searching forward from `from_date`.

    Returns None for "regular" (no specific window) and when no occurrence
    falls inside the search horizon.
    """
    detector = _DETECTORS.get(period)
    if detector is None:
        return None

    start: Optional[date] = None
    previous = False
    for offset in range(_SEARCH_HORIZON_DAYS):
        day = from_date + timedelta(days=offset)
        inside = detector(day)
        if inside and not previous:
            start = day
        if previous and not inside and start is not None:
            return start, day - timedelta(days=1)
        previous = inside
    if start is not None:
        return start, from_date + timedelta(days=_SEARCH_HORIZON_DAYS - 1)
    return None


@dataclass
class ProductUplift:
    product_sku: str
    product_name: str
    source_primal: str
    # Normal-week equivalent: the lookback window de-seasonalised, so the
    # uplift below is applied to a regular week rather than to whatever
    # season the window happened to span.
    baseline_weekly_kg: float
    planned_weekly_kg: float
    extra_weekly_kg: float
    extra_weekly_packs: float
    extra_primal_kg: float
    extra_cutting_minutes: float
    extra_trim_waste_kg: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PrimalUplift:
    source_primal: str
    extra_primal_kg: float
    extra_boxes: float
    boxes_on_hand: float
    normal_target_boxes_high: float
    boxes_to_order: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeasonalPlan:
    period: str
    multiplier: float
    multiplier_source: str
    window_start: Optional[str]
    window_end: Optional[str]
    days_until: Optional[int]
    as_of: str
    # How the baseline was derived - surfaced so a reader knows the uplift is
    # applied to a normal week, not to whatever season the lookback spanned.
    baseline_basis: str = ("de-seasonalised normal week: each day's sales "
                           "divided by its own confirmed multiplier")

    products: list[ProductUplift] = field(default_factory=list)
    primals: list[PrimalUplift] = field(default_factory=list)

    total_extra_weekly_kg: float = 0.0
    total_extra_primal_kg: float = 0.0
    total_extra_boxes: float = 0.0
    # Baseline and planned totals are reported alongside the delta, not just
    # the delta: every consumer (narration, API, the UI's outcome cards) shows
    # before-and-after, and a block carrying only the delta forces each of
    # them to reconstruct the other two.
    total_baseline_trim_waste_kg: float = 0.0
    total_planned_trim_waste_kg: float = 0.0
    total_extra_trim_waste_kg: float = 0.0

    baseline_weekly_cutting_minutes: float = 0.0
    planned_weekly_cutting_minutes: float = 0.0
    weekly_cutter_minutes_available: float = 0.0
    baseline_cutter_utilization_pct: float = 0.0
    planned_cutter_utilization_pct: float = 0.0
    exceeds_cutter_capacity: bool = False
    extra_cutter_minutes_needed: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["products"] = [p.to_dict() for p in self.products]
        d["primals"] = [p.to_dict() for p in self.primals]
        return d


def resolve_period(text: str) -> Optional[str]:
    """Map operator phrasing onto a named period. Returns None if none matches."""
    lowered = text.lower()
    # "the holidays" means Christmas in this shop's calendar — it is the
    # only period with its own confirmed multiplier that operators refer to
    # that way. A stat-holiday long weekend is said as "long weekend".
    if any(w in lowered for w in ("christmas", "xmas", "holiday season",
                                  "the holidays", "holidays", "festive",
                                  "december")):
        return "christmas"
    if any(w in lowered for w in ("long weekend", "long-weekend", "stat holiday",
                                  "statutory holiday", "bank holiday", "victoria day",
                                  "canada day", "labour day", "labor day",
                                  "thanksgiving")):
        return "long_weekend"
    if any(w in lowered for w in ("summer", "bbq", "barbecue", "grilling season",
                                  "july", "august")):
        return "summer"
    return None


def seasonal_plan(period: str, as_of: Optional[date] = None,
                  product_sku: Optional[str] = None,
                  lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> SeasonalPlan:
    """
    How much more to produce for a named season, and what it costs.

    Baselines come from observed history over the lookback window, so the
    answer reflects what the shop actually does. The uplift is the confirmed
    seasonal multiplier - nothing here is estimated or inferred.
    """
    if period not in PERIODS:
        raise ValueError(f"Unknown period '{period}'. "
                         f"Known periods: {', '.join(sorted(PERIODS))}")

    sales = ops_data.load_sales()
    as_of = as_of or max(sales["sale_date"])
    start = as_of - timedelta(days=lookback_days - 1)
    weeks = lookback_days / 7
    multiplier = PERIODS[period]

    window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= as_of)]
    if product_sku is not None:
        if product_sku not in products_by_sku:
            raise ValueError(f"Unknown product_sku '{product_sku}'")
        window = window[window["sku"] == product_sku]

    # DE-SEASONALISE the baseline before applying the target uplift.
    #
    # Each day's sales are divided by the multiplier that applied on THAT day,
    # producing a normal-week-equivalent baseline. Without this, asking about
    # Christmas from inside the Christmas window applies 1.9x on top of a
    # window that is already running at ~1.32x, double-counts the season, and
    # reports a capacity overrun that does not exist (measured: 119% cutter
    # utilization vs. the true 91%). Every number used here is a confirmed
    # multiplier from seasonality.py, so this is a correction, not a model.
    baseline_kg: dict[str, float] = defaultdict(float)
    deseasonalised_days: dict[str, float] = defaultdict(float)
    for row in window.itertuples(index=False):
        day_multiplier = week_multiplier(row.sale_date) or 1.0
        baseline_kg[row.sku] += units_to_kg(row.sku, float(row.units_sold)) / day_multiplier
        deseasonalised_days[row.sku] += 1.0

    box_weights = primal_reference_box_weight_kg()
    products: list[ProductUplift] = []
    primal_extra_kg: dict[str, float] = defaultdict(float)

    for sku, total_kg in sorted(baseline_kg.items()):
        product = products_by_sku[sku]
        yp = yield_profile_by_sku[sku]
        weekly = total_kg / weeks
        extra_kg = weekly * (multiplier - 1)
        extra_packs = extra_kg / product.avg_pack_weight_kg
        extra_primal = extra_kg / yp.yield_pct

        products.append(ProductUplift(
            product_sku=sku,
            product_name=product.name,
            source_primal=yp.source_primal,
            baseline_weekly_kg=round(weekly, 2),
            planned_weekly_kg=round(weekly * multiplier, 2),
            extra_weekly_kg=round(extra_kg, 2),
            extra_weekly_packs=round(extra_packs, 1),
            extra_primal_kg=round(extra_primal, 2),
            extra_cutting_minutes=round(extra_packs * labor_minutes_per_pack(yp, product), 1),
            extra_trim_waste_kg=round(extra_primal * yp.trim_waste_pct, 2),
        ))
        primal_extra_kg[yp.source_primal] += extra_primal

    primals: list[PrimalUplift] = []
    for primal, extra_kg in sorted(primal_extra_kg.items()):
        extra_boxes = extra_kg / box_weights[primal]
        try:
            position = stock_position(primal, as_of=as_of, lookback_days=lookback_days)
            on_hand, target_high = position.boxes_on_hand, position.target_boxes_high
        except ValueError:
            on_hand, target_high = 0.0, 0.0
        # Boxes to order = the extra the period needs, less whatever the
        # cooler is already carrying above its normal top-of-range.
        headroom = max(0.0, on_hand - target_high)
        primals.append(PrimalUplift(
            source_primal=primal,
            extra_primal_kg=round(extra_kg, 2),
            extra_boxes=round(extra_boxes, 2),
            boxes_on_hand=on_hand,
            normal_target_boxes_high=target_high,
            boxes_to_order=round(max(0.0, extra_boxes - headroom), 2),
        ))

    # Shop-wide cutting load always covers every product, even when the plan
    # is scoped to one: the extra work lands on the same cutters. De-seasonalised
    # for the same reason as the product baselines above.
    all_days = sorted(set(sales[(sales["sale_date"] >= start)
                                & (sales["sale_date"] <= as_of)]["sale_date"]))
    baseline_minutes = sum(
        required_cutting_minutes_for_day(d) / (week_multiplier(d) or 1.0)
        for d in all_days) / weeks
    extra_minutes = sum(p.extra_cutting_minutes for p in products)
    available = avg_weekly_cutter_minutes(as_of=as_of, lookback_days=lookback_days)
    planned_minutes = baseline_minutes + extra_minutes

    # Baseline trim waste across every product in the window, so the plan can
    # report before/after rather than only the increase.
    baseline_waste = sum(
        (row.baseline_weekly_kg / yield_profile_by_sku[row.product_sku].yield_pct)
        * yield_profile_by_sku[row.product_sku].trim_waste_pct
        for row in products)

    window_range = period_window(period, as_of + timedelta(days=1))
    days_until = (window_range[0] - as_of).days if window_range else None

    return SeasonalPlan(
        period=period,
        multiplier=multiplier,
        multiplier_source="confirmed seasonal multiplier (simulation/seasonality.py)",
        window_start=window_range[0].isoformat() if window_range else None,
        window_end=window_range[1].isoformat() if window_range else None,
        days_until=days_until,
        as_of=as_of.isoformat(),
        products=products,
        primals=primals,
        total_extra_weekly_kg=round(sum(p.extra_weekly_kg for p in products), 2),
        total_extra_primal_kg=round(sum(p.extra_primal_kg for p in products), 2),
        total_extra_boxes=round(sum(p.extra_boxes for p in primals), 2),
        total_baseline_trim_waste_kg=round(baseline_waste, 2),
        total_planned_trim_waste_kg=round(baseline_waste * multiplier, 2),
        total_extra_trim_waste_kg=round(sum(p.extra_trim_waste_kg for p in products), 2),
        baseline_weekly_cutting_minutes=round(baseline_minutes, 1),
        planned_weekly_cutting_minutes=round(planned_minutes, 1),
        weekly_cutter_minutes_available=round(available, 1),
        baseline_cutter_utilization_pct=(
            round(baseline_minutes / available, 4) if available else 0.0),
        planned_cutter_utilization_pct=(
            round(planned_minutes / available, 4) if available else 0.0),
        exceeds_cutter_capacity=planned_minutes > available,
        extra_cutter_minutes_needed=round(max(0.0, planned_minutes - available), 1),
    )


def seasonal_calendar(as_of: Optional[date] = None, horizon_days: int = 120) -> list[dict]:
    """Elevated trading periods coming up in the next `horizon_days`, each with
    its confirmed multiplier. Answers 'what's coming that we should plan for'."""
    as_of = as_of or max(ops_data.load_sales()["sale_date"])
    out: list[dict] = []
    for period in ("christmas", "long_weekend", "summer"):
        window = period_window(period, as_of + timedelta(days=1))
        if not window:
            continue
        start, end = window
        if (start - as_of).days > horizon_days:
            continue
        out.append({
            "period": period,
            "multiplier": PERIODS[period],
            "starts": start.isoformat(),
            "ends": end.isoformat(),
            "days_until": (start - as_of).days,
        })
    return sorted(out, key=lambda row: row["days_until"])


def multiplier_for_day(day: date) -> dict:
    """The seasonal multiplier that applies to one calendar day, and why."""
    return {
        "date": day.isoformat(),
        "multiplier": week_multiplier(day),
        "is_christmas_period": is_christmas_period(day),
        "is_long_weekend_period": is_long_weekend_period(day),
        "is_summer": is_summer(day),
    }


# ---------------------------------------------------------------------------
# What actually happened LAST time this period came around
# ---------------------------------------------------------------------------

@dataclass
class PeriodActuals:
    """Observed trade during the most recent occurrence of a named period.

    The planning path applies a confirmed multiplier to a de-seasonalised
    baseline. That answers "how much more should we cut". It does not answer
    "what happened last Christmas", which is the question an operator
    actually asks first - and it is answerable from history rather than from
    a multiplier, so it is answered from history here.
    """
    period: str
    window_start: str
    window_end: str
    days: int
    total_kg: float
    total_revenue: float
    avg_kg_per_day: float
    # The same figure for ordinary trading either side of the window, so the
    # observed uplift is a comparison the operator can check, not a claim.
    normal_avg_kg_per_day: float
    observed_multiplier: Optional[float]
    confirmed_multiplier: float
    top_products: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def last_occurrence_window(period: str, before: date) -> Optional[tuple[date, date]]:
    """
    The most recent COMPLETED window for `period`, searching backwards.

    period_window() looks forward to the next occurrence, for planning. This
    looks backwards, for comparison. A window still in progress at `before`
    is skipped: half a Christmas is not a Christmas to compare against.
    """
    detector = _DETECTORS.get(period)
    if detector is None:
        return None

    end: Optional[date] = None
    following = False
    for offset in range(_SEARCH_HORIZON_DAYS):
        day = before - timedelta(days=offset)
        inside = detector(day)
        if inside and not following and day < before:
            end = day
        if following and not inside and end is not None:
            return day + timedelta(days=1), end
        following = inside
    return None


def period_actuals(period: str, as_of: Optional[date] = None,
                   normal_days: int = 28, limit: int = 5) -> Optional[PeriodActuals]:
    """
    What the shop actually sold during the last occurrence of `period`.

    Returns None when the history does not cover a completed occurrence -
    reported as "no comparable period in the data" rather than answered from
    a multiplier, which would be presenting an assumption as an observation.
    """
    if period not in PERIODS:
        raise ValueError(f"Unknown period '{period}'")

    sales = ops_data.load_sales()
    as_of = as_of or max(sales["sale_date"])
    window = last_occurrence_window(period, as_of)
    if window is None:
        return None
    start, end = window

    earliest = min(sales["sale_date"])
    if end < earliest:
        return None

    in_window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= end)]
    if in_window.empty:
        return None

    kg_by_day: dict[date, float] = {}
    kg_by_sku: dict[str, float] = {}
    revenue = 0.0
    for row in in_window.itertuples(index=False):
        kg = units_to_kg(row.sku, float(row.units_sold))
        kg_by_day[row.sale_date] = kg_by_day.get(row.sale_date, 0.0) + kg
        kg_by_sku[row.sku] = kg_by_sku.get(row.sku, 0.0) + kg
        revenue += float(row.revenue)

    # Ordinary trade for comparison: the `normal_days` before the window,
    # excluding any day the seasonality model marks as elevated, so the
    # comparison is against a normal week rather than the run-up.
    normal_start = start - timedelta(days=normal_days)
    normal = sales[(sales["sale_date"] >= normal_start)
                   & (sales["sale_date"] < start)]
    normal_by_day: dict[date, float] = {}
    for row in normal.itertuples(index=False):
        if week_multiplier(row.sale_date) > WINTER_MULTIPLIER:
            continue
        normal_by_day[row.sale_date] = normal_by_day.get(row.sale_date, 0.0) + units_to_kg(
            row.sku, float(row.units_sold))

    avg = sum(kg_by_day.values()) / len(kg_by_day) if kg_by_day else 0.0
    normal_avg = sum(normal_by_day.values()) / len(normal_by_day) if normal_by_day else 0.0

    top = sorted(kg_by_sku.items(), key=lambda kv: -kv[1])[:limit]
    return PeriodActuals(
        period=period,
        window_start=start.isoformat(),
        window_end=end.isoformat(),
        days=(end - start).days + 1,
        total_kg=round(sum(kg_by_sku.values()), 2),
        total_revenue=round(revenue, 2),
        avg_kg_per_day=round(avg, 2),
        normal_avg_kg_per_day=round(normal_avg, 2),
        observed_multiplier=round(avg / normal_avg, 3) if normal_avg > 0 else None,
        confirmed_multiplier=PERIODS[period],
        top_products=[
            {"sku": sku,
             "product_name": products_by_sku[sku].name,
             "total_kg": round(kg, 2)}
            for sku, kg in top
        ],
    )
