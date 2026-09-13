"""
Forward inventory projection: will we run out, and when?

Everything before this module answered inventory questions in the present
tense - "you have 21.1 boxes, 2.49 days of cover". That is a snapshot, and
a snapshot cannot answer the question an operator actually has, which is
whether the cooler survives the next ten days given what is on order and
what is forecast to sell.

    projected primal kg
        = opening stock
        + supply expected to arrive   (open POs, discounted for reliability)
        - primal drawn down by forecast demand

---------------------------------------------------------------------------
WHY TRIM WASTE IS NOT A SEPARATE SUBTRACTION
---------------------------------------------------------------------------
It is tempting to write the equation as

    opening + incoming + production - demand - waste

but that double-counts. This ledger is in RAW PRIMAL kg, and demand is in
FINISHED PRODUCT kg, so the two are bridged by dividing by yield_pct:

    primal drawn = finished demand / yield_pct

At an 84% yield, 100kg of demand draws 119kg of primal - and the 19kg
difference IS the trim waste. Subtracting waste again on top would charge
the cooler for it twice. `trim_waste_kg` is reported in the result for
visibility, but it is derived FROM the draw, never subtracted from it.

Nor is production a separate addition. Production does not create primal;
it converts primal into finished product. It appears here only as the
consumption side, which is already what the demand draw represents.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from data.catalog import (
    primal_reference_box_weight_kg, yield_profiles_by_primal,
)
from simulation import ops_data
from simulation.demand_forecast import expected_kg_for_day
from simulation.inventory_calc import DEFAULT_LOOKBACK_DAYS, stock_position
from simulation.seasonality import is_closed

# How far forward to project by default. 14 days covers two full delivery
# cycles for every supplier (longest lead time is 5 days), so an order placed
# today has time to land inside the horizon. ASSUMPTION - a reporting choice,
# not a user-confirmed planning horizon.
DEFAULT_HORIZON_DAYS = 14

# Statuses that represent supply still expected to arrive.
OPEN_STATUSES = {"ordered", "in_transit", "delayed"}

# A projected closing position below this many days of cover is flagged even
# though it is not technically a stockout - it is the same 1.5-day line the
# recommendation node uses, kept in one place conceptually but restated here
# because this module must stand alone. ASSUMPTION.
LOW_COVER_DAYS = 1.5


@dataclass
class DayProjection:
    """One day on the forward ledger."""
    day: date
    opening_kg: float
    arriving_kg: float
    demand_primal_kg: float
    closing_kg: float
    closing_boxes: float
    is_stockout: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d


@dataclass
class SupplyProjection:
    source_primal: str
    as_of: date
    horizon_days: int
    horizon_end: date

    opening_kg: float
    opening_boxes: float
    reference_box_weight_kg: float

    incoming_scheduled_kg: float
    incoming_expected_kg: float
    incoming_order_count: int
    supply_reliability: float
    reliability_basis: str

    forecast_product_demand_kg: float
    forecast_primal_draw_kg: float
    implied_trim_waste_kg: float

    projected_closing_kg: float
    projected_closing_boxes: float
    projected_days_of_cover: Optional[float]
    avg_daily_primal_kg: float

    shortfall_kg: float
    stockout_expected: bool
    earliest_stockout_date: Optional[date]
    days_until_stockout: Optional[int]
    status: str
    daily: list[DayProjection] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["horizon_end"] = self.horizon_end.isoformat()
        d["earliest_stockout_date"] = (self.earliest_stockout_date.isoformat()
                                       if self.earliest_stockout_date else None)
        d["daily"] = [x.to_dict() for x in self.daily]
        return d


# ---------------------------------------------------------------------------
# Vendor reliability, measured
# ---------------------------------------------------------------------------

@dataclass
class SupplierReliability:
    scope: str
    orders: int
    on_time_rate: Optional[float]
    avg_fill_rate: Optional[float]
    short_shipment_rate: Optional[float]
    avg_days_late: Optional[float]
    combined_reliability: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def supplier_reliability(source_primal: Optional[str] = None,
                         supplier: Optional[str] = None,
                         lookback_days: int = 180,
                         as_of: Optional[date] = None) -> SupplierReliability:
    """
    How much of an ordered quantity actually turns up, measured.

    `combined_reliability` is fill rate alone, NOT fill x on-time. A late
    delivery still arrives - it is a timing risk the day-by-day ladder
    already models by placing it on its expected date - whereas a short
    shipment is quantity that never comes. Multiplying the two would
    double-count lateness as loss and make every projection pessimistic.
    """
    orders = ops_data.load_purchase_orders()
    as_of = as_of or _latest_stock_date()
    start = as_of - timedelta(days=lookback_days - 1)

    closed = orders[orders["actual_arrival"].notna()].copy()
    closed = closed[(closed["actual_arrival"] >= start)
                    & (closed["actual_arrival"] <= as_of)]

    scope = "all suppliers"
    if source_primal is not None:
        closed = closed[closed["source_primal"] == source_primal]
        scope = source_primal
    if supplier is not None:
        closed = closed[closed["supplier"] == supplier]
        scope = supplier

    if closed.empty:
        return SupplierReliability(scope, 0, None, None, None, None, None)

    late_days = [(row.actual_arrival - row.expected_arrival).days
                 for row in closed.itertuples(index=False)]
    on_time = sum(1 for d in late_days if d <= 0) / len(late_days)

    fills = [float(row.received_kg) / float(row.quantity_kg)
             for row in closed.itertuples(index=False)
             if row.quantity_kg and float(row.quantity_kg) > 0
             and pd.notna(row.received_kg)]
    avg_fill = sum(fills) / len(fills) if fills else None
    short_rate = (sum(1 for f in fills if f < 0.999) / len(fills)) if fills else None

    return SupplierReliability(
        scope=scope,
        orders=int(len(closed)),
        on_time_rate=round(on_time, 4),
        avg_fill_rate=round(avg_fill, 4) if avg_fill is not None else None,
        short_shipment_rate=round(short_rate, 4) if short_rate is not None else None,
        avg_days_late=round(sum(max(0, d) for d in late_days) / len(late_days), 2),
        combined_reliability=round(avg_fill, 4) if avg_fill is not None else None,
    )


# ---------------------------------------------------------------------------
# Incoming supply
# ---------------------------------------------------------------------------

def _latest_stock_date() -> date:
    return max(ops_data.load_primal_stock()["as_of"])


def open_orders(source_primal: Optional[str] = None,
                as_of: Optional[date] = None,
                horizon_days: int = DEFAULT_HORIZON_DAYS) -> list[dict]:
    """Purchase orders still expected to arrive inside the horizon."""
    orders = ops_data.load_purchase_orders()
    as_of = as_of or _latest_stock_date()
    end = as_of + timedelta(days=horizon_days)

    rows = orders[orders["status"].isin(OPEN_STATUSES)]
    rows = rows[(rows["expected_arrival"] > as_of) & (rows["expected_arrival"] <= end)]
    if source_primal is not None:
        rows = rows[rows["source_primal"] == source_primal]

    return [{
        "po_id": row.po_id,
        "source_primal": row.source_primal,
        "supplier": row.supplier,
        "order_date": row.order_date.isoformat(),
        "expected_arrival": row.expected_arrival.isoformat(),
        "quantity_kg": round(float(row.quantity_kg), 2),
        "status": row.status,
    } for row in rows.sort_values("expected_arrival").itertuples(index=False)]


def incoming_by_day(source_primal: str, as_of: date,
                    horizon_days: int) -> dict[date, float]:
    """Expected arrival kg per day, discounted for measured fill rate."""
    reliability = supplier_reliability(source_primal, as_of=as_of)
    factor = reliability.combined_reliability
    if factor is None:
        factor = 1.0

    per_day: dict[date, float] = defaultdict(float)
    orders = ops_data.load_purchase_orders()
    end = as_of + timedelta(days=horizon_days)
    rows = orders[(orders["source_primal"] == source_primal)
                  & (orders["status"].isin(OPEN_STATUSES))
                  & (orders["expected_arrival"] > as_of)
                  & (orders["expected_arrival"] <= end)]
    for row in rows.itertuples(index=False):
        per_day[row.expected_arrival] += float(row.quantity_kg) * factor
    return dict(per_day)


# ---------------------------------------------------------------------------
# Forecast draw
# ---------------------------------------------------------------------------

def forecast_primal_draw(source_primal: str, day: date,
                         demand_multiplier: float = 1.0) -> tuple[float, float]:
    """
    (finished product kg, raw primal kg) this primal must supply on `day`.

    Sums every yield path off the primal: each path's forecast finished
    demand divided by that path's own yield_pct, because two paths off one
    primal do not convert at the same rate.
    """
    if is_closed(day):
        return 0.0, 0.0
    product_kg = 0.0
    primal_kg = 0.0
    for yp in yield_profiles_by_primal().get(source_primal, []):
        expected = expected_kg_for_day(yp.product_sku, day) * demand_multiplier
        if expected <= 0:
            continue
        product_kg += expected
        primal_kg += expected / yp.yield_pct
    return product_kg, primal_kg


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------

def project_supply(source_primal: str,
                   as_of: Optional[date] = None,
                   horizon_days: int = DEFAULT_HORIZON_DAYS,
                   demand_multiplier: float = 1.0,
                   lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> SupplyProjection:
    """
    Walk the cooler forward day by day.

    A day-by-day ladder rather than one closing figure, because the closing
    figure hides the shape: a primal can finish the horizon comfortably and
    still hit zero on day four while waiting for a delivery on day five. The
    date it runs out is the actionable number, not the balance at the end.
    """
    as_of = as_of or _latest_stock_date()
    position = stock_position(source_primal, as_of=as_of, lookback_days=lookback_days)
    box_kg = primal_reference_box_weight_kg()[source_primal]

    arrivals = incoming_by_day(source_primal, as_of, horizon_days)
    reliability = supplier_reliability(source_primal, as_of=as_of)

    scheduled = sum(float(o["quantity_kg"]) for o in
                    open_orders(source_primal, as_of, horizon_days))
    expected_incoming = sum(arrivals.values())

    balance = position.kg_on_hand
    daily: list[DayProjection] = []
    total_product = 0.0
    total_draw = 0.0
    earliest_stockout: Optional[date] = None

    for offset in range(1, horizon_days + 1):
        day = as_of + timedelta(days=offset)
        arriving = arrivals.get(day, 0.0)
        product_kg, draw = forecast_primal_draw(source_primal, day, demand_multiplier)

        opening = balance
        balance = opening + arriving - draw
        stockout = balance <= 0
        if stockout and earliest_stockout is None:
            earliest_stockout = day
        # The cooler cannot go negative; it just runs dry. Letting the
        # balance go negative would silently net a later delivery against a
        # shortfall that already cost the shop sales on an earlier day.
        balance = max(0.0, balance)

        total_product += product_kg
        total_draw += draw
        daily.append(DayProjection(
            day=day,
            opening_kg=round(opening, 2),
            arriving_kg=round(arriving, 2),
            demand_primal_kg=round(draw, 2),
            closing_kg=round(balance, 2),
            closing_boxes=round(balance / box_kg, 2),
            is_stockout=stockout,
        ))

    daily_rate = position.avg_daily_primal_kg
    cover = round(balance / daily_rate, 2) if daily_rate > 0 else None

    # Shortfall = demand the horizon cannot cover from opening + incoming.
    available = position.kg_on_hand + expected_incoming
    shortfall = max(0.0, total_draw - available)

    if earliest_stockout is not None:
        status = "stockout_expected"
    elif cover is not None and cover < LOW_COVER_DAYS:
        status = "tight"
    elif position.status in ("below_target", "critical"):
        status = "below_target"
    else:
        status = "ok"

    # Trim waste implied by the draw - reported, never subtracted. See the
    # module docstring: it is already inside the yield division.
    waste = total_draw - total_product

    return SupplyProjection(
        source_primal=source_primal,
        as_of=as_of,
        horizon_days=horizon_days,
        horizon_end=as_of + timedelta(days=horizon_days),
        opening_kg=round(position.kg_on_hand, 2),
        opening_boxes=position.boxes_on_hand,
        reference_box_weight_kg=round(box_kg, 2),
        incoming_scheduled_kg=round(scheduled, 2),
        incoming_expected_kg=round(expected_incoming, 2),
        incoming_order_count=len(open_orders(source_primal, as_of, horizon_days)),
        supply_reliability=round(reliability.combined_reliability or 1.0, 4),
        reliability_basis=(f"{reliability.orders} closed orders"
                           if reliability.orders else "no closed orders; assuming full fill"),
        forecast_product_demand_kg=round(total_product, 2),
        forecast_primal_draw_kg=round(total_draw, 2),
        implied_trim_waste_kg=round(waste, 2),
        projected_closing_kg=round(balance, 2),
        projected_closing_boxes=round(balance / box_kg, 2),
        projected_days_of_cover=cover,
        avg_daily_primal_kg=position.avg_daily_primal_kg,
        shortfall_kg=round(shortfall, 2),
        stockout_expected=earliest_stockout is not None,
        earliest_stockout_date=earliest_stockout,
        days_until_stockout=((earliest_stockout - as_of).days
                             if earliest_stockout else None),
        status=status,
        daily=daily,
    )


def project_all_supply(as_of: Optional[date] = None,
                       horizon_days: int = DEFAULT_HORIZON_DAYS,
                       demand_multiplier: float = 1.0) -> list[SupplyProjection]:
    """Every primal, soonest-to-run-out first."""
    stock = ops_data.load_primal_stock()
    as_of = as_of or max(stock["as_of"])
    primals = sorted(set(stock["source_primal"]))
    out = [project_supply(p, as_of=as_of, horizon_days=horizon_days,
                          demand_multiplier=demand_multiplier) for p in primals]
    return sorted(out, key=lambda p: (
        p.days_until_stockout if p.days_until_stockout is not None else 10_000,
        p.projected_days_of_cover if p.projected_days_of_cover is not None else 10_000,
    ))
