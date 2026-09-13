"""
Generates the supply-side and execution-side history:

- PurchaseOrder:     incoming raw primal, with lead times, delays and short shipments
- ActualProduction:  what each scheduled cutting run actually produced

FLAGGED ASSUMPTIONS (none of these were confirmed with the project owner;
they are the knobs to correct first against real vendor and production
records, and they are stated here for the same reason SALES_NOISE_CV and
STOCKOUT_PROB are stated in history_generator.py):

- Lead times of 2-5 days by supplier (LEAD_TIME_DAYS)
- 8% of orders arrive late, by 1-3 days (LATE_PROB, LATE_DAYS)
- 6% of orders arrive short, at 70-95% of the ordered quantity (SHORT_PROB)
- Undisrupted production lands at ~100% of plan, sd 3% (EXECUTION_*)
- 7% of cutting runs hit a disruption (DISRUPTION_PROB)

---------------------------------------------------------------------------
WHY PURCHASE ORDERS ARE DERIVED FROM THE STOCK SERIES, NOT GENERATED BESIDE IT
---------------------------------------------------------------------------

primal_stock.csv already encodes every delivery that ever happened: any day
a primal's box count goes UP, something arrived. Generating a second,
independent random stream of purchase orders would have produced a file that
looks plausible on its own and contradicts the stock series everywhere - POs
landing on days stock did not move, deliveries in the stock series with no
PO behind them. Every downstream "projected inventory = on hand + incoming"
calculation would then be built on two sources that disagree.

So deliveries are READ OUT of the stock series (a positive day-over-day
delta, converted to kg at that primal's reference box weight) and a purchase
order is back-filled for each one: order_date = arrival - lead_time, and
received_kg = exactly the kg that appeared. `quantity_kg` (what was ORDERED)
is then >= received_kg, and the gap is the short shipment. The two files
cannot disagree, because one is computed from the other.

The only genuinely NEW orders are the open ones past the end of the history
(see `generate_open_orders`), which is the part the projected-inventory
equation actually needs and which by definition cannot be read out of a
stock series that has not happened yet.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.catalog import primal_reference_box_weight_kg, yield_profiles_by_primal
from models.domain import (
    ActualProduction, ProductionStatus, PurchaseOrder, PurchaseOrderStatus,
)
from simulation.seasonality import is_closed

RNG_SEED = 4242

# --- Vendor assumptions -----------------------------------------------------
# Three suppliers, split the way a small shop actually buys: one beef house,
# one pork house, one specialty importer for the wagyu. ASSUMPTION.
SUPPLIERS: dict[str, str] = {
    "beef": "Northlake Beef Co.",
    "pork": "Riverside Pork Supply",
    "specialty": "Kuroge Specialty Imports",
}

LEAD_TIME_DAYS: dict[str, int] = {"beef": 2, "pork": 3, "specialty": 5}

LATE_PROB = 0.08
LATE_DAYS = (1, 3)
SHORT_PROB = 0.06
SHORT_FILL_RANGE = (0.70, 0.95)

# Minimum box-count movement that counts as a delivery rather than rounding
# noise in the stock series (which is stored to one decimal place).
MIN_DELIVERY_BOXES = 0.05

# Shops order slightly more than they consume, so the cooler drifts up rather
# than running to exactly zero between deliveries. 8% over replacement.
# ASSUMPTION - the knob that sets how comfortable the forward book looks.
ORDER_BUFFER = 1.08

# --- Production execution assumptions ---------------------------------------
# An undisrupted run lands NEAR plan, scattered either side of it: a cutter
# working a box takes what the box gives, so finishing at 98% or 102% of a
# planned kg figure is a completed run, not a failure. Centring this below
# 1.0 would model a shop that chronically under-cuts and would slowly starve.
EXECUTION_MEAN = 1.0
EXECUTION_SD = 0.03
DISRUPTION_PROB = 0.07
DISRUPTION_REASONS = ("labor_short", "equipment", "quality_hold")
DISRUPTION_SEVERITY = (0.45, 0.85)   # fraction of plan achieved when disrupted

# STATUS IS DRIVEN BY CAUSE, NOT BY A RATIO.
#
# An earlier version classified any run under 99% of plan as PARTIAL, which
# made 4,373 of 6,052 runs "partial" - normal weight variance was being
# recorded as a failed cutting run, and the status column carried no signal
# at all because almost every row had the same value. A run is partial
# because SOMETHING WENT WRONG (it has a shortfall_reason); the threshold
# below is only a backstop for a disruption severe enough that calling it
# completed would be absurd.
COMPLETED_THRESHOLD = 0.90


def supplier_for(primal: str) -> tuple[str, str]:
    """(supplier_key, supplier_name) for a primal."""
    lowered = primal.lower()
    if "wagyu" in lowered:
        key = "specialty"
    elif "pork" in lowered:
        key = "pork"
    else:
        key = "beef"
    return key, SUPPLIERS[key]


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------------
# Deliveries implied by the stock series
# ---------------------------------------------------------------------------

def implied_deliveries(stock: pd.DataFrame) -> list[tuple[str, date, float]]:
    """
    Every (primal, date, boxes_received) the stock history implies.

    A delivery is a positive day-over-day change in boxes_on_hand. Stock is
    consumed every open day, so an increase can only mean something arrived
    - and the size of the increase understates the delivery by exactly that
    day's consumption, which is fine: this reconstructs the NET arrival the
    stock series actually recorded, and net is what the inventory equation
    needs to stay self-consistent.
    """
    out: list[tuple[str, date, float]] = []
    for primal, group in stock.sort_values(["source_primal", "as_of"]).groupby("source_primal"):
        previous = None
        for row in group.itertuples(index=False):
            if previous is not None:
                delta = float(row.boxes_on_hand) - previous
                if delta >= MIN_DELIVERY_BOXES:
                    out.append((str(primal), row.as_of, delta))
            previous = float(row.boxes_on_hand)
    return out


def generate_purchase_orders(stock: pd.DataFrame,
                             seed: int = RNG_SEED) -> list[PurchaseOrder]:
    """
    One delivered PurchaseOrder per delivery the stock series implies.

    received_kg is exactly the kg that appeared in the cooler, so this file
    reconciles against primal_stock.csv by construction. quantity_kg is what
    was ordered, which is larger on a short shipment.
    """
    rng = np.random.default_rng(seed)
    box_kg = primal_reference_box_weight_kg()
    orders: list[PurchaseOrder] = []

    for i, (primal, arrival, boxes) in enumerate(implied_deliveries(stock)):
        received_kg = boxes * box_kg.get(primal, 20.0)
        if received_kg <= 0:
            continue

        key, supplier = supplier_for(primal)
        lead = LEAD_TIME_DAYS[key]

        late = rng.random() < LATE_PROB
        days_late = int(rng.integers(LATE_DAYS[0], LATE_DAYS[1] + 1)) if late else 0
        expected = arrival - timedelta(days=days_late)
        order_date = expected - timedelta(days=lead)

        short = rng.random() < SHORT_PROB
        if short:
            fill = float(rng.uniform(*SHORT_FILL_RANGE))
            ordered_kg = received_kg / fill
            status = PurchaseOrderStatus.SHORT_SHIPPED
        else:
            ordered_kg = received_kg
            status = PurchaseOrderStatus.DELIVERED

        orders.append(PurchaseOrder(
            po_id=f"PO-{arrival.isoformat().replace('-', '')}-{i:05d}",
            source_primal=primal,
            order_date=order_date,
            expected_arrival=expected,
            quantity_kg=round(ordered_kg, 2),
            status=status,
            supplier=supplier,
            actual_arrival=arrival,
            received_kg=round(received_kg, 2),
        ))

    return orders


def generate_open_orders(stock: pd.DataFrame, as_of: date,
                         horizon_days: int = 16,
                         seed: int = RNG_SEED + 1) -> list[PurchaseOrder]:
    """
    Orders placed but not yet arrived as of the end of the history.

    This is the part that cannot be derived - it has not happened - and the
    part the projected-inventory equation is built on.

    ORDERS ARE SIZED FROM CONSUMPTION, NOT FROM OBSERVED STOCK DELTAS.
    An earlier version averaged the primal's recent implied deliveries and
    projected that size forward. That was wrong in a quiet way:
    implied_deliveries() reads a NET day-over-day increase, which is the
    delivery minus that same day's consumption, so the averages it produces
    are systematically smaller than the deliveries that actually arrived.
    Sizing forward orders from them under-supplied every primal - incoming
    ran at 37-84% of forecast draw and eight of seventeen primals projected
    a stockout inside two weeks, which would describe a shop going out of
    business rather than one in steady state.

    A shop orders to REPLACE WHAT IT USES, so the quantity is the primal's
    measured daily consumption multiplied by the days between deliveries,
    plus a small buffer.

    That consumption is DE-SEASONALISED first, for the same reason
    seasonal_planning.seasonal_plan de-seasonalises its baseline. The history
    ends on 31 December, so the trailing 28 days are the Christmas run-up
    and average ~1.3x a normal day. Sizing January's orders off December's
    consumption over-orders by roughly a third - measured, it inflated every
    cooler by 24-62% across the horizon, which reads as a shop stockpiling
    for no reason. Dividing each day by the multiplier that applied on that
    day gives a normal-day rate to order against.
    """
    from simulation.inventory_calc import primal_kg_consumed_by_day  # local: avoids a cycle
    from simulation.seasonality import week_multiplier

    rng = np.random.default_rng(seed)
    box_kg = primal_reference_box_weight_kg()

    recent_start = as_of - timedelta(days=27)
    recent = [(p, d, b) for p, d, b in implied_deliveries(stock)
              if recent_start <= d <= as_of]

    dates_by_primal: dict[str, list[date]] = defaultdict(list)
    for primal, d, _boxes in recent:
        dates_by_primal[primal].append(d)

    # De-seasonalised mean daily draw per primal, over open days only.
    consumed = primal_kg_consumed_by_day(recent_start, as_of)
    totals: dict[str, float] = defaultdict(float)
    open_days: dict[str, set] = defaultdict(set)
    for (primal, d), kg in consumed.items():
        totals[primal] += kg / (week_multiplier(d) or 1.0)
        open_days[primal].add(d)
    daily_draw = {p: totals[p] / len(open_days[p]) for p in totals if open_days[p]}

    orders: list[PurchaseOrder] = []
    counter = 0
    for primal in sorted(dates_by_primal):
        per_day = daily_draw.get(primal, 0.0)
        if per_day <= 0:
            continue
        # Deliveries per open day over the window -> days between deliveries.
        cadence = max(1, round(28 / max(1, len(dates_by_primal[primal]))))

        key, supplier = supplier_for(primal)
        lead = LEAD_TIME_DAYS[key]

        arrival = as_of + timedelta(days=cadence)
        while (arrival - as_of).days <= horizon_days:
            if is_closed(arrival):
                arrival += timedelta(days=1)
                continue
            qty_kg = (per_day * cadence * ORDER_BUFFER
                      * float(rng.uniform(0.9, 1.1)))
            order_date = arrival - timedelta(days=lead)

            delayed = rng.random() < LATE_PROB
            status = (PurchaseOrderStatus.DELAYED if delayed
                      else PurchaseOrderStatus.IN_TRANSIT
                      if order_date <= as_of
                      else PurchaseOrderStatus.ORDERED)

            counter += 1
            orders.append(PurchaseOrder(
                po_id=f"PO-OPEN-{arrival.isoformat().replace('-', '')}-{counter:04d}",
                source_primal=primal,
                order_date=order_date,
                expected_arrival=arrival,
                quantity_kg=round(qty_kg, 2),
                status=status,
                supplier=supplier,
                actual_arrival=None,
                received_kg=None,
            ))
            arrival += timedelta(days=cadence)

    return orders


# ---------------------------------------------------------------------------
# Actual production against the schedule
# ---------------------------------------------------------------------------

def generate_actual_production(schedule: pd.DataFrame, stock: pd.DataFrame,
                               seed: int = RNG_SEED + 2) -> list[ActualProduction]:
    """
    What each scheduled cutting run actually produced.

    Execution is not independent noise - it is CAUSED, which is the point of
    generating it at all:

      - a primal that was at zero boxes that morning cannot be cut, so the
        run is starved and the reason is 'stock_short'. This is read from the
        real stock series, so a stockout in primal_stock.csv produces a
        matching production failure rather than the two files telling
        different stories about the same day.
      - otherwise a run mostly lands near plan, with an occasional
        disruption drawn from DISRUPTION_REASONS.

    That causal link is what makes the execution rate worth measuring: it
    carries a signal (which primals are unreliable, and why) rather than
    being a constant with noise on top.
    """
    rng = np.random.default_rng(seed)

    stock_lookup: dict[tuple[str, date], float] = {
        (str(row.source_primal), row.as_of): float(row.boxes_on_hand)
        for row in stock.itertuples(index=False)
    }

    records: list[ActualProduction] = []
    for row in schedule.sort_values(["schedule_date", "source_primal"]).itertuples(index=False):
        d = row.schedule_date
        primal = str(row.source_primal)
        planned = float(row.planned_primal_kg)
        if planned <= 0:
            continue

        on_hand = stock_lookup.get((primal, d))
        reason = None

        if on_hand is not None and on_hand <= 0.0:
            # Nothing in the cooler: the run cannot happen at all.
            rate = 0.0
            reason = "stock_short"
        elif rng.random() < DISRUPTION_PROB:
            rate = float(rng.uniform(*DISRUPTION_SEVERITY))
            reason = str(rng.choice(DISRUPTION_REASONS))
        else:
            # Undisrupted: natural variance either side of plan.
            rate = max(0.0, float(rng.normal(EXECUTION_MEAN, EXECUTION_SD)))

        actual = planned * rate
        if rate <= 0:
            status = ProductionStatus.CANCELLED
        elif reason is None and rate >= COMPLETED_THRESHOLD:
            status = ProductionStatus.COMPLETED
        else:
            status = ProductionStatus.PARTIAL

        records.append(ActualProduction(
            production_date=d,
            source_primal=primal,
            planned_primal_kg=round(planned, 2),
            actual_primal_kg=round(actual, 2),
            status=status,
            shortfall_reason=reason,
        ))

    return records


# ---------------------------------------------------------------------------
# Schedule status / priority backfill
# ---------------------------------------------------------------------------

def schedule_priority(primal: str, tiers: dict[str, str]) -> str:
    """High for the primals carrying the shop's volume, low for the slowest.

    Derived from the velocity tier rather than hand-assigned, so it cannot
    drift out of step with what actually sells.
    """
    tier = tiers.get(primal)
    tier_value = getattr(tier, "value", tier)
    return {"best": "high", "medium": "normal", "low": "low"}.get(tier_value, "normal")


def primal_cut_paths(primal: str) -> list[str]:
    """SKUs a primal feeds - used to explain a schedule row in the CSV."""
    return [yp.product_sku for yp in yield_profiles_by_primal().get(primal, [])]
