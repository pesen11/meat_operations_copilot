"""
Tests for simulation/supply_calc.py.

The magnitude assertions here exist because of a bug this module actually
had: forward purchase orders were sized from net stock deltas, which made
incoming supply run at 37-84% of forecast draw and projected a stockout for
eight of seventeen primals inside two weeks. Every arithmetic test passed
while that was true - the ledger added up perfectly, it just described a
shop going out of business. `test_supply_roughly_balances_consumption` is
the test that would have caught it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from simulation import ops_data
from simulation.supply_calc import (
    DEFAULT_HORIZON_DAYS, forecast_primal_draw, incoming_by_day, open_orders,
    project_all_supply, project_supply, supplier_reliability,
)


@pytest.fixture(scope="module")
def as_of() -> date:
    return max(ops_data.load_primal_stock()["as_of"])


# ---------------------------------------------------------------------------
# Magnitude: does the forward book describe a shop in steady state?
# ---------------------------------------------------------------------------

def test_supply_roughly_balances_consumption(as_of):
    """A shop in steady state orders roughly what it uses. Far below 1.0 and
    it starves; far above and it is stockpiling for no reason. This is the
    assertion that catches a mis-sized forward order book."""
    for projection in project_all_supply(as_of=as_of):
        if projection.forecast_primal_draw_kg <= 0:
            continue
        ratio = projection.incoming_expected_kg / projection.forecast_primal_draw_kg
        assert 0.85 < ratio < 1.45, (
            f"{projection.source_primal}: incoming supply is {ratio:.2f}x "
            f"forecast draw over the horizon - not a steady-state shop"
        )


def test_most_primals_do_not_project_a_stockout(as_of):
    """A handful of tight primals is realistic; a majority means the forward
    book is broken, not that the shop is in trouble."""
    rows = project_all_supply(as_of=as_of)
    stockouts = [p for p in rows if p.stockout_expected]
    assert len(stockouts) <= len(rows) // 4, (
        f"{len(stockouts)} of {len(rows)} primals project a stockout"
    )


def test_closing_stock_stays_in_a_sane_box_range(as_of):
    for projection in project_all_supply(as_of=as_of):
        assert projection.projected_closing_boxes < 120, (
            f"{projection.source_primal} closes at "
            f"{projection.projected_closing_boxes} boxes - a cooler that size "
            f"does not exist in this shop"
        )


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_daily_ledger_chains(as_of):
    """Each day's opening must be the previous day's closing. If this breaks
    the ladder is not a ledger, it is a list of independent guesses."""
    projection = project_supply("Blade Eyes", as_of=as_of)
    for previous, current in zip(projection.daily, projection.daily[1:]):
        assert current.opening_kg == pytest.approx(previous.closing_kg, abs=0.02)


def test_daily_arithmetic_holds(as_of):
    projection = project_supply("Blade Eyes", as_of=as_of)
    for day in projection.daily:
        expected = day.opening_kg + day.arriving_kg - day.demand_primal_kg
        assert day.closing_kg == pytest.approx(max(0.0, expected), abs=0.02)


def test_opening_matches_current_stock_position(as_of):
    from simulation.inventory_calc import stock_position
    projection = project_supply("Blade Eyes", as_of=as_of)
    position = stock_position("Blade Eyes", as_of=as_of)
    assert projection.opening_kg == pytest.approx(position.kg_on_hand, abs=0.02)
    assert projection.daily[0].opening_kg == pytest.approx(position.kg_on_hand, abs=0.02)


def test_balance_never_goes_negative(as_of):
    """The cooler runs dry; it does not go into debt. A negative balance
    would let a later delivery silently net against a shortfall that already
    cost the shop sales."""
    for projection in project_all_supply(as_of=as_of):
        for day in projection.daily:
            assert day.closing_kg >= 0


def test_horizon_length_is_respected(as_of):
    projection = project_supply("Blade Eyes", as_of=as_of, horizon_days=7)
    assert len(projection.daily) == 7
    assert projection.horizon_end == as_of + timedelta(days=7)
    assert projection.daily[0].day == as_of + timedelta(days=1)


# ---------------------------------------------------------------------------
# Waste is reported, never subtracted
# ---------------------------------------------------------------------------

def test_trim_waste_is_the_gap_between_draw_and_demand(as_of):
    """Waste must equal primal draw minus finished demand - i.e. it is
    already inside the yield division. If it were also subtracted from the
    ledger the cooler would be charged for it twice."""
    projection = project_supply("Blade Eyes", as_of=as_of)
    assert projection.implied_trim_waste_kg == pytest.approx(
        projection.forecast_primal_draw_kg - projection.forecast_product_demand_kg,
        abs=0.05)
    assert projection.implied_trim_waste_kg > 0


def test_draw_exceeds_finished_demand(as_of):
    """Yield is always below 1.0, so raw draw must exceed finished output."""
    for projection in project_all_supply(as_of=as_of):
        if projection.forecast_product_demand_kg > 0:
            assert projection.forecast_primal_draw_kg > projection.forecast_product_demand_kg


def test_closed_days_draw_nothing(as_of):
    """New Year's Day is an Ontario stat holiday: the shop is shut, so the
    cooler must not be drawn down on it."""
    product_kg, primal_kg = forecast_primal_draw("Blade Eyes", date(2026, 1, 1))
    assert product_kg == 0.0
    assert primal_kg == 0.0


def test_demand_multiplier_scales_the_draw(as_of):
    base_product, base_primal = forecast_primal_draw("Blade Eyes", date(2026, 1, 5))
    up_product, up_primal = forecast_primal_draw("Blade Eyes", date(2026, 1, 5),
                                                 demand_multiplier=1.5)
    assert up_product == pytest.approx(base_product * 1.5, rel=1e-6)
    assert up_primal == pytest.approx(base_primal * 1.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Incoming supply
# ---------------------------------------------------------------------------

def test_open_orders_are_in_the_future_and_open(as_of):
    for order in open_orders("Blade Eyes", as_of=as_of):
        assert order["status"] in {"ordered", "in_transit", "delayed"}
        assert date.fromisoformat(order["expected_arrival"]) > as_of


def test_expected_incoming_is_discounted_below_scheduled(as_of):
    """Measured fill rate is under 100%, so expected arrivals must be below
    ordered quantity. Equal would mean the reliability discount is not
    being applied at all."""
    projection = project_supply("Blade Eyes", as_of=as_of)
    assert projection.incoming_scheduled_kg > 0
    assert projection.incoming_expected_kg <= projection.incoming_scheduled_kg
    assert projection.supply_reliability < 1.0


def test_incoming_by_day_sums_to_expected_total(as_of):
    projection = project_supply("Blade Eyes", as_of=as_of)
    per_day = incoming_by_day("Blade Eyes", as_of, DEFAULT_HORIZON_DAYS)
    assert sum(per_day.values()) == pytest.approx(
        projection.incoming_expected_kg, abs=0.05)


def test_no_open_orders_for_unknown_primal(as_of):
    assert open_orders("Not A Primal", as_of=as_of) == []


# ---------------------------------------------------------------------------
# Supplier reliability
# ---------------------------------------------------------------------------

def test_reliability_rates_are_plausible(as_of):
    reliability = supplier_reliability(as_of=as_of)
    assert reliability.orders > 0
    assert 0.80 < reliability.on_time_rate <= 1.0
    assert 0.90 < reliability.avg_fill_rate <= 1.0
    assert 0.0 <= reliability.short_shipment_rate < 0.20


def test_reliability_uses_fill_only_not_fill_times_on_time(as_of):
    """A late delivery still arrives. Folding lateness into the quantity
    discount would double-count it - the day-by-day ladder already models
    timing by placing each order on its expected date."""
    reliability = supplier_reliability(as_of=as_of)
    assert reliability.combined_reliability == reliability.avg_fill_rate


def test_reliability_of_unknown_scope_is_unknown_not_perfect(as_of):
    reliability = supplier_reliability("Not A Primal", as_of=as_of)
    assert reliability.orders == 0
    assert reliability.avg_fill_rate is None
    assert reliability.combined_reliability is None


# ---------------------------------------------------------------------------
# Ordering and status
# ---------------------------------------------------------------------------

def test_all_supply_sorted_by_urgency(as_of):
    rows = project_all_supply(as_of=as_of)
    keys = [(p.days_until_stockout if p.days_until_stockout is not None else 10_000)
            for p in rows]
    assert keys == sorted(keys)


def test_status_matches_stockout_flag(as_of):
    for projection in project_all_supply(as_of=as_of):
        if projection.stockout_expected:
            assert projection.status == "stockout_expected"
            assert projection.earliest_stockout_date is not None
            assert projection.days_until_stockout >= 1
        else:
            assert projection.earliest_stockout_date is None


def test_unknown_primal_raises(as_of):
    with pytest.raises(ValueError):
        project_supply("Not A Primal", as_of=as_of)


def test_to_dict_is_json_safe(as_of):
    import json
    projection = project_supply("Blade Eyes", as_of=as_of)
    json.dumps(projection.to_dict())
