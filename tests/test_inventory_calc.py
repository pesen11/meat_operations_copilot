"""
Tests for simulation/inventory_calc.py.

Two kinds of test here, deliberately:
- Structural tests against the real generated history (statuses are valid,
  cover is non-negative, scenario arithmetic is internally consistent).
- Closed-form tests on a tiny synthetic frame, where the expected number is
  computed by hand in the test so a regression in the module can't quietly
  agree with itself.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from data.catalog import products_by_sku, yield_profile_by_sku, primal_reference_box_weight_kg
from simulation import inventory_calc as ic
from simulation import ops_data
from simulation.yield_calc import labor_minutes_per_pack


LAST_DAY = date(2025, 12, 31)


# ---------------------------------------------------------------------------
# Closed-form tests on a hand-built frame
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_history(monkeypatch):
    """
    Two open days of chuck-roast-only sales, so every derived number can be
    checked by hand. Chuck Roast: yield 0.88, pack 2.2kg, 15 labor min per
    30kg box of Blade Eyes.
    """
    sales = pd.DataFrame([
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 2), "units_sold": 44.0, "revenue": 1033.56},
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 3), "units_sold": 88.0, "revenue": 2067.12},
    ])
    stock = pd.DataFrame([
        {"source_primal": "Blade Eyes", "as_of": date(2025, 6, 3),
         "boxes_on_hand": 10.0, "velocity_tier": "best"},
    ])
    labor = pd.DataFrame([
        {"shift_date": date(2025, 6, 2), "role": "cutter", "available_minutes": 1020.0, "headcount": 2},
        {"shift_date": date(2025, 6, 3), "role": "cutter", "available_minutes": 1020.0, "headcount": 2},
    ])
    monkeypatch.setattr(ops_data, "load_sales", lambda: sales.copy())
    monkeypatch.setattr(ops_data, "load_primal_stock", lambda: stock.copy())
    monkeypatch.setattr(ops_data, "load_labor", lambda: labor.copy())
    return sales


def test_primal_draw_is_finished_kg_divided_by_yield(tiny_history):
    consumed = ic.primal_kg_consumed_by_day(date(2025, 6, 2), date(2025, 6, 3))
    # 44 kg of finished chuck roast at 88% yield needs 50 kg of Blade Eyes.
    assert consumed[("Blade Eyes", date(2025, 6, 2))] == pytest.approx(44 / 0.88)
    assert consumed[("Blade Eyes", date(2025, 6, 3))] == pytest.approx(88 / 0.88)


def test_avg_daily_primal_kg_divides_by_open_days_not_calendar_days(tiny_history):
    # Window is 28 days but only 2 of them have sales, so the average must be
    # (50 + 100) / 2 = 75, not (50 + 100) / 28.
    avg = ic.avg_daily_primal_kg(as_of=date(2025, 6, 3), lookback_days=28)
    assert avg["Blade Eyes"] == pytest.approx(75.0)


def test_days_of_cover_is_kg_on_hand_over_daily_draw(tiny_history):
    pos = ic.stock_position("Blade Eyes", as_of=date(2025, 6, 3), lookback_days=28)
    # 10 boxes x 30 kg = 300 kg on hand, 75 kg/day draw -> 4.0 days.
    assert pos.kg_on_hand == pytest.approx(300.0)
    assert pos.days_of_cover == pytest.approx(4.0)
    assert pos.status == "below_target"  # best tier wants 20-25 boxes


def test_required_cutting_minutes_matches_packs_times_per_pack_labor(tiny_history):
    product = products_by_sku["chuck-roast"]
    yp = yield_profile_by_sku["chuck-roast"]
    expected = (44.0 / product.avg_pack_weight_kg) * labor_minutes_per_pack(yp, product)
    assert ic.required_cutting_minutes_for_day(date(2025, 6, 2)) == pytest.approx(expected)


def test_cutter_capacity_slack_and_utilization(tiny_history):
    cap = ic.cutter_capacity(date(2025, 6, 2))
    assert cap.headcount == 2
    assert cap.available_minutes == 1020.0
    assert cap.slack_minutes == pytest.approx(1020.0 - cap.required_cutting_minutes, abs=0.05)
    assert cap.utilization_pct == pytest.approx(cap.required_cutting_minutes / 1020.0, abs=1e-4)
    assert cap.is_over_capacity is False


def test_scenario_impact_scales_linearly_and_conserves_identities(tiny_history):
    si = ic.scenario_impact("chuck-roast", 1.15, as_of=date(2025, 6, 3), lookback_days=28)

    # baseline weekly kg = 132 kg over a 4-week window = 33 kg/week
    assert si.baseline_weekly_product_kg == pytest.approx(33.0)
    assert si.extra_weekly_product_kg == pytest.approx(33.0 * 0.15)
    assert si.scenario_weekly_product_kg == pytest.approx(
        si.baseline_weekly_product_kg + si.extra_weekly_product_kg, abs=0.01)

    # primal kg = product kg / yield, extra boxes = extra primal kg / 30
    assert si.baseline_weekly_primal_kg == pytest.approx(33.0 / 0.88, abs=0.01)
    assert si.extra_weekly_boxes == pytest.approx(si.extra_weekly_primal_kg / 30.0, abs=0.01)

    # waste scales off PRIMAL throughput, not finished product
    yp = yield_profile_by_sku["chuck-roast"]
    assert si.extra_weekly_trim_waste_kg == pytest.approx(
        si.extra_weekly_primal_kg * yp.trim_waste_pct, abs=0.01)


def test_scenario_impact_cover_falls_when_demand_rises(tiny_history):
    si = ic.scenario_impact("chuck-roast", 1.5, as_of=date(2025, 6, 3), lookback_days=28)
    assert si.scenario_days_of_cover < si.baseline_days_of_cover


def test_scenario_impact_multiplier_of_one_is_a_no_op(tiny_history):
    si = ic.scenario_impact("chuck-roast", 1.0, as_of=date(2025, 6, 3), lookback_days=28)
    assert si.extra_weekly_product_kg == pytest.approx(0.0)
    assert si.extra_weekly_cutting_minutes == pytest.approx(0.0)
    assert si.baseline_days_of_cover == si.scenario_days_of_cover


def test_scenario_impact_rejects_bad_input(tiny_history):
    with pytest.raises(ValueError):
        ic.scenario_impact("chuck-roast", 0.0)
    with pytest.raises(ValueError):
        ic.scenario_impact("not-a-real-sku", 1.1)


def test_scenario_utilization_uses_shop_wide_minutes(monkeypatch):
    """
    A second product's cutting minutes must show up in the baseline
    utilization even when the scenario is about chuck roast. This is the
    guard against measuring one product's lift against only its own minutes,
    which would make capacity look far safer than it is.
    """
    sales = pd.DataFrame([
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 2), "units_sold": 44.0, "revenue": 1033.56},
        {"sku": "ground-beef-medium", "sale_date": date(2025, 6, 2), "units_sold": 200.0, "revenue": 2798.0},
    ])
    stock = pd.DataFrame([{"source_primal": "Blade Eyes", "as_of": date(2025, 6, 2),
                           "boxes_on_hand": 10.0, "velocity_tier": "best"}])
    labor = pd.DataFrame([{"shift_date": date(2025, 6, 2), "role": "cutter",
                           "available_minutes": 1020.0, "headcount": 2}])
    monkeypatch.setattr(ops_data, "load_sales", lambda: sales.copy())
    monkeypatch.setattr(ops_data, "load_primal_stock", lambda: stock.copy())
    monkeypatch.setattr(ops_data, "load_labor", lambda: labor.copy())

    si = ic.scenario_impact("chuck-roast", 1.15, as_of=date(2025, 6, 2), lookback_days=28)

    # Chuck roast's own contribution, computed independently here.
    chuck = products_by_sku["chuck-roast"]
    chuck_minutes = ((44.0 / chuck.avg_pack_weight_kg)
                     * labor_minutes_per_pack(yield_profile_by_sku["chuck-roast"], chuck))
    # Ground beef is cut on the same cutters, so shop-wide baseline minutes
    # must be strictly larger than chuck roast's share.
    assert si.baseline_weekly_cutting_minutes > chuck_minutes / 4
    assert si.baseline_weekly_cutting_minutes == pytest.approx(
        ic.required_cutting_minutes_for_day(date(2025, 6, 2)) / 4, abs=0.1)


# ---------------------------------------------------------------------------
# Structural tests against the real generated history
# ---------------------------------------------------------------------------

VALID_STATUSES = {"stockout", "critical", "below_target", "ok", "over_target"}


def test_all_stock_positions_cover_every_primal():
    positions = ic.all_stock_positions(as_of=LAST_DAY)
    primals = set(primal_reference_box_weight_kg())
    assert {p.source_primal for p in positions} == primals


def test_stock_positions_are_internally_consistent():
    for pos in ic.all_stock_positions(as_of=LAST_DAY):
        assert pos.status in VALID_STATUSES
        assert pos.boxes_on_hand >= 0
        assert pos.kg_on_hand == pytest.approx(
            pos.boxes_on_hand * pos.reference_box_weight_kg, abs=0.05)
        if pos.days_of_cover is not None:
            assert pos.days_of_cover >= 0


def test_tenderloin_uses_real_box_weight_not_piece_weight():
    """Regression guard on the bug CLAUDE.md documents: tenderloin stock is
    tracked in ~40kg delivery boxes, not ~4.5kg cutting pieces."""
    pos = ic.stock_position("Tenderloin", as_of=LAST_DAY)
    assert pos.reference_box_weight_kg == pytest.approx(40.0)
    assert pos.target_boxes_low == 1.0 and pos.target_boxes_high == 2.0


def test_cutter_capacity_on_a_real_day():
    cap = ic.cutter_capacity(LAST_DAY)
    assert cap.available_minutes > 0
    assert cap.required_cutting_minutes > 0
    assert cap.utilization_pct == pytest.approx(
        cap.required_cutting_minutes / cap.available_minutes, abs=1e-4)


def test_cutter_capacity_raises_on_a_closed_holiday():
    with pytest.raises(ValueError):
        ic.cutter_capacity(date(2025, 12, 25))  # Christmas: shop closed


def test_scenario_impact_on_real_history_is_directionally_sane():
    si = ic.scenario_impact("chuck-roast", 1.15, as_of=LAST_DAY)
    assert si.extra_weekly_primal_kg > 0
    assert si.extra_weekly_boxes > 0
    assert si.extra_weekly_trim_waste_kg > 0
    assert si.scenario_cutter_utilization_pct > si.baseline_cutter_utilization_pct
    assert si.scenario_days_of_cover < si.baseline_days_of_cover


def test_stockout_day_count_is_bounded_by_window_length():
    start, end = date(2025, 1, 1), date(2025, 3, 31)
    count = ic.stockout_day_count("Blade Eyes", start, end)
    assert 0 <= count <= (end - start).days + 1
