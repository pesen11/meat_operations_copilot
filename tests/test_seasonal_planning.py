"""Tests for simulation/seasonal_planning.py."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from data.catalog import yield_profile_by_sku
from simulation import ops_data
from simulation import seasonal_planning as sp

LAST_DAY = date(2025, 12, 31)


class TestPeriodResolution:
    @pytest.mark.parametrize("text,expected", [
        ("How much production increase should we do during christmas?", "christmas"),
        ("what should we plan for xmas", "christmas"),
        ("how much more for the long weekend", "long_weekend"),
        ("plan for the victoria day stat holiday", "long_weekend"),
        ("what about summer bbq season", "summer"),
        ("how much for grilling season", "summer"),
    ])
    def test_operator_phrasing_maps_to_a_period(self, text, expected):
        assert sp.resolve_period(text) == expected

    def test_unrelated_text_resolves_to_nothing(self):
        assert sp.resolve_period("how do I clean the grinder") is None
        assert sp.resolve_period("increase chuck roast by 15%") is None


class TestMultipliers:
    def test_multipliers_are_the_confirmed_values(self):
        """These came from the project owner. If this test fails, someone
        changed a confirmed business number."""
        assert sp.PERIODS["christmas"] == 1.9
        assert sp.PERIODS["long_weekend"] == 1.7
        assert sp.PERIODS["summer"] == 1.4
        assert sp.PERIODS["regular"] == 1.0

    def test_unknown_period_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown period"):
            sp.seasonal_plan("easter")


class TestPeriodWindow:
    def test_christmas_window_ends_on_the_25th(self):
        window = sp.period_window("christmas", date(2025, 11, 1))
        assert window is not None
        start, end = window
        assert end == date(2025, 12, 25)
        assert start == date(2025, 12, 19)

    def test_window_search_rolls_into_next_year(self):
        window = sp.period_window("christmas", date(2025, 12, 26))
        assert window is not None
        assert window[1] == date(2026, 12, 25)

    def test_regular_period_has_no_window(self):
        assert sp.period_window("regular", date(2025, 6, 1)) is None


@pytest.fixture
def tiny_history(monkeypatch):
    """One product, 4 weeks, 28 kg/week, so uplift math is checkable by hand."""
    rows = []
    start = date(2025, 11, 3)
    for i in range(28):
        rows.append({"sku": "chuck-roast", "sale_date": start + __import__("datetime").timedelta(days=i),
                     "units_sold": 4.0, "revenue": 93.96})
    sales = pd.DataFrame(rows)
    stock = pd.DataFrame([{"source_primal": "Blade Eyes", "as_of": date(2025, 11, 30),
                           "boxes_on_hand": 21.0, "velocity_tier": "best"}])
    labor = pd.DataFrame([{"shift_date": start + __import__("datetime").timedelta(days=i),
                           "role": "cutter", "available_minutes": 1020.0, "headcount": 2}
                          for i in range(28)])
    monkeypatch.setattr(ops_data, "load_sales", lambda: sales.copy())
    monkeypatch.setattr(ops_data, "load_primal_stock", lambda: stock.copy())
    monkeypatch.setattr(ops_data, "load_labor", lambda: labor.copy())
    return sales


class TestPlanArithmetic:
    def test_uplift_is_baseline_times_the_confirmed_multiplier(self, tiny_history):
        plan = sp.seasonal_plan("christmas", as_of=date(2025, 11, 30))
        row = plan.products[0]
        # 4 kg/day x 28 days / 4 weeks = 28 kg/week
        assert row.baseline_weekly_kg == pytest.approx(28.0)
        assert row.planned_weekly_kg == pytest.approx(28.0 * 1.9)
        assert row.extra_weekly_kg == pytest.approx(28.0 * 0.9)

    def test_extra_primal_is_extra_product_over_yield(self, tiny_history):
        plan = sp.seasonal_plan("christmas", as_of=date(2025, 11, 30))
        row = plan.products[0]
        yp = yield_profile_by_sku["chuck-roast"]
        assert row.extra_primal_kg == pytest.approx(row.extra_weekly_kg / yp.yield_pct, abs=0.01)

    def test_waste_scales_off_primal_not_product(self, tiny_history):
        plan = sp.seasonal_plan("christmas", as_of=date(2025, 11, 30))
        row = plan.products[0]
        yp = yield_profile_by_sku["chuck-roast"]
        assert row.extra_trim_waste_kg == pytest.approx(
            row.extra_primal_kg * yp.trim_waste_pct, abs=0.01)

    def test_boxes_use_the_real_delivery_box_weight(self, tiny_history):
        plan = sp.seasonal_plan("christmas", as_of=date(2025, 11, 30))
        primal = plan.primals[0]
        assert primal.source_primal == "Blade Eyes"
        assert primal.extra_boxes == pytest.approx(primal.extra_primal_kg / 30.0, abs=0.01)

    def test_regular_period_is_a_no_op(self, tiny_history):
        plan = sp.seasonal_plan("regular", as_of=date(2025, 11, 30))
        assert plan.total_extra_weekly_kg == pytest.approx(0.0)
        assert plan.total_extra_boxes == pytest.approx(0.0)
        assert plan.exceeds_cutter_capacity is False

    def test_bigger_multiplier_means_more_of_everything(self, tiny_history):
        summer = sp.seasonal_plan("summer", as_of=date(2025, 11, 30))
        christmas = sp.seasonal_plan("christmas", as_of=date(2025, 11, 30))
        assert christmas.total_extra_weekly_kg > summer.total_extra_weekly_kg
        assert christmas.total_extra_boxes > summer.total_extra_boxes
        assert christmas.total_extra_trim_waste_kg > summer.total_extra_trim_waste_kg

    def test_boxes_to_order_credits_existing_overstock(self, tiny_history):
        """A cooler already carrying above its normal top-of-range should not
        be told to order that surplus again."""
        plan = sp.seasonal_plan("christmas", as_of=date(2025, 11, 30))
        primal = plan.primals[0]
        # 21 boxes on hand vs a 20-25 target: no headroom above 25, so the
        # full extra requirement must still be ordered.
        assert primal.boxes_on_hand == 21.0
        assert primal.boxes_to_order == pytest.approx(primal.extra_boxes, abs=0.01)


class TestDeseasonalisedBaseline:
    """Regression: asking about Christmas FROM INSIDE the Christmas window
    used to apply the 1.9x uplift on top of a baseline already running at
    ~1.32x, double-counting the season and reporting a capacity overrun that
    did not exist (119% vs. the true ~91%)."""

    @pytest.mark.parametrize("as_of", [
        date(2025, 7, 15),    # quiet summer-ish window
        date(2025, 11, 30),   # normal window, no elevated days
        date(2025, 12, 31),   # window CONTAINS the Christmas peak
    ])
    def test_plan_is_stable_whenever_it_is_asked(self, as_of):
        plan = sp.seasonal_plan("christmas", as_of=as_of)
        # Same shop, same confirmed uplift: the answer must not swing on the
        # accident of which season the lookback window happened to span.
        assert 0.85 <= plan.planned_cutter_utilization_pct <= 0.97, (
            as_of, plan.planned_cutter_utilization_pct)
        assert plan.exceeds_cutter_capacity is False

    def test_baseline_is_lower_than_raw_sales_in_an_elevated_window(self):
        """Dec 4-31 contains the Christmas peak, so the de-seasonalised
        baseline must come out BELOW the raw kg actually sold."""
        from simulation import ops_data
        from simulation.units import units_to_kg

        as_of = date(2025, 12, 31)
        start = as_of - timedelta(days=27)
        sales = ops_data.load_sales()
        window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= as_of)]
        raw_weekly = sum(units_to_kg(r.sku, float(r.units_sold))
                         for r in window.itertuples(index=False)) / 4

        plan = sp.seasonal_plan("christmas", as_of=as_of)
        baseline_weekly = sum(p.baseline_weekly_kg for p in plan.products)
        assert baseline_weekly < raw_weekly

    def test_plan_states_how_the_baseline_was_derived(self):
        plan = sp.seasonal_plan("christmas", as_of=date(2025, 12, 31))
        assert "de-seasonalised" in plan.baseline_basis


class TestPlanOnRealHistory:
    def test_christmas_plan_is_directionally_sane(self):
        plan = sp.seasonal_plan("christmas", as_of=date(2025, 11, 30))
        assert plan.multiplier == 1.9
        assert plan.total_extra_weekly_kg > 0
        assert plan.total_extra_boxes > 0
        assert plan.planned_cutter_utilization_pct > plan.baseline_cutter_utilization_pct
        assert plan.days_until is not None and plan.days_until > 0

    def test_plan_covers_every_selling_product(self):
        plan = sp.seasonal_plan("christmas", as_of=LAST_DAY)
        assert len(plan.products) == 19

    def test_scoping_to_one_product_narrows_the_product_list(self):
        plan = sp.seasonal_plan("christmas", as_of=LAST_DAY, product_sku="chuck-roast")
        assert [p.product_sku for p in plan.products] == ["chuck-roast"]

    def test_scoped_plan_still_reports_shop_wide_capacity(self):
        """The extra work lands on the same cutters regardless of which
        product the plan is about - reporting only that product's minutes
        would make capacity look far safer than it is."""
        scoped = sp.seasonal_plan("christmas", as_of=LAST_DAY, product_sku="chuck-roast")
        whole = sp.seasonal_plan("christmas", as_of=LAST_DAY)
        assert scoped.baseline_weekly_cutting_minutes == pytest.approx(
            whole.baseline_weekly_cutting_minutes, abs=0.5)

    def test_unknown_sku_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown product_sku"):
            sp.seasonal_plan("christmas", product_sku="not-a-sku")

    def test_calendar_lists_upcoming_periods_in_order(self):
        rows = sp.seasonal_calendar(as_of=date(2025, 11, 1), horizon_days=120)
        assert rows
        assert [r["days_until"] for r in rows] == sorted(r["days_until"] for r in rows)
        assert all(r["multiplier"] >= 1.0 for r in rows)

    def test_multiplier_for_a_christmas_day_is_the_christmas_value(self):
        assert sp.multiplier_for_day(date(2025, 12, 22))["multiplier"] == 1.9
        assert sp.multiplier_for_day(date(2025, 12, 22))["is_christmas_period"] is True
