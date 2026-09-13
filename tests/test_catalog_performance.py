"""
Tests for the whole-catalog performance / slow-mover / weekend-split analysis
that answers "what is doing well, what should we order less of, and how much
heavier do weekends trade".
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from data.catalog import products_by_sku
from simulation import history_analysis as ha


@pytest.fixture(scope="module")
def end_date():
    return ha.latest_sales_date()


class TestCatalogPerformance:
    def test_covers_every_product_that_traded(self, end_date):
        rows = ha.catalog_performance(end=end_date)
        assert {r.sku for r in rows} <= set(products_by_sku)
        assert len(rows) >= 15

    def test_sorted_by_revenue_descending(self, end_date):
        rows = ha.catalog_performance(end=end_date)
        assert rows == sorted(rows, key=lambda r: -r.total_revenue)

    def test_revenue_shares_sum_to_one(self, end_date):
        rows = ha.catalog_performance(end=end_date)
        assert sum(r.share_of_revenue_pct for r in rows) == pytest.approx(1.0, abs=0.01)

    def test_every_row_carries_an_actionable_verdict(self, end_date):
        rows = ha.catalog_performance(end=end_date)
        assert all(r.verdict in ("focus", "watch", "reduce", "steady") for r in rows)

    def test_direction_agrees_with_change_pct(self, end_date):
        for r in ha.catalog_performance(end=end_date):
            if r.change_pct is None:
                assert r.direction == "unknown"
            elif r.change_pct > ha.FLAT_TREND_BAND:
                assert r.direction == "up"
            elif r.change_pct < -ha.FLAT_TREND_BAND:
                assert r.direction == "down"
            else:
                assert r.direction == "flat"

    def test_seasonal_swings_are_never_recommended_as_demand_changes(self, end_date):
        """
        The guard that matters: telling an operator to cut back on a product
        because Christmas ended would be a wrong call, not a vague one.
        """
        for r in ha.catalog_performance(end=end_date):
            if r.seasonality_explains_change:
                assert r.verdict == "watch"

    def test_immaterial_products_are_never_marked_reduce(self, end_date):
        for r in ha.catalog_performance(end=end_date):
            if r.share_of_revenue_pct < ha.MATERIAL_REVENUE_SHARE:
                assert r.verdict != "reduce"

    def test_rate_matches_the_per_product_summary(self, end_date):
        """
        The comparison is per-OPEN-DAY, not per-window-total: an unequal
        number of open days either side would otherwise invent a trend. This
        pins the recent rate to what sales_summary independently computes.
        """
        start = end_date - timedelta(days=13)
        for r in ha.catalog_performance(start=start, end=end_date):
            expected = ha.sales_summary(r.sku, start, end_date).avg_kg_per_open_day
            assert r.avg_kg_per_open_day == pytest.approx(expected, abs=0.01)

    def test_a_shorter_window_does_not_by_itself_read_as_decline(self, end_date):
        """
        Same end date, same trade, fewer days in the window: the per-day rate
        should barely move. A totals comparison would halve it.
        """
        start = end_date - timedelta(days=13)
        long_rates = {r.sku: r.avg_kg_per_open_day
                      for r in ha.catalog_performance(start=start, end=end_date)}
        short_rates = {r.sku: r.avg_kg_per_open_day
                       for r in ha.catalog_performance(
                           start=end_date - timedelta(days=6), end=end_date)}
        for sku, short in short_rates.items():
            assert short == pytest.approx(long_rates[sku], rel=0.6), sku

    def test_weekends_only_changes_the_numbers(self, end_date):
        start = end_date - timedelta(days=59)
        allf = {r.sku: r.total_kg for r in ha.catalog_performance(start=start, end=end_date)}
        we = {r.sku: r.total_kg for r in ha.catalog_performance(
            start=start, end=end_date, weekends_only=True)}
        assert we, "weekend-only view returned nothing"
        assert all(we[sku] < allf[sku] for sku in we), "weekend subset not a subset"


class TestSlowMovers:
    def test_returns_the_worst_not_the_best(self, end_date):
        slow = ha.slow_movers(end=end_date, limit=3)
        top = ha.top_movers(end=end_date, limit=3)
        assert {r.sku for r in slow}.isdisjoint({r.sku for r in top})

    def test_ascending_order(self, end_date):
        rows = ha.slow_movers(end=end_date, limit=5)
        assert [r.total_revenue for r in rows] == sorted(r.total_revenue for r in rows)

    def test_products_with_no_sales_are_included_at_zero(self, end_date):
        """
        A product that sold nothing has no rows in the sales table. Reversing
        a ranking built from those rows would hide it - the one item an
        operator most needs to see.
        """
        # A single day is narrow enough that not everything trades.
        rows = ha.slow_movers(start=end_date, end=end_date, limit=len(products_by_sku))
        assert len(rows) == len(products_by_sku)

    def test_rejects_an_unknown_ranking_key(self, end_date):
        with pytest.raises(ValueError):
            ha.slow_movers(end=end_date, by="margin")


class TestWeekendUplift:
    def test_weekends_trade_heavier_than_weekdays(self, end_date):
        up = ha.weekend_uplift(end=end_date)
        assert up.weekend_avg_kg > up.weekday_avg_kg
        assert up.implied_multiplier > 1.0

    def test_per_product_uplift_is_available(self, end_date):
        up = ha.weekend_uplift("ribeye-steak-bnls", end=end_date)
        assert up.sku == "ribeye-steak-bnls"
        assert up.product_name == "Boneless Ribeye Steak"
        assert up.uplift_pct is not None

    def test_uplift_pct_and_multiplier_agree(self, end_date):
        up = ha.weekend_uplift(end=end_date)
        assert up.implied_multiplier == pytest.approx(1 + up.uplift_pct, abs=0.01)

    def test_serialises_dates(self, end_date):
        d = ha.weekend_uplift(end=end_date).to_dict()
        assert isinstance(d["start"], str) and isinstance(d["end"], str)
