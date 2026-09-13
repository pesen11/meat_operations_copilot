"""Tests for simulation/history_analysis.py."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from simulation import history_analysis as ha
from simulation import ops_data

LAST_DAY = date(2025, 12, 31)


@pytest.fixture
def tiny_history(monkeypatch):
    """
    Four days of chuck roast: two weekdays at 40kg, two weekend days at 60kg.
    2025-06-02 is a Monday, so 06-07/06-08 are Sat/Sun.
    """
    rows = [
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 2), "units_sold": 40.0, "revenue": 939.6},
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 3), "units_sold": 40.0, "revenue": 939.6},
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 7), "units_sold": 60.0, "revenue": 1409.4},
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 8), "units_sold": 60.0, "revenue": 1409.4},
    ]
    sales = pd.DataFrame(rows)
    monkeypatch.setattr(ops_data, "load_sales", lambda: sales.copy())
    return sales


def test_sales_summary_splits_weekday_and_weekend(tiny_history):
    s = ha.sales_summary("chuck-roast", date(2025, 6, 2), date(2025, 6, 8))
    assert s.open_days == 4
    assert s.total_kg == pytest.approx(200.0)
    assert s.avg_kg_per_open_day == pytest.approx(50.0)
    assert s.avg_kg_weekday == pytest.approx(40.0)
    assert s.avg_kg_weekend == pytest.approx(60.0)
    assert s.best_day_kg == pytest.approx(60.0)


def test_sales_summary_rejects_unknown_sku(tiny_history):
    with pytest.raises(ValueError):
        ha.sales_summary("no-such-sku")


def test_each_unit_products_are_normalised_to_kg(monkeypatch):
    """
    Tomahawk is sold EACH at ~1.75kg per tray. 10 trays must report as
    17.5kg, not 10 - otherwise it looks like the slowest mover in the shop
    when ranked against kg-denominated cuts.
    """
    sales = pd.DataFrame([
        {"sku": "tomahawk-steak", "sale_date": date(2025, 6, 2), "units_sold": 10.0, "revenue": 962.3},
    ])
    monkeypatch.setattr(ops_data, "load_sales", lambda: sales.copy())
    s = ha.sales_summary("tomahawk-steak", date(2025, 6, 2), date(2025, 6, 2))
    assert s.total_kg == pytest.approx(17.5)


def test_trend_flags_growth(monkeypatch):
    """14 days at 10kg followed by 14 days at 20kg is a +100% trend."""
    rows = []
    start = date(2025, 3, 1)
    for i in range(28):
        d = start + timedelta(days=i)
        kg = 10.0 if i < 14 else 20.0
        rows.append({"sku": "chuck-roast", "sale_date": d, "units_sold": kg, "revenue": kg * 23.49})
    sales = pd.DataFrame(rows)
    monkeypatch.setattr(ops_data, "load_sales", lambda: sales.copy())

    t = ha.sales_trend("chuck-roast", as_of=start + timedelta(days=27), window_days=14)
    assert t.recent_avg_kg_per_day == pytest.approx(20.0)
    assert t.prior_avg_kg_per_day == pytest.approx(10.0)
    assert t.change_pct == pytest.approx(1.0)
    assert t.direction == "up"
    # March has no stat holiday or summer window in either half, so the
    # calendar cannot be credited for this one.
    assert t.seasonality_explains_change is False


def test_trend_reports_flat_inside_the_noise_band(monkeypatch):
    rows = []
    start = date(2025, 3, 1)
    for i in range(28):
        d = start + timedelta(days=i)
        kg = 10.0 if i < 14 else 10.2  # +2%, inside the 5% band
        rows.append({"sku": "chuck-roast", "sale_date": d, "units_sold": kg, "revenue": kg * 23.49})
    monkeypatch.setattr(ops_data, "load_sales", lambda: pd.DataFrame(rows).copy())

    t = ha.sales_trend("chuck-roast", as_of=start + timedelta(days=27), window_days=14)
    assert t.direction == "flat"
    assert t.seasonality_explains_change is False


def test_trend_credits_seasonality_when_the_calendar_moved_too(monkeypatch):
    """
    Sales that rise exactly as much as the seasonal multiplier rises should
    be attributed to the calendar, not narrated as a real demand shift.
    Window pair straddles the Christmas run-up (1.9x, confirmed).
    """
    rows = []
    start = date(2025, 11, 28)
    for i in range(28):
        d = start + timedelta(days=i)
        kg = 10.0 * ha.week_multiplier(d)
        rows.append({"sku": "chuck-roast", "sale_date": d, "units_sold": kg, "revenue": kg * 23.49})
    monkeypatch.setattr(ops_data, "load_sales", lambda: pd.DataFrame(rows).copy())

    t = ha.sales_trend("chuck-roast", as_of=start + timedelta(days=27), window_days=14)
    assert t.direction == "up"
    assert t.recent_avg_seasonal_multiplier > t.prior_avg_seasonal_multiplier
    assert t.seasonality_explains_change is True


def test_top_movers_ranks_and_shares_sum_to_one(tiny_history):
    rows = ha.top_movers(date(2025, 6, 2), date(2025, 6, 8), limit=5)
    assert rows[0].sku == "chuck-roast"
    assert sum(r.share_of_revenue_pct for r in rows) == pytest.approx(1.0, abs=1e-3)


def test_top_movers_by_kg_and_by_revenue_can_differ():
    """On the real catalog, the cheapest high-volume cut should outrank a
    premium cut on kg but not necessarily on revenue."""
    by_kg = {r.sku for r in ha.top_movers(limit=3, by="kg", end=LAST_DAY)}
    by_rev = {r.sku for r in ha.top_movers(limit=3, by="revenue", end=LAST_DAY)}
    assert by_kg and by_rev
    assert by_kg != by_rev


def test_top_movers_rejects_bad_sort_key():
    with pytest.raises(ValueError):
        ha.top_movers(by="margin")


# ---------------------------------------------------------------------------
# Structural checks against the real generated history
# ---------------------------------------------------------------------------

def test_summary_on_real_history_is_positive():
    s = ha.sales_summary("chuck-roast", end=LAST_DAY)
    assert s.total_kg > 0
    assert s.total_revenue > 0
    assert s.open_days > 0
    assert s.avg_kg_weekend > s.avg_kg_weekday  # confirmed weekend-heavy pattern


def test_trailing_average_matches_summary_average():
    avg = ha.trailing_avg_kg_per_day("chuck-roast", as_of=LAST_DAY, days=28)
    s = ha.sales_summary("chuck-roast", LAST_DAY - timedelta(days=27), LAST_DAY)
    assert avg == pytest.approx(s.avg_kg_per_open_day)


def test_weekly_revenue_series_buckets_are_whole_weeks():
    series = ha.weekly_revenue_series(LAST_DAY - timedelta(days=69), LAST_DAY)
    assert len(series) == 10
    assert all(row["revenue"] > 0 for row in series)
    assert series[-1]["week_ending"] == LAST_DAY.isoformat()


# --- catalog_sales_summary --------------------------------------------------
# The whole-catalog total. Added because a question naming no product ("what
# were our sales yesterday") had no total to report: `sales_summary` requires
# a SKU, so the narration could only rank products against each other.

@pytest.fixture
def two_product_history(monkeypatch):
    """
    Two products over the same four days, so a catalog total is not simply one
    product's total. 2025-06-07/08 are Sat/Sun.
    """
    rows = [
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 2), "units_sold": 40.0, "revenue": 939.6},
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 3), "units_sold": 40.0, "revenue": 939.6},
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 7), "units_sold": 60.0, "revenue": 1409.4},
        {"sku": "chuck-roast", "sale_date": date(2025, 6, 8), "units_sold": 60.0, "revenue": 1409.4},
        {"sku": "ribeye-steak-bnls", "sale_date": date(2025, 6, 2), "units_sold": 10.0, "revenue": 500.0},
        {"sku": "ribeye-steak-bnls", "sale_date": date(2025, 6, 7), "units_sold": 20.0, "revenue": 1000.0},
    ]
    sales = pd.DataFrame(rows)
    monkeypatch.setattr(ops_data, "load_sales", lambda: sales.copy())
    return sales


def test_catalog_total_is_the_sum_of_its_products(two_product_history):
    """The catalog total must equal the per-product totals added up."""
    start, end = date(2025, 6, 2), date(2025, 6, 8)
    cat = ha.catalog_sales_summary(start, end)
    parts = [ha.sales_summary(sku, start, end) for sku in ("chuck-roast", "ribeye-steak-bnls")]

    assert cat.products_sold == 2
    assert cat.total_kg == pytest.approx(sum(p.total_kg for p in parts), abs=0.01)
    assert cat.total_revenue == pytest.approx(sum(p.total_revenue for p in parts), abs=0.01)


def test_catalog_open_days_counts_days_not_rows(two_product_history):
    """Two products selling on one day is one open day, not two."""
    cat = ha.catalog_sales_summary(date(2025, 6, 2), date(2025, 6, 8))
    assert cat.open_days == 4
    assert cat.avg_kg_per_open_day == pytest.approx(cat.total_kg / 4, abs=0.01)


def test_catalog_weekend_split_and_best_day(two_product_history):
    """Best day is the heaviest DAY across the catalog, not the heaviest row."""
    cat = ha.catalog_sales_summary(date(2025, 6, 2), date(2025, 6, 8))
    assert cat.best_day == date(2025, 6, 7)
    assert cat.avg_kg_weekend > cat.avg_kg_weekday
    assert cat.best_day_revenue == pytest.approx(1409.4 + 1000.0, abs=0.01)


def test_catalog_single_day_window(two_product_history):
    """A one-day window is a legitimate report, not an error."""
    cat = ha.catalog_sales_summary(date(2025, 6, 2), date(2025, 6, 2))
    assert cat.open_days == 1
    assert cat.products_sold == 2
    assert cat.total_revenue == pytest.approx(939.6 + 500.0, abs=0.01)


def test_catalog_empty_window_is_zero_not_a_crash(two_product_history):
    cat = ha.catalog_sales_summary(date(2025, 7, 1), date(2025, 7, 7))
    assert cat.open_days == 0
    assert cat.total_kg == 0.0
    assert cat.best_day is None


def test_catalog_matches_catalog_performance_total():
    """
    Two independent code paths over the REAL history must agree. This is the
    cross-check that self-consistency alone would miss.
    """
    end = ha.latest_sales_date()
    start = end - timedelta(days=27)
    cat = ha.catalog_sales_summary(start, end)
    perf = ha.catalog_performance(start, end)

    assert cat.total_kg == pytest.approx(sum(p.total_kg for p in perf), abs=0.5)
    assert cat.total_revenue == pytest.approx(sum(p.total_revenue for p in perf), abs=0.5)


def test_catalog_weekly_revenue_is_a_plausible_shop():
    """
    Magnitude, not just internal consistency - the lesson from the primal-cost
    bug, where 57 self-consistent tests all passed on a catalog that could not
    pay rent. Cut products are confirmed at ~$140k/week (35% of a $400k store),
    so a 7-day total belongs in a band around that, not off by an order of
    magnitude.
    """
    end = ha.latest_sales_date()
    cat = ha.catalog_sales_summary(end - timedelta(days=6), end)
    assert 80_000 < cat.total_revenue < 400_000, cat.total_revenue
    assert cat.open_days <= 7
    assert cat.products_sold > 1
