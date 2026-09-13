"""
Tests for simulation/uncertainty.py.

The headline test here is `test_measured_volatility_recovers_injected_noise`.
history_generator.py injects day-to-day noise at SALES_NOISE_CV = 0.12, and
the estimator measures volatility off residuals without ever importing that
constant. Recovering ~0.12 is therefore a genuine check that the estimator
works - the kind of test that would still mean something against real sales
data, rather than a constant compared against itself.

`test_risk_responds_to_stress` exists because of a real bug: an earlier
version compared 14-day demand totals against 14-day supply totals and
returned 0.000 for every primal in the catalog, including ones whose daily
ledger ran dry repeatedly. A probability that is always zero passes every
arithmetic test ever written for it.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from simulation import ops_data
from simulation.history_generator import SALES_NOISE_CV
from simulation.uncertainty import (
    DEMAND_AUTOCORRELATION, FALLBACK_CV, RISK_BANDS, _aggregate_cv,
    demand_interval, demand_volatility, primal_demand_interval,
    stockout_probability, supply_needed_for_service_level,
)


@pytest.fixture(scope="module")
def as_of() -> date:
    return max(ops_data.load_sales()["sale_date"])


# ---------------------------------------------------------------------------
# The estimator actually estimates
# ---------------------------------------------------------------------------

def test_measured_volatility_recovers_injected_noise(as_of):
    """The generator injects 12% per-product noise. Measuring it off
    residuals must recover roughly that, without importing the constant into
    the estimator. This is the test that says the estimator works."""
    measured = demand_volatility("chuck-roast", end=as_of)
    assert measured.measured
    assert measured.coefficient_of_variation == pytest.approx(SALES_NOISE_CV, abs=0.03), (
        f"estimator returned {measured.coefficient_of_variation}, "
        f"generator injected {SALES_NOISE_CV}"
    )


def test_every_product_volatility_is_in_a_plausible_band(as_of):
    from data.catalog import products_by_sku
    for sku in products_by_sku:
        volatility = demand_volatility(sku, end=as_of)
        assert 0.02 < volatility.coefficient_of_variation < 0.60, (
            f"{sku} CV {volatility.coefficient_of_variation} is implausible"
        )


def test_catalog_volatility_is_lower_than_single_product(as_of):
    """Independent per-product noise partially cancels when aggregated, so
    the shop total must be steadier than any one product in it. If this
    inverts, the catalog measurement is averaging per-row ratios instead of
    aggregating per day."""
    catalog = demand_volatility(None, end=as_of)
    single = demand_volatility("chuck-roast", end=as_of)
    assert catalog.coefficient_of_variation < single.coefficient_of_variation


def test_unknown_product_falls_back_and_says_so(as_of):
    volatility = demand_volatility("not-a-sku", end=as_of)
    assert not volatility.measured
    assert "fallback" in volatility.basis or "default" in volatility.basis


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_cv_shrinks_with_more_days():
    assert _aggregate_cv(0.12, 9) < _aggregate_cv(0.12, 4) < _aggregate_cv(0.12, 1)


def test_aggregate_cv_is_widened_by_autocorrelation():
    """Independence would give exactly cv/sqrt(n). The correction must make
    the interval WIDER, never narrower - a narrowing correction would make
    the system more confident for a reason nobody can defend."""
    independent = 0.12 / math.sqrt(9)
    assert _aggregate_cv(0.12, 9) > independent
    assert _aggregate_cv(0.12, 9, autocorrelation=0.0) == pytest.approx(independent)


def test_autocorrelation_constant_is_a_moderate_positive():
    assert 0.0 < DEMAND_AUTOCORRELATION < 0.6


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------

def test_interval_brackets_its_point_estimate():
    interval = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 11))
    assert interval.lower_kg < interval.point_estimate_kg < interval.upper_kg


def test_interval_lower_bound_is_never_negative():
    """A symmetric normal interval on a volatile low-volume product puts its
    floor below zero, which is not a quantity of meat. Lognormal cannot."""
    from data.catalog import products_by_sku
    for sku in products_by_sku:
        interval = demand_interval(sku, date(2026, 1, 5), date(2026, 1, 6))
        assert interval.lower_kg >= 0


def test_wider_confidence_gives_a_wider_interval():
    narrow = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 11),
                             confidence=0.80)
    wide = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 11),
                           confidence=0.99)
    assert wide.upper_kg > narrow.upper_kg
    assert wide.lower_kg < narrow.lower_kg


def test_longer_window_has_a_proportionally_tighter_interval():
    """Absolute width grows with the window; RELATIVE width must shrink."""
    short = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 7))
    long = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 25))
    assert long.point_estimate_kg > short.point_estimate_kg
    short_rel = (short.upper_kg - short.lower_kg) / short.point_estimate_kg
    long_rel = (long.upper_kg - long.lower_kg) / long.point_estimate_kg
    assert long_rel < short_rel


def test_demand_multiplier_scales_the_whole_interval():
    # Tolerance is 0.02 kg, not a relative epsilon: every field is rounded to
    # two decimals on the way out, so doubling a rounded value and rounding a
    # doubled value can differ in the last cent. That is storage precision,
    # not a scaling error.
    base = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 11))
    up = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 11),
                         demand_multiplier=2.0)
    assert up.point_estimate_kg == pytest.approx(base.point_estimate_kg * 2, abs=0.02)
    assert up.upper_kg == pytest.approx(base.upper_kg * 2, abs=0.02)
    assert up.lower_kg == pytest.approx(base.lower_kg * 2, abs=0.02)


def test_unknown_sku_raises():
    with pytest.raises(ValueError):
        demand_interval("not-a-sku", date(2026, 1, 5), date(2026, 1, 11))


def test_primal_interval_exceeds_its_finished_product_demand():
    """Primal draw is finished demand divided by yield, so it must be larger."""
    primal = primal_demand_interval("Blade Eyes", date(2026, 1, 5), date(2026, 1, 11))
    product = demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 11))
    assert primal.point_estimate_kg > product.point_estimate_kg
    assert primal.lower_kg < primal.point_estimate_kg < primal.upper_kg


def test_unknown_primal_raises():
    with pytest.raises(ValueError):
        primal_demand_interval("Not A Primal", date(2026, 1, 5), date(2026, 1, 11))


# ---------------------------------------------------------------------------
# Stockout risk
# ---------------------------------------------------------------------------

def test_normal_demand_is_low_risk(as_of):
    """A shop in steady state should not be on the brink. If normal trading
    is high risk, the forward order book is broken."""
    risk = stockout_probability("Blade Eyes", as_of=as_of)
    assert risk.stockout_probability < 0.10
    assert risk.headroom_kg > 0


def test_risk_responds_to_stress(as_of):
    """The bug this pins: a risk model that reports 0.000 under Christmas
    demand, when the day ledger shows the cooler running dry on day five."""
    calm = stockout_probability("Blade Eyes", as_of=as_of, demand_multiplier=1.0)
    stressed = stockout_probability("Blade Eyes", as_of=as_of, demand_multiplier=1.9)
    assert stressed.stockout_probability > 0.9
    assert stressed.stockout_probability > calm.stockout_probability
    assert stressed.peak_risk_date is not None


def test_risk_is_monotone_in_demand(as_of):
    probabilities = [
        stockout_probability("Blade Eyes", as_of=as_of, demand_multiplier=m).stockout_probability
        for m in (1.0, 1.2, 1.3, 1.5, 1.9)
    ]
    assert probabilities == sorted(probabilities)


def test_risk_curve_has_a_usable_middle(as_of):
    """If risk only ever reads 0.0 or 1.0 it cannot rank scenario options.
    There has to be a multiplier where the answer is genuinely uncertain."""
    middling = [
        stockout_probability("Blade Eyes", as_of=as_of, demand_multiplier=m).stockout_probability
        for m in (1.2, 1.25, 1.3, 1.35)
    ]
    assert any(0.02 < p < 0.98 for p in middling), (
        "stockout probability is effectively binary across the whole range"
    )


def test_extra_supply_reduces_risk(as_of):
    without = stockout_probability("Blade Eyes", as_of=as_of, demand_multiplier=1.9)
    with_extra = stockout_probability("Blade Eyes", as_of=as_of,
                                      demand_multiplier=1.9, extra_supply_kg=3000)
    assert with_extra.stockout_probability < without.stockout_probability


def test_peak_risk_is_the_worst_day_not_the_last(as_of):
    risk = stockout_probability("Blade Eyes", as_of=as_of, demand_multiplier=1.4)
    assert risk.stockout_probability == max(d.probability for d in risk.daily)
    assert risk.stockout_probability >= risk.horizon_end_probability


def test_daily_ladder_covers_the_horizon(as_of):
    risk = stockout_probability("Blade Eyes", as_of=as_of, horizon_days=10)
    assert len(risk.daily) == 10
    assert risk.daily[0].day == as_of + timedelta(days=1)


def test_cumulative_demand_is_non_decreasing(as_of):
    risk = stockout_probability("Blade Eyes", as_of=as_of)
    values = [d.cumulative_demand_kg for d in risk.daily]
    assert values == sorted(values)


def test_service_level_complements_probability(as_of):
    risk = stockout_probability("Blade Eyes", as_of=as_of, demand_multiplier=1.3)
    assert risk.service_level == pytest.approx(1 - risk.stockout_probability, abs=1e-4)


def test_risk_band_matches_thresholds(as_of):
    for multiplier in (1.0, 1.2, 1.3, 1.5, 1.9):
        risk = stockout_probability("Blade Eyes", as_of=as_of,
                                    demand_multiplier=multiplier)
        expected = "high"
        for threshold, label in RISK_BANDS:
            if risk.stockout_probability < threshold:
                expected = label
                break
        assert risk.risk_band == expected


# ---------------------------------------------------------------------------
# Solving for a service level
# ---------------------------------------------------------------------------

def test_solver_finds_the_quantity_that_hits_the_target(as_of):
    out = supply_needed_for_service_level("Blade Eyes", 0.95, as_of=as_of,
                                          demand_multiplier=1.9)
    assert not out["already_met"]
    assert out["extra_needed_kg"] > 0
    assert out["resulting_stockout_probability"] <= 0.0501, (
        "solver returned a quantity that does not actually hit the target"
    )


def test_solver_short_circuits_when_target_already_met(as_of):
    out = supply_needed_for_service_level("Blade Eyes", 0.95, as_of=as_of,
                                          demand_multiplier=1.0)
    assert out["already_met"]
    assert out["extra_needed_kg"] == 0.0


def test_higher_service_level_costs_more_stock(as_of):
    modest = supply_needed_for_service_level("Blade Eyes", 0.90, as_of=as_of,
                                             demand_multiplier=1.9)
    strict = supply_needed_for_service_level("Blade Eyes", 0.99, as_of=as_of,
                                             demand_multiplier=1.9)
    assert strict["extra_needed_kg"] > modest["extra_needed_kg"]


def test_to_dict_is_json_safe(as_of):
    import json
    json.dumps(stockout_probability("Blade Eyes", as_of=as_of).to_dict())
    json.dumps(demand_interval("chuck-roast", date(2026, 1, 5), date(2026, 1, 11)).to_dict())
