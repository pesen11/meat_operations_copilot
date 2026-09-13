"""
Tests for simulation/execution_rates.py.

Per the project's working convention these assert on the MAGNITUDE of the
outcome, not only on internal consistency: the primal-cost bug got past 57
self-consistent tests because none of them asserted the shop could pay rent.
So "the rate is between 0 and 1" is not enough here - the rate has to land
in a band that describes a shop that actually functions.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from simulation import ops_data
from simulation.execution_rates import (
    AT_RISK_THRESHOLD, MIN_RUNS_FOR_RELIABLE_RATE, RELIABLE_THRESHOLD,
    all_execution_rates, effective_rate_for, execution_rate, expected_actual_kg,
)


@pytest.fixture(scope="module")
def latest() -> date:
    return max(ops_data.load_actual_production()["production_date"])


# ---------------------------------------------------------------------------
# Magnitude: does this describe a shop that works?
# ---------------------------------------------------------------------------

def test_shop_execution_rate_is_operationally_plausible(latest):
    """A shop completing under 70% of plan would be in crisis; one at 100%
    has no disruptions modelled at all and the whole module is pointless.
    Both ends are real failure modes this pins."""
    rate = execution_rate(None, end=latest, window_days=365)
    assert rate.rate is not None
    assert 0.80 < rate.rate < 0.99, (
        f"shop-wide execution rate {rate.rate} is outside the band that "
        f"describes a functioning shop with real disruptions"
    )


def test_every_primal_rate_is_plausible(latest):
    for row in all_execution_rates(end=latest, window_days=365):
        if row.rate is None:
            continue
        assert 0.5 < row.rate <= 1.05, f"{row.scope} execution rate {row.rate}"


def test_worst_primal_is_meaningfully_worse_than_best(latest):
    """If every primal executed identically the rate would carry no signal
    and keying it by primal would be pointless ceremony."""
    rates = [r.rate for r in all_execution_rates(end=latest, window_days=365)
             if r.rate is not None]
    assert max(rates) - min(rates) > 0.05, (
        "per-primal execution rates are too tightly clustered to justify "
        "keying the rate by primal"
    )


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_rate_equals_actual_over_planned(latest):
    rate = execution_rate("Blade Eyes", end=latest, window_days=365)
    assert rate.planned_kg > 0
    assert rate.rate == pytest.approx(rate.actual_kg / rate.planned_kg, abs=1e-4)


def test_shortfall_is_planned_minus_actual(latest):
    rate = execution_rate("Blade Eyes", end=latest, window_days=365)
    assert rate.shortfall_kg == pytest.approx(
        rate.planned_kg - rate.actual_kg, abs=0.02)


def test_run_counts_partition_the_window(latest):
    rate = execution_rate("Blade Eyes", end=latest, window_days=365)
    assert (rate.completed_runs + rate.partial_runs
            + rate.cancelled_runs) == rate.runs


def test_shortfall_by_reason_is_sorted_descending(latest):
    rate = execution_rate(None, end=latest, window_days=365)
    values = list(rate.shortfall_by_reason.values())
    assert values == sorted(values, reverse=True)
    assert rate.top_shortfall_reason == next(iter(rate.shortfall_by_reason))


def test_stock_short_is_a_real_recorded_cause(latest):
    """The generator links a cancelled run to an empty cooler on the same
    day. If that link ever breaks, the causal chain between primal_stock.csv
    and actual_production.csv has been severed and the two files are telling
    different stories about the same day."""
    rate = execution_rate(None, end=latest, window_days=365)
    assert "stock_short" in rate.shortfall_by_reason
    assert rate.shortfall_by_reason["stock_short"] > 0


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def test_window_is_respected(latest):
    narrow = execution_rate(None, end=latest, window_days=7)
    wide = execution_rate(None, end=latest, window_days=365)
    assert narrow.runs < wide.runs
    assert narrow.start > wide.start
    assert narrow.end == wide.end == latest


def test_empty_window_reports_none_not_zero(latest):
    """A window with no production is 'we do not know', not 'we executed
    0%'. Returning 0.0 here would read as total failure."""
    far_past = latest - timedelta(days=5000)
    rate = execution_rate(None, start=far_past,
                          end=far_past + timedelta(days=3))
    assert rate.rate is None
    assert rate.reliability == "unknown"


def test_unknown_primal_yields_no_rate(latest):
    rate = execution_rate("Not A Primal", end=latest, window_days=365)
    assert rate.rate is None
    assert rate.runs == 0


# ---------------------------------------------------------------------------
# Classification and fallback
# ---------------------------------------------------------------------------

def test_reliability_bands_match_thresholds(latest):
    for row in all_execution_rates(end=latest, window_days=365):
        if row.rate is None:
            assert row.reliability == "unknown"
        elif row.rate >= RELIABLE_THRESHOLD:
            assert row.reliability == "reliable"
        elif row.rate >= AT_RISK_THRESHOLD:
            assert row.reliability == "watch"
        else:
            assert row.reliability == "at_risk"


def test_all_rates_sorted_worst_first(latest):
    rates = [r.rate for r in all_execution_rates(end=latest, window_days=365)]
    assert rates == sorted(rates, key=lambda r: r if r is not None else 1.0)


def test_effective_rate_falls_back_for_thin_samples(latest):
    """A primal with too few runs must borrow the shop-wide rate AND say so
    - a rate off four runs is not a rate."""
    rate, basis = effective_rate_for("Not A Primal", end=latest)
    assert "shop-wide" in basis or "no production history" in basis
    assert 0.5 < rate <= 1.0


def test_effective_rate_uses_own_data_when_sample_is_large(latest):
    own = execution_rate("Blade Eyes", end=latest, window_days=365)
    assert own.runs >= MIN_RUNS_FOR_RELIABLE_RATE
    rate, basis = effective_rate_for("Blade Eyes", end=latest)
    assert "Blade Eyes" in basis
    assert "shop-wide" not in basis


# ---------------------------------------------------------------------------
# The headline use: discounting a plan
# ---------------------------------------------------------------------------

def test_expected_actual_discounts_the_plan(latest):
    out = expected_actual_kg(400.0, "Blade Eyes", end=latest)
    assert out["expected_actual_kg"] < 400.0, (
        "a plan must be discounted by execution reliability, not booked in full"
    )
    assert out["expected_actual_kg"] == pytest.approx(
        400.0 * out["execution_rate"], abs=0.02)
    assert out["expected_shortfall_kg"] == pytest.approx(
        400.0 - out["expected_actual_kg"], abs=0.02)


def test_expected_actual_carries_its_basis(latest):
    """An unattributed discount is a number an operator cannot argue with."""
    out = expected_actual_kg(100.0, "Blade Eyes", end=latest)
    assert out["rate_basis"]
    assert "runs" in out["rate_basis"]


def test_expected_actual_of_zero_plan_is_zero(latest):
    out = expected_actual_kg(0.0, "Blade Eyes", end=latest)
    assert out["expected_actual_kg"] == 0.0
    assert out["expected_shortfall_kg"] == 0.0
