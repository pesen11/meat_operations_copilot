"""
Demand uncertainty: prediction intervals and stockout probability.

Every projection in this project has been a point estimate. "Christmas needs
4,769 kg more" reads as a fact, but it is the centre of a distribution, and
an operator deciding how much primal to buy needs the width of that
distribution far more than they need another decimal place on the centre.

    point estimate:  you need 1,240 kg
    with the width:  1,080-1,430 kg, and at 700 kg on hand plus 400 incoming
                     there is an 8% chance of running out

The second answer supports a decision. The first only supports a nod.

---------------------------------------------------------------------------
THE VOLATILITY IS MEASURED, NOT ASSUMED
---------------------------------------------------------------------------
history_generator.py injects day-to-day noise at SALES_NOISE_CV = 0.12, and
it would be circular to import that constant and call it the uncertainty:
that would test the generator against itself and would be worthless the
moment real sales data replaced the synthetic history.

So volatility is measured off the residuals - actual sales divided by the
seasonal expectation for that day - exactly as it would be against real
data. tests/test_uncertainty.py then asserts the measurement RECOVERS ~0.12,
which is a real check on the estimator rather than a restatement of a
constant.

---------------------------------------------------------------------------
WHAT THE MODEL ASSUMES, STATED PLAINLY
---------------------------------------------------------------------------
1. Residuals are lognormal. Sales are non-negative and right-skewed, so a
   normal interval would put mass below zero on low-volume products.
2. Day-to-day residuals are only PARTIALLY independent. Aggregating n days
   scales sigma by sqrt(n), which is then widened by DEMAND_AUTOCORRELATION
   because real demand shocks persist. That correction is an assumption, not
   a measurement, and is flagged as such on every aggregate result.
3. Supply arrives on its expected date. Delivery-timing risk lives in
   supply_calc's day ladder; this module reads that ladder rather than
   re-modelling it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np

from data.catalog import products_by_sku, yield_profiles_by_primal
from simulation import ops_data
from simulation.demand_forecast import expected_kg_for_day
from simulation.seasonality import is_closed
from simulation.units import units_to_kg

# Window over which residual volatility is measured. A full year, because
# volatility is a slow-moving property and a short window measures whatever
# happened to be going on that month. ASSUMPTION.
DEFAULT_VOLATILITY_WINDOW_DAYS = 365

# Below this many observations a per-product volatility is not trustworthy
# and the catalog-wide figure is used instead.
MIN_OBSERVATIONS_FOR_VOLATILITY = 30

# Fallback when nothing can be measured at all. Deliberately WIDE: an
# unmeasurable product is more uncertain than a measured one, and defaulting
# to a narrow band would present ignorance as precision. ASSUMPTION.
FALLBACK_CV = 0.25

DEFAULT_CONFIDENCE = 0.95

# Day-to-day demand autocorrelation. Strict independence understates the
# width of a multi-day total, because real demand shocks persist - a cold
# snap or a competitor's promotion runs for days, not for one afternoon.
# Var(sum of n correlated days) ~ n * sigma^2 * (1+rho)/(1-rho), so a rho of
# 0.25 widens an aggregate interval by ~29%. ASSUMPTION: not measured from
# this history (the generator draws each day independently, so measuring it
# here would return ~0 and encode the generator's simplification as a fact
# about butchery). It is a deliberately conservative correction in the
# direction real demand is known to err.
DEMAND_AUTOCORRELATION = 0.25

# Two-sided normal quantiles, tabulated so scipy is not a dependency.
_Z = {0.50: 0.6745, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def _z_for(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    nearest = min(_Z, key=lambda c: abs(c - confidence))
    return _Z[nearest]


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Measured volatility
# ---------------------------------------------------------------------------

@dataclass
class DemandVolatility:
    scope: str
    coefficient_of_variation: float
    observations: int
    measured: bool
    basis: str

    def to_dict(self) -> dict:
        return asdict(self)


def _residual_ratios(sku: Optional[str], start: date, end: date) -> list[float]:
    """actual / seasonal expectation, per open day."""
    sales = ops_data.load_sales()
    window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= end)]
    if sku is not None:
        window = window[window["sku"] == sku]

    ratios: list[float] = []
    if sku is not None:
        for row in window.itertuples(index=False):
            expected = expected_kg_for_day(row.sku, row.sale_date)
            if expected <= 0:
                continue
            ratios.append(units_to_kg(row.sku, float(row.units_sold)) / expected)
        return ratios

    # Catalog-wide: aggregate per DAY first. Averaging per-row ratios would
    # measure the volatility of a single product line, not of the shop - the
    # shop's total is steadier than any one product in it, because
    # independent product noise partially cancels.
    by_day: dict[date, tuple[float, float]] = {}
    for row in window.itertuples(index=False):
        expected = expected_kg_for_day(row.sku, row.sale_date)
        if expected <= 0:
            continue
        actual = units_to_kg(row.sku, float(row.units_sold))
        got, want = by_day.get(row.sale_date, (0.0, 0.0))
        by_day[row.sale_date] = (got + actual, want + expected)
    return [a / e for a, e in by_day.values() if e > 0]


def demand_volatility(sku: Optional[str] = None,
                      end: Optional[date] = None,
                      window_days: int = DEFAULT_VOLATILITY_WINDOW_DAYS) -> DemandVolatility:
    """
    Coefficient of variation of demand around its seasonal expectation.

    `sku=None` measures the whole catalog. Falls back to the catalog figure,
    and then to FALLBACK_CV, rather than reporting a confident number off a
    handful of observations.
    """
    sales = ops_data.load_sales()
    end = end or max(sales["sale_date"])
    start = end - timedelta(days=window_days - 1)

    ratios = _residual_ratios(sku, start, end)
    scope = sku or "catalog"

    if len(ratios) >= MIN_OBSERVATIONS_FOR_VOLATILITY:
        mean = float(np.mean(ratios))
        if mean > 0:
            cv = float(np.std(ratios, ddof=1)) / mean
            return DemandVolatility(
                scope=scope, coefficient_of_variation=round(cv, 4),
                observations=len(ratios), measured=True,
                basis=f"{len(ratios)} days of residuals to {end.isoformat()}")

    if sku is not None:
        catalog = demand_volatility(None, end=end, window_days=window_days)
        return DemandVolatility(
            scope=scope,
            coefficient_of_variation=catalog.coefficient_of_variation,
            observations=len(ratios), measured=False,
            basis=f"catalog-wide fallback; {scope} had only {len(ratios)} observations")

    return DemandVolatility(scope=scope, coefficient_of_variation=FALLBACK_CV,
                            observations=len(ratios), measured=False,
                            basis="no measurable history; wide default applied")


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------

@dataclass
class DemandInterval:
    scope: str
    start: date
    end: date
    days: int
    point_estimate_kg: float
    lower_kg: float
    upper_kg: float
    confidence: float
    daily_cv: float
    aggregate_cv: float
    volatility_measured: bool
    independence_assumed: bool
    basis: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d


def _aggregate_cv(daily_cv: float, open_days: int,
                  autocorrelation: float = DEMAND_AUTOCORRELATION) -> float:
    """Daily CV -> CV of an n-day total.

    Totalling n independent days multiplies the mean by n and the standard
    deviation by sqrt(n), so the CV of the total shrinks by sqrt(n). That
    shrinkage is why a weekly figure is far more predictable than a daily
    one - and why quoting a daily CV on a weekly total would badly overstate
    the width.

    The inflation factor corrects that shrinkage for persistence: positive
    autocorrelation means a high day is more likely to be followed by
    another high day, so the total is less self-cancelling than independence
    implies. See DEMAND_AUTOCORRELATION.
    """
    if open_days <= 0:
        return daily_cv
    rho = min(0.9, max(0.0, autocorrelation))
    inflation = math.sqrt((1 + rho) / (1 - rho))
    return daily_cv / math.sqrt(open_days) * inflation


def demand_interval(sku: str, start: date, end: date,
                    confidence: float = DEFAULT_CONFIDENCE,
                    demand_multiplier: float = 1.0) -> DemandInterval:
    """
    Expected kg over a window, with a lognormal prediction interval.

    Lognormal rather than normal so the lower bound cannot go negative - on
    a low-volume product with a 25% CV, a symmetric 95% interval puts its
    floor below zero, which is not a quantity of meat.
    """
    if sku not in products_by_sku:
        raise ValueError(f"Unknown product_sku '{sku}'")

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    open_days = [d for d in days if not is_closed(d)]
    point = sum(expected_kg_for_day(sku, d) for d in open_days) * demand_multiplier

    volatility = demand_volatility(sku, end=end)
    daily_cv = volatility.coefficient_of_variation
    agg_cv = _aggregate_cv(daily_cv, len(open_days))

    if point <= 0 or agg_cv <= 0:
        lower = upper = max(0.0, point)
    else:
        sigma = math.sqrt(math.log(1 + agg_cv ** 2))
        z = _z_for(confidence)
        lower = point * math.exp(-z * sigma)
        upper = point * math.exp(z * sigma)

    return DemandInterval(
        scope=sku, start=start, end=end, days=len(days),
        point_estimate_kg=round(point, 2),
        lower_kg=round(lower, 2), upper_kg=round(upper, 2),
        confidence=confidence,
        daily_cv=round(daily_cv, 4),
        aggregate_cv=round(agg_cv, 4),
        volatility_measured=volatility.measured,
        independence_assumed=True,
        basis=volatility.basis,
    )


def primal_demand_interval(source_primal: str, start: date, end: date,
                           confidence: float = DEFAULT_CONFIDENCE,
                           demand_multiplier: float = 1.0) -> DemandInterval:
    """
    Raw primal kg a primal must supply over a window, with an interval.

    Each yield path's finished-product interval is converted to primal by
    dividing by that path's own yield_pct, then the bounds are summed. That
    sum is CONSERVATIVE - it assumes every path is simultaneously at its own
    upper bound, which independence says is unlikely - and conservative is
    the right direction for a stock decision.
    """
    paths = yield_profiles_by_primal().get(source_primal, [])
    if not paths:
        raise ValueError(f"Unknown primal '{source_primal}'")

    point = lower = upper = 0.0
    cvs: list[float] = []
    measured = True
    for yp in paths:
        interval = demand_interval(yp.product_sku, start, end, confidence,
                                   demand_multiplier)
        point += interval.point_estimate_kg / yp.yield_pct
        lower += interval.lower_kg / yp.yield_pct
        upper += interval.upper_kg / yp.yield_pct
        cvs.append(interval.aggregate_cv)
        measured = measured and interval.volatility_measured

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    open_days = [d for d in days if not is_closed(d)]
    agg_cv = float(np.mean(cvs)) if cvs else 0.0
    daily_cv = agg_cv * math.sqrt(max(1, len(open_days)))

    return DemandInterval(
        scope=source_primal, start=start, end=end, days=len(days),
        point_estimate_kg=round(point, 2),
        lower_kg=round(lower, 2), upper_kg=round(upper, 2),
        confidence=confidence,
        daily_cv=round(daily_cv, 4),
        aggregate_cv=round(agg_cv, 4),
        volatility_measured=measured,
        independence_assumed=True,
        basis=f"{len(paths)} yield path(s) off {source_primal}, summed at their bounds",
    )


# ---------------------------------------------------------------------------
# Stockout probability
# ---------------------------------------------------------------------------

@dataclass
class DayRisk:
    """Stockout probability on one day of the horizon."""
    day: date
    cumulative_demand_kg: float
    cumulative_supply_kg: float
    headroom_kg: float
    probability: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d


@dataclass
class StockoutRisk:
    source_primal: str
    as_of: date
    horizon_days: int
    available_kg: float
    expected_demand_kg: float
    demand_sigma_kg: float
    headroom_kg: float
    stockout_probability: float
    risk_band: str
    service_level: float
    peak_risk_date: Optional[date]
    days_until_peak_risk: Optional[int]
    horizon_end_probability: float
    independence_assumed: bool
    basis: str
    daily: list["DayRisk"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["peak_risk_date"] = (self.peak_risk_date.isoformat()
                               if self.peak_risk_date else None)
        d["daily"] = [x.to_dict() for x in self.daily]
        return d


# Risk bands. ASSUMPTIONS - the line an operator would draw between "fine",
# "watch it" and "do something today".
RISK_BANDS = ((0.05, "low"), (0.15, "moderate"), (0.30, "elevated"))


def _band(probability: float) -> str:
    for threshold, label in RISK_BANDS:
        if probability < threshold:
            return label
    return "high"


def _lognormal_exceedance(threshold: float, mu: float, cv: float) -> float:
    """P(X > threshold) for X lognormal with mean `mu` and CV `cv`."""
    if mu <= 0:
        return 0.0
    if cv <= 0:
        return 1.0 if threshold < mu else 0.0
    if threshold <= 0:
        return 1.0
    shape = math.sqrt(math.log(1 + cv ** 2))
    scale = math.log(mu) - 0.5 * shape ** 2
    return 1.0 - _phi((math.log(threshold) - scale) / shape)


def _forecast_draw(source_primal: str, day: date, demand_multiplier: float):
    from simulation.supply_calc import forecast_primal_draw  # local: avoids a cycle
    return forecast_primal_draw(source_primal, day, demand_multiplier)


def stockout_probability(source_primal: str,
                         as_of: Optional[date] = None,
                         horizon_days: int = 14,
                         demand_multiplier: float = 1.0,
                         extra_supply_kg: float = 0.0) -> StockoutRisk:
    """
    Probability of running dry at ANY point in the horizon.

    -----------------------------------------------------------------------
    WHY THIS WALKS THE LADDER INSTEAD OF COMPARING 14-DAY TOTALS
    -----------------------------------------------------------------------
    An earlier version compared total horizon demand against total available
    supply. It returned 0.000 for every primal in the catalog, including
    ones whose day-by-day ledger runs dry repeatedly - because aggregating
    fourteen days shrinks the CV by sqrt(n) until the two totals sit eleven
    sigma apart, and because a total cannot express TIMING. A cooler that
    empties on day five while its next delivery lands on day six has a
    perfectly healthy fourteen-day balance and a stockout on day five.

    So risk is evaluated at every day k: P(cumulative demand through k >
    opening + everything scheduled to arrive through k). The horizon's
    figure is the maximum over k, which is where the cooler is genuinely
    tightest. That is a lower bound on the true any-day probability - the
    daily events overlap heavily, so their union sits only slightly above
    their maximum - and a lower bound is the honest direction to err in a
    number whose job is to trigger an order.
    """
    from simulation.supply_calc import incoming_by_day, project_supply  # local: avoids a cycle

    projection = project_supply(source_primal, as_of=as_of,
                                horizon_days=horizon_days,
                                demand_multiplier=demand_multiplier)
    as_of = projection.as_of
    arrivals = incoming_by_day(source_primal, as_of, horizon_days)

    # Daily CV for this primal, taken from a one-day interval so the same
    # measured volatility feeds both the interval API and the risk API.
    one_day = as_of + timedelta(days=1)
    volatility_cv = primal_demand_interval(
        source_primal, one_day, one_day,
        demand_multiplier=demand_multiplier).daily_cv

    cumulative_supply = projection.opening_kg + extra_supply_kg
    cumulative_demand = 0.0
    open_days = 0
    daily: list[DayRisk] = []

    for offset in range(1, horizon_days + 1):
        day = as_of + timedelta(days=offset)
        cumulative_supply += arrivals.get(day, 0.0)
        _product, draw = _forecast_draw(source_primal, day, demand_multiplier)
        cumulative_demand += draw
        if draw > 0:
            open_days += 1

        cv = _aggregate_cv(volatility_cv, open_days) if open_days else 0.0
        probability = _lognormal_exceedance(cumulative_supply, cumulative_demand, cv)

        daily.append(DayRisk(
            day=day,
            cumulative_demand_kg=round(cumulative_demand, 2),
            cumulative_supply_kg=round(cumulative_supply, 2),
            headroom_kg=round(cumulative_supply - cumulative_demand, 2),
            probability=round(probability, 4),
        ))

    peak = max(daily, key=lambda r: r.probability) if daily else None
    probability = peak.probability if peak else 0.0
    total_demand = daily[-1].cumulative_demand_kg if daily else 0.0
    total_supply = daily[-1].cumulative_supply_kg if daily else 0.0
    final_cv = _aggregate_cv(volatility_cv, max(1, open_days))

    return StockoutRisk(
        source_primal=source_primal,
        as_of=as_of,
        horizon_days=horizon_days,
        available_kg=round(total_supply, 2),
        expected_demand_kg=round(total_demand, 2),
        demand_sigma_kg=round(total_demand * final_cv, 2),
        headroom_kg=round(total_supply - total_demand, 2),
        stockout_probability=round(probability, 4),
        risk_band=_band(probability),
        service_level=round(1 - probability, 4),
        peak_risk_date=peak.day if peak and peak.probability > 0 else None,
        days_until_peak_risk=((peak.day - as_of).days
                              if peak and peak.probability > 0 else None),
        horizon_end_probability=daily[-1].probability if daily else 0.0,
        independence_assumed=False,
        basis=(f"opening {projection.opening_kg} kg + expected incoming "
               f"{projection.incoming_expected_kg} kg"
               + (f" + {round(extra_supply_kg, 2)} kg extra" if extra_supply_kg else "")
               + f"; worst day of {horizon_days}"),
        daily=daily,
    )


def supply_needed_for_service_level(source_primal: str,
                                    target_service_level: float = 0.95,
                                    as_of: Optional[date] = None,
                                    horizon_days: int = 14,
                                    demand_multiplier: float = 1.0,
                                    max_extra_kg: Optional[float] = None) -> dict:
    """
    How much extra primal buys a given service level.

    Solved by bisection on `extra_supply_kg` rather than by inverting a
    quantile, because the risk is now a maximum over a day ladder and has no
    closed form - extra supply has to be walked through the same ledger to
    see which day it actually rescues. Risk is monotone decreasing in the
    extra quantity, so bisection converges cleanly; 24 iterations is well
    past float precision at these magnitudes.
    """
    target_risk = 1.0 - target_service_level
    base = stockout_probability(source_primal, as_of=as_of,
                                horizon_days=horizon_days,
                                demand_multiplier=demand_multiplier)

    if base.stockout_probability <= target_risk:
        return {
            "source_primal": source_primal,
            "target_service_level": target_service_level,
            "expected_demand_kg": base.expected_demand_kg,
            "available_kg": base.available_kg,
            "extra_needed_kg": 0.0,
            "current_stockout_probability": base.stockout_probability,
            "resulting_stockout_probability": base.stockout_probability,
            "horizon_days": horizon_days,
            "already_met": True,
        }

    high = (max_extra_kg if max_extra_kg is not None
            else max(base.expected_demand_kg, 1.0) * 2)
    low = 0.0
    for _ in range(24):
        mid = (low + high) / 2
        risk = stockout_probability(source_primal, as_of=as_of,
                                    horizon_days=horizon_days,
                                    demand_multiplier=demand_multiplier,
                                    extra_supply_kg=mid).stockout_probability
        if risk <= target_risk:
            high = mid
        else:
            low = mid

    final = stockout_probability(source_primal, as_of=as_of,
                                 horizon_days=horizon_days,
                                 demand_multiplier=demand_multiplier,
                                 extra_supply_kg=high)
    return {
        "source_primal": source_primal,
        "target_service_level": target_service_level,
        "expected_demand_kg": base.expected_demand_kg,
        "available_kg": base.available_kg,
        "extra_needed_kg": round(high, 2),
        "current_stockout_probability": base.stockout_probability,
        "resulting_stockout_probability": final.stockout_probability,
        "horizon_days": horizon_days,
        "already_met": False,
    }
