"""
Sweep a ladder of production levels and find the boundary.

An operator asking "how much more should I make for Christmas" is not asking
for a number. They are choosing between options, and what they need is the
EDGE: the largest increase the shop can actually execute, or the smallest
one that covers demand at an acceptable risk of running out.

A point estimate cannot express an edge. "Christmas needs 1.9x" is true and
says nothing about whether 1.9x is reachable, what it costs, or what the
next-best rung looks like if it is not. So this evaluates a ladder:

    +5%   stockout risk 18%   cutter 84%   feasible
    +10%  stockout risk 10%   cutter 88%   feasible
    +15%  stockout risk  4%   cutter 92%   feasible, above comfort line
    +20%  stockout risk  2%   cutter 96%   feasible, above comfort line
    +25%  stockout risk  1%   cutter 101%  NOT feasible - capacity

and then applies a stated selection policy to pick one, so the choice is
auditable rather than a matter of taste.

Every rung is computed by the SAME deterministic functions the single-point
path uses (scenario_impact, catalog_impact, weekly_projection). A sweep that
re-derived its own arithmetic would be a second implementation free to
disagree with the first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

from data.catalog import demand_by_sku, products_by_sku, yield_profile_by_sku
from models.decision import ScenarioOption
from simulation.bottlenecks import catalog_impact
from simulation.cost_calc import weekly_projection
from simulation.inventory_calc import DEFAULT_LOOKBACK_DAYS, scenario_impact
from simulation.uncertainty import stockout_probability

# The ladder, as fractional changes from today. Spans a meaningful cut
# through to a near-doubling because both directions are real questions -
# "should we scale back brisket" is as common as "can we do more ribeye".
# ASSUMPTION: the rungs are a presentation choice, not a business input.
DEFAULT_LADDER = (-0.20, -0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.90)

# Selection policy thresholds. ASSUMPTIONS, stated here rather than buried
# in a comparison, because they decide what the system recommends.
MAX_ACCEPTABLE_STOCKOUT_RISK = 0.05    # the service level a rung must buy
MAX_SAFE_CUTTER_UTILIZATION = 0.90     # matches agents/nodes.py's comfort line
HARD_CUTTER_LIMIT = 1.0                # past this, the week does not fit


@dataclass
class ScenarioSweep:
    """A ladder of evaluated options and the one the policy selects."""
    scope: str                       # a SKU, or "catalog"
    as_of: date
    horizon_days: int
    baseline_label: str
    options: list[ScenarioOption]
    recommended: Optional[ScenarioOption]
    recommendation_reason: str
    target_multiplier: Optional[float] = None
    target_is_feasible: Optional[bool] = None
    policy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "as_of": self.as_of.isoformat(),
            "horizon_days": self.horizon_days,
            "baseline_label": self.baseline_label,
            "options": [o.to_dict() for o in self.options],
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "recommendation_reason": self.recommendation_reason,
            "target_multiplier": self.target_multiplier,
            "target_is_feasible": self.target_is_feasible,
            "policy": self.policy,
        }


def _label(multiplier: float) -> str:
    change = multiplier - 1.0
    if abs(change) < 1e-9:
        return "no change"
    return f"{change:+.0%}"


def _feasibility(utilization: float, exceeds: bool,
                 stockout_risk: Optional[float],
                 risk_scope: Optional[str] = None) -> tuple[bool, list[str]]:
    """A rung is feasible if the shop could physically execute it.

    Note what is NOT here: margin. A less profitable option is a worse
    option, not an impossible one, and folding profitability into
    feasibility would let the sweep silently rule out a choice the operator
    is entitled to make.
    """
    reasons: list[str] = []
    if exceeds or utilization > HARD_CUTTER_LIMIT:
        reasons.append(f"Cutting would need {utilization:.0%} of available "
                       f"cutter minutes")
    if stockout_risk is not None and stockout_risk >= 0.50:
        # Name the primal. A bare "Primal would run out" tells an operator
        # nothing they can act on, and reads as a placeholder that was never
        # filled in.
        subject = risk_scope or "Primal supply"
        reasons.append(f"{subject} would run out with {stockout_risk:.0%} probability")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Single product
# ---------------------------------------------------------------------------

def _primal_multiplier_for_product(impact, source_primal: str,
                                   as_of: Optional[date],
                                   lookback_days: int) -> float:
    """
    Translate a PRODUCT multiplier into the multiplier its PRIMAL actually sees.

    These are not the same number whenever a primal feeds more than one
    product, and treating them as the same badly overstates supply risk.
    Blade Eyes is cut 65% into thin-sliced blades and 35% into chuck roast,
    so a 50% increase in chuck roast raises the primal's total draw by about
    17%, not 50%. Passing the product's 1.5 straight through reported a 99.8%
    chance of running Blade Eyes dry for a change that barely moves it.

        primal multiplier = 1 + (extra primal this path needs)
                                / (the primal's whole current weekly draw)

    The denominator is the sales-derived draw across every path off the
    primal - the same figure days-of-cover is computed from - so a
    single-path primal collapses back to exactly the product multiplier.
    """
    from simulation.inventory_calc import avg_daily_primal_kg

    weekly_primal_draw = avg_daily_primal_kg(
        as_of=as_of, lookback_days=lookback_days).get(source_primal, 0.0) * 7
    if weekly_primal_draw <= 0:
        return 1.0
    return 1.0 + impact.extra_weekly_primal_kg / weekly_primal_draw



def sweep_product(product_sku: str,
                  as_of: Optional[date] = None,
                  ladder: tuple[float, ...] = DEFAULT_LADDER,
                  target_multiplier: Optional[float] = None,
                  horizon_days: int = 14,
                  lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> ScenarioSweep:
    """Evaluate one product at every rung on the ladder."""
    if product_sku not in products_by_sku:
        raise ValueError(f"Unknown product_sku '{product_sku}'")

    product = products_by_sku[product_sku]
    yp = yield_profile_by_sku[product_sku]
    demand = demand_by_sku[product_sku]

    multipliers = sorted({round(1 + c, 4) for c in ladder} |
                         ({round(target_multiplier, 4)} if target_multiplier else set()))
    multipliers = [m for m in multipliers if m > 0]

    baseline_margin = weekly_projection(yp, product, demand).weekly_margin
    options: list[ScenarioOption] = []
    resolved_as_of = as_of

    for multiplier in multipliers:
        impact = scenario_impact(product_sku, multiplier, as_of=as_of,
                                 lookback_days=lookback_days)
        resolved_as_of = resolved_as_of or date.today()
        projection = weekly_projection(yp, product, demand,
                                       demand_multiplier=multiplier)
        risk = stockout_probability(
            yp.source_primal, as_of=as_of, horizon_days=horizon_days,
            demand_multiplier=_primal_multiplier_for_product(impact, yp.source_primal,
                                                             as_of, lookback_days))
        feasible, reasons = _feasibility(
            impact.scenario_cutter_utilization_pct,
            impact.exceeds_cutter_capacity,
            risk.stockout_probability,
            risk_scope=yp.source_primal)

        options.append(ScenarioOption(
            label=_label(multiplier),
            demand_multiplier=multiplier,
            extra_weekly_product_kg=impact.extra_weekly_product_kg,
            extra_weekly_primal_kg=impact.extra_weekly_primal_kg,
            extra_weekly_boxes=impact.extra_weekly_boxes,
            cutter_utilization_pct=impact.scenario_cutter_utilization_pct,
            exceeds_cutter_capacity=impact.exceeds_cutter_capacity,
            extra_weekly_trim_waste_kg=impact.extra_weekly_trim_waste_kg,
            weekly_margin=projection.weekly_margin,
            delta_weekly_margin=round(projection.weekly_margin - baseline_margin, 2),
            stockout_probability=risk.stockout_probability,
            days_of_cover=impact.scenario_days_of_cover,
            feasible=feasible,
            infeasible_reasons=reasons,
        ))

    from simulation import ops_data
    resolved_as_of = as_of or max(ops_data.load_sales()["sale_date"])

    chosen, reason = select_option(options, target_multiplier)
    return ScenarioSweep(
        scope=product_sku,
        as_of=resolved_as_of,
        horizon_days=horizon_days,
        baseline_label=f"{product.name} at current demand",
        options=options,
        recommended=chosen,
        recommendation_reason=reason,
        target_multiplier=target_multiplier,
        target_is_feasible=_target_feasible(options, target_multiplier),
        policy=_policy_dict(),
    )


# ---------------------------------------------------------------------------
# Whole catalog
# ---------------------------------------------------------------------------

def sweep_catalog(as_of: Optional[date] = None,
                  ladder: tuple[float, ...] = DEFAULT_LADDER,
                  target_multiplier: Optional[float] = None,
                  horizon_days: int = 14,
                  lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> ScenarioSweep:
    """
    Evaluate the whole catalog at every rung.

    Stockout risk at catalog level is the WORST primal's, not an average.
    Averaging would let sixteen comfortable primals hide the one that is
    about to run dry, and the shop stops cutting when any single primal it
    needs is empty - not when the mean primal is empty.
    """
    multipliers = sorted({round(1 + c, 4) for c in ladder} |
                         ({round(target_multiplier, 4)} if target_multiplier else set()))
    multipliers = [m for m in multipliers if m > 0]

    options: list[ScenarioOption] = []
    resolved_as_of = as_of

    for multiplier in multipliers:
        impact = catalog_impact(multiplier, as_of=as_of,
                                horizon_days=horizon_days,
                                lookback_days=lookback_days)
        resolved_as_of = resolved_as_of or impact.as_of

        risks = [row.get("stockout_probability") for row in impact.per_primal
                 if row.get("stockout_probability") is not None]
        worst_risk = max(risks) if risks else None

        worst_primal = None
        if risks:
            worst_primal = max(
                (r for r in impact.per_primal
                 if r.get("stockout_probability") is not None),
                key=lambda r: r["stockout_probability"])["source_primal"]
        feasible, reasons = _feasibility(
            impact.cutter_utilization_pct, impact.exceeds_cutter_capacity,
            worst_risk, risk_scope=worst_primal)
        if impact.bottlenecks and impact.bottlenecks[0].severity == "over_limit":
            top = impact.bottlenecks[0]
            if top.kind in ("capacity", "stock") and top.headline not in reasons:
                reasons.append(top.headline)
                feasible = False

        options.append(ScenarioOption(
            label=_label(multiplier),
            demand_multiplier=multiplier,
            extra_weekly_product_kg=impact.total_extra_weekly_product_kg,
            extra_weekly_primal_kg=impact.total_extra_weekly_primal_kg,
            extra_weekly_boxes=impact.total_extra_weekly_boxes,
            cutter_utilization_pct=impact.cutter_utilization_pct,
            exceeds_cutter_capacity=impact.exceeds_cutter_capacity,
            extra_weekly_trim_waste_kg=impact.total_extra_trim_waste_kg,
            weekly_margin=None,
            delta_weekly_margin=None,
            stockout_probability=worst_risk,
            days_of_cover=None,
            feasible=feasible,
            infeasible_reasons=reasons,
        ))

    chosen, reason = select_option(options, target_multiplier)
    return ScenarioSweep(
        scope="catalog",
        as_of=resolved_as_of or date.today(),
        horizon_days=horizon_days,
        baseline_label="whole catalog at current demand",
        options=options,
        recommended=chosen,
        recommendation_reason=reason,
        target_multiplier=target_multiplier,
        target_is_feasible=_target_feasible(options, target_multiplier),
        policy=_policy_dict(),
    )


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------

def _policy_dict() -> dict:
    return {
        "max_acceptable_stockout_risk": MAX_ACCEPTABLE_STOCKOUT_RISK,
        "max_safe_cutter_utilization": MAX_SAFE_CUTTER_UTILIZATION,
        "hard_cutter_limit": HARD_CUTTER_LIMIT,
        "rule": ("When the operator names a size, that size is evaluated and "
                 "the nearest feasible rung is offered if it does not fit. "
                 "When they ask how much, the largest feasible rung that "
                 "stays inside the cutter comfort line is chosen; if none "
                 "does, the largest merely-feasible rung is chosen and the "
                 "comfort breach is reported."),
    }


def _target_feasible(options: list[ScenarioOption],
                     target: Optional[float]) -> Optional[bool]:
    if target is None:
        return None
    for option in options:
        if abs(option.demand_multiplier - target) < 1e-6:
            return option.feasible
    return None


def select_option(options: list[ScenarioOption],
                  target_multiplier: Optional[float] = None
                  ) -> tuple[Optional[ScenarioOption], str]:
    """
    Apply the stated policy. Returns (option, human-readable reason).

    Two different questions, two different rules:

    - The operator NAMED a size. Their number is the answer; the policy's
      only job is to say whether it fits and, when it does not, to offer the
      nearest rung that does. Substituting a different number for one the
      operator chose would be overriding them, not advising them.
    - The operator ASKED for a size. Then the policy picks: the largest rung
      that is both feasible and inside the comfort line, because on a
      seasonal ramp the constraint is how much can be produced, not how
      little.
    """
    if not options:
        return None, "No options were evaluated."

    increases = [o for o in options if o.demand_multiplier > 1.0]

    if target_multiplier is not None:
        named = next((o for o in options
                      if abs(o.demand_multiplier - target_multiplier) < 1e-6), None)
        if named is not None and named.feasible:
            if named.cutter_utilization_pct > MAX_SAFE_CUTTER_UTILIZATION:
                return named, (
                    f"{named.label} is achievable, but cutting would run at "
                    f"{named.cutter_utilization_pct:.0%} - above the "
                    f"{MAX_SAFE_CUTTER_UTILIZATION:.0%} comfort line.")
            return named, f"{named.label} fits within current capacity and supply."

        feasible_below = [o for o in options
                          if o.feasible and o.demand_multiplier < target_multiplier]
        if feasible_below:
            best = max(feasible_below, key=lambda o: o.demand_multiplier)
            blocker = (named.infeasible_reasons[0] if named and named.infeasible_reasons
                       else "it exceeds available capacity or supply")
            return best, (f"{_label(target_multiplier)} is not achievable - {blocker}. "
                          f"{best.label} is the largest change that is.")
        return named, (f"{_label(target_multiplier)} is not achievable, and no "
                       f"smaller increase on the ladder is either.")

    comfortable = [o for o in increases
                   if o.feasible and o.cutter_utilization_pct <= MAX_SAFE_CUTTER_UTILIZATION]
    if comfortable:
        best = max(comfortable, key=lambda o: o.demand_multiplier)
        return best, (f"{best.label} is the largest increase that stays inside "
                      f"the {MAX_SAFE_CUTTER_UTILIZATION:.0%} cutter comfort line.")

    feasible = [o for o in increases if o.feasible]
    if feasible:
        best = max(feasible, key=lambda o: o.demand_multiplier)
        return best, (f"{best.label} is the largest achievable increase, but it "
                      f"runs cutting at {best.cutter_utilization_pct:.0%}, above "
                      f"the {MAX_SAFE_CUTTER_UTILIZATION:.0%} comfort line.")

    no_change = next((o for o in options if abs(o.demand_multiplier - 1.0) < 1e-6), None)
    return no_change, ("No increase on the ladder is achievable with current "
                       "capacity and supply.")


def smallest_option_meeting_risk(options: list[ScenarioOption],
                                 max_risk: float = MAX_ACCEPTABLE_STOCKOUT_RISK
                                 ) -> Optional[ScenarioOption]:
    """
    The least disruptive rung that still holds stockout risk under `max_risk`.

    The mirror of `select_option`'s ramp rule, for the opposite question:
    not "how much can we push" but "what is the minimum that keeps us safe".
    """
    qualifying = [o for o in options
                  if o.feasible and o.stockout_probability is not None
                  and o.stockout_probability <= max_risk]
    if not qualifying:
        return None
    return min(qualifying, key=lambda o: abs(o.demand_multiplier - 1.0))
