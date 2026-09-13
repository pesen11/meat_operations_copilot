"""
The decision: several scored factors, one auditable call.

What this replaces: `_verdict()` in agents/nodes.py checked cutter
utilization against 90%, days of cover against 1.5, and margin delta against
zero. It was deterministic, which was the important part, and it was
effectively single-threshold - capacity decided almost every answer.
Everything else the system computed (waste, supply, execution reliability,
stockout exposure) was narrated and then not used to decide anything.

An operations person does not decide that way. They weigh several things at
once, and they let any one of them veto.

    score 0.0 = as bad as this factor gets
    score 1.0 = no concern at all

so the blended `decision_score` reads as "how comfortable is this", higher
always better, and no factor can invert the convention. Any factor may set
`is_blocking`, which no weighted average can outvote: a capacity overrun is
not compensated for by a good margin, because the week still does not fit.

A factor that CANNOT be assessed sets `assessed=False` and is excluded from
the blend rather than scored 1.0. "We did not look" and "we looked and it is
fine" must not produce the same number - that is how a system becomes
confident about things it never checked.

Every threshold here is an ASSUMPTION and is named as a module constant, for
the same reason the verdict thresholds were: they are the line between "go
ahead" and "a human should think about this", and they are the first thing
to correct against real operating experience.
"""

from __future__ import annotations

from typing import Any, Optional

from models.decision import (
    Action, DecisionFactor, FactorName, Recommendation, ScenarioOption, Verdict,
)
from models.provenance import Evidence, derived, from_business_rule, from_history, weakest

# --- Capacity ---------------------------------------------------------------
CAPACITY_COMFORT = 0.90     # above this, a caution; matches the old verdict line
CAPACITY_EASY = 0.70        # at or below this, no concern at all
CAPACITY_HARD_LIMIT = 1.00  # above this the week does not fit - blocking

# --- Inventory --------------------------------------------------------------
COVER_CRITICAL_DAYS = 1.0   # under a day of cutting left - blocking
COVER_COMFORT_DAYS = 3.0    # at or above this, no concern
STOCKOUT_BLOCKING = 0.50    # a coin-flip chance of running dry is not a plan
STOCKOUT_CONCERN = 0.15     # above this, a stated caution
STOCKOUT_COMFORT = 0.05     # the service level the sweep policy targets
COVER_CONCERN_DAYS = 1.5    # matches the old verdict's "no room for a missed
                            # delivery" line - a caution, not a blocker

# --- Economics --------------------------------------------------------------
# A change that loses money is blocking, as it was before. The score above
# zero scales against baseline margin so a $5 gain on a $3,000 base is not
# treated as a strong result.
MARGIN_STRONG_GAIN = 0.10   # +10% of baseline weekly margin = full marks

# --- Waste ------------------------------------------------------------------
# Waste rises with volume, so the question is whether it rises FASTER than
# volume. Proportional waste is not a finding.
WASTE_DISPROPORTION_CONCERN = 1.15   # 15% worse than proportional

# --- Execution --------------------------------------------------------------
EXECUTION_COMFORT = 0.95
EXECUTION_CONCERN = 0.85

# --- Supply -----------------------------------------------------------------
SUPPLY_COMFORT_RATIO = 1.00   # incoming covers forecast draw
SUPPLY_CONCERN_RATIO = 0.75

# Relative importance in the blended score. Capacity and inventory dominate
# because they are physical constraints; waste and execution are quality
# signals that should colour a decision without deciding it. ASSUMPTIONS.
FACTOR_WEIGHTS: dict[FactorName, float] = {
    FactorName.CAPACITY: 3.0,
    FactorName.INVENTORY: 3.0,
    FactorName.SUPPLY: 2.0,
    FactorName.ECONOMICS: 2.0,
    FactorName.EXECUTION: 1.0,
    FactorName.WASTE: 1.0,
}

# Decision-score bands, applied only when nothing blocks.
SCORE_PROCEED = 0.75
SCORE_CAUTION = 0.45


# The score at which a factor becomes a stated caution. Anything scoring
# below this appears in `cautions` and drags the blend accordingly.
CAUTION_SCORE = 0.5


def _ramp(value: float, good: float, bad: float) -> float:
    """Linear score in [0,1]: 1.0 at `good`, 0.0 at `bad`, either direction."""
    if good == bad:
        return 1.0
    raw = (value - bad) / (good - bad)
    return max(0.0, min(1.0, raw))


def _ramp_through(value: float, good: float, threshold: float) -> float:
    """
    Score 1.0 at `good`, exactly CAUTION_SCORE at `threshold`, 0.0 beyond.

    Every named threshold in this module claims to be the line between "fine"
    and "a human should look at this". A plain _ramp does not honour that
    claim: with good=0.95 and an arbitrary floor of 0.70, an execution rate
    sitting exactly on the 0.85 concern line scored 0.60 and raised no
    caution at all, so the constant named the wrong place. Anchoring the ramp
    so the threshold lands on CAUTION_SCORE makes crossing it mean precisely
    what the constant's name says it means.
    """
    if good == threshold:
        return 1.0
    return _ramp(value, good=good, bad=good + (threshold - good) / CAUTION_SCORE)


# ---------------------------------------------------------------------------
# Individual factors
# ---------------------------------------------------------------------------

def capacity_factor(labor: dict) -> DecisionFactor:
    utilization = labor.get("scenario_cutter_utilization_pct")
    if utilization is None:
        return DecisionFactor(
            name=FactorName.CAPACITY, score=0.0, weight=FACTOR_WEIGHTS[FactorName.CAPACITY],
            assessed=False, headline="Cutting capacity was not assessed.")

    exceeds = bool(labor.get("exceeds_cutter_capacity")) or utilization > CAPACITY_HARD_LIMIT
    score = _ramp_through(utilization, good=CAPACITY_EASY, threshold=CAPACITY_COMFORT)

    if exceeds:
        headline = (f"Cutting needs {utilization:.0%} of available cutter "
                    f"minutes - more than exists.")
    elif utilization > CAPACITY_COMFORT:
        headline = (f"Cutting would run at {utilization:.0%}, above the "
                    f"{CAPACITY_COMFORT:.0%} comfort line.")
    else:
        headline = f"Cutting runs at {utilization:.0%} of capacity, with room to spare."

    return DecisionFactor(
        name=FactorName.CAPACITY, score=score,
        weight=FACTOR_WEIGHTS[FactorName.CAPACITY],
        is_blocking=exceeds, headline=headline,
        detail={k: labor.get(k) for k in (
            "baseline_cutter_utilization_pct", "scenario_cutter_utilization_pct",
            "extra_weekly_cutting_minutes", "weekly_cutter_minutes_available")},
        evidence=derived(utilization, "cutter minutes required vs available"),
    )


def inventory_factor(inventory: dict, risk: Optional[dict] = None) -> DecisionFactor:
    cover = inventory.get("scenario_days_of_cover")
    probability = (risk or {}).get("stockout_probability")

    if cover is None and probability is None:
        return DecisionFactor(
            name=FactorName.INVENTORY, score=0.0,
            weight=FACTOR_WEIGHTS[FactorName.INVENTORY], assessed=False,
            headline="Stock position was not assessed.")

    scores: list[float] = []
    blocking = False
    parts: list[str] = []

    if cover is not None:
        scores.append(_ramp_through(cover, good=COVER_COMFORT_DAYS,
                                    threshold=COVER_CONCERN_DAYS))
        if cover < COVER_CRITICAL_DAYS:
            blocking = True
            parts.append(f"cover falls to {cover} days, under one day of cutting")
        else:
            parts.append(f"{cover} days of cover")

    if probability is not None:
        scores.append(_ramp_through(probability, good=STOCKOUT_COMFORT,
                                    threshold=STOCKOUT_CONCERN))
        if probability >= STOCKOUT_BLOCKING:
            blocking = True
        parts.append(f"{probability:.0%} chance of running out")

    return DecisionFactor(
        name=FactorName.INVENTORY, score=min(scores),
        weight=FACTOR_WEIGHTS[FactorName.INVENTORY],
        is_blocking=blocking,
        headline="Stock: " + ", ".join(parts) + ".",
        detail={"scenario_days_of_cover": cover,
                "stockout_probability": probability,
                "peak_risk_date": (risk or {}).get("peak_risk_date")},
        evidence=from_history(probability if probability is not None else cover,
                              "forward ledger over the planning horizon"),
    )


def supply_factor(supply: dict) -> DecisionFactor:
    draw = supply.get("forecast_primal_draw_kg")
    incoming = supply.get("incoming_expected_kg")
    opening = supply.get("opening_kg", 0.0)

    if draw is None or incoming is None or draw <= 0:
        return DecisionFactor(
            name=FactorName.SUPPLY, score=0.0, weight=FACTOR_WEIGHTS[FactorName.SUPPLY],
            assessed=False, headline="Incoming supply was not assessed.")

    ratio = (opening + incoming) / draw
    score = _ramp_through(ratio, good=SUPPLY_COMFORT_RATIO + 0.2,
                          threshold=SUPPLY_CONCERN_RATIO)
    reliability = supply.get("supply_reliability")

    if ratio < SUPPLY_CONCERN_RATIO:
        headline = (f"Stock plus incoming covers only {ratio:.0%} of forecast "
                    f"primal draw.")
    else:
        headline = (f"Stock plus incoming covers {ratio:.0%} of forecast draw.")
    if reliability is not None and reliability < 0.97:
        headline += f" Vendors fill {reliability:.0%} of ordered quantity."

    return DecisionFactor(
        name=FactorName.SUPPLY, score=score, weight=FACTOR_WEIGHTS[FactorName.SUPPLY],
        headline=headline,
        detail={"opening_kg": opening, "incoming_expected_kg": incoming,
                "forecast_primal_draw_kg": draw, "coverage_ratio": round(ratio, 4),
                "supply_reliability": reliability},
        evidence=from_history(round(ratio, 4), "open purchase orders vs forecast draw"),
    )


def economics_factor(margin: dict) -> DecisionFactor:
    delta = margin.get("delta_weekly_margin")
    baseline = margin.get("baseline_weekly_margin")

    if delta is None:
        if baseline is None:
            return DecisionFactor(
                name=FactorName.ECONOMICS, score=0.0,
                weight=FACTOR_WEIGHTS[FactorName.ECONOMICS], assessed=False,
                headline="No margin comparison was available.")
        return DecisionFactor(
            name=FactorName.ECONOMICS, score=1.0,
            weight=FACTOR_WEIGHTS[FactorName.ECONOMICS],
            headline=f"Weekly margin stays at ${baseline:,.2f}.",
            detail=dict(margin),
            evidence=derived(baseline, "baseline weekly margin"))

    if delta < 0:
        return DecisionFactor(
            name=FactorName.ECONOMICS, score=0.0,
            weight=FACTOR_WEIGHTS[FactorName.ECONOMICS], is_blocking=True,
            headline=f"Weekly margin falls by ${abs(delta):,.2f}.",
            detail=dict(margin),
            evidence=derived(delta, "scenario minus baseline weekly margin"))

    reference = abs(baseline) if baseline else max(abs(delta), 1.0)
    score = _ramp(delta / reference, good=MARGIN_STRONG_GAIN, bad=0.0)
    return DecisionFactor(
        name=FactorName.ECONOMICS, score=score,
        weight=FACTOR_WEIGHTS[FactorName.ECONOMICS],
        headline=f"Weekly margin rises by ${delta:,.2f}.",
        detail=dict(margin),
        evidence=derived(delta, "scenario minus baseline weekly margin"))


def waste_factor(waste: dict) -> DecisionFactor:
    """
    Waste is judged on PROPORTIONALITY, not on absolute increase.

    Cutting more meat produces more trim; that is arithmetic, not a problem.
    The question worth flagging is whether waste is rising faster than the
    volume driving it, which would mean the extra work is being done less
    efficiently than the baseline.
    """
    baseline = waste.get("baseline_weekly_trim_waste_kg")
    scenario = waste.get("scenario_weekly_trim_waste_kg")
    multiplier = waste.get("demand_multiplier")

    if baseline is None or scenario is None or not baseline:
        return DecisionFactor(
            name=FactorName.WASTE, score=0.0, weight=FACTOR_WEIGHTS[FactorName.WASTE],
            assessed=False, headline="Trim waste was not assessed.")

    actual_ratio = scenario / baseline
    expected_ratio = multiplier if multiplier else actual_ratio
    disproportion = actual_ratio / expected_ratio if expected_ratio else 1.0
    score = _ramp_through(disproportion, good=1.0,
                          threshold=WASTE_DISPROPORTION_CONCERN)

    # Prefer the extra figure the outcome already carries. Re-deriving it as
    # scenario - baseline lands a cent away (46.13 -> 53.04 subtracts to 6.91
    # where the outcome says 6.92, because the outcome computed it from extra
    # primal kg x trim pct). Two spellings of one quantity is exactly what
    # evals/number_guard.py flags, so the narration and the card must read
    # the same field.
    extra = waste.get("extra_weekly_trim_waste_kg")
    if extra is None:
        extra = round(scenario - baseline, 2)

    if disproportion > WASTE_DISPROPORTION_CONCERN:
        headline = (f"Trim waste rises {disproportion:.0%} faster than volume, "
                    f"to {scenario:,.2f} kg a week.")
    else:
        headline = (f"Trim waste rises by {extra:,.2f} kg a week, in proportion "
                    f"to the extra volume.")

    return DecisionFactor(
        name=FactorName.WASTE, score=score, weight=FACTOR_WEIGHTS[FactorName.WASTE],
        headline=headline,
        detail={"baseline_weekly_trim_waste_kg": baseline,
                "scenario_weekly_trim_waste_kg": scenario,
                "extra_weekly_trim_waste_kg": extra,
                "disproportion": round(disproportion, 4)},
        evidence=derived(extra, "extra trim waste from the extra primal"))


def execution_factor(execution: dict) -> DecisionFactor:
    rate = execution.get("execution_rate") or execution.get("rate")
    if rate is None:
        return DecisionFactor(
            name=FactorName.EXECUTION, score=0.0,
            weight=FACTOR_WEIGHTS[FactorName.EXECUTION], assessed=False,
            headline="Production reliability was not assessed.")

    score = _ramp_through(rate, good=EXECUTION_COMFORT, threshold=EXECUTION_CONCERN)
    scope = execution.get("scope") or "shop"
    scope = "The shop" if scope == "shop" else scope
    if rate < EXECUTION_CONCERN:
        headline = (f"{scope} completes only {rate:.0%} of planned cutting - "
                    f"a plan sized at 100% will fall short.")
    else:
        headline = f"{scope} completes {rate:.0%} of planned cutting."

    return DecisionFactor(
        name=FactorName.EXECUTION, score=score,
        weight=FACTOR_WEIGHTS[FactorName.EXECUTION],
        headline=headline,
        detail={k: execution.get(k) for k in
                ("scope", "rate", "execution_rate", "runs", "top_shortfall_reason",
                 "reliability", "rate_basis")},
        evidence=from_history(rate, execution.get("rate_basis") or
                              "planned vs actual production"),
    )


# ---------------------------------------------------------------------------
# The blend
# ---------------------------------------------------------------------------

def effective_multiplier(outcome: dict) -> float:
    """
    The multiplier actually applied, which is not always `demand_multiplier`.

    On a seasonal plan the operator supplies no size, so `demand_multiplier`
    stays 1.0 and the real uplift lives on `seasonal.multiplier` as the
    shop's confirmed business rule. Reading the wrong one made the waste
    factor compare a 1.9x scenario against a 1.0x expectation and conclude
    that trim waste was rising "190% faster than volume" - when it was rising
    exactly in proportion. That scored waste 0.00, dragged the whole decision
    to 0.45, and named waste as the limiting factor on a plan whose real
    constraint was cutter capacity.
    """
    seasonal = outcome.get("seasonal") or {}
    if seasonal.get("multiplier"):
        return float(seasonal["multiplier"])
    return float(outcome.get("demand_multiplier") or 1.0)


def build_factors(outcome: dict) -> list[DecisionFactor]:
    """Assemble every factor the outcome carries enough data to assess."""
    waste = dict(outcome.get("waste") or {})
    waste["demand_multiplier"] = effective_multiplier(outcome)

    # A whole-catalog plan has no single primal, so the single-primal blocks
    # are absent and these aggregates stand in for them. Without the
    # fallback, inventory and supply both reported "not assessed" on exactly
    # the questions that span the most primals.
    inventory = outcome.get("inventory") or {}
    risk = outcome.get("stockout_risk") or outcome.get("worst_stockout_risk")
    supply = outcome.get("supply") or outcome.get("supply_totals") or {}

    return [
        capacity_factor(outcome.get("labor") or {}),
        inventory_factor(inventory, risk),
        supply_factor(supply),
        economics_factor(outcome.get("margin") or {}),
        execution_factor(outcome.get("execution") or {}),
        waste_factor(waste),
    ]


def blended_score(factors: list[DecisionFactor]) -> Optional[float]:
    assessed = [f for f in factors if f.assessed]
    if not assessed:
        return None
    total_weight = sum(f.weight for f in assessed)
    return sum(f.score * f.weight for f in assessed) / total_weight


def _confidence(factors: list[DecisionFactor],
                evidence: Optional[Evidence]) -> float:
    """
    How much the system trusts its own call.

    Driven by coverage (how many factors could be assessed at all) and by the
    authority of the weakest evidence underneath it. A recommendation built
    on two assessed factors and a system default is not as good as one built
    on six and a measurement, and saying so is the point.
    """
    coverage = sum(1 for f in factors if f.assessed) / max(1, len(factors))
    authority = (evidence.authority / 6.0) if evidence is not None else 0.5
    return round(max(0.05, min(1.0, 0.5 * coverage + 0.5 * authority)), 4)


def decide(outcome: dict,
           sweep: Optional[Any] = None,
           operator_supplied_a_size: bool = True,
           evidence: Optional[Evidence] = None) -> Recommendation:
    """
    Turn a projected outcome into a structured Recommendation.

    `sweep` is an optional ScenarioSweep; when present its options and its
    selected rung are carried through, so a "how much should we" question
    returns the ladder it was chosen from rather than a bare number.
    """
    factors = build_factors(outcome)
    score = blended_score(factors)

    blockers = [f.headline for f in factors if f.is_blocking]
    cautions = [f.headline for f in factors
                if not f.is_blocking and f.assessed and f.score < CAUTION_SCORE]
    reasons = [f.headline for f in factors
               if f.assessed and f.score >= CAUTION_SCORE and not f.is_blocking]
    risks = [f.headline for f in factors
             if f.assessed and CAUTION_SCORE <= f.score < SCORE_PROCEED]

    intent = outcome.get("intent")
    informational = intent in ("history", "catalog", "inventory_status", "unsupported")

    if informational and not blockers:
        verdict = Verdict.INFORMATIONAL
        action = Action.NO_ACTION
    elif blockers:
        verdict = Verdict.DO_NOT_PROCEED
        action = Action.DO_NOT_CHANGE
    elif score is None:
        verdict = Verdict.INFORMATIONAL
        action = Action.NO_ACTION
    elif cautions or score < SCORE_PROCEED:
        verdict = (Verdict.PROCEED_WITH_CAUTION if score >= SCORE_CAUTION
                   else Verdict.DO_NOT_PROCEED)
        action = (Action.INCREASE if verdict == Verdict.PROCEED_WITH_CAUTION
                  and (outcome.get("demand_multiplier") or 1.0) > 1.0
                  else Action.MAINTAIN)
    else:
        verdict = Verdict.PROCEED
        multiplier = outcome.get("demand_multiplier") or 1.0
        action = (Action.INCREASE if multiplier > 1.0
                  else Action.DECREASE if multiplier < 1.0
                  else Action.MAINTAIN)

    assessed = [f for f in factors if f.assessed]
    limiting = min(assessed, key=lambda f: f.score).name if assessed else None

    options: list[ScenarioOption] = []
    chosen_label: Optional[str] = None
    recommended_multiplier: Optional[float] = None
    if sweep is not None:
        options = list(sweep.options)
        if sweep.recommended is not None:
            chosen_label = sweep.recommended.label
            # Only SUPPLY a size when the operator did not. Substituting a
            # number for one they chose would be overriding them.
            if not operator_supplied_a_size:
                recommended_multiplier = sweep.recommended.demand_multiplier
        if sweep.recommendation_reason:
            reasons.insert(0, sweep.recommendation_reason)

    basis = weakest(evidence, *[f.evidence for f in factors if f.evidence])

    return Recommendation(
        action=action,
        verdict=verdict,
        recommended_multiplier=recommended_multiplier,
        recommended_change_pct=(round(recommended_multiplier - 1.0, 4)
                                if recommended_multiplier is not None else None),
        confidence=_confidence(factors, basis),
        decision_score=score,
        factors=factors,
        blockers=blockers,
        cautions=cautions,
        reasons=reasons,
        risks=risks,
        options=options,
        chosen_option=chosen_label,
        requires_approval=verdict in (Verdict.PROCEED, Verdict.PROCEED_WITH_CAUTION),
        limiting_factor=limiting,
        confidence_basis=basis,
    )
