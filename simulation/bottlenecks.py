"""
What is actually binding, when a plan spans the whole catalog.

The problem this replaces: a whole-catalog seasonal plan covers seventeen
primals, so `yield_analyzer` set

    baseline_days_of_cover = None
    scenario_days_of_cover = None

and explained, correctly, that one combined days-of-cover figure across
seventeen primals would be meaningless. That reasoning was right and the
result was still unhelpful - the operator got a hole where the constraint
should have been. "Meaningless to average" does not mean "nothing to say";
it means the right answer is a RANKING, not a mean.

Three layers, which is what a catalog-wide answer actually needs:

    catalog     one number per resource, where one number is honest
                (total extra kg, total cutter minutes - these genuinely add)
    per-primal  the same measures, not averaged
    bottleneck  the handful that bind, worst first, each with a named cause

A bottleneck is a constraint that is *close to or past* its limit, scored by
how tight it is, so "which three things do I fix" has an answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

from data.catalog import (
    primal_reference_box_weight_kg, products_by_sku, yield_profile_by_sku,
    yield_profiles_by_primal,
)
from simulation import ops_data
from simulation.execution_rates import (
    AT_RISK_THRESHOLD, RELIABLE_THRESHOLD, effective_rate_for,
)
from simulation.inventory_calc import (
    DEFAULT_LOOKBACK_DAYS, avg_weekly_cutter_minutes, stock_position,
)
from simulation.uncertainty import stockout_probability

# A constraint at or above this fraction of its limit is reported as binding.
# Below it, the constraint has real slack and naming it would bury the ones
# that matter. ASSUMPTION - the same judgement as the 90% cutter comfort
# line, applied generally.
BINDING_THRESHOLD = 0.85

# Severity bands for the ranked list.
SEVERITY_BANDS = ((1.0, "over_limit"), (0.95, "critical"), (BINDING_THRESHOLD, "tight"))


@dataclass
class Bottleneck:
    """One binding constraint, named and scored."""
    kind: str                  # capacity | stock | supply | execution
    scope: str                 # primal name, or "shop"
    headline: str
    utilization: float         # 1.0 = exactly at the limit
    severity: str              # over_limit | critical | tight
    detail: dict
    remedy: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CatalogImpact:
    """Whole-catalog rollup with the constraints named rather than averaged."""
    as_of: date
    demand_multiplier: float
    horizon_days: int

    total_extra_weekly_product_kg: float
    total_extra_weekly_primal_kg: float
    total_extra_weekly_boxes: float
    total_extra_trim_waste_kg: float
    total_extra_cutting_minutes: float
    weekly_cutter_minutes_available: float
    cutter_utilization_pct: float
    exceeds_cutter_capacity: bool

    primals_assessed: int
    bottlenecks: list[Bottleneck] = field(default_factory=list)
    per_primal: list[dict] = field(default_factory=list)
    limiting_constraint: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["bottlenecks"] = [b.to_dict() for b in self.bottlenecks]
        return d


def _severity(utilization: float) -> str:
    for threshold, label in SEVERITY_BANDS:
        if utilization >= threshold:
            return label
    return "ok"


def catalog_impact(demand_multiplier: float,
                   as_of: Optional[date] = None,
                   horizon_days: int = 14,
                   lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                   include_risk: bool = True) -> CatalogImpact:
    """
    Scale the whole catalog and report what binds.

    Baselines come from observed sales over the lookback window and are
    DE-SEASONALISED before the multiplier is applied, exactly as
    seasonal_planning.seasonal_plan does.

    That is not optional. The history ends on 31 December, so the trailing
    28 days are the Christmas run-up and already run ~1.3x a normal week.
    Applying a 1.9x Christmas uplift on top of that double-counts the season
    and reports a capacity overrun that does not exist - measured here
    before the correction, cutter utilization came out at 119.1%, which is
    the same wrong number seasonal_planning.py records having hit for the
    same reason. Dividing each day by the multiplier that applied on that
    day gives a normal-week baseline to scale.
    """
    sales = ops_data.load_sales()
    as_of = as_of or max(sales["sale_date"])
    from datetime import timedelta
    start = as_of - timedelta(days=lookback_days - 1)
    weeks = lookback_days / 7
    window = sales[(sales["sale_date"] >= start) & (sales["sale_date"] <= as_of)]

    from simulation.units import units_to_kg
    from simulation.yield_calc import labor_minutes_per_pack

    from simulation.seasonality import week_multiplier

    baseline_kg: dict[str, float] = {}
    for row in window.itertuples(index=False):
        day_multiplier = week_multiplier(row.sale_date) or 1.0
        baseline_kg[row.sku] = baseline_kg.get(row.sku, 0.0) + units_to_kg(
            row.sku, float(row.units_sold)) / day_multiplier

    box_kg = primal_reference_box_weight_kg()
    extra_primal_by_primal: dict[str, float] = {}
    total_product = total_primal = total_waste = total_minutes = 0.0

    for sku, total_kg in baseline_kg.items():
        product = products_by_sku.get(sku)
        yp = yield_profile_by_sku.get(sku)
        if product is None or yp is None:
            continue
        weekly = total_kg / weeks
        extra_kg = weekly * (demand_multiplier - 1)
        extra_primal = extra_kg / yp.yield_pct
        extra_packs = extra_kg / product.avg_pack_weight_kg

        total_product += extra_kg
        total_primal += extra_primal
        total_waste += extra_primal * yp.trim_waste_pct
        total_minutes += extra_packs * labor_minutes_per_pack(yp, product)
        extra_primal_by_primal[yp.source_primal] = (
            extra_primal_by_primal.get(yp.source_primal, 0.0) + extra_primal)

    # --- Shop-wide cutting load ------------------------------------------
    from simulation.inventory_calc import required_cutting_minutes_for_day
    baseline_minutes = sum(
        required_cutting_minutes_for_day(d) / (week_multiplier(d) or 1.0)
        for d in sorted(set(window["sale_date"]))) / weeks
    available = avg_weekly_cutter_minutes(as_of=as_of, lookback_days=lookback_days)
    planned_minutes = baseline_minutes + total_minutes
    utilization = (planned_minutes / available) if available else 0.0

    bottlenecks: list[Bottleneck] = []

    if utilization >= BINDING_THRESHOLD:
        bottlenecks.append(Bottleneck(
            kind="capacity", scope="shop",
            headline=(f"Cutting would run at {utilization:.0%} of available "
                      f"cutter minutes"),
            utilization=round(utilization, 4),
            severity=_severity(utilization),
            detail={
                "baseline_weekly_cutting_minutes": round(baseline_minutes, 1),
                "planned_weekly_cutting_minutes": round(planned_minutes, 1),
                "extra_weekly_cutting_minutes": round(total_minutes, 1),
                "weekly_cutter_minutes_available": round(available, 1),
            },
            remedy=("Add cutter hours or move some runs earlier in the week"
                    if utilization >= 1.0 else
                    "Watch the roster - one absence takes this over the line"),
        ))

    # --- Per-primal ------------------------------------------------------
    per_primal: list[dict] = []
    for primal in sorted(extra_primal_by_primal):
        extra_kg = extra_primal_by_primal[primal]
        row: dict = {
            "source_primal": primal,
            "extra_weekly_primal_kg": round(extra_kg, 2),
            "extra_weekly_boxes": round(extra_kg / box_kg.get(primal, 20.0), 2),
            "paths": [yp.product_sku for yp in yield_profiles_by_primal().get(primal, [])],
        }

        try:
            position = stock_position(primal, as_of=as_of, lookback_days=lookback_days)
            row["boxes_on_hand"] = position.boxes_on_hand
            row["days_of_cover"] = position.days_of_cover
            row["stock_status"] = position.status
        except ValueError:
            row["stock_status"] = "unknown"

        rate, basis = effective_rate_for(primal, end=as_of)
        row["execution_rate"] = round(rate, 4)
        row["execution_basis"] = basis

        if include_risk:
            try:
                risk = stockout_probability(primal, as_of=as_of,
                                            horizon_days=horizon_days,
                                            demand_multiplier=demand_multiplier)
                row["stockout_probability"] = risk.stockout_probability
                row["risk_band"] = risk.risk_band
                row["peak_risk_date"] = (risk.peak_risk_date.isoformat()
                                         if risk.peak_risk_date else None)

                if risk.stockout_probability >= 0.15:
                    bottlenecks.append(Bottleneck(
                        kind="stock", scope=primal,
                        headline=(f"{primal} has a "
                                  f"{risk.stockout_probability:.0%} chance of "
                                  f"running out within {horizon_days} days"),
                        # Express risk on the same 1.0-is-the-limit scale as
                        # capacity so the ranking below compares like with like.
                        utilization=round(min(2.0, 1.0 + risk.stockout_probability), 4),
                        severity=_severity(1.0 + risk.stockout_probability),
                        detail={
                            "stockout_probability": risk.stockout_probability,
                            "expected_demand_kg": risk.expected_demand_kg,
                            "available_kg": risk.available_kg,
                            "headroom_kg": risk.headroom_kg,
                            "peak_risk_date": (risk.peak_risk_date.isoformat()
                                               if risk.peak_risk_date else None),
                        },
                        remedy=f"Bring forward or increase the next {primal} order",
                    ))
            except (ValueError, KeyError):
                row["stockout_probability"] = None

        # Execution reliability is a constraint in its own right: a primal
        # that only ever completes 83% of plan cannot supply a plan sized as
        # though it completes 100%.
        # Execution reliability is graded with the bands execution_rates
        # already defines and tests, not with a second scale invented here.
        # An earlier version mapped any rate under 90% onto "over_limit",
        # which put a chronic, well-understood 78% primal at the top of the
        # list above genuine emergencies.
        if rate < RELIABLE_THRESHOLD:
            severity = "critical" if rate < AT_RISK_THRESHOLD else "tight"
            bottlenecks.append(Bottleneck(
                kind="execution", scope=primal,
                headline=(f"{primal} historically completes only {rate:.0%} of "
                          f"its planned cutting"),
                utilization=round(RELIABLE_THRESHOLD / rate, 4) if rate > 0 else 2.0,
                severity=severity,
                detail={"execution_rate": round(rate, 4), "basis": basis},
                remedy=(f"Plan {primal} at {1 / rate:.2f}x the target, or fix "
                        f"the cause of the shortfall"),
            ))

        per_primal.append(row)

    # Rank by severity class first, then by how tight the constraint is.
    # Sorting on utilization alone compared numbers that are not on one
    # scale - a stockout probability and a capacity ratio are both "1.4" for
    # entirely different reasons.
    severity_rank = {"over_limit": 0, "critical": 1, "tight": 2, "ok": 3}
    kind_rank = {"capacity": 0, "stock": 1, "supply": 2, "execution": 3}
    bottlenecks.sort(key=lambda b: (severity_rank.get(b.severity, 9),
                                    kind_rank.get(b.kind, 9),
                                    -b.utilization))

    return CatalogImpact(
        as_of=as_of,
        demand_multiplier=demand_multiplier,
        horizon_days=horizon_days,
        total_extra_weekly_product_kg=round(total_product, 2),
        total_extra_weekly_primal_kg=round(total_primal, 2),
        total_extra_weekly_boxes=round(
            sum(extra_primal_by_primal[p] / box_kg.get(p, 20.0)
                for p in extra_primal_by_primal), 2),
        total_extra_trim_waste_kg=round(total_waste, 2),
        total_extra_cutting_minutes=round(total_minutes, 1),
        weekly_cutter_minutes_available=round(available, 1),
        cutter_utilization_pct=round(utilization, 4),
        exceeds_cutter_capacity=planned_minutes > available,
        primals_assessed=len(per_primal),
        bottlenecks=bottlenecks,
        per_primal=sorted(per_primal, key=lambda r: -r["extra_weekly_primal_kg"]),
        limiting_constraint=(f"{bottlenecks[0].kind}:{bottlenecks[0].scope}"
                             if bottlenecks else None),
    )


def top_bottlenecks(demand_multiplier: float, limit: int = 3,
                    as_of: Optional[date] = None,
                    horizon_days: int = 14) -> list[dict]:
    """The few constraints worth naming in a narration."""
    impact = catalog_impact(demand_multiplier, as_of=as_of, horizon_days=horizon_days)
    return [b.to_dict() for b in impact.bottlenecks[:limit]]
