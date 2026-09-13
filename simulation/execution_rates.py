"""
How reliably the shop executes its own cutting plan.

This is the number that turns a PLAN into a FORECAST. Every projection in
this project used to read production_schedule.planned_primal_kg and treat it
as supply that will exist. It will not: measured across 2025 the shop
completes 90% of planned kg, and one primal (Blade Eyes, the highest-volume
one) completes 83%. Planning 400kg of Blade Eyes and booking 400kg of
finished product against it overstates supply by roughly 70kg every time.

Same rule as the rest of simulation/: plain arithmetic over the history,
unit-tested, no model anywhere near it.

WHY THE RATE IS KEYED BY PRIMAL AND NOT TAKEN SHOP-WIDE
-------------------------------------------------------
The 90% shop-wide figure is an average over primals whose real rates span
83-98%, and the spread is not noise - it is caused. High-velocity primals
run their coolers closer to empty, so they are the ones whose runs get
cancelled for stock. Applying one blended rate to every primal would make
the reliable ones look worse than they are and, far more dangerously, make
the unreliable ones look better.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Optional

from simulation import ops_data

# Window over which a rate is measured when the caller does not say. 90 days
# rather than the 28 used elsewhere: a disruption is a ~7% event, so a
# 28-day window on one primal contains only a couple of them and the rate
# swings on whether a single bad day landed inside it. ASSUMPTION - not a
# user-confirmed number, chosen so the denominator is large enough to be
# stable rather than because the shop thinks in quarters.
DEFAULT_EXECUTION_WINDOW_DAYS = 90

# Below this many scheduled runs, a rate is reported but flagged unreliable
# and callers are expected to fall back to the shop-wide figure.
MIN_RUNS_FOR_RELIABLE_RATE = 20

# Reliability bands. ASSUMPTIONS - the line between "this primal is fine"
# and "this primal needs looking at" is an operating judgement, stated here
# rather than buried in a comparison.
RELIABLE_THRESHOLD = 0.95
AT_RISK_THRESHOLD = 0.85


@dataclass
class ExecutionRate:
    """Planned vs actual for one primal (or the whole shop) over a window."""
    scope: str                     # primal name, or "shop"
    start: date
    end: date
    planned_kg: float
    actual_kg: float
    rate: Optional[float]          # actual / planned; None when nothing was planned
    runs: int
    completed_runs: int
    partial_runs: int
    cancelled_runs: int
    shortfall_kg: float
    shortfall_by_reason: dict[str, float]
    top_shortfall_reason: Optional[str]
    reliability: str               # reliable | watch | at_risk | unknown
    sample_is_reliable: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d


def _latest_production_date() -> date:
    return max(ops_data.load_actual_production()["production_date"])


def _classify(rate: Optional[float]) -> str:
    if rate is None:
        return "unknown"
    if rate >= RELIABLE_THRESHOLD:
        return "reliable"
    if rate >= AT_RISK_THRESHOLD:
        return "watch"
    return "at_risk"


def execution_rate(source_primal: Optional[str] = None,
                   start: Optional[date] = None,
                   end: Optional[date] = None,
                   window_days: int = DEFAULT_EXECUTION_WINDOW_DAYS) -> ExecutionRate:
    """
    Planned vs actual primal kg over a window.

    `source_primal=None` measures the whole shop. The shortfall breakdown by
    reason is the operationally useful half: a primal missing plan because
    the cooler was empty needs an ordering fix, and one missing plan for
    labour needs a rostering fix. Reporting only the rate hides which.
    """
    production = ops_data.load_actual_production()
    end = end or _latest_production_date()
    start = start or (end - timedelta(days=window_days - 1))

    window = production[(production["production_date"] >= start)
                        & (production["production_date"] <= end)]
    scope = "shop"
    if source_primal is not None:
        window = window[window["source_primal"] == source_primal]
        scope = source_primal

    planned = float(window["planned_primal_kg"].sum())
    actual = float(window["actual_primal_kg"].sum())
    rate = (actual / planned) if planned > 0 else None

    by_reason: dict[str, float] = {}
    for row in window.itertuples(index=False):
        reason = row.shortfall_reason
        if not isinstance(reason, str) or not reason:
            continue
        gap = float(row.planned_primal_kg) - float(row.actual_primal_kg)
        if gap > 0:
            by_reason[reason] = by_reason.get(reason, 0.0) + gap

    statuses = window["status"].value_counts().to_dict()
    runs = int(len(window))

    return ExecutionRate(
        scope=scope,
        start=start,
        end=end,
        planned_kg=round(planned, 2),
        actual_kg=round(actual, 2),
        rate=round(rate, 4) if rate is not None else None,
        runs=runs,
        completed_runs=int(statuses.get("completed", 0)),
        partial_runs=int(statuses.get("partial", 0)),
        cancelled_runs=int(statuses.get("cancelled", 0)),
        shortfall_kg=round(max(0.0, planned - actual), 2),
        shortfall_by_reason={k: round(v, 2) for k, v in
                             sorted(by_reason.items(), key=lambda kv: -kv[1])},
        top_shortfall_reason=(max(by_reason, key=by_reason.get) if by_reason else None),
        reliability=_classify(rate),
        sample_is_reliable=runs >= MIN_RUNS_FOR_RELIABLE_RATE,
    )


def all_execution_rates(end: Optional[date] = None,
                        window_days: int = DEFAULT_EXECUTION_WINDOW_DAYS) -> list[ExecutionRate]:
    """Every primal's execution rate, worst first - which is the order an
    operator wants, because the worst one is the thing to fix."""
    production = ops_data.load_actual_production()
    primals = sorted(set(production["source_primal"]))
    rates = [execution_rate(p, end=end, window_days=window_days) for p in primals]
    return sorted(rates, key=lambda r: (r.rate if r.rate is not None else 1.0))


def effective_rate_for(source_primal: str, end: Optional[date] = None,
                       window_days: int = DEFAULT_EXECUTION_WINDOW_DAYS) -> tuple[float, str]:
    """
    The rate to actually plan with, and where it came from.

    Falls back to the shop-wide figure when a primal has too few runs to
    measure, and says so - a rate derived from four cutting runs is not a
    rate, and silently using it would put a number nobody can defend in
    front of an operator.
    """
    own = execution_rate(source_primal, end=end, window_days=window_days)
    if own.rate is not None and own.sample_is_reliable:
        return own.rate, f"{source_primal} measured over {own.runs} runs"

    shop = execution_rate(None, end=end, window_days=window_days)
    if shop.rate is None:
        return 1.0, "no production history; assuming plans execute in full"
    return shop.rate, (f"shop-wide rate ({shop.runs} runs); {source_primal} had "
                       f"only {own.runs} runs, too few to measure")


def expected_actual_kg(planned_kg: float, source_primal: str,
                       end: Optional[date] = None) -> dict:
    """
    Discount a planned quantity by how reliably that primal actually gets cut.

    This is the whole point of the module in one function: 400kg planned at
    a measured 83.4% is a 333.6kg expectation. Returns the basis alongside
    the number so the narration can attribute it rather than asserting it.
    """
    rate, basis = effective_rate_for(source_primal, end=end)
    return {
        "source_primal": source_primal,
        "planned_kg": round(planned_kg, 2),
        "execution_rate": round(rate, 4),
        "expected_actual_kg": round(planned_kg * rate, 2),
        "expected_shortfall_kg": round(planned_kg * (1 - rate), 2),
        "rate_basis": basis,
    }
