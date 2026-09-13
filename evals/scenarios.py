"""
Ground-truth operational scenarios: inject a known fault, check the diagnosis.

Every eval in this project so far has asked whether the ANSWER looked right -
did the narration read well, did its numbers appear in the source, did it
decline when it should. None of them could ask the question an operations
system actually has to pass:

    We broke something specific. Did it notice, and did it name the right
    thing?

That question is only answerable when you control the fault. So each case
here perturbs the data layer in a defined way - a supplier stops delivering,
a cutter calls in sick, demand spikes - runs the full graph against the
perturbed world, and asserts on the DIAGNOSIS: the right intent, the right
agents dispatched, the right bottleneck named, the right call made.

---------------------------------------------------------------------------
WHY PERTURBATION RATHER THAN MORE DATASETS
---------------------------------------------------------------------------
A second year of synthetic history would add rows and no new information -
the system already handles a normal year, and adding more normal makes no
claim testable that is not testable now. A controlled fault is different:
because the fault is known, the correct answer is known, and the system can
be scored rather than admired.

---------------------------------------------------------------------------
WHAT IS SCORED
---------------------------------------------------------------------------
Six dimensions, each independently checkable, because "the answer was good"
is not a measurement:

    intent        did it understand the question
    dispatch      did it run the agents that question needs, and no others
    evidence      did the required evidence actually reach the outcome
    diagnosis     did it name the constraint we broke
    decision      did it reach the right verdict
    grounding     number guard + claim guard both clean

A case can fail `diagnosis` while passing `decision`, which is exactly the
interesting failure: right answer, wrong reason. Averaging the six into one
number would hide it, so they are reported separately.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from simulation import ops_data


# ---------------------------------------------------------------------------
# Controlled perturbation of the data layer
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def perturbed(*, labor_scale: float = 1.0,
              stock_scale: float = 1.0,
              stock_scale_primal: Optional[str] = None,
              drop_open_orders: bool = False,
              delay_open_orders_days: int = 0,
              execution_scale: float = 1.0,
              sales_scale: float = 1.0):
    """
    Temporarily rewrite what the read layer returns.

    Patches simulation.ops_data's loaders rather than the CSVs, so a scenario
    cannot leave the repository dirty if it raises half-way through, and two
    scenarios cannot interfere with each other. The lru_cache is cleared on
    both entry and exit: without the exit clear, a perturbed frame would
    survive into the next case and quietly poison it.
    """
    originals = {
        "load_labor": ops_data.load_labor,
        "load_primal_stock": ops_data.load_primal_stock,
        "load_purchase_orders": ops_data.load_purchase_orders,
        "load_actual_production": ops_data.load_actual_production,
        "load_sales": ops_data.load_sales,
    }

    def _labor():
        df = originals["load_labor"]()
        if labor_scale != 1.0:
            df = df.copy()
            df["available_minutes"] = df["available_minutes"] * labor_scale
            df["headcount"] = (df["headcount"] * labor_scale).round().astype(int)
        return df

    def _stock():
        df = originals["load_primal_stock"]()
        if stock_scale != 1.0:
            df = df.copy()
            mask = (df["source_primal"] == stock_scale_primal
                    if stock_scale_primal else slice(None))
            if stock_scale_primal:
                df.loc[mask, "boxes_on_hand"] = df.loc[mask, "boxes_on_hand"] * stock_scale
            else:
                df["boxes_on_hand"] = df["boxes_on_hand"] * stock_scale
        return df

    def _orders():
        df = originals["load_purchase_orders"]()
        open_mask = df["actual_arrival"].isna()
        if drop_open_orders:
            df = df[~open_mask].copy()
        elif delay_open_orders_days:
            df = df.copy()
            shifted = pd.to_datetime(df.loc[open_mask, "expected_arrival"]) + \
                pd.Timedelta(days=delay_open_orders_days)
            df.loc[open_mask, "expected_arrival"] = [d.date() for d in shifted]
        return df

    def _production():
        df = originals["load_actual_production"]()
        if execution_scale != 1.0:
            df = df.copy()
            df["actual_primal_kg"] = df["actual_primal_kg"] * execution_scale
        return df

    def _sales():
        df = originals["load_sales"]()
        if sales_scale != 1.0:
            df = df.copy()
            df["units_sold"] = df["units_sold"] * sales_scale
            df["revenue"] = df["revenue"] * sales_scale
        return df

    ops_data.clear_cache()
    ops_data.load_labor = _labor
    ops_data.load_primal_stock = _stock
    ops_data.load_purchase_orders = _orders
    ops_data.load_actual_production = _production
    ops_data.load_sales = _sales
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(ops_data, name, fn)
        ops_data.clear_cache()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@dataclass
class ScenarioCase:
    """One controlled fault and the diagnosis it should produce."""
    name: str
    question: str
    fault: str                                   # plain-English description
    perturbation: dict[str, Any] = field(default_factory=dict)

    expect_intent: Optional[str] = None
    expect_agents: Optional[set[str]] = None
    expect_not_agents: set[str] = field(default_factory=set)
    expect_evidence: set[str] = field(default_factory=set)
    expect_bottleneck_kind: Optional[str] = None
    expect_bottleneck_scope: Optional[str] = None
    expect_verdict_in: Optional[set[str]] = None
    expect_limiting_factor: Optional[str] = None
    custom_check: Optional[Callable[[dict], Optional[str]]] = None


def _no_open_orders_means_supply_risk(state: dict) -> Optional[str]:
    outcome = state.get("projected_outcome") or {}
    at_risk = outcome.get("primals_at_supply_risk") or []
    supply = outcome.get("supply_projection") or {}
    if at_risk or supply.get("stockout_expected"):
        return None
    return "cancelling every open order produced no supply risk at all"


def _labor_cut_raises_utilization(state: dict) -> Optional[str]:
    labor = (state.get("projected_outcome") or {}).get("labor") or {}
    utilization = labor.get("scenario_cutter_utilization_pct")
    if utilization is None:
        return "no cutter utilization was reported"
    if utilization <= 0.75:
        return (f"cutting utilization is {utilization:.0%} after a 40% labour "
                f"cut - the reduction did not reach the capacity maths")
    return None


CASES: list[ScenarioCase] = [
    # --- Control: nothing is broken -------------------------------------
    ScenarioCase(
        name="baseline_healthy",
        question="what if we increase chuck roast production by 10%?",
        fault="none - the control case",
        expect_intent="scenario",
        expect_agents={"inventory", "production", "historical", "supply"},
        expect_evidence={"stock_position", "cutter_capacity", "sales_history",
                         "incoming_supply", "execution_rate"},
        expect_verdict_in={"proceed", "proceed_with_caution"},
    ),

    # --- Labour ----------------------------------------------------------
    ScenarioCase(
        name="labour_cut_40pct",
        question="what if we increase chuck roast production by 25%?",
        fault="40% of cutter hours removed",
        perturbation={"labor_scale": 0.6},
        expect_intent="scenario",
        expect_bottleneck_kind="capacity",
        expect_limiting_factor="capacity",
        custom_check=_labor_cut_raises_utilization,
    ),

    # --- Supply ----------------------------------------------------------
    ScenarioCase(
        name="all_deliveries_cancelled",
        question="will we run short of blade eyes?",
        fault="every open purchase order cancelled",
        perturbation={"drop_open_orders": True},
        expect_agents={"inventory", "supply"},
        custom_check=_no_open_orders_means_supply_risk,
    ),
    ScenarioCase(
        name="deliveries_delayed_5_days",
        question="what do we need to restock?",
        fault="every open order pushed back 5 days",
        perturbation={"delay_open_orders_days": 5},
        expect_intent="inventory_status",
        expect_evidence={"stock_position", "incoming_supply"},
    ),

    # --- Stock -----------------------------------------------------------
    ScenarioCase(
        name="cooler_at_quarter_stock",
        question="what do we need to restock?",
        fault="every primal's on-hand boxes cut to 25%",
        perturbation={"stock_scale": 0.25},
        expect_intent="inventory_status",
        expect_evidence={"stock_position"},
    ),

    # --- Execution -------------------------------------------------------
    ScenarioCase(
        name="production_execution_halved",
        question="what if we increase chuck roast production by 15%?",
        fault="actual production halved against plan",
        perturbation={"execution_scale": 0.5},
        expect_intent="scenario",
        expect_limiting_factor="execution",
    ),

    # --- Demand ----------------------------------------------------------
    ScenarioCase(
        name="christmas_planning",
        question="how much more do we need to produce for christmas?",
        fault="none - tests the seasonal business rule under real supply",
        expect_intent="seasonal_planning",
        expect_agents={"inventory", "production", "historical", "supply"},
        expect_evidence={"seasonal_rule", "stock_position", "cutter_capacity"},
    ),

    # --- Dispatch discipline ---------------------------------------------
    ScenarioCase(
        name="history_needs_only_history",
        question="what were our best sellers last month?",
        fault="none - tests that irrelevant agents are NOT dispatched",
        expect_intent="history",
        expect_agents={"historical"},
        expect_not_agents={"inventory", "production", "supply"},
        expect_evidence={"sales_history"},
        expect_verdict_in={"informational"},
    ),
    ScenarioCase(
        name="sop_question_is_declined",
        question="how do I trim a brisket fat cap?",
        fault="none - tests routing to the SOP corpus rather than simulation",
        expect_intent="unsupported",
        expect_verdict_in={"informational"},
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    fault: str
    question: str
    dimensions: dict[str, bool]
    failures: list[str] = field(default_factory=list)
    verdict: Optional[str] = None
    limiting_factor: Optional[str] = None
    bottleneck: Optional[str] = None

    @property
    def passed(self) -> bool:
        return all(self.dimensions.values())

    def to_dict(self) -> dict:
        return {
            "name": self.name, "fault": self.fault, "question": self.question,
            "passed": self.passed, "dimensions": self.dimensions,
            "failures": self.failures, "verdict": self.verdict,
            "limiting_factor": self.limiting_factor, "bottleneck": self.bottleneck,
        }


def run_case(case: ScenarioCase) -> CaseResult:
    """Run one scenario against a freshly-built graph in a perturbed world."""
    from agents.graph import build_graph, run_question
    from evals import claim_guard, number_guard
    from langgraph.checkpoint.memory import InMemorySaver

    failures: list[str] = []
    dimensions: dict[str, bool] = {}

    with perturbed(**case.perturbation):
        graph = build_graph(InMemorySaver())
        state = run_question(case.question, thread_id=f"sc-{case.name}", graph=graph)

        plan = state.get("plan") or {}
        outcome = state.get("projected_outcome") or {}
        recommendation = state.get("recommendation") or {}

        # --- intent ---
        if case.expect_intent is not None:
            ok = plan.get("intent") == case.expect_intent
            dimensions["intent"] = ok
            if not ok:
                failures.append(f"intent: expected {case.expect_intent}, "
                                f"got {plan.get('intent')}")

        # --- dispatch ---
        branches = set(state.get("branches") or [])
        if case.expect_agents is not None or case.expect_not_agents:
            ok = True
            if case.expect_agents is not None and not case.expect_agents <= branches:
                ok = False
                failures.append(f"dispatch: missing {sorted(case.expect_agents - branches)}")
            wrongly_run = case.expect_not_agents & branches
            if wrongly_run:
                ok = False
                failures.append(f"dispatch: ran unnecessary {sorted(wrongly_run)}")
            dimensions["dispatch"] = ok

        # --- evidence ---
        if case.expect_evidence:
            from agents import planner
            from models.request import EvidenceKind
            missing = [k for k in case.expect_evidence
                       if not planner.evidence_present(EvidenceKind(k), outcome)]
            dimensions["evidence"] = not missing
            if missing:
                failures.append(f"evidence: {sorted(missing)} never reached the outcome")

        # --- diagnosis ---
        bottlenecks = outcome.get("bottlenecks") or []
        top = bottlenecks[0] if bottlenecks else None
        if case.expect_bottleneck_kind or case.expect_bottleneck_scope:
            ok = top is not None
            if ok and case.expect_bottleneck_kind:
                ok = any(b["kind"] == case.expect_bottleneck_kind for b in bottlenecks[:3])
            if ok and case.expect_bottleneck_scope:
                ok = any(b["scope"] == case.expect_bottleneck_scope for b in bottlenecks[:3])
            dimensions["diagnosis"] = ok
            if not ok:
                got = [f"{b['kind']}:{b['scope']}" for b in bottlenecks[:3]] or ["none"]
                failures.append(
                    f"diagnosis: expected "
                    f"{case.expect_bottleneck_kind or case.expect_bottleneck_scope}, "
                    f"top constraints were {got}")

        # --- decision ---
        if case.expect_verdict_in or case.expect_limiting_factor:
            ok = True
            if case.expect_verdict_in:
                ok = recommendation.get("verdict") in case.expect_verdict_in
                if not ok:
                    failures.append(f"decision: verdict {recommendation.get('verdict')} "
                                    f"not in {sorted(case.expect_verdict_in)}")
            if case.expect_limiting_factor:
                got = recommendation.get("limiting_factor")
                if got != case.expect_limiting_factor:
                    ok = False
                    failures.append(f"decision: limiting factor {got}, "
                                    f"expected {case.expect_limiting_factor}")
            dimensions["decision"] = ok

        # --- grounding ---
        numbers = number_guard.guard_state(state)
        claims = claim_guard.guard_state(state)
        dimensions["grounding"] = numbers.passed and claims.passed
        if not numbers.passed:
            failures.append(f"grounding: {len(numbers.violations)} ungrounded number(s)")
        if not claims.passed:
            failures.append(f"grounding: {len(claims.violations)} unsupported claim(s)")

        # --- case-specific ---
        if case.custom_check is not None:
            problem = case.custom_check(state)
            dimensions["custom"] = problem is None
            if problem:
                failures.append(f"custom: {problem}")

        return CaseResult(
            name=case.name, fault=case.fault, question=case.question,
            dimensions=dimensions, failures=failures,
            verdict=recommendation.get("verdict"),
            limiting_factor=recommendation.get("limiting_factor"),
            bottleneck=(f"{top['kind']}:{top['scope']}" if top else None),
        )


def run_all(cases: Optional[list[ScenarioCase]] = None) -> dict:
    """Run every scenario and summarise by dimension."""
    cases = cases if cases is not None else CASES
    results = [run_case(c) for c in cases]

    by_dimension: dict[str, dict[str, int]] = {}
    for result in results:
        for dimension, ok in result.dimensions.items():
            bucket = by_dimension.setdefault(dimension, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(ok)

    passed = sum(1 for r in results if r.passed)
    return {
        "cases": [r.to_dict() for r in results],
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "by_dimension": {
            name: {**counts,
                   "rate": round(counts["passed"] / counts["total"], 4)}
            for name, counts in sorted(by_dimension.items())
        },
    }


def main() -> int:
    from agents import llm

    report = run_all()
    mode = "LIVE (Claude)" if llm.is_live() else "OFFLINE (deterministic)"
    print(f"\nGround-truth operational scenarios - mode: {mode}")
    print("=" * 78)
    for case in report["cases"]:
        mark = "PASS" if case["passed"] else "FAIL"
        print(f"  [{mark}] {case['name']:<28} {case['fault']}")
        for failure in case["failures"]:
            print(f"         - {failure}")
    print("-" * 78)
    print(f"  {report['passed']}/{report['total']} cases "
          f"({report['pass_rate']:.0%})")
    for name, counts in report["by_dimension"].items():
        print(f"    {name:<12} {counts['passed']}/{counts['total']} "
              f"({counts['rate']:.0%})")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
