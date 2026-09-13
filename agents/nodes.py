"""
Graph nodes for the ops copilot.

Division of labour, and the reason for it:

- The three data agents call TOOLS DIRECTLY rather than letting Claude pick
  them. Once the Manager has produced a validated Plan, which tools to call
  is fully determined - there is no judgement left for a model to add, only
  latency, cost, and a chance to call the wrong one. An LLM-driven variant
  is still available via agents.llm.run_tool_loop for genuinely open-ended
  questions; it is not on the critical path for a scenario.

- The Recommendation node's VERDICT is also deterministic (threshold rules
  below). Claude writes the prose around it. So "LLM never computes
  numbers" extends here to "LLM never decides the call either" - it explains
  a decision that Python already made, which is what makes the narration
  checkable by evals/number_guard.py.
"""

from __future__ import annotations

from typing import Any, Optional

from agents import llm, planner
from agents.intent import parse_question
from agents.state import OpsState, trace_event
from models.request import EvidenceKind
from simulation.supply_tools import (
    get_all_execution_rates, get_all_supply_projections, get_bottlenecks,
    get_catalog_impact, get_catalog_scenario_ladder, get_execution_rate,
    get_open_purchase_orders, get_product_scenario_ladder, get_stockout_risk,
    get_supplier_reliability, get_supply_projection,
)
from simulation.history_tools import (
    get_catalog_performance, get_catalog_sales_summary, get_history_range,
    get_sales_summary, get_sales_trend,
    get_slow_movers, get_top_movers, get_weekend_uplift,
)
from simulation.inventory_tools import (
    get_all_stock_positions, get_cutter_capacity, get_scenario_impact, get_stock_position,
)
from simulation.history_analysis import (
    DEFAULT_TREND_WINDOW_DAYS, MIN_COMPARISON_DAYS as MIN_TREND_DAYS,
)
from simulation.seasonal_tools import (
    get_last_period_actuals, get_seasonal_calendar, get_seasonal_plan,
)
from simulation.tools import (
    get_pack_economics, get_weekly_projection, get_yield_breakdown, list_products,
)

# --- Verdict thresholds -----------------------------------------------------
# ASSUMPTIONS, not user-confirmed numbers. They are the line between "go
# ahead" and "this needs a human to think about it", so they are stated here
# rather than buried in an if-statement, and they are the first thing to
# correct against real operating experience.
#
# - Cutter utilization above 90% leaves no room for a heavy Saturday or one
#   person off sick.
# - Under 1.5 days of primal cover, a single missed vendor delivery (which
#   history_generator models at 4% per delivery day) becomes a stockout.
# - A scenario that reduces weekly margin is never auto-recommended,
#   however comfortable the capacity picture looks.
MAX_SAFE_CUTTER_UTILIZATION = 0.90
MIN_SAFE_DAYS_OF_COVER = 1.5


def _safe(node: str, fn, *args, **kwargs) -> tuple[Optional[Any], list[str]]:
    """Run a tool, converting failure into a recorded error instead of a dead
    graph - one agent failing should still leave the other two branches'
    findings visible to the operator."""
    try:
        return fn(*args, **kwargs), []
    except Exception as exc:
        return None, [f"{node}: {type(exc).__name__}: {exc}"]


# ---------------------------------------------------------------------------
# Manager / Supervisor
# ---------------------------------------------------------------------------

def manager(state: OpsState) -> dict:
    """Parse the operator's question into a validated Plan."""
    plan = parse_question(state["question"], state.get("as_of"))
    return {
        "plan": plan,
        "trace": [trace_event("manager", f"Interpreted as '{plan['intent']}'",
                              product_sku=plan.get("product_sku"),
                              demand_multiplier=plan.get("demand_multiplier"),
                              resolved_by=plan.get("resolved_by"))],
    }


def supervisor(state: OpsState) -> dict:
    """
    Dispatch the agents this QUESTION needs, and only those.

    Driven by the plan's required_evidence rather than by a fixed branch list
    per intent: an agent runs because the question needs something it can
    supply, not because the graph has a node for it. "How much did we produce
    last Christmas" needs sales history and nothing else, and sweeping stock,
    capacity and supply for it only puts irrelevant numbers in front of the
    narration, where they get printed.

    Agents that are NOT dispatched still emit a trace event saying so. The
    previous design ran every specialist unconditionally, with idle ones
    returning {"skipped": True}, so that the progress stream showed three
    agents reporting in - a rationale about the STREAM, which is satisfiable
    without doing the work. The stream is now strictly more informative: it
    names every agent AND says why one sat out.
    """
    plan = state["plan"]
    kinds = {EvidenceKind(k) for k in plan.get("required_evidence", [])}
    if not kinds:
        kinds = planner.required_evidence(plan["intent"])

    branches = planner.agents_for(kinds)
    skipped = planner.skipped_agents(branches)

    events = [trace_event("supervisor",
                          f"Dispatching to: {', '.join(branches) or 'none'}",
                          branches=branches, skipped=skipped,
                          required_evidence=sorted(k.value for k in kinds))]
    for name in skipped:
        events.append(trace_event(
            f"{name}_agent",
            f"Skipped - this question needs no {name} evidence",
            skipped=True))

    return {"branches": branches, "trace": events}


# ---------------------------------------------------------------------------
# Parallel specialist agents
# ---------------------------------------------------------------------------

def inventory_agent(state: OpsState) -> dict:
    plan = state["plan"]
    if "inventory" not in state.get("branches", []):
        return {"inventory": {"skipped": True}}

    as_of = plan.get("as_of")
    errors: list[str] = []
    out: dict[str, Any] = {}

    positions, err = _safe("inventory", get_all_stock_positions.invoke, {"as_of": as_of})
    errors += err
    if positions is not None:
        out["stock_positions"] = positions
        out["at_risk"] = [p for p in positions
                          if p["status"] in ("stockout", "critical", "below_target")]

    capacity, err = _safe("inventory", get_cutter_capacity.invoke, {"shift_date": as_of})
    errors += err
    if capacity is not None:
        out["cutter_capacity"] = capacity

    if positions:
        # "What do I need to restock soon?" is answered by ordering on days
        # of cover, not by the status label alone: an 'ok' primal with 1.8
        # days left is the next thing to run out.
        out["restock_priority"] = sorted(
            [p for p in positions if p.get("days_of_cover") is not None],
            key=lambda p: p["days_of_cover"])[:5]

    if plan["intent"] == "seasonal_planning":
        period = plan.get("period")
        calendar, err = _safe("inventory", get_seasonal_calendar.invoke, {"as_of": as_of})
        errors += err
        if calendar is not None:
            out["seasonal_calendar"] = calendar

        # "How should we prepare for the upcoming holiday" names no specific
        # season. Rather than answer with a bare calendar, plan for the
        # nearest elevated period the calendar actually shows - a
        # deterministic choice, not a guess about which one they meant.
        if not period and calendar:
            period = calendar[0]["period"]
            out["period_inferred_from_calendar"] = period

        if period:
            seasonal, err = _safe("inventory", get_seasonal_plan.invoke, {
                "period": period, "as_of": as_of,
                "product_sku": plan.get("product_sku"),
            })
            errors += err
            if seasonal is not None:
                out["seasonal_plan"] = seasonal

            # What actually happened last time this period came round. The
            # plan applies a confirmed multiplier; this is observed history,
            # and the two are reported side by side rather than blended.
            actuals, err = _safe("inventory", get_last_period_actuals.invoke,
                                 {"period": period, "as_of": as_of})
            errors += err
            if actuals is not None:
                out["last_period_actuals"] = actuals

    if plan["intent"] == "scenario" and plan.get("product_sku"):
        impact, err = _safe("inventory", get_scenario_impact.invoke, {
            "product_sku": plan["product_sku"],
            "demand_multiplier": plan["demand_multiplier"],
            "as_of": as_of,
        })
        errors += err
        if impact is not None:
            out["scenario_impact"] = impact

    if plan.get("source_primal"):
        pos, err = _safe("inventory", get_stock_position.invoke, {
            "source_primal": plan["source_primal"], "as_of": as_of})
        errors += err
        if pos is not None:
            out["focus_primal"] = pos

    return {
        "inventory": out,
        "errors": errors,
        "trace": [trace_event("inventory_agent",
                              f"Checked {len(out.get('stock_positions', []))} primals; "
                              f"{len(out.get('at_risk', []))} need attention")],
    }


def production_agent(state: OpsState) -> dict:
    plan = state["plan"]
    if "production" not in state.get("branches", []):
        return {"production": {"skipped": True}}

    errors: list[str] = []
    out: dict[str, Any] = {}
    sku = plan.get("product_sku")

    if plan["intent"] == "catalog" or not sku:
        catalog, err = _safe("production", list_products.invoke, {})
        errors += err
        if catalog is not None:
            out["catalog"] = catalog

    if sku:
        for key, tool, args in (
            ("yield_breakdown", get_yield_breakdown, {"product_sku": sku}),
            ("pack_economics", get_pack_economics, {"product_sku": sku}),
            ("baseline_projection", get_weekly_projection, {"product_sku": sku}),
        ):
            value, err = _safe("production", tool.invoke, args)
            errors += err
            if value is not None:
                out[key] = value

        multiplier = plan["demand_multiplier"]
        if plan["intent"] == "seasonal_planning" and plan.get("period"):
            # The uplift is the shop's confirmed seasonal multiplier, not a
            # number the operator or the model supplied.
            from simulation.seasonal_planning import PERIODS
            multiplier = PERIODS[plan["period"]]
        if plan["intent"] in ("scenario", "seasonal_planning") and multiplier != 1.0:
            value, err = _safe("production", get_weekly_projection.invoke, {
                "product_sku": sku, "demand_multiplier": multiplier})
            errors += err
            if value is not None:
                out["scenario_projection"] = value

    # --- Capacity and constraints, which is what makes this an agent -------
    #
    # This node used to call list_products() and stop whenever no SKU was
    # named, so a whole-catalog seasonal question got "Ran yield/economics
    # for the whole catalog" and contributed nothing to the decision - the
    # inventory agent was doing the seasonal arithmetic on its own. Production
    # now supplies the evidence only it can: what the shop's cutting capacity
    # actually is under this plan, and which constraints bind.
    multiplier = plan.get("demand_multiplier", 1.0)
    if plan["intent"] == "seasonal_planning" and plan.get("period"):
        from simulation.seasonal_planning import PERIODS
        multiplier = PERIODS[plan["period"]]

    if plan["intent"] in ("scenario", "seasonal_planning"):
        impact, err = _safe("production", get_catalog_impact.invoke,
                            {"demand_multiplier": multiplier, "as_of": plan.get("as_of")})
        errors += err
        if impact is not None:
            out["capacity_assessment"] = {
                "demand_multiplier": impact["demand_multiplier"],
                "baseline_cutter_utilization_pct": round(
                    impact["cutter_utilization_pct"]
                    - (impact["total_extra_cutting_minutes"]
                       / impact["weekly_cutter_minutes_available"])
                    if impact["weekly_cutter_minutes_available"] else 0.0, 4),
                "scenario_cutter_utilization_pct": impact["cutter_utilization_pct"],
                "extra_weekly_cutting_minutes": impact["total_extra_cutting_minutes"],
                "weekly_cutter_minutes_available": impact["weekly_cutter_minutes_available"],
                "exceeds_cutter_capacity": impact["exceeds_cutter_capacity"],
            }
            out["bottlenecks"] = impact["bottlenecks"][:5]
            out["limiting_constraint"] = impact["limiting_constraint"]
            out["per_primal_impact"] = impact["per_primal"][:8]
            out["catalog_totals"] = {
                "total_extra_weekly_product_kg": impact["total_extra_weekly_product_kg"],
                "total_extra_weekly_primal_kg": impact["total_extra_weekly_primal_kg"],
                "total_extra_weekly_boxes": impact["total_extra_weekly_boxes"],
                "total_extra_trim_waste_kg": impact["total_extra_trim_waste_kg"],
            }

        # The ladder of options. A "how much should we" question is a choice
        # between rungs, and answering it with one number hides the edge.
        ladder_tool = get_product_scenario_ladder if sku else get_catalog_scenario_ladder
        args = {"as_of": plan.get("as_of")}
        if sku:
            args["product_sku"] = sku
        # Evaluate a target whenever one genuinely EXISTS - whether the
        # operator named it or the shop's confirmed seasonal rule supplies
        # it. An earlier version passed a target only when the operator named
        # one, so "how much more for Christmas" swept a generic ladder and
        # never checked the 1.9x the shop actually plans to: the answer
        # recommended +20% and never mentioned that Christmas needs +90%.
        #
        # The operator-supplied flag still matters, but downstream: it
        # controls whether the recommendation SUBSTITUTES a size, not whether
        # the ladder evaluates one.
        if multiplier != 1.0:
            args["target_multiplier"] = multiplier
        ladder, err = _safe("production", ladder_tool.invoke, args)
        errors += err
        if ladder is not None:
            out["scenario_ladder"] = ladder

    headline = (f"Ran yield/economics for {sku}" if sku
                else "Assessed shop-wide capacity and constraints")
    if out.get("limiting_constraint"):
        headline += f"; limiting constraint {out['limiting_constraint']}"

    return {
        "production": out,
        "errors": errors,
        "trace": [trace_event("production_agent", headline,
                              limiting_constraint=out.get("limiting_constraint"))],
    }


def supply_agent(state: OpsState) -> dict:
    """
    Incoming supply and production reliability.

    The evidence nothing else in the graph could provide. Before this node
    existed, every projection implicitly assumed two things that are both
    false: that whatever is on order arrives in full and on time, and that a
    production plan executes at 100%. Measured on this history, vendors fill
    98.9% of ordered quantity and the shop completes 90% of planned cutting -
    83% for its highest-volume primal. A forward answer that ignores both is
    optimistic by roughly a sixth on the cuts that matter most.
    """
    plan = state["plan"]
    if "supply" not in state.get("branches", []):
        return {"supply": {"skipped": True}}

    errors: list[str] = []
    out: dict[str, Any] = {}
    as_of = plan.get("as_of")
    primal = plan.get("source_primal")

    multiplier = plan.get("demand_multiplier", 1.0)
    if plan["intent"] == "seasonal_planning" and plan.get("period"):
        from simulation.seasonal_planning import PERIODS
        multiplier = PERIODS[plan["period"]]

    reliability, err = _safe("supply", get_supplier_reliability.invoke,
                             {"source_primal": primal, "as_of": as_of})
    errors += err
    if reliability is not None:
        out["supplier_reliability"] = reliability

    execution, err = _safe("supply", get_execution_rate.invoke,
                           {"source_primal": primal, "end": as_of})
    errors += err
    if execution is not None:
        out["execution"] = execution

    if primal:
        projection, err = _safe("supply", get_supply_projection.invoke, {
            "source_primal": primal, "as_of": as_of,
            "demand_multiplier": multiplier})
        errors += err
        if projection is not None:
            # The day-by-day ledger is long and the narration cannot use it;
            # the frontend can. Keep the head of it, drop the rest.
            projection["daily"] = projection.get("daily", [])[:14]
            out["supply_projection"] = projection

        risk, err = _safe("supply", get_stockout_risk.invoke, {
            "source_primal": primal, "as_of": as_of,
            "demand_multiplier": multiplier})
        errors += err
        if risk is not None:
            risk.pop("daily", None)
            out["stockout_risk"] = risk

        orders, err = _safe("supply", get_open_purchase_orders.invoke,
                            {"source_primal": primal, "as_of": as_of})
        errors += err
        if orders is not None:
            out["open_orders"] = orders
    else:
        # Project at the multiplier actually being planned for. Leaving this
        # at 1.0 made the supply factor report a comfortable 157% coverage
        # (baseline demand) in the same breath as the inventory factor
        # reported a certain stockout (Christmas demand) - two numbers
        # describing different worlds, printed side by side.
        outlook, err = _safe("supply", get_all_supply_projections.invoke,
                             {"as_of": as_of, "demand_multiplier": multiplier})
        errors += err
        if outlook is not None:
            # Soonest-to-run-out first, so the head of the list is the answer.
            out["supply_outlook"] = [
                {k: v for k, v in row.items() if k != "daily"} for row in outlook[:6]]
            out["primals_at_supply_risk"] = [
                row["source_primal"] for row in outlook
                if row["stockout_expected"]]

            # Catalog-level aggregates, computed over the FULL outlook rather
            # than the truncated head. The decision engine needs a supply and
            # an inventory reading for every question; without these, both
            # factors reported "not assessed" on precisely the whole-catalog
            # questions that touch the most primals - and an unassessed
            # factor is excluded from the score, so a seasonal plan was being
            # decided on capacity, execution and waste alone.
            out["supply_totals"] = {
                "opening_kg": round(sum(r["opening_kg"] for r in outlook), 2),
                "incoming_expected_kg": round(
                    sum(r["incoming_expected_kg"] for r in outlook), 2),
                "incoming_scheduled_kg": round(
                    sum(r["incoming_scheduled_kg"] for r in outlook), 2),
                "forecast_primal_draw_kg": round(
                    sum(r["forecast_primal_draw_kg"] for r in outlook), 2),
                "primals": len(outlook),
                "primals_at_risk": sum(1 for r in outlook if r["stockout_expected"]),
            }

        rates, err = _safe("supply", get_all_execution_rates.invoke, {"end": as_of})
        errors += err
        if rates is not None:
            out["execution_rates"] = rates[:5]

        # The catalog's risk is the WORST primal's, not an average: the shop
        # stops cutting when any one primal it needs is empty, not when the
        # mean primal is empty.
        impact, err = _safe("supply", get_catalog_impact.invoke,
                            {"demand_multiplier": multiplier, "as_of": as_of})
        errors += err
        if impact is not None:
            risky = [r for r in impact["per_primal"]
                     if r.get("stockout_probability") is not None]
            if risky:
                worst = max(risky, key=lambda r: r["stockout_probability"])
                out["worst_stockout_risk"] = {
                    "source_primal": worst["source_primal"],
                    "stockout_probability": worst["stockout_probability"],
                    "risk_band": worst.get("risk_band"),
                    "peak_risk_date": worst.get("peak_risk_date"),
                    "horizon_days": impact["horizon_days"],
                }

    at_risk = len(out.get("primals_at_supply_risk", []))
    rate = (out.get("execution") or {}).get("rate")
    headline = "Checked incoming supply and production reliability"
    if rate is not None:
        headline += f"; {rate:.0%} of planned cutting completes"
    if at_risk:
        headline += f"; {at_risk} primal(s) project a stockout"

    return {
        "supply": out,
        "errors": errors,
        "trace": [trace_event("supply_agent", headline,
                              primals_at_risk=at_risk, execution_rate=rate)],
    }


def historical_agent(state: OpsState) -> dict:
    plan = state["plan"]
    if "historical" not in state.get("branches", []):
        return {"historical": {"skipped": True}}

    errors: list[str] = []
    out: dict[str, Any] = {}
    sku = plan.get("product_sku")
    as_of = plan.get("as_of")

    # The window the operator actually asked about. Every call below is
    # scoped to it. Before this existed the agent passed only `end` and let
    # the 28-day default apply, so "yesterday", "last week" and "last month"
    # all returned the same answer and none of them said which window it was.
    window = plan.get("window") or {}
    start = window.get("start")
    end = window.get("end") or as_of
    weekends_only = bool(window.get("weekends_only"))
    if window:
        out["window"] = window

    rng, err = _safe("historical", get_history_range.invoke, {})
    errors += err
    if rng is not None:
        out["history_range"] = rng

    # The shop's own total for the window. This is NOT gated on `sku`: the
    # per-product `sales_summary` below needs one, so a question naming no
    # product ("what were our sales yesterday") used to produce only
    # rankings - which cuts led, never how much the shop sold.
    totals, err = _safe("historical", get_catalog_sales_summary.invoke,
                        {"start": start, "end": end})
    errors += err
    if totals is not None:
        out["catalog_sales_totals"] = totals

    if sku:
        # Compare like with like: a question about the last 7 days gets a
        # 7-day-vs-previous-7-day trend, not the fixed 28-day default, which
        # narrated a window the operator never asked about. Very short
        # windows keep the default - a one-day trend is noise.
        trend_days = window.get("days") or DEFAULT_TREND_WINDOW_DAYS
        if trend_days < MIN_TREND_DAYS:
            trend_days = DEFAULT_TREND_WINDOW_DAYS
        for key, tool, args in (
            ("sales_summary", get_sales_summary, {"sku": sku, "start": start, "end": end}),
            ("sales_trend", get_sales_trend,
             {"sku": sku, "as_of": end, "window_days": trend_days}),
        ):
            value, err = _safe("historical", tool.invoke, args)
            errors += err
            if value is not None:
                out[key] = value

    movers, err = _safe("historical", get_top_movers.invoke,
                        {"limit": 5, "start": start, "end": end})
    errors += err
    if movers is not None:
        out["top_movers"] = movers

    # Rankings alone cannot answer "what is doing well and what should we
    # order less of" - the biggest seller can still be shrinking. The
    # per-product verdicts come from deterministic analysis, not narration.
    perf, err = _safe("historical", get_catalog_performance.invoke,
                      {"start": start, "end": end, "weekends_only": weekends_only})
    errors += err
    if perf is not None:
        out["catalog_performance"] = perf
        out["focus_products"] = [r for r in perf if r["verdict"] == "focus"][:5]
        out["reduce_products"] = [r for r in perf if r["verdict"] == "reduce"][:5]

    slow, err = _safe("historical", get_slow_movers.invoke,
                      {"limit": 5, "start": start, "end": end})
    errors += err
    if slow is not None:
        out["slow_movers"] = slow

    if weekends_only or plan["intent"] == "seasonal_planning":
        uplift, err = _safe("historical", get_weekend_uplift.invoke,
                            {"sku": sku, "end": end})
        errors += err
        if uplift is not None:
            out["weekend_uplift"] = uplift

    return {
        "historical": out,
        "errors": errors,
        "trace": [trace_event("historical_agent",
                              f"Summarised sales history for {sku or 'the catalog'}")],
    }


# ---------------------------------------------------------------------------
# Yield analyzer (join) - pure assembly, no new arithmetic
# ---------------------------------------------------------------------------

def yield_analyzer(state: OpsState) -> dict:
    """
    Join the three branches into one 'Projected Outcome' record.

    Every value here is copied verbatim from a tool result. The one derived
    figure - margin delta - is a subtraction of two numbers the deterministic
    cost_calc produced, kept here so the frontend card and the narration read
    from the same field rather than each subtracting for themselves.
    """
    plan = state["plan"]
    inventory = state.get("inventory") or {}
    production = state.get("production") or {}
    historical = state.get("historical") or {}
    supply = state.get("supply") or {}
    supplementary = state.get("supplementary") or {}

    outcome: dict[str, Any] = {
        "intent": plan["intent"],
        "product_sku": plan.get("product_sku"),
        "source_primal": plan.get("source_primal"),
        "demand_multiplier": plan.get("demand_multiplier", 1.0),
    }

    baseline = production.get("baseline_projection")
    scenario = production.get("scenario_projection")
    if baseline and scenario:
        outcome["margin"] = {
            "baseline_weekly_margin": baseline["weekly_margin"],
            "scenario_weekly_margin": scenario["weekly_margin"],
            "delta_weekly_margin": round(
                scenario["weekly_margin"] - baseline["weekly_margin"], 2),
            "baseline_weekly_revenue": baseline["weekly_revenue"],
            "scenario_weekly_revenue": scenario["weekly_revenue"],
        }
    elif baseline:
        outcome["margin"] = {"baseline_weekly_margin": baseline["weekly_margin"],
                             "baseline_weekly_revenue": baseline["weekly_revenue"]}

    impact = inventory.get("scenario_impact")
    if impact:
        outcome["inventory"] = {
            "extra_weekly_primal_kg": impact["extra_weekly_primal_kg"],
            "extra_weekly_boxes": impact["extra_weekly_boxes"],
            "boxes_on_hand": impact["boxes_on_hand"],
            "baseline_days_of_cover": impact["baseline_days_of_cover"],
            "scenario_days_of_cover": impact["scenario_days_of_cover"],
        }
        outcome["labor"] = {
            "extra_weekly_cutting_minutes": impact["extra_weekly_cutting_minutes"],
            "weekly_cutter_minutes_available": impact["weekly_cutter_minutes_available"],
            "baseline_cutter_utilization_pct": impact["baseline_cutter_utilization_pct"],
            "scenario_cutter_utilization_pct": impact["scenario_cutter_utilization_pct"],
            "exceeds_cutter_capacity": impact["exceeds_cutter_capacity"],
        }
        outcome["waste"] = {
            "baseline_weekly_trim_waste_kg": impact["baseline_weekly_trim_waste_kg"],
            "scenario_weekly_trim_waste_kg": impact["scenario_weekly_trim_waste_kg"],
            "extra_weekly_trim_waste_kg": impact["extra_weekly_trim_waste_kg"],
        }

    seasonal = inventory.get("seasonal_plan")
    if seasonal:
        outcome["seasonal"] = {
            "period": seasonal["period"],
            "multiplier": seasonal["multiplier"],
            "multiplier_source": seasonal["multiplier_source"],
            "window_start": seasonal["window_start"],
            "window_end": seasonal["window_end"],
            "days_until": seasonal["days_until"],
            "total_extra_weekly_kg": seasonal["total_extra_weekly_kg"],
            "total_extra_primal_kg": seasonal["total_extra_primal_kg"],
            "total_extra_boxes": seasonal["total_extra_boxes"],
            "total_extra_trim_waste_kg": seasonal["total_extra_trim_waste_kg"],
            # Only the primals that actually need ordering, biggest first -
            # the full 17-row list is noise in a narration.
            "top_primals_to_order": sorted(
                [row for row in seasonal["primals"] if row["boxes_to_order"] > 0],
                key=lambda row: -row["boxes_to_order"])[:5],
        }
        outcome["labor"] = {
            "extra_weekly_cutting_minutes": round(
                seasonal["planned_weekly_cutting_minutes"]
                - seasonal["baseline_weekly_cutting_minutes"], 1),
            "weekly_cutter_minutes_available": seasonal["weekly_cutter_minutes_available"],
            "baseline_cutter_utilization_pct": seasonal["baseline_cutter_utilization_pct"],
            "scenario_cutter_utilization_pct": seasonal["planned_cutter_utilization_pct"],
            "exceeds_cutter_capacity": seasonal["exceeds_cutter_capacity"],
        }
        outcome["waste"] = {
            "baseline_weekly_trim_waste_kg": seasonal["total_baseline_trim_waste_kg"],
            "scenario_weekly_trim_waste_kg": seasonal["total_planned_trim_waste_kg"],
            "extra_weekly_trim_waste_kg": seasonal["total_extra_trim_waste_kg"],
        }
        outcome["inventory"] = {
            "extra_weekly_primal_kg": seasonal["total_extra_primal_kg"],
            "extra_weekly_boxes": seasonal["total_extra_boxes"],
            "boxes_on_hand": round(
                sum(row["boxes_on_hand"] for row in seasonal["primals"]), 2),
            # A whole-catalog seasonal plan spans every primal, so a single
            # days-of-cover figure would be meaningless; per-primal ordering
            # lives in outcome["seasonal"]["top_primals_to_order"].
            "baseline_days_of_cover": None,
            "scenario_days_of_cover": None,
        }
    if inventory.get("seasonal_calendar"):
        outcome["seasonal_calendar"] = inventory["seasonal_calendar"]

    if inventory.get("at_risk"):
        outcome["primals_at_risk"] = inventory["at_risk"]
    if inventory.get("cutter_capacity"):
        outcome["today_cutter_capacity"] = inventory["cutter_capacity"]
    if inventory.get("restock_priority"):
        outcome["restock_priority"] = inventory["restock_priority"]
    # The stock position for the primal the operator actually named. It was
    # being fetched and then dropped here, which is why "how many boxes of
    # ribeye do we have" could not be answered.
    if inventory.get("focus_primal"):
        outcome["focus_primal"] = inventory["focus_primal"]
    if inventory.get("stock_positions") and plan["intent"] == "inventory_status":
        outcome["stock_positions"] = inventory["stock_positions"]
    if inventory.get("last_period_actuals"):
        outcome["last_period_actuals"] = inventory["last_period_actuals"]
    if inventory.get("period_inferred_from_calendar"):
        outcome["period_inferred_from_calendar"] = inventory["period_inferred_from_calendar"]

    # Same story on the history side: the summary was computed every time and
    # never reached the narration, so history questions were answered with a
    # single top-seller line.
    if historical.get("window"):
        outcome["window"] = historical["window"]
    if historical.get("sales_summary"):
        outcome["sales_summary"] = historical["sales_summary"]
    if historical.get("catalog_sales_totals"):
        outcome["catalog_sales_totals"] = historical["catalog_sales_totals"]
    if historical.get("sales_trend"):
        outcome["sales_trend"] = historical["sales_trend"]
    if historical.get("top_movers"):
        outcome["top_movers"] = historical["top_movers"]
    if historical.get("slow_movers"):
        outcome["slow_movers"] = historical["slow_movers"]
    if historical.get("catalog_performance"):
        outcome["catalog_performance"] = historical["catalog_performance"]
    if historical.get("focus_products"):
        outcome["focus_products"] = historical["focus_products"]
    if historical.get("reduce_products"):
        outcome["reduce_products"] = historical["reduce_products"]
    if historical.get("weekend_uplift"):
        outcome["weekend_uplift"] = historical["weekend_uplift"]
    if production.get("catalog"):
        outcome["catalog"] = production["catalog"]
    if production.get("yield_breakdown"):
        outcome["yield_breakdown"] = production["yield_breakdown"]
    if production.get("pack_economics"):
        outcome["pack_economics"] = production["pack_economics"]

    # --- Production: capacity, constraints and the option ladder ----------
    # Capacity from the production agent wins over the single-primal figure
    # the inventory agent derives, because it is measured shop-wide across
    # every product rather than for one scenario's primal. Falling back
    # rather than overwriting keeps single-product scenarios working when
    # the production branch was not dispatched.
    if production.get("capacity_assessment"):
        outcome.setdefault("labor", {})
        outcome["labor"] = {**outcome["labor"], **production["capacity_assessment"]}
    for key in ("bottlenecks", "limiting_constraint", "per_primal_impact",
                "scenario_ladder", "catalog_totals"):
        if production.get(key):
            outcome[key] = production[key]

    # --- Supply and execution ---------------------------------------------
    # Every one of these was computed and then had nowhere to go until this
    # block existed. The project has shipped that exact bug twice
    # (sales_summary, focus_primal): if you add a tool call to an agent, add
    # its line here in the same commit or the narration cannot see it.
    for key in ("supply_projection", "open_orders", "supplier_reliability",
                "stockout_risk", "supply_outlook", "primals_at_supply_risk",
                "supply_totals", "worst_stockout_risk",
                "execution", "execution_rates"):
        if supply.get(key):
            outcome[key] = supply[key]

    if supply.get("supply_projection"):
        projection = supply["supply_projection"]
        outcome["supply"] = {
            "opening_kg": projection["opening_kg"],
            "incoming_expected_kg": projection["incoming_expected_kg"],
            "incoming_scheduled_kg": projection["incoming_scheduled_kg"],
            "forecast_primal_draw_kg": projection["forecast_primal_draw_kg"],
            "supply_reliability": projection["supply_reliability"],
            "projected_closing_boxes": projection["projected_closing_boxes"],
            "stockout_expected": projection["stockout_expected"],
            "earliest_stockout_date": projection["earliest_stockout_date"],
        }

    # Anything a supplementary pass recovered, merged last so it fills holes
    # rather than overwriting a first-pass answer.
    for key, value in supplementary.items():
        outcome.setdefault(key, value)

    return {
        "projected_outcome": outcome,
        "trace": [trace_event("yield_analyzer", "Assembled projected outcome",
                              sections=sorted(outcome.keys()))],
    }


# ---------------------------------------------------------------------------
# Evidence sufficiency
# ---------------------------------------------------------------------------

def evidence_check(state: OpsState) -> dict:
    """
    Did we actually get the evidence this question needs?

    A runtime guard against a failure mode this project has hit twice and
    both times only noticed from the OUTPUT: `historical_agent` computed
    `sales_summary` on every run and `yield_analyzer` never copied it out;
    `inventory_agent` fetched `focus_primal` and the same thing happened. The
    graph completed, a verdict was computed, and the narration quietly
    answered a narrower question than the one asked. Nothing failed, so
    nothing was reported.

    This compares what the plan said it needed against what the outcome
    actually carries, and says so out loud.
    """
    plan = state["plan"]
    outcome = state.get("projected_outcome") or {}
    kinds = {EvidenceKind(k) for k in plan.get("required_evidence", [])}

    gaps = planner.evidence_gaps(kinds, outcome)
    critical = planner.critical_gaps(gaps)
    attempts = state.get("gather_attempts", 0)
    cover = planner.coverage(kinds, outcome)

    if gaps:
        message = (f"Evidence {cover:.0%} complete; missing "
                   f"{', '.join(k.value for k in gaps)}")
    else:
        message = "All required evidence present"

    return {
        "evidence_gaps": [k.value for k in gaps],
        "evidence_coverage": cover,
        "trace": [trace_event("evidence_check", message,
                              coverage=cover,
                              gaps=[k.value for k in gaps],
                              critical_gaps=[k.value for k in critical],
                              will_retry=planner.should_gather_more(gaps, attempts))],
    }


def gather_supplementary(state: OpsState) -> dict:
    """
    Fetch the specific evidence the first pass missed.

    Deliberately narrow: it fetches the missing KINDS, not the whole agent's
    workload again. Re-running a whole branch would re-do work that already
    succeeded and would make the second pass slower than the first for no
    additional evidence.
    """
    plan = state["plan"]
    gaps = {EvidenceKind(k) for k in state.get("evidence_gaps", [])}
    attempts = state.get("gather_attempts", 0)
    as_of = plan.get("as_of")
    primal = plan.get("source_primal")

    errors: list[str] = []
    recovered: dict[str, Any] = {}

    if EvidenceKind.EXECUTION_RATE in gaps:
        value, err = _safe("supplementary", get_execution_rate.invoke,
                           {"source_primal": primal, "end": as_of})
        errors += err
        if value is not None:
            recovered["execution"] = value

    if EvidenceKind.INCOMING_SUPPLY in gaps:
        if primal:
            value, err = _safe("supplementary", get_supply_projection.invoke,
                               {"source_primal": primal, "as_of": as_of})
            errors += err
            if value is not None:
                value["daily"] = value.get("daily", [])[:14]
                recovered["supply_projection"] = value
        else:
            value, err = _safe("supplementary", get_all_supply_projections.invoke,
                               {"as_of": as_of,
                                "demand_multiplier": plan.get("demand_multiplier", 1.0)})
            errors += err
            if value is not None:
                recovered["supply_outlook"] = [
                    {k: v for k, v in row.items() if k != "daily"} for row in value[:6]]

    if EvidenceKind.CUTTER_CAPACITY in gaps:
        value, err = _safe("supplementary", get_cutter_capacity.invoke,
                           {"shift_date": as_of})
        errors += err
        if value is not None:
            recovered["cutter_capacity"] = value

    if EvidenceKind.STOCK_POSITION in gaps:
        value, err = _safe("supplementary", get_all_stock_positions.invoke,
                           {"as_of": as_of})
        errors += err
        if value is not None:
            recovered["stock_positions"] = value

    if EvidenceKind.SALES_HISTORY in gaps:
        window = plan.get("window") or {}
        value, err = _safe("supplementary", get_top_movers.invoke,
                           {"limit": 5, "start": window.get("start"),
                            "end": window.get("end") or as_of})
        errors += err
        if value is not None:
            recovered["top_movers"] = value

    return {
        "supplementary": recovered,
        "gather_attempts": attempts + 1,
        "errors": errors,
        "trace": [trace_event("gather_supplementary",
                              f"Recovered {len(recovered)} missing item(s)",
                              recovered=sorted(recovered))],
    }


def route_after_evidence_check(state: OpsState) -> str:
    """Loop back for one supplementary pass, or move on."""
    gaps = {EvidenceKind(k) for k in state.get("evidence_gaps", [])}
    if planner.should_gather_more(gaps, state.get("gather_attempts", 0)):
        return "gather_supplementary"
    return "recommendation"


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

_NARRATION_SYSTEM = """You are the narration layer of a butcher-shop operations copilot.

Deterministic Python has already computed every number and already decided the
verdict. Your job is to explain the result to a shop operator in plain language.

Hard rules:
1. Use ONLY numbers that appear in the JSON you are given. Never compute,
   re-derive, round differently, convert units, or estimate a number. If a
   figure you want is not in the JSON, describe the situation in words instead.
2. Do not contradict or second-guess the verdict field. Explain it.
3. Write for someone standing on a cutting floor: short sentences, concrete
   nouns, no consultant vocabulary and no bullet-point padding.
4. 120 words maximum. Lead with the answer.
5. Mention the biggest constraint or risk even when the verdict is positive.

An automated check compares every number in your output against the source
data and flags any that does not match, so do not paraphrase figures."""


def _verdict(outcome: dict) -> dict:
    """Deterministic go/no-go. Reasons are returned so narration can explain
    the same decision rather than inventing its own."""
    labor = outcome.get("labor") or {}
    inventory = outcome.get("inventory") or {}
    margin = outcome.get("margin") or {}

    blockers: list[str] = []
    cautions: list[str] = []

    if labor.get("exceeds_cutter_capacity"):
        blockers.append("The extra cutting work exceeds available cutter minutes.")
    elif (labor.get("scenario_cutter_utilization_pct") or 0) > MAX_SAFE_CUTTER_UTILIZATION:
        cautions.append(
            f"Cutter utilization would reach "
            f"{labor['scenario_cutter_utilization_pct']:.0%}, above the "
            f"{MAX_SAFE_CUTTER_UTILIZATION:.0%} comfort line.")

    cover = inventory.get("scenario_days_of_cover")
    if cover is not None and cover < 1.0:
        blockers.append(f"Primal cover falls to {cover} days - under one day of cutting.")
    elif cover is not None and cover < MIN_SAFE_DAYS_OF_COVER:
        cautions.append(f"Primal cover falls to {cover} days, leaving no room for a "
                        f"missed vendor delivery.")

    delta = margin.get("delta_weekly_margin")
    if delta is not None and delta < 0:
        blockers.append(f"Weekly margin falls by ${abs(delta):,.2f}.")

    if blockers:
        verdict = "do_not_proceed"
    elif cautions:
        verdict = "proceed_with_caution"
    elif delta is not None and delta > 0:
        verdict = "proceed"
    else:
        verdict = "informational"

    return {"verdict": verdict, "blockers": blockers, "cautions": cautions}


def _offline_narration(outcome: dict, verdict: dict) -> str:
    """Template narration for when no API key is configured. Deliberately
    plain - every number is interpolated from the outcome dict, so it passes
    the same number guard the live narration has to pass."""
    intent = outcome.get("intent")
    if intent == "unsupported":
        return ("This question is outside what the operations simulation covers. "
                "Procedural and food-safety questions are answered from the SOP "
                "knowledge base instead.")

    parts: list[str] = []

    seasonal = outcome.get("seasonal")
    if seasonal:
        window = ""
        if seasonal.get("window_start"):
            window = (f" ({seasonal['window_start']} to {seasonal['window_end']}, "
                      f"{seasonal['days_until']} days away)")
        parts.append(
            f"For {seasonal['period'].replace('_', ' ')}{window}, the shop's "
            f"confirmed uplift is {seasonal['multiplier']}x normal demand.")
        parts.append(
            f"That is {seasonal['total_extra_weekly_kg']} kg more finished product "
            f"per week, needing {seasonal['total_extra_primal_kg']} kg "
            f"({seasonal['total_extra_boxes']} boxes) more primal and producing "
            f"{seasonal['total_extra_trim_waste_kg']} kg more trim waste.")
        if seasonal.get("top_primals_to_order"):
            listed = ", ".join(
                f"{row['source_primal']} {row['boxes_to_order']} boxes"
                for row in seasonal["top_primals_to_order"])
            parts.append(f"Biggest orders to place: {listed}.")

    if outcome.get("seasonal_calendar") and not seasonal:
        upcoming = ", ".join(
            f"{row['period'].replace('_', ' ')} in {row['days_until']} days "
            f"({row['multiplier']}x)"
            for row in outcome["seasonal_calendar"][:3])
        parts.append(f"Coming up: {upcoming}." if upcoming
                     else "Nothing elevated is coming up in the planning horizon.")

    # --- History -----------------------------------------------------------
    # Stated first and in full. These questions ("how was ribeye last week",
    # "what sold best last month", "what should we order less of") used to
    # produce a single top-seller line regardless of what was asked, because
    # the summary never reached this function.
    window = outcome.get("window") or {}
    window_label = window.get("label")

    # The shop's total leads the answer. Rankings say which cuts led; only
    # this says how much was sold, which is what "what were our sales
    # yesterday" actually asks.
    cat_totals = outcome.get("catalog_sales_totals")
    if cat_totals and cat_totals.get("open_days"):
        day_word = "day" if cat_totals["open_days"] == 1 else "days"
        parts.append(
            f"The shop sold {cat_totals['total_kg']} kg over "
            f"{window_label or 'the window'} for "
            f"{cat_totals['total_revenue']:,.2f} dollars, across "
            f"{cat_totals['products_sold']} products and "
            f"{cat_totals['open_days']} trading {day_word}.")
        if cat_totals["open_days"] > 1:
            parts.append(
                f"That averages {cat_totals['avg_kg_per_open_day']} kg and "
                f"{cat_totals['avg_revenue_per_open_day']:,.2f} dollars a trading "
                f"day, with the best day {cat_totals['best_day']} at "
                f"{cat_totals['best_day_kg']} kg "
                f"({cat_totals['best_day_revenue']:,.2f} dollars).")

    summary = outcome.get("sales_summary")
    if summary:
        parts.append(
            f"{summary['product_name']} sold {summary['total_kg']} kg "
            f"({summary['total_revenue']:,.2f} dollars) over "
            f"{window_label or 'the window'}, across {summary['open_days']} "
            f"trading days - an average of {summary['avg_kg_per_open_day']} kg a day.")
        if summary.get("best_day"):
            parts.append(f"Best single day was {summary['best_day']} at "
                         f"{summary['best_day_kg']} kg.")
        if summary.get("avg_kg_weekend"):
            parts.append(f"Weekdays averaged {summary['avg_kg_weekday']} kg, "
                         f"weekends {summary['avg_kg_weekend']} kg.")

    trend = outcome.get("sales_trend")
    if trend and trend.get("change_pct") is not None:
        direction = {"up": "up", "down": "down", "flat": "flat"}.get(
            trend["direction"], trend["direction"])
        # Name both windows explicitly. Saying "against the previous window"
        # while printing the RECENT window's dates described the comparison
        # backwards - the trend window is also not always the window the
        # question asked about, so neither end can be left implied.
        parts.append(
            f"Day rate from {trend['recent_start']} to {trend['recent_end']} was "
            f"{trend['recent_avg_kg_per_day']} kg, against "
            f"{trend['prior_avg_kg_per_day']} kg from {trend['prior_start']} to "
            f"{trend['prior_end']} - {direction} {abs(trend['change_pct']):.1%}.")
        if trend.get("seasonality_explains_change"):
            parts.append("That move is the calendar, not a change in demand - "
                         "the season's confirmed multiplier accounts for it.")

    uplift = outcome.get("weekend_uplift")
    if uplift and uplift.get("implied_multiplier"):
        who = uplift.get("product_name") or "The shop"
        parts.append(
            f"{who} trades {uplift['weekday_avg_kg']} kg on an average weekday "
            f"against {uplift['weekend_avg_kg']} kg on an average weekend day - "
            f"{uplift['implied_multiplier']}x, measured from "
            f"{uplift['start']} to {uplift['end']}. Size weekend production off "
            f"that, not off a flat uplift.")

    # A ranking, not one name. The whole point of "what were our best
    # sellers" is the list; printing only the first entry answered a
    # different question.
    movers = outcome.get("top_movers")
    if movers and not summary:
        listed = "; ".join(
            f"{row['product_name']} {row['total_kg']} kg "
            f"({row['total_revenue']:,.2f} dollars)"
            for row in movers[:5])
        parts.append(f"Best sellers by revenue over {window_label or 'the window'}: "
                     f"{listed}.")
    elif movers and summary:
        top = movers[0]
        parts.append(f"Top seller over the same window was {top['product_name']} "
                     f"at {top['total_revenue']:,.2f} dollars.")

    slow = outcome.get("slow_movers")
    if slow and not summary:
        listed = "; ".join(
            f"{row['product_name']} {row['total_kg']} kg "
            f"({row['total_revenue']:,.2f} dollars)"
            for row in slow[:3])
        parts.append(f"Slowest over the same window: {listed}.")

    focus = outcome.get("focus_products") if not summary else None
    if focus:
        listed = ", ".join(
            f"{row['product_name']} (up {abs(row['change_pct']):.1%})"
            for row in focus[:4])
        parts.append(f"Growing, and worth pushing: {listed}.")

    reduce_rows = outcome.get("reduce_products") if not summary else None
    if reduce_rows:
        listed = ", ".join(
            f"{row['product_name']} (down {abs(row['change_pct']):.1%})"
            for row in reduce_rows[:4])
        parts.append(f"Falling on real demand, so order less: {listed}.")
    elif outcome.get("catalog_performance") and not focus and not summary:
        parts.append("Nothing is moving enough to change an order on - the rest "
                     "is either steady or explained by the season.")

    # --- Inventory ---------------------------------------------------------
    focus_primal = outcome.get("focus_primal")
    if focus_primal:
        parts.append(
            f"{focus_primal['source_primal']}: {focus_primal['boxes_on_hand']} boxes "
            f"on hand ({focus_primal['kg_on_hand']} kg) as of "
            f"{focus_primal['as_of']}, against a target of "
            f"{focus_primal['target_boxes_low']} to {focus_primal['target_boxes_high']} "
            f"boxes. Status {focus_primal['status'].replace('_', ' ')}, "
            f"{focus_primal['days_of_cover']} days of cover at "
            f"{focus_primal['avg_daily_primal_kg']} kg a day.")

    # --- Forward supply: will we run out, and when --------------------------
    # A stock position is a snapshot; the question an operator actually has is
    # whether the cooler survives the next fortnight. Every figure here is
    # copied from the supply projection, never re-derived.
    supply_view = outcome.get("supply_projection")
    if supply_view:
        parts.append(
            f"Looking {supply_view['horizon_days']} days out, "
            f"{supply_view['source_primal']} opens at {supply_view['opening_kg']} kg "
            f"with {supply_view['incoming_expected_kg']} kg expected in from "
            f"{supply_view['incoming_order_count']} order(s), against "
            f"{supply_view['forecast_primal_draw_kg']} kg of forecast draw.")
        if supply_view["stockout_expected"]:
            parts.append(
                f"On that path it runs dry around "
                f"{supply_view['earliest_stockout_date']}, "
                f"{supply_view['days_until_stockout']} days from now. Bring the "
                f"next order forward.")
        else:
            parts.append(
                f"That closes at {supply_view['projected_closing_boxes']} boxes.")

    risk = outcome.get("stockout_risk")
    if risk and risk.get("stockout_probability", 0) > 0.01:
        parts.append(
            f"Chance of running out of {risk['source_primal']} inside "
            f"{risk['horizon_days']} days: {risk['stockout_probability']:.0%}, "
            f"tightest around {risk['peak_risk_date']}.")

    outlook = outcome.get("supply_outlook")
    at_risk = outcome.get("primals_at_supply_risk")
    if outlook and not supply_view:
        if at_risk:
            parts.append(f"Projected to run short: {', '.join(at_risk[:5])}.")
        else:
            head = outlook[0]
            parts.append(
                f"No primal is projected to run out in the next "
                f"{head['horizon_days']} days. Tightest is "
                f"{head['source_primal']} at "
                f"{head['projected_closing_boxes']} boxes.")

    # --- Execution: a plan is not the same as supply ------------------------
    execution = outcome.get("execution")
    if execution and execution.get("rate") is not None:
        # execution_rate() reports the shop-wide scope as the bare word
        # "shop", which reads as a sentence fragment mid-narration.
        scope = execution.get("scope") or "shop"
        subject = "The shop" if scope == "shop" else scope
        parts.append(
            f"{subject} completes {execution['rate']:.0%} of planned "
            f"cutting, measured over {execution['runs']} runs"
            + (f"; the usual cause of shortfall is "
               f"{execution['top_shortfall_reason'].replace('_', ' ')}."
               if execution.get("top_shortfall_reason") else "."))

    restock = outcome.get("restock_priority")
    if restock and outcome.get("intent") == "inventory_status" and not focus_primal:
        listed = "; ".join(
            f"{row['source_primal']} {row['days_of_cover']} days "
            f"({row['boxes_on_hand']} boxes)"
            for row in restock[:5])
        parts.append(f"Restock soonest, by days of cover: {listed}.")

    # --- Last time this season came round -----------------------------------
    actuals = outcome.get("last_period_actuals")
    if actuals:
        parts.append(
            f"Last {actuals['period'].replace('_', ' ')} "
            f"({actuals['window_start']} to {actuals['window_end']}) the shop sold "
            f"{actuals['total_kg']} kg, {actuals['avg_kg_per_day']} kg a day "
            f"against {actuals['normal_avg_kg_per_day']} kg on a normal day.")
        if actuals.get("top_products"):
            listed = ", ".join(f"{row['product_name']} {row['total_kg']} kg"
                               for row in actuals["top_products"][:4])
            parts.append(f"What moved then: {listed}.")

    sku = outcome.get("product_sku")
    mult = outcome.get("demand_multiplier", 1.0)
    if sku and mult != 1.0:
        direction = "increasing" if mult > 1 else "reducing"
        parts.append(f"Scenario: {direction} {sku} to {mult}x current demand.")

    margin = outcome.get("margin") or {}
    if "delta_weekly_margin" in margin:
        parts.append(f"Weekly margin moves from ${margin['baseline_weekly_margin']:,.2f} "
                     f"to ${margin['scenario_weekly_margin']:,.2f} "
                     f"(${margin['delta_weekly_margin']:,.2f}).")

    inv = outcome.get("inventory") or {}
    # Only narrate the single-primal inventory sentence when there IS a single
    # primal. A whole-catalog seasonal plan spans seventeen of them and has no
    # meaningful combined days-of-cover, and the seasonal block above has
    # already given the per-primal ordering - without this guard the sentence
    # rendered as "194.76 more boxes of None ... from None to None".
    if ("extra_weekly_boxes" in inv and outcome.get("source_primal")
            and inv.get("baseline_days_of_cover") is not None
            and inv.get("scenario_days_of_cover") is not None):
        parts.append(f"That needs {inv['extra_weekly_boxes']} more boxes of "
                     f"{outcome['source_primal']} per week; days of cover goes from "
                     f"{inv['baseline_days_of_cover']} to {inv['scenario_days_of_cover']}.")

    labor = outcome.get("labor") or {}
    if "scenario_cutter_utilization_pct" in labor:
        parts.append(f"Cutter utilization goes from "
                     f"{labor['baseline_cutter_utilization_pct']:.1%} to "
                     f"{labor['scenario_cutter_utilization_pct']:.1%}.")

    waste = outcome.get("waste") or {}
    # The seasonal block already reported the waste increase; don't say it twice.
    if "extra_weekly_trim_waste_kg" in waste and not seasonal:
        parts.append(f"Trim waste rises by {waste['extra_weekly_trim_waste_kg']} kg per week.")

    if outcome.get("primals_at_risk"):
        names = ", ".join(p["source_primal"] for p in outcome["primals_at_risk"][:4])
        parts.append(f"Primals needing attention: {names}.")

    # --- The ladder, and what binds ----------------------------------------
    # An operator choosing a production level wants the EDGE, not one number.
    # Only the rungs either side of the recommendation are named: printing all
    # ten is a table, and this is prose.
    ladder = outcome.get("scenario_ladder")
    if ladder and ladder.get("recommended"):
        chosen = ladder["recommended"]
        parts.append(f"Of the levels checked, {ladder['recommendation_reason']}")
        infeasible = [o for o in ladder.get("options", []) if not o["feasible"]]
        if infeasible:
            smallest = min(infeasible, key=lambda o: o["demand_multiplier"])
            parts.append(
                f"{smallest['label']} is the point it stops being achievable: "
                f"{smallest['infeasible_reasons'][0]}.")
        if chosen.get("stockout_probability") is not None:
            parts.append(
                f"At {chosen['label']}, cutting runs at "
                f"{chosen['cutter_utilization_pct']:.0%} and the chance of "
                f"running short is {chosen['stockout_probability']:.0%}.")

    bottleneck_rows = outcome.get("bottlenecks")
    if bottleneck_rows:
        top = bottleneck_rows[0]
        # The headline is a complete sentence and the remedy is another;
        # joining them with a bare space ran the two together as
        # "...within 14 days Bring forward the next order".
        headline = top["headline"].rstrip(". ")
        remedy = (top.get("remedy") or "").rstrip(". ")
        parts.append(f"Biggest constraint: {headline}."
                     + (f" {remedy}." if remedy else ""))

    # The decision's own limiting factor, named in words. An operator wants
    # the bottleneck identified, not six scores to rank themselves.
    limiting = (verdict or {}).get("limiting_factor")
    score = (verdict or {}).get("decision_score")
    if limiting and score is not None:
        parts.append(f"Overall this scores {score:.2f} out of 1.00, held down by "
                     f"{limiting}.")

    # Blockers and cautions are NOT appended here: the API returns them as
    # their own fields and every surface renders them as a list, so repeating
    # them in the prose printed each one twice.
    return " ".join(parts) or "No projections were produced for this question."


def recommendation(state: OpsState) -> dict:
    """
    Score the decision, then narrate it.

    The verdict now comes from simulation/decision_engine.decide(), which
    weighs six factors and lets any one of them veto, rather than from the
    single capacity threshold `_verdict` applied. What has NOT changed is the
    boundary that matters: Python decides, Claude explains. The model is
    handed a finished Recommendation and a projected outcome and writes prose
    around them - it does not choose the verdict, the action, or the size.

    `_verdict` is kept below as the narrow fallback for states the engine
    cannot score (no assessable factors at all), and because the existing
    regression tests pin its behaviour.
    """
    import json

    from simulation.decision_engine import decide

    outcome = state.get("projected_outcome") or {}
    plan = state.get("plan") or {}

    sweep = _sweep_from(outcome)
    decision = decide(
        outcome,
        sweep=sweep,
        operator_supplied_a_size=bool(plan.get("operator_supplied_a_size", True)),
    )
    # "How much should we make for Christmas" IS a request for a size, and
    # the shop's confirmed multiplier is the answer to it. Attribute it to
    # the business rule rather than presenting it as the system's own pick.
    seasonal = (outcome.get("seasonal") or {})
    if seasonal.get("multiplier") and not plan.get("operator_supplied_a_size"):
        decision.recommended_multiplier = float(seasonal["multiplier"])
        decision.recommended_change_pct = round(float(seasonal["multiplier"]) - 1.0, 4)
    rec_dict = decision.to_dict()

    # Evidence coverage caps confidence. A decision built on two of eight
    # required kinds is not as good as one built on all eight, and the
    # confidence figure is where that has to show up - otherwise a partial
    # answer looks exactly as authoritative as a complete one.
    coverage = state.get("evidence_coverage")
    if coverage is not None:
        rec_dict["confidence"] = round(decision.confidence * coverage, 4)
        rec_dict["evidence_coverage"] = coverage
        if state.get("evidence_gaps"):
            rec_dict["evidence_gaps"] = state["evidence_gaps"]

    payload = {
        **outcome,
        "verdict": rec_dict["verdict"],
        "action": rec_dict["action"],
        "decision_score": rec_dict["decision_score"],
        "limiting_factor": rec_dict["limiting_factor"],
        "blockers": rec_dict["blockers"],
        "cautions": rec_dict["cautions"],
        "factors": rec_dict["factors"],
    }

    offline_text = _offline_narration(outcome, rec_dict)
    result = llm.complete(
        _NARRATION_SYSTEM,
        "Explain this result to the shop operator.\n\n"
        + json.dumps(payload, indent=2, default=str),
        effort="low",
        prefill_offline=offline_text,
    )
    narration = result.text.strip() or offline_text

    rec_dict["narration"] = narration
    rec_dict["narrated_by"] = "claude" if result.live else "template"

    return {
        "recommendation": rec_dict,
        "narration": narration,
        "trace": [trace_event("recommendation",
                              f"Verdict: {rec_dict['verdict']}",
                              verdict=rec_dict["verdict"],
                              action=rec_dict["action"],
                              decision_score=rec_dict["decision_score"],
                              limiting_factor=rec_dict["limiting_factor"],
                              confidence=rec_dict["confidence"],
                              blockers=len(rec_dict["blockers"]),
                              cautions=len(rec_dict["cautions"]))],
    }


def _sweep_from(outcome: dict):
    """Rebuild a ScenarioSweep-shaped object from the ladder the production
    agent already computed, so the decision engine does not re-run the sweep."""
    ladder = outcome.get("scenario_ladder")
    if not ladder:
        return None

    from types import SimpleNamespace

    from models.decision import ScenarioOption

    options = [ScenarioOption(**o) for o in ladder.get("options", [])]
    chosen = ladder.get("recommended")
    return SimpleNamespace(
        options=options,
        recommended=ScenarioOption(**chosen) if chosen else None,
        recommendation_reason=ladder.get("recommendation_reason", ""),
    )


# ---------------------------------------------------------------------------
# Human approval
# ---------------------------------------------------------------------------

def human_approval(state: OpsState) -> dict:
    """
    Pause for a human decision when the graph is recommending a real change.

    interrupt() suspends the run and persists it via the checkpointer; the
    API layer resumes it with Command(resume={"approved": bool, ...}).
    Informational answers and do_not_proceed verdicts don't interrupt -
    there is nothing to approve, and asking anyway trains operators to click
    through prompts without reading them.

    IMPORTANT - this node must stay SYNCHRONOUS, and the whole graph must be
    driven by `graph.stream()`, never `graph.astream()`, on Python 3.10.
    interrupt() calls LangGraph's get_config(), which on Python < 3.11 raises
    "Called get_config outside of a runnable context" whenever it is reached
    from inside a running event loop - by design, not by accident. Under
    astream the approval step therefore blows up exactly where it should have
    paused. api/main.py drives the sync stream on a worker thread and bridges
    its events into asyncio instead; see _stream_graph there.
    """
    rec = state.get("recommendation") or {}
    if not rec.get("requires_approval"):
        return {
            "approval": {"status": "not_required", "verdict": rec.get("verdict")},
            "trace": [trace_event("human_approval", "No approval needed")],
        }

    from langgraph.types import interrupt

    decision = interrupt({
        "kind": "approval_request",
        "question": state.get("question"),
        "verdict": rec.get("verdict"),
        "narration": rec.get("narration"),
        "blockers": rec.get("blockers", []),
        "cautions": rec.get("cautions", []),
        "projected_outcome": state.get("projected_outcome"),
    })

    if isinstance(decision, bool):
        decision = {"approved": decision}
    decision = decision or {}
    approved = bool(decision.get("approved"))

    return {
        "approval": {
            "status": "approved" if approved else "rejected",
            "approved_by": decision.get("approved_by"),
            "note": decision.get("note"),
            "verdict": rec.get("verdict"),
        },
        "trace": [trace_event("human_approval",
                              "Approved by operator" if approved else "Rejected by operator",
                              approved=approved)],
    }
