"""
Which evidence does this question need, and did we actually get it?

Two jobs, both deterministic:

1. REQUIREMENTS. Map a validated request onto the set of EvidenceKinds an
   answer needs, then onto the agents that can supply them. This is what
   makes agent dispatch a function of the QUESTION rather than of the graph's
   shape - "how much did we produce last Christmas" needs sales history and
   nothing else, and running an inventory sweep, a capacity calculation and a
   supply projection for it is wasted work whose main effect is to put
   irrelevant numbers in front of the narration, where they get printed.

2. SUFFICIENCY. After the agents run, check that every required kind was
   actually PRODUCED. This is a runtime guard against a failure this project
   has already had twice: `historical_agent` computed `sales_summary` on
   every run and `yield_analyzer` never copied it out; `inventory_agent`
   fetched `focus_primal` and the same thing happened. Both were invisible -
   the graph completed, the verdict was computed, and the narration simply
   answered a narrower question than the one asked. A sufficiency check turns
   that class of bug from silent into loud.

---------------------------------------------------------------------------
WHY SKIPPED AGENTS STILL REPORT
---------------------------------------------------------------------------
The previous design ran all three specialists unconditionally and had the
idle ones return {"skipped": True}, deliberately, so the SSE stream showed
three agents reporting in. That rationale was about the STREAM, not about the
work - and it is satisfiable without doing the work. The supervisor now emits
a trace event for each agent it declines to dispatch, naming why, so the
frontend learns about every agent and the shop's CPU only pays for the ones
with something to contribute. The stream is strictly more informative than
before: it now says why an agent sat out.
"""

from __future__ import annotations

from typing import Iterable

from models.request import EvidenceKind, Intent

# Which agent can supply which kind of evidence. One kind may have several
# suppliers; one agent supplies several kinds.
AGENT_CAPABILITIES: dict[str, set[EvidenceKind]] = {
    "inventory": {
        EvidenceKind.STOCK_POSITION,
        EvidenceKind.CUTTER_CAPACITY,
        EvidenceKind.SEASONAL_RULE,
    },
    "production": {
        EvidenceKind.PRODUCT_ECONOMICS,
        EvidenceKind.YIELD_PROFILE,
        EvidenceKind.CATALOG_LISTING,
        EvidenceKind.CUTTER_CAPACITY,
    },
    "historical": {
        EvidenceKind.SALES_HISTORY,
        EvidenceKind.DEMAND_FORECAST,
    },
    "supply": {
        EvidenceKind.INCOMING_SUPPLY,
        EvidenceKind.EXECUTION_RATE,
        EvidenceKind.DEMAND_FORECAST,
    },
}

# Where each evidence kind lands in the assembled outcome. The sufficiency
# check reads these paths, so adding a tool call to an agent without adding
# its output key here reports the evidence as missing.
#
# These are the keys `yield_analyzer` writes, NOT the keys the agents produce
# internally - the two differ, deliberately (the inventory agent's
# `cutter_capacity` is copied out as `today_cutter_capacity`). Listing an
# agent-internal name here makes the check fire a false positive and sends
# the graph off to re-fetch evidence it already had, which is what happened
# on every seasonal question until `today_cutter_capacity` and `labor` were
# added below.
EVIDENCE_STATE_KEYS: dict[EvidenceKind, tuple[str, ...]] = {
    EvidenceKind.STOCK_POSITION: ("stock_positions", "focus_primal", "restock_priority",
                                  "primals_at_risk", "seasonal"),
    EvidenceKind.CUTTER_CAPACITY: ("today_cutter_capacity", "cutter_capacity",
                                   "capacity_assessment", "labor"),
    EvidenceKind.INCOMING_SUPPLY: ("supply_projection", "open_orders", "supply_outlook",
                                   "supply", "supplier_reliability"),
    EvidenceKind.EXECUTION_RATE: ("execution", "execution_rates"),
    # supplier_reliability counts: it is the measured fill rate that discounts
    # every incoming quantity, so its presence means supply really was assessed.
    EvidenceKind.SALES_HISTORY: ("sales_summary", "top_movers", "catalog_performance",
                                 "sales_trend", "slow_movers"),
    EvidenceKind.DEMAND_FORECAST: ("demand_interval", "stockout_risk", "weekend_uplift",
                                   "scenario_ladder", "supply_outlook"),
    EvidenceKind.PRODUCT_ECONOMICS: ("pack_economics", "baseline_projection", "margin"),
    EvidenceKind.YIELD_PROFILE: ("yield_breakdown", "catalog", "per_primal_impact",
                                 "catalog_totals"),
    EvidenceKind.SEASONAL_RULE: ("seasonal", "seasonal_plan", "seasonal_calendar",
                                 "last_period_actuals"),
    EvidenceKind.CATALOG_LISTING: ("catalog",),
}

# Kinds whose absence is worth a second pass. A missing sales history changes
# the answer; a missing catalog listing on a scenario question does not.
CRITICAL_KINDS = {
    EvidenceKind.STOCK_POSITION,
    EvidenceKind.CUTTER_CAPACITY,
    EvidenceKind.SALES_HISTORY,
    EvidenceKind.INCOMING_SUPPLY,
    EvidenceKind.EXECUTION_RATE,
}

# One supplementary pass, never more. A second retry that failed the same way
# would fail a third time - the gap means the data does not exist, not that
# the fetch was unlucky - and an agent graph that can loop is an agent graph
# that can hang.
MAX_GATHER_ATTEMPTS = 1


def required_evidence(intent: str, product_sku=None, period=None) -> set[EvidenceKind]:
    """The evidence kinds an intent needs. Mirrors UserRequest.required_evidence
    for callers that only have the flattened Plan dict."""
    from models.request import UserRequest  # noqa: PLC0415
    from models.provenance import Evidence

    try:
        parsed = Intent(intent)
    except ValueError:
        return set()

    stub = UserRequest.model_construct(
        intent=parsed,
        intent_evidence=Evidence(value=intent, source="derived", basis="plan"),
        product_sku=None, source_primal=None, period=None,
        explicit_multiplier=None, window=None, as_of=None,
    )
    return stub.required_evidence()


def agents_for(kinds: Iterable[EvidenceKind]) -> list[str]:
    """The smallest set of agents covering the requested kinds.

    Not a minimal set cover - that would be over-engineering for four agents -
    but every agent returned genuinely supplies at least one required kind,
    which is the property that matters.
    """
    wanted = set(kinds)
    chosen = [name for name, supplies in AGENT_CAPABILITIES.items()
              if supplies & wanted]
    # Stable, readable order for the progress stream.
    order = ["inventory", "production", "historical", "supply"]
    return [name for name in order if name in chosen]


def skipped_agents(dispatched: Iterable[str]) -> list[str]:
    order = ["inventory", "production", "historical", "supply"]
    active = set(dispatched)
    return [name for name in order if name not in active]


def evidence_present(kind: EvidenceKind, outcome: dict) -> bool:
    """True when the assembled outcome actually carries this kind."""
    for key in EVIDENCE_STATE_KEYS.get(kind, ()):
        value = outcome.get(key)
        if value not in (None, {}, [], ""):
            return True
    return False


def evidence_gaps(required: Iterable[EvidenceKind], outcome: dict) -> list[EvidenceKind]:
    """Required kinds the outcome does not carry."""
    return [kind for kind in sorted(required, key=lambda k: k.value)
            if not evidence_present(kind, outcome)]


def critical_gaps(gaps: Iterable[EvidenceKind]) -> list[EvidenceKind]:
    return [kind for kind in gaps if kind in CRITICAL_KINDS]


def coverage(required: Iterable[EvidenceKind], outcome: dict) -> float:
    """Fraction of required evidence actually present, for the confidence math."""
    required = list(required)
    if not required:
        return 1.0
    present = sum(1 for kind in required if evidence_present(kind, outcome))
    return round(present / len(required), 4)


def should_gather_more(gaps: Iterable[EvidenceKind], attempts: int) -> bool:
    """Whether a supplementary pass is worth running."""
    return bool(critical_gaps(gaps)) and attempts < MAX_GATHER_ATTEMPTS
