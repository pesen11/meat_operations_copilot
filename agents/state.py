"""
Shared state for the LangGraph ops graph.

The four data-gathering agents (inventory, production, historical, supply)
run in PARALLEL, so any key they all write needs a reducer or LangGraph
raises a concurrent-update error. `trace` and `errors` therefore accumulate
via operator.add; every other key is written by exactly one node.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class Plan(TypedDict, total=False):
    """What the Manager decided the question is actually asking for.

    Still the contract every node reads, but it is now DERIVED from a
    validated models.request.UserRequest rather than assembled by hand - see
    UserRequest.to_plan(). The fields below the divider are what that
    widening added; everything above is unchanged, so no existing consumer
    had to be touched.
    """
    intent: str                      # scenario | seasonal_planning | inventory_status
                                     # | history | catalog | supply | unsupported
    product_sku: Optional[str]
    source_primal: Optional[str]
    demand_multiplier: float
    period: Optional[str]          # seasonal_planning only: christmas | long_weekend | summer | regular
    as_of: Optional[str]
    # Resolved time window for history questions: {"start","end","label",
    # "days","explicit","weekends_only"}. Without this every history answer
    # silently used the default 28 days regardless of what was asked.
    window: Optional[dict]
    reasoning: str
    resolved_by: str                 # "llm" | "keyword" - provenance of the parse

    # --- added with the request/provenance split --------------------------
    # True when the OPERATOR named a size ("15% more"), False when they asked
    # the system to supply one ("how much more should we?"). Collapsing both
    # into demand_multiplier=1.0 made the two indistinguishable, which is why
    # seasonal planning once had to be special-cased rather than falling out
    # of the model.
    operator_supplied_a_size: bool
    # Evidence kinds this question needs, as string values. The supervisor
    # dispatches agents from this rather than from a fixed branch list.
    required_evidence: list[str]
    # Per-field provenance: {"period": {"value": "christmas", "source": "user",
    # "basis": "..."}}. Lets a narration say WHERE a figure came from, and
    # lets evals/claim_guard.py check that it did.
    provenance: dict[str, Any]


class OpsState(TypedDict, total=False):
    # --- input ---
    question: str
    as_of: Optional[str]

    # --- manager / supervisor ---
    plan: Plan
    branches: list[str]

    # --- parallel agents (each writes exactly one key) ---
    inventory: dict[str, Any]
    production: dict[str, Any]
    historical: dict[str, Any]
    supply: dict[str, Any]

    # --- join ---
    projected_outcome: dict[str, Any]

    # --- evidence sufficiency ---
    # Required kinds the assembled outcome did not carry, and how many
    # supplementary passes have run. Capped by planner.MAX_GATHER_ATTEMPTS:
    # a gap that survives one retry means the data does not exist, not that
    # the fetch was unlucky, and a graph that can loop is a graph that can hang.
    evidence_gaps: list[str]
    evidence_coverage: float
    gather_attempts: int
    supplementary: dict[str, Any]

    # --- decision ---
    recommendation: dict[str, Any]
    narration: str

    # --- human approval ---
    approval: dict[str, Any]

    # --- accumulated across parallel branches ---
    trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


def trace_event(node: str, message: str, **data) -> dict:
    """One progress event. The FastAPI layer streams these over SSE, so keep
    them JSON-serialisable and free of raw objects."""
    return {"node": node, "message": message, "data": data}
