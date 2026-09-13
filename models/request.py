"""
What the operator asked for, as a validated object.

This splits apart three things the old single `Plan` dict conflated:

  A. INTENT       - what kind of answer is wanted
  B. ENTITIES     - which product / primal / period the question names
  C. PARAMETERS   - the sizes and windows the question specifies

They were mixed because keyword_plan() produced all three in one pass, so a
test of "does it recognise a seasonal question" could not be written without
also asserting on window resolution and product matching. Separating them
means each is independently testable, and - more importantly - it makes
room for the distinction the whole provenance layer exists for:

    explicit_multiplier = 1.15      the operator said "15% more"
    explicit_multiplier = None      the operator asked "how much should I?"

Those are different questions. The first supplies a size and asks for its
consequences; the second asks the system to supply the size. Collapsing
both into `demand_multiplier: float = 1.0` made them indistinguishable,
which is why "how much more for Christmas" once had to be special-cased
rather than simply falling out of the model.

`Plan` in agents/state.py is still produced (every existing node reads it)
and is now DERIVED from a UserRequest rather than built by hand - see
`UserRequest.to_plan`. That keeps this a refactor with a widened contract,
not a rewrite of the graph.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from models.provenance import Evidence


class Intent(str, Enum):
    """What kind of answer the operator wants."""
    SCENARIO = "scenario"                    # "what if we cut 15% more X"
    SEASONAL_PLANNING = "seasonal_planning"  # "how much more for Christmas"
    INVENTORY_STATUS = "inventory_status"    # "how many boxes of X do we have"
    HISTORY = "history"                      # "how did we do last week"
    CATALOG = "catalog"                      # "what products do we make"
    SUPPLY = "supply"                        # "what's on order / arriving"
    UNSUPPORTED = "unsupported"              # belongs to the SOP corpus, or nothing


class EvidenceKind(str, Enum):
    """A category of evidence an answer may require.

    The planner maps an Intent onto a set of these, and the graph runs only
    the agents that can supply them. This is the mechanism behind "don't run
    an agent because the architecture says it exists" - the required set is
    derived from the question, and an agent with nothing to contribute is
    never dispatched.
    """
    STOCK_POSITION = "stock_position"
    CUTTER_CAPACITY = "cutter_capacity"
    INCOMING_SUPPLY = "incoming_supply"
    EXECUTION_RATE = "execution_rate"
    SALES_HISTORY = "sales_history"
    DEMAND_FORECAST = "demand_forecast"
    PRODUCT_ECONOMICS = "product_economics"
    YIELD_PROFILE = "yield_profile"
    SEASONAL_RULE = "seasonal_rule"
    CATALOG_LISTING = "catalog_listing"


class UserRequest(BaseModel):
    """
    A validated reading of one operator question.

    Every field that could be questioned carries its provenance. `period`
    resolved from the literal word "christmas" and `period` inferred from
    the calendar because the operator asked "what's coming up?" are both
    "christmas" here - but only one of them is `source="user"`, and the
    narration is expected to say so.
    """

    # --- A. Intent ---
    intent: Intent
    intent_evidence: Evidence[str]

    # --- B. Entities ---
    product_sku: Optional[Evidence[str]] = None
    source_primal: Optional[Evidence[str]] = None
    period: Optional[Evidence[str]] = None

    # --- C. Parameters ---
    explicit_multiplier: Optional[Evidence[float]] = Field(
        default=None,
        description="A size the OPERATOR supplied. None means they did not "
                    "supply one - which for a seasonal question is the "
                    "signal to apply the shop's confirmed multiplier, and "
                    "for a scenario means 'project the baseline'. Never "
                    "defaulted to 1.0: that erases the distinction.",
    )
    window: Optional[dict[str, Any]] = Field(
        default=None, description="Resolved time window (simulation/time_windows.py)",
    )
    as_of: Optional[str] = None

    # --- Provenance of the parse itself ---
    resolved_by: str = Field(default="keyword", description="'llm' | 'keyword'")
    reasoning: str = ""

    @model_validator(mode="after")
    def check_scenario_has_target(self):
        """A scenario question with no product is not answerable as a
        scenario. Catching it here rather than three nodes downstream, where
        it surfaced as an empty projection with no explanation."""
        if self.intent == Intent.SCENARIO and self.product_sku is None:
            raise ValueError(
                "intent=scenario requires a product_sku; route to "
                "unsupported instead of guessing one"
            )
        return self

    # --- Convenience accessors: the value without the wrapper -------------

    @property
    def sku(self) -> Optional[str]:
        return self.product_sku.value if self.product_sku else None

    @property
    def primal(self) -> Optional[str]:
        return self.source_primal.value if self.source_primal else None

    @property
    def period_name(self) -> Optional[str]:
        return self.period.value if self.period else None

    @property
    def multiplier(self) -> Optional[float]:
        return self.explicit_multiplier.value if self.explicit_multiplier else None

    @property
    def operator_supplied_a_size(self) -> bool:
        """True when the operator named a magnitude themselves.

        This is the whole scenario-vs-planning distinction in one property:
        False means the operator is ASKING for the number, so the system
        must supply it from a business rule and say that it did.
        """
        return self.explicit_multiplier is not None

    def required_evidence(self) -> set[EvidenceKind]:
        """Which kinds of evidence this request actually needs.

        Deliberately a function of the REQUEST, not of the graph topology.
        "How much did we produce last Christmas" needs history and nothing
        else; running an inventory sweep and a capacity calculation for it
        is wasted work whose only effect is to put irrelevant numbers in
        front of the narration, where they get printed.
        """
        E = EvidenceKind
        if self.intent == Intent.SCENARIO:
            need = {E.STOCK_POSITION, E.CUTTER_CAPACITY, E.PRODUCT_ECONOMICS,
                    E.YIELD_PROFILE, E.SALES_HISTORY, E.INCOMING_SUPPLY,
                    E.EXECUTION_RATE, E.DEMAND_FORECAST}
        elif self.intent == Intent.SEASONAL_PLANNING:
            need = {E.SEASONAL_RULE, E.STOCK_POSITION, E.CUTTER_CAPACITY,
                    E.SALES_HISTORY, E.YIELD_PROFILE, E.INCOMING_SUPPLY,
                    E.EXECUTION_RATE, E.DEMAND_FORECAST}
        elif self.intent == Intent.INVENTORY_STATUS:
            need = {E.STOCK_POSITION, E.INCOMING_SUPPLY, E.CUTTER_CAPACITY}
        elif self.intent == Intent.SUPPLY:
            need = {E.INCOMING_SUPPLY, E.STOCK_POSITION, E.EXECUTION_RATE}
        elif self.intent == Intent.HISTORY:
            need = {E.SALES_HISTORY}
        elif self.intent == Intent.CATALOG:
            need = {E.CATALOG_LISTING, E.YIELD_PROFILE}
        else:
            need = set()
        return need

    def to_plan(self) -> dict:
        """Flatten to the legacy `Plan` dict the existing nodes read.

        `demand_multiplier` collapses back to 1.0 when the operator supplied
        no size, which is what every downstream consumer already expects.
        The un-collapsed truth stays available on `explicit_multiplier`, so
        a node that needs to know the difference can ask for it.
        """
        return {
            "intent": self.intent.value,
            "product_sku": self.sku,
            "source_primal": self.primal,
            "demand_multiplier": self.multiplier if self.multiplier is not None else 1.0,
            "period": self.period_name,
            "as_of": self.as_of,
            "window": self.window,
            "reasoning": self.reasoning,
            "resolved_by": self.resolved_by,
            "operator_supplied_a_size": self.operator_supplied_a_size,
            "required_evidence": sorted(k.value for k in self.required_evidence()),
            "provenance": {
                name: ev.to_dict()
                for name, ev in (("intent", self.intent_evidence),
                                 ("product_sku", self.product_sku),
                                 ("source_primal", self.source_primal),
                                 ("period", self.period),
                                 ("explicit_multiplier", self.explicit_multiplier))
                if ev is not None
            },
        }
