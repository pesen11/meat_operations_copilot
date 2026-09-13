"""
Structured recommendation: what the system decided, and on what grounds.

The old `_verdict()` returned three strings and two lists. It was already
deterministic - which was the important part - but it was also
single-threshold: cutter utilization above 90% produced a caution, and that
was effectively the whole decision. Everything else (waste, stockout
exposure, supply, execution reliability) was computed, narrated, and then
not used to decide anything.

This models a decision the way an operations person actually makes one:
several independent factors, each scored and each able to veto, combined
into one call whose arithmetic is visible. The LLM's job is unchanged - it
explains the `Recommendation`, it does not produce one.

Scoring convention, fixed once here so no factor can quietly invert it:

    score 0.0 = as bad as this factor gets
    score 1.0 = no concern at all

so a decision score is "how comfortable is this", and higher is always
better. Any factor may set `is_blocking`, which no weighted average can
outvote - a capacity overrun is not compensated for by a good margin.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from models.provenance import Evidence


class Action(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    ORDER = "order"
    NO_ACTION = "no_action"       # informational answers
    DO_NOT_CHANGE = "do_not_change"  # actively advising against a proposed change


class Verdict(str, Enum):
    PROCEED = "proceed"
    PROCEED_WITH_CAUTION = "proceed_with_caution"
    DO_NOT_PROCEED = "do_not_proceed"
    INFORMATIONAL = "informational"


class FactorName(str, Enum):
    CAPACITY = "capacity"
    INVENTORY = "inventory"
    SUPPLY = "supply"
    ECONOMICS = "economics"
    WASTE = "waste"
    EXECUTION = "execution"


class DecisionFactor(BaseModel):
    """One axis of the decision, scored and explained.

    `detail` carries the raw numbers the score came from so the narration
    can quote them and claim_guard can check them. A factor that cannot be
    assessed (no data) sets `assessed=False` and is excluded from the
    weighted score rather than being scored 1.0 - "we did not look" and
    "we looked and it is fine" must not produce the same number.
    """
    name: FactorName
    score: float = Field(ge=0, le=1, description="1.0 = no concern, 0.0 = worst case")
    weight: float = Field(gt=0, description="Relative importance in the blended score")
    assessed: bool = True
    is_blocking: bool = False
    headline: str = Field(description="One clause an operator can read")
    detail: dict[str, Any] = Field(default_factory=dict)
    evidence: Optional[Evidence] = None

    def to_dict(self) -> dict:
        out = {"name": self.name.value, "score": round(self.score, 4),
               "weight": self.weight, "assessed": self.assessed,
               "is_blocking": self.is_blocking, "headline": self.headline,
               "detail": self.detail}
        if self.evidence is not None:
            out["evidence"] = self.evidence.to_dict()
        return out


class ScenarioOption(BaseModel):
    """One rung on a swept ladder of production increases.

    The point of sweeping rather than answering a single point estimate:
    an operator asking "how much more should I make" is choosing between
    options, and the interesting answer is the boundary - the largest
    increase that is still feasible, or the smallest that covers demand.
    """
    label: str
    demand_multiplier: float
    extra_weekly_product_kg: float
    extra_weekly_primal_kg: float
    extra_weekly_boxes: float
    cutter_utilization_pct: float
    exceeds_cutter_capacity: bool
    extra_weekly_trim_waste_kg: float
    weekly_margin: Optional[float] = None
    delta_weekly_margin: Optional[float] = None
    stockout_probability: Optional[float] = None
    days_of_cover: Optional[float] = None
    feasible: bool = True
    infeasible_reasons: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()


class Recommendation(BaseModel):
    """
    The decision, fully structured. The narration layer receives this and
    explains it; it never produces one and never overrides one.
    """
    action: Action
    verdict: Verdict
    recommended_multiplier: Optional[float] = Field(
        default=None,
        description="The size the system recommends, when the question asked "
                    "for one. None for informational answers and for "
                    "scenarios where the operator supplied the size.",
    )
    recommended_change_pct: Optional[float] = None
    confidence: float = Field(ge=0, le=1)
    decision_score: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Weighted blend of assessed factor scores. None when no "
                    "factor could be assessed.",
    )
    factors: list[DecisionFactor] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    options: list[ScenarioOption] = Field(default_factory=list)
    chosen_option: Optional[str] = None
    requires_approval: bool = False
    limiting_factor: Optional[FactorName] = Field(
        default=None,
        description="The lowest-scoring assessed factor - the thing to fix "
                    "first. An operator wants the bottleneck named, not a "
                    "list of six scores to rank themselves.",
    )
    confidence_basis: Optional[Evidence] = Field(
        default=None,
        description="The weakest link in the evidence chain, which is what "
                    "actually caps confidence.",
    )

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "verdict": self.verdict.value,
            "recommended_multiplier": self.recommended_multiplier,
            "recommended_change_pct": self.recommended_change_pct,
            "confidence": round(self.confidence, 4),
            "decision_score": round(self.decision_score, 4) if self.decision_score is not None else None,
            "factors": [f.to_dict() for f in self.factors],
            "blockers": self.blockers,
            "cautions": self.cautions,
            "reasons": self.reasons,
            "risks": self.risks,
            "options": [o.to_dict() for o in self.options],
            "chosen_option": self.chosen_option,
            "requires_approval": self.requires_approval,
            "limiting_factor": self.limiting_factor.value if self.limiting_factor else None,
            "confidence_basis": self.confidence_basis.to_dict() if self.confidence_basis else None,
        }
