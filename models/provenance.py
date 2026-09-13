"""
Where a value came from.

The problem this solves: by the time a number reaches a narration, four
very different things look identical.

    period = "christmas"     the operator typed the word
    period = "christmas"     the calendar inferred it from "what's coming up?"
    multiplier = 1.9         a configured business rule the shop confirmed
    multiplier = 1.87        a figure measured out of the sales history

Those carry completely different authority. A business rule is something
the shop owns and can defend; a measurement is something the data says and
may be contaminated; an inference is the system's own guess. Flattening
them into a bare float means nobody downstream - including the operator
reading the answer - can tell which one they are being handed.

So every value that crosses a node boundary and could be questioned gets
wrapped in `Evidence`, which records its source and, where meaningful, how
it was derived. `evals/claim_guard.py` then checks not only that a narrated
number exists in the state (that is number_guard's job) but that it is
ATTRIBUTED - that the answer says "the shop's confirmed multiplier" for a
business rule rather than implying the system worked it out.

Deliberately NOT a generic container for everything. Wrapping every field
in the codebase would be ceremony; the rule is that a value earns an
Evidence wrapper when an operator could reasonably ask "says who?".
"""

from __future__ import annotations

from datetime import date
from typing import Any, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

ValueSource = Literal[
    "user",             # the operator said it, in the question
    "business_rule",    # a confirmed constant the shop owns (seasonality.py, catalog)
    "historical_data",  # measured from the operational history
    "derived",          # computed deterministically from other evidence
    "system_default",   # a fallback the system chose because nothing was supplied
    "inferred",         # the system picked it without being told (calendar lookup)
    "llm_inference",    # a model's reading of the question - never a number
]

# Ordered weakest -> strongest. Used to decide which source wins when two
# paths supply the same field, and to surface the weakest link in a chain.
SOURCE_AUTHORITY: dict[str, int] = {
    "llm_inference": 0,
    "system_default": 1,
    "inferred": 2,
    "derived": 3,
    "historical_data": 4,
    "business_rule": 5,
    "user": 6,
}

# Sources a NUMBER used in a projection is allowed to come from. The
# project's core rule is that the LLM never computes, so a numeric value
# attributed to llm_inference is a structural violation, not a style
# preference - claim_guard raises on it rather than warning.
NUMERIC_SOURCES_ALLOWED = {"user", "business_rule", "historical_data",
                           "derived", "system_default"}


class Evidence(BaseModel, Generic[T]):
    """One value, plus the answer to 'says who?'.

    `basis` is free text for a human: "confirmed seasonal multiplier",
    "28-day observed mean", "operator stated 15%". It is what the narration
    layer is expected to paraphrase when attributing the figure.
    """
    value: T
    source: ValueSource
    basis: str = Field(description="Short human-readable justification")
    confidence: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Only meaningful for measured or inferred values. A "
                    "business rule has no confidence - it is a policy, not "
                    "an estimate - and leaving it None says so.",
    )
    as_of: Optional[date] = None
    sample_size: Optional[int] = Field(
        default=None, ge=0,
        description="Observations behind a historical_data value. A mean of "
                    "three days and a mean of ninety are not the same claim.",
    )

    @property
    def authority(self) -> int:
        return SOURCE_AUTHORITY[self.source]

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)

    def attribution(self) -> str:
        """One clause a narration can use verbatim."""
        return {
            "user": f"as you specified ({self.basis})",
            "business_rule": f"the shop's confirmed figure ({self.basis})",
            "historical_data": f"measured from history ({self.basis})",
            "derived": f"calculated ({self.basis})",
            "system_default": f"a system default ({self.basis})",
            "inferred": f"inferred by the system ({self.basis})",
            "llm_inference": f"read from your question ({self.basis})",
        }[self.source]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "source": self.source,
                               "basis": self.basis}
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.as_of is not None:
            out["as_of"] = self.as_of.isoformat()
        if self.sample_size is not None:
            out["sample_size"] = self.sample_size
        return out


# --- Constructors -----------------------------------------------------------
# Named helpers rather than raw Evidence(...) calls at every site, so the
# source string cannot be typo'd into something SOURCE_AUTHORITY does not
# know about, and so grepping for "who sets business_rule" is one search.

def from_user(value: T, basis: str) -> Evidence[T]:
    return Evidence(value=value, source="user", basis=basis)


def from_business_rule(value: T, basis: str) -> Evidence[T]:
    return Evidence(value=value, source="business_rule", basis=basis)


def from_history(value: T, basis: str, *, confidence: Optional[float] = None,
                 as_of: Optional[date] = None,
                 sample_size: Optional[int] = None) -> Evidence[T]:
    return Evidence(value=value, source="historical_data", basis=basis,
                    confidence=confidence, as_of=as_of, sample_size=sample_size)


def derived(value: T, basis: str, *, confidence: Optional[float] = None) -> Evidence[T]:
    return Evidence(value=value, source="derived", basis=basis, confidence=confidence)


def system_default(value: T, basis: str) -> Evidence[T]:
    return Evidence(value=value, source="system_default", basis=basis)


def inferred(value: T, basis: str, *, confidence: Optional[float] = None) -> Evidence[T]:
    return Evidence(value=value, source="inferred", basis=basis, confidence=confidence)


def weakest(*items: Optional[Evidence]) -> Optional[Evidence]:
    """The least authoritative evidence in a chain.

    A projection is only as defensible as its flimsiest input, so this is
    what a recommendation should cite when it explains how confident it is.
    """
    present = [e for e in items if e is not None]
    if not present:
        return None
    return min(present, key=lambda e: e.authority)
