"""
Claim/evidence validator: is every material claim actually supported?

number_guard.py answers "does this figure exist in the source data?" - a
STRUCTURAL check, and the enforcement mechanism for "the LLM never computes".
It is deliberately blind to meaning: 2.41 appears in the outcome, so a
narration saying "2.41" passes, whatever it attaches that 2.41 to.

This module asks the next question: is the claim the figure is embedded in
TRUE, and is it ATTRIBUTED? Three failure classes the number guard
structurally cannot see:

    1. RIGHT NUMBER, WRONG QUANTITY
       "cutter utilization goes to 2.41%" - 2.41 is in the data, as days of
       cover. The guard passes it. It is nonsense.

    2. RIGHT NUMBER, WRONG PROVENANCE
       "we calculate Christmas demand at 1.9x" - 1.9 is in the data, but it
       is the shop's CONFIRMED business rule, not something the system
       worked out. Presenting a policy as a derivation invites an operator
       to argue with the system instead of with the policy, and hides that
       the figure is theirs to change.

    3. CONTRADICTION WITH THE VERDICT
       A narration recommending an increase under a do_not_proceed verdict.

(1) and (3) overlap with narration_faithfulness.py, which already catches a
number attached to the wrong quantity and a reversed direction. This module
does not duplicate that arithmetic-adjacent work; it is specifically about
EVIDENCE - whether a claim traces to something that was actually measured,
and whether the answer says where its figures came from.

Like the number guard, the rule list is CLOSED. Adding "or if it sounds
about right" would make this a vibe check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from models.provenance import NUMERIC_SOURCES_ALLOWED

# Claims that assert a measured fact and therefore need a measurement behind
# them. Matched case-insensitively against the narration.
_MEASUREMENT_CLAIMS = (
    "measured", "historically", "over the last", "on average", "averaged",
    "observed", "recorded",
)

# Phrasing that presents a figure as the system's own derivation.
_DERIVATION_PHRASES = (
    "we calculate", "we estimate", "we predict", "our model",
    "i calculate", "i estimate", "the system calculates",
)

# Phrasing that correctly attributes a figure to the shop rather than to the
# system. A business-rule figure should carry one of these.
_BUSINESS_RULE_ATTRIBUTIONS = (
    "confirmed", "the shop's", "shop's", "business rule", "policy",
    "agreed", "standard",
)

# Verdict -> language that would contradict it.
_CONTRADICTIONS: dict[str, tuple[str, ...]] = {
    "do_not_proceed": ("go ahead", "safe to proceed", "recommend increasing",
                       "you should increase", "no concerns", "no risk"),
    "proceed": ("do not proceed", "not recommended", "cannot be done",
                "not achievable"),
}


@dataclass
class ClaimViolation:
    kind: str          # unattributed_business_rule | unsupported_measurement
                       # | contradicts_verdict | ungrounded_source
    detail: str
    excerpt: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "excerpt": self.excerpt}


@dataclass
class ClaimGuardResult:
    passed: bool
    checked: int = 0
    violations: list[ClaimViolation] = field(default_factory=list)
    attributed_claims: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "claims_checked": self.checked,
            "violations": [v.to_dict() for v in self.violations],
            "attributed_claims": self.attributed_claims,
            "note": self.note,
        }


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _mentions(sentence: str, phrases) -> bool:
    lowered = sentence.lower()
    return any(phrase in lowered for phrase in phrases)


def business_rule_values(provenance: dict) -> set[float]:
    """Numeric values the plan recorded as business rules."""
    out: set[float] = set()
    for entry in (provenance or {}).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("source") != "business_rule":
            continue
        value = entry.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.add(float(value))
    return out


def check_numeric_provenance(provenance: dict) -> list[ClaimViolation]:
    """
    No number in the plan may be attributed to the model.

    This is the provenance layer's half of the project's core rule. The
    number guard enforces "the narration states no number the tools did not
    produce"; this enforces "no number entered the plan as a model's
    invention in the first place". A numeric field tagged llm_inference is a
    structural violation, not a style preference.
    """
    violations: list[ClaimViolation] = []
    for field_name, entry in (provenance or {}).items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        source = entry.get("source")
        if source not in NUMERIC_SOURCES_ALLOWED:
            violations.append(ClaimViolation(
                kind="ungrounded_source",
                detail=(f"{field_name} carries the numeric value {value} with "
                        f"source '{source}', which is not an allowed origin for "
                        f"a number ({sorted(NUMERIC_SOURCES_ALLOWED)})"),
            ))
    return violations


def check(narration: str, outcome: dict, recommendation: Optional[dict] = None,
          provenance: Optional[dict] = None) -> ClaimGuardResult:
    """
    Validate the claims in a narration against the evidence behind it.
    """
    narration = (narration or "").strip()
    if not narration:
        return ClaimGuardResult(passed=True, note="no narration to check")

    recommendation = recommendation or {}
    provenance = provenance or {}
    violations: list[ClaimViolation] = list(check_numeric_provenance(provenance))

    sentences = _sentences(narration)
    attributed = 0

    # --- 1. Business-rule figures must be attributed ----------------------
    # The seasonal multiplier is the shop's confirmed number. Saying "we
    # calculate 1.9x" claims authorship of a figure the shop owns.
    rule_values = business_rule_values(provenance)
    seasonal = (outcome.get("seasonal") or {})
    if seasonal.get("multiplier"):
        rule_values.add(float(seasonal["multiplier"]))

    for sentence in sentences:
        stated = {float(m.replace(",", ""))
                  for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", sentence)}
        if not (stated & rule_values):
            continue
        if _mentions(sentence, _DERIVATION_PHRASES):
            violations.append(ClaimViolation(
                kind="unattributed_business_rule",
                detail=("a confirmed business-rule figure is presented as the "
                        "system's own calculation"),
                excerpt=sentence[:200]))
        elif _mentions(sentence, _BUSINESS_RULE_ATTRIBUTIONS):
            attributed += 1

    # --- 2. Measurement claims need a measurement -------------------------
    # "historically completes 87%" is only sayable if an execution rate was
    # actually measured. Without one it is a guess wearing a fact's clothes.
    measured_evidence = any(
        outcome.get(key) for key in
        ("execution", "execution_rates", "sales_summary", "sales_trend",
         "top_movers", "catalog_performance", "supplier_reliability",
         "last_period_actuals", "weekend_uplift", "stockout_risk",
         "worst_stockout_risk", "supply_projection", "supply_outlook"))
    for sentence in sentences:
        if _mentions(sentence, _MEASUREMENT_CLAIMS) and not measured_evidence:
            violations.append(ClaimViolation(
                kind="unsupported_measurement",
                detail="claims a measured fact, but no measurement is in the outcome",
                excerpt=sentence[:200]))

    # --- 3. The narration may not argue with the verdict ------------------
    verdict = recommendation.get("verdict")
    for phrase in _CONTRADICTIONS.get(verdict, ()):
        if phrase in narration.lower():
            violations.append(ClaimViolation(
                kind="contradicts_verdict",
                detail=f"verdict is '{verdict}' but the narration says '{phrase}'",
                excerpt=phrase))

    return ClaimGuardResult(
        passed=not violations,
        checked=len(sentences),
        violations=violations,
        attributed_claims=attributed,
        note=("" if violations else "all claims trace to evidence"),
    )


def guard_state(state: dict) -> ClaimGuardResult:
    """Guard a finished graph state, alongside number_guard.guard_state."""
    return check(
        state.get("narration") or "",
        state.get("projected_outcome") or {},
        state.get("recommendation") or {},
        (state.get("plan") or {}).get("provenance") or {},
    )
