"""
Live groundedness guardrail.

CLAUDE.md step 6.4 asks for a faithfulness check as a runtime guardrail, not
just an offline metric. This module is the runtime half; rag/eval/ reuses the
same functions offline so the two cannot disagree about what "grounded" means.

Two checks, cheapest first:

1. **Numeric grounding (deterministic, always on).** Every number in the
   answer must appear in the context passages. This catches the specific
   failure this project cares most about - a temperature, concentration, or
   weight that drifted - and it is exact, free, and impossible to argue with.
   It is the same principle as evals/number_guard.py applied to prose instead
   of tool output.

2. **Claim grounding (LLM, optional).** Splits the answer into claims and
   asks Claude whether each is supported by the passages - a Ragas-style
   faithfulness score. Runs only when credentials exist and the caller asks
   for it, since it costs a second model call per answer.

A failed numeric check is treated as a hard failure: the pipeline downgrades
the answer rather than serving a number that is not in the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from agents import llm

# Numbers to ignore when checking numeric grounding: citation markers like
# [2], and small ordinals from list phrasing ("step 1", "first of the 3").
_NUMBER_RE = re.compile(r"(?<![\[\w])(\d+(?:[.,]\d+)?)\s*%?")
_CITATION_RE = re.compile(r"\[\d+\]")

# Numbers this small are usually step numbers or counts restated from list
# structure, not factual claims. Above this, a mismatch is meaningful.
IGNORE_BELOW = 10.0


@dataclass
class GroundednessResult:
    grounded: bool
    numeric_ok: bool
    ungrounded_numbers: list[str] = field(default_factory=list)
    faithfulness: Optional[float] = None
    unsupported_claims: list[str] = field(default_factory=list)
    checked_by: str = "numeric"

    def to_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "numeric_ok": self.numeric_ok,
            "ungrounded_numbers": self.ungrounded_numbers,
            "faithfulness": self.faithfulness,
            "unsupported_claims": self.unsupported_claims,
            "checked_by": self.checked_by,
        }


def _numbers_in(text: str) -> set[str]:
    cleaned = _CITATION_RE.sub(" ", text)
    out = set()
    for raw in _NUMBER_RE.findall(cleaned):
        out.add(raw.replace(",", "."))
    return out


def check_numeric_grounding(answer: str, context: str) -> tuple[bool, list[str]]:
    """Every non-trivial number in the answer must appear in the context."""
    context_numbers = _numbers_in(context)
    # Also accept the raw context string containing the literal, which covers
    # formats the regex splits differently (e.g. "0-4" vs "0" and "4").
    ungrounded = []
    for number in sorted(_numbers_in(answer)):
        try:
            if float(number) < IGNORE_BELOW:
                continue
        except ValueError:
            continue
        if number in context_numbers or number in context:
            continue
        # Tolerate a trailing ".0" difference ("200" vs "200.0").
        if number.endswith(".0") and number[:-2] in context_numbers:
            continue
        ungrounded.append(number)
    return (not ungrounded), ungrounded


_FAITHFULNESS_SYSTEM = """You check whether an answer is supported by source passages.

Break the answer into its distinct factual claims. For each claim, decide
whether the passages support it. A claim is supported only if the passages
actually state it - not if the claim is merely plausible, consistent with the
passages, or true in general practice.

Ignore claims that are pure framing ("here is the procedure"). Judge only
substantive claims about procedure, quantity, or requirement.

Return the total number of substantive claims, how many are supported, and
the text of any that are not."""

_FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "total_claims": {"type": "integer"},
        "supported_claims": {"type": "integer"},
        "unsupported": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["total_claims", "supported_claims", "unsupported"],
    "additionalProperties": False,
}


def check_faithfulness(answer: str, context: str) -> tuple[Optional[float], list[str]]:
    """Ragas-style faithfulness: supported claims / total claims."""
    if not llm.is_live():
        return None, []
    try:
        raw = llm.complete_json(
            _FAITHFULNESS_SYSTEM,
            f"Passages:\n\n{context}\n\n---\n\nAnswer:\n\n{answer}",
            _FAITHFULNESS_SCHEMA, model=llm.FAST_MODEL, effort="low", max_tokens=2048)
    except Exception:
        return None, []
    total = max(int(raw.get("total_claims", 0)), 0)
    supported = max(int(raw.get("supported_claims", 0)), 0)
    if total == 0:
        return None, []
    return min(1.0, supported / total), [str(c)[:200] for c in raw.get("unsupported", [])]


def check(answer: str, context: str, *, deep: bool = False) -> GroundednessResult:
    numeric_ok, ungrounded = check_numeric_grounding(answer, context)
    result = GroundednessResult(grounded=numeric_ok, numeric_ok=numeric_ok,
                                ungrounded_numbers=ungrounded)
    if deep:
        faithfulness, unsupported = check_faithfulness(answer, context)
        if faithfulness is not None:
            result.faithfulness = round(faithfulness, 3)
            result.unsupported_claims = unsupported
            result.checked_by = "numeric+llm"
            # A partly-unsupported answer is still reported, but flagged.
            result.grounded = numeric_ok and faithfulness >= 0.8
    return result
