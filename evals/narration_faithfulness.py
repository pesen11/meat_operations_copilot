"""
Narration faithfulness: does the plain-language summary match what the
deterministic tools actually returned?

CLAUDE.md step 8 is explicit that this is a CUSTOM check, not off-the-shelf
Ragas. The reason is that Ragas faithfulness compares an answer against
retrieved text passages; here the "context" is a JSON structure of computed
figures, and the failure modes are different in kind:

  1. A fabricated number          -> caught structurally by number_guard.py.
  2. A number that exists but is
     attached to the wrong thing  -> "trim waste rises 1.92 kg" when 1.92 is
                                     the box count. The number guard passes
                                     it, because 1.92 IS in the tool output.
  3. A reversed direction         -> "margin improves" when delta is negative.
  4. Contradicting the verdict    -> narrating "go ahead" under do_not_proceed.

(2), (3), and (4) are semantic, so they need a judge. (3) and (4) are also
checkable deterministically from the same fields the verdict used, which is
cheaper and never wrong, so they are done in Python first and only the
residue goes to the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from agents import llm
from evals.number_guard import GuardResult, guard_state

_POSITIVE_WORDS = ("improve", "improves", "increase", "increases", "rises", "rise",
                   "grows", "up ", "gain", "better", "higher")
_NEGATIVE_WORDS = ("falls", "fall", "drops", "drop", "decline", "declines",
                   "decrease", "decreases", "shrink", "lower", "worse", "down ")

_APPROVAL_WORDS = ("go ahead", "recommended", "proceed", "should do it",
                   "worth doing", "green light")
_REJECTION_WORDS = ("do not", "don't", "not recommended", "hold off", "avoid",
                    "not worth", "reject")


@dataclass
class FaithfulnessResult:
    passed: bool
    number_guard: GuardResult
    direction_ok: bool = True
    verdict_ok: bool = True
    semantic_score: Optional[float] = None
    issues: list[str] = field(default_factory=list)
    checked_by: str = "deterministic"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "number_guard": self.number_guard.to_dict(),
            "direction_ok": self.direction_ok,
            "verdict_ok": self.verdict_ok,
            "semantic_score": self.semantic_score,
            "issues": self.issues,
            "checked_by": self.checked_by,
        }


def _check_margin_direction(narration: str, outcome: dict) -> tuple[bool, list[str]]:
    """A negative margin delta must not be narrated as an improvement."""
    margin = outcome.get("margin") or {}
    delta = margin.get("delta_weekly_margin")
    if delta is None or abs(delta) < 0.01:
        return True, []

    # Only look at the sentence that mentions margin - "waste rises" in a
    # neighbouring sentence is not a claim about margin.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", narration)
                 if "margin" in s.lower()]
    if not sentences:
        return True, []

    issues = []
    for sentence in sentences:
        lowered = sentence.lower()
        says_up = any(w in lowered for w in _POSITIVE_WORDS)
        says_down = any(w in lowered for w in _NEGATIVE_WORDS)
        if delta < 0 and says_up and not says_down:
            issues.append(f"margin delta is {delta} but narration says it rises: "
                          f"'{sentence.strip()[:120]}'")
        if delta > 0 and says_down and not says_up:
            issues.append(f"margin delta is {delta} but narration says it falls: "
                          f"'{sentence.strip()[:120]}'")
    return not issues, issues


def _check_verdict_agreement(narration: str, verdict: str) -> tuple[bool, list[str]]:
    lowered = narration.lower()
    if verdict == "do_not_proceed":
        endorses = any(w in lowered for w in _APPROVAL_WORDS)
        warns = any(w in lowered for w in _REJECTION_WORDS)
        if endorses and not warns:
            return False, ["verdict is do_not_proceed but the narration endorses it"]
    return True, []


_JUDGE_SYSTEM = """You check whether a plain-language summary correctly describes a JSON result.

You are given a JSON object of figures computed by a deterministic system,
and a summary written about it. Decide whether the summary describes those
figures correctly.

Count these as errors:
- a figure attached to the wrong quantity (calling a box count a waste figure)
- a direction stated backwards (saying something rose when it fell)
- a claim the JSON does not support at all
- a conclusion that contradicts the JSON's own verdict field

Do NOT count these as errors:
- formatting or rounding of a figure (0.6268 shown as "62.7%")
- leaving a figure out (a summary need not mention everything)
- plain, informal wording

Return a score from 0.0 (nothing matches) to 1.0 (everything checks out) and
list any errors you found."""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "errors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "errors"],
    "additionalProperties": False,
}

# Below this the narration is treated as unfaithful. Set where a single
# mis-attributed figure in an otherwise correct summary fails the check.
SEMANTIC_PASS_THRESHOLD = 0.8


def semantic_check(narration: str, outcome: dict,
                   verdict: str) -> tuple[Optional[float], list[str]]:
    if not llm.is_live():
        return None, []
    payload = json.dumps({"verdict": verdict, **outcome}, indent=2, default=str)
    try:
        raw = llm.complete_json(
            _JUDGE_SYSTEM,
            f"JSON result:\n{payload}\n\n---\n\nSummary:\n{narration}",
            _JUDGE_SCHEMA, model=llm.FAST_MODEL, effort="low", max_tokens=2048)
    except Exception:
        return None, []
    score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
    return score, [str(e)[:200] for e in raw.get("errors", [])]


def check_state(state: dict, deep: bool = False) -> FaithfulnessResult:
    narration = (state.get("narration") or "").strip()
    outcome = state.get("projected_outcome") or {}
    rec = state.get("recommendation") or {}
    verdict = rec.get("verdict", "informational")

    guard = guard_state(state)
    issues = [f"ungrounded number: {v.stated} ({v.context})" for v in guard.violations]

    direction_ok, direction_issues = _check_margin_direction(narration, outcome)
    verdict_ok, verdict_issues = _check_verdict_agreement(narration, verdict)
    issues += direction_issues + verdict_issues

    result = FaithfulnessResult(
        passed=guard.passed and direction_ok and verdict_ok,
        number_guard=guard,
        direction_ok=direction_ok,
        verdict_ok=verdict_ok,
        issues=issues,
    )

    if deep and narration and outcome:
        score, errors = semantic_check(narration, outcome, verdict)
        if score is not None:
            result.semantic_score = round(score, 3)
            result.checked_by = "deterministic+llm"
            if errors:
                result.issues.extend(f"judge: {e}" for e in errors)
            if score < SEMANTIC_PASS_THRESHOLD:
                result.passed = False
    return result
