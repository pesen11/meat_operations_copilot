"""
Structural guard: every number the LLM says must come from tool output.

This is the enforcement mechanism for the project's non-negotiable rule -
the LLM never computes numbers. The narration node is given a JSON blob of
deterministic tool results and asked to explain them; this module checks the
prose it produced against that blob and flags any figure that is not there.

CLAUDE.md step 8 calls for exactly this ("catch any LLM-stated number that
doesn't match tool output verbatim, since that would be a silent violation
of the whole project's principle"). It runs both offline as an eval and
online in the API response, because a violation in production is worse than
a violation in CI.

**"Verbatim" needs care, or the guard is useless.** A narration that renders
0.6268 as "62.7%" or 789.6 as "$789.60" has not invented anything - it has
formatted. So a stated number is accepted when it matches a tool value under
one of a SHORT, EXPLICIT list of presentation transforms (below). Anything
outside that list is a violation. The list is deliberately closed: adding
"or within 5%" to make a failure go away would silently license the model to
do arithmetic, which is the exact thing being prevented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Numbers inside a citation marker are references, not claims.
_CITATION_RE = re.compile(r"\[\d+\]")
# Match numbers with optional thousands separators, decimals, and a % suffix.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Figures this small are almost always structural (list positions, "2 of the
# 3 primals", "one day"), not financial or operational claims. Checking them
# produces noise that trains people to ignore the guard.
IGNORE_BELOW = 2.0

# Dates are not quantities. Full ISO dates are stripped FIRST and whole:
# removing only the year from "2025-12-31" leaves "-12-31", which the number
# regex then reads as the numbers -12 and -31 and checks against the source
# data. That is a tokenisation bug, not a missing tolerance - nothing here
# widens what counts as a matching figure.
_ISO_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
# Bare 4-digit tokens that appear as years rather than quantities. The
# lookarounds are load-bearing: a plain \b(19|20)\d{2}\b also matches the
# leading digits of a real quantity like "2044.71 kg", leaving ".71" behind
# to be checked as the number 71 against data that has no such value. A year
# is a 4-digit token with no digit, comma or decimal touching either end.
_DATE_LIKE_RE = re.compile(r"(?<![\d.,])(19|20)\d{2}(?!\.?\d)(?![\d,])")


@dataclass
class NumberViolation:
    stated: str
    value: float
    context: str

    def to_dict(self) -> dict:
        return {"stated": self.stated, "value": self.value, "context": self.context}


@dataclass
class GuardResult:
    passed: bool
    checked: int = 0
    violations: list[NumberViolation] = field(default_factory=list)
    source_value_count: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "numbers_checked": self.checked,
            "violations": [v.to_dict() for v in self.violations],
            "source_values": self.source_value_count,
            "note": self.note,
        }


def collect_source_numbers(data: Any) -> set[float]:
    """Every numeric value anywhere in a nested tool-output structure."""
    found: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return  # bool is an int subclass; True is not the number 1 here
        if isinstance(node, (int, float)):
            found.add(float(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple, set)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            # Tool results can carry numbers inside strings (dates, ids).
            for match in _NUMBER_RE.findall(node):
                try:
                    found.add(float(match.replace(",", "")))
                except ValueError:
                    continue

    walk(data)
    return found


def _acceptable_forms(value: float) -> set[float]:
    """
    The closed list of presentation transforms a narration may apply.

    Each entry is a way of DISPLAYING the same value, never a computation:
      - the value itself
      - rounded to 0, 1 or 2 decimal places (currency and percentage display)
      - the value as a percentage (x100), and that rounded
      - the absolute value, for "margin falls by $118.44" phrasing of -118.44
    """
    forms: set[float] = set()
    for base in (value, abs(value)):
        forms.add(base)
        for places in (0, 1, 2):
            forms.add(round(base, places))
        scaled = base * 100
        forms.add(scaled)
        for places in (0, 1, 2):
            forms.add(round(scaled, places))
    return forms


def _expand(source_values: Iterable[float]) -> set[float]:
    allowed: set[float] = set()
    for value in source_values:
        allowed |= _acceptable_forms(value)
    return allowed


def _stated_numbers(text: str) -> list[tuple[str, float, str]]:
    cleaned = _CITATION_RE.sub(" ", text)
    cleaned = _ISO_DATE_RE.sub(" ", cleaned)
    cleaned = _DATE_LIKE_RE.sub(" ", cleaned)
    out: list[tuple[str, float, str]] = []
    for match in _NUMBER_RE.finditer(cleaned):
        raw = match.group(0)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        start = max(0, match.start() - 35)
        context = cleaned[start:match.end() + 25].replace("\n", " ").strip()
        out.append((raw, value, context))
    return out


# Floats that went through round() do not compare exactly; 1e-6 is far tighter
# than any rounding difference and far looser than float noise.
TOLERANCE = 1e-6


def check(narration: str, source: Any) -> GuardResult:
    """Check a narration against the tool output it claims to describe."""
    source_values = collect_source_numbers(source)
    allowed = _expand(source_values)
    sorted_allowed = sorted(allowed)

    violations: list[NumberViolation] = []
    checked = 0

    for raw, value, context in _stated_numbers(narration):
        if abs(value) < IGNORE_BELOW:
            continue
        checked += 1
        if not _matches(value, sorted_allowed):
            violations.append(NumberViolation(stated=raw, value=value, context=context))

    return GuardResult(
        passed=not violations,
        checked=checked,
        violations=violations,
        source_value_count=len(source_values),
        note=("" if violations else "every stated number traces to tool output"),
    )


def _matches(value: float, sorted_allowed: list[float]) -> bool:
    import bisect

    if not sorted_allowed:
        return False
    i = bisect.bisect_left(sorted_allowed, value)
    for j in (i - 1, i):
        if 0 <= j < len(sorted_allowed) and abs(sorted_allowed[j] - value) <= TOLERANCE:
            return True
    return False


def guard_state(state: dict) -> GuardResult:
    """
    Guard a finished graph state: check the narration against the projected
    outcome the narration node was given.

    A template narration (offline mode) is still checked - it interpolates
    from the same dict, so it should pass, and if it ever stops passing that
    is a real bug in the templating.
    """
    narration = (state.get("narration") or "").strip()
    outcome = state.get("projected_outcome") or {}
    if not narration:
        return GuardResult(passed=True, note="no narration to check")
    if not outcome:
        return GuardResult(passed=True, note="no projected outcome to check against")

    # The verdict strings are generated from the same numbers and are part of
    # what the narration is allowed to restate.
    rec = state.get("recommendation") or {}
    source = {"outcome": outcome,
              "blockers": rec.get("blockers", []),
              "cautions": rec.get("cautions", [])}
    return check(narration, source)
