"""
"Last week", "yesterday", "last weekend" -> a concrete (start, end) date range.

WHY THIS EXISTS
---------------
Every history tool already accepted `start` and `end`. Nothing ever supplied
them. The agent called get_sales_summary(sku, end=as_of) and let the default
28-day window apply, so "how were sales yesterday", "how was last week" and
"what sold best last month" all returned the same 28-day answer — and the
operator could not tell, because the window was never stated back to them.
Parsing the window is the missing half of answering a history question.

Windows are resolved against the LAST DATE IN THE DATA, not against today.
The generated history ends on a fixed date; anchoring "last week" to the
real clock would silently return an empty window and read as "no sales",
which is a wrong answer rather than an unhelpful one. `anchor` is
injectable so this stays a pure function under test.

Everything here is deterministic and unit-tested. No LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

# Longest phrases first: "last two weeks" must be tested before "last week",
# and "this month" before "month". Each entry is (pattern, resolver-key).
# Ordered, not a dict, because the order is the whole point.
_PHRASES: list[tuple[str, str]] = [
    (r"\blast weekend\b", "last_weekend"),
    (r"\bthis weekend\b", "last_weekend"),
    (r"\bon the weekend\b", "last_weekend"),
    (r"\bweekends?\b", "weekends"),
    (r"\byesterday\b", "yesterday"),
    (r"\blast day\b", "yesterday"),
    (r"\bpast day\b", "yesterday"),
    (r"\btoday\b", "latest_day"),
    (r"\blast (\d+) days?\b", "n_days"),
    (r"\bpast (\d+) days?\b", "n_days"),
    (r"\blast (\d+) weeks?\b", "n_weeks"),
    (r"\bpast (\d+) weeks?\b", "n_weeks"),
    (r"\blast (\d+) months?\b", "n_months"),
    (r"\bpast (\d+) months?\b", "n_months"),
    (r"\blast (?:couple of |few )?weeks\b", "two_weeks"),
    (r"\bfortnight\b", "two_weeks"),
    (r"\blast week\b", "last_week"),
    (r"\bpast week\b", "last_week"),
    (r"\bthis week\b", "last_week"),
    (r"\blast month\b", "last_month"),
    (r"\bpast month\b", "last_month"),
    (r"\bthis month\b", "last_month"),
    (r"\blast quarter\b", "last_quarter"),
    (r"\blast (?:year|12 months)\b", "last_year"),
    (r"\bthis year\b", "last_year"),
    (r"\byear to date\b", "last_year"),
    (r"\bytd\b", "last_year"),
    (r"\ball time\b", "all_time"),
    (r"\ball[- ]time\b", "all_time"),
]

# Default when a history question names no window at all. 28 days matches
# what the analysis layer has always used, so an unqualified question keeps
# its previous meaning.
DEFAULT_WINDOW_DAYS = 28

MAX_WINDOW_DAYS = 400  # a year plus slack; guards "last 9999 days"


@dataclass(frozen=True)
class TimeWindow:
    """A resolved window, plus the wording to say it back to the operator."""
    start: date
    end: date
    label: str            # "last week", "the 28 days to 2025-12-28"
    phrase_matched: bool  # False => the default window, not something they asked for
    weekends_only: bool = False

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days": self.days,
            "label": self.label,
            "explicit": self.phrase_matched,
            "weekends_only": self.weekends_only,
        }


def _last_full_weekend(anchor: date) -> tuple[date, date]:
    """
    The most recent Saturday-Sunday pair at or before `anchor`.

    Anchored on Sunday: walk back to the last Sunday that has happened, then
    take the Saturday before it. If the anchor IS a Saturday, that Saturday
    is part of a weekend still in progress, so the completed weekend is the
    previous one — an operator asking "how was last weekend" on a Saturday
    means the one that finished, not the day they are standing in.
    """
    # weekday(): Mon=0 .. Sat=5, Sun=6
    days_since_sunday = (anchor.weekday() + 1) % 7
    sunday = anchor - timedelta(days=days_since_sunday)
    return sunday - timedelta(days=1), sunday


def resolve_window(
    question: str,
    anchor: date,
    default_days: int = DEFAULT_WINDOW_DAYS,
    earliest: Optional[date] = None,
) -> TimeWindow:
    """
    Parse a time window out of an operator's question.

    `anchor` is the last date the data covers. `earliest` clamps the start so
    a window never runs off the front of the dataset. When nothing matches,
    returns the default window with phrase_matched=False so callers can tell
    "they asked for 28 days" from "we assumed 28 days".
    """
    q = question.lower()

    for pattern, kind in _PHRASES:
        match = re.search(pattern, q)
        if not match:
            continue
        number = int(match.group(1)) if match.groups() and match.group(1) else None
        window = _build(kind, anchor, number)
        if window is not None:
            return _clamp(window, earliest, anchor)

    return _clamp(
        TimeWindow(
            start=anchor - timedelta(days=default_days - 1),
            end=anchor,
            label=f"the {default_days} days to {anchor.isoformat()}",
            phrase_matched=False,
        ),
        earliest, anchor,
    )


def _build(kind: str, anchor: date, number: Optional[int]) -> Optional[TimeWindow]:
    if kind == "yesterday":
        day = anchor - timedelta(days=1)
        return TimeWindow(day, day, "yesterday", True)
    if kind == "latest_day":
        return TimeWindow(anchor, anchor, f"{anchor.isoformat()}", True)
    if kind == "last_weekend":
        sat, sun = _last_full_weekend(anchor)
        return TimeWindow(sat, sun, "last weekend", True)
    if kind == "weekends":
        # Every weekend day in the trailing quarter — "how do weekends do"
        # is a question about Saturdays and Sundays in general, not about
        # one particular weekend.
        return TimeWindow(anchor - timedelta(days=89), anchor,
                          "weekends over the last 90 days", True, weekends_only=True)
    if kind == "last_week":
        return TimeWindow(anchor - timedelta(days=6), anchor, "the last 7 days", True)
    if kind == "two_weeks":
        return TimeWindow(anchor - timedelta(days=13), anchor, "the last 14 days", True)
    if kind == "last_month":
        return TimeWindow(anchor - timedelta(days=29), anchor, "the last 30 days", True)
    if kind == "last_quarter":
        return TimeWindow(anchor - timedelta(days=89), anchor, "the last 90 days", True)
    if kind == "last_year":
        return TimeWindow(anchor - timedelta(days=364), anchor, "the last 12 months", True)
    if kind == "all_time":
        return TimeWindow(date(1900, 1, 1), anchor, "all available history", True)
    if number is None or number <= 0:
        return None
    if kind == "n_days":
        days = min(number, MAX_WINDOW_DAYS)
        return TimeWindow(anchor - timedelta(days=days - 1), anchor,
                          f"the last {days} days", True)
    if kind == "n_weeks":
        days = min(number * 7, MAX_WINDOW_DAYS)
        return TimeWindow(anchor - timedelta(days=days - 1), anchor,
                          f"the last {number} weeks", True)
    if kind == "n_months":
        days = min(number * 30, MAX_WINDOW_DAYS)
        return TimeWindow(anchor - timedelta(days=days - 1), anchor,
                          f"the last {number} months", True)
    return None


def _clamp(window: TimeWindow, earliest: Optional[date], anchor: date) -> TimeWindow:
    """Keep a window inside the data. A window entirely outside it would
    return zero sales, which reads as 'we sold nothing' rather than 'we have
    no data for that'."""
    start, end = window.start, window.end
    if earliest is not None and start < earliest:
        start = earliest
    if end > anchor:
        end = anchor
    if start > end:
        start = end
    if (start, end) == (window.start, window.end):
        return window
    return TimeWindow(start, end, window.label, window.phrase_matched, window.weekends_only)


def mentions_a_window(question: str) -> bool:
    """True when the question names any time window at all. Used by the
    intent parser: a question with a time expression is asking about history
    even when it uses none of the history keywords."""
    q = question.lower()
    return any(re.search(pattern, q) for pattern, _ in _PHRASES)
