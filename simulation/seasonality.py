"""
Deterministic seasonality & calendar rules, confirmed with the user:

- Shop is open 7 days/week, closed on Ontario statutory holidays.
- Regular winter week: baseline (1.0x).
- Summer week: 1.4x.
- Long-weekend week: 1.7x.
- Christmas week: 1.9x.

Overlapping cases (e.g. a long weekend that falls in summer) take the
MAX applicable multiplier, not a product of factors — user gave these as
independent peak scenarios, not stacking effects, and multiplying them
would produce spikes nobody confirmed (e.g. 1.4 x 1.7 = 2.38x, well past
the largest number the user actually gave us, which was 1.9x for
Christmas). Capping at the highest confirmed multiplier is the more
defensible assumption pending real correction.

IMPORTANT: elevated periods are computed as day-windows around the actual
holiday/event, NOT by grouping into Mon-Sun calendar weeks. An earlier
version grouped by ISO week and missed the Saturday immediately before a
Monday holiday (it fell in the *previous* ISO week), even though that's
exactly when long-weekend shopping actually spikes. Day-windows avoid that
boundary bug.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import holidays

PROVINCE = "ON"

WINTER_MULTIPLIER = 1.0
SUMMER_MULTIPLIER = 1.4
LONG_WEEKEND_MULTIPLIER = 1.7
CHRISTMAS_WEEK_MULTIPLIER = 1.9

# Summer = Jun, Jul, Aug (standard meat-counter BBQ season assumption; not
# separately confirmed with the user beyond "summer" — flagged if it needs narrowing).
SUMMER_MONTHS = {6, 7, 8}

# Christmas elevated window: the 7 days ending on Dec 25 (people buy in the
# run-up to the holiday, not the week after). Not separately confirmed with
# the user beyond "week of Christmas" — flagged assumption on exact bounds.
CHRISTMAS_WINDOW_DAYS_BEFORE = 6


@lru_cache(maxsize=8)
def _holiday_calendar(year: int) -> "holidays.HolidayBase":
    # Pull one extra year on each side so window lookups near Jan 1 / Dec 31 work.
    return holidays.Canada(subdiv=PROVINCE, years=[year - 1, year, year + 1])


def is_closed(d: date) -> bool:
    """Shop closes on Ontario statutory holidays."""
    return d in _holiday_calendar(d.year)


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6


def is_long_weekend_period(d: date) -> bool:
    """
    True if d falls within the actual long-weekend shopping window around a
    stat holiday: for a Monday holiday, that's the Fri/Sat/Sun before it
    plus the Monday itself; for a Friday holiday, that's the Friday plus
    the Sat/Sun/Mon after it. Checks holidays within +/-4 days of d to stay
    correct near year boundaries.
    """
    cal = _holiday_calendar(d.year)
    for offset in range(-4, 5):
        candidate = d + timedelta(days=offset)
        if candidate not in cal:
            continue
        wd = candidate.weekday()
        if wd == 0:  # Monday holiday -> elevated Fri, Sat, Sun, Mon
            window = {candidate - timedelta(days=i) for i in range(3, -1, -1)}
        elif wd == 4:  # Friday holiday -> elevated Fri, Sat, Sun, Mon
            window = {candidate + timedelta(days=i) for i in range(0, 4)}
        else:
            continue
        if d in window:
            return True
    return False


def is_christmas_period(d: date) -> bool:
    """7-day window ending on Dec 25 (run-up to Christmas)."""
    christmas = date(d.year, 12, 25)
    window_start = christmas - timedelta(days=CHRISTMAS_WINDOW_DAYS_BEFORE)
    return window_start <= d <= christmas


def is_summer(d: date) -> bool:
    return d.month in SUMMER_MONTHS


def week_multiplier(d: date) -> float:
    """The seasonal revenue multiplier that applies to day d."""
    candidates = [WINTER_MULTIPLIER]
    if is_summer(d):
        candidates.append(SUMMER_MULTIPLIER)
    if is_long_weekend_period(d):
        candidates.append(LONG_WEEKEND_MULTIPLIER)
    if is_christmas_period(d):
        candidates.append(CHRISTMAS_WEEK_MULTIPLIER)
    return max(candidates)

