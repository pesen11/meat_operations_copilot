from datetime import date

import pytest

from simulation.time_windows import (
    DEFAULT_WINDOW_DAYS, MAX_WINDOW_DAYS, TimeWindow,
    mentions_a_window, resolve_window,
)

# A Sunday, so the weekend cases have an unambiguous expected answer.
ANCHOR = date(2025, 12, 28)
EARLIEST = date(2025, 1, 1)


def win(question: str, anchor: date = ANCHOR) -> TimeWindow:
    return resolve_window(question, anchor=anchor, earliest=EARLIEST)


class TestNamedWindows:
    def test_yesterday_is_a_single_day(self):
        w = win("how did we do yesterday?")
        assert (w.start, w.end) == (date(2025, 12, 27), date(2025, 12, 27))
        assert w.days == 1

    def test_last_day_is_the_same_as_yesterday(self):
        assert win("how were the sales last day?").start == win("sales yesterday").start

    def test_last_week_is_seven_days(self):
        w = win("how was the ribeye sales last week")
        assert w.days == 7
        assert w.end == ANCHOR

    def test_last_month_is_thirty_days(self):
        assert win("what were our best sellers last month?").days == 30

    def test_last_year_is_twelve_months(self):
        # Unclamped: with `earliest` applied this correctly trims to the
        # first day of data, which TestClamping covers separately.
        assert resolve_window("how did we do this year", anchor=ANCHOR).days == 365

    def test_numeric_windows(self):
        assert win("sales for the last 10 days").days == 10
        assert win("sales over the past 3 weeks").days == 21
        assert win("how did the last 2 months go").days == 60

    def test_all_time_is_clamped_to_available_history(self):
        w = win("best sellers all time")
        assert w.start == EARLIEST


class TestWeekends:
    def test_last_weekend_is_saturday_and_sunday(self):
        w = win("how were the sales last weekend?")
        assert (w.start, w.end) == (date(2025, 12, 27), date(2025, 12, 28))
        assert w.days == 2
        assert not w.weekends_only

    def test_weekend_in_progress_resolves_to_the_completed_one(self):
        """
        Asked on a Saturday, "last weekend" means the one that finished, not
        the day the operator is standing in.
        """
        saturday = date(2025, 12, 27)
        w = resolve_window("how was last weekend", anchor=saturday, earliest=EARLIEST)
        assert (w.start, w.end) == (date(2025, 12, 20), date(2025, 12, 21))

    def test_weekends_plural_is_a_recurring_pattern_not_one_weekend(self):
        """
        "How much should we increase on weekends" asks about Saturdays and
        Sundays in general - answering it with one specific weekend would be
        a different question.
        """
        w = win("how much production should we increase during weekends?")
        assert w.weekends_only
        assert w.days > 2


class TestDefaulting:
    def test_no_phrase_falls_back_to_the_default_window(self):
        w = win("how are ribeye sales")
        assert w.days == DEFAULT_WINDOW_DAYS
        assert not w.phrase_matched

    def test_explicit_flag_distinguishes_asked_from_assumed(self):
        assert win("sales last week").phrase_matched
        assert not win("sales").phrase_matched

    def test_label_states_the_window_back_to_the_operator(self):
        assert win("sales yesterday").label == "yesterday"
        assert "28 days" in win("how are sales").label


class TestClamping:
    def test_window_never_starts_before_the_data(self):
        assert win("sales over the last 365 days", anchor=date(2025, 1, 10)).start == EARLIEST

    def test_window_never_ends_after_the_data(self):
        for q in ("sales last week", "sales yesterday", "sales last month"):
            assert win(q).end <= ANCHOR

    def test_absurd_numbers_are_capped(self):
        w = resolve_window("sales for the last 99999 days", anchor=ANCHOR)
        assert w.days <= MAX_WINDOW_DAYS + 1

    def test_zero_is_not_a_window(self):
        """Falls through to the default rather than producing an empty range."""
        w = win("sales for the last 0 days")
        assert w.days >= 1


class TestLongestMatchWins:
    @pytest.mark.parametrize("question,expected_days", [
        ("sales last 2 weeks", 14),
        ("sales last week", 7),
        ("sales last 3 months", 90),
        ("sales last month", 30),
    ])
    def test_specific_phrase_beats_the_shorter_one_inside_it(self, question, expected_days):
        assert win(question).days == expected_days


class TestMentionsAWindow:
    @pytest.mark.parametrize("question", [
        "how did we do yesterday", "sales last week", "best sellers last month",
        "how was last weekend", "revenue this year", "the last 5 days",
    ])
    def test_detects_time_expressions(self, question):
        assert mentions_a_window(question)

    @pytest.mark.parametrize("question", [
        "what products do we sell", "how many boxes of ribeye do we have",
        "what if we cut 15% more chuck roast",
    ])
    def test_ignores_questions_with_no_time_expression(self, question):
        assert not mentions_a_window(question)
