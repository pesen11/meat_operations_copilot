from datetime import date

from simulation.seasonality import (
    week_multiplier, is_closed, is_weekend,
    WINTER_MULTIPLIER, SUMMER_MULTIPLIER, LONG_WEEKEND_MULTIPLIER, CHRISTMAS_WEEK_MULTIPLIER,
)


class TestClosedDays:
    def test_christmas_is_closed(self):
        assert is_closed(date(2025, 12, 25))

    def test_regular_tuesday_is_open(self):
        assert not is_closed(date(2025, 3, 11))

    def test_new_years_day_is_closed(self):
        assert is_closed(date(2025, 1, 1))


class TestWeekendDetection:
    def test_saturday_is_weekend(self):
        assert is_weekend(date(2025, 5, 17))

    def test_wednesday_is_not_weekend(self):
        assert not is_weekend(date(2025, 5, 14))


class TestSeasonalMultiplier:
    def test_regular_winter_day_is_baseline(self):
        assert week_multiplier(date(2025, 1, 15)) == WINTER_MULTIPLIER

    def test_summer_day_gets_summer_multiplier(self):
        assert week_multiplier(date(2025, 7, 15)) == SUMMER_MULTIPLIER

    def test_saturday_before_monday_holiday_gets_long_weekend_multiplier(self):
        # The bug this test guards against: Sat before a Monday holiday
        # must NOT fall through to baseline just because it's in the
        # "previous" ISO calendar week.
        assert week_multiplier(date(2025, 5, 17)) == LONG_WEEKEND_MULTIPLIER

    def test_monday_holiday_itself_gets_long_weekend_multiplier(self):
        assert week_multiplier(date(2025, 5, 19)) == LONG_WEEKEND_MULTIPLIER

    def test_day_after_long_weekend_returns_to_baseline(self):
        assert week_multiplier(date(2025, 5, 20)) == WINTER_MULTIPLIER

    def test_christmas_eve_gets_christmas_multiplier(self):
        assert week_multiplier(date(2025, 12, 24)) == CHRISTMAS_WEEK_MULTIPLIER

    def test_christmas_day_gets_christmas_multiplier(self):
        assert week_multiplier(date(2025, 12, 25)) == CHRISTMAS_WEEK_MULTIPLIER

    def test_a_week_before_christmas_window_is_baseline(self):
        assert week_multiplier(date(2025, 12, 10)) == WINTER_MULTIPLIER

    def test_overlap_takes_max_not_product(self):
        # Canada Day (Jul 1, a Tuesday in 2025) falls in summer but is not
        # adjacent to a Fri/Mon, so no long-weekend bump — summer alone applies.
        assert week_multiplier(date(2025, 7, 1)) == SUMMER_MULTIPLIER
