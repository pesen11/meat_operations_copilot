"""
Tests for the number guard - the structural enforcement of "the LLM never
computes numbers".

The interesting tests here are the ones that check the guard is not TOO
permissive. A guard that accepts anything is worse than no guard, because it
certifies the invariant while not enforcing it.
"""

from __future__ import annotations

import pytest

from evals.narration_faithfulness import check_state
from evals.number_guard import check, collect_source_numbers, guard_state


class TestSourceCollection:
    def test_collects_nested_numbers(self):
        found = collect_source_numbers(
            {"a": 1.5, "b": {"c": [2.5, 3.5]}, "d": "text 4.5 more"})
        assert {1.5, 2.5, 3.5, 4.5} <= found

    def test_booleans_are_not_numbers(self):
        """True is an int subclass in Python; treating it as the value 1 would
        let a narration state '1' of anything and pass."""
        found = collect_source_numbers({"flag": True, "other": False})
        assert 1.0 not in found and 0.0 not in found


class TestAcceptedFormatting:
    """A narration may re-format a tool value; it may not compute a new one."""

    def test_percentage_rendering_is_accepted(self):
        result = check("Utilization goes to 62.7%.", {"util": 0.6268})
        assert result.passed, result.violations

    def test_currency_rounding_is_accepted(self):
        result = check("Margin is $789.60 per week.", {"margin": 789.6})
        assert result.passed

    def test_thousands_separators_are_accepted(self):
        result = check("Revenue was $106,335.98.", {"revenue": 106335.98})
        assert result.passed

    def test_absolute_value_phrasing_is_accepted(self):
        result = check("Margin falls by $118.44.", {"delta": -118.44})
        assert result.passed

    def test_iso_dates_are_not_read_as_negative_numbers(self):
        """
        Regression: stripping only the YEAR from "2025-12-31" left "-12-31",
        which the number regex read as -12 and -31 and checked against source
        data that contained no such values. Any narration naming a date was
        one step from a false violation.
        """
        result = check("Sales ran from 2025-12-04 to 2025-12-31.", {"kg": 500.0})
        assert result.passed, result.violations

    def test_a_four_digit_quantity_is_not_mistaken_for_a_year(self):
        """
        Regression: "2044.71 kg" starts with 20, so the bare-year rule ate
        "2044" and left ".71" to be checked as 71 - a violation invented out
        of correct output.
        """
        result = check("Short ribs sold 2044.71 kg.", {"kg": 2044.71})
        assert result.passed, result.violations

    def test_a_four_digit_quantity_is_still_checked(self):
        """The fix must not smuggle 4-digit numbers past the guard."""
        result = check("Short ribs sold 2044.71 kg.", {"kg": 1500.0})
        assert not result.passed

    def test_a_bare_year_is_still_ignored(self):
        result = check("Trade was stronger in 2025.", {"kg": 500.0})
        assert result.passed, result.violations

    def test_hyphenated_range_in_prose_is_a_real_violation(self):
        """
        Not a tokenisation bug: "a target of 20-25 boxes" genuinely reads as
        the number -25. The narration says "20 to 25" instead; this pins the
        guard's behaviour so nobody "fixes" it by loosening the parser.
        """
        result = check("Target is 20-25 boxes.", {"low": 20.0, "high": 25.0})
        assert not result.passed

    def test_citation_markers_are_ignored(self):
        result = check("See the procedure [12] for details.", {"x": 3.0})
        assert result.passed

    def test_dates_are_ignored(self):
        result = check("As of 2025 the figure is 42.0.", {"x": 42.0})
        assert result.passed


class TestRejectedFabrication:
    def test_invented_number_is_caught(self):
        result = check("Margin rises to $999.99.", {"margin": 118.44})
        assert not result.passed
        assert result.violations[0].stated == "999.99"

    def test_arithmetic_on_tool_values_is_caught(self):
        """230.0 is 2 x 115.0. The model doing that sum is precisely the
        behaviour this project forbids, so it must NOT be accepted."""
        result = check("That is 230.0 kg in total.", {"weekly": 115.0})
        assert not result.passed

    def test_near_miss_is_caught(self):
        """A guard with a percentage tolerance would pass this. It must not -
        tolerance is how 'the model may do a little arithmetic' creeps in."""
        result = check("Cover falls to 2.6 days.", {"cover": 2.41})
        assert not result.passed

    def test_small_numbers_are_ignored_as_structural(self):
        result = check("There is 1 blocker.", {"x": 500.0})
        assert result.passed

    def test_violation_reports_context(self):
        result = check("Waste rises by 44.4 kg per week.", {"waste": 6.92})
        assert not result.passed
        assert "44.4" in result.violations[0].context


class TestGuardState:
    def test_real_graph_narration_passes(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from agents.graph import build_graph, run_question

        graph = build_graph(checkpointer=InMemorySaver())
        for i, question in enumerate([
            "what if we increase chuck roast production by 15%?",
            "cut ground beef medium back 30%",
            "how many boxes of blade eyes do we have on hand?",
        ]):
            state = run_question(question, thread_id=f"guard-{i}", graph=graph)
            result = guard_state(state)
            assert result.passed, (question, result.violations)

    def test_empty_narration_passes_trivially(self):
        assert guard_state({}).passed


class TestNarrationFaithfulness:
    def test_backwards_margin_direction_is_caught(self):
        state = {
            "narration": "Good news - weekly margin improves under this plan.",
            "projected_outcome": {"margin": {"delta_weekly_margin": -250.0}},
            "recommendation": {"verdict": "do_not_proceed"},
        }
        result = check_state(state)
        assert not result.passed
        assert not result.direction_ok

    def test_correct_margin_direction_passes(self):
        state = {
            "narration": "Weekly margin falls by $250.00 under this plan.",
            "projected_outcome": {"margin": {"delta_weekly_margin": -250.0}},
            "recommendation": {"verdict": "do_not_proceed", "blockers": []},
        }
        result = check_state(state)
        assert result.direction_ok

    def test_narration_contradicting_the_verdict_is_caught(self):
        state = {
            "narration": "This is recommended, go ahead with it.",
            "projected_outcome": {"margin": {"delta_weekly_margin": 10.0}},
            "recommendation": {"verdict": "do_not_proceed"},
        }
        result = check_state(state)
        assert not result.verdict_ok

    def test_direction_word_in_an_unrelated_sentence_is_not_a_violation(self):
        """'Waste rises' must not be read as a claim about margin."""
        state = {
            "narration": ("Weekly margin falls by $250.00. Trim waste rises by "
                          "12.00 kg."),
            "projected_outcome": {
                "margin": {"delta_weekly_margin": -250.0},
                "waste": {"extra_weekly_trim_waste_kg": 12.0},
            },
            "recommendation": {"verdict": "do_not_proceed", "blockers": []},
        }
        result = check_state(state)
        assert result.direction_ok, result.issues
