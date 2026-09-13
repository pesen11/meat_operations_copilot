"""
Tests for simulation/decision_engine.py.

Two properties matter more than any individual threshold:

1. A blocking factor cannot be outvoted by a weighted average. A capacity
   overrun is not compensated for by a good margin, because the week still
   does not fit.
2. An unassessed factor is excluded from the blend, not scored 1.0.
   "We did not look" and "we looked and it is fine" must not produce the same
   number - that is how a system becomes confident about things it never
   checked.
"""

from __future__ import annotations

import pytest

from models.decision import Action, FactorName, Verdict
from simulation.decision_engine import (
    CAPACITY_COMFORT, CAPACITY_HARD_LIMIT, COVER_CRITICAL_DAYS,
    SCORE_PROCEED, STOCKOUT_BLOCKING, WASTE_DISPROPORTION_CONCERN,
    blended_score, build_factors, capacity_factor, decide, economics_factor,
    execution_factor, inventory_factor, supply_factor, waste_factor,
)


def _outcome(**overrides) -> dict:
    """A comfortable scenario, so each test can spoil exactly one thing."""
    base = {
        "intent": "scenario",
        "demand_multiplier": 1.15,
        "labor": {
            "baseline_cutter_utilization_pct": 0.63,
            "scenario_cutter_utilization_pct": 0.63,
            "extra_weekly_cutting_minutes": 28.8,
            "weekly_cutter_minutes_available": 6630.0,
            "exceeds_cutter_capacity": False,
        },
        "inventory": {"baseline_days_of_cover": 2.49, "scenario_days_of_cover": 2.41},
        "margin": {"baseline_weekly_margin": 2904.91,
                   "scenario_weekly_margin": 3340.65,
                   "delta_weekly_margin": 435.74},
        "waste": {"baseline_weekly_trim_waste_kg": 46.13,
                  "scenario_weekly_trim_waste_kg": 53.04,
                  "extra_weekly_trim_waste_kg": 6.92},
        "supply": {"opening_kg": 633.0, "incoming_expected_kg": 2920.0,
                   "forecast_primal_draw_kg": 2541.0, "supply_reliability": 0.9895},
        "execution": {"scope": "Blade Eyes", "execution_rate": 0.98,
                      "rate_basis": "measured over 356 runs"},
        "stockout_risk": {"stockout_probability": 0.0},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Scoring convention
# ---------------------------------------------------------------------------

def test_all_scores_are_in_range():
    for factor in build_factors(_outcome()):
        assert 0.0 <= factor.score <= 1.0


def test_higher_score_always_means_more_comfortable():
    """No factor may invert the convention."""
    easy = capacity_factor({"scenario_cutter_utilization_pct": 0.60,
                            "exceeds_cutter_capacity": False})
    hard = capacity_factor({"scenario_cutter_utilization_pct": 0.95,
                            "exceeds_cutter_capacity": False})
    assert easy.score > hard.score

    rich = economics_factor({"baseline_weekly_margin": 1000.0,
                             "delta_weekly_margin": 200.0})
    thin = economics_factor({"baseline_weekly_margin": 1000.0,
                             "delta_weekly_margin": 5.0})
    assert rich.score > thin.score

    reliable = execution_factor({"execution_rate": 0.99})
    flaky = execution_factor({"execution_rate": 0.80})
    assert reliable.score > flaky.score


# ---------------------------------------------------------------------------
# Blocking beats the average
# ---------------------------------------------------------------------------

def test_capacity_overrun_blocks_despite_excellent_everything_else():
    outcome = _outcome(labor={
        "baseline_cutter_utilization_pct": 0.63,
        "scenario_cutter_utilization_pct": 1.19,
        "extra_weekly_cutting_minutes": 4000.0,
        "weekly_cutter_minutes_available": 6630.0,
        "exceeds_cutter_capacity": True,
    })
    recommendation = decide(outcome)
    assert recommendation.verdict == Verdict.DO_NOT_PROCEED
    assert recommendation.action == Action.DO_NOT_CHANGE
    assert recommendation.blockers
    # The other factors really were good - this is a veto, not a low average.
    assert recommendation.decision_score > 0.5


def test_margin_loss_blocks():
    outcome = _outcome(margin={"baseline_weekly_margin": 2904.91,
                               "scenario_weekly_margin": 2500.0,
                               "delta_weekly_margin": -404.91})
    recommendation = decide(outcome)
    assert recommendation.verdict == Verdict.DO_NOT_PROCEED
    assert any("margin falls" in b for b in recommendation.blockers)


def test_critical_cover_blocks():
    outcome = _outcome(inventory={"baseline_days_of_cover": 2.49,
                                  "scenario_days_of_cover": 0.4})
    recommendation = decide(outcome)
    assert recommendation.verdict == Verdict.DO_NOT_PROCEED
    assert any("under one day" in b for b in recommendation.blockers)


def test_coin_flip_stockout_blocks():
    outcome = _outcome(stockout_risk={"stockout_probability": STOCKOUT_BLOCKING + 0.1})
    recommendation = decide(outcome)
    assert recommendation.verdict == Verdict.DO_NOT_PROCEED


def test_comfortable_scenario_proceeds():
    recommendation = decide(_outcome())
    assert recommendation.verdict == Verdict.PROCEED
    assert recommendation.action == Action.INCREASE
    assert recommendation.blockers == []
    assert recommendation.decision_score >= SCORE_PROCEED
    assert recommendation.requires_approval


# ---------------------------------------------------------------------------
# Unassessed is not "fine"
# ---------------------------------------------------------------------------

def test_missing_data_marks_a_factor_unassessed():
    factor = capacity_factor({})
    assert not factor.assessed
    assert factor.score == 0.0


def test_unassessed_factors_are_excluded_from_the_blend():
    """An unassessed factor scored 0.0 must not drag the average down, and
    must not be silently counted as a pass either."""
    full = build_factors(_outcome())
    partial = build_factors(_outcome(waste={}, execution={}))

    assert blended_score(partial) is not None
    assessed_partial = [f for f in partial if f.assessed]
    assert len(assessed_partial) == len(full) - 2
    expected = (sum(f.score * f.weight for f in assessed_partial)
                / sum(f.weight for f in assessed_partial))
    assert blended_score(partial) == pytest.approx(expected)


def test_no_assessable_factors_gives_no_score():
    recommendation = decide({"intent": "scenario"})
    assert recommendation.decision_score is None
    assert recommendation.verdict == Verdict.INFORMATIONAL


def test_confidence_falls_when_fewer_factors_are_assessed():
    full = decide(_outcome())
    thin = decide(_outcome(waste={}, execution={}, supply={}))
    assert thin.confidence < full.confidence


# ---------------------------------------------------------------------------
# Waste is judged on proportionality
# ---------------------------------------------------------------------------

def test_proportional_waste_is_not_a_finding():
    """Cutting 15% more meat produces ~15% more trim. That is arithmetic,
    not a problem, and flagging it would bury the real findings."""
    factor = waste_factor({"baseline_weekly_trim_waste_kg": 100.0,
                           "scenario_weekly_trim_waste_kg": 115.0,
                           "extra_weekly_trim_waste_kg": 15.0,
                           "demand_multiplier": 1.15})
    assert factor.score == pytest.approx(1.0, abs=0.01)
    assert "in proportion" in factor.headline


def test_disproportionate_waste_scores_worse():
    factor = waste_factor({"baseline_weekly_trim_waste_kg": 100.0,
                           "scenario_weekly_trim_waste_kg": 160.0,
                           "extra_weekly_trim_waste_kg": 60.0,
                           "demand_multiplier": 1.15})
    assert factor.score < 0.6
    assert "faster than volume" in factor.headline


def test_waste_headline_uses_the_canonical_extra_field():
    """Re-deriving extra as scenario - baseline lands a cent from the field
    the outcome already carries, and two spellings of one number is what the
    number guard exists to catch."""
    factor = waste_factor({"baseline_weekly_trim_waste_kg": 46.13,
                           "scenario_weekly_trim_waste_kg": 53.04,
                           "extra_weekly_trim_waste_kg": 6.92,
                           "demand_multiplier": 1.15})
    assert "6.92" in factor.headline
    assert factor.detail["extra_weekly_trim_waste_kg"] == 6.92


# ---------------------------------------------------------------------------
# Limiting factor and provenance
# ---------------------------------------------------------------------------

def test_limiting_factor_is_the_lowest_assessed_score():
    outcome = _outcome(execution={"scope": "Blade Eyes", "execution_rate": 0.80,
                                  "rate_basis": "measured"})
    recommendation = decide(outcome)
    assert recommendation.limiting_factor == FactorName.EXECUTION
    assessed = [f for f in recommendation.factors if f.assessed]
    assert min(f.score for f in assessed) == next(
        f.score for f in assessed if f.name == recommendation.limiting_factor)


def test_execution_surfaces_as_a_constraint_the_old_verdict_never_saw():
    """The point of the rewrite: a primal that completes 83% of plan is a
    real limit on a production increase, and the single-threshold verdict
    had no way to express it."""
    recommendation = decide(_outcome(
        execution={"scope": "Blade Eyes", "execution_rate": 0.83,
                   "rate_basis": "measured over 356 runs"}))
    execution = next(f for f in recommendation.factors
                     if f.name == FactorName.EXECUTION)
    assert execution.assessed
    assert execution.score < 0.5
    assert "83%" in execution.headline


def test_every_assessed_factor_carries_evidence():
    for factor in build_factors(_outcome()):
        if factor.assessed:
            assert factor.evidence is not None
            assert factor.evidence.basis


def test_confidence_basis_is_the_weakest_link():
    recommendation = decide(_outcome())
    assert recommendation.confidence_basis is not None
    authorities = [f.evidence.authority for f in recommendation.factors
                   if f.evidence is not None]
    assert recommendation.confidence_basis.authority == min(authorities)


# ---------------------------------------------------------------------------
# Interaction with a sweep
# ---------------------------------------------------------------------------

def test_a_named_size_is_not_replaced_by_the_sweep(as_of=None):
    """When the operator supplied the size, the system evaluates it. It does
    not substitute its own preferred number."""
    from simulation.scenario_sweep import sweep_product
    sweep = sweep_product("chuck-roast", target_multiplier=1.15)
    recommendation = decide(_outcome(), sweep=sweep, operator_supplied_a_size=True)
    assert recommendation.recommended_multiplier is None
    assert recommendation.options


def test_an_open_question_gets_a_recommended_size():
    from simulation.scenario_sweep import sweep_product
    sweep = sweep_product("chuck-roast")
    recommendation = decide(_outcome(), sweep=sweep, operator_supplied_a_size=False)
    assert recommendation.recommended_multiplier is not None
    assert recommendation.recommended_change_pct == pytest.approx(
        recommendation.recommended_multiplier - 1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Informational questions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent", ["history", "catalog", "inventory_status"])
def test_informational_intents_need_no_approval(intent):
    recommendation = decide({"intent": intent})
    assert recommendation.verdict == Verdict.INFORMATIONAL
    assert recommendation.action == Action.NO_ACTION
    assert not recommendation.requires_approval


def test_to_dict_is_json_safe():
    import json
    json.dumps(decide(_outcome()).to_dict())
