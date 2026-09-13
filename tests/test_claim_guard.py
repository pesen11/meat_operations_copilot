"""
Tests for evals/claim_guard.py and evals/scenarios.py.

The scenario-harness tests here are mostly NEGATIVE controls. A ground-truth
suite that cannot fail is worse than no suite: it reports 100% forever and
everyone stops reading it. So these assert that each scored dimension
actually discriminates, and that the data perturbations genuinely change the
numbers the system sees.
"""

from __future__ import annotations

import pytest

from evals import claim_guard
from evals.scenarios import ScenarioCase, perturbed, run_case


# ---------------------------------------------------------------------------
# Claim guard: what it catches
# ---------------------------------------------------------------------------

def test_business_rule_presented_as_a_calculation_is_caught():
    """1.9x is the shop's confirmed figure. Claiming authorship of it hides
    that the number is theirs to change."""
    result = claim_guard.check(
        "We calculate that Christmas needs 1.9x normal demand.",
        {"seasonal": {"multiplier": 1.9}}, {"verdict": "informational"}, {})
    assert not result.passed
    assert any(v.kind == "unattributed_business_rule" for v in result.violations)


def test_business_rule_correctly_attributed_passes():
    result = claim_guard.check(
        "The shop's confirmed uplift for Christmas is 1.9x normal demand.",
        {"seasonal": {"multiplier": 1.9}}, {"verdict": "informational"}, {})
    assert result.passed
    assert result.attributed_claims >= 1


def test_measurement_claim_without_a_measurement_is_caught():
    result = claim_guard.check(
        "Blade Eyes historically completes 87% of its planned cutting.",
        {}, {"verdict": "proceed"}, {})
    assert not result.passed
    assert any(v.kind == "unsupported_measurement" for v in result.violations)


def test_measurement_claim_with_a_measurement_passes():
    result = claim_guard.check(
        "Blade Eyes historically completes 87% of its planned cutting.",
        {"execution": {"rate": 0.87, "scope": "Blade Eyes"}},
        {"verdict": "proceed"}, {})
    assert result.passed


@pytest.mark.parametrize("verdict,text", [
    ("do_not_proceed", "Go ahead and increase production."),
    ("proceed", "This is not achievable with current capacity."),
])
def test_narration_contradicting_the_verdict_is_caught(verdict, text):
    result = claim_guard.check(text, {}, {"verdict": verdict}, {})
    assert not result.passed
    assert any(v.kind == "contradicts_verdict" for v in result.violations)


def test_a_number_sourced_from_the_model_is_a_structural_violation():
    """The project's core rule at the provenance layer: no number may enter
    the plan as a model's invention."""
    result = claim_guard.check(
        "Fine.", {}, {},
        {"demand_multiplier": {"value": 1.15, "source": "llm_inference",
                               "basis": "model guessed"}})
    assert not result.passed
    assert any(v.kind == "ungrounded_source" for v in result.violations)


def test_a_non_numeric_llm_field_is_allowed():
    """The model IS allowed to read the intent from the question. Only
    NUMBERS are forbidden to originate there."""
    result = claim_guard.check(
        "Fine.", {}, {},
        {"intent": {"value": "scenario", "source": "llm_inference",
                    "basis": "model read of the question"}})
    assert result.passed


def test_empty_narration_is_not_a_violation():
    assert claim_guard.check("", {}, {}, {}).passed


# ---------------------------------------------------------------------------
# Claim guard against real graph output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "what if we increase chuck roast production by 15%?",
    "how much more do i need to produce for christmas?",
    "what were our best sellers last month?",
    "what do we need to restock?",
])
def test_real_narrations_pass_the_claim_guard(question):
    from agents.graph import build_graph, run_question
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(InMemorySaver())
    state = run_question(question, thread_id=f"cg-{abs(hash(question))}", graph=graph)
    result = claim_guard.guard_state(state)
    assert result.passed, [v.to_dict() for v in result.violations]


# ---------------------------------------------------------------------------
# Scenario harness: the perturbations must actually bite
# ---------------------------------------------------------------------------

def test_labour_perturbation_changes_capacity():
    from simulation.inventory_calc import avg_weekly_cutter_minutes
    baseline = avg_weekly_cutter_minutes()
    with perturbed(labor_scale=0.6):
        assert avg_weekly_cutter_minutes() == pytest.approx(baseline * 0.6, rel=0.01)
    assert avg_weekly_cutter_minutes() == pytest.approx(baseline, rel=1e-6)


def test_stock_perturbation_changes_positions():
    from simulation.inventory_calc import all_stock_positions
    baseline = all_stock_positions()[0].boxes_on_hand
    with perturbed(stock_scale=0.25):
        assert all_stock_positions()[0].boxes_on_hand < baseline
    assert all_stock_positions()[0].boxes_on_hand == baseline


def test_dropping_orders_removes_incoming_supply():
    from simulation.supply_calc import open_orders
    baseline = len(open_orders())
    assert baseline > 0
    with perturbed(drop_open_orders=True):
        assert open_orders() == []
    assert len(open_orders()) == baseline


def test_execution_perturbation_changes_the_rate():
    from simulation.execution_rates import execution_rate
    baseline = execution_rate().rate
    with perturbed(execution_scale=0.5):
        assert execution_rate().rate == pytest.approx(baseline * 0.5, rel=0.01)
    assert execution_rate().rate == pytest.approx(baseline, rel=1e-6)


def test_perturbation_is_restored_even_after_an_exception():
    """A scenario that raises must not poison every case after it."""
    from simulation.inventory_calc import avg_weekly_cutter_minutes
    baseline = avg_weekly_cutter_minutes()
    with pytest.raises(RuntimeError):
        with perturbed(labor_scale=0.1):
            raise RuntimeError("boom")
    assert avg_weekly_cutter_minutes() == pytest.approx(baseline, rel=1e-6)


# ---------------------------------------------------------------------------
# Scenario harness: every dimension must be able to fail
# ---------------------------------------------------------------------------

_HISTORY_Q = "what were our best sellers last month?"


@pytest.mark.parametrize("dimension,case", [
    ("intent", ScenarioCase(name="x", question=_HISTORY_Q, fault="n/a",
                            expect_intent="scenario")),
    ("dispatch", ScenarioCase(name="x", question=_HISTORY_Q, fault="n/a",
                              expect_agents={"supply", "production"})),
    ("decision", ScenarioCase(name="x", question=_HISTORY_Q, fault="n/a",
                              expect_verdict_in={"do_not_proceed"})),
    ("evidence", ScenarioCase(name="x", question=_HISTORY_Q, fault="n/a",
                              expect_evidence={"incoming_supply"})),
    ("diagnosis", ScenarioCase(name="x",
                               question="what if we increase chuck roast production by 10%?",
                               fault="n/a", expect_bottleneck_kind="supply")),
])
def test_each_dimension_can_fail(dimension, case):
    result = run_case(case)
    assert not result.passed
    assert result.dimensions.get(dimension) is False, (
        f"{dimension} did not register the deliberate mismatch")


def test_dispatch_flags_an_agent_that_should_not_have_run():
    result = run_case(ScenarioCase(
        name="x", question=_HISTORY_Q, fault="n/a",
        expect_not_agents={"historical"}))
    assert result.dimensions["dispatch"] is False
    assert any("unnecessary" in f for f in result.failures)


def test_the_shipped_cases_all_pass():
    from evals.scenarios import run_all
    report = run_all()
    failing = [c["name"] for c in report["cases"] if not c["passed"]]
    assert not failing, f"failing scenarios: {failing}"
    assert report["by_dimension"]["grounding"]["rate"] == 1.0
