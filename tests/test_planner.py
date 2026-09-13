"""
Tests for agents/planner.py and the request/provenance split.

The properties that matter here are about DISCIPLINE:

- an agent runs because the question needs it, not because a node exists
- a skipped agent is still accounted for in the progress stream
- "the operator said 15%" and "the operator asked how much" stay
  distinguishable all the way down
- evidence that never reached the outcome is reported, not silently absent
"""

from __future__ import annotations

import pytest

from agents import planner
from agents.intent import build_request, extract_explicit_multiplier
from models.request import EvidenceKind, Intent


# ---------------------------------------------------------------------------
# Explicit vs absent size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("what if we increase chuck roast by 15%?", 1.15),
    ("cut ground beef medium back 30%", 0.70),
    ("scale back ribeye 20 percent", 0.80),
])
def test_stated_sizes_are_captured(question, expected):
    assert extract_explicit_multiplier(question) == pytest.approx(expected)


@pytest.mark.parametrize("question", [
    "how much more do we need to produce for christmas?",
    "what should we plan for the long weekend?",
    "how many boxes of ribeye do we have?",
])
def test_absent_sizes_return_none_not_one(question):
    """1.0 and "they did not say" are different facts. Collapsing them is why
    seasonal planning once had to be special-cased."""
    assert extract_explicit_multiplier(question) is None


def test_operator_supplied_flag_distinguishes_the_two_questions():
    named = build_request("what if we increase chuck roast production by 15%?")
    asked = build_request("how much more do i need to produce for christmas?")
    assert named.operator_supplied_a_size
    assert not asked.operator_supplied_a_size
    assert named.multiplier == pytest.approx(1.15)
    assert asked.multiplier is None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_named_product_is_attributed_to_the_user():
    request = build_request("what if we increase chuck roast production by 15%?")
    assert request.product_sku.source == "user"
    assert request.sku == "chuck-roast"


def test_derived_primal_is_attributed_as_inferred():
    """The operator said "chuck roast", not "Blade Eyes". The primal is the
    system's lookup, and must not be presented as something they said."""
    request = build_request("what if we increase chuck roast production by 15%?")
    assert request.source_primal.value == "Blade Eyes"
    assert request.source_primal.source == "inferred"


def test_named_period_is_user_sourced():
    request = build_request("how much more for christmas?")
    assert request.period.value == "christmas"
    assert request.period.source == "user"


def test_unnamed_period_is_not_invented_at_parse_time():
    """"What's coming up" names no season. The calendar infers one later, in
    the inventory agent, and that inference is recorded there - the parser
    must not guess one and label it 'user'."""
    request = build_request("what is coming up that we need to prepare for?")
    assert request.intent == Intent.SEASONAL_PLANNING
    assert request.period is None


def test_every_numeric_evidence_has_an_allowed_source():
    """The project's core rule, enforced at the provenance layer: no number
    may enter the plan as a model's invention."""
    from models.provenance import NUMERIC_SOURCES_ALLOWED
    for question in ("what if we increase chuck roast production by 15%?",
                     "cut ground beef medium back 30%",
                     "how much more for christmas?"):
        for evidence in build_request(question).to_plan()["provenance"].values():
            if isinstance(evidence["value"], (int, float)):
                assert evidence["source"] in NUMERIC_SOURCES_ALLOWED


def test_plan_round_trip_keeps_the_legacy_shape():
    """Every existing node reads the flat Plan. Widening the contract must
    not have changed it."""
    plan = build_request("what if we increase chuck roast production by 15%?").to_plan()
    for key in ("intent", "product_sku", "source_primal", "demand_multiplier",
                "period", "as_of", "window", "reasoning", "resolved_by"):
        assert key in plan
    assert plan["demand_multiplier"] == pytest.approx(1.15)


def test_absent_size_still_collapses_to_one_for_downstream_maths():
    plan = build_request("how much more for christmas?").to_plan()
    assert plan["demand_multiplier"] == 1.0
    assert plan["operator_supplied_a_size"] is False


# ---------------------------------------------------------------------------
# Evidence requirements and dispatch
# ---------------------------------------------------------------------------

def test_history_question_needs_only_history():
    kinds = planner.required_evidence("history")
    assert kinds == {EvidenceKind.SALES_HISTORY}
    assert planner.agents_for(kinds) == ["historical"]


def test_history_question_skips_three_agents():
    assert set(planner.skipped_agents(["historical"])) == {
        "inventory", "production", "supply"}


def test_scenario_needs_the_full_board():
    agents = planner.agents_for(planner.required_evidence("scenario"))
    assert set(agents) == {"inventory", "production", "historical", "supply"}


def test_unsupported_question_dispatches_nothing():
    assert planner.agents_for(planner.required_evidence("unsupported")) == []


def test_every_dispatched_agent_supplies_something_required():
    """The point of dispatch: no agent runs that cannot contribute."""
    for intent in ("scenario", "seasonal_planning", "inventory_status",
                   "history", "catalog", "supply"):
        kinds = planner.required_evidence(intent)
        for agent in planner.agents_for(kinds):
            assert planner.AGENT_CAPABILITIES[agent] & kinds, (
                f"{agent} dispatched for {intent} but supplies nothing it needs")


def test_agent_order_is_stable():
    kinds = planner.required_evidence("scenario")
    assert planner.agents_for(kinds) == planner.agents_for(kinds)


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------

def test_missing_evidence_is_detected():
    gaps = planner.evidence_gaps({EvidenceKind.SALES_HISTORY}, {})
    assert gaps == [EvidenceKind.SALES_HISTORY]


def test_present_evidence_is_recognised_under_any_of_its_keys():
    for key in ("sales_summary", "top_movers", "catalog_performance"):
        assert planner.evidence_present(EvidenceKind.SALES_HISTORY, {key: [1]})


def test_empty_values_do_not_count_as_present():
    """An empty list is not evidence. Counting it would make the sufficiency
    check report success for a tool that returned nothing."""
    for empty in ({}, [], None, ""):
        assert not planner.evidence_present(
            EvidenceKind.SALES_HISTORY, {"top_movers": empty})


def test_coverage_is_a_fraction_of_required_kinds():
    required = {EvidenceKind.SALES_HISTORY, EvidenceKind.STOCK_POSITION}
    assert planner.coverage(required, {"top_movers": [1]}) == pytest.approx(0.5)
    assert planner.coverage(required, {}) == 0.0
    assert planner.coverage(set(), {}) == 1.0


def test_retry_is_capped_at_one_pass():
    """A gap that survives one supplementary fetch means the data does not
    exist. A graph that can loop is a graph that can hang."""
    gaps = [EvidenceKind.SALES_HISTORY]
    assert planner.should_gather_more(gaps, attempts=0)
    assert not planner.should_gather_more(gaps, attempts=planner.MAX_GATHER_ATTEMPTS)


def test_non_critical_gaps_do_not_trigger_a_retry():
    """A missing catalog listing is not worth a second round trip."""
    assert not planner.should_gather_more([EvidenceKind.CATALOG_LISTING], attempts=0)


def test_state_key_map_covers_every_evidence_kind():
    """A kind with no state keys can never be found present, so the
    sufficiency check would report it missing forever and burn the one
    retry on every single request."""
    for kind in EvidenceKind:
        assert planner.EVIDENCE_STATE_KEYS.get(kind), f"{kind.value} has no state keys"
