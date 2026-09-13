"""
Tests for the LangGraph ops graph (agents/).

All of these run in offline mode (OPS_COPILOT_LLM=stub, set by the autouse
fixture) so the suite is deterministic and free. That is possible precisely
because no number in the pipeline comes from the model - see agents/llm.py.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agents import llm
from agents.graph import build_graph, resume_approval, run_question
from agents.intent import keyword_plan, parse_question
from agents.nodes import _verdict, MAX_SAFE_CUTTER_UTILIZATION


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("OPS_COPILOT_LLM", "stub")


@pytest.fixture
def graph():
    """A fresh graph with its own checkpointer per test, so thread ids from
    one test can't collide with another's."""
    return build_graph(checkpointer=InMemorySaver())


class TestOfflineMode:
    def test_stub_env_forces_offline_even_with_a_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
        monkeypatch.setenv("OPS_COPILOT_LLM", "stub")
        assert llm.is_live() is False

    def test_complete_returns_the_offline_prefill(self):
        result = llm.complete("system", "user", prefill_offline="fallback text")
        assert result.live is False
        assert result.text == "fallback text"


class TestIntentParsing:
    @pytest.mark.parametrize("question,sku,multiplier", [
        ("what if we increase chuck roast production by 15%?", "chuck-roast", 1.15),
        ("should we scale back ribeye steak by 20 percent", "ribeye-steak-bnls", 0.8),
        ("cut ground beef medium back 30%", "ground-beef-medium", 0.7),
        ("what happens if we make 25% more pulled pork roast", "pulled-pork-roast", 1.25),
        ("reduce osso buco by 10%", "osso-buco", 0.9),
    ])
    def test_scenario_questions_resolve_sku_and_direction(self, question, sku, multiplier):
        plan = parse_question(question)
        assert plan["intent"] == "scenario"
        assert plan["product_sku"] == sku
        assert plan["demand_multiplier"] == pytest.approx(multiplier)

    @pytest.mark.parametrize("question,intent", [
        ("how many boxes of blade eyes do we have on hand?", "inventory_status"),
        ("what were our top sellers last month", "history"),
        ("what products do we carry", "catalog"),
    ])
    def test_non_scenario_intents(self, question, intent):
        assert parse_question(question)["intent"] == intent

    @pytest.mark.parametrize("question", [
        "how do I trim a brisket fat cap",
        "what is the holding temperature for ground beef",
        "what's the cleaning procedure for the bandsaw",
    ])
    def test_sop_questions_are_not_answered_by_the_simulation(self, question):
        """Procedural questions must NOT be answered with a margin projection -
        they belong to the RAG layer."""
        assert parse_question(question)["intent"] == "unsupported"

    def test_ambiguous_product_words_do_not_guess(self):
        assert keyword_plan("sell more pork")["product_sku"] is None
        assert keyword_plan("more steak please")["product_sku"] is None


class TestSeasonalPlanning:
    """Regression tests for a real reported gap: the system could evaluate a
    scenario the operator proposed, but not answer 'how much should we
    increase for Christmas?' - it returned 'unsupported'."""

    @pytest.mark.parametrize("question,period", [
        ("How much production increase should we do during christmas?", "christmas"),
        ("What should we do for the holidays?", "christmas"),
        ("How much more should we cut for the long weekend?", "long_weekend"),
        ("how much extra should we make for summer bbq season", "summer"),
    ])
    def test_seasonal_questions_are_not_unsupported(self, question, period):
        plan = parse_question(question)
        assert plan["intent"] == "seasonal_planning", question
        assert plan["period"] == period

    def test_open_ended_whats_coming_up_uses_the_calendar(self):
        plan = parse_question("what is coming up that we should prepare for")
        assert plan["intent"] == "seasonal_planning"
        assert plan["period"] is None

    def test_seasonal_run_uses_the_confirmed_multiplier(self, graph):
        result = run_question("How much production increase should we do during christmas?",
                              thread_id="t-xmas", graph=graph, as_of="2025-11-30")
        seasonal = result["projected_outcome"]["seasonal"]
        # 1.9x is the project owner's confirmed Christmas figure - not the
        # model's, and not an estimate.
        assert seasonal["multiplier"] == 1.9
        assert "confirmed" in seasonal["multiplier_source"]
        assert seasonal["total_extra_boxes"] > 0
        assert seasonal["top_primals_to_order"]

    def test_seasonal_run_reports_labor_and_waste(self, graph):
        result = run_question("how much more for christmas", thread_id="t-xmas2",
                              graph=graph, as_of="2025-11-30")
        outcome = result["projected_outcome"]
        labor = outcome["labor"]
        assert (labor["scenario_cutter_utilization_pct"]
                > labor["baseline_cutter_utilization_pct"])
        assert outcome["waste"]["extra_weekly_trim_waste_kg"] > 0

    def test_seasonal_narration_passes_the_number_guard(self, graph):
        from evals.number_guard import guard_state
        for i, q in enumerate(["How much production increase should we do during christmas?",
                               "How much more should we cut for the long weekend?",
                               "what is coming up that we should prepare for"]):
            state = run_question(q, thread_id=f"t-seas-{i}", graph=graph,
                                 as_of="2025-11-30")
            result = guard_state(state)
            assert result.passed, (q, result.violations)

    def test_scenario_questions_still_work(self):
        """The seasonal branch is checked first; it must not swallow an
        ordinary what-if that happens to mention no season."""
        plan = parse_question("what if we increase chuck roast production by 15%?")
        assert plan["intent"] == "scenario"
        assert plan["demand_multiplier"] == pytest.approx(1.15)


class TestGraphRun:
    def test_scenario_run_produces_all_three_impact_dimensions(self, graph):
        result = run_question("what if we increase chuck roast production by 15%?",
                              thread_id="t-scenario", graph=graph)
        outcome = result["projected_outcome"]
        assert outcome["product_sku"] == "chuck-roast"
        assert outcome["demand_multiplier"] == pytest.approx(1.15)
        # inventory, labor, waste - the promise the project makes
        assert "inventory" in outcome and "labor" in outcome and "waste" in outcome
        assert outcome["margin"]["delta_weekly_margin"] > 0

    def test_all_three_agents_report_in_parallel(self, graph):
        result = run_question("what if we increase chuck roast production by 15%?",
                              thread_id="t-parallel", graph=graph)
        nodes = [t["node"] for t in result["trace"]]
        for node in ("inventory_agent", "production_agent", "historical_agent"):
            assert node in nodes
        assert not result.get("errors")

    def test_irrelevant_agents_are_not_dispatched_but_still_report(self, graph):
        """
        A history question needs sales history and nothing else.

        The contract CHANGED here, deliberately. The graph used to run every
        specialist unconditionally and have idle ones return
        {"skipped": True}, so that the SSE stream showed three agents
        reporting in. That rationale was about the stream, not the work, and
        it is satisfiable without doing the work: the supervisor now emits a
        skip event for each agent it declines to dispatch. So the assertion
        is no longer "the node ran and wrote a marker" but "the node did not
        run, AND the operator was told why".
        """
        result = run_question("what were our top sellers last month",
                              thread_id="t-history", graph=graph)

        assert result["historical"]["top_movers"]
        assert result["branches"] == ["historical"]
        # Not dispatched at all - no state key, not an empty marker.
        assert "inventory" not in result
        assert "production" not in result
        assert "supply" not in result

        # ...but the progress stream still accounts for every one of them.
        # `is True`, not truthiness: the supervisor's own event carries a
        # skipped LIST, which is also truthy and is not an agent report.
        skipped = {t["node"] for t in result["trace"]
                   if t.get("data", {}).get("skipped") is True}
        assert skipped == {"inventory_agent", "production_agent", "supply_agent"}

    def test_unsupported_question_still_returns_a_clean_answer(self, graph):
        result = run_question("how do I trim a brisket fat cap",
                              thread_id="t-sop", graph=graph)
        assert result["plan"]["intent"] == "unsupported"
        assert result["recommendation"]["verdict"] == "informational"
        assert "SOP" in result["narration"] or "knowledge base" in result["narration"]
        assert result.get("interrupt") is None

    def test_narration_is_marked_as_template_when_offline(self, graph):
        result = run_question("what if we increase chuck roast production by 15%?",
                              thread_id="t-narr", graph=graph)
        assert result["recommendation"]["narrated_by"] == "template"
        assert result["narration"]


class TestVerdictRules:
    def test_capacity_overrun_blocks(self):
        v = _verdict({"labor": {"exceeds_cutter_capacity": True}})
        assert v["verdict"] == "do_not_proceed"
        assert v["blockers"]

    def test_high_but_feasible_utilization_is_a_caution(self):
        v = _verdict({
            "labor": {"exceeds_cutter_capacity": False,
                      "scenario_cutter_utilization_pct": MAX_SAFE_CUTTER_UTILIZATION + 0.02},
            "margin": {"delta_weekly_margin": 50.0},
        })
        assert v["verdict"] == "proceed_with_caution"
        assert v["cautions"]

    def test_negative_margin_blocks_even_with_spare_capacity(self):
        v = _verdict({
            "labor": {"exceeds_cutter_capacity": False, "scenario_cutter_utilization_pct": 0.3},
            "margin": {"delta_weekly_margin": -250.0},
        })
        assert v["verdict"] == "do_not_proceed"

    def test_cover_under_one_day_blocks(self):
        v = _verdict({
            "inventory": {"scenario_days_of_cover": 0.4},
            "margin": {"delta_weekly_margin": 100.0},
        })
        assert v["verdict"] == "do_not_proceed"

    def test_healthy_scenario_proceeds(self):
        v = _verdict({
            "labor": {"exceeds_cutter_capacity": False, "scenario_cutter_utilization_pct": 0.6},
            "inventory": {"scenario_days_of_cover": 2.4},
            "margin": {"delta_weekly_margin": 118.44},
        })
        assert v["verdict"] == "proceed"
        assert not v["blockers"] and not v["cautions"]


class TestHumanApproval:
    def test_actionable_scenario_interrupts_for_approval(self, graph):
        result = run_question("what if we increase chuck roast production by 15%?",
                              thread_id="t-approve", graph=graph)
        assert result["interrupt"]["kind"] == "approval_request"
        assert result["interrupt"]["verdict"] in ("proceed", "proceed_with_caution")

    def test_resume_records_approval(self, graph):
        run_question("what if we increase chuck roast production by 15%?",
                     thread_id="t-approve-2", graph=graph)
        final = resume_approval("t-approve-2", approved=True,
                                approved_by="shop-manager", graph=graph)
        assert final["approval"]["status"] == "approved"
        assert final["approval"]["approved_by"] == "shop-manager"

    def test_resume_records_rejection(self, graph):
        run_question("what if we increase chuck roast production by 15%?",
                     thread_id="t-reject", graph=graph)
        final = resume_approval("t-reject", approved=False, note="not this week", graph=graph)
        assert final["approval"]["status"] == "rejected"
        assert final["approval"]["note"] == "not this week"

    def test_informational_answers_do_not_ask_for_approval(self, graph):
        result = run_question("what were our top sellers last month",
                              thread_id="t-info", graph=graph)
        assert result.get("interrupt") is None
        assert result["approval"]["status"] == "not_required"
