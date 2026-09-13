"""
The questions an operator actually asks, end to end.

Every case here was reported as broken: either routed to "unsupported", or
routed correctly and then answered with a single unrelated top-seller line
because the data the agent fetched never reached the narration. These are
therefore coverage tests, not unit tests - they assert that a plain question
comes back with the thing it asked for in it.

Offline (stub) narration is deliberate: it is the template path, so a
failure here is a wiring failure, never a model wording change.
"""

from __future__ import annotations

import pytest

from agents.graph import build_graph, run_question
from agents.intent import keyword_plan
from langgraph.checkpoint.memory import InMemorySaver


@pytest.fixture(scope="module")
def graph():
    return build_graph(checkpointer=InMemorySaver())


def answer(graph, question: str, thread: str) -> tuple[str, dict]:
    state = run_question(question, thread_id=thread, graph=graph)
    return (state.get("narration") or ""), (state.get("projected_outcome") or {})


class TestNothingOperationalIsUnsupported:
    """Every one of these used to land in 'unsupported'."""

    @pytest.mark.parametrize("question,expected", [
        ("how was the ribeye sales last week", "history"),
        ("what were our best sellers last month?", "history"),
        ("how were the sales last day?", "history"),
        ("how did we do yesterday?", "history"),
        ("how were the sales last weekend?", "history"),
        ("how much production should we increase during weekends?", "history"),
        ("which items are doing poorly?", "history"),
        ("what should we order less of?", "history"),
        ("what items do i need to restock soon?", "inventory_status"),
        ("how many boxes of ribeye do we have?", "inventory_status"),
        ("how many kgs of chuck roast do we have?", "inventory_status"),
        ("what if we increase chuck roast production by 15%?", "scenario"),
    ])
    def test_routes_to_an_operational_intent(self, question, expected):
        assert keyword_plan(question)["intent"] == expected

    def test_procedural_questions_still_go_to_the_sop_layer(self):
        """The RAG hand-off must not be collateral damage of widening intents."""
        for q in ("what temperature should the cooler be?",
                  "how do i trim a brisket fat cap?",
                  "what is the cleaning procedure for the saw?"):
            assert keyword_plan(q)["intent"] == "unsupported"


class TestTimeWindowsReachTheAnswer:
    def test_the_window_asked_for_is_the_window_reported(self, graph):
        narration, outcome = answer(graph, "how was the ribeye sales last week", "w1")
        assert outcome["window"]["days"] == 7
        assert outcome["window"]["label"] in narration

    def test_different_windows_give_different_answers(self, graph):
        _, week = answer(graph, "how was ribeye last week", "w2")
        _, month = answer(graph, "how was ribeye last month", "w3")
        assert week["sales_summary"]["total_kg"] != month["sales_summary"]["total_kg"]

    def test_yesterday_is_a_single_day(self, graph):
        _, outcome = answer(graph, "how did we do yesterday?", "w4")
        assert outcome["window"]["days"] == 1


class TestHistoryAnswersTheQuestionAsked:
    def test_a_product_question_reports_that_product(self, graph):
        narration, outcome = answer(graph, "how was the ribeye sales last week", "h1")
        assert "Ribeye" in narration
        assert outcome["sales_summary"]["sku"] == "ribeye-steak-bnls"
        assert str(outcome["sales_summary"]["total_kg"]) in narration

    def test_best_sellers_returns_a_ranking_not_one_name(self, graph):
        """
        The reported bug: this returned only "Thinly Sliced Blades" because
        the narration printed top_movers[0] and nothing else.
        """
        narration, outcome = answer(graph, "what were our best sellers last month?", "h2")
        assert len(outcome["top_movers"]) >= 5
        named = sum(1 for row in outcome["top_movers"]
                    if row["product_name"] in narration)
        assert named >= 3, narration

    def test_poor_performers_are_named(self, graph):
        narration, outcome = answer(graph, "which items are doing poorly?", "h3")
        assert outcome["slow_movers"]
        assert any(row["product_name"] in narration for row in outcome["slow_movers"])

    def test_weekend_question_returns_a_measured_uplift(self, graph):
        narration, outcome = answer(
            graph, "how much production should we increase during weekends?", "h4")
        uplift = outcome["weekend_uplift"]
        assert uplift["implied_multiplier"] > 1.0
        assert str(uplift["implied_multiplier"]) in narration


class TestRecommendations:
    def test_catalog_performance_carries_a_verdict_per_product(self, graph):
        _, outcome = answer(graph, "what should we order less of?", "r1")
        assert outcome["catalog_performance"]
        assert all("verdict" in row for row in outcome["catalog_performance"])

    def test_a_one_day_window_never_recommends_an_order_change(self, graph):
        """
        Daily sales carry 12% noise. "How did we do yesterday" is a fine
        question; answering it with "so order less short ribs" is advice
        built on noise.
        """
        narration, outcome = answer(graph, "how did we do yesterday?", "r2")
        assert not outcome.get("reduce_products")
        assert not outcome.get("focus_products")
        assert "order less" not in narration.lower()

    def test_a_full_window_can_recommend(self, graph):
        _, outcome = answer(graph, "which products should we focus on this month?", "r3")
        verdicts = {row["verdict"] for row in outcome["catalog_performance"]}
        assert verdicts & {"focus", "reduce", "watch"}


class TestInventoryQuestions:
    def test_boxes_on_hand_for_a_named_product(self, graph):
        narration, outcome = answer(graph, "how many boxes of ribeye do we have?", "i1")
        focus = outcome["focus_primal"]
        assert focus["source_primal"] == "Bone-in Ribeye"
        assert str(focus["boxes_on_hand"]) in narration
        assert str(focus["kg_on_hand"]) in narration

    def test_restock_question_ranks_by_days_of_cover(self, graph):
        narration, outcome = answer(graph, "what items do i need to restock soon?", "i2")
        priority = outcome["restock_priority"]
        assert priority == sorted(priority, key=lambda r: r["days_of_cover"])
        assert priority[0]["source_primal"] in narration


class TestHolidayPreparation:
    def test_upcoming_holiday_plans_and_cites_last_time(self, graph):
        narration, outcome = answer(
            graph,
            "For the upcoming holiday how should we prepare by taking into "
            "account the history of last holiday sales",
            "s1")
        # Planned forward from a confirmed multiplier...
        assert outcome["seasonal"]["multiplier"] > 1.0
        assert outcome["seasonal"]["top_primals_to_order"]
        # ...and grounded in what actually happened last time.
        actuals = outcome["last_period_actuals"]
        assert actuals["total_kg"] > 0
        assert actuals["top_products"]
        assert str(actuals["total_kg"]) in narration

    def test_observed_and_confirmed_multipliers_are_reported_separately(self, graph):
        """
        The observed figure is history and the confirmed figure is a business
        input. Blending them would present an assumption as a measurement.
        """
        _, outcome = answer(graph, "how should we prepare for christmas?", "s2")
        actuals = outcome["last_period_actuals"]
        assert "observed_multiplier" in actuals
        assert "confirmed_multiplier" in actuals


class TestWholeCatalogSalesTotals:
    """
    "What were our sales yesterday" asks how much the shop sold. It used to
    return only rankings - which cuts led - because `sales_summary` requires a
    SKU and nothing computed a catalog-wide total when no product was named.
    """

    def test_sales_question_with_no_product_reports_a_total(self, graph):
        narration, outcome = answer(graph, "What were our sales yesterday?", "ct1")
        totals = outcome["catalog_sales_totals"]
        assert totals["total_revenue"] > 0
        assert totals["products_sold"] > 1
        # The total must actually reach the operator, not just the outcome -
        # this project has shipped "computed then dropped" twice.
        assert f"{totals['total_revenue']:,.2f}" in narration
        assert str(totals["total_kg"]) in narration

    def test_total_is_not_just_the_top_seller(self, graph):
        """A ranking answered a different question than the one asked."""
        _, outcome = answer(graph, "How much did we sell last week?", "ct2")
        totals = outcome["catalog_sales_totals"]
        top = outcome["top_movers"][0]
        assert totals["total_revenue"] > top["total_revenue"]
        assert totals["total_kg"] > top["total_kg"]

    def test_window_is_respected_and_stated(self, graph):
        narration, outcome = answer(graph, "What were our sales yesterday?", "ct3")
        assert outcome["window"]["days"] == 1
        assert outcome["catalog_sales_totals"]["open_days"] <= 1
        assert "yesterday" in narration

    def test_scenario_question_keeps_production_catalog_totals(self, graph):
        """
        The production agent publishes its own `catalog_totals` (extra volume
        for a scenario). The two must not overwrite each other.
        """
        _, outcome = answer(
            graph, "How much production increase should we do during Christmas?", "ct4")
        assert "total_extra_weekly_product_kg" in outcome["catalog_totals"]
