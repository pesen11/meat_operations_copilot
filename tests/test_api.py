"""
API tests.

Run against the real app with TestClient (which runs the lifespan, so the
index and graph are warmed exactly as in production), in offline LLM mode.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def offline_env():
    import os
    previous = os.environ.get("OPS_COPILOT_LLM")
    os.environ["OPS_COPILOT_LLM"] = "stub"
    yield
    if previous is None:
        os.environ.pop("OPS_COPILOT_LLM", None)
    else:
        os.environ["OPS_COPILOT_LLM"] = previous


@pytest.fixture(scope="module")
def client(offline_env):
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_reports_every_subsystem(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["llm"] in ("live", "offline")
        assert body["data_backend"] in ("csv", "postgres")
        assert body["vector_store"]["chunks"] > 0
        assert body["catalog_products"] == 19

    def test_health_exposes_whether_rag_params_are_tuned(self, client):
        body = client.get("/health").json()
        assert body["rag_params"]["source"] in ("tuned", "default")


class TestReferenceData:
    def test_products(self, client):
        rows = client.get("/products").json()
        assert len(rows) == 19
        assert {r["data_source"] for r in rows} <= {"real", "synthetic"}

    def test_primals_are_sorted(self, client):
        rows = client.get("/primals").json()
        assert rows == sorted(rows)

    def test_stock_covers_every_primal(self, client):
        primals = client.get("/primals").json()
        stock = client.get("/stock").json()
        assert {s["source_primal"] for s in stock} == set(primals)

    def test_stock_rejects_a_bad_date(self, client):
        assert client.get("/stock", params={"as_of": "not-a-date"}).status_code == 400

    def test_top_movers_validates_sort_key(self, client):
        assert client.get("/history/top-movers", params={"by": "margin"}).status_code == 422

    def test_weekly_revenue_is_chronological(self, client):
        rows = client.get("/history/weekly-revenue",
                          params={"start": "2025-10-01", "end": "2025-12-31"}).json()
        weeks = [r["week_ending"] for r in rows]
        assert weeks == sorted(weeks)


class TestSimulate:
    def test_scenario_returns_all_impact_dimensions(self, client):
        body = client.post("/simulate", json={
            "question": "what if we increase chuck roast production by 15%?",
            "thread_id": "test-sim-1"}).json()
        outcome = body["projected_outcome"]
        assert body["plan"]["product_sku"] == "chuck-roast"
        assert outcome["inventory"] and outcome["labor"] and outcome["waste"]
        assert body["recommendation"]["verdict"] in (
            "proceed", "proceed_with_caution", "do_not_proceed", "informational")

    def test_narration_passes_the_number_guard(self, client):
        """The project's core invariant, enforced on every API response."""
        body = client.post("/simulate", json={
            "question": "what if we increase chuck roast production by 15%?",
            "thread_id": "test-sim-2"}).json()
        guard = body["number_guard"]
        assert guard["passed"], guard["violations"]
        assert guard["numbers_checked"] > 0

    def test_thread_id_is_generated_when_omitted(self, client):
        body = client.post("/simulate", json={
            "question": "what were our top sellers last month"}).json()
        assert body["thread_id"].startswith("t-")

    def test_short_question_is_rejected(self, client):
        assert client.post("/simulate", json={"question": "hi"}).status_code == 422

    def test_sop_question_is_not_answered_by_the_simulation(self, client):
        body = client.post("/simulate", json={
            "question": "how do I trim a brisket fat cap",
            "thread_id": "test-sim-3"}).json()
        assert body["plan"]["intent"] == "unsupported"
        assert body["awaiting_approval"] is False


class TestApproval:
    def test_full_approve_cycle(self, client):
        first = client.post("/simulate", json={
            "question": "what if we increase chuck roast production by 15%?",
            "thread_id": "test-approve-1"}).json()
        assert first["awaiting_approval"] is True

        body = client.post("/approve", json={
            "thread_id": "test-approve-1", "approved": True,
            "approved_by": "shop-manager"}).json()
        assert body["approval"]["status"] == "approved"
        assert body["approval"]["approved_by"] == "shop-manager"

    def test_rejection_is_recorded(self, client):
        client.post("/simulate", json={
            "question": "what if we increase chuck roast production by 15%?",
            "thread_id": "test-approve-2"})
        body = client.post("/approve", json={
            "thread_id": "test-approve-2", "approved": False,
            "note": "not this week"}).json()
        assert body["approval"]["status"] == "rejected"
        assert body["approval"]["note"] == "not this week"

    def test_unknown_thread_returns_404(self, client):
        response = client.post("/approve", json={
            "thread_id": "never-started", "approved": True})
        assert response.status_code == 404


class TestStreaming:
    def _events(self, client, question: str, thread_id: str):
        events, current = [], None
        with client.stream("GET", "/simulate/stream",
                           params={"question": question, "thread_id": thread_id}) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("event:"):
                    current = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and current:
                    events.append((current, json.loads(line.split(":", 1)[1].strip())))
        return events

    def test_stream_reports_each_agent(self, client):
        events = self._events(
            client, "what if we increase chuck roast production by 15%?", "test-sse-1")
        names = [name for name, _ in events]
        assert names[0] == "start"
        assert names[-1] == "done"

        nodes = [payload["node"] for name, payload in events if name == "progress"]
        for node in ("manager", "supervisor", "inventory_agent",
                     "production_agent", "historical_agent", "recommendation"):
            assert node in nodes, nodes

    def test_stream_pauses_for_approval_and_can_be_resumed(self, client):
        """Regression guard: interrupt() must survive the sync-stream bridge.
        Driving the graph with astream on Python 3.10 breaks exactly here."""
        events = self._events(
            client, "what if we increase chuck roast production by 15%?", "test-sse-2")
        names = [name for name, _ in events]
        assert "approval_required" in names
        assert "error" not in names, [p for n, p in events if n == "error"]

        body = client.post("/approve", json={
            "thread_id": "test-sse-2", "approved": True, "approved_by": "lead"}).json()
        assert body["approval"]["status"] == "approved"

    def test_stream_emits_a_full_result(self, client):
        events = self._events(client, "what were our top sellers last month",
                              "test-sse-3")
        results = [payload for name, payload in events if name == "result"]
        assert len(results) == 1
        assert results[0]["thread_id"] == "test-sse-3"
        assert results[0]["narration"]


class TestAsk:
    def test_in_corpus_question_is_answered_with_sources(self, client):
        body = client.post("/ask", json={
            "query": "How do I clean the grinder after use?"}).json()
        assert body["answered"] is True
        assert body["sources"]
        assert all(s["data_source"] in ("real", "synthetic") for s in body["sources"])

    def test_out_of_corpus_question_declines_with_200_not_an_error(self, client):
        """A decline is a correct answer. Returning an error status would
        train the frontend to hide it."""
        response = client.post("/ask", json={
            "query": "What temperature should I sous vide a striploin steak to?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answered"] is False
        assert "SOP" in body["answer"] or "can't answer" in body["answer"]

    def test_cache_reports_a_hit_on_a_repeat(self, client):
        payload = {"query": "What chlorine concentration should the sanitizer be at?"}
        client.post("/ask", json=payload)
        second = client.post("/ask", json=payload).json()
        assert second["cache"] in ("exact", "semantic")

    def test_cache_stats_expose_the_threshold_source(self, client):
        body = client.get("/ask/cache-stats").json()
        assert "semantic_threshold" in body
        assert body["threshold_source"] in ("tuned", "explicit", "untuned_default")

    def test_short_query_is_rejected(self, client):
        assert client.post("/ask", json={"query": "x"}).status_code == 422
