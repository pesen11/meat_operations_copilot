"""
The LangGraph graph:

    manager -> supervisor -> {inventory, production, historical, supply}
            -> yield_analyzer -> evidence_check
                                   |            |
                        gaps remain |            | sufficient
                                   v            v
                        gather_supplementary   recommendation -> human_approval -> END
                                   |
                                   +----> yield_analyzer (one pass only)

The specialist nodes fan out in parallel: LangGraph runs every node reachable
from supervisor in the same superstep, and yield_analyzer waits for all of
them because it has an incoming edge from each. That is why OpsState.trace
and .errors carry operator.add reducers - several nodes write them in the
same superstep.

DISPATCH IS CONDITIONAL, AND SKIPPED AGENTS STILL REPORT
--------------------------------------------------------
An earlier version of this graph ran all three specialists unconditionally
and had idle ones return {"skipped": True}, deliberately, so the SSE progress
stream showed three agents reporting in. That rationale was about the STREAM
rather than about the work, and it is satisfiable without doing the work: the
supervisor now emits a trace event for every agent it declines to dispatch,
naming why, and only the agents that can supply required evidence actually
run. The stream ends up strictly more informative than before, because it now
says why an agent sat out.

THE SUFFICIENCY LOOP IS CAPPED AT ONE PASS
-------------------------------------------
planner.MAX_GATHER_ATTEMPTS is 1. A gap that survives one supplementary fetch
means the data does not exist, not that the fetch was unlucky, and a graph
that can loop is a graph that can hang in front of an operator.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    evidence_check, gather_supplementary, historical_agent, human_approval,
    inventory_agent, manager, production_agent, recommendation,
    route_after_evidence_check, supervisor, supply_agent, yield_analyzer,
)
from agents.state import OpsState

# Graph node name per branch key used by the supervisor and the planner.
BRANCH_NODES: dict[str, str] = {
    "inventory": "inventory_agent",
    "production": "production_agent",
    "historical": "historical_agent",
    "supply": "supply_agent",
}


def route_from_supervisor(state: OpsState) -> list[str]:
    """
    Fan out to exactly the agents this question needs.

    Returning a list is LangGraph's conditional fan-out: every named node runs
    in the same superstep. An empty branch list (an unsupported question)
    routes straight to the join, because a graph that dispatches nothing must
    still reach an answer rather than stalling.
    """
    branches = state.get("branches") or []
    nodes = [BRANCH_NODES[b] for b in branches if b in BRANCH_NODES]
    return nodes or ["yield_analyzer"]


def build_graph(checkpointer: Any = None):
    """Compile the ops graph. A checkpointer is REQUIRED for interrupt() to
    work - without one there is nowhere to persist the paused run."""
    builder = StateGraph(OpsState)

    builder.add_node("manager", manager)
    builder.add_node("supervisor", supervisor)
    builder.add_node("inventory_agent", inventory_agent)
    builder.add_node("production_agent", production_agent)
    builder.add_node("historical_agent", historical_agent)
    builder.add_node("supply_agent", supply_agent)
    builder.add_node("yield_analyzer", yield_analyzer)
    builder.add_node("evidence_check", evidence_check)
    builder.add_node("gather_supplementary", gather_supplementary)
    builder.add_node("recommendation", recommendation)
    builder.add_node("human_approval", human_approval)

    builder.add_edge(START, "manager")
    builder.add_edge("manager", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        [*BRANCH_NODES.values(), "yield_analyzer"],
    )
    for node in BRANCH_NODES.values():
        builder.add_edge(node, "yield_analyzer")

    builder.add_edge("yield_analyzer", "evidence_check")
    builder.add_conditional_edges(
        "evidence_check",
        route_after_evidence_check,
        ["gather_supplementary", "recommendation"],
    )
    builder.add_edge("gather_supplementary", "yield_analyzer")

    builder.add_edge("recommendation", "human_approval")
    builder.add_edge("human_approval", END)

    return builder.compile(checkpointer=checkpointer or default_checkpointer())


def default_checkpointer():
    """Postgres-backed when DATABASE_URL is set and langgraph-checkpoint-postgres
    is installed; in-memory otherwise. In-memory means sessions do not survive
    a restart - fine for local use, not for a deployed backend."""
    url = os.getenv("DATABASE_URL")
    if url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415
            saver = PostgresSaver.from_conn_string(url).__enter__()
            saver.setup()
            return saver
        except Exception:
            pass  # fall through to memory rather than failing to start
    from langgraph.checkpoint.memory import InMemorySaver
    return InMemorySaver()


_compiled = None


def get_graph():
    """Process-wide compiled graph (compilation is not free, and the FastAPI
    layer needs the same checkpointer across requests to resume interrupts)."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_question(question: str, thread_id: str = "local", as_of: Optional[str] = None,
                 graph=None) -> dict:
    """
    Run one question to completion or to the approval interrupt.

    Returns the final state plus an `interrupt` key when the graph paused for
    human approval - resume it with resume_approval().
    """
    graph = graph or get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"question": question, "as_of": as_of}, config=config)

    pending = result.get("__interrupt__")
    if pending:
        payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
        result = {**{k: v for k, v in result.items() if k != "__interrupt__"},
                  "interrupt": payload}
    return result


def resume_approval(thread_id: str, approved: bool, approved_by: Optional[str] = None,
                    note: Optional[str] = None, graph=None) -> dict:
    """Resume a run paused at human_approval with the operator's decision."""
    from langgraph.types import Command

    graph = graph or get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        Command(resume={"approved": approved, "approved_by": approved_by, "note": note}),
        config=config,
    )
