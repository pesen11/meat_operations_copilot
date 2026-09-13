"""
FastAPI backend for the Meat-Cutting Operations Copilot.

    uvicorn api.main:app --reload

Three surfaces:

- `/simulate` and `/simulate/stream` run the LangGraph ops graph. The
  streaming variant emits one SSE event per graph node as it completes, so
  the frontend can show the Inventory / Production / Historical agents
  reporting in rather than a spinner. The three run in parallel, so events
  arrive interleaved - the `node` field, not arrival order, says who spoke.
- `/approve` resumes a run paused at the human-approval interrupt. State is
  held by the LangGraph checkpointer and keyed by thread_id.
- `/ask` is the SOP RAG endpoint. A declined answer returns HTTP 200 with
  `answered: false`, NOT an error: "the SOPs don't cover this" is a correct
  answer, and returning 404 would train a frontend to hide it.

Sync graph nodes run in a threadpool via `astream`, so a long tool call
cannot block the event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from agents import llm
from agents.graph import get_graph, resume_approval
from api.schemas import (
    ApprovalRequest, ApprovalResponse, AskRequest, AskResponse, HealthResponse,
    ProductOut, SimulateRequest, SimulateResponse, StockOut,
)

# Comma-separated origins, e.g. "http://localhost:3000,https://ops.example.com".
# Defaults to local Next.js dev only - a wildcard default would ship an open
# CORS policy to whoever deploys this first.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "OPS_COPILOT_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the expensive singletons once at startup rather than making the
    # first request pay for index load + graph compile.
    from rag.pipeline import get_pipeline

    await asyncio.to_thread(get_graph)
    await asyncio.to_thread(get_pipeline)
    yield


app = FastAPI(
    title="Meat-Cutting Operations Copilot",
    version="1.0.0",
    description=(
        "Operations intelligence for a butcher shop. All yield, labor, waste "
        "and margin figures are computed by deterministic, unit-tested Python; "
        "the LLM only routes questions and narrates results."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & reference data
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from data.catalog import products_by_sku
    from rag import tuning
    from rag.vectorstore.base import get_store
    from simulation import ops_data

    def _gather():
        try:
            store_stats: dict[str, Any] = get_store().stats()
        except Exception as exc:
            store_stats = {"error": str(exc)}
        params = tuning.current()
        return store_stats, {
            "source": params.source,
            "embedder": params.embedder,
            "reranker": params.reranker,
            "rerank_top_n": params.rerank_top_n,
            "semantic_cache_threshold": params.semantic_threshold,
        }, ("postgres" if ops_data._postgres_available() else "csv")

    store_stats, rag_params, backend = await asyncio.to_thread(_gather)

    # Name the checkpointer class actually compiled into the graph, not what
    # the environment implies it should be.
    try:
        saver = type(get_graph().checkpointer).__name__
    except Exception:
        saver = "unknown"

    return HealthResponse(
        status="ok",
        llm="live" if llm.is_live() else "offline",
        checkpointer=saver,
        vector_store=store_stats,
        data_backend=backend,
        rag_params=rag_params,
        catalog_products=len(products_by_sku),
    )


@app.get("/products", response_model=list[ProductOut])
async def products() -> list[ProductOut]:
    from simulation.tools import list_products

    rows = await asyncio.to_thread(list_products.invoke, {})
    return [ProductOut(**row) for row in rows]


@app.get("/primals", response_model=list[str])
async def primals() -> list[str]:
    from simulation.inventory_tools import list_primals

    return await asyncio.to_thread(list_primals.invoke, {})


@app.get("/stock", response_model=list[StockOut])
async def stock(as_of: Optional[str] = Query(default=None)) -> list[StockOut]:
    """Current stock position for every primal - the dashboard's inventory view."""
    from simulation.inventory_tools import get_all_stock_positions

    try:
        rows = await asyncio.to_thread(get_all_stock_positions.invoke, {"as_of": as_of})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [StockOut(**row) for row in rows]


@app.get("/history/weekly-revenue")
async def weekly_revenue(start: Optional[str] = None, end: Optional[str] = None):
    from simulation.history_tools import get_weekly_revenue_series

    try:
        return await asyncio.to_thread(
            get_weekly_revenue_series.invoke, {"start": start, "end": end})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/history/top-movers")
async def top_movers(limit: int = Query(default=5, ge=1, le=20),
                     by: str = Query(default="revenue", pattern="^(revenue|kg)$"),
                     end: Optional[str] = None):
    from simulation.history_tools import get_top_movers

    try:
        return await asyncio.to_thread(
            get_top_movers.invoke, {"limit": limit, "by": by, "end": end})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _to_response(thread_id: str, question: str, state: dict) -> SimulateResponse:
    from evals.claim_guard import guard_state as claim_guard_state
    from evals.number_guard import guard_state

    plan = state.get("plan") or {}
    rec = state.get("recommendation") or {}
    interrupt = state.get("interrupt")

    return SimulateResponse(
        thread_id=thread_id,
        question=question,
        plan={
            "intent": plan.get("intent", "unsupported"),
            "product_sku": plan.get("product_sku"),
            "source_primal": plan.get("source_primal"),
            "demand_multiplier": plan.get("demand_multiplier", 1.0),
            "reasoning": plan.get("reasoning", ""),
            "resolved_by": plan.get("resolved_by", "keyword"),
            "operator_supplied_a_size": plan.get("operator_supplied_a_size", True),
            "provenance": plan.get("provenance", {}),
            "required_evidence": plan.get("required_evidence", []),
        },
        projected_outcome=state.get("projected_outcome") or {},
        recommendation={
            "verdict": rec.get("verdict", "informational"),
            "blockers": rec.get("blockers", []),
            "cautions": rec.get("cautions", []),
            "narration": rec.get("narration", ""),
            "narrated_by": rec.get("narrated_by", "template"),
            "requires_approval": rec.get("requires_approval", False),
            "action": rec.get("action", "no_action"),
            "decision_score": rec.get("decision_score"),
            "confidence": rec.get("confidence", 0.0),
            "limiting_factor": rec.get("limiting_factor"),
            "factors": rec.get("factors", []),
            "options": rec.get("options", []),
            "chosen_option": rec.get("chosen_option"),
            "recommended_multiplier": rec.get("recommended_multiplier"),
            "recommended_change_pct": rec.get("recommended_change_pct"),
            "reasons": rec.get("reasons", []),
            "risks": rec.get("risks", []),
            "evidence_coverage": rec.get("evidence_coverage"),
            "evidence_gaps": rec.get("evidence_gaps", []),
        },
        narration=state.get("narration", ""),
        awaiting_approval=interrupt is not None,
        approval=state.get("approval"),
        trace=state.get("trace", []),
        errors=state.get("errors", []),
        number_guard=guard_state(state).to_dict(),
        claim_guard=claim_guard_state(state).to_dict(),
    )


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    from agents.graph import run_question
    from db import postgres

    thread_id = request.thread_id or f"t-{uuid.uuid4().hex[:12]}"
    try:
        state = await asyncio.to_thread(
            run_question, request.question, thread_id, request.as_of)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    response = _to_response(thread_id, request.question, state)
    await asyncio.to_thread(
        postgres.log_decision, thread_id, request.question,
        dict(state.get("plan") or {}), state.get("projected_outcome") or {},
        response.recommendation.verdict, response.narration,
        response.recommendation.narrated_by)
    return response


def _sse(event: str, data: Any) -> dict:
    return {"event": event, "data": json.dumps(data, default=str)}


_STREAM_SENTINEL = object()


def _run_graph_into_queue(question: str, thread_id: str, as_of: Optional[str],
                          queue: "asyncio.Queue", loop) -> None:
    """
    Drive the SYNCHRONOUS graph stream on a worker thread, pushing each update
    onto an asyncio queue.

    Why not `graph.astream()`, which would be the obvious choice: the
    human-approval node calls LangGraph's interrupt(), which reads the run
    config from a contextvar via get_config(). On Python < 3.11 get_config()
    deliberately raises when called from inside a running event loop, so the
    async path fails at precisely the approval step - the whole point of the
    graph. Running the sync stream in its own thread keeps interrupt() in a
    plain synchronous context where it works, and the queue bridge keeps the
    event loop free to flush SSE frames as they arrive.

    `stream_mode="updates"` yields {node_name: state_delta} per completed
    node, which is the granularity the progress UI wants without shipping the
    whole accumulated state on every tick.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    def put(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        for update in graph.stream({"question": question, "as_of": as_of},
                                   config=config, stream_mode="updates"):
            put(update)
    except Exception as exc:
        put({"__error__": f"{type(exc).__name__}: {exc}"})
    finally:
        put(_STREAM_SENTINEL)


async def _stream_graph(question: str, thread_id: str,
                        as_of: Optional[str]) -> AsyncIterator[dict]:
    """Emit one SSE event per completed graph node, then a final result."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    yield _sse("start", {"thread_id": thread_id, "question": question})

    worker = loop.run_in_executor(
        None, _run_graph_into_queue, question, thread_id, as_of, queue, loop)

    final_state: dict[str, Any] = {"trace": [], "errors": []}
    interrupt_payload = None
    failed = False

    while True:
        update = await queue.get()
        if update is _STREAM_SENTINEL:
            break
        if "__error__" in update:
            failed = True
            yield _sse("error", {"node": "graph", "errors": [update["__error__"]]})
            continue

        for node, delta in (update or {}).items():
            if node == "__interrupt__":
                first = delta[0] if isinstance(delta, (list, tuple)) and delta else delta
                interrupt_payload = getattr(first, "value", first)
                continue
            if not isinstance(delta, dict):
                continue
            for event in delta.get("trace", []) or []:
                yield _sse("progress", event)
            if delta.get("errors"):
                yield _sse("error", {"node": node, "errors": delta["errors"]})
            final_state.update({k: v for k, v in delta.items()
                                if k not in ("trace", "errors")})
            final_state["trace"].extend(delta.get("trace", []) or [])
            final_state["errors"].extend(delta.get("errors", []) or [])

    await worker

    if failed:
        yield _sse("done", {"thread_id": thread_id, "failed": True})
        return

    if interrupt_payload is not None:
        final_state["interrupt"] = interrupt_payload
        yield _sse("approval_required", interrupt_payload)

    response = _to_response(thread_id, question, final_state)
    yield _sse("result", response.model_dump())
    yield _sse("done", {"thread_id": thread_id,
                        "awaiting_approval": response.awaiting_approval})


@app.post("/simulate/stream")
async def simulate_stream(request: SimulateRequest):
    """SSE variant of /simulate with live per-agent progress."""
    thread_id = request.thread_id or f"t-{uuid.uuid4().hex[:12]}"
    return EventSourceResponse(
        _stream_graph(request.question, thread_id, request.as_of))


@app.get("/simulate/stream")
async def simulate_stream_get(question: str = Query(min_length=3, max_length=1000),
                              thread_id: Optional[str] = None,
                              as_of: Optional[str] = None):
    """GET variant, because the browser EventSource API cannot POST."""
    return EventSourceResponse(
        _stream_graph(question, thread_id or f"t-{uuid.uuid4().hex[:12]}", as_of))


@app.post("/approve", response_model=ApprovalResponse)
async def approve(request: ApprovalRequest) -> ApprovalResponse:
    from db import postgres

    try:
        state = await asyncio.to_thread(
            resume_approval, request.thread_id, request.approved,
            request.approved_by, request.note)
    except Exception as exc:
        # The usual cause is a thread_id with no paused run - either it was
        # never started, it already completed, or the process restarted and
        # the in-memory checkpointer lost it.
        raise HTTPException(
            status_code=404,
            detail=f"No run awaiting approval for thread '{request.thread_id}' "
                   f"({type(exc).__name__}: {exc})") from exc

    approval = state.get("approval") or {}
    await asyncio.to_thread(
        postgres.log_approval, request.thread_id,
        approval.get("status", "unknown"), request.approved_by, request.note)
    return ApprovalResponse(thread_id=request.thread_id, approval=approval,
                            trace=state.get("trace", []))


# ---------------------------------------------------------------------------
# SOP knowledge base
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Answer a procedural question from the SOP corpus.

    A question the SOPs do not cover returns 200 with `answered: false` and
    an explanation. That is the designed behaviour, not a failure.
    """
    from rag.pipeline import get_pipeline

    def _run():
        return get_pipeline().answer(request.query, use_cache=request.use_cache)

    try:
        response = await asyncio.to_thread(_run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return AskResponse(**response.to_dict())


@app.get("/ask/cache-stats")
async def cache_stats():
    from rag.cache import get_cache

    cache = await asyncio.to_thread(get_cache)
    return {**cache.stats.to_dict(), "entries": cache.size()}
