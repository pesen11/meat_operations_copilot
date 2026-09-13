"""
The RAG pipeline: cache -> retrieve (wide) -> rerank (narrow) -> generate ->
groundedness guardrail.

This is the only module the API and the agent layer import. Everything else
in rag/ is a stage it composes.

Ordering note: the cache is checked BEFORE retrieval, so a hit costs one
embedding rather than a retrieval plus two model calls. Cache writes happen
only for answers that pass the groundedness guardrail - caching a
hallucination would serve it repeatedly and make the failure look like a
systematic problem rather than a one-off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rag import groundedness, tuning
from rag.cache import QueryCache, get_cache
from rag.generate import GeneratedAnswer, build_context, generate
from rag.reranker import Reranker, RerankedChunk, get_reranker
from rag.retriever import RERANK_TOP_N, Retriever, get_retriever


@dataclass
class RagResponse:
    query: str
    answer: str
    answered: bool
    sources: list[dict] = field(default_factory=list)
    cache: str = "miss"                  # exact | semantic | miss
    latency_ms: float = 0.0
    groundedness: Optional[dict] = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "answered": self.answered,
            "sources": self.sources,
            "cache": self.cache,
            "latency_ms": round(self.latency_ms, 1),
            "groundedness": self.groundedness,
            "stats": self.stats,
        }


def _sources(chunks: list[RerankedChunk], cited: list[str]) -> list[dict]:
    out = []
    for r in chunks:
        c = r.scored.chunk
        out.append({
            "chunk_id": c.chunk_id,
            "document": c.doc_title,
            "doc_id": c.doc_id,
            "heading": c.heading,
            "data_source": c.data_source,
            "relevance": round(r.relevance, 3),
            "cited": c.chunk_id in cited,
        })
    return out


class RagPipeline:
    def __init__(self, retriever: Optional[Retriever] = None,
                 reranker: Optional[Reranker] = None,
                 cache: Optional[QueryCache] = None,
                 top_n: Optional[int] = None,
                 use_cache: bool = True,
                 deep_groundedness: bool = False) -> None:
        self.retriever = retriever or get_retriever()
        self.reranker = reranker or get_reranker()
        self.cache = cache or get_cache()
        # top_n is swept by rag/eval/tune.py; RERANK_TOP_N is only the
        # pre-tuning default.
        self.top_n = top_n if top_n is not None else tuning.current().rerank_top_n
        self.use_cache = use_cache
        self.deep_groundedness = deep_groundedness

    def answer(self, query: str, *, use_cache: Optional[bool] = None) -> RagResponse:
        start = time.perf_counter()
        caching = self.use_cache if use_cache is None else use_cache

        if caching:
            cached, how = self.cache.get(query)
            if cached is not None:
                response = RagResponse(**{**cached, "query": query})
                response.cache = how
                response.latency_ms = (time.perf_counter() - start) * 1000
                return response

        retrieval = self.retriever.retrieve(query)
        reranked = self.reranker.rerank(query, retrieval.candidates, self.top_n)
        result: GeneratedAnswer = generate(query, reranked)

        context = build_context(reranked)
        ground = (groundedness.check(result.answer, context, deep=self.deep_groundedness)
                  if result.answered else None)

        answer_text = result.answer
        answered = result.answered
        if ground is not None and not ground.numeric_ok:
            # A number that is not in the source is the one failure mode this
            # project refuses to ship. Withhold rather than serve it.
            answer_text = (
                "I found relevant procedures but could not verify every figure in "
                "the drafted answer against the source text, so I'm not showing it. "
                "The source sections are listed below - read them directly."
            )
            answered = False

        response = RagResponse(
            query=query,
            answer=answer_text,
            answered=answered,
            sources=_sources(reranked, result.cited_chunk_ids),
            cache="miss",
            groundedness=ground.to_dict() if ground else None,
            stats={
                **retrieval.stats,
                "reranked": len(reranked),
                "reranker": getattr(self.reranker, "name", "unknown"),
                "generated_by": result.generated_by,
            },
        )
        response.latency_ms = (time.perf_counter() - start) * 1000

        # Only cache answers that passed the guardrail.
        if caching and (ground is None or ground.numeric_ok):
            payload = response.to_dict()
            payload.pop("query", None)
            payload.pop("latency_ms", None)
            payload.pop("cache", None)
            self.cache.set(query, payload)

        return response

    def retrieve_only(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Retrieval without generation - used by the eval harness to measure
        recall@k independently of whatever the generator does with it."""
        return [s.to_dict() for s in self.retriever.retrieve(query, top_k=top_k).candidates]


_pipeline: Optional[RagPipeline] = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None


def answer(query: str) -> RagResponse:
    return get_pipeline().answer(query)
