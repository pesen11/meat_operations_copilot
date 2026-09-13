"""
Retrieval stage - deliberately WIDE.

CLAUDE.md step 6.2: optimise for recall, not top-k precision, because a
missed relevant chunk is unrecoverable while an irrelevant one is fixable by
the reranker downstream. So this stage fetches RETRIEVE_TOP_K (wide) and
hands all of it to rag/reranker.py, which cuts it to RERANK_TOP_N (narrow)
before anything reaches the generation prompt.

Two recall widenings beyond a plain top-k, both aimed at the same failure:

- **Neighbour expansion.** A procedure's steps are split across adjacent
  chunks; retrieving step 3 without step 4 produces a confidently incomplete
  answer. When a chunk scores well, its immediate neighbours in the same
  document are pulled in as candidates too.
- **Sibling-section expansion.** Chunks sharing a heading path with a strong
  hit are pulled in, so a table and the prose explaining it stay together.

Both only ADD candidates. Precision is the reranker's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rag.query_expansion import expand_query, expansion_terms
from rag.vectorstore.base import ScoredChunk, VectorStore, get_store

# Wide net. ~50 chunks in this corpus, so 14 is ~28% of it - proportionally
# generous, which is the intent. Scale this with corpus size, not linearly:
# at 5,000 chunks a top-k of 1,400 would be absurd, 40-60 is the usual shape.
RETRIEVE_TOP_K = 14
# What the model actually sees. Small and high-precision.
RERANK_TOP_N = 4
# Candidates below this hybrid score are not worth expanding around.
EXPANSION_SCORE_FLOOR = 0.25
# Expansion candidates inherit a fraction of their parent's score so they
# enter the reranker ranked below genuine hits rather than above them.
EXPANSION_SCORE_FACTOR = 0.45


@dataclass
class RetrievalResult:
    query: str
    candidates: list[ScoredChunk]
    stats: dict[str, Any]


class Retriever:
    def __init__(self, store: Optional[VectorStore] = None,
                 top_k: int = RETRIEVE_TOP_K) -> None:
        self.store = store or get_store()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 expand: bool = True, expand_terms: bool = True) -> RetrievalResult:
        k = top_k or self.top_k
        # Search with the expanded query, but everything downstream (reranking,
        # generation, caching) sees the operator's ORIGINAL wording. Expansion
        # is a retrieval aid; it must not leak into what the model is asked.
        search_query = expand_query(query) if expand_terms else query
        added_terms = expansion_terms(query) if expand_terms else []

        primary = self.store.search(search_query, top_k=k)
        seen = {s.chunk.chunk_id for s in primary}
        candidates = list(primary)

        added = self._expand(primary, seen) if expand else []
        candidates.extend(added)

        return RetrievalResult(
            query=query,
            candidates=candidates,
            stats={
                "retrieved": len(primary),
                "expanded": len(added),
                "expansion_terms": added_terms,
                "total_candidates": len(candidates),
                "top_score": round(primary[0].score, 4) if primary else 0.0,
            },
        )

    def _expand(self, primary: list[ScoredChunk], seen: set[str]) -> list[ScoredChunk]:
        all_chunks = self.store.all_chunks()
        by_id = {c.chunk_id: i for i, c in enumerate(all_chunks)}
        added: list[ScoredChunk] = []

        for hit in primary:
            if hit.score < EXPANSION_SCORE_FLOOR:
                continue
            idx = by_id.get(hit.chunk.chunk_id)
            if idx is None:
                continue

            neighbours = [idx - 1, idx + 1]
            for n in neighbours:
                if not (0 <= n < len(all_chunks)):
                    continue
                candidate = all_chunks[n]
                if candidate.doc_id != hit.chunk.doc_id or candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                added.append(ScoredChunk(
                    chunk=candidate,
                    score=hit.score * EXPANSION_SCORE_FACTOR,
                    components={"expansion": 1.0, "parent": hit.score},
                ))

            for candidate in all_chunks:
                if (candidate.doc_id == hit.chunk.doc_id
                        and candidate.heading_path == hit.chunk.heading_path
                        and candidate.chunk_id not in seen):
                    seen.add(candidate.chunk_id)
                    added.append(ScoredChunk(
                        chunk=candidate,
                        score=hit.score * EXPANSION_SCORE_FACTOR,
                        components={"sibling": 1.0, "parent": hit.score},
                    ))
        return added


_default: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _default
    if _default is None:
        _default = Retriever()
    return _default


def reset_retriever() -> None:
    """Drop the cached retriever - used by tests and after a reindex."""
    global _default
    _default = None
