"""
Reranking / filtering stage.

CLAUDE.md step 6.3: the model must only see good chunks. Retrieval is wide
and noisy by design; this stage scores relevance properly and truncates to a
small set before anything reaches the generation prompt.

Two rerankers, same interface:

- `LLMReranker` (default when credentials exist): Claude Haiku 4.5 scores
  every candidate's relevance to the query in ONE batched call, returning
  structured JSON. Haiku rather than Opus because this is a bounded,
  well-specified judgement made once per candidate - the latency and cost
  budget (CLAUDE.md step 6.5) does not justify a frontier model for it.

- `LexicalReranker` (offline default): a cross-encoder-style scorer that
  looks at query/chunk interaction rather than a bag of words - term
  coverage, phrase proximity, and heading agreement. Weaker than an LLM, but
  it runs in CI for free and meaningfully beats "just take the retriever's
  top-n", which is what the eval numbers show.

Why an LLM filter rather than a hosted cross-encoder model: adding a
sentence-transformers dependency to get a cross-encoder means a torch
install, a model download, and a second inference runtime in a project that
otherwise has one. Claude is already a dependency, and Haiku's latency at
this candidate count is comparable.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Protocol

from agents import llm
from rag.embeddings import tokenize
from rag.retriever import RERANK_TOP_N
from rag.vectorstore.base import ScoredChunk

# Truncation is governed by TWO floors, both tuned in rag/tuning.py:
#
#   min_relevance  - absolute sanity bar. Measured on this corpus, gold and
#                    non-gold lexical scores overlap heavily (gold minimum
#                    0.35, non-gold median 0.40), so an absolute floor CANNOT
#                    separate them. It is set low and only rejects noise.
#   relative_floor - keep chunks scoring at least this fraction of the best
#                    chunk. This is what controls precision, and unlike an
#                    absolute floor it can never discard the best chunk, so a
#                    good match is never dropped to an empty result.
#
# Deciding "nothing here answers the question" is a separate judgement made in
# rag/generate.py, not by thresholding this score.
MIN_RELEVANCE = 0.20

# Truncation sent to the reranker per chunk. Enough to judge relevance
# without paying for the whole corpus on every query.
RERANK_CHUNK_PREVIEW_CHARS = 700


@dataclass
class RerankedChunk:
    chunk_id: str
    relevance: float
    reason: str
    scored: ScoredChunk

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "relevance": round(self.relevance, 3),
                "reason": self.reason, "heading": self.scored.chunk.heading,
                "doc_id": self.scored.chunk.doc_id}


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk],
               top_n: int) -> list[RerankedChunk]: ...


# ---------------------------------------------------------------------------
# Lexical (offline) reranker
# ---------------------------------------------------------------------------

class LexicalReranker:
    """
    Query-chunk interaction scoring, no model required.

    Three signals, all genuinely cross-encoder-ish in that they look at the
    pair rather than at each side independently:
      - coverage:  how much of the query's INFORMATION the chunk covers
      - proximity: do the matched terms appear near each other
      - heading:   does the section heading agree with the query

    Coverage is IDF-weighted, not a plain term count. Unweighted coverage
    punishes long questions: "how do I know at opening whether we have enough
    primal to get through the day" has nine content words, so the chunk that
    genuinely answers it scores 4/9 and falls below the relevance floor,
    while a short question's mediocre match scores 2/3 and survives.

    The IDF is computed over the CORPUS, not over the candidate set. Using
    candidate-local IDF is tempting and wrong: when retrieval does its job on
    a broad query it returns a tight topical cluster, so the topic words
    ("safety", "food", "hygiene") appear in nearly every candidate, their
    local IDF collapses toward zero, and an off-topic chunk that happens to
    contain one rare query word outranks the chunks that actually answer the
    question. That is a measured failure, not a hypothetical - it put
    "Quality control sign-off" above "Cross-contamination control" for "what
    safety protocols should I follow?". Corpus IDF is what BM25 uses, for
    this reason.
    """

    name = "lexical"

    W_COVERAGE = 0.55
    W_PROXIMITY = 0.20
    W_HEADING = 0.25

    def rerank(self, query: str, candidates: list[ScoredChunk],
               top_n: int = RERANK_TOP_N) -> list[RerankedChunk]:
        # Score the EXPANDED query, exactly as the retriever did. This stage
        # is a lexical scorer, so it needs the same lexical aid; scoring the
        # raw query left it strictly weaker than the stage feeding it, and a
        # question like "what safety protocols should I follow?" reduces to
        # {safety, protocol, follow} - of which "protocol" appears nowhere in
        # the corpus. Expansion stops at this boundary: the LLM reranker and
        # the generation prompt both still see the operator's own wording.
        from rag.query_expansion import expand_query

        q_set = set(tokenize(expand_query(query)))
        if not q_set or not candidates:
            return []

        idf = self._corpus_idf(q_set, candidates)
        total_idf = sum(idf.values()) or 1.0
        out: list[RerankedChunk] = []

        for cand in candidates:
            text_tokens = tokenize(cand.chunk.text)
            head_tokens = set(tokenize(cand.chunk.heading))

            hits = q_set & set(text_tokens)
            coverage = sum(idf[t] for t in hits) / total_idf
            proximity = _proximity(q_set, text_tokens)
            head_hits = q_set & head_tokens
            heading = sum(idf[t] for t in head_hits) / total_idf

            relevance = (self.W_COVERAGE * coverage
                         + self.W_PROXIMITY * proximity
                         + self.W_HEADING * heading)
            missing = sorted(q_set - hits, key=lambda t: -idf[t])[:3]
            reason = (f"covers {len(hits)}/{len(q_set)} query terms"
                      + (f"; missing {', '.join(missing)}" if missing else ""))
            out.append(RerankedChunk(cand.chunk.chunk_id, relevance, reason, cand))

        return truncate(out, top_n)

    @staticmethod
    def _corpus_idf(q_set: set[str], candidates: list[ScoredChunk]) -> dict[str, float]:
        """IDF of each query term over the whole corpus.

        Falls back to candidate-set statistics only if the corpus cannot be
        loaded, so the reranker still works on an ad-hoc candidate list.
        """
        import math

        stats = _corpus_document_frequencies()
        if stats is None:
            n = len(candidates)
            docs = [set(tokenize(c.chunk.text)) | set(tokenize(c.chunk.heading))
                    for c in candidates]
            freq = {t: sum(1 for d in docs if t in d) for t in q_set}
        else:
            n, freq = stats[0], {t: stats[1].get(t, 0) for t in q_set}

        idf: dict[str, float] = {}
        for term in q_set:
            df = freq.get(term, 0)
            # +0.1 floor so a term absent from the corpus entirely cannot make
            # total_idf zero.
            idf[term] = math.log(1 + (n + 0.5) / (df + 0.5)) if df else 0.1
        return idf


@lru_cache(maxsize=1)
def _corpus_document_frequencies():
    """(chunk count, {term: how many chunks contain it}) over the SOP corpus."""
    try:
        from rag.chunker import chunk_corpus
        chunks = chunk_corpus()
    except Exception:
        return None
    freq: Counter = Counter()
    for chunk in chunks:
        freq.update(set(tokenize(chunk.embedding_text())))
    return len(chunks), dict(freq)


def reset_corpus_statistics() -> None:
    """Drop the cached corpus IDF - call after re-chunking or a corpus edit."""
    _corpus_document_frequencies.cache_clear()


def _proximity(q_set: set[str], text_tokens: list[str]) -> float:
    """1.0 when every matched query term sits inside one tight window, decaying
    as the matches spread out. A chunk that mentions 'grinder' in step 1 and
    'sanitize' forty lines later is a weaker match for 'sanitize the grinder'
    than one where they appear in the same sentence."""
    positions = [i for i, t in enumerate(text_tokens) if t in q_set]
    if len(positions) < 2:
        return 1.0 if positions else 0.0
    distinct = len({text_tokens[i] for i in positions})
    if distinct < 2:
        return 0.5
    span = positions[-1] - positions[0] + 1
    ideal = distinct
    return min(1.0, ideal / span * 2)


# ---------------------------------------------------------------------------
# LLM reranker
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """You score how well each candidate passage answers a question.

You are given a question and a numbered list of passages from a butcher
shop's standard operating procedures. For each passage return a relevance
score from 0.0 to 1.0 and a very short reason.

Scoring guide:
- 1.0  The passage directly and completely answers the question.
- 0.7  The passage contains most of the answer, or one necessary part of a
       multi-part answer.
- 0.4  The passage is about the right topic but does not answer the question.
- 0.1  Same vocabulary, different subject (for example, cleaning the slicer
       when the question is about cleaning the grinder).
- 0.0  Unrelated.

Be strict. Passages that merely share words with the question are 0.1, not
0.4. A passage about a different piece of equipment, a different species, or
a different cut is not relevant no matter how similar the wording, and
marking it relevant is the single most damaging error you can make here.

Score every passage you are given. Do not skip any."""

_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relevance": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "relevance", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


class LLMReranker:
    """Claude Haiku as a relevance filter. Falls back to the lexical reranker
    if the call fails - a reranker outage should degrade quality, not take
    the whole answer path down."""

    name = "llm"

    def __init__(self, model: str = llm.FAST_MODEL) -> None:
        self.model = model
        self._fallback = LexicalReranker()

    def rerank(self, query: str, candidates: list[ScoredChunk],
               top_n: int = RERANK_TOP_N) -> list[RerankedChunk]:
        if not candidates:
            return []
        if not llm.is_live():
            return self._fallback.rerank(query, candidates, top_n)

        listing = "\n\n".join(
            f"[{i}] ({c.chunk.doc_title} > {c.chunk.heading})\n"
            f"{c.chunk.text[:RERANK_CHUNK_PREVIEW_CHARS]}"
            for i, c in enumerate(candidates)
        )
        user = f"Question: {query}\n\nPassages:\n\n{listing}"

        try:
            raw = llm.complete_json(_RERANK_SYSTEM, user, _RERANK_SCHEMA,
                                    model=self.model, effort="low", max_tokens=4096)
        except Exception:
            return self._fallback.rerank(query, candidates, top_n)

        scored: list[RerankedChunk] = []
        for row in raw.get("scores", []):
            idx = row.get("index")
            if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
                continue
            relevance = float(row.get("relevance", 0.0))
            scored.append(RerankedChunk(
                chunk_id=candidates[idx].chunk.chunk_id,
                relevance=max(0.0, min(1.0, relevance)),
                reason=str(row.get("reason", ""))[:160],
                scored=candidates[idx],
            ))

        if not scored:
            return self._fallback.rerank(query, candidates, top_n)

        return truncate(scored, top_n)


def truncate(scored: list[RerankedChunk], top_n: int) -> list[RerankedChunk]:
    """Rank, then cut with the relative + absolute floors from rag/tuning.py."""
    from rag import tuning

    if not scored:
        return []
    params = tuning.current()
    ranked = sorted(scored, key=lambda r: -r.relevance)
    best = ranked[0].relevance
    if best < params.min_relevance:
        return []
    cutoff = max(params.min_relevance, best * params.relative_floor)
    return [r for r in ranked if r.relevance >= cutoff][:top_n]


def get_reranker(kind: Optional[str] = None) -> Reranker:
    """LLM reranker when credentials exist, lexical otherwise."""
    if kind == "lexical":
        return LexicalReranker()
    if kind == "llm":
        return LLMReranker()
    return LLMReranker() if llm.is_live() else LexicalReranker()
