"""
Empirically tuned parameters for the RAG pipeline.

CLAUDE.md is explicit that thresholds here must be validated against the eval
set rather than guessed (step 6.5 says it about the semantic cache; the same
discipline applies to every threshold in this layer). This module is the one
place those numbers live, and `rag/eval/tune.py` is what sets them.

Defaults below are STARTING POINTS. Run:

    python -m rag.eval.tune

which sweeps them against rag/eval/dataset.py and writes the chosen values to
rag/index/tuned_params.json. `current()` prefers that file and reports which
source it used, so nothing silently claims to be tuned when it isn't.

A tuned file records the embedder and reranker it was tuned with, and is
ignored if either changed - a threshold calibrated on TF-IDF scores means
nothing on Voyage scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

TUNED_PATH = Path(__file__).resolve().parent / "index" / "tuned_params.json"


@dataclass
class TunedParams:
    # --- reranker truncation ---
    # Absolute floor: a sanity bar only. The score distributions of gold and
    # non-gold chunks overlap heavily under lexical scoring (measured, not
    # assumed), so an absolute floor cannot separate them - set it low enough
    # that it only rejects genuine noise.
    min_relevance: float = 0.20
    # Relative floor: keep chunks scoring at least this fraction of the best
    # chunk's score. This is what actually controls precision, and unlike an
    # absolute floor it cannot return an empty set when a good match exists.
    relative_floor: float = 0.55
    # How many chunks reach the generation prompt. Swept, not assumed: too few
    # makes multi-hop questions unanswerable, too many is exactly the "don't
    # dump top-k into context" failure CLAUDE.md step 6.3 warns about.
    rerank_top_n: int = 4

    # --- declining (offline extractive path only) ---
    # Best-chunk relevance below which the extractive answerer declines.
    extractive_min_relevance: float = 0.40
    # Fraction of the query's content words that appear NOWHERE in the corpus,
    # above which the question is treated as out of scope. This is the only
    # signal available offline that separates "the corpus doesn't cover this"
    # from "the corpus covers it in different words".
    oov_decline_threshold: float = 0.34

    # --- semantic cache ---
    semantic_threshold: float = 0.97

    # provenance
    embedder: str = "tfidf"
    reranker: str = "lexical"
    source: str = "default"
    tuned_at: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load(embedder: str = "tfidf", reranker: str = "lexical") -> TunedParams:
    """Tuned params for this (embedder, reranker) pair, or the defaults."""
    defaults = TunedParams(embedder=embedder, reranker=reranker)
    if not TUNED_PATH.exists():
        return defaults
    try:
        data = json.loads(TUNED_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return defaults
    if data.get("embedder") != embedder or data.get("reranker") != reranker:
        defaults.notes = (f"tuned file exists but was calibrated for "
                          f"{data.get('embedder')}/{data.get('reranker')}; ignored")
        return defaults
    known = {f for f in TunedParams.__dataclass_fields__}
    params = TunedParams(**{k: v for k, v in data.items() if k in known})
    params.source = "tuned"
    return params


def save(params: TunedParams) -> Path:
    from datetime import datetime, timezone
    params.source = "tuned"
    params.tuned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")
    return TUNED_PATH


_current: Optional[TunedParams] = None


def current() -> TunedParams:
    global _current
    if _current is None:
        import os
        from agents import llm
        embedder = (os.getenv("OPS_COPILOT_EMBEDDINGS") or "tfidf").lower()
        reranker = "llm" if llm.is_live() else "lexical"
        _current = load(embedder, reranker)
    return _current


def reset() -> None:
    global _current
    _current = None
