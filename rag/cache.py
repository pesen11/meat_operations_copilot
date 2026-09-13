"""
Two-layer query cache (CLAUDE.md step 6.5).

Layer 1 - **exact**: SHA-256 of the normalised query plus any filters.
Cheap, zero false positives, catches the genuinely repeated question.

Layer 2 - **semantic**: cosine similarity between the query's embedding and
cached queries' embeddings. Catches "how do I trim a brisket" against
"brisket trimming procedure", which layer 1 misses entirely.

**The semantic threshold is the dangerous knob in this file.** Too low and
the cache serves a confidently wrong answer to a different question - a
failure that is invisible in latency metrics and looks like a hallucination
to the user. CLAUDE.md says to validate it empirically rather than guess, so:

  - The default below is a *placeholder*, not a tuned value.
  - `rag/eval/tune_cache.py` sweeps thresholds against the labelled eval set,
    measuring true hits (paraphrases of the same question) against false
    hits (different questions that happen to be similar), and writes the
    chosen value to rag/index/cache_threshold.json.
  - `load_tuned_threshold()` reads that file when present. If it's absent,
    the cache runs at a deliberately conservative threshold and says so in
    its stats, rather than silently using a guess.
  - The right threshold is provider-specific. TF-IDF similarities and Voyage
    similarities do not live on the same scale, so the tuned file records
    which embedder it was tuned for and is ignored if the embedder changed.

The third caching layer, prompt caching (`cache_control: ephemeral` on the
static system prompt), is implemented in agents/llm.py and stacks with both
of these rather than replacing them.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rag.embeddings import EmbeddingProvider, cosine_similarity, get_embedder

# Conservative placeholder used only when no tuned threshold exists. High
# enough that it will mostly miss rather than mostly false-hit - a cache that
# does nothing is merely slow; a cache that serves the wrong answer is wrong.
UNTUNED_SEMANTIC_THRESHOLD = 0.97

DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_ENTRIES = 512


def normalise_query(query: str) -> str:
    """Whitespace/case/trailing-punctuation normalisation. Deliberately does
    NOT stem or drop stopwords: 'can I refreeze this' and 'can I not refreeze
    this' must not collapse to the same exact-cache key."""
    return " ".join(query.lower().split()).rstrip("?!. ")


def exact_key(query: str, filters: Optional[dict] = None) -> str:
    payload = json.dumps({"q": normalise_query(query),
                          "f": filters or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    key: str
    query: str
    value: Any
    created_at: float
    embedding: Optional[np.ndarray] = None
    hits: int = 0


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    threshold: float = UNTUNED_SEMANTIC_THRESHOLD
    threshold_source: str = "untuned_default"

    def to_dict(self) -> dict:
        total = self.exact_hits + self.semantic_hits + self.misses
        return {
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "hit_rate": round((self.exact_hits + self.semantic_hits) / total, 4) if total else 0.0,
            "semantic_threshold": round(self.threshold, 4),
            "threshold_source": self.threshold_source,
        }


def load_tuned_threshold(embedder_name: str) -> tuple[float, str]:
    """Read the empirically tuned threshold, if one exists for this embedder.

    Delegates to rag/tuning.py, which owns every swept parameter and already
    refuses a file calibrated for a different embedding space - a threshold
    tuned on TF-IDF similarities is not merely wrong on Voyage similarities,
    it is wrong while looking authoritative.
    """
    from rag import tuning

    params = tuning.current()
    if params.source != "tuned" or params.embedder != embedder_name:
        return UNTUNED_SEMANTIC_THRESHOLD, "untuned_default"
    return float(params.semantic_threshold), "tuned"


class QueryCache:
    """Thread-safe in-process cache with exact + semantic lookup.

    In-process means it does not survive a restart and is not shared between
    workers. For a single-node deployment that is the right trade (no Redis
    dependency); a multi-worker deployment should back this with Redis and
    keep the same interface.
    """

    def __init__(self, embedder: Optional[EmbeddingProvider] = None,
                 threshold: Optional[float] = None,
                 ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.embedder = embedder or get_embedder()
        name = getattr(self.embedder, "name", "unknown")
        tuned, source = load_tuned_threshold(name)
        self.threshold = threshold if threshold is not None else tuned
        self.threshold_source = "explicit" if threshold is not None else source
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.stats = CacheStats(threshold=self.threshold,
                                threshold_source=self.threshold_source)

    # --- internals ---------------------------------------------------------
    def _expired(self, entry: CacheEntry) -> bool:
        return self.ttl > 0 and (time.time() - entry.created_at) > self.ttl

    def _purge(self) -> None:
        dead = [k for k, e in self._entries.items() if self._expired(e)]
        for k in dead:
            del self._entries[k]
        while len(self._entries) > self.max_entries:
            oldest = min(self._entries.values(), key=lambda e: e.created_at)
            del self._entries[oldest.key]
            self.stats.evictions += 1

    def _embed(self, query: str) -> Optional[np.ndarray]:
        try:
            vec = self.embedder.embed_query(query)
            return vec if getattr(vec, "size", 0) else None
        except Exception:
            # Semantic layer is an optimisation; losing it must not break the
            # exact layer or the request.
            return None

    # --- public API --------------------------------------------------------
    def get(self, query: str, filters: Optional[dict] = None) -> tuple[Optional[Any], str]:
        """Returns (value, how) where how is 'exact', 'semantic', or 'miss'."""
        with self._lock:
            self._purge()
            key = exact_key(query, filters)
            entry = self._entries.get(key)
            if entry and not self._expired(entry):
                entry.hits += 1
                self.stats.exact_hits += 1
                return entry.value, "exact"

            if not self._entries:
                self.stats.misses += 1
                return None, "miss"

            vec = self._embed(query)
            if vec is None:
                self.stats.misses += 1
                return None, "miss"

            candidates = [e for e in self._entries.values()
                          if e.embedding is not None and not self._expired(e)]
            if not candidates:
                self.stats.misses += 1
                return None, "miss"

            matrix = np.vstack([e.embedding for e in candidates])
            sims = cosine_similarity(vec, matrix)
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.threshold:
                hit = candidates[best]
                hit.hits += 1
                self.stats.semantic_hits += 1
                return hit.value, "semantic"

            self.stats.misses += 1
            return None, "miss"

    def set(self, query: str, value: Any, filters: Optional[dict] = None) -> None:
        with self._lock:
            key = exact_key(query, filters)
            self._entries[key] = CacheEntry(
                key=key, query=query, value=value,
                created_at=time.time(), embedding=self._embed(query),
            )
            self.stats.stores += 1
            self._purge()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def similarity_to_cached(self, query: str) -> list[tuple[str, float]]:
        """(cached query, similarity) pairs - used by the threshold tuner and
        for debugging a suspicious cache hit."""
        with self._lock:
            vec = self._embed(query)
            entries = [e for e in self._entries.values() if e.embedding is not None]
            if vec is None or not entries:
                return []
            sims = cosine_similarity(vec, np.vstack([e.embedding for e in entries]))
            return sorted(((e.query, float(s)) for e, s in zip(entries, sims)),
                          key=lambda kv: -kv[1])


_default_cache: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = QueryCache()
    return _default_cache


def reset_cache() -> None:
    global _default_cache
    _default_cache = None
