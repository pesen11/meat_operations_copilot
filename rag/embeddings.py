"""
Embedding providers.

Two are implemented:

- `tfidf` (DEFAULT): a deterministic TF-IDF vectoriser fitted on the SOP
  corpus itself. No network, no key, no cost, identical output on every run.
  This is what makes the whole retrieval eval suite runnable in CI and on a
  laptop with no credentials.
- `voyage`: real dense semantic embeddings via Voyage AI (Anthropic's
  recommended embedding partner - Anthropic's own API has no embeddings
  endpoint). Set VOYAGE_API_KEY and OPS_COPILOT_EMBEDDINGS=voyage.

**Be honest about what the default can and cannot do.** TF-IDF matches
words, not meaning. "how do I trim a brisket" and "brisket trimming
procedure" share a stem and score well; "how cold should the cooler be" and
"temperature requirements for refrigerated storage" share almost nothing and
score badly. That is precisely why:

  1. retrieval is hybrid (TF-IDF + BM25 + heading match), not dense-only;
  2. the semantic cache threshold is TUNED PER PROVIDER against the eval set
     (rag/eval/tune_cache.py) rather than hardcoded - a threshold that is
     safe for TF-IDF is far too loose for real dense embeddings, and vice
     versa.

Swapping to `voyage` should raise recall on paraphrased queries. The eval
harness measures that rather than assuming it.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Iterable, Optional, Protocol

import numpy as np

DEFAULT_PROVIDER = "tfidf"
VOYAGE_MODEL = "voyage-3"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words carrying no retrieval signal in a corpus where every document is an
# SOP about meat. Kept deliberately short - aggressive stoplists hurt recall,
# and recall is the metric this project optimises for (CLAUDE.md step 6.2).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "be", "for",
    "on", "at", "it", "this", "that", "with", "as", "by", "from", "not", "no",
    "do", "does", "how", "what", "when", "which", "any", "all", "must", "can",
    "if", "then", "than", "so", "into", "out", "up", "down", "per", "each",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with a light suffix strip.

    The stemming is crude on purpose: 'trimming'/'trimmed'/'trim' and
    'cleaning'/'cleaned'/'clean' must collide, because operators and SOP
    authors pick different forms of the same verb. A real stemmer would be
    better; this one is dependency-free and its failure mode (over-merging a
    rare word) costs precision, which reranking recovers, rather than recall,
    which nothing recovers.
    """
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        for suffix in ("ing", "ies", "ed", "es", "s"):
            if len(raw) > len(suffix) + 3 and raw.endswith(suffix):
                raw = raw[: -len(suffix)]
                break
        tokens.append(raw)
    return tokens


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def fit(self, documents: list[str]) -> None: ...
    def embed_documents(self, documents: list[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...
    def state(self) -> dict[str, Any]: ...
    def load_state(self, state: dict[str, Any]) -> None: ...


class TfidfEmbedder:
    """Deterministic TF-IDF with L2-normalised vectors, so a dot product is
    cosine similarity."""

    name = "tfidf"

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    @property
    def dim(self) -> int:
        return len(self.vocabulary)

    def fit(self, documents: list[str]) -> None:
        df: Counter = Counter()
        for doc in documents:
            df.update(set(tokenize(doc)))
        self.vocabulary = {term: i for i, term in enumerate(sorted(df))}
        n = len(documents)
        idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for term, i in self.vocabulary.items():
            # Smoothed IDF; +1 keeps a term appearing in every document at a
            # small positive weight rather than exactly zero.
            idf[i] = math.log((1 + n) / (1 + df[term])) + 1.0
        self.idf = idf

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        counts = Counter(tokenize(text))
        if not counts:
            return vec
        max_count = max(counts.values())
        for term, count in counts.items():
            idx = self.vocabulary.get(term)
            if idx is None:  # out-of-vocabulary query terms simply don't vote
                continue
            vec[idx] = (0.5 + 0.5 * count / max_count) * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        if not self.vocabulary:
            self.fit(documents)
        return np.vstack([self._vector(d) for d in documents]) if documents else np.zeros((0, self.dim))

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def state(self) -> dict[str, Any]:
        return {"provider": self.name,
                "vocabulary": self.vocabulary,
                "idf": self.idf.tolist()}

    def load_state(self, state: dict[str, Any]) -> None:
        self.vocabulary = dict(state["vocabulary"])
        self.idf = np.asarray(state["idf"], dtype=np.float32)


class VoyageEmbedder:
    """Dense semantic embeddings via Voyage AI.

    Anthropic does not serve an embeddings endpoint, and Voyage is the
    partner Anthropic points at for this, so it is the sanctioned way to get
    real semantic retrieval into an otherwise all-Claude stack.
    """

    name = "voyage"

    def __init__(self, model: str = VOYAGE_MODEL) -> None:
        self.model = model
        self._dim = 1024
        self._client = None

    @property
    def dim(self) -> int:
        return self._dim

    def _get_client(self):
        if self._client is None:
            import voyageai  # optional dependency
            self._client = voyageai.Client()
        return self._client

    def fit(self, documents: list[str]) -> None:
        return  # nothing to fit for a hosted model

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        result = self._get_client().embed(texts, model=self.model, input_type=input_type)
        arr = np.asarray(result.embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._dim = arr.shape[1]
        return arr / norms

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.zeros((0, self._dim), dtype=np.float32)
        # Voyage caps batch size; 128 is comfortably inside it.
        batches = [documents[i:i + 128] for i in range(0, len(documents), 128)]
        return np.vstack([self._embed(b, "document") for b in batches])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query], "query")[0]

    def state(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "dim": self._dim}

    def load_state(self, state: dict[str, Any]) -> None:
        self.model = state.get("model", self.model)
        self._dim = int(state.get("dim", self._dim))


_PROVIDERS = {"tfidf": TfidfEmbedder, "voyage": VoyageEmbedder}


def get_embedder(provider: Optional[str] = None) -> EmbeddingProvider:
    name = (provider or os.getenv("OPS_COPILOT_EMBEDDINGS") or DEFAULT_PROVIDER).lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown embedding provider '{name}'. "
                         f"Available: {', '.join(sorted(_PROVIDERS))}")
    return _PROVIDERS[name]()


def embedder_from_state(state: dict[str, Any]) -> EmbeddingProvider:
    """Rebuild the embedder an index was written with, so a query is embedded
    in exactly the same space as the stored vectors."""
    embedder = get_embedder(state.get("provider", DEFAULT_PROVIDER))
    embedder.load_state(state)
    return embedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a matrix of rows.
    Both sides are already L2-normalised, so this is a dot product; the
    explicit normalisation guards against a provider that forgets."""
    if b.size == 0:
        return np.zeros(0, dtype=np.float32)
    a_norm = np.linalg.norm(a)
    if a_norm:
        a = a / a_norm
    b_norms = np.linalg.norm(b, axis=1)
    b_norms[b_norms == 0] = 1.0
    return (b @ a) / b_norms
