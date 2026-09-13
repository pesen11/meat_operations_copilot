"""
Local file-backed vector store.

Numpy matrix + a JSON sidecar, persisted under rag/index/. At 50 chunks an
exact brute-force scan is microseconds, so there is no index structure and
no approximation - which also means recall@k measured here is true recall,
not ANN recall. When this moves to pgvector at real scale, the eval numbers
from here are the ceiling to compare against.

Scoring is HYBRID, per CLAUDE.md step 6.2 (optimise for recall):

  dense   - TF-IDF/Voyage cosine over heading + body
  lexical - BM25 over the same text
  heading - direct token overlap with the heading path

Three weak signals with different failure modes beat one strong one here.
BM25 finds an exact rare term ("PTFE", "200 ppm") that a normalised vector
dilutes; the dense score handles distributed phrasing; the heading bonus
breaks ties between near-identical procedure chunks that differ only by
which machine they belong to.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rag.chunker import Chunk, chunk_corpus
from rag.embeddings import (
    EmbeddingProvider, cosine_similarity, embedder_from_state, get_embedder, tokenize,
)
from rag.vectorstore.base import ScoredChunk

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
DEFAULT_INDEX_NAME = "sop_index"

# Hybrid weights. Dense leads; BM25 is a strong second because exact rare
# terms matter a lot in procedural text; the heading bonus is deliberately
# small - it should break ties, not decide matches on its own.
# ASSUMPTION: hand-set starting point, validated by rag/eval/run_eval.py.
# Change them together and re-run the eval; do not tune one in isolation.
W_DENSE = 0.50
W_BM25 = 0.38
W_HEADING = 0.12

BM25_K1 = 1.5
BM25_B = 0.75


class LocalVectorStore:
    def __init__(self, index_name: str = DEFAULT_INDEX_NAME,
                 index_dir: Optional[Path] = None,
                 embedder: Optional[EmbeddingProvider] = None) -> None:
        self.index_dir = index_dir or INDEX_DIR
        self.index_name = index_name
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter = Counter()
        self._avg_len: float = 0.0

    # --- paths -------------------------------------------------------------
    @property
    def _meta_path(self) -> Path:
        return self.index_dir / f"{self.index_name}.json"

    @property
    def _matrix_path(self) -> Path:
        return self.index_dir / f"{self.index_name}.npy"

    # --- build / load ------------------------------------------------------
    def build(self, chunks: Optional[list[Chunk]] = None) -> None:
        self.chunks = chunks if chunks is not None else chunk_corpus()
        if not self.chunks:
            raise ValueError("Refusing to build an empty index.")
        texts = [c.embedding_text() for c in self.chunks]

        self.embedder = self.embedder or get_embedder()
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed_documents(texts)
        self._build_lexical(texts)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._matrix_path, self.matrix)
        self._meta_path.write_text(json.dumps({
            "index_name": self.index_name,
            "embedder": self.embedder.state(),
            "chunks": [c.to_dict() for c in self.chunks],
        }, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self._meta_path.exists():
            raise FileNotFoundError(
                f"No index at {self._meta_path}. Run `python -m rag.index build` first.")
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self.embedder = embedder_from_state(meta["embedder"])
        self.chunks = [
            Chunk(chunk_id=c["chunk_id"], doc_id=c["doc_id"], doc_title=c["doc_title"],
                  document_type=c["document_type"], data_source=c["data_source"],
                  heading_path=c["heading_path"], text=c["text"],
                  char_count=c["char_count"], metadata=c.get("metadata", {}))
            for c in meta["chunks"]
        ]
        self.matrix = np.load(self._matrix_path)
        self._build_lexical([c.embedding_text() for c in self.chunks])

    def is_built(self) -> bool:
        return self._meta_path.exists() and self._matrix_path.exists()

    def ensure_loaded(self) -> None:
        if not self.chunks:
            if self.is_built():
                self.load()
            else:
                self.build()

    def all_chunks(self) -> list[Chunk]:
        self.ensure_loaded()
        return list(self.chunks)

    # --- lexical side ------------------------------------------------------
    def _build_lexical(self, texts: list[str]) -> None:
        self._doc_tokens = [tokenize(t) for t in texts]
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            self._doc_freq.update(set(tokens))
        self._avg_len = (sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)
                         if self._doc_tokens else 0.0)

    def _bm25_scores(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        n = len(self._doc_tokens)
        scores = np.zeros(n, dtype=np.float32)
        if not q_tokens or n == 0:
            return scores
        for i, tokens in enumerate(self._doc_tokens):
            counts = Counter(tokens)
            length = len(tokens) or 1
            total = 0.0
            for term in q_tokens:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / (self._avg_len or 1))
                total += idf * (tf * (BM25_K1 + 1)) / denom
            scores[i] = total
        return scores

    def _heading_scores(self, query: str) -> np.ndarray:
        q_tokens = set(tokenize(query))
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if not q_tokens:
            return scores
        for i, chunk in enumerate(self.chunks):
            h_tokens = set(tokenize(chunk.heading))
            if h_tokens:
                scores[i] = len(h_tokens & q_tokens) / len(q_tokens)
        return scores

    # --- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 12) -> list[ScoredChunk]:
        self.ensure_loaded()
        dense = cosine_similarity(self.embedder.embed_query(query), self.matrix)
        bm25 = _minmax(self._bm25_scores(query))
        heading = self._heading_scores(query)
        combined = W_DENSE * dense + W_BM25 * bm25 + W_HEADING * heading

        order = np.argsort(-combined)[:top_k]
        return [
            ScoredChunk(
                chunk=self.chunks[i],
                score=float(combined[i]),
                components={"dense": float(dense[i]), "bm25": float(bm25[i]),
                            "heading": float(heading[i])},
            )
            for i in order
        ]

    def stats(self) -> dict[str, Any]:
        self.ensure_loaded()
        return {
            "backend": "local",
            "index_path": str(self._meta_path),
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "embedder": getattr(self.embedder, "name", "unknown"),
            "dim": int(self.matrix.shape[1]) if self.matrix.size else 0,
        }


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 so BM25 (unbounded) can be summed with cosine (0..1).
    An all-zero vector - no query term present anywhere - stays all-zero
    rather than becoming a uniform 0.5 that dilutes the dense signal."""
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)
