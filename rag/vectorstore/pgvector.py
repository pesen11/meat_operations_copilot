"""
pgvector backend - the Neon Postgres target from the original architecture.

Same table lives in the same database as the operational tables (db/schema.sql),
which is the reason for choosing pgvector over a separate Pinecone index at
this data scale: one connection string, one backup, one place to join
SOP chunks against operational data later.

Scoring mirrors rag/vectorstore/local.py exactly - dense cosine + BM25 +
heading bonus, same weights - so eval numbers are comparable between
backends. The dense half runs in Postgres (`<=>` cosine distance, which is
what the ivfflat index accelerates); the lexical half runs in Python over the
candidate set. Pushing BM25 into Postgres full-text search would be faster at
scale but scores differently, and a backend that quietly changes the ranking
would make the two sets of eval numbers incomparable, which is worse than
slower.

Requires: pip install -e ".[postgres]" and DATABASE_URL set.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Any, Optional

import numpy as np

from rag.chunker import Chunk, chunk_corpus
from rag.embeddings import (
    EmbeddingProvider, embedder_from_state, get_embedder, tokenize,
)
from rag.vectorstore.base import ScoredChunk
from rag.vectorstore.local import (
    BM25_B, BM25_K1, W_BM25, W_DENSE, W_HEADING, _minmax,
)

TABLE = "sop_chunks"
STATE_TABLE = "sop_index_state"

# Candidates pulled from Postgres by dense similarity before the lexical
# rerank runs in Python. Generous on purpose: a chunk that BM25 would have
# ranked first is invisible if the dense pre-filter never returns it, which
# is the unrecoverable failure mode CLAUDE.md step 6.2 warns about.
DENSE_CANDIDATE_MULTIPLIER = 6
MIN_DENSE_CANDIDATES = 60


class PgVectorStore:
    def __init__(self, dsn: Optional[str] = None,
                 embedder: Optional[EmbeddingProvider] = None) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError("DATABASE_URL is not set; cannot use the pgvector backend.")
        import psycopg  # noqa: F401 - fail fast if the driver is missing
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter = Counter()
        self._avg_len = 0.0

    # --- connection --------------------------------------------------------
    def _connect(self):
        import psycopg
        return psycopg.connect(self.dsn)

    def _ensure_schema(self, conn, dim: int) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    chunk_id       TEXT PRIMARY KEY,
                    doc_id         TEXT NOT NULL,
                    doc_title      TEXT NOT NULL,
                    document_type  TEXT NOT NULL,
                    data_source    TEXT NOT NULL,
                    heading_path   JSONB NOT NULL,
                    text           TEXT NOT NULL,
                    char_count     INTEGER NOT NULL,
                    metadata       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding      vector({dim})
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
                    index_name TEXT PRIMARY KEY,
                    state      JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_doc_id_idx ON {TABLE} (doc_id)")
        conn.commit()

    # --- build / load ------------------------------------------------------
    def build(self, chunks: Optional[list[Chunk]] = None) -> None:
        self.chunks = chunks if chunks is not None else chunk_corpus()
        if not self.chunks:
            raise ValueError("Refusing to build an empty index.")
        texts = [c.embedding_text() for c in self.chunks]

        self.embedder = self.embedder or get_embedder()
        self.embedder.fit(texts)
        matrix = self.embedder.embed_documents(texts)
        dim = int(matrix.shape[1])

        with self._connect() as conn:
            self._ensure_schema(conn, dim)
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE {TABLE}")
                for chunk, vector in zip(self.chunks, matrix):
                    cur.execute(
                        f"""INSERT INTO {TABLE} (chunk_id, doc_id, doc_title, document_type,
                                data_source, heading_path, text, char_count, metadata, embedding)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (chunk.chunk_id, chunk.doc_id, chunk.doc_title, chunk.document_type,
                         chunk.data_source, json.dumps(chunk.heading_path), chunk.text,
                         chunk.char_count, json.dumps(chunk.metadata),
                         _to_pgvector(vector)),
                    )
                # ivfflat needs rows present before the index is created, and
                # a list count above the row count is wasted work.
                lists = max(1, min(100, len(self.chunks) // 10))
                cur.execute(f"DROP INDEX IF EXISTS {TABLE}_embedding_idx")
                cur.execute(f"""CREATE INDEX {TABLE}_embedding_idx ON {TABLE}
                                USING ivfflat (embedding vector_cosine_ops)
                                WITH (lists = {lists})""")
                cur.execute(
                    f"""INSERT INTO {STATE_TABLE} (index_name, state) VALUES (%s, %s)
                        ON CONFLICT (index_name) DO UPDATE
                        SET state = EXCLUDED.state, updated_at = now()""",
                    ("sop_index", json.dumps(self.embedder.state())),
                )
            conn.commit()
        self._build_lexical(texts)

    def load(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT state FROM {STATE_TABLE} WHERE index_name = 'sop_index'")
            row = cur.fetchone()
            if not row:
                raise FileNotFoundError(
                    "No pgvector index state found. Run `python -m rag.index build` first.")
            self.embedder = embedder_from_state(row[0])

            cur.execute(f"""SELECT chunk_id, doc_id, doc_title, document_type, data_source,
                                   heading_path, text, char_count, metadata
                            FROM {TABLE} ORDER BY chunk_id""")
            self.chunks = [
                Chunk(chunk_id=r[0], doc_id=r[1], doc_title=r[2], document_type=r[3],
                      data_source=r[4], heading_path=list(r[5]), text=r[6],
                      char_count=r[7], metadata=dict(r[8] or {}))
                for r in cur.fetchall()
            ]
        self._build_lexical([c.embedding_text() for c in self.chunks])

    def is_built(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(f"SELECT to_regclass('{TABLE}')")
                if cur.fetchone()[0] is None:
                    return False
                cur.execute(f"SELECT count(*) FROM {TABLE}")
                return cur.fetchone()[0] > 0
        except Exception:
            return False

    def ensure_loaded(self) -> None:
        if not self.chunks:
            self.load() if self.is_built() else self.build()

    def all_chunks(self) -> list[Chunk]:
        self.ensure_loaded()
        return list(self.chunks)

    # --- lexical side (identical scoring to the local backend) -------------
    def _build_lexical(self, texts: list[str]) -> None:
        self._doc_tokens = [tokenize(t) for t in texts]
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            self._doc_freq.update(set(tokens))
        self._avg_len = (sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)
                         if self._doc_tokens else 0.0)

    def _bm25_for(self, query: str, indices: list[int]) -> dict[int, float]:
        q_tokens = tokenize(query)
        n = len(self._doc_tokens)
        out = {i: 0.0 for i in indices}
        if not q_tokens or n == 0:
            return out
        for i in indices:
            counts = Counter(self._doc_tokens[i])
            length = len(self._doc_tokens[i]) or 1
            total = 0.0
            for term in q_tokens:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / (self._avg_len or 1))
                total += idf * (tf * (BM25_K1 + 1)) / denom
            out[i] = total
        return out

    # --- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 12) -> list[ScoredChunk]:
        self.ensure_loaded()
        qvec = self.embedder.embed_query(query)
        limit = max(MIN_DENSE_CANDIDATES, top_k * DENSE_CANDIDATE_MULTIPLIER)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT chunk_id, 1 - (embedding <=> %s) AS similarity
                    FROM {TABLE} ORDER BY embedding <=> %s LIMIT %s""",
                (_to_pgvector(qvec), _to_pgvector(qvec), limit),
            )
            dense_rows = cur.fetchall()

        by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        indices = [by_id[r[0]] for r in dense_rows if r[0] in by_id]
        dense = {by_id[r[0]]: float(r[1]) for r in dense_rows if r[0] in by_id}

        bm25_raw = self._bm25_for(query, indices)
        bm25_scaled = dict(zip(indices, _minmax(np.array([bm25_raw[i] for i in indices],
                                                         dtype=np.float32))))
        q_tokens = set(tokenize(query))

        scored: list[ScoredChunk] = []
        for i in indices:
            chunk = self.chunks[i]
            h_tokens = set(tokenize(chunk.heading))
            heading = len(h_tokens & q_tokens) / len(q_tokens) if q_tokens and h_tokens else 0.0
            total = (W_DENSE * dense.get(i, 0.0)
                     + W_BM25 * float(bm25_scaled.get(i, 0.0))
                     + W_HEADING * heading)
            scored.append(ScoredChunk(chunk=chunk, score=total, components={
                "dense": dense.get(i, 0.0),
                "bm25": float(bm25_scaled.get(i, 0.0)),
                "heading": heading,
            }))
        scored.sort(key=lambda s: -s.score)
        return scored[:top_k]

    def stats(self) -> dict[str, Any]:
        self.ensure_loaded()
        return {
            "backend": "pgvector",
            "table": TABLE,
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "embedder": getattr(self.embedder, "name", "unknown"),
        }


def _to_pgvector(vector) -> str:
    """pgvector's text input format. psycopg sends it as a string literal;
    the column type casts it."""
    return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"
