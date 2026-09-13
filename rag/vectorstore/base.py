"""
Vector store interface + backend selection.

The retriever talks only to this interface, so the pgvector backend and the
local file backend are genuinely interchangeable: the same eval suite runs
against either, and the numbers are comparable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from rag.chunker import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    # Component scores, kept for debugging and for the eval report - when
    # recall drops it matters whether the dense or the lexical side missed.
    components: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk.chunk_id,
                "doc_id": self.chunk.doc_id,
                "heading": self.chunk.heading,
                "score": round(self.score, 4),
                "components": {k: round(v, 4) for k, v in (self.components or {}).items()},
                "text": self.chunk.text}


class VectorStore(Protocol):
    def build(self, chunks: list[Chunk]) -> None: ...
    def load(self) -> None: ...
    def is_built(self) -> bool: ...
    def all_chunks(self) -> list[Chunk]: ...
    def search(self, query: str, top_k: int) -> list[ScoredChunk]: ...
    def stats(self) -> dict[str, Any]: ...


def get_store(backend: Optional[str] = None, **kwargs) -> VectorStore:
    """
    Pick a backend.

    Explicit `backend` wins; otherwise OPS_COPILOT_VECTOR_STORE; otherwise
    pgvector when DATABASE_URL is set, else the local file store. Falling back
    to local rather than failing means a developer with no Postgres still gets
    a working pipeline, which is the point of the abstraction.
    """
    name = (backend or os.getenv("OPS_COPILOT_VECTOR_STORE") or "").lower()
    if not name:
        name = "pgvector" if os.getenv("DATABASE_URL") else "local"

    if name == "pgvector":
        try:
            from rag.vectorstore.pgvector import PgVectorStore
            return PgVectorStore(**kwargs)
        except Exception as exc:  # missing driver, unreachable DB, no extension
            import warnings
            warnings.warn(f"pgvector backend unavailable ({exc}); using the local "
                          f"file store instead.", RuntimeWarning)
            name = "local"

    if name == "local":
        from rag.vectorstore.local import LocalVectorStore
        return LocalVectorStore(**kwargs)

    raise ValueError(f"Unknown vector store backend '{name}'")
