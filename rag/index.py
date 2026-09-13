"""
Index build CLI.

    python -m rag.index build      # chunk, embed, and store the SOP corpus
    python -m rag.index stats      # what's in the current index
    python -m rag.index search "how do I clean the grinder"

Backend and embedder come from the environment (see rag/vectorstore/base.py
and rag/embeddings.py); `build --backend pgvector` overrides for one run.
"""

from __future__ import annotations

import argparse
import json
import sys

from rag.chunker import chunk_corpus
from rag.vectorstore.base import get_store


def cmd_build(args) -> int:
    chunks = chunk_corpus()
    store = get_store(args.backend)
    store.build(chunks)
    stats = store.stats()
    print(f"Indexed {stats['chunks']} chunks from {stats['documents']} documents "
          f"({stats['backend']} backend, {stats['embedder']} embeddings).")
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
    for doc_id, count in sorted(by_doc.items()):
        print(f"  {doc_id}: {count} chunks")
    return 0


def cmd_stats(args) -> int:
    store = get_store(args.backend)
    if not store.is_built():
        print("No index built yet. Run `python -m rag.index build`.", file=sys.stderr)
        return 1
    print(json.dumps(store.stats(), indent=2))
    return 0


def cmd_search(args) -> int:
    from rag.pipeline import RagPipeline
    from rag.retriever import Retriever

    retriever = Retriever(store=get_store(args.backend))
    pipeline = RagPipeline(retriever=retriever)
    hits = retriever.retrieve(args.query).candidates[:args.top_k]
    print(f"Query: {args.query}\n")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. [{hit.score:.3f}] {hit.chunk.doc_id} > {hit.chunk.heading}")
        print(f"   {hit.chunk.text[:160].replace(chr(10), ' ')}...")
    if args.answer:
        print("\n--- answer ---")
        print(json.dumps(pipeline.answer(args.query).to_dict(), indent=2)[:4000])
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rag.index", description="SOP index management")
    parser.add_argument("--backend", default=None,
                        help="local | pgvector (default: pgvector if DATABASE_URL is set)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="build or rebuild the index")
    sub.add_parser("stats", help="show index statistics")

    search = sub.add_parser("search", help="run a retrieval query")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--answer", action="store_true", help="also generate an answer")

    args = parser.parse_args(argv)
    return {"build": cmd_build, "stats": cmd_stats, "search": cmd_search}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
