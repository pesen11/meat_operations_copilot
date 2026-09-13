"""
Load the SOP corpus from data/sops/*.md.

Every document carries YAML frontmatter, including `data_source`. That tag
is preserved all the way through chunking, retrieval, and into the
generation prompt, so an answer drawn from synthetic material can always be
identified as such - the project's data-provenance requirement does not stop
at the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

SOP_DIR = Path(__file__).resolve().parent.parent / "data" / "sops"


@dataclass
class SopDocument:
    doc_id: str            # filename stem
    path: Path
    title: str
    document_type: str
    data_source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def is_synthetic(self) -> bool:
        return self.data_source != "real"


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a '---' delimited YAML frontmatter block off the top of a file.

    Returns ({}, text) for a file without frontmatter rather than raising -
    a new SOP dropped into the folder without a header should still be
    searchable, just with less metadata.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, parts[2].lstrip("\n")


def load_document(path: Path) -> SopDocument:
    meta, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    return SopDocument(
        doc_id=path.stem,
        path=path,
        title=str(meta.get("title", path.stem)),
        document_type=str(meta.get("document_type", "unknown")),
        data_source=str(meta.get("data_source", "synthetic")),
        metadata={k: v for k, v in meta.items()
                  if k not in ("title", "document_type", "data_source")},
        body=body,
    )


def load_corpus(sop_dir: Optional[Path] = None) -> list[SopDocument]:
    """Load every .md file in the SOP directory, sorted by filename so chunk
    ids are stable across runs (the eval set references chunk ids)."""
    directory = sop_dir or SOP_DIR
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No SOP documents found in {directory}")
    return [load_document(p) for p in paths]
