"""
Header-aware markdown chunking.

Why not a fixed character window: these SOPs are written as numbered
sections where the heading carries most of the disambiguating meaning. A
blind 800-character window cuts "## 5. Grinder" off the top of the grinder
cleaning steps, and the resulting chunk is nearly indistinguishable from
the slicer's - which is exactly the failure mode the adversarial eval cases
are built to catch.

So: split on headings, keep the full heading path on every chunk, and only
then split oversized sections, repeating the heading path on each part.

Tables are kept whole wherever possible. A markdown table split down the
middle loses its header row, and a temperature table without its column
headers is worse than useless - it reads as a list of plausible numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

from rag.loader import SopDocument, load_corpus

# Target chunk size in characters. Sized so a whole numbered subsection
# usually fits in one chunk: the SOP sections in this corpus run 400-1500
# characters. ASSUMPTION - tune against retrieval eval results, not by feel.
TARGET_CHUNK_CHARS = 1400
# Hard ceiling; above this a section is split even mid-prose.
MAX_CHUNK_CHARS = 2200
# Overlap carried between split parts of one oversized section, so a fact
# that lands on a split boundary appears in both halves.
OVERLAP_CHARS = 180
# Sections shorter than this are merged into the following chunk rather than
# standing alone (a bare "## 2. Scope" one-liner retrieves noise).
MIN_CHUNK_CHARS = 120

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    document_type: str
    data_source: str
    heading_path: list[str]
    text: str
    char_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)

    def embedding_text(self) -> str:
        """What actually gets embedded: heading path prepended to the body.

        The heading is repeated into the embedded text on purpose. 'Clean the
        blade from the centre outwards' is ambiguous between the slicer and
        the band saw until 'Slicer' is part of the text being embedded.
        """
        return f"{self.doc_title}\n{self.heading}\n\n{self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["heading"] = self.heading
        return d


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def _sections(body: str) -> list[tuple[list[str], str]]:
    """Split a document body into (heading_path, text) sections."""
    lines = body.splitlines()
    stack: list[tuple[int, str]] = []
    sections: list[tuple[list[str], str]] = []
    buffer: list[str] = []
    current: list[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append((list(current), text))
        buffer.clear()

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = [t for _, t in stack]
        else:
            buffer.append(line)
    flush()
    return sections


def _split_long_text(text: str) -> list[str]:
    """Split an oversized section on paragraph boundaries, with overlap.

    Table blocks are treated as single unsplittable paragraphs so a table
    never loses its header row.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    blocks: list[str] = []
    current: list[str] = []
    in_table = False
    for line in text.splitlines():
        row = _is_table_row(line)
        if row and not in_table and current:
            blocks.append("\n".join(current))
            current = []
        if not row and in_table:
            blocks.append("\n".join(current))
            current = []
        in_table = row
        if line.strip() == "" and not in_table:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))

    parts: list[str] = []
    buf = ""
    for block in blocks:
        candidate = f"{buf}\n\n{block}".strip() if buf else block
        if len(candidate) > TARGET_CHUNK_CHARS and buf:
            parts.append(buf)
            tail = buf[-OVERLAP_CHARS:] if len(buf) > OVERLAP_CHARS else buf
            buf = f"{tail}\n\n{block}".strip()
        else:
            buf = candidate
    if buf:
        parts.append(buf)
    return parts


def chunk_document(doc: SopDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: Optional[tuple[list[str], str]] = None

    for heading_path, text in _sections(doc.body):
        if pending is not None:
            heading_path = pending[0] if not heading_path else heading_path
            text = f"{pending[1]}\n\n{text}"
            pending = None
        if len(text) < MIN_CHUNK_CHARS:
            pending = (heading_path, text)
            continue
        for part in _split_long_text(text):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks):03d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                document_type=doc.document_type,
                data_source=doc.data_source,
                heading_path=heading_path or [doc.title],
                text=part,
                char_count=len(part),
                metadata={"path": str(doc.path)},
            ))

    if pending is not None:  # trailing short section still deserves a chunk
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{len(chunks):03d}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            document_type=doc.document_type,
            data_source=doc.data_source,
            heading_path=pending[0] or [doc.title],
            text=pending[1],
            char_count=len(pending[1]),
            metadata={"path": str(doc.path)},
        ))
    return chunks


def chunk_corpus(docs: Optional[list[SopDocument]] = None) -> list[Chunk]:
    docs = docs if docs is not None else load_corpus()
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc))
    return out
