"""
Grounded answer generation.

CLAUDE.md step 6.4: answer from retrieved context only, and say so plainly
when the answer is not there. Three mechanisms, because the instruction
alone is not enough and the requirement explicitly says to verify rather
than assume:

1. **The prompt.** Explicit, with the refusal phrasing specified verbatim so
   both the eval harness and the API can detect a decline reliably instead
   of pattern-matching on hedging language.
2. **A pre-generation gate.** If the reranker returned nothing above the
   relevance floor, no model call happens at all - the decline is returned
   directly. This is the cheapest and most reliable defence, since a model
   shown zero relevant chunks is exactly the situation where it is most
   tempted to fill the gap from parametric knowledge.
3. **A post-generation groundedness check** (rag/groundedness.py), run as a
   live guardrail rather than only offline.

The system prompt is byte-stable and sent as a cached block; the question and
the chunks go in the user turn, after the cache breakpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agents import llm
from rag.reranker import RerankedChunk

# Exact refusal string. The eval set, the API, and the groundedness check all
# key off this, so it must not drift - change it here and nowhere else.
NO_ANSWER_MARKER = "NOT_IN_CORPUS"
NO_ANSWER_TEXT = (
    "I can't answer that from the shop's SOPs - the procedures on file don't "
    "cover it. Ask the department lead, or check whether a written procedure "
    "for this exists."
)

_SYSTEM = """You answer questions about a butcher shop's standard operating procedures.

You will be given a question and a set of numbered passages taken from the
shop's own SOP documents. Those passages are the ONLY source you may use.

Rules:

1. Answer strictly from the passages provided. Do not add general food-safety
   knowledge, industry convention, regulatory detail, or anything you know
   from training. If the passages disagree with what you believe to be true,
   follow the passages - they are this shop's actual procedure.

2. If the passages do not contain the answer, reply with exactly this and
   nothing else:
   NOT_IN_CORPUS

   Use it when the passages are about a related but different subject - a
   different machine, a different cut, a different species, a different
   stage. Partial topical overlap is not an answer. Declining is the correct,
   expected outcome for a question the SOPs do not cover; a plausible-sounding
   answer assembled from adjacent passages is a serious failure, not a
   helpful one.

3. When you do answer, cite the passages you used by their bracketed number,
   like [2]. Cite every passage you drew on.

4. Preserve specifics exactly as written - temperatures, concentrations,
   weights, times, thicknesses. Never round, convert, or restate a figure in
   different units.

5. Be direct and practical. Write for someone on a cutting floor who needs to
   act on this now. Lead with the answer; add conditions afterwards.

6. If the passages give a general rule AND a specific exception that applies
   to the question, state the exception. Reporting only the general rule when
   an exception covers the asked-about case is a wrong answer."""


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    answered: bool
    cited_chunk_ids: list[str] = field(default_factory=list)
    context_chunk_ids: list[str] = field(default_factory=list)
    generated_by: str = "claude"
    usage: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "answered": self.answered,
            "cited_chunk_ids": self.cited_chunk_ids,
            "context_chunk_ids": self.context_chunk_ids,
            "generated_by": self.generated_by,
        }


def build_context(chunks: list[RerankedChunk]) -> str:
    """Numbered passages, each labelled with its document and heading.

    The label is not decoration: it is what lets the model tell the grinder
    procedure from the slicer procedure when the step text is nearly
    identical, and what makes a citation traceable back to a file.
    """
    blocks = []
    for i, r in enumerate(chunks, start=1):
        c = r.scored.chunk
        provenance = "synthetic example SOP" if c.data_source != "real" else "real SOP"
        blocks.append(
            f"[{i}] {c.doc_title} > {c.heading}  ({provenance})\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


def _extract_citations(answer: str, chunks: list[RerankedChunk]) -> list[str]:
    import re
    cited: list[str] = []
    for match in re.findall(r"\[(\d+)\]", answer):
        idx = int(match) - 1
        if 0 <= idx < len(chunks):
            chunk_id = chunks[idx].chunk_id
            if chunk_id not in cited:
                cited.append(chunk_id)
    return cited


def _offline_answer(query: str, chunks: list[RerankedChunk]) -> str:
    """
    Offline extractive answer: quote the best passage rather than compose one.

    This is not a pretend LLM. It returns the top passage's text verbatim with
    its citation, which is trivially grounded (it IS the source) and keeps the
    whole pipeline - retrieval, reranking, declining, caching, evaluation -
    exercisable with no credentials. It reads like a quotation because it is
    one.

    It declines on two signals, both tuned in rag/tuning.py: a best-chunk
    relevance floor, and the out-of-scope vocabulary check in rag/scope.py.
    A live model can read four passages and conclude that none of them
    answers the question; a quoting function cannot, so it needs those
    signals instead.
    """
    from rag import tuning
    from rag.scope import looks_out_of_scope

    params = tuning.current()
    if not chunks or chunks[0].relevance < params.extractive_min_relevance:
        return NO_ANSWER_MARKER
    if looks_out_of_scope(query)[0]:
        return NO_ANSWER_MARKER
    c = chunks[0].scored.chunk
    return f"From {c.doc_title} - {c.heading} [1]:\n\n{c.text.strip()}"


def generate(query: str, chunks: list[RerankedChunk], *,
             model: str = llm.OPS_MODEL) -> GeneratedAnswer:
    context_ids = [r.chunk_id for r in chunks]

    # Gate 2: no relevant context means no model call.
    if not chunks:
        return GeneratedAnswer(query=query, answer=NO_ANSWER_TEXT, answered=False,
                               context_chunk_ids=[], generated_by="gate")

    offline_text = _offline_answer(query, chunks)
    user = (f"Question: {query}\n\n"
            f"Passages:\n\n{build_context(chunks)}")

    result = llm.complete(_SYSTEM, user, model=model, effort="low",
                          max_tokens=2048, prefill_offline=offline_text)
    text = (result.text or offline_text).strip()

    if NO_ANSWER_MARKER in text:
        return GeneratedAnswer(query=query, answer=NO_ANSWER_TEXT, answered=False,
                               context_chunk_ids=context_ids,
                               generated_by="claude" if result.live else "extractive",
                               usage=result.usage)

    return GeneratedAnswer(
        query=query,
        answer=text,
        answered=True,
        cited_chunk_ids=_extract_citations(text, chunks),
        context_chunk_ids=context_ids,
        generated_by="claude" if result.live else "extractive",
        usage=result.usage,
    )
