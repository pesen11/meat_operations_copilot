"""
Optional Ragas integration for the SOP RAG evals.

CLAUDE.md names Ragas for faithfulness and answer_relevancy. Ragas pulls a
heavy dependency tree, so it is an extra (`pip install -e ".[eval]"`) rather
than a hard requirement, and the built-in Claude-judge implementations in
rag/groundedness.py and rag/eval/run_eval.py are the default. This adapter
runs real Ragas when it is installed, so the numbers can be compared.

**The blind spot the project owner already hit, carried forward.** Ragas
answer_relevancy scores a correct "I don't know" near zero: the metric
generates questions from the answer and compares them to the original, and
a refusal generates nothing resembling the question. That makes a system
that correctly declines look worse than one that confidently makes things
up - and drags the aggregate down in a way that creates pressure to "fix"
the one behaviour you most want to keep.

So `evaluate()` here EXCLUDES no-answer cases from relevancy by default, and
reports their correct-decline rate as a separate figure. That is not the
metric being massaged; it is the metric being applied to the population it
is valid for. Pass `include_declines=True` to see the uncorrected number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class RagasReport:
    available: bool
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    scored_cases: int = 0
    excluded_declines: int = 0
    correct_decline_rate: Optional[float] = None
    note: str = ""
    per_case: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "scored_cases": self.scored_cases,
            "excluded_declines": self.excluded_declines,
            "correct_decline_rate": self.correct_decline_rate,
            "note": self.note,
        }


def build_samples(pipeline, cases, include_declines: bool = False):
    """Run the pipeline over eval cases and shape rows for Ragas.

    Returns (rows, declines_excluded, correct_decline_rate).
    """
    from rag.generate import build_context

    rows: list[dict] = []
    excluded = 0
    decline_total = 0
    decline_correct = 0

    for case in cases:
        response = pipeline.answer(case.query, use_cache=False)
        retrieval = pipeline.retriever.retrieve(case.query)
        reranked = pipeline.reranker.rerank(case.query, retrieval.candidates,
                                            pipeline.top_n)
        contexts = [r.scored.chunk.text for r in reranked]

        if not case.expects_answer:
            decline_total += 1
            decline_correct += 0 if response.answered else 1

        if not response.answered and not include_declines:
            excluded += 1
            continue

        rows.append({
            "question": case.query,
            "answer": response.answer,
            "contexts": contexts or [""],
            "ground_truth": case.reference_answer or "",
            "case_id": case.case_id,
            "category": case.category,
        })

    rate = (decline_correct / decline_total) if decline_total else None
    return rows, excluded, rate


def evaluate(pipeline=None, cases=None, include_declines: bool = False) -> RagasReport:
    from rag.eval.dataset import EVAL_CASES
    from rag.pipeline import RagPipeline

    pipeline = pipeline or RagPipeline(use_cache=False)
    cases = cases if cases is not None else EVAL_CASES

    rows, excluded, decline_rate = build_samples(pipeline, cases, include_declines)

    if not ragas_available():
        return RagasReport(
            available=False, scored_cases=len(rows), excluded_declines=excluded,
            correct_decline_rate=decline_rate,
            note=("Ragas is not installed. The built-in Claude-judge equivalents "
                  "in rag/groundedness.py and rag/eval/run_eval.py --deep were "
                  "used instead. Install with: pip install -e \".[eval]\""),
            per_case=rows,
        )

    from agents import llm
    if not llm.is_live():
        return RagasReport(
            available=True, scored_cases=len(rows), excluded_declines=excluded,
            correct_decline_rate=decline_rate,
            note="Ragas is installed but needs a live LLM judge; no credentials "
                 "configured, so no scores were computed.",
            per_case=rows,
        )

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        dataset = Dataset.from_list([
            {k: v for k, v in row.items() if k in
             ("question", "answer", "contexts", "ground_truth")}
            for row in rows
        ])
        result = ragas_evaluate(dataset,
                                metrics=[faithfulness, answer_relevancy, context_precision])
        scores = dict(result)
        return RagasReport(
            available=True,
            faithfulness=_as_float(scores.get("faithfulness")),
            answer_relevancy=_as_float(scores.get("answer_relevancy")),
            context_precision=_as_float(scores.get("context_precision")),
            scored_cases=len(rows),
            excluded_declines=excluded,
            correct_decline_rate=decline_rate,
            note=("Declined answers excluded from relevancy: Ragas scores a "
                  "correct refusal near zero, which would penalise exactly the "
                  "behaviour the no-answer cases exist to enforce."
                  if not include_declines else
                  "Declines INCLUDED - relevancy is depressed by correct refusals."),
            per_case=rows,
        )
    except Exception as exc:
        return RagasReport(
            available=True, scored_cases=len(rows), excluded_declines=excluded,
            correct_decline_rate=decline_rate,
            note=f"Ragas run failed: {type(exc).__name__}: {exc}",
            per_case=rows,
        )


def _as_float(value) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
