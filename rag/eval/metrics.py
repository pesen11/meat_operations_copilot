"""
Retrieval and generation metrics.

Definitions, stated explicitly because "recall@k" means several different
things in practice and comparing across definitions is meaningless:

- recall@k      : fraction of a case's GOLD chunks that appear in the top k.
                  Averaged per-case, then across cases (macro), so a case
                  with three gold chunks does not outweigh one with a single
                  gold chunk.
- hit_rate@k    : fraction of CASES with at least one gold chunk in the top k.
                  The number that actually predicts whether the generator can
                  answer at all.
- precision@k   : fraction of the top k that is gold. Low precision here is
                  expected and acceptable - the retriever is deliberately
                  wide, and the reranker is what fixes precision.
- mrr           : mean reciprocal rank of the FIRST gold chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Optional


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(gold)


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & set(gold)) / len(top)


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / i
    return 0.0


@dataclass
class RetrievalScore:
    case_id: str
    category: str
    query: str
    gold: list[str]
    retrieved: list[str]
    reranked: list[str] = field(default_factory=list)
    recall: dict[int, float] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    rerank_recall: float = 0.0
    rerank_precision: float = 0.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "gold": self.gold,
            "recall": {str(k): round(v, 3) for k, v in self.recall.items()},
            "hit": {str(k): v for k, v in self.hit.items()},
            "precision": {str(k): round(v, 3) for k, v in self.precision.items()},
            "mrr": round(self.mrr, 3),
            "rerank_recall": round(self.rerank_recall, 3),
            "rerank_precision": round(self.rerank_precision, 3),
            "retrieved_top5": self.retrieved[:5],
            "reranked": self.reranked,
        }


def score_retrieval(case_id: str, category: str, query: str, gold: list[str],
                    retrieved: list[str], reranked: Optional[list[str]] = None,
                    ks: Iterable[int] = (1, 3, 5, 10, 14)) -> RetrievalScore:
    reranked = reranked or []
    score = RetrievalScore(
        case_id=case_id, category=category, query=query,
        gold=gold, retrieved=retrieved, reranked=reranked,
        mrr=reciprocal_rank(retrieved, gold),
    )
    for k in ks:
        score.recall[k] = recall_at_k(retrieved, gold, k)
        score.hit[k] = hit_at_k(retrieved, gold, k)
        score.precision[k] = precision_at_k(retrieved, gold, k)
    if reranked:
        score.rerank_recall = recall_at_k(reranked, gold, len(reranked))
        score.rerank_precision = precision_at_k(reranked, gold, len(reranked))
    return score


def aggregate(scores: list[RetrievalScore], ks: Iterable[int] = (1, 3, 5, 10, 14)) -> dict:
    """Macro-average across cases, plus a per-category breakdown.

    The breakdown matters more than the headline: an aggregate that looks
    healthy while every adversarial case fails is the exact situation
    CLAUDE.md warns about when it says not to trust an aggregate metric
    blindly.
    """
    if not scores:
        return {}
    out = {
        "cases": len(scores),
        "mrr": round(mean(s.mrr for s in scores), 3),
        "recall_at_k": {str(k): round(mean(s.recall[k] for s in scores), 3) for k in ks},
        "hit_rate_at_k": {str(k): round(mean(s.hit[k] for s in scores), 3) for k in ks},
        "precision_at_k": {str(k): round(mean(s.precision[k] for s in scores), 3) for k in ks},
    }
    reranked = [s for s in scores if s.reranked]
    if reranked:
        out["rerank"] = {
            "recall": round(mean(s.rerank_recall for s in reranked), 3),
            "precision": round(mean(s.rerank_precision for s in reranked), 3),
            "avg_chunks_shown": round(mean(len(s.reranked) for s in reranked), 2),
        }

    categories = sorted({s.category for s in scores})
    out["by_category"] = {
        cat: {
            "cases": len([s for s in scores if s.category == cat]),
            "recall_at_5": round(mean(s.recall[5] for s in scores if s.category == cat), 3),
            "hit_rate_at_5": round(mean(s.hit[5] for s in scores if s.category == cat), 3),
            "mrr": round(mean(s.mrr for s in scores if s.category == cat), 3),
        }
        for cat in categories
    }
    return out


# ---------------------------------------------------------------------------
# Generation-side metrics
# ---------------------------------------------------------------------------

@dataclass
class GenerationScore:
    case_id: str
    category: str
    answered: bool
    should_answer: bool
    correct_decision: bool
    numeric_grounded: Optional[bool] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    reference_overlap: Optional[float] = None
    answer: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "answered": self.answered,
            "should_answer": self.should_answer,
            "correct_decision": self.correct_decision,
            "numeric_grounded": self.numeric_grounded,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "reference_overlap": self.reference_overlap,
            "note": self.note,
        }


def aggregate_generation(scores: list[GenerationScore]) -> dict:
    if not scores:
        return {}
    answerable = [s for s in scores if s.should_answer]
    unanswerable = [s for s in scores if not s.should_answer]

    out = {
        "cases": len(scores),
        "decision_accuracy": round(mean(1.0 if s.correct_decision else 0.0
                                        for s in scores), 3),
    }
    if answerable:
        out["answer_rate_on_answerable"] = round(
            mean(1.0 if s.answered else 0.0 for s in answerable), 3)
        overlaps = [s.reference_overlap for s in answerable if s.reference_overlap is not None]
        if overlaps:
            out["reference_overlap"] = round(mean(overlaps), 3)
    if unanswerable:
        # The headline number for CLAUDE.md step 6.4: does it actually decline?
        out["correct_decline_rate"] = round(
            mean(0.0 if s.answered else 1.0 for s in unanswerable), 3)
        out["hallucinated"] = [s.case_id for s in unanswerable if s.answered]

    grounded = [s.numeric_grounded for s in scores if s.numeric_grounded is not None]
    if grounded:
        out["numeric_grounded_rate"] = round(mean(1.0 if g else 0.0 for g in grounded), 3)
    faith = [s.faithfulness for s in scores if s.faithfulness is not None]
    if faith:
        out["faithfulness"] = round(mean(faith), 3)
    relevancy = [s.answer_relevancy for s in scores if s.answer_relevancy is not None]
    if relevancy:
        out["answer_relevancy"] = round(mean(relevancy), 3)
    return out
