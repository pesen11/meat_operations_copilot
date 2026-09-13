"""
RAG evaluation harness.

    python -m rag.eval.run_eval                 # retrieval + generation
    python -m rag.eval.run_eval --retrieval     # retrieval only (fast, free)
    python -m rag.eval.run_eval --deep          # + LLM faithfulness/relevancy
    python -m rag.eval.run_eval --json out.json

Two deliberate choices, both from CLAUDE.md step 6.1:

1. **Per-category output, not just a headline.** A mean that hides four
   failing adversarial cases behind twelve passing direct ones is worse than
   no metric. The report prints per-category numbers and names every failing
   case so they can be read by hand.

2. **"I don't know" is scored as a correct decision, not as a low-relevance
   answer.** This is the Ragas blind spot the project owner already hit:
   answer_relevancy scores a correct refusal near zero, which drags the
   aggregate down and creates pressure to "fix" the one behaviour you most
   want. Here, no-answer cases are scored on `correct_decline_rate` and are
   excluded from relevancy averages entirely.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from rag.eval.dataset import EVAL_CASES, EvalCase, resolve_gold, summary
from rag.eval.metrics import (
    GenerationScore, RetrievalScore, aggregate, aggregate_generation, score_retrieval,
)
from rag.embeddings import tokenize
from rag.generate import build_context
from rag.pipeline import RagPipeline
from rag.retriever import RERANK_TOP_N, RETRIEVE_TOP_K

K_VALUES = (1, 3, 5, 10, RETRIEVE_TOP_K)


def _reference_overlap(answer: str, reference: str) -> float:
    """Token recall of the reference answer's content words in the produced
    answer. A blunt proxy, reported as a sanity signal rather than a score to
    optimise - it rewards restating the source, which is exactly what a
    grounded extractive answer does."""
    ref_tokens = set(tokenize(reference))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & set(tokenize(answer))) / len(ref_tokens)


def evaluate_retrieval(pipeline: RagPipeline,
                       cases: Optional[list[EvalCase]] = None) -> list[RetrievalScore]:
    cases = cases or EVAL_CASES
    chunks = pipeline.retriever.store.all_chunks()
    scores: list[RetrievalScore] = []

    for case in cases:
        if not case.expects_answer:
            continue  # no gold chunks to measure against
        gold = resolve_gold(case, chunks)
        retrieval = pipeline.retriever.retrieve(case.query)
        retrieved = [c.chunk.chunk_id for c in retrieval.candidates]
        reranked = pipeline.reranker.rerank(case.query, retrieval.candidates,
                                            pipeline.top_n)
        scores.append(score_retrieval(
            case.case_id, case.category, case.query, gold, retrieved,
            [r.chunk_id for r in reranked], ks=K_VALUES,
        ))
    return scores


def evaluate_generation(pipeline: RagPipeline,
                        cases: Optional[list[EvalCase]] = None,
                        deep: bool = False) -> list[GenerationScore]:
    from rag import groundedness

    cases = cases or EVAL_CASES
    scores: list[GenerationScore] = []

    for case in cases:
        response = pipeline.answer(case.query, use_cache=False)
        should = case.expects_answer
        correct = response.answered == should

        score = GenerationScore(
            case_id=case.case_id,
            category=case.category,
            answered=response.answered,
            should_answer=should,
            correct_decision=correct,
            answer=response.answer,
        )
        if response.groundedness:
            score.numeric_grounded = response.groundedness.get("numeric_ok")

        if should and response.answered and case.reference_answer:
            score.reference_overlap = _reference_overlap(response.answer,
                                                          case.reference_answer)

        if deep and response.answered:
            retrieval = pipeline.retriever.retrieve(case.query)
            reranked = pipeline.reranker.rerank(case.query, retrieval.candidates,
                                                pipeline.top_n)
            context = build_context(reranked)
            faith, unsupported = groundedness.check_faithfulness(response.answer, context)
            score.faithfulness = faith
            if unsupported:
                score.note = f"unsupported: {unsupported[0][:120]}"
            score.answer_relevancy = _answer_relevancy(case.query, response.answer)

        if not should and response.answered:
            score.note = "HALLUCINATION RISK: answered a question the corpus " \
                         "does not cover"
        scores.append(score)
    return scores


_RELEVANCY_SYSTEM = """You judge whether an answer actually addresses the question asked.

Score 0.0 to 1.0:
- 1.0 fully addresses the question
- 0.5 partially addresses it, or answers a neighbouring question
- 0.0 does not address it

Judge only relevance to the question. Do NOT penalise an answer for being
short, for citing sources, or for declining - a refusal is handled elsewhere
and should not reach you."""

_RELEVANCY_SCHEMA = {
    "type": "object",
    "properties": {"relevancy": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["relevancy", "reason"],
    "additionalProperties": False,
}


def _answer_relevancy(query: str, answer: str) -> Optional[float]:
    from agents import llm
    if not llm.is_live():
        return None
    try:
        raw = llm.complete_json(_RELEVANCY_SYSTEM,
                                f"Question: {query}\n\nAnswer: {answer}",
                                _RELEVANCY_SCHEMA, model=llm.FAST_MODEL,
                                effort="low", max_tokens=1024)
        return max(0.0, min(1.0, float(raw.get("relevancy", 0.0))))
    except Exception:
        return None


def _print_report(retrieval: list[RetrievalScore], generation: list[GenerationScore]) -> None:
    from agents import llm
    from rag import tuning

    params = tuning.current()
    live = llm.is_live()

    print("=" * 74)
    print("RAG EVALUATION")
    print("=" * 74)
    print(f"Eval set: {json.dumps(summary())}")
    mode = ("LIVE (Claude reranker + generator)" if live
            else "OFFLINE (lexical reranker + extractive generator)")
    print(f"Mode    : {mode}")
    if not live:
        print("          Offline numbers are a FLOOR, not the shipped behaviour:")
        print("          declining and reranking are both harder without a model.")
    print(f"Params  : {params.source} "
          f"(embedder={params.embedder}, reranker={params.reranker}, "
          f"top_n={params.rerank_top_n}, relative_floor={params.relative_floor})")
    if params.source != "tuned":
        print("          WARNING: thresholds are DEFAULTS, not tuned. "
              "Run `python -m rag.eval.tune`.")
    print(f"Retrieval top_k={RETRIEVE_TOP_K}  ->  reranked to top_n={params.rerank_top_n}\n")

    if retrieval:
        agg = aggregate(retrieval, ks=K_VALUES)
        print("--- RETRIEVAL ---")
        print(f"  MRR              : {agg['mrr']}")
        print(f"  recall@k         : {agg['recall_at_k']}")
        print(f"  hit rate@k       : {agg['hit_rate_at_k']}")
        print(f"  precision@k      : {agg['precision_at_k']}")
        if "rerank" in agg:
            r = agg["rerank"]
            print(f"  after rerank     : recall={r['recall']} precision={r['precision']} "
                  f"({r['avg_chunks_shown']} chunks shown)")
        print("\n  by category:")
        for cat, stats in agg["by_category"].items():
            print(f"    {cat:<14} n={stats['cases']:<3} recall@5={stats['recall_at_5']:<6} "
                  f"hit@5={stats['hit_rate_at_5']:<6} mrr={stats['mrr']}")

        misses = [s for s in retrieval if s.hit[5] == 0.0]
        if misses:
            print(f"\n  MISSED at k=5 ({len(misses)}):")
            for s in misses:
                print(f"    - {s.case_id} [{s.category}]: {s.query}")
                print(f"      gold={s.gold}  got={s.retrieved[:3]}")
        dropped = [s for s in retrieval if s.hit[5] == 1.0 and s.rerank_recall == 0.0]
        if dropped:
            print(f"\n  RETRIEVED BUT DROPPED BY RERANKER ({len(dropped)}):")
            for s in dropped:
                print(f"    - {s.case_id}: gold={s.gold} kept={s.reranked}")

    if generation:
        agg = aggregate_generation(generation)
        print("\n--- GENERATION ---")
        for key in ("decision_accuracy", "answer_rate_on_answerable",
                    "correct_decline_rate", "numeric_grounded_rate",
                    "reference_overlap", "faithfulness", "answer_relevancy"):
            if key in agg:
                print(f"  {key:<26}: {agg[key]}")
        if agg.get("hallucinated"):
            print(f"\n  ANSWERED A NO-ANSWER CASE ({len(agg['hallucinated'])}):")
            for case_id in agg["hallucinated"]:
                s = next(g for g in generation if g.case_id == case_id)
                print(f"    - {case_id}: {s.answer[:160]}")
        wrong = [g for g in generation if not g.correct_decision and g.should_answer]
        if wrong:
            print(f"\n  DECLINED AN ANSWERABLE CASE ({len(wrong)}):")
            for g in wrong:
                print(f"    - {g.case_id} [{g.category}]")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the SOP RAG pipeline")
    parser.add_argument("--retrieval", action="store_true",
                        help="retrieval metrics only (no generation)")
    parser.add_argument("--deep", action="store_true",
                        help="add LLM faithfulness and answer-relevancy (needs credentials)")
    parser.add_argument("--category", default=None, help="run one category only")
    parser.add_argument("--json", default=None, help="write full results to this path")
    args = parser.parse_args(argv)

    cases = EVAL_CASES
    if args.category:
        cases = [c for c in EVAL_CASES if c.category == args.category]
        if not cases:
            print(f"No cases in category '{args.category}'", file=sys.stderr)
            return 1

    pipeline = RagPipeline(use_cache=False)
    retrieval = evaluate_retrieval(pipeline, cases)
    generation = [] if args.retrieval else evaluate_generation(pipeline, cases, deep=args.deep)
    _print_report(retrieval, generation)

    if args.json:
        payload = {
            "summary": summary(),
            "retrieval_aggregate": aggregate(retrieval, ks=K_VALUES),
            "retrieval_cases": [s.to_dict() for s in retrieval],
            "generation_aggregate": aggregate_generation(generation),
            "generation_cases": [s.to_dict() for s in generation],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json}")

    # Non-zero exit when the behaviour the project treats as non-negotiable
    # regresses: answering something the corpus does not cover.
    if generation:
        agg = aggregate_generation(generation)
        if agg.get("hallucinated"):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
