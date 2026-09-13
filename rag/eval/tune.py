"""
Threshold tuning against the labelled eval set.

    python -m rag.eval.tune            # sweep, report, write tuned_params.json
    python -m rag.eval.tune --dry-run  # sweep and report, write nothing

CLAUDE.md is explicit that the semantic-cache threshold must be validated
empirically rather than guessed (step 6.5). The same argument applies to the
reranker floors and the out-of-scope threshold, so all four are swept here.

What is being traded off, per group:

- **Reranker floors** (min_relevance, relative_floor): keeping gold chunks
  (recall) against not burying them in noise (precision). Recall is weighted
  higher, per step 6.2 - a dropped gold chunk cannot be recovered downstream,
  a surviving noisy chunk can be ignored by the generator.

- **Decline thresholds** (extractive_min_relevance, oov_decline_threshold):
  correctly declining out-of-corpus questions against not refusing questions
  the corpus does answer. Scored as balanced accuracy so the sweep cannot win
  by declining everything - which is the degenerate optimum an unbalanced
  score would happily find.

- **Semantic cache threshold**: true hits (a paraphrase of a cached question)
  against false hits (a DIFFERENT question that happens to look similar).
  These are scored asymmetrically on purpose: a false cache hit serves a
  wrong answer with no indication anything went wrong, while a missed cache
  hit costs only latency. The sweep therefore takes the highest threshold
  that still catches paraphrases, not the one that maximises raw hit rate,
  and refuses to choose a value with ANY false positive.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import product
from typing import Optional

from rag import tuning
from rag.embeddings import cosine_similarity, get_embedder
from rag.eval.dataset import EVAL_CASES, paraphrase_pairs, resolve_gold
from rag.eval.metrics import recall_at_k
from rag.pipeline import RagPipeline
from rag.reranker import RerankedChunk, get_reranker
from rag.retriever import RERANK_TOP_N, Retriever
from rag.scope import oov_rate

# Recall is worth more than precision here (CLAUDE.md step 6.2).
RECALL_WEIGHT = 0.75
PRECISION_WEIGHT = 0.25

# The eval set has 25 queries. A sweep on 25 queries can find a threshold
# that sits 0.01 above the most-similar unrelated pair IT HAPPENED TO SEE,
# which is overfitting, not tuning - real traffic will contain a closer pair.
# So the semantic cache threshold gets two guards the sweep cannot override:
#   - an absolute floor, below which no amount of eval evidence justifies
#     treating two queries as the same question;
#   - a margin above the worst observed false pair, so the chosen value is
#     not sitting on the edge of the observed distribution.
# Both loosen automatically as the eval set grows, because max_false rises.
MIN_SAFE_CACHE_THRESHOLD = 0.60
CACHE_THRESHOLD_MARGIN = 0.05


def _collect(pipeline: RagPipeline):
    """Score every candidate once, so the sweep re-thresholds cached scores
    instead of re-running retrieval hundreds of times."""
    chunks = pipeline.retriever.store.all_chunks()
    rows = []
    for case in EVAL_CASES:
        candidates = pipeline.retriever.retrieve(case.query).candidates
        # top_n large + floors disabled: we want the full scored list.
        scored = _score_all(pipeline, case.query, candidates)
        gold = resolve_gold(case, chunks) if case.expects_answer else []
        rows.append({
            "case": case,
            "gold": gold,
            "scored": scored,
            "oov": oov_rate(case.query)[0],
        })
    return rows


def _score_all(pipeline: RagPipeline, query: str,
               candidates) -> list[RerankedChunk]:
    saved = tuning.current()
    # Temporarily disable both floors so `rerank` returns everything scored.
    tuning._current = replace(saved, min_relevance=0.0, relative_floor=0.0)
    try:
        return pipeline.reranker.rerank(query, candidates, top_n=10_000)
    finally:
        tuning._current = saved


def _apply_floors(scored: list[RerankedChunk], min_rel: float, rel_floor: float,
                  top_n: int) -> list[RerankedChunk]:
    if not scored:
        return []
    ranked = sorted(scored, key=lambda r: -r.relevance)
    if ranked[0].relevance < min_rel:
        return []
    cutoff = max(min_rel, ranked[0].relevance * rel_floor)
    return [r for r in ranked if r.relevance >= cutoff][:top_n]


def sweep_reranker(rows) -> tuple[float, float, int, dict]:
    """Sweeps top_n too: it is the knob that most directly trades the
    'model only sees good chunks' requirement (step 6.3) against recall on
    multi-hop cases, which need three chunks to be answerable at all."""
    best = None
    answerable = [r for r in rows if r["case"].expects_answer]

    for min_rel, rel_floor, top_n in product(
            [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90],
            [3, 4, 5, 6]):
        recalls, precisions, shown = [], [], []
        for row in answerable:
            kept = _apply_floors(row["scored"], min_rel, rel_floor, top_n)
            kept_ids = [k.chunk_id for k in kept]
            recalls.append(recall_at_k(kept_ids, row["gold"], len(kept_ids) or 1))
            precisions.append(
                len(set(kept_ids) & set(row["gold"])) / len(kept_ids) if kept_ids else 0.0)
            shown.append(len(kept_ids))
        recall = sum(recalls) / len(recalls)
        precision = sum(precisions) / len(precisions)
        score = RECALL_WEIGHT * recall + PRECISION_WEIGHT * precision
        entry = {"min_relevance": min_rel, "relative_floor": rel_floor,
                 "top_n": top_n,
                 "recall": round(recall, 4), "precision": round(precision, 4),
                 "avg_shown": round(sum(shown) / len(shown), 2),
                 "score": round(score, 4)}
        # Tie-break toward FEWER chunks shown: same score with less noise in
        # the generation prompt is strictly better.
        if best is None or entry["score"] > best["score"] or (
                entry["score"] == best["score"] and entry["avg_shown"] < best["avg_shown"]):
            best = entry
    return best["min_relevance"], best["relative_floor"], best["top_n"], best


def sweep_decline(rows, min_rel: float, rel_floor: float,
                  top_n: int = RERANK_TOP_N) -> tuple[float, float, dict]:
    """Balanced accuracy over (correctly answers) and (correctly declines)."""
    best = None
    for extractive_min, oov_threshold in product(
            [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            [0.20, 0.25, 0.30, 0.34, 0.40, 0.50, 0.60, 1.01]):
        answer_correct, decline_correct = [], []
        for row in rows:
            kept = _apply_floors(row["scored"], min_rel, rel_floor, top_n)
            would_answer = bool(kept) and kept[0].relevance >= extractive_min \
                and row["oov"] <= oov_threshold
            if row["case"].expects_answer:
                answer_correct.append(1.0 if would_answer else 0.0)
            else:
                decline_correct.append(0.0 if would_answer else 1.0)
        sens = sum(answer_correct) / len(answer_correct) if answer_correct else 0.0
        spec = sum(decline_correct) / len(decline_correct) if decline_correct else 0.0
        entry = {"extractive_min_relevance": extractive_min,
                 "oov_decline_threshold": oov_threshold,
                 "answer_rate_on_answerable": round(sens, 4),
                 "correct_decline_rate": round(spec, 4),
                 "balanced_accuracy": round((sens + spec) / 2, 4)}
        if best is None or entry["balanced_accuracy"] > best["balanced_accuracy"] or (
                entry["balanced_accuracy"] == best["balanced_accuracy"]
                and entry["correct_decline_rate"] > best["correct_decline_rate"]):
            best = entry
    return (best["extractive_min_relevance"], best["oov_decline_threshold"], best)


def sweep_cache_threshold() -> tuple[float, dict]:
    """
    Highest threshold that still catches every known paraphrase, subject to
    zero false positives against every other question pair.

    Asymmetric on purpose: a false hit returns a confidently wrong cached
    answer, a miss costs only latency.
    """
    embedder = get_embedder()
    texts = [c.query for c in EVAL_CASES]
    # TF-IDF needs a fitted vocabulary; fit on the corpus the cache will see.
    from rag.chunker import chunk_corpus
    embedder.fit([c.embedding_text() for c in chunk_corpus()] + texts)

    vectors = {c.case_id: embedder.embed_query(c.query) for c in EVAL_CASES}
    pairs = paraphrase_pairs()

    true_sims = [float(cosine_similarity(vectors[p.case_id],
                                         vectors[o.case_id].reshape(1, -1))[0])
                 for o, p in pairs]
    false_sims = []
    by_id = {c.case_id: c for c in EVAL_CASES}
    paraphrase_of = {p.case_id: p.paraphrase_of for _, p in pairs}
    for a in EVAL_CASES:
        for b in EVAL_CASES:
            if a.case_id >= b.case_id:
                continue
            if paraphrase_of.get(a.case_id) == b.case_id or \
               paraphrase_of.get(b.case_id) == a.case_id:
                continue
            false_sims.append(float(cosine_similarity(
                vectors[a.case_id], vectors[b.case_id].reshape(1, -1))[0]))

    max_false = max(false_sims) if false_sims else 0.0
    min_true = min(true_sims) if true_sims else 1.0

    if min_true > max_false + CACHE_THRESHOLD_MARGIN:
        # Separable with room to spare: sit in the gap, biased to the safe side.
        raw = max_false + 0.6 * (min_true - max_false)
        note = "paraphrases and distinct questions are separable"
    else:
        # Not separable. Refuse any value that would produce a false hit; put
        # the margin above the worst false pair and accept that some
        # paraphrases miss the cache.
        raw = max_false + CACHE_THRESHOLD_MARGIN
        note = ("NOT separable at this embedder: some distinct questions score "
                "as similar as some paraphrases do. Threshold set above the "
                "worst observed false pair, so paraphrase cache hits are "
                "sacrificed to avoid serving a wrong cached answer.")

    if raw < MIN_SAFE_CACHE_THRESHOLD:
        note += (f" Sweep suggested {raw:.3f}, raised to the "
                 f"{MIN_SAFE_CACHE_THRESHOLD} safety floor: a value that low is "
                 f"not supportable from {len(EVAL_CASES)} queries and would "
                 f"false-hit on unseen traffic.")
        raw = MIN_SAFE_CACHE_THRESHOLD
    threshold = round(min(0.995, raw), 4)

    caught = sum(1 for s in true_sims if s >= threshold)
    return threshold, {
        "threshold": threshold,
        "paraphrase_pairs": len(pairs),
        "paraphrases_cached": caught,
        "min_true_similarity": round(min_true, 4),
        "max_false_similarity": round(max_false, 4),
        "false_positives": sum(1 for s in false_sims if s >= threshold),
        "note": note,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tune RAG thresholds against the eval set")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)

    pipeline = RagPipeline(use_cache=False)
    rows = _collect(pipeline)

    min_rel, rel_floor, top_n, rerank_report = sweep_reranker(rows)
    extractive_min, oov_threshold, decline_report = sweep_decline(
        rows, min_rel, rel_floor, top_n)
    cache_threshold, cache_report = sweep_cache_threshold()

    print("=" * 74)
    print("THRESHOLD TUNING")
    print("=" * 74)
    print("\n[reranker floors]")
    print(json.dumps(rerank_report, indent=2))
    print("\n[decline thresholds]")
    print(json.dumps(decline_report, indent=2))
    print("\n[semantic cache threshold]")
    print(json.dumps(cache_report, indent=2))

    params = tuning.TunedParams(
        min_relevance=min_rel,
        relative_floor=rel_floor,
        extractive_min_relevance=extractive_min,
        oov_decline_threshold=oov_threshold,
        rerank_top_n=top_n,
        semantic_threshold=cache_threshold,
        embedder=getattr(pipeline.retriever.store.embedder, "name", "tfidf"),
        reranker=getattr(pipeline.reranker, "name", "lexical"),
        notes=cache_report["note"],
    )

    if args.dry_run:
        print("\n--dry-run: nothing written.")
    else:
        path = tuning.save(params)
        tuning.reset()
        print(f"\nWrote {path}")
        print(json.dumps(params.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
