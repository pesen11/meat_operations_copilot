"""
Out-of-scope detection for the offline path.

The problem this solves, stated honestly: **lexical relevance scores cannot
tell an unanswerable question from an answerable one.** Measured on this
corpus, the worst gold chunk scores 0.35 and the median non-gold chunk scores
0.40 - the distributions overlap, so no threshold on that score separates
them. Thresholding it harder only trades hallucinations for refusals of
questions the corpus does answer.

A different signal does carry information: whether the question's content
words exist in the corpus AT ALL. "sous vide", "overtime", "thigh", "kilo",
"freezer" appear nowhere in these SOPs. Vocabulary absence is evidence of
topic absence in a way that similarity ranking is not, because ranking is
relative - something always ranks first.

Limits, which matter as much as the signal:

- It is a weak signal used only where nothing better is available. When
  credentials exist, Claude reads the passages and makes this judgement
  properly; this check then acts as a cheap pre-filter, not the decision.
- It cannot catch an out-of-scope question phrased entirely in in-corpus
  vocabulary ("what do we pay for beef" - every word is in the corpus). The
  eval set contains exactly that case (`no-answer-supplier-price` is partly
  of this kind) and the offline path's score on it is reported rather than
  engineered away.
- The threshold is tuned in rag/eval/tune.py, not guessed.
"""

from __future__ import annotations

from functools import lru_cache

from rag.embeddings import tokenize

# Function words and contentless verbs. A query term being out-of-vocabulary
# only means something if the term carries topic information: "we", "know",
# and "enough" being absent from an SOP corpus says nothing at all.
#
# This list is general English, NOT tuned to the eval set - adding a word here
# because one eval case needs it is how a metric gets gamed.
_NON_CONTENT = {
    "we", "i", "you", "they", "it", "he", "she", "our", "my", "your", "their",
    "us", "them", "me", "who", "whom", "whose", "there", "here", "some", "many",
    "much", "more", "most", "less", "least", "other", "another", "same",
    "enough", "whether", "either", "neither", "such", "every", "few", "several",
    "know", "get", "got", "go", "going", "make", "made", "take", "taken",
    "want", "need", "let", "put", "use", "using", "used", "keep", "kept",
    "have", "has", "had", "should", "would", "could", "will", "shall", "may",
    "might", "supposed", "ok", "okay", "please", "thing", "things", "stuff",
    "way", "ways", "long", "far", "back", "before", "after", "while", "during",
    "am", "was", "were", "been", "being", "did", "done", "doing", "say", "said",
}


@lru_cache(maxsize=1)
def _corpus_vocabulary() -> frozenset:
    from rag.chunker import chunk_corpus
    vocab: set[str] = set()
    for chunk in chunk_corpus():
        vocab |= set(tokenize(chunk.embedding_text()))
    return frozenset(vocab)


def reset_vocabulary_cache() -> None:
    _corpus_vocabulary.cache_clear()


def oov_rate(query: str) -> tuple[float, list[str]]:
    """
    Fraction of the query's CONTENT words absent from the whole corpus, plus
    the absent words themselves (for the API's explanation of a decline).

    Computed on the EXPANDED query (rag/query_expansion.py), not the raw one.
    A word the expansion map can translate into corpus vocabulary is a
    phrasing difference, not an unknown subject: "protocols" does not appear
    anywhere in the corpus, but it means "procedures", which appears
    everywhere. Scoring the raw query made that a refusal.
    """
    from rag.query_expansion import expand_query

    vocab = _corpus_vocabulary()
    content = [t for t in set(tokenize(expand_query(query))) if t not in _NON_CONTENT]
    if not content:
        return 0.0, []
    missing = sorted(t for t in content if t not in vocab)
    return len(missing) / len(content), missing


def looks_out_of_scope(query: str, threshold: float | None = None) -> tuple[bool, dict]:
    """True when the question's vocabulary suggests the corpus does not cover it."""
    from rag import tuning

    limit = threshold if threshold is not None else tuning.current().oov_decline_threshold
    rate, missing = oov_rate(query)
    return rate > limit, {"oov_rate": round(rate, 3), "unknown_terms": missing,
                          "threshold": limit}
