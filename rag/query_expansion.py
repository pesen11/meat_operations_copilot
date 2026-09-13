"""
Query expansion for lexical retrieval.

**This exists because the default embedder is lexical, and it is a crutch.**
TF-IDF cannot connect "how cold does a delivery need to be" to a passage
that says "product must arrive at or below 4°C" - there is not one shared
content word. A real dense embedding model closes that gap on its own, which
is why this module is a *retrieval-time query rewrite* and not a change to
the corpus or the index: switch OPS_COPILOT_EMBEDDINGS to `voyage` and it
does nothing harmful, it just stops being load-bearing.

Measured effect is reported by `python -m rag.eval.run_eval`; the two cases
it was built for are `danger-zone` and `temp-receiving-para`.

Design rules:
- Expansion only ADDS terms, never removes or replaces. Recall is the metric
  being protected (CLAUDE.md step 6.2).
- Added terms are appended once, so they weigh less than the operator's own
  repeated words rather than drowning them out.
- The map is small, hand-written, and domain-specific. It is not a thesaurus:
  a general synonym list would connect "cut" to "reduce" and pull cost
  scenarios into food-safety queries.
"""

from __future__ import annotations

import re

# Operator phrasing -> vocabulary the SOPs actually use.
# Keys are matched as whole words (or phrases) in the lowercased query.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    # temperature / cold chain
    "cold": ("temperature", "refrigerated", "cooler"),
    "warm": ("temperature", "danger", "zone"),
    "fridge": ("cooler", "refrigerated"),
    "freezer": ("frozen", "cold", "storage"),
    "chilled": ("refrigerated", "cooler", "temperature"),
    # receiving
    "delivery": ("receiving", "arrive", "vendor", "manifest"),
    "deliveries": ("receiving", "arrive", "vendor"),
    "arrives": ("receiving", "arrive"),
    "shows up": ("receiving", "arrive"),
    "truck": ("delivery", "receiving", "driver"),
    "supplier": ("vendor",),
    # disposal / waste
    "throw away": ("discard", "waste", "disposition"),
    "thrown away": ("discard", "waste"),
    "bin": ("waste", "discard"),
    "sit out": ("danger", "zone", "ambient", "room", "temperature"),
    "left out": ("danger", "zone", "ambient", "temperature"),
    "goes off": ("shelf", "life", "use-by", "spoil"),
    "go bad": ("shelf", "life", "use-by"),
    "expiry": ("use-by", "shelf", "life"),
    "expires": ("use-by", "shelf", "life"),
    # equipment
    "saw": ("band", "saw", "blade"),
    "mincer": ("grinder",),
    "sealer": ("vacuum", "sealer"),
    # cutting
    "spec": ("specification", "thickness", "target", "weight"),
    "portion": ("portioning", "pack", "weight"),
    "fat cap": ("trim", "fat", "cap"),
    # stock
    "run out": ("stockout", "coverage", "short"),
    "ran out": ("stockout", "coverage", "short"),
    "enough": ("coverage", "cover"),
    "how much do we have": ("coverage", "boxes", "hand"),
    # people / shifts
    "opening": ("opening", "checklist", "morning", "shift"),
    "closing": ("closing", "checklist", "close"),

    # --- vocabulary for the documents themselves ---------------------------
    # Operators say "protocols", "rules", "guidelines"; the SOPs say
    # "procedure" and "practices". None of these synonyms appear in the corpus
    # at all, so without this a perfectly answerable question looks like it is
    # about a subject the corpus has never heard of. This is the class of gap
    # that made "what are some safety protocols?" get refused.
    "protocol": ("procedure", "practices", "sop", "requirement"),
    "protocols": ("procedure", "practices", "sop", "requirement"),
    "rules": ("procedure", "practices", "requirement", "sop"),
    "guideline": ("procedure", "practices", "requirement"),
    "guidelines": ("procedure", "practices", "requirement"),
    "policy": ("procedure", "sop", "practices"),
    "policies": ("procedure", "sop", "practices"),
    "standards": ("spec", "requirement", "procedure"),
    "steps": ("procedure", "checklist"),
    "process": ("procedure", "practices"),
    "instructions": ("procedure", "spec", "checklist"),
    "best practice": ("procedure", "practices", "requirement"),
    "safety": ("safety", "food", "hygiene", "contamination", "sanitize"),
    "hygiene": ("hygiene", "ppe", "hand", "wash", "personal"),
}


def expand_query(query: str) -> str:
    """Return the query with domain synonyms appended once each."""
    lowered = query.lower()
    additions: list[str] = []
    seen: set[str] = set(re.findall(r"[a-z]+", lowered))

    for trigger, terms in _SYNONYMS.items():
        if " " in trigger:
            present = trigger in lowered
        else:
            present = re.search(rf"\b{re.escape(trigger)}\b", lowered) is not None
        if not present:
            continue
        for term in terms:
            if term not in seen:
                additions.append(term)
                seen.add(term)

    return f"{query} {' '.join(additions)}" if additions else query


def expansion_terms(query: str) -> list[str]:
    """Just the added terms - for eval reporting and debugging."""
    expanded = expand_query(query)
    return expanded[len(query):].split() if expanded != query else []
