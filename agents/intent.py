"""
Question -> structured Plan.

This is one of only two jobs Claude has in this project (the other is
narration). Even here the model is not trusted blindly: whatever it returns
is validated against the real catalog, and an unknown SKU is rejected rather
than passed downstream where it would become a confusing KeyError.

Offline (no API key) the keyword parser below runs instead. It is weaker on
phrasing but produces the same Plan shape, so the graph behaves identically
- which is what makes the whole pipeline testable without credentials.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from data.catalog import products_by_sku, yield_profile_by_sku, primal_reference_box_weight_kg
from agents import llm
from agents.state import Plan
from models.provenance import Evidence
from simulation.history_analysis import earliest_sales_date, latest_sales_date
from simulation.seasonal_planning import PERIODS, resolve_period
from simulation.time_windows import mentions_a_window, resolve_window

INTENTS = ("scenario", "seasonal_planning", "inventory_status", "history",
           "catalog", "unsupported")

# Byte-stable system prompt - it is sent as a cached block, so interpolating
# anything per-request here would silently destroy the cache hit rate.
_SYSTEM = """You are the Manager node of a butcher-shop operations copilot.

Your ONLY job is to turn an operator's question into a structured plan. You
never compute, estimate, or state any number about the business; deterministic
Python does all of that after you hand off.

Choose exactly one intent:
- "scenario": a what-if about changing production or demand for a SPECIFIC
  product by a size the operator supplies ("what if we cut 15% more chuck
  roast", "should we scale back brisket 20%").
- "seasonal_planning": asks HOW MUCH to produce for a season or event, rather
  than proposing a size themselves ("how much more should we make for
  Christmas?", "what should we plan for the long weekend?", "what's coming up
  that we need to prepare for?"). The shop has confirmed seasonal multipliers,
  so this is answerable even when no product and no percentage is named.
- "inventory_status": a question about current stock, boxes on hand, kg on
  hand, coverage, what needs restocking or reordering, stockouts, or
  cutting-labor capacity. "How many boxes of ribeye do we have" and "what
  do I need to restock" are both this, NOT scenarios.
- "history": a question about past sales, trends, best OR worst sellers, how
  a product or the shop performed over any period ("how did we do
  yesterday", "how was ribeye last week", "what is selling poorly", "what
  should we order less of"), and how weekends trade versus weekdays. Any
  question carrying a time expression is this unless it proposes a change.
- "catalog": a question about what products/primals exist, prices, or yields.
- "unsupported": anything this operations system cannot answer from its own
  data (food-safety procedure, HR, general cooking questions, and so on).

For "scenario" you must identify the product_sku from the catalog list given
in the user message, and a demand_multiplier: 1.15 for "15% more", 0.85 for
"15% less", 1.0 if no size is stated. If the question names a product that is
not in the catalog list, set product_sku to null and intent to "unsupported".

For "seasonal_planning" set `period` to one of christmas, long_weekend,
summer, or regular. product_sku may be null (plan the whole catalog) or a
specific SKU if the operator named one. Do NOT set demand_multiplier - the
confirmed seasonal multiplier is applied downstream, and a number from you
would override a business figure the shop already owns.

Never invent a SKU. Use only SKUs that appear verbatim in the catalog list."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "product_sku": {"type": ["string", "null"]},
        "source_primal": {"type": ["string", "null"]},
        "demand_multiplier": {"type": "number"},
        "period": {"type": ["string", "null"],
                   "enum": ["christmas", "long_weekend", "summer", "regular", None]},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "product_sku", "source_primal", "demand_multiplier",
                 "period", "reasoning"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Deterministic keyword parser (offline path, and the validator for the LLM)
# ---------------------------------------------------------------------------

_INCREASE_WORDS = ("more", "increase", "increasing", "up", "boost", "scale up", "raise", "extra")
# "back" is listed on its own because operators split the phrasal verb around
# the product: "cut ground beef medium back 30%" has no contiguous "cut back".
_DECREASE_WORDS = ("less", "fewer", "decrease", "reduce", "reducing", "cut back",
                   "scale back", " back", "down", "drop", "trim back")

_INVENTORY_WORDS = ("stock", "inventory", "boxes", "on hand", "cover", "coverage",
                    "stockout", "short", "capacity", "labour", "labor", "cutter",
                    # "how many kg of X do we have" is an inventory question,
                    # not a scenario - without these it fell through to the
                    # bare-SKU branch and was simulated as a what-if.
                    "do we have", "have left", "left in", "restock", "reorder",
                    "re-order", "run out", "running out", "need to order",
                    "in the cooler", "in the freezer", "on the floor")
_HISTORY_WORDS = ("last month", "trend", "history", "historical", "sold", "selling",
                  "best seller", "bestseller", "top seller", "past", "last week", "year",
                  # Performance questions - these are history questions even
                  # when no time word appears, and each one used to land in
                  # "unsupported".
                  "sales", "revenue", "how did we do", "how are we doing",
                  "how was", "how were", "performance", "performing",
                  "doing well", "doing good", "doing poorly", "doing badly",
                  "worst", "slowest", "slow mover", "poor", "underperform",
                  "order less", "order fewer", "cut back on", "drop from",
                  "moving well", "not moving", "dead stock", "focus on")
_CATALOG_WORDS = ("what products", "which products", "list", "catalog", "catalogue",
                  "price", "yield", "primals")
_SCENARIO_WORDS = ("what if", "what happens", "if we", "should we", "scenario",
                   "increase", "decrease", "more", "less", "reduce", "boost")

# Phrasing that asks HOW MUCH rather than proposing a size. Paired with a
# recognised season, this is a planning question the shop's confirmed
# multipliers can answer even with no product and no percentage named.
_PLANNING_WORDS = ("how much", "how many", "should we do", "should we make",
                   "should we cut", "should we produce", "should we order",
                   "what should we", "how do we prepare", "prepare for",
                   "plan for", "planning for", "get ready", "ramp up",
                   "what is coming", "whats coming", "what's coming",
                   "coming up")

# Procedural / food-safety phrasing belongs to the SOP knowledge base, not to
# the simulation graph. These questions route to "unsupported" here so the API
# layer can hand them to the RAG pipeline instead of the ops agents - without
# this, "how do I trim a brisket fat cap" matches the Brisket product and gets
# answered with a margin projection, which is a genuinely wrong answer rather
# than merely an unhelpful one.
_SOP_WORDS = ("how do i", "how should", "procedure", "sop", "temperature", "temp ",
              "food safety", "sanitis", "sanitiz", "clean", "label", "allergen",
              "cross-contamination", "cross contamination", "haccp", "recall",
              "trim the", "trimming", "how to", "what's the correct", "spec for",
              "thickness", "shelf life", "store ", "storage")

# Name words too generic to identify a product on their own.
_GENERIC_NAME_WORDS = {"beef", "pork", "steak", "roast", "whole", "thin", "thinly",
                       "sliced", "cut", "boneless", "medium", "meat", "removed",
                       "cross", "trimmed", "eye", "of", "bnls", "cap"}

# A product matches when the question mentions most of its distinctive words
# AND is clearly ahead of the next-best product. Tuned against the phrasings
# in tests/test_intent.py - move these together, not independently.
MIN_PRODUCT_MATCH_SCORE = 0.6
MATCH_MARGIN = 0.15


def _match_product(question: str) -> Optional[str]:
    """
    Match a product by exact SKU, exact name, then by best token overlap.

    Token overlap is scored rather than first-match, because operators say
    "ribeye steak" for "Boneless Ribeye Steak" and "chuck roast" for "Chuck
    Roast"; a first-match-wins loop returns whichever happens to be earlier
    in the catalog. Generic words are excluded from scoring so that "steak"
    alone matches nothing rather than matching the first steak in the list.
    """
    q = question.lower()
    for sku in products_by_sku:
        if sku.lower() in q:
            return sku
    for sku, product in sorted(products_by_sku.items(), key=lambda kv: -len(kv[1].name)):
        if product.name.lower() in q:
            return sku

    q_tokens = set(re.findall(r"[a-z]+", q))
    scores: list[tuple[float, str]] = []
    for sku, product in products_by_sku.items():
        # A parenthetical is an alias, not part of the required name: nobody
        # asks for "Pulled Pork Roast (Boston Butt)" in full, and counting
        # "boston"/"butt" as required tokens makes "pulled pork roast" fail.
        primary = re.sub(r"\(.*?\)", " ", product.name.lower())
        name_tokens = set(re.findall(r"[a-z]+", primary))
        distinctive = name_tokens - _GENERIC_NAME_WORDS
        if not distinctive:
            continue
        hits = distinctive & q_tokens
        if not hits:
            continue
        # Fraction of the product's distinctive words the question mentions,
        # with a small bonus for also matching generic words ("ribeye steak"
        # should beat a bare "ribeye").
        score = len(hits) / len(distinctive) + 0.1 * len(name_tokens & q_tokens & _GENERIC_NAME_WORDS)
        scores.append((score, sku))

    if not scores:
        return None
    scores.sort(reverse=True)
    best_score, best_sku = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0

    # Accept only a clear winner. Requiring a full distinctive-token match
    # would reject "osso buco" for "Osso Buco Cross-Cut Shank"; accepting any
    # partial hit would let a bare "pork" pick a product at random. A
    # majority of the distinctive words, well ahead of the runner-up, is the
    # line that handles both.
    if best_score >= MIN_PRODUCT_MATCH_SCORE and best_score >= runner_up + MATCH_MARGIN:
        return best_sku
    return None


def _match_primal(question: str) -> Optional[str]:
    q = question.lower()
    for primal in sorted(primal_reference_box_weight_kg(), key=len, reverse=True):
        if primal.lower() in q:
            return primal
    return None


def _extract_multiplier(question: str) -> float:
    """Pull a percentage out of the question and give it a direction.

    Defaults to 1.0 (no change) when no percentage is present - a scenario
    question with no size is still a valid baseline request, and guessing a
    magnitude would be inventing a number.
    """
    q = question.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if not match:
        match = re.search(r"(?:by\s+)?(\d+(?:\.\d+)?)\s*percent", q)
    if not match:
        return 1.0
    pct = float(match.group(1)) / 100

    # Direction is decided by the word nearest the percentage, not by "any
    # decrease word anywhere" - "increase chuck roast 15% instead of cutting
    # back on ribeye" contains both.
    span_start = max(0, match.start() - 60)
    context = q[span_start:match.end() + 40]
    decrease = any(w in context for w in _DECREASE_WORDS)
    increase = any(w in context for w in _INCREASE_WORDS)
    if decrease and not increase:
        return round(1 - pct, 4)
    return round(1 + pct, 4)


def extract_explicit_multiplier(question: str) -> Optional[float]:
    """
    The size the OPERATOR stated, or None if they stated none.

    _extract_multiplier() defaults to 1.0 when no percentage appears, which
    is the right value for a projection and the wrong value for a decision:
    it makes "what if we keep production flat" and "how much more should we
    make" indistinguishable, because both arrive downstream as 1.0. The
    second question is asking the SYSTEM to supply the number, and only a
    None can say so.
    """
    q = question.lower()
    if not (re.search(r"\d+(?:\.\d+)?\s*%", q)
            or re.search(r"(?:by\s+)?\d+(?:\.\d+)?\s*percent", q)):
        return None
    return _extract_multiplier(question)


def looks_like_sop_question(question: str) -> bool:
    """True for procedural/food-safety questions that belong to the SOP
    knowledge base rather than the simulation graph."""
    q = question.lower()
    return any(w in q for w in _SOP_WORDS)


def resolve_question_window(question: str, as_of: Optional[str] = None) -> dict:
    """
    The time window a question is asking about, as a plain dict on the Plan.

    Anchored to the last date in the DATA, not to today: the generated
    history ends on a fixed date, so anchoring "last week" to the wall clock
    returns an empty window that reads as "we sold nothing" - a wrong answer
    rather than an unhelpful one.
    """
    anchor = date.fromisoformat(as_of) if as_of else latest_sales_date()
    return resolve_window(question, anchor=anchor,
                          earliest=earliest_sales_date()).to_dict()


def keyword_plan(question: str, as_of: Optional[str] = None) -> Plan:
    q = question.lower()
    sku = _match_product(question)
    primal = _match_primal(question)
    multiplier = _extract_multiplier(question)
    window = resolve_question_window(question, as_of)

    # Seasonal planning is checked BEFORE the product/percentage logic,
    # because these questions deliberately name neither. "How much more
    # should we make for Christmas?" has no SKU and no number, which is
    # exactly why it used to fall through to "unsupported".
    period = resolve_period(question)
    asks_what_is_coming = any(
        w in q for w in ("coming up", "what is coming", "whats coming",
                         "what's coming", "prepare for", "upcoming",
                         "get ready"))
    if period is None and asks_what_is_coming:
        # No season named: answer with the seasonal calendar rather than
        # guessing which period they meant.
        return Plan(
            intent="seasonal_planning",
            product_sku=None,
            source_primal=None,
            demand_multiplier=1.0,
            period=None,
            as_of=as_of,
            window=window,
            reasoning="Asks what is coming up; answered with the seasonal "
                      "calendar rather than a single period's plan.",
            resolved_by="keyword",
        )
    if period and (any(w in q for w in _PLANNING_WORDS) or sku is None):
        return Plan(
            intent="seasonal_planning",
            product_sku=sku,
            source_primal=(yield_profile_by_sku[sku].source_primal if sku else primal),
            demand_multiplier=1.0,
            period=period,
            as_of=as_of,
            window=window,
            reasoning=f"Seasonal planning question for the {period} period; the "
                      f"shop's confirmed multiplier is applied downstream.",
            resolved_by="keyword",
        )

    if looks_like_sop_question(question):
        return Plan(
            intent="unsupported",
            product_sku=None,
            source_primal=primal,
            demand_multiplier=1.0,
            period=None,
            as_of=as_of,
            window=window,
            reasoning="Procedural/food-safety question - belongs to the SOP "
                      "knowledge base (rag/), not the operations simulation.",
            resolved_by="keyword",
        )

    # Inventory is tested BEFORE the scenario branch. "How many kg of chuck
    # roast do we have" names a product and contains no percentage, and used
    # to fall through to the bare-SKU branch and get simulated as a what-if -
    # an answer to a question nobody asked.
    inventory_signal = any(w in q for w in _INVENTORY_WORDS)
    history_signal = any(w in q for w in _HISTORY_WORDS)
    scenario_signal = any(w in q for w in _SCENARIO_WORDS)

    if inventory_signal and multiplier == 1.0:
        intent = "inventory_status"
    elif sku and (scenario_signal or multiplier != 1.0):
        intent = "scenario"
    elif inventory_signal:
        intent = "inventory_status"
    elif history_signal or window["weekends_only"] or window["explicit"]:
        # A question carrying any time expression at all is asking about
        # history, even when it uses none of the history keywords: "how did
        # we do yesterday" has no keyword and is unambiguously a history
        # question.
        intent = "history"
    elif any(w in q for w in _CATALOG_WORDS):
        intent = "catalog"
    elif sku:
        intent = "scenario"
    else:
        intent = "unsupported"

    return Plan(
        intent=intent,
        product_sku=sku,
        source_primal=primal or (yield_profile_by_sku[sku].source_primal if sku else None),
        demand_multiplier=multiplier,
        period=None,
        as_of=as_of,
        window=window,
        reasoning="Parsed by keyword matching (no LLM credentials configured).",
        resolved_by="keyword",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _catalog_listing() -> str:
    lines = [f"- {sku}: {p.name} (primal: {yield_profile_by_sku[sku].source_primal})"
             for sku, p in products_by_sku.items()]
    return "\n".join(lines)


def build_request(question: str, as_of: Optional[str] = None) -> "UserRequest":
    """
    Parse a question into a validated UserRequest, with provenance.

    Layered ON TOP of the existing intent logic rather than replacing it:
    _plan_for() still decides the intent, the SKU and the period exactly as
    before (and is still validated against the catalog), and this function's
    only job is to record WHERE each resolved field came from.

    The distinctions it captures, which the flat Plan could not:

      period "christmas" because the operator typed the word   -> user
      period "christmas" because the calendar found it next    -> inferred
      multiplier 1.15 because the operator said "15% more"     -> user
      multiplier absent because they asked how much            -> None

    Those look identical in a dict of floats and strings, and they carry
    completely different authority.
    """
    from models.provenance import from_user, inferred as inferred_from, system_default
    from models.request import Intent, UserRequest

    plan = _plan_for(question, as_of)
    keyword_sku = _match_product(question)
    stated_period = resolve_period(question)
    stated_multiplier = extract_explicit_multiplier(question)

    try:
        intent = Intent(plan["intent"])
    except ValueError:
        intent = Intent.UNSUPPORTED

    # A scenario with no product is not answerable as a scenario; UserRequest
    # validates this, so demote here rather than let it raise downstream.
    if intent == Intent.SCENARIO and not plan.get("product_sku"):
        intent = Intent.UNSUPPORTED

    resolved_by = plan.get("resolved_by", "keyword")
    intent_evidence = (from_user(intent.value, "matched question keywords")
                       if resolved_by == "keyword"
                       else Evidence(value=intent.value, source="llm_inference",
                                     basis="model read of the question"))

    sku_evidence = None
    if plan.get("product_sku"):
        sku = plan["product_sku"]
        # Keyword matching requires the product's own distinctive words to be
        # present, so a keyword hit really is the operator naming it. A SKU
        # only the model produced is the model's reading, however well it
        # validated against the catalog.
        sku_evidence = (from_user(sku, "named in the question")
                        if sku == keyword_sku
                        else Evidence(value=sku, source="llm_inference",
                                      basis="model matched it to the catalog"))

    primal_evidence = None
    if plan.get("source_primal"):
        primal = plan["source_primal"]
        primal_evidence = (
            from_user(primal, "named in the question") if _match_primal(question)
            else inferred_from(primal, f"the primal {plan.get('product_sku')} is cut from"))

    period_evidence = None
    if plan.get("period"):
        period = plan["period"]
        period_evidence = (from_user(period, "named in the question")
                           if stated_period == period
                           else inferred_from(period, "nearest elevated period on the calendar"))

    multiplier_evidence = None
    if stated_multiplier is not None:
        multiplier_evidence = from_user(
            stated_multiplier, f"operator asked for {stated_multiplier:.0%} of current demand")
    elif plan.get("demand_multiplier", 1.0) != 1.0 and resolved_by == "llm":
        multiplier_evidence = Evidence(
            value=float(plan["demand_multiplier"]), source="llm_inference",
            basis="model read a size from the question")

    return UserRequest(
        intent=intent,
        intent_evidence=intent_evidence,
        product_sku=sku_evidence,
        source_primal=primal_evidence,
        period=period_evidence,
        explicit_multiplier=multiplier_evidence,
        window=plan.get("window"),
        as_of=as_of,
        resolved_by=resolved_by,
        reasoning=plan.get("reasoning", ""),
    )


def parse_question(question: str, as_of: Optional[str] = None) -> Plan:
    """
    Parse a question into a validated Plan.

    Now derived from build_request() so every consumer gets the provenance
    and the explicit-vs-inferred distinction for free, while the Plan shape
    every existing node reads is unchanged.
    """
    return build_request(question, as_of).to_plan()


def _plan_for(question: str, as_of: Optional[str] = None) -> Plan:
    """The original parse: LLM when live, keywords otherwise."""
    fallback = keyword_plan(question, as_of)

    if not llm.is_live():
        return fallback

    user = (
        f"Catalog:\n{_catalog_listing()}\n\n"
        f"Operator question: {question}"
    )
    try:
        raw = llm.complete_json(_SYSTEM, user, _SCHEMA, effort="low")
    except Exception:
        # A planning failure must not take the whole request down - the
        # keyword parse is a genuine answer, just a blunter one.
        return fallback

    intent = raw.get("intent")
    if intent not in INTENTS:
        return fallback

    sku = raw.get("product_sku")
    if sku is not None and sku not in products_by_sku:
        # The model named something that isn't ours. Trust the catalog, not
        # the model: retry the match deterministically, and demote to
        # unsupported if that finds nothing either.
        sku = fallback["product_sku"]
        if sku is None and intent == "scenario":
            intent = "unsupported"

    multiplier = raw.get("demand_multiplier", 1.0)
    if not isinstance(multiplier, (int, float)) or multiplier <= 0:
        multiplier = fallback["demand_multiplier"]

    primal = raw.get("source_primal")
    if primal not in primal_reference_box_weight_kg():
        primal = yield_profile_by_sku[sku].source_primal if sku else fallback["source_primal"]

    # A seasonal plan without a valid period is not answerable. Trust the
    # deterministic resolver over the model here, and demote rather than
    # guess a season.
    period = raw.get("period")
    if period not in PERIODS:
        period = resolve_period(question)
    if intent == "seasonal_planning" and period is None:
        intent = fallback["intent"]
        period = fallback.get("period")

    return Plan(
        intent=intent,
        product_sku=sku,
        source_primal=primal,
        demand_multiplier=float(multiplier),
        period=period,
        as_of=as_of,
        # The window is resolved deterministically either way. Asking the
        # model for dates would be asking it for numbers, and "last week"
        # has one correct answer that Python can compute.
        window=fallback["window"],
        reasoning=str(raw.get("reasoning", ""))[:500],
        resolved_by="llm",
    )
