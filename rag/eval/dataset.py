"""
The labelled RAG eval set.

CLAUDE.md step 6.1: build this BEFORE optimising anything. Every retrieval
weight, the reranker's relevance floor, and the semantic-cache threshold are
tuned against this file, so it is written first and treated as the fixed
target rather than adjusted to flatter a result.

**Gold chunks are referenced by (doc_id, heading substring), not by chunk
id.** Chunk ids shift whenever chunking parameters change, and an eval set
that silently stops pointing at the right passage after a chunker tweak is
worse than no eval set - it reports a regression that is really a
relabelling. `resolve_gold()` maps these to live chunk ids at eval time and
raises loudly if a label matches nothing.

Case categories:
  - `direct`     : the answer is stated plainly in one section.
  - `paraphrase` : same question as a `direct` case, worded differently.
                   These are the semantic-cache tuning set (true hits).
  - `multi_hop`  : needs two sections; tests recall, not just top-1.
  - `adversarial`: near-miss vocabulary that a bag-of-words system gets
                   wrong. The pulled-pork fat cap case is the one CLAUDE.md
                   specifically calls out.
  - `broad`      : open-ended "what should I know about X" questions. These
                   were MISSING from the first version of this eval set, and
                   their absence was a real hole: the threshold sweep tuned
                   against a set where every answerable question used corpus
                   vocabulary and named one narrow fact, so it happily chose
                   an aggressive decline threshold. The first broad question a
                   real user asked ("what are some safety protocols I should
                   follow?") was refused, because "protocol" appears zero
                   times in the corpus, which says "procedure". An eval set
                   shapes the system; its gaps become the system's gaps.
  - `no_answer`  : genuinely not in the corpus. The system must decline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rag.chunker import Chunk


@dataclass
class GoldRef:
    doc_id: str
    heading_contains: str


@dataclass
class EvalCase:
    case_id: str
    query: str
    category: str
    gold: list[GoldRef] = field(default_factory=list)
    reference_answer: Optional[str] = None
    # For paraphrase cases: the case_id this is a paraphrase of. Used by the
    # cache tuner as a known true-positive pair.
    paraphrase_of: Optional[str] = None
    notes: str = ""

    @property
    def expects_answer(self) -> bool:
        return self.category != "no_answer"


EVAL_CASES: list[EvalCase] = [
    # --- direct ------------------------------------------------------------
    EvalCase(
        case_id="temp-receiving",
        query="What temperature does product have to arrive at?",
        category="direct",
        gold=[GoldRef("food_safety_sop", "Temperature control")],
        reference_answer=(
            "Product must arrive at or below 4°C (40°F). Any delivery arriving "
            "above 4°C is rejected and logged."),
    ),
    EvalCase(
        case_id="danger-zone",
        query="How long can meat sit out before it has to be thrown away?",
        category="direct",
        gold=[GoldRef("food_safety_sop", "Temperature control")],
        reference_answer=(
            "The danger zone is 4°C to 60°C. Cumulative time in that range must "
            "not exceed 2 hours before the product is re-refrigerated, packaged "
            "and chilled, or discarded."),
    ),
    EvalCase(
        case_id="grinder-clean",
        query="How do I clean the grinder after use?",
        category="direct",
        gold=[GoldRef("equipment_cleaning_maintenance_sop", "Grinder")],
        reference_answer=(
            "Unplug it, remove the auger, knife, plate and ring, push out "
            "residual product (do not run the machine empty to clear it), then "
            "wash, rinse, sanitize and air-dry. Inspect the knife and plate as a "
            "matched pair. Strip the head completely once daily."),
    ),
    EvalCase(
        case_id="sanitizer-ppm",
        query="What chlorine concentration should the sanitizer be at?",
        category="direct",
        gold=[GoldRef("equipment_cleaning_maintenance_sop", "Cleaning chemicals")],
        reference_answer=(
            "Chlorine sanitizer for food-contact surfaces is 100-200 ppm with a "
            "60 second contact time, no rinse. Verify with a test strip, not by "
            "eye, and make it fresh each shift."),
    ),
    EvalCase(
        case_id="ribeye-thickness",
        query="How thick should ribeye steaks be cut?",
        category="direct",
        gold=[GoldRef("meat_cutting_instructions", "Steaks")],
        reference_answer=(
            "Boneless ribeye is cut 1 to 1.25 inches (2.5-3.2 cm) thick, to a "
            "1.6 kg target pack weight, with the fat cap trimmed to 1/8-1/4 inch "
            "(3-6 mm) and the natural marbling left alone."),
    ),
    EvalCase(
        case_id="ground-shelf-life",
        query="What is the use-by date on ground beef?",
        category="direct",
        gold=[GoldRef("food_safety_sop", "Labeling")],
        reference_answer=(
            "Ground and trim products carry a 2-day use-by from the cut date - "
            "the shortest in the catalog."),
    ),
    EvalCase(
        case_id="tenderloin-box",
        query="How much of a tenderloin box should I open at once?",
        category="direct",
        gold=[GoldRef("meat_cutting_instructions", "per-piece primals")],
        reference_answer=(
            "Only pull the number of ~4.5 kg pieces needed for today's and "
            "tomorrow's planned volume - typically 2-3 - and reseal and return "
            "the rest of the ~40 kg box to cold storage immediately. Do not "
            "process the whole box while you are in there."),
    ),
    EvalCase(
        case_id="vendor-stockout",
        query="What do I do when the vendor is out of a primal?",
        category="direct",
        gold=[GoldRef("receiving_and_vendor_sop", "Vendor stockout")],
        reference_answer=(
            "Log the stockout with the primal, date and expected recovery "
            "(usually three to four delivery days). Check coverage - boxes on "
            "hand over normal daily draw - and escalate to the department lead "
            "the same morning if it is under two days. A lead may re-route a "
            "shared primal toward the faster-moving product, but never "
            "substitute a different primal into a product's spec."),
    ),
    EvalCase(
        case_id="closing-waste-log",
        query="What has to be recorded on the waste log at close?",
        category="direct",
        gold=[GoldRef("opening_closing_checklists", "Closing checklist")],
        reference_answer=(
            "Every item pulled from the case that is not sellable tomorrow, with "
            "the product, the weight, and the reason. The log is what separates "
            "trim loss from over-production later."),
    ),
    EvalCase(
        case_id="species-changeover",
        query="What do I have to do between cutting beef and cutting pork?",
        category="direct",
        gold=[GoldRef("food_safety_sop", "Cross-contamination")],
        reference_answer=(
            "A full changeover: remove all visible debris, wash with hot water "
            "and food-safe detergent, sanitize with an approved food-contact "
            "sanitizer for its specified contact time, and air dry - do not wipe "
            "dry with a shared cloth. Change gloves too. Dedicated or freshly "
            "sanitized boards and knives per species; never move between species "
            "without the changeover."),
    ),

    # --- paraphrases (semantic cache true-positive pairs) ------------------
    EvalCase(
        case_id="temp-receiving-para",
        query="How cold does a delivery need to be when it shows up?",
        category="paraphrase",
        paraphrase_of="temp-receiving",
        gold=[GoldRef("food_safety_sop", "Temperature control")],
        reference_answer=(
            "Product must arrive at or below 4°C (40°F); anything warmer is "
            "rejected and logged."),
    ),
    EvalCase(
        case_id="grinder-clean-para",
        query="Grinder cleaning procedure",
        category="paraphrase",
        paraphrase_of="grinder-clean",
        gold=[GoldRef("equipment_cleaning_maintenance_sop", "Grinder")],
        reference_answer=(
            "Unplug, strip the auger, knife, plate and ring, push residual "
            "product out, then wash, rinse, sanitize and air-dry; full head "
            "strip daily and between species."),
    ),
    EvalCase(
        case_id="ribeye-thickness-para",
        query="ribeye steak cutting spec thickness",
        category="paraphrase",
        paraphrase_of="ribeye-thickness",
        gold=[GoldRef("meat_cutting_instructions", "Steaks")],
        reference_answer=(
            "1 to 1.25 inches (2.5-3.2 cm), 1.6 kg pack, fat cap 1/8-1/4 inch."),
    ),

    # --- multi-hop ---------------------------------------------------------
    EvalCase(
        case_id="delivery-full-check",
        query="What do I check when a delivery arrives, and what temperature is required?",
        category="multi_hop",
        gold=[GoldRef("receiving_and_vendor_sop", "Inspection at the dock"),
              GoldRef("food_safety_sop", "Temperature control")],
        reference_answer=(
            "At the dock, before the driver leaves: probe temperature (at or "
            "below 4°C, per the food safety SOP), count cases against the "
            "manifest, spot-weigh at least one box per primal, check packaging "
            "seals, and confirm legible pack dates and lot codes."),
        notes="Requires one chunk from each of two documents.",
    ),
    EvalCase(
        case_id="saw-clean-and-schedule",
        query="How often is the band saw deep cleaned and what does the deep clean involve?",
        category="multi_hop",
        gold=[GoldRef("equipment_cleaning_maintenance_sop", "Band saw"),
              GoldRef("equipment_cleaning_maintenance_sop", "Preventive maintenance")],
        reference_answer=(
            "Weekly, at the Sunday close: the daily strip and clean plus removing "
            "and cleaning the blade tensioner and checking the wheel tires for "
            "cracking or embedded bone chips. Wheel tires are also on the weekly "
            "preventive maintenance list."),
    ),
    EvalCase(
        case_id="opening-coverage",
        query="How do I know at opening whether we have enough primal to get through the day?",
        category="multi_hop",
        gold=[GoldRef("opening_closing_checklists", "Opening checklist"),
              GoldRef("receiving_and_vendor_sop", "Vendor stockout")],
        reference_answer=(
            "After receiving and put-away, count boxes on hand per primal against "
            "its normal daily draw. Anything under two days of coverage is flagged "
            "to the lead before cutting starts."),
    ),

    # --- adversarial -------------------------------------------------------
    EvalCase(
        case_id="pulled-pork-fat-cap",
        query="How tight should I trim the fat cap on a pulled pork roast?",
        category="adversarial",
        gold=[GoldRef("meat_cutting_instructions", "Roasts")],
        reference_answer=(
            "Do not trim it tight. Pulled pork (Boston butt) is the one roast "
            "where a 1/4 to 1/2 inch (6-12 mm) fat cap is left on deliberately - "
            "it renders during slow cooking and keeps the pulled product moist. "
            "This is the opposite of the steak and other-roast trim standard."),
        notes="THE adversarial case CLAUDE.md names: a system that averages "
              "across similar 'fat trim' chunks reports the general tight-trim "
              "rule and gets this wrong.",
    ),
    EvalCase(
        case_id="pork-belly-trim",
        query="How much fat do I trim off thin-cut pork bellies?",
        category="adversarial",
        gold=[GoldRef("meat_cutting_instructions", "Thin-sliced")],
        reference_answer=(
            "None. The alternating fat and lean layers are the product - do not "
            "trim the fat layers. Slice at 1/8 inch across the belly for an even "
            "fat-to-lean ratio per slice."),
        notes="Second 'do not trim' exception; distractor chunks all say trim.",
    ),
    EvalCase(
        case_id="slicer-not-saw",
        query="Which direction do I clean the slicer blade?",
        category="adversarial",
        gold=[GoldRef("equipment_cleaning_maintenance_sop", "Slicer")],
        reference_answer=(
            "From the centre outwards, with a cloth - never a scraper, and never "
            "toward the edge."),
        notes="Band saw and grinder cleaning chunks share almost all vocabulary.",
    ),
    EvalCase(
        case_id="osso-buco-not-short-ribs",
        query="How much do I trim an osso buco shank?",
        category="adversarial",
        gold=[GoldRef("meat_cutting_instructions", "Ribs")],
        reference_answer=(
            "Minimal exterior silverskin only. Leave the connective tissue and "
            "the marrow bone intact - both render during a slow braise. It is a "
            "band saw cross-cut, not a knife cut."),
        notes="Sits in the same table as short ribs, which has a different spec.",
    ),

    # --- broad / open-ended ------------------------------------------------
    # Deliberately phrased the way an operator actually asks, including words
    # the corpus never uses ("protocols", "rules", "need to know").
    EvalCase(
        case_id="broad-safety-protocols",
        query="What are some safety protocols that I should follow?",
        category="broad",
        gold=[GoldRef("food_safety_sop", "Cross-contamination"),
              GoldRef("food_safety_sop", "Personal hygiene"),
              GoldRef("food_safety_sop", "Temperature control")],
        reference_answer=(
            "The core food-safety practices are temperature control (receive at "
            "or below 4°C, hold coolers at 0-4°C, and keep cumulative time in "
            "the 4-60°C danger zone under 2 hours), cross-contamination control "
            "(dedicated or freshly sanitized boards and knives per species, a "
            "full wash-rinse-sanitize-air-dry changeover between species, and "
            "glove changes), and personal hygiene (hand washing on arrival and "
            "after every task change, clean aprons, hair restraints, and "
            "reporting illness before a shift)."),
        notes="THE case a real user hit. 'protocols' appears nowhere in the "
              "corpus; the corpus says 'procedures'. Must not be refused.",
    ),
    EvalCase(
        case_id="broad-hygiene-rules",
        query="What are the hygiene rules for staff?",
        category="broad",
        gold=[GoldRef("food_safety_sop", "Personal hygiene")],
        reference_answer=(
            "Wash hands on arrival, before starting, after breaks, after the "
            "restroom, and after any change of task. Wear a clean apron and, "
            "where applicable, hair and beard restraints. Anyone with fever, "
            "vomiting, diarrhea, jaundice, or an infected cut on exposed skin "
            "must not handle open product and must tell a supervisor before "
            "the shift."),
        notes="'rules' is another word the corpus never uses.",
    ),
    EvalCase(
        case_id="broad-new-cutter",
        query="What does a new cutter need to know before starting on the floor?",
        category="broad",
        gold=[GoldRef("meat_cutting_instructions", "General procedure"),
              GoldRef("food_safety_sop", "Personal hygiene")],
        reference_answer=(
            "Stage and inspect the primal against its delivery label and check "
            "it is within temperature before cutting. Trim first, portion "
            "second - but trim standards differ by product, so follow the "
            "product's own spec rather than one blanket rule. Weigh every pack; "
            "target weight has a ±10% tolerance and out-of-tolerance packs get "
            "re-portioned, not mislabeled. Wash hands, wear a clean apron, and "
            "use cut gloves."),
        notes="Open-ended onboarding question spanning two documents.",
    ),
    EvalCase(
        case_id="broad-equipment-daily",
        query="What cleaning do I need to do on the machines every day?",
        category="broad",
        gold=[GoldRef("equipment_cleaning_maintenance_sop", "Scope and responsibility"),
              GoldRef("equipment_cleaning_maintenance_sop", "Band saw"),
              GoldRef("equipment_cleaning_maintenance_sop", "Grinder")],
        reference_answer=(
            "Every powered machine is unplugged first. The band saw is stripped "
            "and cleaned at close; the grinder is stripped after each use and "
            "the head fully stripped daily; the slicer is cleaned after each "
            "use with the gauge plate at zero; the vacuum sealer chamber and "
            "seal bar are wiped at close. Cutting tables are cleaned after each "
            "species change and at close."),
    ),

    # --- no answer ---------------------------------------------------------
    EvalCase(
        case_id="no-answer-sous-vide",
        query="What temperature should I sous vide a striploin steak to?",
        category="no_answer",
        notes="Cooking question. The corpus is about cutting and storing, not "
              "cooking - and 'temperature' plus 'striploin' both retrieve well, "
              "so the retriever WILL return confident-looking chunks.",
    ),
    EvalCase(
        case_id="no-answer-wages",
        query="What is the overtime rate for a cutter working a stat holiday?",
        category="no_answer",
        notes="HR/payroll. 'cutter' and 'stat holiday' both appear in the corpus.",
    ),
    EvalCase(
        case_id="no-answer-chicken",
        query="What is the cutting spec for chicken thighs?",
        category="no_answer",
        notes="Chicken is explicitly out of scope; the corpus mentions chicken "
              "only in the cross-contamination rule, which is a tempting "
              "near-miss.",
    ),
    EvalCase(
        case_id="no-answer-supplier-price",
        query="What do we pay per kilo for Blade Eyes?",
        category="no_answer",
        notes="Pricing is not in the SOPs. The receiving SOP's box-weight table "
              "is a strong lexical match and contains numbers, which is exactly "
              "the setup for a fabricated figure.",
    ),
    EvalCase(
        case_id="no-answer-freezer",
        query="How long can we keep primals in the freezer before they go off?",
        category="no_answer",
        notes="Corpus covers refrigeration and use-by from cut date, never "
              "freezing. Shelf-life table is a strong near-miss.",
    ),
]


# Cases the OFFLINE (lexical) path is known to get wrong, named individually
# rather than hidden behind an averaged threshold. Naming them keeps the test
# strict: a DIFFERENT case leaking still fails, which an aggregate bar of
# ">= 0.8" would quietly absorb.
#
# no-answer-chicken: "what is the cutting spec for chicken thighs?" - chicken
#   IS in the corpus (the cross-contamination rule names it as a species to
#   separate), so only "thigh" is unknown vocabulary. Knowing that chicken
#   HANDLING is covered while chicken CUTTING is out of scope is a semantic
#   judgement; lexical signals cannot make it, and the measured score
#   distributions confirm no threshold separates it. The live path (Claude
#   reading the passages) is expected to decline this correctly - the live
#   bar stays at 1.0.
KNOWN_OFFLINE_DECLINE_GAPS = {"no-answer-chicken"}


def resolve_gold(case: EvalCase, chunks: list[Chunk]) -> list[str]:
    """Map a case's gold references onto live chunk ids.

    Raises rather than returning an empty list for an unmatched label: a
    silently unresolvable gold reference turns into a reported recall failure
    that has nothing to do with retrieval quality.
    """
    resolved: list[str] = []
    for ref in case.gold:
        matches = [c.chunk_id for c in chunks
                   if c.doc_id == ref.doc_id
                   and ref.heading_contains.lower() in c.heading.lower()]
        if not matches:
            raise ValueError(
                f"Eval case '{case.case_id}' references "
                f"{ref.doc_id} > '{ref.heading_contains}', which matches no chunk. "
                f"The SOP text or the chunker changed - fix the label, don't "
                f"delete the case.")
        resolved.extend(matches)
    return sorted(set(resolved))


def cases_by_category(category: str) -> list[EvalCase]:
    return [c for c in EVAL_CASES if c.category == category]


def paraphrase_pairs() -> list[tuple[EvalCase, EvalCase]]:
    """(original, paraphrase) pairs - the cache tuner's true positives."""
    by_id = {c.case_id: c for c in EVAL_CASES}
    pairs = []
    for case in EVAL_CASES:
        if case.paraphrase_of and case.paraphrase_of in by_id:
            pairs.append((by_id[case.paraphrase_of], case))
    return pairs


def summary() -> dict:
    from collections import Counter
    counts = Counter(c.category for c in EVAL_CASES)
    return {"total": len(EVAL_CASES), **counts}
