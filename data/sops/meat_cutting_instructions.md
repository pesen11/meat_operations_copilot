---
document_type: cutting_instruction_sheet
title: Meat Cutting & Portioning Specifications — Full Catalog
data_source: synthetic
status: EXAMPLE / TEMPLATE — for portfolio project use, not a real operational document
version: 1.0
effective_date: 2025-01-01
applies_to: all 19 products in the catalog (11 real, 8 synthetic — see data/catalog.py)
supersedes: steak_cutting_instructions.md (steaks-only version)
---

# Meat Cutting & Portioning Specifications — Full Catalog

> **Note on this document:** This is a synthetic example SOP built for the
> Operations Copilot project, modeled on general industry cutting
> conventions. It is not derived from a specific employer's real
> procedures and should not be treated as a regulatory or food-safety
> reference. Product names, target weights, and yields below are
> cross-referenced to this project's own catalog (`data/catalog.py`, both
> `data/seed_yield_data.py` [real] and `data/synthetic_yield_data.py`
> [synthetic]) so retrieval-augmented answers can be checked against
> known values.

## 1. Purpose

This sheet defines the cutting spec — thickness/size, target pack weight,
and fat-trim requirement — for every product currently in the catalog,
covering steaks, roasts, ribs/braising cuts, thin-sliced products, ground
product, stew meat, chops, and specialty cross-cuts. Different cut types
need different quality parameters (a roast is judged on shape/tie and fat
cap, not thickness; ground product is judged on grind size and fat ratio,
not either); this document is organized by cut type for that reason.

## 2. General procedure (applies to every cut type below)

1. **Stage and inspect** the primal. Confirm it matches the expected cut
   (check against the delivery label) and is within acceptable temperature
   (see Food Safety SOP, Section 3) before cutting begins.
2. **Trim first, portion second**, with one exception: braising cuts and
   pulled-pork-style roasts often keep more fat than a steak would (see
   their sections below) — "trim first" does not mean "trim to the same
   standard as a steak" for every product.
3. **Weigh every pack.** Target weight has a ±10% tolerance unless a
   product's section says otherwise. Packs outside tolerance get
   re-portioned, not force-labeled at the wrong weight.
4. **Tray and wrap immediately** after cutting. Label with product SKU,
   cut date, and use-by date per Food Safety SOP Section 6.
5. **Sanitize the block and knife** between different primals, and always
   between species (beef/pork/wagyu) — see Food Safety SOP Section 4.
6. **Match the tool to the cut.** Bone-in cross-cuts (osso buco) and
   bone-in steaks (tomahawk) require a band saw, not a knife, for clean,
   safe portioning through bone.

## 3. Steaks

Cut against the grain (except flank and skirt — see notes), thickness
checked with a ruler/gauge on the first steak of every primal and every
5th steak after that. Do not eyeball thickness on high-price cuts.

| Product (SKU) | Source primal | Thickness | Target pack weight | Fat trim spec | Notes |
|---|---|---|---|---|---|
| Striploin Steak Boneless (`striploin-steak-bnls`) | Striploin Whole | 1 – 1.25 in (2.5–3.2 cm) | 1.6 kg | External fat cap to max 1/4 in (6 mm); no silverskin | Check for consistent marbling across the loin |
| Boneless Ribeye Steak (`ribeye-steak-bnls`) | Bone-in Ribeye | 1 – 1.25 in (2.5–3.2 cm) | 1.6 kg | Fat cap to 1/8–1/4 in (3–6 mm); leave natural marbling | Bone removed before portioning; reserve bones separately |
| Top Sirloin Steak Cap Removed (`top-sirloin-steak-cr`) | Top Sirloin Butt | 0.75 – 1 in (2–2.5 cm) | 1.4 kg | Fat cap AND silverskin fully removed; external fat to 1/8 in (3 mm) max | Coulotte/cap muscle separated before portioning |
| Flank Steak (`flank-steak`) | Flank | Sold whole, not sliced into multiple steaks | 0.8 kg | Trim silverskin and surface fat; naturally lean | Cut *with* the grain when portioning the whole piece |
| Tenderloin Steak (`tenderloin-steak`) | Tenderloin (PER_PIECE — see Section 8) | 1.5 – 2 in (3.8–5 cm) | 0.8 kg | Silverskin MUST be fully removed; trim chain muscle off separately | Highest-value cut in the catalog — never rush the silverskin trim |
| Skirt Steak (`skirt-steak`) | Plate | Sold whole, not sliced into multiple steaks | 0.6 kg | Remove the tough outer membrane completely; minimal fat naturally | Cut *with* the grain; membrane, not fat, is the main concern |
| Tomahawk Steak (`tomahawk-steak`) | 6X6 Frenched | 2 in (5 cm), bone-in | ~1.75 kg (sold each, one per tray) | Fat cap to 1/4 in (6 mm); bone frenched 6 in from the eye | Band saw required; presentation matters as much as trim |
| Boneless Pork Chops (`pork-chops-bnls`) | Pork Loin | 0.75 – 1 in (2–2.5 cm) | 0.5 kg (typically 2 chops/pack) | Fat cap to 1/8 in (3 mm) | Same thickness discipline as beef steaks; check every pack, chops vary more in shape than beef loin cuts |

## 4. Roasts

Judged on target weight, fat cap (which varies intentionally by product —
do not apply the same trim standard to every roast), and tie/shape
consistency for anything netted or tied.

| Product (SKU) | Source primal | Target pack weight | Fat trim spec | Tie/shape | Notes |
|---|---|---|---|---|---|
| Chuck Roast (`chuck-roast`) | Blade Eyes | 2.2 kg | Fat cap trimmed to approx. 1/4 in (6 mm); remove large connective-tissue seams, keep internal marbling | Tie with butcher's twine every 1–1.5 in for even shape/cooking | One of two yield paths off Blade Eyes (35% of primal volume — see `cut_plan_share` in `data/seed_yield_data.py`) |
| Beef Brisket Whole Trimmed (`brisket-whole-trimmed`) | Brisket | 4.5 kg | Trim external fat cap down to approx. 1/4 in (6 mm) even layer across the whole piece — do not trim to bare meat, the fat cap is needed for low-and-slow cooking | No tie; sold as a whole trimmed packer cut | Largest single-piece product in the catalog |
| Eye of Round Roast (`eye-of-round-roast`) | Round | 1.8 kg | Naturally very lean — trim silverskin fully, minimal fat to remove | Tie only if the piece needs help holding a uniform shape | One of two yield paths off Round (55% of primal volume) |
| Pulled Pork Roast — Boston Butt (`pulled-pork-roast`) | Pork Shoulder | 2.0 kg | **Do NOT over-trim.** Leave a 1/4–1/2 in (6–12 mm) fat cap intentionally — it renders during slow cooking and keeps the pulled product moist | Net or tie only if boneless and needs shape support | This is the one roast in the catalog where standard "trim tight" instinct is wrong — flag for new cutters |

## 5. Ribs / braising cuts

| Product (SKU) | Source primal | Portion spec | Target pack weight | Fat trim spec | Notes |
|---|---|---|---|---|---|
| Simmering Short Ribs (`short-ribs-simmering`) | Chuck Flat | English-cut, bone-in, approx. 1.5–2 in (3.8–5 cm) wide per portion | 1.2 kg (approx. 2–3 ribs per pack) | Trim exterior fat cap to approx. 1/4 in (6 mm) — do not strip fully, fat renders and adds flavor during braising | Band saw for the bone; knife for final portion separation |
| Osso Buco Cross-Cut Shank (`osso-buco`) | Shank | Cross-cut through the bone, 1.5–2 in (3.8–5 cm) thick | 0.7 kg (1–2 pieces per pack) | Trim minimal exterior silverskin only — leave connective tissue and marrow bone intact, both render during slow braise | Band saw required for the cross-cut; this is not a knife cut |

## 6. Thin-sliced products

| Product (SKU) | Source primal | Slice thickness | Target pack weight | Fat trim spec | Notes |
|---|---|---|---|---|---|
| Thinly Sliced Blades (`blade-thin-sliced`) | Blade Eyes | 1/8 – 3/16 in (3–5 mm) | 2.0 kg | Trim major connective-tissue seams before slicing; keep marbling | Second of two yield paths off Blade Eyes (65% of primal volume — the larger share) |
| Thinly Cut Pork Bellies (`pork-belly-thin-cut`) | Pork Belly | 1/8 in (3 mm) | 2.5 kg | **Do not trim the fat layers** — the alternating fat/lean layers are the product | Slice across the belly for even fat/lean ratio per slice |
| Wagyu Flat Iron Thin Sliced (`wagyu-flatiron-thin`) | Wagyu Flat Iron | 1/8 in (3 mm) | 0.8 kg | Trim the connective-tissue seam through the center of the flat iron completely before slicing; minimal fat trim (well-marbled) | Slice across the grain in thin sheets |

## 7. Ground & cubed products

| Product (SKU) | Source primal | Size spec | Target pack weight | Fat/trim spec | Notes |
|---|---|---|---|---|---|
| Ground Beef Medium (`ground-beef-medium`) | Chuck Primal | Medium grinding plate, approx. 4.5 mm (3/16 in) holes; double-grind for uniform texture | 0.5 kg | Target approx. 80/20 lean-to-fat ratio | Grind fresh same-day where possible; ground product has the shortest shelf life in the catalog (see Food Safety SOP Section 6) |
| Beef Stew Meat (`beef-stew-meat`) | Round | Uniform 1 – 1.5 in (2.5–3.8 cm) cubes | 1.0 kg | Trim connective tissue and silverskin from round trim before cubing | Second of two yield paths off Round (45% of primal volume); cube size uniformity matters more than exact weight for cook consistency |

## 8. Special handling: per-piece primals (Tenderloin)

Tenderloin does not arrive pre-portioned like the other primals in this
document — it comes in a full box (~40 kg) but only 2-3 individual pieces
(~4.5 kg each) get pulled and cut at a time, because a full box would not
sell through before the product's quality degrades. When staging a
tenderloin cut:

1. Confirm today's + tomorrow's planned tenderloin volume with the
   Production Schedule before opening the box.
2. Pull only the number of pieces needed for that volume — do not fully
   open and process the box "while you're in there."
3. Reseal and return the unused portion of the box to cold storage
   immediately.

## 9. Quality control sign-off

Every batch cut should have the following checked and initialed before
going to the case:
- [ ] Size spec (thickness / cube size / slice thickness / grind size, as
      applicable to the product) matches Sections 3–7
- [ ] Pack weight within ±10% of target
- [ ] Trim matches spec for that specific product — note that trim
      standards deliberately differ by product (compare Section 3 vs.
      Section 4's pulled pork entry) rather than applying one blanket rule
- [ ] Label shows correct SKU, cut date, and use-by date
- [ ] Case/cooler temperature confirmed at or below spec before stocking
