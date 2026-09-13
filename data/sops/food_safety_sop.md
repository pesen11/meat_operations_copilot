---
document_type: food_safety_sop
title: Food Safety Standard Operating Procedures — Meat Cutting Department
data_source: synthetic
status: EXAMPLE / TEMPLATE — for portfolio project use, not a real operational document
version: 1.0
effective_date: 2025-01-01
---

# Food Safety Standard Operating Procedures — Meat Cutting Department

> **Note on this document:** This is a synthetic example SOP built for the
> Operations Copilot project. It illustrates the *kind* of content a real
> food-safety SOP contains, modeled on general, widely-known food safety
> principles (temperature danger zone, cross-contamination control,
> traceability). It is **not** a substitute for a real food safety plan —
> a real operation's procedures must be reviewed and signed off by a
> qualified food safety professional against current CFIA and Ontario
> Food Premises Regulation requirements. Do not treat any temperature,
> time, or procedure below as verified regulatory guidance.

## 1. Purpose

This SOP defines the minimum food safety practices for receiving,
storing, cutting, and packaging meat products, to prevent
cross-contamination, control bacterial growth, and maintain product
traceability.

## 2. Scope

Applies to all staff handling raw meat, poultry, or fish in the cutting
room, cooler, and display case areas.

## 3. Temperature control

| Stage | Requirement |
|---|---|
| Receiving | Product must arrive at or below 4°C (40°F). Reject and log any delivery arriving above 4°C. |
| Cooler storage | Maintain 0–4°C (32–40°F). Check and log cooler temperature at shift start and mid-shift. |
| Cutting room (ambient) | Keep primals refrigerated until immediately before cutting; do not stage more product at room temperature than can be processed within 30 minutes. |
| Danger zone | 4°C–60°C (40°F–140°F) is the bacterial growth danger zone. Cumulative time any product spends in this range must not exceed 2 hours before it's re-refrigerated, packaged and chilled, or discarded. |
| Display case | Maintain at or below 4°C (40°F); check and log temperature at case open, midday, and close. |

## 4. Cross-contamination control

1. **Species separation.** Beef, pork, and any other species (chicken,
   fish) must use dedicated or freshly sanitized cutting boards, knives,
   and equipment — never move between species without a full changeover
   (Section 4.2).
2. **Color-coded equipment.** Where available, use color-coded cutting
   boards/handles by species (e.g., one color for beef, another for
   pork, another for fish) to make cross-contamination visually obvious
   and preventable.
3. **Changeover procedure between species or between raw and
   ready-to-eat product:**
   - Remove all visible product debris from surfaces and equipment.
   - Wash with hot water and food-safe detergent.
   - Sanitize with an approved food-contact sanitizer at the
     manufacturer's specified contact time.
   - Air dry before resuming work — do not wipe dry with a shared cloth.
4. **Hands and gloves.** Change gloves between species, after handling
   packaging/labels, after touching your face or phone, and any time
   gloves are visibly soiled or torn. Wash hands before donning new
   gloves, not just between glove changes.

## 5. Personal hygiene & PPE

- Wash hands on arrival, before starting work, after breaks, after
  using the restroom, and after any change of task.
- Wear clean aprons and, where applicable, hair/beard restraints.
- Any staff member with symptoms of illness (fever, vomiting, diarrhea,
  jaundice, or an infected cut/wound on exposed skin) must not handle
  open product and should be reported to a supervisor before starting a
  shift.

## 6. Labeling & traceability

Every packaged product must be labeled with:
- Product SKU / name
- Cut date
- Use-by date (per the shelf-life table below)
- Lot/batch reference sufficient to trace back to the source primal
  delivery, for recall purposes

| Product category | Typical use-by from cut date |
|---|---|
| Ground/trim products | 2 days |
| Portioned steaks/roasts (beef) | 4–5 days |
| Portioned pork | 3–4 days |
| Thin-sliced products (e.g., thin-cut pork belly, thin-sliced flat iron) | 2–3 days |

*(These are illustrative example values only — a real operation must set
these based on its own validated shelf-life testing, not this document.)*

## 7. Vendor delivery discrepancies

If a delivery doesn't match what was ordered (wrong item, short count, or
temperature out of spec on arrival):
1. Do not accept the affected item(s) into inventory.
2. Photograph and log the discrepancy (item, expected vs. received,
   temperature reading if applicable).
3. Notify the vendor and a supervisor before the driver leaves, where
   possible.
4. Log the incident against that primal for future vendor
   reliability tracking (see `PrimalStockLevel` / stockout handling in
   the main project — vendor delivery gaps are already modeled in the
   synthetic history data).

## 8. Recall procedure (outline)

1. Identify the affected SKU(s) and lot/batch range using the
   traceability labels from Section 6.
2. Pull all matching product from the display case and cooler
   immediately.
3. Check the sales log for the affected date range to identify how much
   product may already have been sold.
4. Notify a supervisor and follow escalation per current regulatory
   requirements (this step must be filled in with real, current CFIA/
   provincial guidance for an actual operation — not covered by this
   example document).

## 9. Cleaning schedule (minimum)

| Area/equipment | Frequency |
|---|---|
| Cutting boards, knives | Between species changeovers, and at minimum every 4 hours during continuous same-species use |
| Cutting room floor | End of every shift |
| Cooler shelving | Weekly, or immediately after any spill |
| Display case interior | Daily |
| Scales | Daily, and immediately after any product contact spill |

## 10. Sign-off

Each shift's opening cutter should confirm and initial:
- [ ] Cooler and case temperatures logged and within range
- [ ] Cutting boards/knives sanitized and changeover-ready
- [ ] No expired product remaining in the case from the prior day
- [ ] PPE available and in use
