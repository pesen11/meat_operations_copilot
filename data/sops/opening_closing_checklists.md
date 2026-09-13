---
document_type: shift_checklist
title: Opening & Closing Checklists — Meat Cutting Department
data_source: synthetic
status: EXAMPLE / TEMPLATE — for portfolio project use, not a real operational document
version: 1.0
effective_date: 2025-01-01
related: food_safety_sop.md, equipment_cleaning_maintenance_sop.md, receiving_and_vendor_sop.md
---

# Opening & Closing Checklists — Meat Cutting Department

> **Note on this document:** Synthetic example SOP for the Operations Copilot
> project. Shift times below match the staffing pattern this project
> simulates (see `simulation/history_generator.py`) and are illustrative, not
> a real employer's schedule.

## 1. Purpose

The two checklists that bracket every trading day. These are sequences, not
menus — the order matters, because each step's output is the next step's
input. The morning coverage count, for example, is only meaningful after the
delivery has been received and put away.

## 2. Shift pattern

| Shift | Start | Roles | Length |
|---|---|---|---|
| Morning | 05:00 | 2 cutters, 1 wrapper, 1 fishmonger, 1 counter | 8.5 h |
| Mid | 09:00–10:30 | 1 counter/wrapper (2 on weekends) | 8.5 h |
| Closing | 13:30 | 2 closers (3 on weekends) | 8.5 h |

The shop trades seven days a week and closes on Ontario statutory holidays.
Weekends add one mid-shift person and one closer, because weekend volume
runs well above a weekday.

## 3. Opening checklist — morning shift (05:00)

**Before any product is touched:**

1. **Cooler and case temperatures.** Read and log every cooler and display
   case. Coolers must be 0–4°C. A cooler out of range overnight is a
   product-disposition decision for the department lead — do not begin
   cutting out of a cooler that failed overnight.
2. **Hand wash and PPE.** Clean apron, cut gloves available at each station,
   no jewellery.
3. **Equipment visual check.** Cords, plugs, guards intact on band saw,
   grinder, slicer, sealer. Anything tagged out of service stays out.
4. **Sanitizer made fresh.** Mix the shift's sanitizer solution and verify
   concentration with a test strip; log the reading. Yesterday's bucket is
   discarded, not topped up.

**Receiving (05:00–07:30):**

5. Receive the day's delivery against the manifest per the Receiving SOP:
   probe temperature, count boxes, spot-weigh, check seals and lot codes.
6. Put away FIFO — new stock behind existing stock of the same primal.
7. Update the box count per primal.

**Planning the cut:**

8. **Count coverage.** For each primal, boxes on hand against normal daily
   draw. Anything under two days of coverage is flagged to the lead before
   cutting starts, not after.
9. **Read the production plan** for today and tomorrow. This department cuts
   on a today-plus-next-day basis; the plan covers both, so today's cut
   includes tomorrow's buffer.
10. **Check yesterday's leftovers first.** Product already cut and still
    within its sell-by is cut into today's plan before new primal is opened.
    Opening a new box while yesterday's packs are still in the back is how
    waste is created.
11. **Confirm the cut plan for shared primals.** Blade Eyes feeds both chuck
    roast and thinly sliced blades; the day's split between them is the
    lead's call, informed by what the case actually needs.

**Case setup (by 08:00):**

12. Rotate the display case: older product forward, new product behind.
13. Pull anything at or past its sell-by date; record it on the waste log.
14. Fill gaps in the case from the cut plan, front-facing labels out.
15. Confirm every pack in the case carries a legible label with product
    name, weight, price, pack date, and sell-by.

## 4. Closing checklist — closing shift (from 13:30, completed by close)

**Product:**

1. **Pull and assess the case.** Anything not sellable tomorrow comes out.
   Record each item on the waste log with product, weight, and reason — the
   waste log is the only way trim loss can later be distinguished from
   over-production, and the two have completely different fixes.
2. **Re-wrap and re-date** anything being held that is still within spec.
   Never re-date a pack to a later date than its original sell-by.
3. **Return unsold cut product** to the cooler on labelled trays, not in the
   case overnight.
4. **Note tomorrow's shortfalls** for the morning shift: anything that ran
   out today, and anything the case could not be filled with.

**Equipment (see Equipment SOP for the full method):**

5. Band saw: strip, clean, sanitize, air-dry, reassemble.
6. Grinder: full head strip and clean.
7. Slicer: gauge plate to zero, strip, clean, reassemble.
8. Vacuum sealer: chamber and seal bar wiped, PTFE tape checked.
9. Knives washed, sanitized, and racked; no knife left in a sink.

**Area:**

10. Cutting tables scraped, washed, sanitized. Table inserts removed and
    washed separately.
11. Floors swept and washed; drains cleared and flushed.
12. Tote bins and lugs through wash-up; none left stacked wet.
13. Waste and bone barrels out; blade disposal box checked and not
    overfilled.

**Records and security:**

14. **Log closing cooler and case temperatures.** A cooler drifting up at
    close is a problem to solve tonight, not to discover at 05:00.
15. Log sanitizer concentration at close.
16. Complete the waste log and total it for the day.
17. Note any equipment fault in the maintenance log and tag the machine out
    of service.
18. Cooler doors confirmed closed and latched; lights and power off at the
    board; department secured.

## 5. Weekend and holiday variations

- **Saturday close** carries Sunday as well, because there is no Sunday
  delivery. Confirm Sunday's cut can be covered from stock on hand before
  leaving on Saturday night.
- **Sunday** runs on Saturday's stock. A primal that is short on Sunday
  cannot be recovered until Monday morning's delivery.
- **The day before a statutory holiday** is a closed day followed by a
  reopening: cut and hold only what will still be sellable when the shop
  reopens, and expect elevated volume in the run-up to long weekends and
  Christmas.

## 6. Handover

At each shift change the outgoing lead tells the incoming lead, out loud and
in this order:

1. Anything out of temperature, and what was done about it.
2. Any equipment tagged out of service.
3. Any primal short, and the coverage position.
4. Anything cut but not yet in the case.
5. Anything the next shift has been asked to do that is not on this list.

A handover written on a note and left on the bench is not a handover. If the
incoming lead did not hear it, it did not happen.
