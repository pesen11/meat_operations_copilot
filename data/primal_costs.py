"""
Estimated vendor cost per kg of RAW PRIMAL, as delivered.

WHY THIS FILE EXISTS
--------------------
Earlier versions of this project had no primal cost data at all, and
estimated it as a percentage discount off the *finished product's retail
price*. That was wrong in a way that only showed up once the numbers were
totalled: the price_per_kg values in seed_yield_data.py are RETAIL SHELF
PRICES — what a customer pays over the counter — not what the shop pays a
vendor for a whole primal in bulk. Discounting retail by 20-40% and then
paying that rate for the whole primal (trim included) put cost of goods at
80-97% of revenue and left the whole store making roughly $900/week per
item, which is not an operation that can pay rent. See CLAUDE.md.

The real vendor price list is confidential and not available to this
project. So these numbers are CALIBRATED ESTIMATES, not invoice data:
each primal's cost is back-solved from the retail value its own cut plan
realises, at a target gross margin (see TARGET_GROSS_MARGIN_BY_PRIMAL
below), then rounded to the nearest $0.05 so the table reads like a price
list rather than a formula. Verified against the catalog: the blended
gross margin across all 19 products lands at 39.0% after cutting labor
(40.1% on primal cost alone) — the 38-40% range the project owner
confirmed is realistic for this shop.

ASSUMPTION — FLAGGED, same as primal_discount_pct was before it, and the
first thing to replace with real data. Dropping true invoice costs into
PRIMAL_COST_PER_KG is the *only* change needed; every calculator, tool,
agent and eval downstream reads through it and keeps working.

WHY COST IS KEYED BY PRIMAL, NOT BY PRODUCT
-------------------------------------------
You buy a primal, not a cut. Blade Eyes feeds both Chuck Roast and Thinly
Sliced Blades; Round feeds both Eye of Round Roast and Stew Meat. Those
paths must share one cost per kg or the economics are incoherent — the
old per-product discount let the same primal cost two different amounts
depending on which cut you asked about. Keying the table by primal makes
that impossible by construction, and data/catalog.py::check_primal_costs
asserts it.

A consequence worth expecting: per-PRODUCT margins on a multi-path primal
are not equal to the primal's target. Round is bought at one price and
yields a 48.7%-margin roast alongside 31.9%-margin stew meat. The primal
clears its 43.6% target overall; the split between paths is a cut-plan
outcome, not a pricing error.
"""

from __future__ import annotations

# Target gross margin per primal, used to back-solve the costs below.
# Tiered rather than flat, because a butcher shop does not earn the same
# percentage on every cut: premium steaks carry a high ticket price at a
# thinner percentage, while grinds and value cuts carry the margin.
# ASSUMPTION — the tiering pattern is industry-typical, the exact values
# are chosen to blend to the owner-confirmed 38-40%.
_PREMIUM = 0.336  # ribeye, striploin, tenderloin, wagyu, tomahawk, skirt
_ROUTINE = 0.416  # mainstream steaks and roasts
_VALUE = 0.436    # grinds, pork, braising and stewing cuts

TARGET_GROSS_MARGIN_BY_PRIMAL: dict[str, float] = {
    # --- real primals (seed_yield_data.py) ---
    "Blade Eyes": _ROUTINE,
    "Striploin Whole": _PREMIUM,
    "Bone-in Ribeye": _PREMIUM,
    "Chuck Flat": _ROUTINE,
    "Top Sirloin Butt": _ROUTINE,
    "Flank": _ROUTINE,
    "Pork Belly": _VALUE,
    "Wagyu Flat Iron": _PREMIUM,
    "6X6 Frenched": _PREMIUM,
    "Tenderloin": _PREMIUM,
    # --- synthetic primals (synthetic_yield_data.py) ---
    "Brisket": _VALUE,
    "Round": _VALUE,
    "Plate": _PREMIUM,
    "Shank": _VALUE,
    "Chuck Primal": _VALUE,
    "Pork Loin": _VALUE,
    "Pork Shoulder": _VALUE,
}

# Estimated vendor cost, $ per kg of raw primal as delivered (you pay for
# the trim and bone too — the yield_pct is applied downstream, in
# cost_calc.py, not here).
#
# Regenerate after changing prices, yields or cut plans:
#     python -m data.primal_costs --recalibrate
PRIMAL_COST_PER_KG: dict[str, float] = {
    # --- real primals ---
    "Blade Eyes": 12.20,
    "Striploin Whole": 22.25,
    "Bone-in Ribeye": 22.25,
    "Chuck Flat": 21.55,
    "Top Sirloin Butt": 16.65,
    "Flank": 16.80,
    "Pork Belly": 6.60,
    "Wagyu Flat Iron": 36.35,
    "6X6 Frenched": 32.50,
    "Tenderloin": 33.75,
    # --- synthetic primals ---
    "Brisket": 9.10,
    "Round": 9.05,
    "Plate": 18.10,
    "Shank": 8.60,
    "Chuck Primal": 7.25,
    "Pork Loin": 7.20,
    "Pork Shoulder": 5.60,
}


def cost_per_kg(source_primal: str) -> float | None:
    """Estimated vendor $/kg for a primal, or None if it isn't in the table."""
    return PRIMAL_COST_PER_KG.get(source_primal)


def recalibrate() -> dict[str, float]:
    """
    Re-derive PRIMAL_COST_PER_KG from the current catalog.

    For each primal: the retail value its cut plan realises per kg of raw
    primal is sum(cut_plan_share * yield_pct * retail_price) across its
    paths; cost is that value less the primal's target gross margin.

    Deliberately NOT called at import time — the table above is committed
    data, so that editing a retail price does not silently move every cost
    in the model. Run it, eyeball the diff, paste it in.
    """
    from data import catalog  # local import: avoids a circular import at module load

    costs: dict[str, float] = {}
    for primal, paths in catalog.yield_profiles_by_primal().items():
        retail_value_per_primal_kg = sum(
            p.cut_plan_share * p.yield_pct * catalog.products_by_sku[p.product_sku].price_per_kg
            for p in paths
        )
        target = TARGET_GROSS_MARGIN_BY_PRIMAL[primal]
        # round to the nearest $0.05 — a price list, not a formula
        costs[primal] = round(retail_value_per_primal_kg * (1 - target) * 20) / 20
    return costs


if __name__ == "__main__":
    import sys

    if "--recalibrate" in sys.argv:
        fresh = recalibrate()
        print("PRIMAL_COST_PER_KG: dict[str, float] = {")
        for primal, cost in fresh.items():
            drift = cost - PRIMAL_COST_PER_KG.get(primal, cost)
            flag = f"   # <- was {PRIMAL_COST_PER_KG[primal]:.2f}" if abs(drift) > 1e-9 else ""
            print(f'    "{primal}": {cost:.2f},{flag}')
        print("}")
    else:
        print(f"{len(PRIMAL_COST_PER_KG)} primals costed. Use --recalibrate to re-derive.")
