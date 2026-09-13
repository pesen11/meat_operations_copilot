"""
Seed data transcribed directly from the user's real workplace-derived
table (Primal.docx). These are real-ish numbers ("similar but not exact"
to their actual workplace) — the anchor data the rest of the synthetic
dataset gets built around.

Each row becomes one Product + one YieldProfile. DemandProfile rows
capture the weekday/weekend average sales, for shaping synthetic
HistoricalSale generation later.
"""

from __future__ import annotations

from data.primal_costs import cost_per_kg
from models.domain import Product, YieldProfile, DemandProfile, GradeLabel, Unit, LaborBasis, DataSource

# (sku, name) kept short + slug-like; feel free to rename later.
# cut_plan_share: Blade Eyes feeds two products. Confirmed with user:
# 65% of Blade Eyes volume goes to Thinly Sliced Blades, 35% to Chuck Roast.

_ROWS = [
    # source_primal, product name, sku, price_per_kg, labor_min_per_box, yield_pct,
    # avg_box_weight_kg, avg_pack_weight_kg, avg_sales_weekday_kg, avg_sales_weekend_kg, grade, cut_plan_share
    ("Blade Eyes", "Chuck Roast", "chuck-roast", 23.49, 15, 0.88, 30, 2.2, 40, 55, GradeLabel.UNGRADED, 0.35),
    ("Striploin Whole", "Striploin Steak Boneless", "striploin-steak-bnls", 42.99, 23, 0.78, 35, 1.6, 50, 75, GradeLabel.UNGRADED, 1.0),
    ("Bone-in Ribeye", "Boneless Ribeye Steak", "ribeye-steak-bnls", 49.99, 20, 0.67, 30, 1.6, 15, 25, GradeLabel.UNGRADED, 1.0),
    ("Chuck Flat", "Simmering Short Ribs", "short-ribs-simmering", 44.99, 20, 0.82, 35, 1.2, 70, 95, GradeLabel.UNGRADED, 1.0),
    ("Top Sirloin Butt", "Top Sirloin Steak Cap Removed", "top-sirloin-steak-cr", 33.99, 15, 0.84, 35, 1.4, 25, 45, GradeLabel.UNGRADED, 1.0),
    ("Flank", "Flank Steak", "flank-steak", 35.99, 10, 0.80, 20, 0.8, 15, 25, GradeLabel.UNGRADED, 1.0),
    ("Blade Eyes", "Thinly Sliced Blades", "blade-thin-sliced", 24.99, 15, 0.84, 30, 2.0, 120, 190, GradeLabel.UNGRADED, 0.65),
    ("Pork Belly", "Thinly Cut Pork Bellies", "pork-belly-thin-cut", 12.99, 7, 0.90, 25, 2.5, 100, 150, GradeLabel.UNGRADED, 1.0),
    ("Wagyu Flat Iron", "Wagyu Flat Iron Thin Sliced", "wagyu-flatiron-thin", 65.99, 8, 0.83, 13, 0.8, 10, 18, GradeLabel.WAGYU, 1.0),
    ("6X6 Frenched", "Tomahawk Steak", "tomahawk-steak", 54.99, 8, 0.89, 25, None, 12, 18, GradeLabel.UNGRADED, 1.0),
]

# Tenderloin is handled separately: it's PER_PIECE, not PER_BOX — only a
# couple of primal pieces get pulled from the ~40kg box at a time, 5 min
# labor per piece, because cutting the whole box wouldn't sell before
# spoiling. Confirmed with user: avg 4-5kg per piece, using midpoint 4.5kg.
_TENDERLOIN_AVG_PIECE_KG = 4.5  # confirmed range 4-5kg

# Tomahawk is "one per tray" rather than a fixed kg pack weight — treat it
# as an EACH-unit product instead of KG, with avg_pack_weight_kg left as a
# placeholder derived from box_weight / (yield_pct implied count). Flagged
# for the user to confirm real avg piece weight.
_TOMAHAWK_AVG_EACH_KG = 1.75  # confirmed range 1.5-2kg

products: list[Product] = []
yield_profiles: list[YieldProfile] = []
demand_profiles: list[DemandProfile] = []

for (primal, name, sku, price, labor_min, yield_pct, box_kg, pack_kg,
     wd_sales, we_sales, grade, cut_share) in _ROWS:

    unit = Unit.KG
    pack_weight = pack_kg
    if pack_kg is None:
        # Tomahawk special case: sold by the piece.
        unit = Unit.EACH
        pack_weight = _TOMAHAWK_AVG_EACH_KG

    products.append(Product(
        sku=sku, name=name, unit=unit,
        price_per_kg=price, avg_pack_weight_kg=pack_weight,
        data_source=DataSource.REAL,
    ))

    yield_profiles.append(YieldProfile(
        source_primal=primal, product_sku=sku, grade=grade,
        yield_pct=yield_pct, labor_basis=LaborBasis.PER_BOX,
        labor_minutes_per_box=labor_min, avg_box_weight_kg=box_kg,
        cut_plan_share=cut_share, data_source=DataSource.REAL,
        primal_cost_per_kg=cost_per_kg(primal),
    ))

    demand_profiles.append(DemandProfile(
        product_sku=sku,
        avg_sales_weekday_kg=wd_sales, avg_sales_weekend_kg=we_sales,
        data_source=DataSource.REAL,
    ))

# --- Tenderloin: PER_PIECE labor basis ---
products.append(Product(
    sku="tenderloin-steak", name="Tenderloin Steak", unit=Unit.KG,
    price_per_kg=65.99, avg_pack_weight_kg=0.8, data_source=DataSource.REAL,
))
yield_profiles.append(YieldProfile(
    source_primal="Tenderloin", product_sku="tenderloin-steak", grade=GradeLabel.UNGRADED,
    yield_pct=0.77, labor_basis=LaborBasis.PER_PIECE,
    labor_minutes_per_piece=5, avg_piece_weight_kg=_TENDERLOIN_AVG_PIECE_KG,
    cut_plan_share=1.0, data_source=DataSource.REAL,
    primal_cost_per_kg=cost_per_kg("Tenderloin"),
    physical_delivery_box_weight_kg=40.0,  # real box weight, confirmed with user; distinct from the
                                            # ~4.5kg piece weight used for labor/yield math above
))
demand_profiles.append(DemandProfile(
    product_sku="tenderloin-steak",
    avg_sales_weekday_kg=4, avg_sales_weekend_kg=9, data_source=DataSource.REAL,
))
