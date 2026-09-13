"""
Synthetic primal/product data — NOT transcribed from the user's real
workplace numbers. Modeled on general industry-realistic yield %, labor
times, and pricing patterns for cuts not covered in seed_yield_data.py.

Every row here is tagged data_source=SYNTHETIC so it stays distinguishable
from the real rows in seed_yield_data.py end to end (models, tests,
eventual README). These numbers are plausible, not audited — treat them as
a starting point to correct with real experience the same way the real
seed data got corrected in conversation.

Includes one deliberate multi-yield-path example (Round -> Eye of Round
Roast + Stew Meat) to exercise the cut_plan_share feature with clean,
non-real data, since editing the user's real Blade Eyes rows wasn't
appropriate.
"""

from __future__ import annotations

from data.primal_costs import cost_per_kg
from models.domain import Product, YieldProfile, DemandProfile, GradeLabel, Unit, LaborBasis, DataSource

_ROWS = [
    # source_primal, product name, sku, price_per_kg, labor_min_per_box, yield_pct,
    # avg_box_weight_kg, avg_pack_weight_kg, avg_sales_weekday_kg, avg_sales_weekend_kg, grade, cut_plan_share
    ("Brisket", "Beef Brisket Whole Trimmed", "brisket-whole-trimmed", 18.99, 40, 0.85, 40, 4.5, 20, 35, GradeLabel.UNGRADED, 1.0),
    ("Round", "Eye of Round Roast", "eye-of-round-roast", 19.99, 12, 0.90, 30, 1.8, 25, 35, GradeLabel.UNGRADED, 0.55),
    ("Round", "Beef Stew Meat", "beef-stew-meat", 16.99, 14, 0.80, 30, 1.0, 30, 40, GradeLabel.UNGRADED, 0.45),
    ("Plate", "Skirt Steak", "skirt-steak", 38.99, 12, 0.70, 15, 0.6, 10, 18, GradeLabel.UNGRADED, 1.0),
    ("Shank", "Osso Buco Cross-Cut Shank", "osso-buco", 17.99, 15, 0.85, 20, 0.7, 8, 15, GradeLabel.UNGRADED, 1.0),
    ("Chuck Primal", "Ground Beef Medium", "ground-beef-medium", 13.99, 18, 0.92, 35, 0.5, 150, 220, GradeLabel.UNGRADED, 1.0),
    ("Pork Loin", "Boneless Pork Chops", "pork-chops-bnls", 14.99, 12, 0.85, 22, 0.5, 60, 90, GradeLabel.UNGRADED, 1.0),
    ("Pork Shoulder", "Pulled Pork Roast (Boston Butt)", "pulled-pork-roast", 10.99, 10, 0.90, 30, 2.0, 25, 45, GradeLabel.UNGRADED, 1.0),
]

products: list[Product] = []
yield_profiles: list[YieldProfile] = []
demand_profiles: list[DemandProfile] = []

for (primal, name, sku, price, labor_min, yield_pct, box_kg, pack_kg,
     wd_sales, we_sales, grade, cut_share) in _ROWS:

    products.append(Product(
        sku=sku, name=name, unit=Unit.KG,
        price_per_kg=price, avg_pack_weight_kg=pack_kg,
        data_source=DataSource.SYNTHETIC,
    ))

    yield_profiles.append(YieldProfile(
        source_primal=primal, product_sku=sku, grade=grade,
        yield_pct=yield_pct, labor_basis=LaborBasis.PER_BOX,
        labor_minutes_per_box=labor_min, avg_box_weight_kg=box_kg,
        cut_plan_share=cut_share, data_source=DataSource.SYNTHETIC,
        primal_cost_per_kg=cost_per_kg(primal),
    ))

    demand_profiles.append(DemandProfile(
        product_sku=sku,
        avg_sales_weekday_kg=wd_sales, avg_sales_weekend_kg=we_sales,
        data_source=DataSource.SYNTHETIC,
    ))
