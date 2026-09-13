"""
Unified catalog: real (seed_yield_data) + synthetic (synthetic_yield_data)
combined into single lists, plus a sanity check that multi-path primals'
cut_plan_share values actually sum to ~1.0.
"""

from __future__ import annotations

from collections import defaultdict

from data import seed_yield_data as real
from data import synthetic_yield_data as synthetic
from models.domain import Product, YieldProfile, DemandProfile

products: list[Product] = [*real.products, *synthetic.products]
yield_profiles: list[YieldProfile] = [*real.yield_profiles, *synthetic.yield_profiles]
demand_profiles: list[DemandProfile] = [*real.demand_profiles, *synthetic.demand_profiles]

products_by_sku: dict[str, Product] = {p.sku: p for p in products}
demand_by_sku: dict[str, DemandProfile] = {d.product_sku: d for d in demand_profiles}
yield_profile_by_sku: dict[str, YieldProfile] = {yp.product_sku: yp for yp in yield_profiles}


def yield_profiles_by_primal() -> dict[str, list[YieldProfile]]:
    grouped: dict[str, list[YieldProfile]] = defaultdict(list)
    for yp in yield_profiles:
        grouped[yp.source_primal].append(yp)
    return dict(grouped)


def primal_reference_box_weight_kg() -> dict[str, float]:
    """
    Reference delivery-box weight per primal, for converting kg <-> box counts.

    Prefers each path's physical_delivery_box_weight_kg (the REAL box that
    gets delivered); falls back to processing_unit_weight_kg, which is the
    same thing for PER_BOX primals but wrong for PER_PIECE ones like
    tenderloin (~40kg box vs. the ~4.5kg piece labor math is anchored to).
    Single source of truth for both the history generator and inventory_calc.
    """
    ref: dict[str, float] = {}
    for yp in yield_profiles:
        if yp.source_primal in ref:
            continue
        ref[yp.source_primal] = yp.physical_delivery_box_weight_kg or yp.processing_unit_weight_kg
    return ref


def check_cut_plan_shares(tolerance: float = 0.01) -> list[str]:
    """Returns a list of human-readable warnings for any primal whose
    yield paths' cut_plan_share values don't sum to ~1.0."""
    warnings = []
    for primal, paths in yield_profiles_by_primal().items():
        total = sum(p.cut_plan_share for p in paths)
        if abs(total - 1.0) > tolerance:
            skus = ", ".join(p.product_sku for p in paths)
            warnings.append(f"{primal}: cut_plan_share sums to {total:.2f} across [{skus}], expected ~1.0")
    return warnings


def check_primal_costs() -> list[str]:
    """
    Returns warnings for primals that are uncosted, or whose paths disagree
    on what the primal costs.

    You buy a primal, not a cut: two paths off Blade Eyes cannot have been
    bought at two different prices. data/primal_costs.py keys cost by
    primal so this holds by construction — this check exists to catch a
    hand-edited row that breaks it.
    """
    warnings = []
    for primal, paths in yield_profiles_by_primal().items():
        costs = {p.primal_cost_per_kg for p in paths}
        if costs == {None}:
            warnings.append(f"{primal}: no primal_cost_per_kg — add it to data/primal_costs.py")
        elif len(costs) > 1:
            shown = ", ".join("uncosted" if c is None else f"${c:.2f}"
                              for c in sorted(costs, key=lambda c: (c is None, c)))
            warnings.append(f"{primal}: paths disagree on vendor cost ({shown}); one primal, one price")
    return warnings


if __name__ == "__main__":
    print(f"Products: {len(products)}  ({sum(1 for p in products if p.data_source.value == 'real')} real, "
          f"{sum(1 for p in products if p.data_source.value == 'synthetic')} synthetic)")
    print(f"Primals: {len(yield_profiles_by_primal())}")
    for label, warnings in (("cut_plan_share", check_cut_plan_shares()),
                            ("primal cost", check_primal_costs())):
        if warnings:
            print(f"\n{label} warnings:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print(f"\nAll {label} checks pass.")
