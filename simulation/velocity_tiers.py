"""
Classifies each primal into a velocity tier (low/medium/best seller),
which determines the box-count range it's normally stocked at.
Confirmed with user: low ~5-6 boxes, medium ~8-12 boxes, best ~20-25 boxes.

Classification is based on each primal's total weekly primal-kg throughput
(derived from all its yield paths' product demand / yield_pct), ranked
into terciles across the catalog's primals. This is inference, not a
number the user gave directly — flagged as an assumption; correct the
_TIER_BOX_RANGES or the tier assignment logic if the real cutoffs differ.
"""

from __future__ import annotations

from collections import defaultdict

from data.catalog import yield_profiles, products_by_sku, demand_by_sku
from models.domain import VelocityTier

TIER_BOX_RANGES: dict[VelocityTier, tuple[float, float]] = {
    VelocityTier.LOW: (5, 6),
    VelocityTier.MEDIUM: (8, 12),
    VelocityTier.BEST: (20, 25),
}


def primal_weekly_kg_throughput() -> dict[str, float]:
    """Total weekly kg of PRIMAL (not finished product) each primal needs to
    supply, summed across all its yield paths, derived from each path's
    product demand divided by that path's yield_pct."""
    throughput: dict[str, float] = defaultdict(float)
    for yp in yield_profiles:
        dp = demand_by_sku[yp.product_sku]
        weekly_product_kg = dp.avg_sales_weekday_kg * 5 + dp.avg_sales_weekend_kg * 2
        throughput[yp.source_primal] += weekly_product_kg / yp.yield_pct
    return dict(throughput)


def classify_velocity_tiers() -> dict[str, VelocityTier]:
    """Ranks primals by weekly kg throughput into terciles -> tier labels."""
    throughput = primal_weekly_kg_throughput()
    ranked = sorted(throughput.items(), key=lambda kv: kv[1])
    n = len(ranked)
    tiers: dict[str, VelocityTier] = {}
    for i, (primal, _kg) in enumerate(ranked):
        if i < n / 3:
            tiers[primal] = VelocityTier.LOW
        elif i < 2 * n / 3:
            tiers[primal] = VelocityTier.MEDIUM
        else:
            tiers[primal] = VelocityTier.BEST
    return tiers
