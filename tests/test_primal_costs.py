"""
Catalog-level economics checks.

These are the tests that would have caught the bug data/primal_costs.py
exists to fix. Every prior test asserted that the math was *internally
consistent* — margin equals revenue minus costs, and so on — which it was.
Nothing asserted that the resulting shop could pay rent, so a cost model
that put COGS at 80-97% of revenue passed 57 tests and shipped.

The band below is the project owner's confirmed figure for this shop:
roughly 38-40% gross margin, net of primal cost and cutting labor.
"""

from __future__ import annotations

import pytest

from data import catalog
from data.primal_costs import PRIMAL_COST_PER_KG, recalibrate
from simulation.cost_calc import pack_economics, primal_cost_per_kg, weekly_projection

TARGET_MARGIN_LOW = 0.38
TARGET_MARGIN_HIGH = 0.40


def _weekly_totals() -> tuple[float, float]:
    revenue = margin = 0.0
    for product in catalog.products:
        yp = catalog.yield_profile_by_sku[product.sku]
        demand = catalog.demand_by_sku[product.sku]
        proj = weekly_projection(yp, product, demand)
        revenue += proj.weekly_revenue
        margin += proj.weekly_margin
    return revenue, margin


class TestBlendedMargin:
    def test_catalog_blended_margin_is_in_the_confirmed_band(self):
        revenue, margin = _weekly_totals()
        assert TARGET_MARGIN_LOW <= margin / revenue <= TARGET_MARGIN_HIGH

    def test_weekly_margin_is_an_operationally_plausible_amount(self):
        """
        The symptom that surfaced the bug: the old model returned margins in
        the hundreds of dollars per item per week, on a store doing $400k/wk.
        """
        _, margin = _weekly_totals()
        assert margin > 50_000

    def test_every_product_clears_a_positive_margin(self):
        for product in catalog.products:
            yp = catalog.yield_profile_by_sku[product.sku]
            e = pack_economics(yp, product)
            assert e.margin_per_pack > 0, f"{product.sku} loses money per pack"

    def test_no_product_margin_is_wildly_off_the_blend(self):
        """
        Individual cuts legitimately vary - premium steaks run thinner than
        grinds, and a multi-path primal splits its margin unevenly across
        paths. A value outside this range means a bad price or yield, not a
        cut-plan effect.
        """
        for product in catalog.products:
            yp = catalog.yield_profile_by_sku[product.sku]
            e = pack_economics(yp, product)
            assert 0.25 <= e.margin_pct <= 0.55, f"{product.sku} at {e.margin_pct:.1%}"


class TestVendorCosts:
    def test_every_catalog_primal_is_costed(self):
        missing = sorted(set(catalog.yield_profiles_by_primal()) - set(PRIMAL_COST_PER_KG))
        assert not missing, f"uncosted primals: {missing}"

    def test_catalog_reports_no_cost_warnings(self):
        assert catalog.check_primal_costs() == []

    def test_paths_off_one_primal_share_one_cost(self):
        """You buy a primal, not a cut — Blade Eyes cannot cost two prices."""
        for primal, paths in catalog.yield_profiles_by_primal().items():
            costs = {p.primal_cost_per_kg for p in paths}
            assert len(costs) == 1, f"{primal} has {len(costs)} different costs"

    def test_vendor_cost_is_well_below_retail(self):
        """
        The core correction: price_per_kg is a RETAIL shelf price, and a
        primal is bought in bulk untrimmed. Paying anywhere near shelf price
        per kg of primal is what produced the 9.5% blended margin.
        """
        for product in catalog.products:
            yp = catalog.yield_profile_by_sku[product.sku]
            cost = primal_cost_per_kg(product, yp)
            assert cost < product.price_per_kg * 0.7, f"{product.sku} primal near retail"

    def test_committed_table_matches_a_fresh_recalibration(self):
        """
        The table is committed data, not computed at import — so it can drift
        if a retail price or yield changes without `python -m data.primal_costs
        --recalibrate` being re-run. This is that drift alarm.
        """
        fresh = recalibrate()
        drifted = {
            primal: (PRIMAL_COST_PER_KG[primal], cost)
            for primal, cost in fresh.items()
            if PRIMAL_COST_PER_KG[primal] != pytest.approx(cost)
        }
        assert not drifted, f"stale costs (committed, fresh): {drifted}"
