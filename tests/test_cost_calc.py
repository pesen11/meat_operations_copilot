import pytest

from models.domain import Product, YieldProfile, LaborBasis
from simulation.cost_calc import (
    primal_cost_per_kg, pack_economics, weekly_projection,
    DEFAULT_TARGET_GROSS_MARGIN_PCT, DEFAULT_LABOR_RATE_PER_HOUR,
)
from models.domain import DemandProfile


@pytest.fixture
def chuck_product():
    return Product(sku="chuck-roast", name="Chuck Roast", price_per_kg=23.49, avg_pack_weight_kg=2.2)


@pytest.fixture
def chuck_yield():
    return YieldProfile(
        source_primal="Blade Eyes", product_sku="chuck-roast",
        yield_pct=0.88, labor_basis=LaborBasis.PER_BOX,
        labor_minutes_per_box=15, avg_box_weight_kg=30,
        primal_cost_per_kg=12.20,  # data/primal_costs.py
    )


@pytest.fixture
def ribeye_product():
    return Product(sku="ribeye-steak-bnls", name="Boneless Ribeye Steak", price_per_kg=49.99, avg_pack_weight_kg=1.6)


@pytest.fixture
def ribeye_yield_uncosted():
    """No vendor cost on the profile - exercises the fallback path."""
    return YieldProfile(
        source_primal="Bone-in Ribeye", product_sku="ribeye-steak-bnls",
        yield_pct=0.67, labor_basis=LaborBasis.PER_BOX,
        labor_minutes_per_box=20, avg_box_weight_kg=30,
    )


@pytest.fixture
def ribeye_yield_costed():
    return YieldProfile(
        source_primal="Bone-in Ribeye", product_sku="ribeye-steak-bnls",
        yield_pct=0.67, labor_basis=LaborBasis.PER_BOX,
        labor_minutes_per_box=20, avg_box_weight_kg=30,
        primal_cost_per_kg=22.25,  # data/primal_costs.py
    )


class TestPrimalCostResolution:
    def test_costed_primal_is_used_verbatim(self, chuck_product, chuck_yield):
        assert primal_cost_per_kg(chuck_product, chuck_yield) == pytest.approx(12.20)

    def test_uncosted_primal_falls_back_to_target_margin(self, ribeye_product, ribeye_yield_uncosted):
        cost = primal_cost_per_kg(ribeye_product, ribeye_yield_uncosted)
        assert cost == pytest.approx(
            ribeye_product.price_per_kg * ribeye_yield_uncosted.yield_pct
            * (1 - DEFAULT_TARGET_GROSS_MARGIN_PCT)
        )

    def test_primal_costs_much_less_per_kg_than_retail(self, ribeye_product, ribeye_yield_costed):
        """
        The bug this model replaced: cost was derived as a small discount off
        the RETAIL price, so the shop was effectively modelled as buying
        primals at roughly what it sells them for. A primal that costs more
        than about two-thirds of shelf price cannot carry a real shop.
        """
        cost = primal_cost_per_kg(ribeye_product, ribeye_yield_costed)
        assert cost < ribeye_product.price_per_kg * 0.67

    def test_cost_is_yield_aware(self, ribeye_product, ribeye_yield_uncosted):
        """
        The specific defect in the old discount model: it ignored yield, so
        a 22.5% discount on a 67%-yield cut still put COGS above revenue.
        Under the fallback, halving the yield must halve the primal price
        you can afford to pay.
        """
        low = ribeye_yield_uncosted.model_copy(update={"yield_pct": 0.335})
        assert primal_cost_per_kg(ribeye_product, low) == pytest.approx(
            primal_cost_per_kg(ribeye_product, ribeye_yield_uncosted) / 2
        )

    def test_low_yield_premium_cut_still_clears_a_real_margin(
        self, ribeye_product, ribeye_yield_costed
    ):
        """Ribeye is the worst case - 67% yield on the highest-priced beef."""
        e = pack_economics(ribeye_yield_costed, ribeye_product)
        assert e.margin_pct > 0.25


class TestPackEconomics:
    def test_revenue_is_price_times_pack_weight(self, chuck_yield, chuck_product):
        e = pack_economics(chuck_yield, chuck_product)
        assert e.revenue_per_pack == pytest.approx(23.49 * 2.2, abs=0.01)

    def test_margin_equals_revenue_minus_costs(self, chuck_yield, chuck_product):
        e = pack_economics(chuck_yield, chuck_product)
        assert e.margin_per_pack == pytest.approx(
            e.revenue_per_pack - e.primal_cost_per_pack - e.labor_cost_per_pack, abs=0.01
        )

    def test_margin_pct_is_margin_over_revenue(self, chuck_yield, chuck_product):
        e = pack_economics(chuck_yield, chuck_product)
        assert e.margin_pct == pytest.approx(e.margin_per_pack / e.revenue_per_pack, abs=0.001)


class TestWeeklyProjection:
    @pytest.fixture
    def chuck_demand(self):
        return DemandProfile(product_sku="chuck-roast", avg_sales_weekday_kg=40, avg_sales_weekend_kg=55)

    def test_baseline_weekly_kg_matches_7day_demand(self, chuck_yield, chuck_product, chuck_demand):
        proj = weekly_projection(chuck_yield, chuck_product, chuck_demand)
        assert proj.weekly_kg_sold == pytest.approx(40 * 5 + 55 * 2)

    def test_demand_multiplier_scales_linearly(self, chuck_yield, chuck_product, chuck_demand):
        base = weekly_projection(chuck_yield, chuck_product, chuck_demand)
        scaled = weekly_projection(chuck_yield, chuck_product, chuck_demand, demand_multiplier=1.15)
        assert scaled.weekly_kg_sold == pytest.approx(base.weekly_kg_sold * 1.15, abs=0.01)
        assert scaled.weekly_labor_minutes == pytest.approx(base.weekly_labor_minutes * 1.15, abs=0.1)

    def test_margin_increases_with_higher_demand(self, chuck_yield, chuck_product, chuck_demand):
        base = weekly_projection(chuck_yield, chuck_product, chuck_demand)
        scaled = weekly_projection(chuck_yield, chuck_product, chuck_demand, demand_multiplier=1.15)
        assert scaled.weekly_margin > base.weekly_margin
