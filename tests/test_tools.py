import pytest

from simulation.tools import list_products, get_yield_breakdown, get_pack_economics, get_weekly_projection


class TestListProducts:
    def test_returns_all_catalog_products(self):
        result = list_products.invoke({})
        assert len(result) == 19

    def test_each_entry_has_expected_keys(self):
        result = list_products.invoke({})
        assert set(result[0].keys()) == {"sku", "name", "source_primal", "price_per_kg", "data_source"}


class TestGetYieldBreakdown:
    def test_known_values_for_chuck_roast(self):
        result = get_yield_breakdown.invoke({"product_sku": "chuck-roast"})
        assert result["packs_per_processing_unit"] == 12.0
        assert result["labor_minutes_per_pack"] == 1.25

    def test_unknown_sku_raises(self):
        with pytest.raises(ValueError, match="Unknown product_sku"):
            get_yield_breakdown.invoke({"product_sku": "nonexistent"})


class TestGetPackEconomics:
    def test_margin_matches_direct_calculation(self):
        result = get_pack_economics.invoke({"product_sku": "chuck-roast"})
        assert result["margin_per_pack"] == pytest.approx(20.615, abs=0.01)
        assert result["margin_pct"] == pytest.approx(0.3989, abs=0.001)


class TestGetWeeklyProjection:
    def test_default_multiplier_is_baseline(self):
        result = get_weekly_projection.invoke({"product_sku": "chuck-roast"})
        assert result["demand_multiplier"] == 1.0
        assert result["weekly_margin"] == pytest.approx(2904.91, abs=0.5)

    def test_scenario_multiplier_scales_correctly(self):
        result = get_weekly_projection.invoke({"product_sku": "chuck-roast", "demand_multiplier": 1.15})
        assert result["weekly_margin"] == pytest.approx(3340.65, abs=0.5)