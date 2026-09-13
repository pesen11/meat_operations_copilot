import pytest

from models.domain import Product, YieldProfile, LaborBasis
from simulation.yield_calc import (
    usable_product_kg, trim_waste_kg, packs_per_processing_unit,
    labor_minutes_per_pack, box_breakdown,
)


@pytest.fixture
def chuck_product():
    return Product(sku="chuck-roast", name="Chuck Roast", price_per_kg=23.49, avg_pack_weight_kg=2.2)


@pytest.fixture
def chuck_yield():
    return YieldProfile(
        source_primal="Blade Eyes", product_sku="chuck-roast",
        yield_pct=0.88, labor_basis=LaborBasis.PER_BOX,
        labor_minutes_per_box=15, avg_box_weight_kg=30,
    )


@pytest.fixture
def tenderloin_product():
    return Product(sku="tenderloin-steak", name="Tenderloin Steak", price_per_kg=65.99, avg_pack_weight_kg=0.8)


@pytest.fixture
def tenderloin_yield():
    return YieldProfile(
        source_primal="Tenderloin", product_sku="tenderloin-steak",
        yield_pct=0.77, labor_basis=LaborBasis.PER_PIECE,
        labor_minutes_per_piece=5, avg_piece_weight_kg=4.5,
    )


class TestPerBox:
    def test_usable_product_kg(self, chuck_yield):
        assert usable_product_kg(chuck_yield) == pytest.approx(30 * 0.88)

    def test_trim_waste_defaults_to_complement_of_yield(self, chuck_yield):
        assert chuck_yield.trim_waste_pct == pytest.approx(0.12)
        assert trim_waste_kg(chuck_yield) == pytest.approx(30 * 0.12)

    def test_packs_per_box(self, chuck_yield, chuck_product):
        expected = (30 * 0.88) / 2.2
        assert packs_per_processing_unit(chuck_yield, chuck_product) == pytest.approx(expected)

    def test_labor_minutes_per_pack(self, chuck_yield, chuck_product):
        packs = (30 * 0.88) / 2.2
        assert labor_minutes_per_pack(chuck_yield, chuck_product) == pytest.approx(15 / packs)

    def test_box_breakdown_rounds_and_matches(self, chuck_yield, chuck_product):
        b = box_breakdown(chuck_yield, chuck_product)
        assert b.source_primal == "Blade Eyes"
        assert b.packs_per_box == pytest.approx(12.0, abs=0.01)
        assert b.labor_minutes_per_pack == pytest.approx(1.25, abs=0.01)


class TestPerPiece:
    def test_processing_unit_uses_piece_weight_not_box(self, tenderloin_yield):
        assert tenderloin_yield.processing_unit_weight_kg == 4.5
        assert tenderloin_yield.labor_minutes_per_processing_unit == 5

    def test_usable_product_kg_from_piece(self, tenderloin_yield):
        assert usable_product_kg(tenderloin_yield) == pytest.approx(4.5 * 0.77)

    def test_labor_minutes_per_pack_independent_of_box_size(self, tenderloin_yield, tenderloin_product):
        # No avg_box_weight_kg was ever set for this profile — confirms per-piece
        # math never touches box weight at all.
        assert tenderloin_yield.avg_box_weight_kg is None
        packs = (4.5 * 0.77) / 0.8
        assert labor_minutes_per_pack(tenderloin_yield, tenderloin_product) == pytest.approx(5 / packs)


class TestValidation:
    def test_per_box_missing_fields_raises(self):
        with pytest.raises(ValueError, match="labor_basis=per_box requires"):
            YieldProfile(
                source_primal="Blade Eyes", product_sku="chuck-roast",
                yield_pct=0.88, labor_basis=LaborBasis.PER_BOX,
                # avg_box_weight_kg / labor_minutes_per_box both missing
            )

    def test_per_piece_missing_fields_raises(self):
        with pytest.raises(ValueError, match="labor_basis=per_piece requires"):
            YieldProfile(
                source_primal="Tenderloin", product_sku="tenderloin-steak",
                yield_pct=0.77, labor_basis=LaborBasis.PER_PIECE,
                # avg_piece_weight_kg / labor_minutes_per_piece both missing
            )

    def test_yield_plus_waste_over_one_raises(self):
        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            YieldProfile(
                source_primal="Blade Eyes", product_sku="chuck-roast",
                yield_pct=0.88, trim_waste_pct=0.5,
                labor_basis=LaborBasis.PER_BOX,
                labor_minutes_per_box=15, avg_box_weight_kg=30,
            )

    def test_product_sku_mismatch_raises(self, chuck_yield):
        wrong_product = Product(sku="other-sku", name="X", price_per_kg=1, avg_pack_weight_kg=1)
        with pytest.raises(ValueError, match="does not match"):
            packs_per_processing_unit(chuck_yield, wrong_product)
