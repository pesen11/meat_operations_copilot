"""
Deterministic yield/labor calculators.

Per the project's core design rule: the LLM never computes these numbers.
These are plain, unit-testable Python functions that get wrapped as
LangChain @tool later. Everything here operates on Pydantic domain models
and returns plain floats/dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.domain import Product, YieldProfile


@dataclass
class BoxBreakdown:
    """Derived, per-box-to-per-pack breakdown for one yield path."""
    product_sku: str
    source_primal: str
    usable_product_kg: float
    trim_waste_kg: float
    packs_per_box: float
    labor_minutes_per_pack: float


def usable_product_kg(yp: YieldProfile) -> float:
    """Kg of finished product recoverable from one processing unit (box or piece)."""
    return yp.processing_unit_weight_kg * yp.yield_pct


def trim_waste_kg(yp: YieldProfile) -> float:
    """Kg lost to trim/waste from one processing unit (box or piece)."""
    return yp.processing_unit_weight_kg * yp.trim_waste_pct


def packs_per_processing_unit(yp: YieldProfile, product: Product) -> float:
    """
    How many sellable packs one processing unit (box or piece) yields.

    unit weight -> (x yield_pct) -> usable product weight -> (/ pack weight)
    -> pack count. This is a real count (can be fractional in aggregate;
    round only when simulating a single physical box/piece).
    """
    if product.sku != yp.product_sku:
        raise ValueError(f"Product {product.sku} does not match YieldProfile.product_sku {yp.product_sku}")
    return usable_product_kg(yp) / product.avg_pack_weight_kg


def labor_minutes_per_pack(yp: YieldProfile, product: Product) -> float:
    """
    Per-pack labor, derived by spreading the processing-unit-level labor
    time (open/trim/tie/cut/tray for a box, or trim/cut/tray for a single
    piece) across however many packs that unit actually yields.
    """
    ppu = packs_per_processing_unit(yp, product)
    if ppu <= 0:
        raise ValueError("packs_per_processing_unit must be positive to derive per-pack labor")
    return yp.labor_minutes_per_processing_unit / ppu


def box_breakdown(yp: YieldProfile, product: Product) -> BoxBreakdown:
    return BoxBreakdown(
        product_sku=product.sku,
        source_primal=yp.source_primal,
        usable_product_kg=round(usable_product_kg(yp), 3),
        trim_waste_kg=round(trim_waste_kg(yp), 3),
        packs_per_box=round(packs_per_processing_unit(yp, product), 2),
        labor_minutes_per_pack=round(labor_minutes_per_pack(yp, product), 2),
    )
