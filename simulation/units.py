"""
Unit normalisation between sold "units" and kilograms.

HistoricalSale.units_sold means different things per product: for Unit.KG
products it is already kilograms, for Unit.EACH products (tomahawk steak,
which is sold one-per-tray rather than by weight) it is a pack count. Every
kg-denominated calculation downstream has to normalise first, so it happens
here once rather than being re-derived — and mis-derived — per call site.
"""

from __future__ import annotations

from data.catalog import products_by_sku
from models.domain import Unit


def units_to_kg(sku: str, units: float) -> float:
    """Convert a product's sold units into kilograms of finished product."""
    product = products_by_sku[sku]
    if product.unit == Unit.EACH:
        return units * product.avg_pack_weight_kg
    return units


def kg_to_packs(sku: str, kg: float) -> float:
    """Convert kilograms of finished product into finished pack count."""
    return kg / products_by_sku[sku].avg_pack_weight_kg
