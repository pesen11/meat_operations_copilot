"""
LangChain @tool wrappers around simulation/yield_calc.py and
simulation/cost_calc.py. These tools do a catalog lookup and call an
already-tested deterministic function — they never compute anything
themselves.
"""

from __future__ import annotations

from langchain_core.tools import tool

from data.catalog import products_by_sku, yield_profile_by_sku, demand_by_sku
from simulation.yield_calc import box_breakdown
from simulation.cost_calc import pack_economics, weekly_projection


def _lookup(product_sku: str):
    if product_sku not in products_by_sku:
        raise ValueError(f"Unknown product_sku '{product_sku}'. Call list_products to see valid SKUs.")
    return products_by_sku[product_sku], yield_profile_by_sku[product_sku]


@tool
def list_products() -> list[dict]:
    """List every product in the catalog with its SKU, name, source primal,
    and price per kg. Use this first if you don't already know a product's
    exact SKU string."""
    return [
        {
            "sku": sku,
            "name": p.name,
            "source_primal": yield_profile_by_sku[sku].source_primal,
            "price_per_kg": p.price_per_kg,
            "data_source": p.data_source.value,
        }
        for sku, p in products_by_sku.items()
    ]


@tool
def get_yield_breakdown(product_sku: str) -> dict:
    """Get the yield/waste/labor breakdown for one box or piece of a
    product's source primal: usable product kg, trim waste kg, packs
    produced, and labor minutes per pack."""
    product, yp = _lookup(product_sku)
    b = box_breakdown(yp, product)
    return {
        "product_sku": b.product_sku,
        "source_primal": b.source_primal,
        "usable_product_kg": b.usable_product_kg,
        "trim_waste_kg": b.trim_waste_kg,
        "packs_per_processing_unit": b.packs_per_box,
        "labor_minutes_per_pack": b.labor_minutes_per_pack,
    }


@tool
def get_pack_economics(product_sku: str) -> dict:
    """Get the per-pack economics for a product: revenue, primal cost,
    labor cost, margin (dollar and percent)."""
    product, yp = _lookup(product_sku)
    e = pack_economics(yp, product)
    return {
        "product_sku": e.product_sku,
        "revenue_per_pack": e.revenue_per_pack,
        "primal_cost_per_pack": e.primal_cost_per_pack,
        "labor_cost_per_pack": e.labor_cost_per_pack,
        "margin_per_pack": e.margin_per_pack,
        "margin_pct": e.margin_pct,
    }


@tool
def get_weekly_projection(product_sku: str, demand_multiplier: float = 1.0) -> dict:
    """Project weekly revenue, primal cost, labor minutes/cost, and margin
    for a product from its historical weekday/weekend demand. Use
    demand_multiplier for 'what if' scenarios — e.g. 1.15 for a 15%
    production increase."""
    product, yp = _lookup(product_sku)
    dp = demand_by_sku[product_sku]
    proj = weekly_projection(yp, product, dp, demand_multiplier=demand_multiplier)
    return {
        "product_sku": proj.product_sku,
        "demand_multiplier": demand_multiplier,
        "weekly_kg_sold": proj.weekly_kg_sold,
        "weekly_revenue": proj.weekly_revenue,
        "weekly_primal_cost": proj.weekly_primal_cost,
        "weekly_labor_minutes": proj.weekly_labor_minutes,
        "weekly_labor_cost": proj.weekly_labor_cost,
        "weekly_margin": proj.weekly_margin,
    }


ALL_TOOLS = [list_products, get_yield_breakdown, get_pack_economics, get_weekly_projection]