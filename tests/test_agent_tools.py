"""
Tests for the inventory/history tool belts.

These deliberately test the *wrapper contract* rather than re-testing the
arithmetic (that lives in test_inventory_calc.py / test_history_analysis.py):
that each tool is invokable through LangChain's interface, returns
JSON-serialisable output an LLM can actually consume, and delegates rather
than computing.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from simulation.inventory_tools import (
    INVENTORY_TOOLS, list_primals, get_stock_position, get_all_stock_positions,
    get_cutter_capacity, get_scenario_impact, get_stockout_days,
)
from simulation.history_tools import (
    HISTORY_TOOLS, get_history_range, get_sales_summary, get_sales_trend,
    get_top_movers, get_weekly_revenue_series,
)
from simulation.tools import ALL_TOOLS

LAST_DAY = "2025-12-31"


class TestToolRegistration:
    def test_every_tool_has_a_description(self):
        for t in [*ALL_TOOLS, *INVENTORY_TOOLS, *HISTORY_TOOLS]:
            assert t.description and len(t.description) > 20, f"{t.name} needs a real description"

    def test_tool_names_are_unique_across_all_belts(self):
        names = [t.name for t in [*ALL_TOOLS, *INVENTORY_TOOLS, *HISTORY_TOOLS]]
        assert len(names) == len(set(names))


class TestInventoryTools:
    def test_list_primals_matches_catalog(self):
        primals = list_primals.invoke({})
        assert "Blade Eyes" in primals
        assert "Tenderloin" in primals
        assert primals == sorted(primals)

    def test_stock_position_is_json_serialisable(self):
        result = get_stock_position.invoke(
            {"source_primal": "Blade Eyes", "as_of": LAST_DAY})
        json.dumps(result)  # dates must already be strings
        assert result["as_of"] == LAST_DAY
        assert result["status"] in {
            "stockout", "critical", "below_target", "ok", "over_target"}

    def test_all_stock_positions_covers_every_primal(self):
        rows = get_all_stock_positions.invoke({"as_of": LAST_DAY})
        assert len(rows) == len(list_primals.invoke({}))
        json.dumps(rows)

    def test_cutter_capacity_is_json_serialisable(self):
        result = get_cutter_capacity.invoke({"shift_date": LAST_DAY})
        json.dumps(result)
        assert result["role"] == "cutter"
        assert result["shift_date"] == LAST_DAY

    def test_scenario_impact_returns_the_three_impact_dimensions(self):
        result = get_scenario_impact.invoke(
            {"product_sku": "chuck-roast", "demand_multiplier": 1.15, "as_of": LAST_DAY})
        json.dumps(result)
        # inventory, labor, waste - the three things the project promises
        assert "extra_weekly_boxes" in result
        assert "extra_weekly_cutting_minutes" in result
        assert "extra_weekly_trim_waste_kg" in result
        assert result["demand_multiplier"] == 1.15

    def test_scenario_impact_rejects_unknown_sku(self):
        with pytest.raises(ValueError, match="Unknown product_sku"):
            get_scenario_impact.invoke({"product_sku": "nope", "demand_multiplier": 1.1})

    def test_stockout_days_window_is_inclusive(self):
        result = get_stockout_days.invoke(
            {"source_primal": "Blade Eyes", "start": "2025-01-01", "end": "2025-01-31"})
        assert result["window_days"] == 31
        assert 0 <= result["stockout_days"] <= 31


class TestHistoryTools:
    def test_history_range_covers_2025(self):
        result = get_history_range.invoke({})
        assert result["earliest"].startswith("2025-")
        assert result["latest"] == LAST_DAY

    def test_sales_summary_is_json_serialisable(self):
        result = get_sales_summary.invoke({"sku": "chuck-roast", "end": LAST_DAY})
        json.dumps(result)
        assert result["total_kg"] > 0

    def test_sales_trend_exposes_the_seasonality_flag(self):
        result = get_sales_trend.invoke({"sku": "chuck-roast", "as_of": LAST_DAY})
        json.dumps(result)
        assert "seasonality_explains_change" in result
        assert result["direction"] in {"up", "down", "flat", "unknown"}

    def test_top_movers_respects_limit(self):
        rows = get_top_movers.invoke({"limit": 3, "end": LAST_DAY})
        assert len(rows) == 3
        json.dumps(rows)

    def test_weekly_revenue_series_is_chronological(self):
        rows = get_weekly_revenue_series.invoke(
            {"start": "2025-10-01", "end": LAST_DAY})
        weeks = [r["week_ending"] for r in rows]
        assert weeks == sorted(weeks)
