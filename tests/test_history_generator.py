from datetime import date

from simulation.history_generator import (
    generate_sales_history, generate_production_schedule,
    generate_primal_stock_history, generate_labor_history,
    _primal_reference_weight_kg, TIER_STOCK_OVERRIDES,
)


class TestPrimalReferenceWeight:
    def test_tenderloin_uses_real_box_weight_not_piece_weight(self):
        ref = _primal_reference_weight_kg()
        assert ref["Tenderloin"] == 40.0  # not 4.5 (the piece weight used for labor math)

    def test_per_box_primal_uses_its_box_weight(self):
        ref = _primal_reference_weight_kg()
        assert ref["Blade Eyes"] == 30.0

    def test_tenderloin_has_override_range(self):
        assert TIER_STOCK_OVERRIDES["Tenderloin"] == (1.0, 2.0)


class TestPrimalStockInvariants:
    def test_stock_never_negative(self):
        stock = generate_primal_stock_history(date(2025, 1, 1), date(2025, 3, 31))
        assert all(s.boxes_on_hand >= 0 for s in stock)

    def test_tenderloin_stays_within_reasonable_range_of_override(self):
        stock = generate_primal_stock_history(date(2025, 1, 1), date(2025, 3, 31))
        tenderloin_vals = [s.boxes_on_hand for s in stock if s.source_primal == "Tenderloin"]
        assert max(tenderloin_vals) <= 2.5  # override target is 1-2, allow small headroom
        assert min(tenderloin_vals) >= 0.0

    def test_no_stock_records_on_closed_days(self):
        stock = generate_primal_stock_history(date(2025, 12, 20), date(2025, 12, 27))
        dates_present = {s.as_of for s in stock}
        assert date(2025, 12, 25) not in dates_present
        assert date(2025, 12, 26) not in dates_present


class TestSalesHistory:
    def test_no_sales_on_closed_days(self):
        sales = generate_sales_history(date(2025, 12, 20), date(2025, 12, 27))
        dates_present = {s.sale_date for s in sales}
        assert date(2025, 12, 25) not in dates_present

    def test_regular_week_revenue_close_to_baseline(self):
        # Week of Jan 13, 2025 has no holidays and is outside summer/Christmas windows.
        sales = generate_sales_history(date(2025, 1, 13), date(2025, 1, 19))
        total_revenue = sum(s.revenue for s in sales)
        # Noise means it won't hit $140,000 exactly, but should be in a sane band.
        assert 100_000 < total_revenue < 180_000

    def test_all_revenue_and_units_non_negative(self):
        sales = generate_sales_history(date(2025, 6, 1), date(2025, 6, 7))
        assert all(s.units_sold >= 0 and s.revenue >= 0 for s in sales)


class TestProductionSchedule:
    def test_planned_kg_positive_for_active_primals(self):
        entries = generate_production_schedule(date(2025, 3, 10), date(2025, 3, 10))
        assert len(entries) > 0
        assert all(e.planned_primal_kg > 0 for e in entries)

    def test_no_entries_on_closed_days(self):
        entries = generate_production_schedule(date(2025, 12, 25), date(2025, 12, 25))
        assert entries == []


class TestLaborHistory:
    def test_weekday_role_counts(self):
        labor = generate_labor_history(date(2025, 3, 10), date(2025, 3, 10))  # a Monday
        by_role = {l.role: l.headcount for l in labor}
        assert by_role["cutter"] == 2
        assert by_role["closing"] == 2
        assert by_role["mid_wrapper_counter"] == 1

    def test_weekend_adds_extra_staff(self):
        labor = generate_labor_history(date(2025, 3, 15), date(2025, 3, 15))  # a Saturday
        by_role = {l.role: l.headcount for l in labor}
        assert by_role["cutter"] == 2  # unchanged per user's description
        assert by_role["closing"] == 3  # +1
        assert by_role["mid_wrapper_counter"] == 2  # +1

    def test_no_labor_on_closed_days(self):
        labor = generate_labor_history(date(2025, 12, 25), date(2025, 12, 25))
        assert labor == []
