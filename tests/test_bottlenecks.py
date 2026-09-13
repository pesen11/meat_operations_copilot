"""
Tests for simulation/bottlenecks.py.

`test_matches_seasonal_plan_utilization` is the important one. catalog_impact
and seasonal_plan compute shop-wide cutter utilization by separate routes,
and both must de-seasonalise their baseline first. When bottlenecks.py did
not, it reported 119.1% against seasonal_plan's 91.8% - the exact failure
seasonal_planning.py's docstring records having hit for the same reason.
Two independent paths agreeing to four decimals is worth far more than
either one agreeing with itself.
"""

from __future__ import annotations

from datetime import date

import pytest

from simulation import ops_data
from simulation.bottlenecks import (
    BINDING_THRESHOLD, catalog_impact, top_bottlenecks,
)
from simulation.seasonal_planning import PERIODS, seasonal_plan


@pytest.fixture(scope="module")
def as_of() -> date:
    return max(ops_data.load_sales()["sale_date"])


# ---------------------------------------------------------------------------
# Cross-validation against the independently-written seasonal path
# ---------------------------------------------------------------------------

def test_matches_seasonal_plan_utilization(as_of):
    """Both paths de-seasonalise, so both must land on the same number."""
    plan = seasonal_plan("christmas", as_of=as_of)
    impact = catalog_impact(PERIODS["christmas"], as_of=as_of)
    assert impact.cutter_utilization_pct == pytest.approx(
        plan.planned_cutter_utilization_pct, abs=0.002)


def test_matches_seasonal_plan_extra_primal(as_of):
    plan = seasonal_plan("christmas", as_of=as_of)
    impact = catalog_impact(PERIODS["christmas"], as_of=as_of)
    assert impact.total_extra_weekly_primal_kg == pytest.approx(
        plan.total_extra_primal_kg, rel=0.01)


def test_christmas_does_not_report_a_phantom_overrun(as_of):
    """119% was the double-counted figure. The true one is ~92%, which is
    tight but feasible - and the difference between those two numbers is the
    difference between 'roster carefully' and 'this is impossible'."""
    impact = catalog_impact(PERIODS["christmas"], as_of=as_of)
    assert impact.cutter_utilization_pct < 1.0
    assert not impact.exceeds_cutter_capacity


# ---------------------------------------------------------------------------
# Magnitude
# ---------------------------------------------------------------------------

def test_baseline_multiplier_adds_nothing(as_of):
    impact = catalog_impact(1.0, as_of=as_of)
    assert impact.total_extra_weekly_product_kg == pytest.approx(0.0, abs=0.01)
    assert impact.total_extra_weekly_primal_kg == pytest.approx(0.0, abs=0.01)
    assert impact.total_extra_cutting_minutes == pytest.approx(0.0, abs=0.1)


def test_baseline_utilization_is_plausible(as_of):
    """A de-seasonalised normal week should leave real slack. Near 100% would
    mean the shop has no capacity to absorb any season at all."""
    impact = catalog_impact(1.0, as_of=as_of)
    assert 0.30 < impact.cutter_utilization_pct < 0.75


def test_extra_scales_with_the_multiplier(as_of):
    small = catalog_impact(1.1, as_of=as_of, include_risk=False)
    large = catalog_impact(1.2, as_of=as_of, include_risk=False)
    assert large.total_extra_weekly_primal_kg == pytest.approx(
        small.total_extra_weekly_primal_kg * 2, rel=0.02)


def test_primal_draw_exceeds_finished_product(as_of):
    impact = catalog_impact(1.5, as_of=as_of, include_risk=False)
    assert impact.total_extra_weekly_primal_kg > impact.total_extra_weekly_product_kg


def test_every_primal_is_assessed(as_of):
    impact = catalog_impact(1.2, as_of=as_of, include_risk=False)
    known = len(set(ops_data.load_primal_stock()["source_primal"]))
    assert impact.primals_assessed == known


# ---------------------------------------------------------------------------
# Bottleneck ranking
# ---------------------------------------------------------------------------

def test_capacity_bottleneck_appears_only_when_binding(as_of):
    calm = catalog_impact(1.0, as_of=as_of, include_risk=False)
    assert not any(b.kind == "capacity" for b in calm.bottlenecks)
    assert calm.cutter_utilization_pct < BINDING_THRESHOLD

    stressed = catalog_impact(2.4, as_of=as_of, include_risk=False)
    assert stressed.cutter_utilization_pct >= BINDING_THRESHOLD
    assert any(b.kind == "capacity" for b in stressed.bottlenecks)


def test_stress_surfaces_stock_bottlenecks(as_of):
    impact = catalog_impact(1.9, as_of=as_of)
    assert any(b.kind == "stock" for b in impact.bottlenecks)


def test_bottlenecks_are_ranked_by_severity_then_kind(as_of):
    impact = catalog_impact(1.9, as_of=as_of)
    severity_rank = {"over_limit": 0, "critical": 1, "tight": 2, "ok": 3}
    kind_rank = {"capacity": 0, "stock": 1, "supply": 2, "execution": 3}
    keys = [(severity_rank[b.severity], kind_rank[b.kind]) for b in impact.bottlenecks]
    assert keys == sorted(keys)


def test_chronic_execution_is_not_ranked_above_an_emergency(as_of):
    """A primal that always completes 78% of plan is a known, chronic issue.
    It must not outrank a live stockout - an earlier scale labelled it
    'over_limit' and floated it to the top of the list."""
    impact = catalog_impact(1.9, as_of=as_of)
    kinds = [b.kind for b in impact.bottlenecks]
    if "stock" in kinds and "execution" in kinds:
        assert kinds.index("stock") < kinds.index("execution")


def test_execution_bottlenecks_are_never_over_limit(as_of):
    impact = catalog_impact(1.0, as_of=as_of, include_risk=False)
    for bottleneck in impact.bottlenecks:
        if bottleneck.kind == "execution":
            assert bottleneck.severity in ("critical", "tight")


def test_every_bottleneck_names_a_remedy(as_of):
    """A constraint with no suggested action is a complaint, not a finding."""
    impact = catalog_impact(1.9, as_of=as_of)
    for bottleneck in impact.bottlenecks:
        assert bottleneck.remedy
        assert bottleneck.headline
        assert bottleneck.scope


def test_limiting_constraint_is_the_first_bottleneck(as_of):
    impact = catalog_impact(1.9, as_of=as_of)
    assert impact.limiting_constraint == (
        f"{impact.bottlenecks[0].kind}:{impact.bottlenecks[0].scope}")


def test_top_bottlenecks_respects_limit(as_of):
    assert len(top_bottlenecks(1.9, limit=3, as_of=as_of)) <= 3


# ---------------------------------------------------------------------------
# Per-primal detail replaces the meaningless global average
# ---------------------------------------------------------------------------

def test_per_primal_is_sorted_by_size(as_of):
    impact = catalog_impact(1.9, as_of=as_of, include_risk=False)
    values = [r["extra_weekly_primal_kg"] for r in impact.per_primal]
    assert values == sorted(values, reverse=True)


def test_per_primal_rows_carry_stock_and_execution(as_of):
    impact = catalog_impact(1.5, as_of=as_of, include_risk=False)
    for row in impact.per_primal:
        assert "extra_weekly_boxes" in row
        assert "execution_rate" in row
        assert row["paths"], f"{row['source_primal']} has no yield paths"


def test_to_dict_is_json_safe(as_of):
    import json
    json.dumps(catalog_impact(1.9, as_of=as_of).to_dict())
