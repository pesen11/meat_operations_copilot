"""
Tests for simulation/scenario_sweep.py.

`test_product_multiplier_is_translated_to_its_primal` pins a real bug: the
sweep originally passed a PRODUCT multiplier straight into the PRIMAL's
stockout model. Blade Eyes feeds two products, so a 90% increase in chuck
roast was being modelled as a 90% increase in Blade Eyes draw and reported a
99.8% chance of running dry. The true figure is a 19% increase and 1.3%.
"""

from __future__ import annotations

from datetime import date

import pytest

from simulation import ops_data
from simulation.scenario_sweep import (
    DEFAULT_LADDER, HARD_CUTTER_LIMIT, MAX_ACCEPTABLE_STOCKOUT_RISK,
    MAX_SAFE_CUTTER_UTILIZATION, select_option, smallest_option_meeting_risk,
    sweep_catalog, sweep_product,
)


@pytest.fixture(scope="module")
def as_of() -> date:
    return max(ops_data.load_sales()["sale_date"])


@pytest.fixture(scope="module")
def chuck(as_of):
    return sweep_product("chuck-roast", as_of=as_of, target_multiplier=1.15)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def test_ladder_covers_every_rung_plus_the_target(chuck):
    multipliers = {o.demand_multiplier for o in chuck.options}
    for change in DEFAULT_LADDER:
        assert round(1 + change, 4) in multipliers
    assert 1.15 in multipliers


def test_options_are_ordered_by_multiplier(chuck):
    values = [o.demand_multiplier for o in chuck.options]
    assert values == sorted(values)


def test_no_change_rung_changes_nothing(chuck):
    baseline = next(o for o in chuck.options if o.demand_multiplier == 1.0)
    assert baseline.extra_weekly_product_kg == pytest.approx(0.0, abs=0.01)
    assert baseline.delta_weekly_margin == pytest.approx(0.0, abs=0.01)
    assert baseline.label == "no change"


def test_extra_kg_is_monotone_across_the_ladder(chuck):
    values = [o.extra_weekly_product_kg for o in chuck.options]
    assert values == sorted(values)


def test_decreases_are_negative(chuck):
    cut = next(o for o in chuck.options if o.demand_multiplier == 0.8)
    assert cut.extra_weekly_product_kg < 0
    assert cut.delta_weekly_margin < 0


def test_margin_moves_with_volume(chuck):
    """More product at a fixed cost structure must earn more margin."""
    for previous, current in zip(chuck.options, chuck.options[1:]):
        assert current.delta_weekly_margin >= previous.delta_weekly_margin


# ---------------------------------------------------------------------------
# The product -> primal translation
# ---------------------------------------------------------------------------

def test_product_multiplier_is_translated_to_its_primal(as_of):
    """Blade Eyes feeds two products. A big increase in one of them must not
    be modelled as the same increase in the primal's total draw."""
    from simulation.inventory_calc import avg_daily_primal_kg, scenario_impact
    from simulation.scenario_sweep import _primal_multiplier_for_product

    impact = scenario_impact("chuck-roast", 1.9, as_of=as_of)
    effective = _primal_multiplier_for_product(impact, "Blade Eyes", as_of, 28)
    assert 1.0 < effective < 1.9, (
        f"a 90% chuck-roast increase became a {effective:.2f}x draw on a "
        f"primal it only partly feeds"
    )

    weekly = avg_daily_primal_kg(as_of=as_of).get("Blade Eyes", 0.0) * 7
    assert effective == pytest.approx(
        1 + impact.extra_weekly_primal_kg / weekly, abs=1e-4)


def test_single_path_primal_translates_one_to_one(as_of):
    """A primal with exactly one yield path must collapse back to the
    product's own multiplier - otherwise the translation is not a
    generalisation, it is a distortion."""
    from simulation.inventory_calc import scenario_impact
    from simulation.scenario_sweep import _primal_multiplier_for_product
    from data.catalog import yield_profile_by_sku, yield_profiles_by_primal

    sku = "brisket-whole-trimmed"
    primal = yield_profile_by_sku[sku].source_primal
    assert len(yield_profiles_by_primal()[primal]) == 1

    impact = scenario_impact(sku, 1.5, as_of=as_of)
    effective = _primal_multiplier_for_product(impact, primal, as_of, 28)
    assert effective == pytest.approx(1.5, abs=0.05)


def test_risk_stays_plausible_across_the_ladder(chuck):
    for option in chuck.options:
        assert 0.0 <= option.stockout_probability <= 1.0


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def test_feasibility_is_about_capacity_and_supply_not_margin(chuck):
    """A loss-making option is a bad choice, not an impossible one. Ruling it
    out as infeasible would remove a decision the operator is entitled to."""
    cut = next(o for o in chuck.options if o.demand_multiplier == 0.8)
    assert cut.delta_weekly_margin < 0
    assert cut.feasible


def test_infeasible_options_say_why(chuck):
    for option in chuck.options:
        if not option.feasible:
            assert option.infeasible_reasons


def test_capacity_overrun_makes_an_option_infeasible(as_of):
    sweep = sweep_catalog(as_of=as_of, ladder=(0.0, 0.9, 1.4))
    stretched = [o for o in sweep.options if o.cutter_utilization_pct > HARD_CUTTER_LIMIT]
    assert stretched, "ladder never reaches a capacity overrun"
    for option in stretched:
        assert not option.feasible


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------

def test_named_target_is_honoured_when_feasible(chuck):
    """The operator chose 15%. The policy's job is to say whether it fits,
    not to substitute a different number."""
    assert chuck.recommended.demand_multiplier == 1.15
    assert chuck.target_is_feasible is True


def test_infeasible_target_falls_back_to_largest_feasible(as_of):
    sweep = sweep_catalog(as_of=as_of, ladder=(0.0, 0.05, 0.10, 1.4),
                          target_multiplier=2.4)
    assert sweep.target_is_feasible is False
    assert sweep.recommended is not None
    assert sweep.recommended.demand_multiplier < 2.4
    assert sweep.recommended.feasible
    assert "not achievable" in sweep.recommendation_reason


def test_open_question_picks_the_largest_comfortable_increase(as_of):
    """No target named: the policy chooses, and must choose the largest rung
    inside the comfort line rather than the smallest or an arbitrary one."""
    sweep = sweep_catalog(as_of=as_of, ladder=(0.0, 0.05, 0.10, 0.20, 1.4))
    comfortable = [o for o in sweep.options
                   if o.demand_multiplier > 1.0 and o.feasible
                   and o.cutter_utilization_pct <= MAX_SAFE_CUTTER_UTILIZATION]
    assert sweep.recommended.demand_multiplier == max(
        o.demand_multiplier for o in comfortable)


def test_selection_reason_is_always_populated(chuck, as_of):
    assert chuck.recommendation_reason
    assert sweep_catalog(as_of=as_of, ladder=(0.0, 0.1)).recommendation_reason


def test_policy_thresholds_are_reported(chuck):
    """A recommendation whose rule is invisible cannot be argued with."""
    assert chuck.policy["max_safe_cutter_utilization"] == MAX_SAFE_CUTTER_UTILIZATION
    assert chuck.policy["hard_cutter_limit"] == HARD_CUTTER_LIMIT
    assert chuck.policy["rule"]


def test_select_option_handles_an_empty_ladder():
    chosen, reason = select_option([], None)
    assert chosen is None
    assert reason


def test_no_feasible_increase_falls_back_to_no_change(as_of):
    sweep = sweep_catalog(as_of=as_of, ladder=(0.0, 1.4, 2.0))
    if not any(o.feasible for o in sweep.options if o.demand_multiplier > 1.0):
        assert sweep.recommended.demand_multiplier == 1.0


# ---------------------------------------------------------------------------
# The minimum-safe-quantity mirror
# ---------------------------------------------------------------------------

def test_smallest_option_meeting_risk_is_the_least_disruptive(chuck):
    chosen = smallest_option_meeting_risk(chuck.options)
    assert chosen is not None
    assert chosen.stockout_probability <= MAX_ACCEPTABLE_STOCKOUT_RISK
    qualifying = [o for o in chuck.options
                  if o.feasible and o.stockout_probability <= MAX_ACCEPTABLE_STOCKOUT_RISK]
    assert abs(chosen.demand_multiplier - 1.0) == min(
        abs(o.demand_multiplier - 1.0) for o in qualifying)


def test_smallest_option_returns_none_when_nothing_qualifies():
    assert smallest_option_meeting_risk([]) is None


# ---------------------------------------------------------------------------
# Catalog sweep
# ---------------------------------------------------------------------------

def test_catalog_risk_is_the_worst_primal_not_the_average(as_of):
    """The shop stops cutting when any one primal it needs is empty, not
    when the average primal is empty."""
    from simulation.bottlenecks import catalog_impact
    sweep = sweep_catalog(as_of=as_of, ladder=(0.9,))
    option = sweep.options[0]
    impact = catalog_impact(option.demand_multiplier, as_of=as_of)
    risks = [r["stockout_probability"] for r in impact.per_primal
             if r.get("stockout_probability") is not None]
    assert option.stockout_probability == pytest.approx(max(risks), abs=1e-4)


def test_catalog_sweep_has_no_per_product_margin(as_of):
    """Margin across seventeen primals with different cut plans is not one
    number; reporting it would invent a figure nothing computed."""
    sweep = sweep_catalog(as_of=as_of, ladder=(0.0, 0.1))
    for option in sweep.options:
        assert option.weekly_margin is None
        assert option.days_of_cover is None


def test_unknown_sku_raises(as_of):
    with pytest.raises(ValueError):
        sweep_product("not-a-sku", as_of=as_of)


def test_to_dict_is_json_safe(chuck):
    import json
    json.dumps(chuck.to_dict())
