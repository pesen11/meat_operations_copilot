"use client";

/**
 * "Projected Outcome" cards.
 *
 * Deliberately stat tiles rather than charts. Each of these is a single
 * before/after pair - a bar chart of two bars carries no more information
 * than the two numbers and reads slower. The one place a chart earns its
 * space on this page is the baseline-vs-scenario comparison below, where
 * four measures share a frame.
 *
 * Numbers are FORMATTED here, never recomputed: no arithmetic in this file
 * beyond choosing a unit suffix. The backend's number guard checks the
 * narration against these same values, so a display-side calculation would
 * put the UI and the guard out of step.
 */

import type { ProjectedOutcome } from "@/lib/api";

// Every formatter tolerates a missing value. The backend is supposed to send
// complete blocks, and it does - but a partial payload previously called
// .toFixed() on undefined and blanked the entire page, which is a far worse
// failure than one tile reading "-". A display component should degrade, not
// take the app down.
const has = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const money = (value: number) =>
  value.toLocaleString("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 2,
  });

const percent = (value: number) =>
  `${(value * 100).toFixed(1)}%`;

const signed = (value: number, format: (v: number) => string) =>
  `${value >= 0 ? "+" : "-"}${format(Math.abs(value))}`;

function Tile({
  label,
  value,
  delta,
  sub,
}: {
  label: string;
  value: string;
  delta?: string;
  sub?: string;
}) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {delta ? <div className="delta">{delta}</div> : null}
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  );
}

export function OutcomeCards({ outcome }: { outcome: ProjectedOutcome }) {
  const { margin, inventory, labor, waste } = outcome;
  const hasScenario = outcome.demand_multiplier !== 1;

  const tiles: React.ReactNode[] = [];

  if (has(margin?.scenario_weekly_margin) && has(margin?.delta_weekly_margin)) {
    tiles.push(
      <Tile
        key="margin"
        label="Weekly margin"
        value={money(margin.scenario_weekly_margin!)}
        delta={`${signed(margin.delta_weekly_margin!, money)} vs. ${money(
          margin.baseline_weekly_margin ?? 0,
        )} today`}
      />,
    );
  } else if (has(margin?.baseline_weekly_margin)) {
    tiles.push(
      <Tile
        key="margin"
        label="Weekly margin"
        value={money(margin.baseline_weekly_margin!)}
        sub="Current run rate"
      />,
    );
  }

  if (inventory && has(inventory.extra_weekly_boxes)) {
    tiles.push(
      <Tile
        key="inventory"
        label="Extra primal needed"
        value={`${inventory.extra_weekly_boxes.toFixed(2)} boxes/wk`}
        delta={
          has(inventory.extra_weekly_primal_kg)
            ? `${inventory.extra_weekly_primal_kg.toFixed(2)} kg of ${
                outcome.source_primal ?? "primal across all primals"
              }`
            : undefined
        }
        sub={has(inventory.boxes_on_hand)
          ? `${inventory.boxes_on_hand} boxes on hand now`
          : undefined}
      />,
    );
    if (has(inventory.scenario_days_of_cover)) {
      tiles.push(
        <Tile
          key="cover"
          label="Days of cover"
          value={`${inventory.scenario_days_of_cover!.toFixed(2)} days`}
          delta={
            has(inventory.baseline_days_of_cover)
              ? `from ${inventory.baseline_days_of_cover.toFixed(2)} days today`
              : undefined
          }
        />,
      );
    }
  }

  if (labor && has(labor.scenario_cutter_utilization_pct)) {
    tiles.push(
      <Tile
        key="labor"
        label="Cutter utilization"
        value={percent(labor.scenario_cutter_utilization_pct)}
        delta={`from ${percent(labor.baseline_cutter_utilization_pct)} — ${
          labor.extra_weekly_cutting_minutes.toFixed(1)
        } more cutting minutes/wk`}
        sub={
          labor.exceeds_cutter_capacity
            ? "Exceeds available cutter minutes"
            : `${labor.weekly_cutter_minutes_available.toFixed(0)} minutes available/wk`
        }
      />,
    );
  }

  if (waste && has(waste.scenario_weekly_trim_waste_kg)) {
    tiles.push(
      <Tile
        key="waste"
        label="Trim waste"
        value={`${waste.scenario_weekly_trim_waste_kg.toFixed(2)} kg/wk`}
        delta={
          has(waste.extra_weekly_trim_waste_kg) && has(waste.baseline_weekly_trim_waste_kg)
            ? `${signed(waste.extra_weekly_trim_waste_kg, (v) => `${v.toFixed(2)} kg`)} vs. ${
                waste.baseline_weekly_trim_waste_kg.toFixed(2)} kg today`
            : undefined
        }
      />,
    );
  }

  if (!tiles.length) return null;

  return (
    <section className="card" aria-labelledby="outcome-heading">
      <h3 id="outcome-heading">Projected outcome</h3>
      {hasScenario && outcome.product_sku ? (
        <p className="card-note">
          {outcome.product_sku} at {outcome.demand_multiplier}× current demand,
          from {outcome.source_primal}.
        </p>
      ) : null}
      <div className="tiles">{tiles}</div>
    </section>
  );
}
