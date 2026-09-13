"use client";

/**
 * Baseline vs. scenario, four measures in one frame.
 *
 * The measures have wildly different units and magnitudes (dollars, kg,
 * minutes, a ratio), so they are INDEXED to baseline = 100 rather than plotted
 * on two axes. A dual-axis chart is the single most common chart mistake and
 * it is not available here; indexing to a common base is the documented fix,
 * and it is also the more useful reading - "waste grows faster than margin" is
 * the question this chart answers.
 *
 * Absolute values live in the stat tiles above and in the table below, so the
 * indexed view never hides the real numbers.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ProjectedOutcome } from "@/lib/api";

const has = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

interface Row {
  measure: string;
  baselineIndex: number;
  scenarioIndex: number;
  baselineLabel: string;
  scenarioLabel: string;
}

function buildRows(outcome: ProjectedOutcome): Row[] {
  const rows: Row[] = [];
  const index = (baseline: number, scenario: number) =>
    baseline === 0 ? 100 : (scenario / baseline) * 100;

  const { margin, inventory, labor, waste } = outcome;

  if (has(margin?.baseline_weekly_margin) && has(margin?.scenario_weekly_margin)) {
    rows.push({
      measure: "Margin",
      baselineIndex: 100,
      scenarioIndex: index(margin.baseline_weekly_margin, margin.scenario_weekly_margin),
      baselineLabel: `$${margin.baseline_weekly_margin.toFixed(2)}/wk`,
      scenarioLabel: `$${margin.scenario_weekly_margin.toFixed(2)}/wk`,
    });
  }
  if (has(waste?.baseline_weekly_trim_waste_kg)
      && has(waste?.scenario_weekly_trim_waste_kg)) {
    rows.push({
      measure: "Trim waste",
      baselineIndex: 100,
      scenarioIndex: index(
        waste.baseline_weekly_trim_waste_kg,
        waste.scenario_weekly_trim_waste_kg,
      ),
      baselineLabel: `${waste.baseline_weekly_trim_waste_kg.toFixed(2)} kg/wk`,
      scenarioLabel: `${waste.scenario_weekly_trim_waste_kg.toFixed(2)} kg/wk`,
    });
  }
  if (has(labor?.baseline_cutter_utilization_pct)
      && has(labor?.scenario_cutter_utilization_pct)) {
    rows.push({
      measure: "Cutter load",
      baselineIndex: 100,
      scenarioIndex: index(
        labor.baseline_cutter_utilization_pct,
        labor.scenario_cutter_utilization_pct,
      ),
      baselineLabel: `${(labor.baseline_cutter_utilization_pct * 100).toFixed(1)}%`,
      scenarioLabel: `${(labor.scenario_cutter_utilization_pct * 100).toFixed(1)}%`,
    });
  }
  if (
    inventory?.baseline_days_of_cover != null &&
    inventory.scenario_days_of_cover != null
  ) {
    rows.push({
      measure: "Days of cover",
      baselineIndex: 100,
      scenarioIndex: index(
        inventory.baseline_days_of_cover,
        inventory.scenario_days_of_cover,
      ),
      baselineLabel: `${inventory.baseline_days_of_cover.toFixed(2)} days`,
      scenarioLabel: `${inventory.scenario_days_of_cover.toFixed(2)} days`,
    });
  }
  return rows;
}

function ComparisonTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as Row;
  return (
    <div className="viz-tooltip">
      <div style={{ fontWeight: 650, marginBottom: 4 }}>{row.measure}</div>
      <div>
        <span className="k">Today: </span>
        <span className="v">{row.baselineLabel}</span>
      </div>
      <div>
        <span className="k">Scenario: </span>
        <span className="v">{row.scenarioLabel}</span>
      </div>
      <div style={{ marginTop: 4 }}>
        <span className="k">Indexed: </span>
        <span className="v">{row.scenarioIndex.toFixed(1)}</span>
        <span className="k"> (today = 100)</span>
      </div>
    </div>
  );
}

export function ScenarioComparison({ outcome }: { outcome: ProjectedOutcome }) {
  const rows = buildRows(outcome);
  if (rows.length < 2) return null;

  return (
    <section className="card" aria-labelledby="comparison-heading">
      <h3 id="comparison-heading">Today vs. scenario</h3>
      <p className="card-note">
        Indexed to today = 100, because these four measures are in different
        units. Hover a bar for the real figures; the table below lists them all.
      </p>

      {/* Legend rendered as plain HTML rather than Recharts' <Legend>:
          v3 dropped the `payload` prop, and its internal ordering puts the
          scenario series first, which reverses the left-to-right reading of
          the bars. Owning the markup also keeps the labels in text tokens
          with the colour carried by the swatch beside them. */}
      <ul className="viz-legend">
        <li>
          <span className="swatch" style={{ background: "var(--series-1)" }} aria-hidden="true" />
          Today
        </li>
        <li>
          <span className="swatch" style={{ background: "var(--series-2)" }} aria-hidden="true" />
          Scenario
        </li>
      </ul>

      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={rows}
            margin={{ top: 8, right: 12, bottom: 4, left: 4 }}
            barGap={2}
          >
            <CartesianGrid
              stroke="var(--gridline)"
              strokeDasharray="0"
              vertical={false}
            />
            <XAxis
              dataKey="measure"
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              axisLine={{ stroke: "var(--baseline)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              width={44}
            />
            <Tooltip
              content={<ComparisonTooltip />}
              cursor={{ fill: "var(--gridline)", opacity: 0.35 }}
            />
            <Bar
              dataKey="baselineIndex"
              name="Today"
              fill="var(--series-1)"
              radius={[4, 4, 0, 0]}
              maxBarSize={38}
            >
              {rows.map((row) => (
                <Cell key={`b-${row.measure}`} stroke="var(--surface-1)" strokeWidth={2} />
              ))}
            </Bar>
            <Bar
              dataKey="scenarioIndex"
              name="Scenario"
              fill="var(--series-2)"
              radius={[4, 4, 0, 0]}
              maxBarSize={38}
            >
              {rows.map((row) => (
                <Cell key={`s-${row.measure}`} stroke="var(--surface-1)" strokeWidth={2} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="table-wrap" style={{ marginTop: 12 }}>
        <table>
          <caption className="muted" style={{ captionSide: "bottom", textAlign: "left", paddingTop: 8 }}>
            The same figures as the chart, unindexed.
          </caption>
          <thead>
            <tr>
              <th scope="col">Measure</th>
              <th scope="col" className="num">Today</th>
              <th scope="col" className="num">Scenario</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.measure}>
                <th scope="row" style={{ fontWeight: 500, textTransform: "none", fontSize: "13.5px", color: "var(--text-primary)", letterSpacing: 0 }}>
                  {row.measure}
                </th>
                <td className="num">{row.baselineLabel}</td>
                <td className="num">{row.scenarioLabel}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
