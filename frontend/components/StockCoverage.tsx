"use client";

/**
 * Days of cover per primal.
 *
 * Magnitude compared across categories -> horizontal bars, sorted ascending so
 * the primals that need attention are at the top where they get read first.
 *
 * Colour here is a STATUS encoding (stockout / critical / below target / ok /
 * over target), not a categorical one, so it uses the fixed status palette and
 * never the series hues. Two of those steps sit below 3:1 contrast on the
 * light surface by design; the mitigation is that every bar carries its status
 * as a text label in the table beneath and in the tooltip, so the colour never
 * has to carry the meaning on its own.
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
import type { StockPosition } from "@/lib/api";

const STATUS_COLOR: Record<StockPosition["status"], string> = {
  stockout: "var(--status-critical)",
  critical: "var(--status-critical)",
  below_target: "var(--status-serious)",
  ok: "var(--status-good)",
  over_target: "var(--status-warning)",
};

export const STATUS_LABEL: Record<StockPosition["status"], string> = {
  stockout: "Out of stock",
  critical: "Critical",
  below_target: "Below target",
  ok: "On target",
  over_target: "Over target",
};

function CoverageTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as StockPosition;
  return (
    <div className="viz-tooltip">
      <div style={{ fontWeight: 650, marginBottom: 4 }}>{row.source_primal}</div>
      <div>
        <span className="k">Cover: </span>
        <span className="v">
          {row.days_of_cover === null ? "n/a" : `${row.days_of_cover.toFixed(2)} days`}
        </span>
      </div>
      <div>
        <span className="k">On hand: </span>
        <span className="v">{row.boxes_on_hand} boxes</span>
        <span className="k"> (target {row.target_boxes_low}–{row.target_boxes_high})</span>
      </div>
      <div>
        <span className="k">Status: </span>
        <span className="v">{STATUS_LABEL[row.status]}</span>
      </div>
    </div>
  );
}

export function StockCoverage({ rows }: { rows: StockPosition[] }) {
  const data = [...rows]
    .filter((row) => row.days_of_cover !== null)
    .sort((a, b) => (a.days_of_cover ?? 0) - (b.days_of_cover ?? 0));

  if (!data.length) return <p className="muted">No stock positions available.</p>;

  return (
    <div style={{ width: "100%", height: Math.max(260, data.length * 26) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 20, bottom: 4, left: 4 }}
        >
          <CartesianGrid stroke="var(--gridline)" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={{ stroke: "var(--baseline)" }}
            tickLine={false}
            label={{
              value: "days of cover",
              position: "insideBottomRight",
              offset: -2,
              fill: "var(--text-muted)",
              fontSize: 11,
            }}
          />
          <YAxis
            type="category"
            dataKey="source_primal"
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={126}
          />
          <Tooltip
            content={<CoverageTooltip />}
            cursor={{ fill: "var(--gridline)", opacity: 0.35 }}
          />
          <Bar dataKey="days_of_cover" radius={[0, 4, 4, 0]} maxBarSize={16}>
            {data.map((row) => (
              <Cell
                key={row.source_primal}
                fill={STATUS_COLOR[row.status]}
                stroke="var(--surface-1)"
                strokeWidth={2}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
