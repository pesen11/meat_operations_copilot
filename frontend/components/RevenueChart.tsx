"use client";

/**
 * Whole-catalog revenue by week.
 *
 * One series, so no legend box - the heading names it. Change over time with
 * ~13 dense points is a line, not bars. A crosshair tooltip is the default
 * interaction for a line chart and is shipped here rather than treated as an
 * enhancement.
 */

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WeeklyRevenuePoint } from "@/lib/api";

const shortDate = (iso: string) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-CA", {
    month: "short",
    day: "numeric",
  });

const compactMoney = (value: number) =>
  `$${(value / 1000).toFixed(0)}k`;

function RevenueTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload as WeeklyRevenuePoint;
  return (
    <div className="viz-tooltip">
      <div className="k">Week ending {shortDate(point.week_ending)}</div>
      <div className="v">
        {point.revenue.toLocaleString("en-CA", {
          style: "currency",
          currency: "CAD",
          maximumFractionDigits: 0,
        })}
      </div>
    </div>
  );
}

export function RevenueChart({ data }: { data: WeeklyRevenuePoint[] }) {
  if (!data.length) {
    return <p className="muted">No revenue history in this window.</p>;
  }
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="var(--gridline)" vertical={false} />
          <XAxis
            dataKey="week_ending"
            tickFormatter={shortDate}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={{ stroke: "var(--baseline)" }}
            tickLine={false}
            minTickGap={24}
          />
          <YAxis
            tickFormatter={compactMoney}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={52}
          />
          <Tooltip
            content={<RevenueTooltip />}
            cursor={{ stroke: "var(--baseline)", strokeWidth: 1 }}
          />
          <Line
            type="monotone"
            dataKey="revenue"
            name="Weekly revenue"
            stroke="var(--series-1)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4.5, strokeWidth: 2, stroke: "var(--surface-1)" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
