"use client";

/**
 * Tables for the non-scenario answers: history, rankings, stock positions
 * and the last-season comparison.
 *
 * OutcomeCards renders before/after tiles, which only exist for a scenario.
 * A history or inventory question produced no tiles at all, so the page
 * showed a paragraph of narration under an empty heading and looked broken
 * even once the backend was answering correctly. These sections render only
 * when their block is present, so a scenario answer is unchanged.
 *
 * Same rule as everywhere else on the frontend: numbers are FORMATTED here,
 * never recomputed. Every value below is printed as the API sent it.
 */

import type { ProjectedOutcome } from "@/lib/api";

const money = (value: number) =>
  value.toLocaleString("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  });

const kg = (value: number) => `${value.toLocaleString("en-CA")} kg`;

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

interface MoverRow {
  sku: string;
  product_name: string;
  total_kg: number;
  total_revenue: number;
  share_of_revenue_pct: number;
}

interface PerformanceRow extends MoverRow {
  change_pct: number | null;
  direction: string;
  verdict: string;
  seasonality_explains_change: boolean;
}

interface StockRow {
  source_primal: string;
  boxes_on_hand: number;
  kg_on_hand: number;
  days_of_cover: number | null;
  status: string;
}

// The status palette is reserved for status. Direction is not status, so it
// gets words, not a colour - "down" on a product the calendar explains is
// not a warning, and colouring it like one would be a lie told in CSS.
const VERDICT_WORDS: Record<string, string> = {
  focus: "Growing — push it",
  reduce: "Falling — order less",
  watch: "Watch",
  steady: "Steady",
};

function MoverTable({ rows, caption }: { rows: MoverRow[]; caption: string }) {
  return (
    <div className="table-wrap" style={{ marginTop: 12 }}>
      <table>
        <caption className="card-note">{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Product</th>
            <th scope="col" className="num">Sold</th>
            <th scope="col" className="num">Revenue</th>
            <th scope="col" className="num">Share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.sku}>
              <th scope="row">{row.product_name}</th>
              <td className="num">{kg(row.total_kg)}</td>
              <td className="num">{money(row.total_revenue)}</td>
              <td className="num">{pct(row.share_of_revenue_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AnswerDetails({ outcome }: { outcome: ProjectedOutcome }) {
  const window = outcome.window as
    | { label?: string; start?: string; end?: string; explicit?: boolean }
    | undefined;
  const summary = outcome.sales_summary as
    | {
        product_name: string;
        total_kg: number;
        total_revenue: number;
        open_days: number;
        avg_kg_per_open_day: number;
        avg_kg_weekday: number;
        avg_kg_weekend: number;
        best_day: string | null;
        best_day_kg: number;
      }
    | undefined;
  // The shop's own total. Rankings say which cuts led; only this says how
  // much was sold, which is what a question naming no product asks.
  const catalogTotals = outcome.catalog_sales_totals as
    | {
        total_kg: number;
        total_revenue: number;
        open_days: number;
        products_sold: number;
        avg_kg_per_open_day: number;
        avg_revenue_per_open_day: number;
        avg_kg_weekday: number;
        avg_kg_weekend: number;
        best_day: string | null;
        best_day_kg: number;
        best_day_revenue: number;
      }
    | undefined;
  const topMovers = outcome.top_movers as MoverRow[] | undefined;
  const slowMovers = outcome.slow_movers as MoverRow[] | undefined;
  const performance = outcome.catalog_performance as PerformanceRow[] | undefined;
  const restock = outcome.restock_priority as StockRow[] | undefined;
  const focusPrimal = outcome.focus_primal as
    | (StockRow & {
        as_of: string;
        target_boxes_low: number;
        target_boxes_high: number;
        avg_daily_primal_kg: number;
        velocity_tier: string;
      })
    | undefined;
  const uplift = outcome.weekend_uplift as
    | {
        product_name: string | null;
        weekday_avg_kg: number;
        weekend_avg_kg: number;
        implied_multiplier: number | null;
        start: string;
        end: string;
      }
    | undefined;
  const actuals = outcome.last_period_actuals as
    | {
        period: string;
        window_start: string;
        window_end: string;
        total_kg: number;
        total_revenue: number;
        avg_kg_per_day: number;
        normal_avg_kg_per_day: number;
        observed_multiplier: number | null;
        confirmed_multiplier: number;
        top_products: { sku: string; product_name: string; total_kg: number }[];
      }
    | undefined;

  const actionable = (performance ?? []).filter(
    (row) => row.verdict === "focus" || row.verdict === "reduce",
  );

  const hasAnything =
    catalogTotals ||
    summary || topMovers || slowMovers || restock || focusPrimal || uplift || actuals;
  if (!hasAnything) return null;

  return (
    <>
      {catalogTotals && catalogTotals.open_days > 0 ? (
        <section className="card" aria-labelledby="totals-heading">
          <h3 id="totals-heading">Total sales</h3>
          <p className="card-note">
            Whole catalog, {catalogTotals.products_sold} products.{" "}
            {window?.label ?? "recent sales"}
            {window?.start ? ` (${window.start} to ${window.end})` : null}
            {window && window.explicit === false ? " — no period given, default used" : null}
          </p>
          <div className="tiles">
            <div className="tile">
              <div className="label">Revenue</div>
              <div className="value">{money(catalogTotals.total_revenue)}</div>
              <div className="sub">
                over {catalogTotals.open_days} trading
                {catalogTotals.open_days === 1 ? " day" : " days"}
              </div>
            </div>
            <div className="tile">
              <div className="label">Sold</div>
              <div className="value">{kg(catalogTotals.total_kg)}</div>
            </div>
            {catalogTotals.open_days > 1 ? (
              <div className="tile">
                <div className="label">Average day</div>
                <div className="value">{money(catalogTotals.avg_revenue_per_open_day)}</div>
                <div className="sub">
                  {kg(catalogTotals.avg_kg_per_open_day)} — weekday{" "}
                  {catalogTotals.avg_kg_weekday} / weekend {catalogTotals.avg_kg_weekend}
                </div>
              </div>
            ) : null}
            {catalogTotals.open_days > 1 && catalogTotals.best_day ? (
              <div className="tile">
                <div className="label">Best day</div>
                <div className="value">{money(catalogTotals.best_day_revenue)}</div>
                <div className="sub">
                  {catalogTotals.best_day} — {kg(catalogTotals.best_day_kg)}
                </div>
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      {summary ? (
        <section className="card" aria-labelledby="summary-heading">
          <h3 id="summary-heading">{summary.product_name}</h3>
          {/* The window is stated, always. An answer that does not say
              which period it covers cannot be checked by the reader. */}
          <p className="card-note">
            {window?.label ?? "recent sales"}
            {window?.start ? ` (${window.start} to ${window.end})` : null}
            {window && window.explicit === false ? " — no period given, default used" : null}
          </p>
          <div className="tiles">
            <div className="tile">
              <div className="label">Sold</div>
              <div className="value">{kg(summary.total_kg)}</div>
              <div className="sub">over {summary.open_days} trading days</div>
            </div>
            <div className="tile">
              <div className="label">Revenue</div>
              <div className="value">{money(summary.total_revenue)}</div>
            </div>
            <div className="tile">
              <div className="label">Average day</div>
              <div className="value">{kg(summary.avg_kg_per_open_day)}</div>
              <div className="sub">
                weekday {summary.avg_kg_weekday} / weekend {summary.avg_kg_weekend}
              </div>
            </div>
            {summary.best_day ? (
              <div className="tile">
                <div className="label">Best day</div>
                <div className="value">{kg(summary.best_day_kg)}</div>
                <div className="sub">{summary.best_day}</div>
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      {uplift && uplift.implied_multiplier ? (
        <section className="card" aria-labelledby="weekend-heading">
          <h3 id="weekend-heading">Weekend vs. weekday</h3>
          <p className="card-note">
            Measured from {uplift.start} to {uplift.end}
            {uplift.product_name ? ` for ${uplift.product_name}` : " across the catalog"}.
          </p>
          <div className="tiles">
            <div className="tile">
              <div className="label">Average weekday</div>
              <div className="value">{kg(uplift.weekday_avg_kg)}</div>
            </div>
            <div className="tile">
              <div className="label">Average weekend day</div>
              <div className="value">{kg(uplift.weekend_avg_kg)}</div>
            </div>
            <div className="tile">
              <div className="label">Weekend uplift</div>
              <div className="value">{uplift.implied_multiplier}×</div>
              <div className="sub">observed, not a target</div>
            </div>
          </div>
        </section>
      ) : null}

      {topMovers || slowMovers ? (
        <section className="card" aria-labelledby="movers-heading">
          <h3 id="movers-heading">Rankings</h3>
          {window?.label ? <p className="card-note">Over {window.label}.</p> : null}
          {topMovers ? <MoverTable rows={topMovers} caption="Best sellers by revenue" /> : null}
          {slowMovers ? <MoverTable rows={slowMovers} caption="Slowest by revenue" /> : null}
        </section>
      ) : null}

      {actionable.length ? (
        <section className="card" aria-labelledby="actions-heading">
          <h3 id="actions-heading">What to change</h3>
          <p className="card-note">
            Movement the season does not explain, on products big enough to matter.
          </p>
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <thead>
                <tr>
                  <th scope="col">Product</th>
                  <th scope="col" className="num">Change</th>
                  <th scope="col" className="num">Share</th>
                  <th scope="col">Call</th>
                </tr>
              </thead>
              <tbody>
                {actionable.map((row) => (
                  <tr key={row.sku}>
                    <th scope="row">{row.product_name}</th>
                    <td className="num">
                      {row.change_pct === null
                        ? "—"
                        : `${row.change_pct > 0 ? "+" : "−"}${pct(Math.abs(row.change_pct))}`}
                    </td>
                    <td className="num">{pct(row.share_of_revenue_pct)}</td>
                    <td>{VERDICT_WORDS[row.verdict] ?? row.verdict}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {focusPrimal ? (
        <section className="card" aria-labelledby="focus-primal-heading">
          <h3 id="focus-primal-heading">{focusPrimal.source_primal} on hand</h3>
          <p className="card-note">As of {focusPrimal.as_of}.</p>
          <div className="tiles">
            <div className="tile">
              <div className="label">Boxes</div>
              <div className="value">{focusPrimal.boxes_on_hand}</div>
              <div className="sub">
                target {focusPrimal.target_boxes_low} to {focusPrimal.target_boxes_high}
              </div>
            </div>
            <div className="tile">
              <div className="label">Weight</div>
              <div className="value">{kg(focusPrimal.kg_on_hand)}</div>
            </div>
            <div className="tile">
              <div className="label">Days of cover</div>
              <div className="value">{focusPrimal.days_of_cover ?? "—"}</div>
              <div className="sub">at {focusPrimal.avg_daily_primal_kg} kg a day</div>
            </div>
            <div className="tile">
              <div className="label">Status</div>
              {/* Status spelled out, never colour alone. */}
              <div className="value">
                <span className={`badge status-${focusPrimal.status}`}>
                  <span className="dot" aria-hidden="true" />
                  {focusPrimal.status.replace(/_/g, " ")}
                </span>
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {restock && !focusPrimal ? (
        <section className="card" aria-labelledby="restock-heading">
          <h3 id="restock-heading">Restock soonest</h3>
          <p className="card-note">Ordered by days of cover, lowest first.</p>
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <thead>
                <tr>
                  <th scope="col">Primal</th>
                  <th scope="col" className="num">Boxes</th>
                  <th scope="col" className="num">On hand</th>
                  <th scope="col" className="num">Cover (days)</th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {restock.map((row) => (
                  <tr key={row.source_primal}>
                    <th scope="row">{row.source_primal}</th>
                    <td className="num">{row.boxes_on_hand}</td>
                    <td className="num">{kg(row.kg_on_hand)}</td>
                    <td className="num">{row.days_of_cover ?? "—"}</td>
                    <td>
                      <span className={`badge status-${row.status}`}>
                        <span className="dot" aria-hidden="true" />
                        {row.status.replace(/_/g, " ")}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {actuals ? (
        <section className="card" aria-labelledby="actuals-heading">
          <h3 id="actuals-heading">
            Last {actuals.period.replace(/_/g, " ")}, for comparison
          </h3>
          <p className="card-note">
            {actuals.window_start} to {actuals.window_end}. The observed figure is
            history; the confirmed multiplier is a business input. Both are shown
            rather than blended.
          </p>
          <div className="tiles">
            <div className="tile">
              <div className="label">Sold then</div>
              <div className="value">{kg(actuals.total_kg)}</div>
              <div className="sub">{money(actuals.total_revenue)}</div>
            </div>
            <div className="tile">
              <div className="label">Per day</div>
              <div className="value">{kg(actuals.avg_kg_per_day)}</div>
              <div className="sub">vs {actuals.normal_avg_kg_per_day} kg normally</div>
            </div>
            <div className="tile">
              <div className="label">Observed uplift</div>
              <div className="value">
                {actuals.observed_multiplier ? `${actuals.observed_multiplier}×` : "—"}
              </div>
              <div className="sub">confirmed {actuals.confirmed_multiplier}×</div>
            </div>
          </div>
          {actuals.top_products.length ? (
            <p className="card-note" style={{ marginTop: 12 }}>
              What moved then:{" "}
              {actuals.top_products
                .map((row) => `${row.product_name} ${row.total_kg} kg`)
                .join(", ")}
              .
            </p>
          ) : null}
        </section>
      ) : null}
    </>
  );
}
