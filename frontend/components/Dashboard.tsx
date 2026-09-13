"use client";

import { useEffect, useState } from "react";
import {
  api,
  type Health,
  type StockPosition,
  type TopMover,
  type WeeklyRevenuePoint,
} from "@/lib/api";
import { RevenueChart } from "@/components/RevenueChart";
import { STATUS_LABEL, StockCoverage } from "@/components/StockCoverage";

export function Dashboard() {
  const [stock, setStock] = useState<StockPosition[] | null>(null);
  const [revenue, setRevenue] = useState<WeeklyRevenuePoint[] | null>(null);
  const [movers, setMovers] = useState<TopMover[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      api.stock(),
      api.weeklyRevenue("2025-10-01", "2025-12-31"),
      api.topMovers(6, "revenue"),
      api.health(),
    ])
      .then(([stockRows, revenueRows, moverRows, healthRow]) => {
        if (cancelled) return;
        setStock(stockRows);
        setRevenue(revenueRows);
        setMovers(moverRows);
        setHealth(healthRow);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not reach the backend.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="card notice error" role="alert">
        {error}
        <div className="muted" style={{ marginTop: 6 }}>
          Start the backend with <code>uvicorn api.main:app --reload</code>.
        </div>
      </div>
    );
  }

  if (!stock || !revenue || !movers) {
    return (
      <div className="card">
        <span className="spinner" /> Loading operational data…
      </div>
    );
  }

  const attention = stock.filter((row) =>
    ["stockout", "critical", "below_target"].includes(row.status),
  );

  return (
    <>
      <section className="card">
        <h3>At a glance</h3>
        <div className="tiles">
          <div className="tile">
            <div className="label">Primals tracked</div>
            <div className="value">{stock.length}</div>
            <div className="sub">as of {stock[0]?.as_of}</div>
          </div>
          <div className="tile">
            <div className="label">Needing attention</div>
            <div className="value">{attention.length}</div>
            <div className="sub">
              {attention.length
                ? attention.map((row) => row.source_primal).join(", ")
                : "All primals on target"}
            </div>
          </div>
          <div className="tile">
            <div className="label">Top seller</div>
            <div className="value" style={{ fontSize: 18 }}>
              {movers[0]?.product_name ?? "—"}
            </div>
            <div className="sub">
              {movers[0]
                ? `${movers[0].total_revenue.toLocaleString("en-CA", {
                    style: "currency",
                    currency: "CAD",
                    maximumFractionDigits: 0,
                  })} over the window`
                : ""}
            </div>
          </div>
        </div>
      </section>

      <section className="card" aria-labelledby="revenue-heading">
        <h3 id="revenue-heading">Weekly revenue, Oct–Dec 2025</h3>
        <p className="card-note">
          Whole catalog, bucketed into full 7-day weeks counting back from the
          last day of data.
        </p>
        <RevenueChart data={revenue} />
      </section>

      <section className="card" aria-labelledby="coverage-heading">
        <h3 id="coverage-heading">Primal stock coverage</h3>
        <p className="card-note">
          Days of cutting each primal's on-hand boxes support, at its recent
          average draw. Sorted shortest first.
        </p>
        <StockCoverage rows={stock} />

        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table>
            <caption
              className="muted"
              style={{ captionSide: "bottom", textAlign: "left", paddingTop: 8 }}
            >
              The chart's data in full. Status is stated in words here as well as
              by colour.
            </caption>
            <thead>
              <tr>
                <th scope="col">Primal</th>
                <th scope="col">Tier</th>
                <th scope="col" className="num">Boxes</th>
                <th scope="col" className="num">Target</th>
                <th scope="col" className="num">Cover (days)</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {[...stock]
                .sort((a, b) => (a.days_of_cover ?? 0) - (b.days_of_cover ?? 0))
                .map((row) => (
                  <tr key={row.source_primal}>
                    <th
                      scope="row"
                      style={{
                        fontWeight: 500,
                        textTransform: "none",
                        fontSize: "13.5px",
                        color: "var(--text-primary)",
                        letterSpacing: 0,
                      }}
                    >
                      {row.source_primal}
                    </th>
                    <td>{row.velocity_tier}</td>
                    <td className="num">{row.boxes_on_hand}</td>
                    <td className="num">
                      {row.target_boxes_low}–{row.target_boxes_high}
                    </td>
                    <td className="num">
                      {row.days_of_cover === null ? "—" : row.days_of_cover.toFixed(2)}
                    </td>
                    <td>
                      <span className={`badge status-${row.status}`}>
                        <span className="dot" aria-hidden="true" />
                        {STATUS_LABEL[row.status]}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" aria-labelledby="movers-heading">
        <h3 id="movers-heading">Top sellers by revenue</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Product</th>
                <th scope="col" className="num">Revenue</th>
                <th scope="col" className="num">Kg</th>
                <th scope="col" className="num">Share</th>
              </tr>
            </thead>
            <tbody>
              {movers.map((mover) => (
                <tr key={mover.sku}>
                  <th
                    scope="row"
                    style={{
                      fontWeight: 500,
                      textTransform: "none",
                      fontSize: "13.5px",
                      color: "var(--text-primary)",
                      letterSpacing: 0,
                    }}
                  >
                    {mover.product_name}
                  </th>
                  <td className="num">
                    {mover.total_revenue.toLocaleString("en-CA", {
                      style: "currency",
                      currency: "CAD",
                      maximumFractionDigits: 0,
                    })}
                  </td>
                  <td className="num">{mover.total_kg.toFixed(1)}</td>
                  <td className="num">
                    {(mover.share_of_revenue_pct * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {health ? (
        <p className="footnote">
          Backend: {health.data_backend} data · {health.catalog_products} products ·
          SOP index {String(health.vector_store.chunks ?? "?")} chunks ·
          language model {health.llm}.
        </p>
      ) : null}
    </>
  );
}
