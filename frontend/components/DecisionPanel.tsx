"use client";

/**
 * The decision, shown as the several things it actually weighed.
 *
 * The verdict card above this says WHAT was decided. This says why: six
 * scored factors, which one is binding, and — where the question was "how
 * much" — the ladder of levels the recommendation was chosen from.
 *
 * Two display rules that carry real meaning rather than being styling:
 *
 * 1. An UNASSESSED factor is rendered as "not assessed", never as a score.
 *    The backend deliberately excludes it from the blend, because "we did
 *    not look" and "we looked and it is fine" are different statements.
 *    Drawing it as an empty bar would silently turn the first into the
 *    second.
 *
 * 2. Status is spelled out in words everywhere it appears, never conveyed by
 *    colour alone — the same rule the stock table follows. A red bar and a
 *    green bar are indistinguishable to a red-green colourblind cutter
 *    standing over a tablet.
 *
 * No arithmetic here beyond choosing a unit suffix. Every figure is
 * formatted, never recomputed, so the UI cannot drift from the number guard
 * that checked the narration against these same values.
 */

import type { DecisionFactor, Recommendation, ScenarioOption } from "@/lib/api";

const FACTOR_LABEL: Record<string, string> = {
  capacity: "Cutting capacity",
  inventory: "Stock position",
  supply: "Incoming supply",
  economics: "Margin",
  waste: "Trim waste",
  execution: "Production reliability",
};

// Score bands, matching simulation/decision_engine.py's CAUTION_SCORE and
// SCORE_PROCEED. The words are the signal; the colour only reinforces them.
function band(score: number): { label: string; tone: string } {
  if (score >= 0.75) return { label: "clear", tone: "good" };
  if (score >= 0.5) return { label: "watch", tone: "warning" };
  return { label: "concern", tone: "critical" };
}

const percent = (value: number) => `${(value * 100).toFixed(0)}%`;

const money = (value: number) =>
  value.toLocaleString("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  });

function FactorRow({ factor, limiting }: { factor: DecisionFactor; limiting: boolean }) {
  if (!factor.assessed) {
    return (
      <tr>
        <th scope="row">{FACTOR_LABEL[factor.name] ?? factor.name}</th>
        <td colSpan={2} className="muted">
          Not assessed — no data for this factor
        </td>
      </tr>
    );
  }

  const { label, tone } = band(factor.score);
  return (
    <tr>
      <th scope="row">
        {FACTOR_LABEL[factor.name] ?? factor.name}
        {limiting ? <span className="pill"> limiting</span> : null}
      </th>
      <td style={{ whiteSpace: "nowrap" }}>
        <span className={`badge status-${tone}`}>
          <span className="dot" aria-hidden="true" />
          {factor.is_blocking ? "blocking" : label}
        </span>{" "}
        <span className="muted">{factor.score.toFixed(2)}</span>
      </td>
      <td>{factor.headline}</td>
    </tr>
  );
}

function OptionRow({ option, chosen }: { option: ScenarioOption; chosen: boolean }) {
  return (
    <tr className={chosen ? "chosen" : undefined}>
      <th scope="row">
        {option.label}
        {chosen ? <span className="pill"> recommended</span> : null}
      </th>
      <td>{option.extra_weekly_product_kg.toFixed(0)} kg</td>
      <td>{option.extra_weekly_boxes.toFixed(1)}</td>
      <td>{percent(option.cutter_utilization_pct)}</td>
      <td>
        {option.stockout_probability === null
          ? "—"
          : percent(option.stockout_probability)}
      </td>
      <td>
        {option.delta_weekly_margin === null
          ? "—"
          : money(option.delta_weekly_margin)}
      </td>
      <td>
        {option.feasible ? (
          "achievable"
        ) : (
          <span title={option.infeasible_reasons.join("; ")}>
            not achievable
          </span>
        )}
      </td>
    </tr>
  );
}

export function DecisionPanel({ recommendation }: { recommendation: Recommendation }) {
  const factors = recommendation.factors ?? [];
  const options = recommendation.options ?? [];
  if (factors.length === 0 && options.length === 0) return null;

  return (
    <>
      {factors.length > 0 ? (
        <section className="card" aria-labelledby="factors-heading">
          <h3 id="factors-heading">What this weighed</h3>
          <p className="card-note">
            Six factors, scored by deterministic rules. Any one of them can
            block on its own — a capacity overrun is not made acceptable by a
            good margin.
            {recommendation.decision_score !== null ? (
              <>
                {" "}Overall {recommendation.decision_score.toFixed(2)} out of
                1.00.
              </>
            ) : null}
            {recommendation.evidence_coverage !== null &&
            recommendation.evidence_coverage < 1 ? (
              <>
                {" "}Confidence is capped at{" "}
                {percent(recommendation.confidence)} because only{" "}
                {percent(recommendation.evidence_coverage)} of the required
                evidence was available
                {recommendation.evidence_gaps.length > 0
                  ? ` (missing: ${recommendation.evidence_gaps.join(", ")})`
                  : ""}
                .
              </>
            ) : null}
          </p>

          <div className="table-scroll">
            <table>
              <caption className="card-note" style={{ captionSide: "bottom", textAlign: "left", paddingTop: 8 }}>
                Any factor marked blocking stops the recommendation on its own.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Factor</th>
                  <th scope="col">Assessment</th>
                  <th scope="col">Detail</th>
                </tr>
              </thead>
              <tbody>
                {factors.map((factor) => (
                  <FactorRow
                    key={factor.name}
                    factor={factor}
                    limiting={factor.name === recommendation.limiting_factor}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {options.length > 0 ? (
        <section className="card" aria-labelledby="options-heading">
          <h3 id="options-heading">Levels considered</h3>
          <p className="card-note">
            Each level run through the same deterministic model.
            {recommendation.chosen_option ? (
              <> Recommended: {recommendation.chosen_option}.</>
            ) : null}{" "}
            {recommendation.reasons[0] ?? ""}
          </p>

          <div className="table-scroll">
            <table>
              <caption className="card-note" style={{ captionSide: "bottom", textAlign: "left", paddingTop: 8 }}>
                Hover a &ldquo;not achievable&rdquo; row to see what blocks it.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Level</th>
                  <th scope="col">Extra product</th>
                  <th scope="col">Extra boxes</th>
                  <th scope="col">Cutter use</th>
                  <th scope="col">Chance of running out</th>
                  <th scope="col">Margin change</th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {options.map((option) => (
                  <OptionRow
                    key={option.label}
                    option={option}
                    chosen={option.label === recommendation.chosen_option}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <style jsx>{`
        .pill {
          font-size: 0.75rem;
          font-weight: 500;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        tr.chosen th,
        tr.chosen td {
          font-weight: 600;
          background: color-mix(in srgb, var(--series-1) 8%, transparent);
        }
        .table-scroll {
          overflow-x: auto;
        }
      `}</style>
    </>
  );
}
