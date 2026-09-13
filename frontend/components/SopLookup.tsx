"use client";

/**
 * SOP lookup - the RAG surface.
 *
 * A declined answer is rendered as a normal, calm result, not an error state.
 * "The SOPs don't cover this" is the correct response to an out-of-corpus
 * question, and styling it like a failure would push people to rephrase until
 * they get a confident-sounding answer instead - which is exactly the
 * behaviour the no-answer eval cases exist to prevent.
 */

import { useState } from "react";
import { api, type AskResponse } from "@/lib/api";

const EXAMPLES = [
  "What are some safety protocols that I should follow?",
  "How do I clean the grinder after use?",
  "How tight should I trim the fat cap on a pulled pork roast?",
  "What chlorine concentration should the sanitizer be at?",
  "What do I do when the vendor is out of a primal?",
  "What temperature should I sous vide a striploin to?",
];

export function SopLookup() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ask = async (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 3 || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.ask(trimmed));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <section className="card" aria-labelledby="sop-heading">
        <h2 id="sop-heading">Look up a procedure</h2>
        <p className="card-note">
          Answers come only from the shop's own SOP documents, with the source
          section cited. If the procedures on file don't cover something, this
          says so rather than guessing.
        </p>
        <form
          className="ask-row"
          onSubmit={(event) => {
            event.preventDefault();
            ask(query);
          }}
        >
          <input
            type="text"
            value={query}
            placeholder="How do I clean the grinder after use?"
            aria-label="Procedure question"
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="submit" className="primary" disabled={loading || query.trim().length < 3}>
            {loading ? <><span className="spinner" /> Searching</> : "Ask"}
          </button>
        </form>
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              disabled={loading}
              onClick={() => {
                setQuery(example);
                ask(example);
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {error ? (
        <div className="card notice error" role="alert">
          {error}
        </div>
      ) : null}

      {result ? (
        <section className="card" aria-live="polite">
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 10 }}>
            <span className={`badge ${result.answered ? "status-ok" : "status-warning"}`}>
              <span className="dot" aria-hidden="true" />
              {result.answered ? "Answered from the SOPs" : "Not covered by the SOPs"}
            </span>
            {result.cache !== "miss" ? (
              <span className="badge">cached ({result.cache})</span>
            ) : null}
            <span className="muted">{result.latency_ms.toFixed(0)} ms</span>
          </div>

          <p className="narration">{result.answer}</p>

          {result.sources.length > 0 ? (
            <>
              <h3 style={{ marginTop: 16 }}>
                {result.answered ? "Sources" : "Closest sections"}
              </h3>
              <ul className="sources">
                {result.sources.map((source) => (
                  <li key={source.chunk_id} className={source.cited ? "cited" : ""}>
                    <strong>{source.heading}</strong>
                    <br />
                    {source.document} · relevance {source.relevance.toFixed(2)}
                    {source.data_source !== "real" ? " · synthetic example SOP" : ""}
                    {source.cited ? " · cited" : ""}
                  </li>
                ))}
              </ul>
            </>
          ) : null}

          {result.groundedness ? (
            <p className="footnote">
              Groundedness check:{" "}
              {result.groundedness.numeric_ok
                ? "every figure in this answer appears in the source text."
                : "a figure could not be matched to the source text, so the answer was withheld."}
            </p>
          ) : null}
        </section>
      ) : null}
    </>
  );
}
