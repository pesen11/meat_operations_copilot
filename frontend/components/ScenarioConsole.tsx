"use client";

/**
 * The scenario page: ask a question, watch the agents work, read the outcome,
 * approve or reject.
 *
 * Progress comes over SSE, so the three parallel agents appear as they finish
 * rather than after a silent wait. The approval bar only appears when the
 * backend actually paused for it - informational answers never ask, which is
 * what keeps the prompt meaningful when it does appear.
 */

import { useCallback, useRef, useState } from "react";
import {
  api,
  streamSimulation,
  type SimulateResponse,
  type TraceEvent,
} from "@/lib/api";
import { AnswerDetails } from "@/components/AnswerDetails";
import { DecisionPanel } from "@/components/DecisionPanel";
import { OutcomeCards } from "@/components/OutcomeCards";
import { ScenarioComparison } from "@/components/ScenarioComparison";

const EXAMPLES = [
  "How much production increase should we do during Christmas?",
  "What if we increase chuck roast production by 15%?",
  "What is coming up that we should prepare for?",
  "Should we scale back ribeye steak by 20 percent?",
  "How many boxes of blade eyes do we have on hand?",
  "What were our top sellers last month?",
];

const VERDICT_LABEL: Record<string, string> = {
  proceed: "Proceed",
  proceed_with_caution: "Proceed with caution",
  do_not_proceed: "Do not proceed",
  informational: "For information",
};

const VERDICT_STATUS: Record<string, string> = {
  proceed: "status-good",
  proceed_with_caution: "status-warning",
  do_not_proceed: "status-critical",
  informational: "status-ok",
};

function newThreadId() {
  return `ui-${Math.random().toString(36).slice(2, 10)}`;
}

export function ScenarioConsole() {
  const [question, setQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<TraceEvent[]>([]);
  const [result, setResult] = useState<SimulateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [decision, setDecision] = useState<string | null>(null);
  // Lazily initialised, never during server render: Math.random() in a
  // useRef initialiser runs on every render and diverges between the
  // server and client pass. The value is only read after run() has
  // already replaced it, so an empty initial value is safe.
  const threadRef = useRef<string>("");
  if (!threadRef.current) threadRef.current = newThreadId();
  const cancelRef = useRef<(() => void) | null>(null);

  const run = useCallback((text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 3 || running) return;

    cancelRef.current?.();
    threadRef.current = newThreadId();
    setRunning(true);
    setProgress([]);
    setResult(null);
    setError(null);
    setDecision(null);

    cancelRef.current = streamSimulation(trimmed, threadRef.current, {
      onProgress: (event) => setProgress((events) => [...events, event]),
      onResult: (payload) => setResult(payload),
      onError: (message) => setError(message),
      onDone: () => setRunning(false),
    });
  }, [running]);

  const decide = useCallback(
    async (approved: boolean) => {
      try {
        const response = await api.approve(
          threadRef.current,
          approved,
          "shop-manager",
        );
        setDecision(String(response.approval.status ?? (approved ? "approved" : "rejected")));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not record the decision.");
      }
    },
    [],
  );

  const verdict = result?.recommendation.verdict ?? "informational";
  const guard = result?.number_guard;

  return (
    <>
      <section className="card" aria-labelledby="ask-heading">
        <h2 id="ask-heading">Ask an operations question</h2>
        <p className="card-note">
          Ask what to do for a season, propose a what-if, or check stock and
          sales history. Procedural questions belong on the SOP lookup tab.
        </p>
        <form
          className="ask-row"
          onSubmit={(event) => {
            event.preventDefault();
            run(question);
          }}
        >
          <input
            type="text"
            value={question}
            placeholder="What if we increase chuck roast production by 15%?"
            aria-label="Operations question"
            onChange={(event) => setQuestion(event.target.value)}
          />
          <button type="submit" className="primary" disabled={running || question.trim().length < 3}>
            {running ? <><span className="spinner" /> Running</> : "Run scenario"}
          </button>
        </form>
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              disabled={running}
              onClick={() => {
                setQuestion(example);
                run(example);
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {progress.length > 0 ? (
        <section className="card" aria-labelledby="progress-heading" aria-live="polite">
          <h3 id="progress-heading">Agent progress</h3>
          <ol className="progress">
            {progress.map((event, i) => (
              <li key={`${event.node}-${i}`}>
                <span className="node">{event.node}</span>
                <span>{event.message}</span>
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {error ? (
        <div className="card notice error" role="alert">
          {error}
          <div className="muted" style={{ marginTop: 6 }}>
            Is the backend running? <code>uvicorn api.main:app --reload</code>
          </div>
        </div>
      ) : null}

      {result ? (
        <>
          <section className="card" aria-labelledby="verdict-heading">
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 10 }}>
              <h2 id="verdict-heading" style={{ margin: 0 }}>
                {VERDICT_LABEL[verdict] ?? verdict}
              </h2>
              <span className={`badge ${VERDICT_STATUS[verdict] ?? ""}`}>
                <span className="dot" aria-hidden="true" />
                {VERDICT_LABEL[verdict] ?? verdict}
              </span>
            </div>

            <p className="narration">{result.narration}</p>

            {result.recommendation.blockers.length > 0 ? (
              <ul style={{ marginTop: 12 }}>
                {result.recommendation.blockers.map((blocker) => (
                  <li key={blocker}>{blocker}</li>
                ))}
              </ul>
            ) : null}
            {result.recommendation.cautions.length > 0 ? (
              <ul style={{ marginTop: 8 }}>
                {result.recommendation.cautions.map((caution) => (
                  <li key={caution}>{caution}</li>
                ))}
              </ul>
            ) : null}

            {guard ? (
              <p className="footnote">
                {guard.passed
                  ? `Verified: all ${guard.numbers_checked} figures in this summary trace back to tool output.`
                  : `Warning: ${guard.violations.length} figure(s) in this summary could not be traced to tool output — ${guard.violations
                      .map((violation) => violation.stated)
                      .join(", ")}.`}
                {" "}Summary written by {result.recommendation.narrated_by === "claude" ? "Claude" : "a template (no API key configured)"}.
              </p>
            ) : null}
          </section>

          {result.awaiting_approval && !decision ? (
            <section className="card" aria-labelledby="approval-heading">
              <h3 id="approval-heading">Approval needed</h3>
              <p className="card-note">
                This scenario recommends a real production change. Nothing is
                recorded until you decide.
              </p>
              <div className="ask-row">
                <button type="button" className="approve" onClick={() => decide(true)}>
                  Approve
                </button>
                <button type="button" className="reject" onClick={() => decide(false)}>
                  Reject
                </button>
              </div>
            </section>
          ) : null}

          {decision ? (
            <div className="card notice" role="status">
              Recorded as <strong>{decision}</strong> for thread{" "}
              <code>{result.thread_id}</code>.
            </div>
          ) : null}

          <DecisionPanel recommendation={result.recommendation} />
          <OutcomeCards outcome={result.projected_outcome} />
          <AnswerDetails outcome={result.projected_outcome} />
          <ScenarioComparison outcome={result.projected_outcome} />
        </>
      ) : null}
    </>
  );
}
