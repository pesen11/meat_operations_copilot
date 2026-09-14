# Meat-Cutting Operations Copilot

An operations intelligence system for a butcher shop — not a chatbot. It
simulates production decisions ("if we increase chuck roast production by 15%,
what happens to inventory, labour, and waste?") and returns computed,
structured outcomes.

**[Live demo](https://meat-operations-copilot.vercel.app)** ·
**[API docs](https://ops-copilot-api-j7zz.onrender.com/docs)** ·
**[Health](https://ops-copilot-api-j7zz.onrender.com/health)**

> The hosted demo runs in offline mode — all arithmetic is identical to the
> live path; narration is templated rather than written by Claude. It is on a
> free tier, so the first request after an idle period takes ~30s to wake.

---

## The core design principle

**The LLM never computes numbers.** All yield, labour, waste and margin maths
is deterministic, unit-tested Python in `simulation/`. The language model does
exactly two jobs: routing a question to the right tools, and narrating the
results in plain language.

This is enforced, not merely intended:

- `evals/number_guard.py` extracts every number from the model's narration and
  traces it back to the tool output it claims to describe. Re-formatting
  (`0.6268` to `62.7%`, `789.6` to `$789.60`) is permitted; arithmetic is not.
- The guard runs **in the API response**, not only in CI, so a violation is
  visible to the operator rather than only to a test run.
- `evals/claim_guard.py` catches what the number guard structurally cannot: a
  business-rule figure presented as a calculation, a measurement claim with no
  measurement behind it, a narration contradicting its own verdict.
- The evaluation suite exits non-zero if the invariant ever breaks.

One consequence is worth stating plainly: **the entire test suite runs with no
API key and no database.** If a change makes a test require credentials, that
change has probably moved arithmetic into the LLM layer.

---

## Quick start

Requires Python 3.10+ and Node 20+.

```bash
git clone https://github.com/pesen11/meat_operations_copilot.git
cd meat_operations_copilot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
```

Generated data is committed and the SOP vector index builds itself on first
use, so there is no build step. Run the two services in separate terminals:

```bash
# Terminal 1 — API on :8000
uvicorn api.main:app --reload --port 8000
```

```bash
# Terminal 2 — UI on :3000
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000**. Interactive API documentation is at **/docs**,
and **/health** reports which mode every subsystem is running in.

### Verifying the install

```bash
pytest tests/ -q                                   # 574 tests, no credentials needed
python -m evals.run_all                            # guards + RAG evaluation
python -m rag.index search "how do I clean the grinder"
```

### Regenerating data (optional)

```bash
python data/generate_history.py    # seeded, so reproducible
python -m rag.index build          # force a rebuild of the SOP index
```

### Ports

The frontend calls the API at `http://127.0.0.1:8000`, and the API's CORS
allowlist defaults to `localhost:3000`. To change either, set both sides:

```bash
export OPS_COPILOT_CORS_ORIGINS="http://localhost:3001"   # API terminal
export NEXT_PUBLIC_API_URL="http://127.0.0.1:8001"        # UI terminal
```

A "Failed to fetch" banner against a healthy API is almost always this: the
browser's origin is not in the allowlist. Note that `NEXT_PUBLIC_*` values are
compiled into the bundle at build time, so changing one requires a rebuild.

---

## Configuration

Nothing below is required. Each upgrades a capability in place.

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | Claude handles intent parsing, SOP reranking, generation and narration. Without it, deterministic fallbacks run and every *number* is identical. |
| `DATABASE_URL` | Reads Postgres instead of CSVs, moves the SOP index to pgvector, persists LangGraph checkpoints across restarts, and writes to `decision_log`. Apply `db/schema.sql`, then run `python -m db.load_csv`. |
| `VOYAGE_API_KEY` + `OPS_COPILOT_EMBEDDINGS=voyage` | Dense semantic embeddings instead of TF-IDF. Re-run `rag.index build` and `rag.eval.tune` — thresholds are embedder-specific. |
| `OPS_COPILOT_CORS_ORIGINS` | Comma-separated allowed origins. Defaults to `localhost:3000`. |
| `OPS_COPILOT_LLM=stub` | Forces offline mode even when a key is present. This is what CI uses. |

Copy `.env.example` to `.env` to set these locally.

### Offline versus live

| | Offline (default) | With `ANTHROPIC_API_KEY` |
|---|---|---|
| Narration | Template | Claude |
| SOP reranking | Lexical scorer | Claude Haiku relevance filter |
| SOP answers | Best passage quoted | Composed across passages |
| **All arithmetic** | **Identical** | **Identical** |

`rag/index/tuned_params.json` is calibrated for the *offline* reranker and is
ignored once the LLM reranker is active — a threshold tuned on one scorer's
scale means nothing on another's. Re-tune the live path with
`python -m rag.eval.tune`.

---

## Architecture

```
question
  → manager          parse into a validated request, with provenance
  → supervisor       dispatch ONLY the agents this question needs
  → inventory  ─┐
    production   ├─ run IN PARALLEL, calling deterministic tools
    historical   │
    supply      ─┘
  → yield analyzer   join into one projected outcome (no new maths)
  → evidence check   did the required evidence actually arrive?
      ├─ gaps ──→ gather supplementary ──→ back to the analyzer (once only)
      └─ ok ────→ recommendation
  → recommendation   deterministic decision + LLM narration of it
  → human approval   interrupt() — pauses until an operator decides
```

Three properties hold throughout:

- **Dispatch follows the question, not the diagram.** "What were our best
  sellers last month" runs *one* agent. The other three are never invoked,
  though the progress stream still names them and says why they sat out.
- **The decision is deterministic** (`simulation/decision_engine.py`): six
  scored factors — capacity, stock, supply, margin, waste, execution
  reliability — blended by weight, with any one able to veto. A capacity
  overrun is not made acceptable by a good margin. The model explains a
  decision Python already made, which is what makes the narration checkable.
- **Evidence that never arrived is reported, not assumed.** Coverage below
  100% caps the recommendation's confidence, and an unassessed factor reads
  "not assessed" rather than scoring well.

### Project layout

```
models/         Pydantic domain models and provenance-tagged evidence types
data/           Catalog (real + synthetic), cost table, history generator, SOP corpus
simulation/     DETERMINISTIC MATHS — no LLM anywhere in this package
agents/         LangGraph orchestration; agents/llm.py is the only caller of Claude
rag/            SOP knowledge base: chunking, hybrid retrieval, reranking, guardrails
api/            FastAPI: /simulate, SSE stream, /approve, /ask, dashboard endpoints
db/             Postgres schema, CSV loader, decision audit log
evals/          Number guard, claim guard, narration faithfulness, Ragas, fault suite
frontend/       Next.js: scenarios, dashboard, SOP lookup
tests/          574 tests
```

---

## Decision support, not a point estimate

Asking "how much more for Christmas?" returns a ladder, not a number:

```
LEVEL   EXTRA PRODUCT  EXTRA BOXES  CUTTER USE  CHANCE OF RUNNING OUT  STATUS
+10%          530 kg         21.6         53%                     0%  achievable
+15%          795 kg         32.5         56%                     2%  achievable
+20%  ◀      1060 kg         43.3         58%                    12%  achievable
+25%         1325 kg         54.1         60%                    37%  not achievable
+90%         4770 kg        194.8         92%                   100%  not achievable
```

The shop's confirmed Christmas uplift is **1.9× (+90%)**. The honest answer is
that it is not reachable at current ordering — Blade Eyes runs out — and
**+20% is the ceiling** until more primal is ordered. A single point estimate
("Christmas needs 4,770 kg more") is true and supports no decision.

Stockout probability is measured rather than assumed. Residual volatility is
estimated from the sales history — the estimator recovers the 12% noise the
generator injected, without importing the constant — and risk is walked day by
day across the horizon rather than compared on 14-day totals. A cooler that
empties on day five while its delivery lands on day six has a healthy
fortnightly balance and a stockout on day five.

---

## Closing the loop between plan and reality

Three datasets measure what actually happened, not what was scheduled:

| | Measured |
|---|---|
| Vendor fill rate | **98.9%** of ordered quantity arrives |
| On-time delivery | **92.1%** |
| Shop-wide execution | **93%** of planned cutting completes |
| Worst primal (Wagyu Flat Iron) | **78%** |
| Highest-volume primal (Blade Eyes) | **87%** — it runs its cooler closest to empty |

That spread is why execution is keyed by primal rather than taken shop-wide:
the 78–98% range is *caused*, not noise. Planning 400 kg of Blade Eyes and
booking 400 kg of finished product overstates supply every time.

Purchase orders are **derived from the stock series, not generated beside it**.
Any day the box count rises, something arrived — so deliveries are read out of
the stock history and an order is back-filled for each. The two cannot
disagree, because one is computed from the other.

---

## Evaluated against controlled faults

`python -m evals.scenarios` breaks something specific and checks the diagnosis:

```
[PASS] labour_cut_40pct             40% of cutter hours removed
[PASS] all_deliveries_cancelled     every open purchase order cancelled
[PASS] deliveries_delayed_5_days    every open order pushed back 5 days
[PASS] cooler_at_quarter_stock      on-hand boxes cut to 25%
[PASS] production_execution_halved  actual production halved against plan
[PASS] history_needs_only_history   irrelevant agents are NOT dispatched
```

Six dimensions are scored separately — intent, dispatch, evidence, diagnosis,
decision, grounding — because "right answer, wrong reason" is the interesting
failure and a single average would hide it. The test suite asserts that each
dimension *can* fail: a ground-truth suite that always passes is worse than
none.

---

## The RAG layer

1. **Eval set first.** 29 labelled queries in `rag/eval/dataset.py` — direct,
   paraphrase, multi-hop, adversarial, broad, and five with **no answer in the
   corpus**. Gold chunks are referenced by (document, heading) rather than
   chunk id, so re-chunking cannot silently orphan the labels.
2. **Recall-optimised retrieval.** Hybrid TF-IDF + BM25 + heading match,
   top-k 14 (~28% of the corpus), plus neighbour and sibling expansion. A
   missed chunk is unrecoverable; an irrelevant one is the reranker's problem.
3. **Reranking before generation.** Claude Haiku scores every candidate and
   truncates to four, with a lexical cross-encoder-style scorer as the offline
   fallback. The model never sees raw top-k.
4. **Grounded generation.** Answers come only from retrieved context, with an
   exact refusal marker, a pre-generation gate, and a numeric groundedness
   guardrail that withholds any answer whose figures cannot be traced.
5. **Two-layer cache.** Exact (hashed) plus semantic (embedding similarity),
   with the similarity threshold swept against the eval set, and prompt caching
   on the static system prompts.

Every threshold here is swept by `python -m rag.eval.tune`, not chosen by feel.
`rag/tuning.py` reports whether the values in force are tuned or defaults.

Current **offline** measurements (`python -m rag.eval.run_eval`):

| Metric | Value |
|---|---|
| Hit rate @5 | 0.96 |
| Recall @5 | 0.87 |
| Recall @14 | 0.95 |
| MRR | 0.81 |
| Decision accuracy | 0.93 |
| Correct decline rate | 0.80 |
| Numeric grounded rate | 1.00 |

These are a **floor, not the shipped behaviour** — the harness prints which
mode it ran in, so offline and live numbers are never mistaken for each other.

The decline rate is **0.80, not 1.00**, and that is deliberate reporting. An
earlier eval set scored 1.00 only because it contained no broad, open-ended
questions, so the threshold sweep tuned into that blind spot and the first real
broad question ("what safety protocols should I follow?") was wrongly refused.
A `broad` category was added; the remaining gap is recorded by name in
`KNOWN_OFFLINE_DECLINE_GAPS` rather than averaged away. Telling chicken
*handling* (covered) from chicken *cutting* (out of scope) is a semantic
judgement that no lexical threshold separates — the live path is expected to
score 1.00 there.

---

## Data provenance

This project uses **synthetic data modelled on real industry patterns**,
documented per row. Every `Product`, `YieldProfile` and `DemandProfile` carries
a `data_source` tag.

- **Real** (11 products across 8 primals): transcribed from an actual
  meat-cutting operation's yield, labour and pricing table, adjusted by the
  domain expert who built this project — "similar but not exact" to their
  workplace.
- **Synthetic** (8 products across 6 primals): generated to extend catalog
  coverage, modelled on industry-realistic yield and labour patterns, not
  audited against a real operation.
- The five SOP documents in `data/sops/` are **all synthetic**. That tag
  survives chunking and retrieval and is shown in each answer's source list, so
  an answer drawn from example material is identifiable as such.

No real employer data, financials, or proprietary business records appear
anywhere in this repository.

---

## Key assumptions

All are flagged inline in code as named constants, so they are easy to find and
correct against real data.

**Confirmed with the domain expert**

- Labour rate: a flat **$27/hr** average.
- Cut-product revenue: **35%** of a confirmed **$400k/week** whole-store baseline.
- Blade Eyes cut plan: **65%** thinly sliced, **35%** chuck roast.
- Tenderloin tracked in real **40 kg** delivery boxes (one to two on hand),
  decoupled from the ~4.5 kg piece weight used for labour maths.
- Blended gross margin of **39.0%** across the catalog, net of primal cost and
  cutting labour, with individual cuts spanning 32–49%.

**Estimated — primal cost.** One vendor price per kg per primal, in
`data/primal_costs.py`. The real vendor list is confidential, so each value is
back-solved from the retail value that primal's cut plan realises at a tiered
target margin, then rounded like a price list. Replacing this table with real
invoices is a data edit; nothing downstream changes. Locked by
`tests/test_primal_costs.py`.

**Not separately confirmed**

- 12% daily sales noise, 4% vendor stockout probability, 28-day reporting
  windows.
- The supply and execution layer in `simulation/supply_generator.py` — vendor
  lead times of two to five days, 8% late deliveries, 6% short shipments, a 7%
  production disruption rate, and an 8% over-replacement order buffer. These
  shape every forward projection.
- **Every decision threshold.** The six factor weights and their cut-offs (90%
  cutter comfort, 1.5 days of cover, 15%/50% stockout bands, 85%/95% execution
  bands, 15% waste disproportion) live in `simulation/decision_engine.py`, and
  the sweep's selection policy in `simulation/scenario_sweep.py`. **These decide
  what the system recommends** — a 92% cutter utilisation reading as "proceed
  with caution" rather than "proceed" is one constant. They are the first thing
  to correct against real operating experience.
- **Demand autocorrelation (0.25) is assumed, not measured.** It widens
  multi-day prediction intervals by roughly 29%. It is not measured here
  because the generator draws each day independently, so measuring it would
  return ~0 and encode the generator's simplification as a fact about butchery.
  It is a deliberately conservative correction in the direction real demand
  errs.

---

## Deployment

The hosted demo runs on:

| Layer | Platform | Notes |
|---|---|---|
| Frontend | Vercel | Root directory `frontend`. Set `NEXT_PUBLIC_API_URL` **before** the first build — it is compiled into the bundle. |
| API | Render | `render.yaml` blueprint. A persistent container rather than serverless: `/simulate/stream` is long-lived SSE over a synchronous worker thread. |
| Database | Neon Postgres | Apply `db/schema.sql`, then run `python -m db.load_csv`. |

Set `DATABASE_URL` and `OPS_COPILOT_CORS_ORIGINS` in the Render dashboard. Both
are declared `sync: false` in the blueprint, so values never enter git.

Check `/health` after deploying. It reports `checkpointer`, and
`checkpointer_fallback_reason` when Postgres was not used — that fallback is
otherwise silent and costs the approval interrupt on every restart.

---

## Known limitations

- **No finished-product inventory layer.** The model tracks primal boxes;
  finished packs are treated as a thin, short-cycle buffer.
- **No multi-step remediation search.** The system reports that a plan is
  infeasible and what binds, but does not search for a fix — "could we add
  labour, move production earlier, reallocate primal?"
- **Labour is modelled for the cutter role only**, not per role.
- **Offline out-of-corpus detection is a weak signal.** It leans on vocabulary
  absence (`rag/scope.py`), doing a job the model does properly when
  credentials are present.
- **The decision thresholds are unvalidated** against a real operation. See
  *Key assumptions*.

---

## Licence

MIT
