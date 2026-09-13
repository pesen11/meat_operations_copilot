# Meat-Cutting / Processing Operations Copilot

An operations intelligence system for a butcher shop — not a chatbot. It
simulates decisions ("if we increase chuck roast production by 15%, what
happens to inventory, labor, and waste?") and returns computed, structured
outcomes, not hallucinated ones.

## Core design principle

**The LLM never computes numbers.** All yield/labor/waste/margin math is
deterministic, unit-tested Python in `simulation/`. The language model does
exactly two jobs: routing a question to the right tools, and narrating the
results in plain language.

This is enforced, not just intended:

- `evals/number_guard.py` extracts every number from the model's narration
  and checks it against the tool output it claims to describe. Re-formatting
  (`0.6268` → `62.7%`, `789.6` → `$789.60`) is allowed; arithmetic is not.
- The guard runs **in the API response** as well as in CI, so a violation in
  production is visible to the operator, not just to a test run.
- The evaluation suite exits non-zero if the invariant ever breaks.

## Data provenance — real vs. synthetic

This project uses **synthetic data modeled on real industry patterns**,
documented per row:

- **Real** (`data_source="real"`): transcribed from an actual meat-cutting
  operation's yield/labor/pricing table, adjusted by the domain expert
  building this project ("similar but not exact" to their real workplace).
  Covers 11 products across 8 primals/sub-primals.
- **Synthetic** (`data_source="synthetic"`): generated to extend catalog
  coverage, modeled on general industry-realistic yield % and labor patterns,
  not audited against a real operation. Covers 8 additional products across
  6 primals.
- The five SOP documents in `data/sops/` are **all synthetic**, tagged
  `data_source: synthetic` in their frontmatter. That tag survives chunking
  and retrieval and is shown in the answer's source list, so an answer drawn
  from example material is identifiable as such.

No real employer data, financials, or proprietary business records are used
anywhere in this project.

## Running it locally

Verified on this machine: Python 3.10.7, Node 22.12, npm 10.9, Windows.

### One-time setup

```powershell
cd D:\claudeProjects\meat_operations_copilot\ops-copilot

.\mt_venv\Scripts\Activate.ps1        # PowerShell
#  or: source mt_venv/Scripts/activate  (Git Bash)

pip install -e .                      # deps + makes the project importable

python data/generate_history.py       # writes data/generated/*.csv  (seeded)
python -m rag.index build             # chunks + embeds the 5 SOP documents

cd frontend
npm install                           # already done here, safe to repeat
cd ..
```

If `Activate.ps1` is blocked by execution policy, either use
`.\mt_venv\Scripts\activate.bat` (cmd) or run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

### Every time — two terminals

**Terminal 1 — backend (port 8000):**

```powershell
cd D:\claudeProjects\meat_operations_copilot\ops-copilot
.\mt_venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — frontend (port 3000):**

```powershell
cd D:\claudeProjects\meat_operations_copilot\ops-copilot\frontend
npm run dev
```

Then open **http://localhost:3000**.

- `npm run dev` picks up edits live. For the production build instead:
  `npm run build` then `npm run start`.
- The API's interactive docs are at **http://127.0.0.1:8000/docs**, and
  **/health** reports which mode everything is in.

### Ports matter

The frontend calls the backend at `http://127.0.0.1:8000` and the backend's
CORS allowlist defaults to `localhost:3000` only. Run them on those ports and
it just works. To use different ones, set both sides:

```powershell
$env:OPS_COPILOT_CORS_ORIGINS = "http://localhost:3001"   # backend terminal
$env:NEXT_PUBLIC_API_URL      = "http://127.0.0.1:8001"   # frontend terminal
```

A "Failed to fetch" banner in the UI with a healthy backend is almost always
this: the origin the browser used is not in the backend's allowlist.

### If a port is already in use

`next start` exits with `EADDRINUSE` and, confusingly, the OLD server keeps
answering on that port - so you can get a 200 from a stale build. Find and
stop the holder:

```powershell
Get-NetTCPConnection -LocalPort 3000 -State Listen | Select-Object OwningProcess
Stop-Process -Id <that id> -Force
```

### Without an API key (the default)

Everything runs. All arithmetic is identical; narration is templated and SOP
answers quote their best-matching section rather than composing across
sections. The UI says so explicitly: *"Summary written by a template (no API
key configured)"*.

### With an API key

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # then restart the backend
```

Claude then handles intent parsing, SOP reranking, answer generation, and
narration. Note that `rag/index/tuned_params.json` is calibrated for the
OFFLINE reranker, so it is ignored once the LLM reranker is active (by
design - a threshold tuned on one scorer's scale is meaningless on
another's). Re-tune for the live path with:

```powershell
python -m rag.eval.tune
```

### Checking it works

```powershell
pytest tests/ -q             # 266 tests, no key or database needed
python -m evals.run_all      # narration guard + RAG evaluation
python -m rag.index search "how do I clean the grinder"
```

## What's built

```
models/domain.py          Pydantic domain models

data/
  seed_yield_data.py      Real data, transcribed from the owner's table
  synthetic_yield_data.py Synthetic data extending catalog coverage
  catalog.py              Merges real + synthetic, validates consistency
  primal_costs.py         Estimated vendor $/kg per primal (calibrated)
  generate_history.py     Full-year synthetic history generator
  generated/              Output CSVs (sales, stock, labor, production,
                          purchase orders, actual production, plus two
                          derived reference exports of the catalog)
  sops/                   5 synthetic SOP documents (the RAG corpus)

simulation/               DETERMINISTIC MATH — no LLM anywhere in here
  yield_calc.py           Box/piece -> packs, waste, per-pack labor
  cost_calc.py            Margin math on vendor primal cost + labor
  seasonality.py          Ontario stat holidays, seasonal multipliers
  demand_forecast.py      DemandProfile -> expected kg for a given day
  velocity_tiers.py       Primals -> low/medium/best seller tiers
  history_generator.py    A year of sales/stock/labor/schedule
  ops_data.py             Read layer over the history (CSV or Postgres)
  inventory_calc.py       Stock cover, cutter capacity, scenario impact
  history_analysis.py     Sales summaries, trends, movers, per-product verdicts
  time_windows.py         "last week" / "yesterday" -> concrete date ranges
  supply_generator.py     Purchase orders + actual production (causally linked
                          to the stock series, not generated beside it)
  execution_rates.py      Planned vs actual cutting — turns a plan into a forecast
  supply_calc.py          Forward inventory ledger: will we run out, and when
  uncertainty.py          Prediction intervals and P(stockout), measured not assumed
  bottlenecks.py          What binds on a whole-catalog plan, ranked
  scenario_sweep.py       A ladder of production levels + a stated selection policy
  decision_engine.py      Six scored factors, any one able to veto
  seasonal_planning.py    "How much more for Christmas?" - confirmed uplifts
                           applied to a de-seasonalised baseline
  units.py                Sold-units <-> kg normalisation
  tools.py                Yield/economics tools (LangChain @tool)
  inventory_tools.py      Inventory + scenario tools
  history_tools.py        Sales-history tools
  seasonal_tools.py       Seasonal plan + calendar tools

agents/                   LangGraph orchestration
  llm.py                  The one place this project calls Claude
  intent.py               Question -> validated plan
  nodes.py                Manager, supervisor, 3 agents, analyzer, approval
  graph.py                Graph assembly, checkpointing, resume
  state.py                Shared state + parallel-branch reducers

rag/                      SOP knowledge base
  loader.py chunker.py    Frontmatter-aware loading, header-aware chunking
  embeddings.py           TF-IDF (default, offline) or Voyage
  vectorstore/            local (file) and pgvector (Neon) backends
  retriever.py            Recall-optimised wide retrieval + expansion
  query_expansion.py      Domain synonym rewrite (lexical-retrieval aid)
  reranker.py             Claude Haiku filter, lexical fallback
  generate.py             Grounded generation; declines when uncovered
  groundedness.py         Numeric + faithfulness guardrail
  cache.py                Exact + semantic query cache
  scope.py                Out-of-corpus detection
  tuning.py               Empirically tuned thresholds
  eval/                   Labelled eval set, metrics, harness, tuner

api/main.py               FastAPI: /simulate, SSE stream, /approve, /ask
db/                       Postgres schema, loader, decision audit log
evals/                    Number guard, narration faithfulness, Ragas, suite
frontend/                 Next.js: scenarios, dashboard, SOP lookup
tests/                    266 tests
```

## How a scenario flows

```
question
  -> manager          parse into a validated request, with provenance
  -> supervisor       dispatch ONLY the agents this question needs
  -> inventory  ─┐
     production   ├─ run IN PARALLEL, calling deterministic tools
     historical   │
     supply      ─┘
  -> yield analyzer   join into one "projected outcome" (no new math)
  -> evidence check   did the required evidence actually arrive?
       ├─ gaps ──> gather supplementary ──> back to the analyzer (once only)
       └─ ok ────> recommendation
  -> recommendation   deterministic decision + LLM narration of it
  -> human approval   interrupt() — pauses until an operator decides
```

Three properties hold throughout:

- **Dispatch follows the question, not the diagram.** "What were our best
  sellers last month" runs ONE agent. The other three are never invoked —
  though the progress stream still names them and says why they sat out.
- **The decision is deterministic** (`simulation/decision_engine.py`): six
  scored factors — capacity, stock, supply, margin, waste, execution
  reliability — blended by weight, with any single factor able to veto. A
  capacity overrun is not made acceptable by a good margin. The model
  explains a decision Python already made, which is what makes the narration
  checkable.
- **Evidence that never arrived is reported, not assumed.** Coverage below
  100% caps the recommendation's confidence, and a factor with no data reads
  "not assessed" rather than scoring well.

## Decision support, not a point estimate

Asking "how much more for Christmas" returns a ladder, not a number:

```
LEVEL   EXTRA PRODUCT  EXTRA BOXES  CUTTER USE  CHANCE OF RUNNING OUT  STATUS
+10%          530 kg         21.6         53%                     0%  achievable
+15%          795 kg         32.5         56%                     2%  achievable
+20%  ◀      1060 kg         43.3         58%                    12%  achievable
+25%         1325 kg         54.1         60%                    37%  not achievable
+90%         4770 kg        194.8         92%                   100%  not achievable
```

The shop's confirmed Christmas uplift is **1.9x (+90%)**. The honest answer
is that it is not reachable at current ordering — Blade Eyes runs out — and
**+20% is the ceiling** until more primal is ordered. A single point estimate
("Christmas needs 4,770 kg more") is true and supports no decision.

Stockout probability is measured, not assumed: residual volatility comes out
of the sales history (the estimator recovers the 12% noise the generator
injected, without importing the constant), and risk is evaluated day by day
across the horizon rather than on 14-day totals — a cooler that empties on
day five while its delivery lands on day six has a healthy fortnightly
balance and a stockout on day five.

## Measuring what actually happens

Three datasets close the loop between plan and reality:

| | measured |
|---|---|
| Vendor fill rate | **98.9%** of ordered quantity arrives |
| On-time delivery | **92.1%** |
| Shop-wide execution | **90%** of planned cutting completes |
| Worst primal (Blade Eyes) | **83%** — it runs its cooler closest to empty |

That last row is why execution is keyed by primal rather than taken
shop-wide: the 83-98% spread is caused, not noise. Planning 400 kg of Blade
Eyes and booking 400 kg of finished product overstates supply by ~70 kg
every time.

## Evaluated against controlled faults

`python -m evals.scenarios` breaks something specific and checks the
diagnosis:

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
failure and one average would hide it. The test suite asserts each dimension
can FAIL: a ground-truth suite that always passes is worse than none.

## The RAG layer

Built to the requirements in `CLAUDE.md` step 6:

1. **Eval set first.** 25 labelled queries in `rag/eval/dataset.py` — direct,
   paraphrase, multi-hop, adversarial, and 5 with **no answer in the corpus**.
   Gold chunks are referenced by (document, heading), not chunk id, so a
   chunker change can't silently orphan the labels.
2. **Recall-optimised retrieval.** Hybrid dense + BM25 + heading match,
   top-k 14 (~28% of the corpus), plus neighbour and sibling expansion.
3. **Reranking before generation.** Claude Haiku scores every candidate and
   truncates to 4; a lexical cross-encoder-style scorer is the offline
   fallback. The model never sees raw top-k.
4. **Grounded generation.** Answers only from retrieved context, with an
   exact refusal marker, a pre-generation gate, and a numeric groundedness
   guardrail that withholds an answer whose figures can't be traced.
5. **Two-layer cache.** Exact (hashed) + semantic (embedding similarity),
   with the similarity threshold **swept against the eval set**, plus prompt
   caching on the static system prompts.

Current offline measurements (`python -m rag.eval.run_eval`):

| metric | value |
|---|---|
| hit rate @5 | 0.96 |
| recall @5 | 0.87 |
| MRR | 0.81 |
| rerank precision | 0.48 |
| decision accuracy | 0.93 |
| correct decline rate | 0.80 |
| numeric grounded rate | 1.00 |

The decline rate is **0.80, not 1.00**, and that is the honest number. An
earlier version of the eval set scored 1.00 only because it contained no
broad, open-ended questions - every answerable case used corpus vocabulary
and named one narrow fact, so the threshold sweep tuned into that blind spot
and the first real broad question ("what are some safety protocols I should
follow?") was wrongly refused. A `broad` category was added; one no-answer
case ("cutting spec for chicken thighs") now leaks offline and is recorded by
name in `KNOWN_OFFLINE_DECLINE_GAPS` rather than averaged away. Telling
chicken HANDLING (covered) from chicken CUTTING (out of scope) is a semantic
judgement no lexical threshold separates - the live path is expected to score
1.00 here.

These are the **offline** figures (lexical reranker, extractive generator) —
a floor, not the shipped behaviour. With credentials, Claude does the
reranking and generation and both should improve; the harness reports which
mode it ran in so the numbers are never misread.

## Key assumptions (all flagged inline in code too)

- Primal cost = an estimated vendor price per kg of raw primal, one price per
  primal, in `data/primal_costs.py`. The real vendor list is confidential, so
  each is back-solved from the retail value its cut plan realises at a target
  gross margin, tiered premium/routine/value, then rounded like a price list.
  Blended result: **39.0%** across the catalog net of primal cost and cutting
  labor — the owner-confirmed 38-40% — with individual cuts spanning 32-49%.
  Replacing this table with real invoices is a data edit; nothing downstream
  changes. Locked by `tests/test_primal_costs.py`.
- Labor rate: flat $27/hr average.
- Cut-product revenue = 35% of a confirmed $400k/week whole-store baseline.
- Blade Eyes split: 65% thinly sliced / 35% chuck roast — confirmed.
- Tenderloin tracked in real 40kg delivery boxes (~1–2 on hand), decoupled
  from the ~4.5kg piece weight used for labor math — confirmed.
- **Not separately confirmed:** 12% daily sales noise; 4% vendor stockout
  probability; 28-day reporting windows; all RAG thresholds (these are swept,
  not guessed — `python -m rag.eval.tune`).
- **Not separately confirmed — the supply and execution layer.** Vendor lead
  times (2–5 days), 8% late deliveries, 6% short shipments, ~100% undisrupted
  production with a 7% disruption rate, and the 8% over-replacement order
  buffer are all in `simulation/supply_generator.py`. They shape every
  forward projection.
- **Not separately confirmed — every decision threshold.** The six factor
  weights and the cut-off for each (90% cutter comfort, 1.5 days of cover,
  15%/50% stockout bands, 85%/95% execution bands, 15% waste disproportion)
  live as named constants in `simulation/decision_engine.py`, and the sweep's
  selection policy in `simulation/scenario_sweep.py`. **These decide what the
  system recommends** — a 92% cutter utilization reading as "proceed with
  caution" rather than "proceed" is one constant. They are the first thing to
  correct against real operating experience.
- **Demand autocorrelation (0.25) is assumed, not measured.** It widens
  multi-day prediction intervals by ~29%. It is not measured here because the
  generator draws each day independently, so measuring it would return ~0 and
  encode the generator's simplification as a fact about butchery. It is a
  deliberately conservative correction in the direction real demand errs.

## Optional services

Nothing below is required; each upgrades a capability in place.

| Set this | What changes |
|---|---|
| `ANTHROPIC_API_KEY` | Claude does intent parsing, reranking, narration, and the LLM-judge evals. Without it, deterministic fallbacks run and every *number* is identical. |
| `DATABASE_URL` | `simulation/ops_data.py` reads Postgres instead of CSVs; the SOP index moves to pgvector; LangGraph checkpoints persist across restarts; decisions are written to `decision_log`. Apply `db/schema.sql`, then `python -m db.load_csv`. |
| `VOYAGE_API_KEY` + `OPS_COPILOT_EMBEDDINGS=voyage` | Real dense semantic embeddings instead of TF-IDF. Re-run `rag.index build` and `rag.eval.tune` — thresholds are embedder-specific. |
| `OPS_COPILOT_LLM=stub` | Forces offline mode even with a key (what CI uses). |
| `OPS_COPILOT_CORS_ORIGINS` | Comma-separated allowed origins for the API. Defaults to localhost:3000 only. |

## Status

- [x] 1. Domain models with real yield/labor numbers
- [x] 2. Deterministic simulation functions, tested
- [x] 3. Synthetic historical data
- [x] 4. Simulation functions wrapped as LangChain tools
- [x] 5. LangGraph graph with parallel agents + human approval
- [x] 6. RAG layer for SOPs, with eval set, reranking, and tuned caching
- [x] 7. FastAPI backend (async, SSE, checkpointed sessions)
- [x] 8. Evaluation suite (number guard, narration faithfulness, Ragas, CI)
- [x] 9. Next.js frontend
- [x] Seasonal planning ("how much more for Christmas?")
- [x] 10. Supply/execution data, uncertainty, multi-factor decision engine,
      provenance, evidence-driven dispatch, controlled-fault evals
- [ ] Deployment — config is written (`frontend/vercel.json`), not deployed
