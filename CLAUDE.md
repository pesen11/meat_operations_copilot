# CLAUDE.md — Project Handoff: Meat-Cutting Operations Copilot

This file is context for picking up this project in Claude Code. Read this
before making changes — it captures decisions made in an earlier session
that aren't otherwise obvious from the code.

## What this project is

An operations intelligence system for a butcher shop (not a chatbot). It
simulates decisions — e.g. "if we increase chuck roast production by 15%,
what happens to inventory, labor, and waste?" — and returns computed,
structured outcomes.

**Core design principle, non-negotiable: the LLM never computes numbers.**
All yield/labor/waste/margin math is deterministic, unit-tested Python.
The LLM's job is orchestration (which tool to call) and narration
(explaining results in plain language) only. Preserve this boundary in
everything built from here on, including the RAG layer below — retrieval
and generation are still not the place for arithmetic.

## Progress so far

### Step 1 — Domain models (`models/domain.py`)
Pydantic models: `Product`, `YieldProfile`, `DemandProfile`,
`PrimalStockLevel`, `HistoricalSale`, `LaborAvailability`,
`ProductionScheduleEntry`. Notable design choices:
- `source_primal` is free text, not an enum — real primal names vary too
  much by shop/species to constrain.
- `YieldProfile.labor_basis` is `PER_BOX` or `PER_PIECE` — some cuts
  (tenderloin) are only ever partially processed from a box, so labor/yield
  math anchors to a piece weight, not the box weight.
- `YieldProfile.physical_delivery_box_weight_kg` is separate from the
  processing-unit weight used for labor math — needed because tenderloin's
  real delivery box (~40kg) is much bigger than the ~4.5kg piece actually
  cut from it. Stock tracking uses the real box weight; labor/yield math
  uses the piece weight. Don't conflate these two fields.
- `DataSource` enum (`real` / `synthetic`) tags every `Product`,
  `YieldProfile`, and `DemandProfile` row — see Data Provenance below.

### Step 2 — Deterministic simulation math (`simulation/yield_calc.py`, `simulation/cost_calc.py`)
- `yield_calc.py`: box/piece → usable product kg, trim waste kg, packs
  produced, labor minutes per pack.
- `cost_calc.py`: per-pack economics (revenue, primal cost, labor cost,
  margin) and weekly projections with a `demand_multiplier` param for
  "what if" scenarios. Primal cost is *estimated* — no real invoice cost
  data exists — and comes from `data/primal_costs.py`, one vendor $/kg
  per primal. See Assumptions below, and read that module's docstring
  before touching cost: an earlier version derived cost as a discount off
  the RETAIL price and put the whole shop at 9.5% margin.
- 57 pytest tests across `tests/test_yield_calc.py`, `tests/test_cost_calc.py`.

### Step 3 — Synthetic historical data (`simulation/seasonality.py`, `simulation/demand_forecast.py`, `simulation/velocity_tiers.py`, `simulation/history_generator.py`, `data/generate_history.py`)
Generates a full year (2025) of daily sales, primal stock, labor
availability, and production schedule, output to `data/generated/*.csv`.
- Seasonality: Ontario stat holidays (via the `holidays` package), with
  seasonal multipliers applied as **day-windows around actual events**,
  not naive Mon-Sun calendar-week grouping (an earlier version had a real
  bug here — see `simulation/seasonality.py` docstring for the full
  explanation if touching this file).
- Revenue baseline: $400k/week whole-store, cut products confirmed at
  ~35% of that ($140k/week), reconciled against the catalog's implied
  revenue via a single global scale factor (`CATALOG_SCALE_FACTOR` in
  `demand_forecast.py`).

### Step 4 — LangChain tool wrappers (`simulation/tools.py`)
Four `@tool`-decorated functions (`list_products`, `get_yield_breakdown`,
`get_pack_economics`, `get_weekly_projection`) that do a catalog lookup and
call the already-tested functions from step 2. This is the pattern to
repeat for any new tool: **the tool function itself contains zero
arithmetic**, only lookup + delegation + response shaping.
Requires `langchain-core` added to `pyproject.toml` / `requirements.txt`.
Also requires one small addition to `data/catalog.py`:
`yield_profile_by_sku: dict[str, YieldProfile] = {yp.product_sku: yp for yp in yield_profiles}`
— tools look products up by SKU, and this was missing from the original
catalog module.

### SOP corpus (`data/sops/`) — prerequisite for Step 6, not yet wired into RAG
Two markdown documents exist now, both tagged synthetic in their YAML
frontmatter (`data_source: synthetic`), both cross-referenced to real
catalog values (pack weights, yields) so RAG answers can eventually be
checked against known-correct numbers:
- `meat_cutting_instructions.md` — cutting/portioning spec for all 19
  catalog products, organized by cut type (steaks, roasts, ribs/braising,
  thin-sliced, ground/cubed) since quality parameters differ by type.
  Deliberately includes one internal "gotcha" worth knowing about if
  building eval cases: pulled pork (Boston butt) is the one roast where
  cutters should *not* trim the fat cap tight, unlike every steak/other
  roast in the doc — a naive retrieval+generation system might average
  across similar-looking "fat trim" chunks and get this one wrong, which
  makes it a good adversarial eval case for groundedness.
- `food_safety_sop.md` — temperature control, cross-contamination/species
  separation, labeling/traceability, vendor discrepancy handling (ties to
  the stockout modeling in `history_generator.py`), recall outline,
  cleaning schedule.

**Neither document is chunked, embedded, or in any vector store yet** —
they are source material only. Step 6 below is still fully unbuilt; these
two files just remove what was previously the first blocker (no corpus to
work with at all).

### Project setup notes
- `pip install -e .` (uses `pyproject.toml`) — do NOT rely on `PYTHONPATH`
  env vars; a version of this project shipped once without proper
  packaging and broke on a fresh machine. Every runnable script and every
  test must work via plain `pytest` / `python data/whatever.py` with just
  `pip install -e .` done once.
- `pytest tests/ -v` should currently show 57 passing (before adding any
  step 5/6 tests).

## Data provenance (read before adding any numbers)

Every `Product` / `YieldProfile` / `DemandProfile` row is tagged
`data_source: "real" | "synthetic"`. Real rows came from the project
owner's actual (adjusted, "similar but not exact") workplace data — 11
products across 8 primals. Synthetic rows extend catalog coverage — 8
products across 6 primals, modeled on general industry patterns, not
audited against a real operation. **Preserve this tagging on anything new
— it's a stated README/project requirement, not incidental metadata.**

## Key assumptions in the numbers (flagged, all confirmed with project owner except where noted)

- Primal cost = an estimated **vendor price per kg of raw primal**, one
  per primal, in `data/primal_costs.py` and carried on
  `YieldProfile.primal_cost_per_kg`. The real vendor list is confidential,
  so each value is back-solved from the retail value that primal's cut
  plan realises, at a target gross margin tiered premium (33.6%) / routine
  (41.6%) / value (43.6%), then rounded to $0.05. Blended result: **39.0%**
  across the catalog net of primal cost and cutting labor — the owner's
  confirmed 38-40% — individual cuts spanning 32-49%.

  **This replaced a wrong model, and the wrongness is worth remembering.**
  Cost used to be a 22.5%/37.5% discount off `price_per_kg` — but that is
  a RETAIL shelf price for trimmed, cut, packaged product, and the whole
  primal (trim and bone included) was then bought at that rate. COGS came
  out at 80-97% of revenue; the catalog cleared 9.5% and items made ~$900
  a week. The discount was never yield-aware, so no value of it could fix
  it: at 88% yield a 22.5% discount already puts COGS at 88% of revenue
  before labor. Don't reintroduce a retail-anchored cost estimate. Cost is
  keyed by primal, not product, because you buy a primal, not a cut —
  `catalog.check_primal_costs()` enforces that, and a multi-path primal
  therefore splits its margin unevenly across paths (Round yields a 48.7%
  roast beside 31.9% stew meat; the primal itself clears its target).
  `tests/test_primal_costs.py` locks the band. Real invoices drop straight
  into `PRIMAL_COST_PER_KG`; nothing downstream changes.
- Labor rate: flat **$27/hr** average — confirmed.
- Blade Eyes cut-plan split: **65% Thinly Sliced Blades / 35% Chuck
  Roast** — confirmed (this required a bug fix; both paths had
  incorrectly been set to `cut_plan_share=1.0`, which should sum to ~1.0
  across paths from one primal, not equal 1.0 on each).
- Tenderloin stock tracked in real **40kg boxes** (~1-2 kept on hand),
  decoupled from the ~4.5kg piece weight used for labor/yield math —
  confirmed, fixed a real bug where stock was being tracked in
  piece-equivalents and dipping to zero constantly.
- **Not separately confirmed — flagged as tunable if generated data looks
  off:** day-to-day sales noise (12% coefficient of variation), vendor
  stockout probability (4% per eligible delivery day per primal).


## Steps 5-9 — built

All remaining steps are now implemented. `pytest tests/ -q` shows **356
passing**; `python -m evals.run_all` exits 0.

### Step 5 — LangGraph agent graph (`agents/`)
Manager → Supervisor → parallel {Inventory, Production, Historical} → Yield
Analyzer → Recommendation → Human Approval (`interrupt()`), with a
checkpointer.

Decisions worth knowing before changing this:

- **The data agents call tools directly; Claude does not pick them.** Once
  the Manager has produced a validated plan, which tools to call is fully
  determined — an LLM adds latency and a chance to call the wrong one, and
  no judgement. `agents/llm.run_tool_loop` exists for genuinely open-ended
  questions but is not on the scenario path.
- **The verdict is deterministic too** (`_verdict` in `agents/nodes.py`).
  Claude narrates a decision Python already made. That is what makes the
  narration checkable by `evals/number_guard.py`.
- **`human_approval` must stay SYNCHRONOUS and the graph must be driven by
  `graph.stream()`, never `graph.astream()`, on Python 3.10.** LangGraph's
  `interrupt()` calls `get_config()`, which on Python < 3.11 deliberately
  raises inside a running event loop. Under `astream` the approval step
  therefore fails at exactly the point it should pause. `api/main.py` runs
  the sync stream on a worker thread and bridges events into asyncio; there
  is a regression test for this in `tests/test_api.py`.
- New deterministic calculators were added first, with tests, per the
  project convention: `simulation/inventory_calc.py` (stock cover, cutter
  capacity, scenario impact) and `simulation/history_analysis.py`
  (summaries, trends, movers), plus `simulation/ops_data.py` as the single
  read layer over the history (CSV or Postgres).

### Step 6 — RAG layer (`rag/`)

**Corpus decision (flagged, as this file asked):** the corpus was expanded
from 2 documents to **5** before any retrieval tuning —
`receiving_and_vendor_sop.md`, `equipment_cleaning_maintenance_sop.md`, and
`opening_closing_checklists.md` were added. Reason: with 2 documents a top-k
of any useful size returns most of the corpus, so recall@k is trivially 1.0
and reranking cannot be measured at all — the eval set would have been unable
to demonstrate the very tradeoffs points 6.2 and 6.3 require. The new
documents deliberately share vocabulary with the existing ones ("clean"
across food-safety and equipment, "temperature" across food-safety and
receiving) to create genuine distractors. All three are tagged
`data_source: synthetic`.

Requirement-by-requirement:

1. **Eval set built first** — `rag/eval/dataset.py`, 25 labelled cases
   across direct / paraphrase / multi_hop / adversarial / no_answer, with
   reference answers. Gold chunks are referenced by (doc_id, heading
   substring) and resolved at eval time, so re-chunking cannot silently
   orphan a label — `resolve_gold` raises instead.
2. **Recall-optimised retrieval** — hybrid dense + BM25 + heading, top-k 14,
   plus neighbour/sibling expansion. recall@k is measured explicitly.
3. **Reranking before generation** — Claude Haiku relevance filter
   (`LLMReranker`), lexical cross-encoder-style fallback offline. Truncation
   uses a *relative* floor, not only an absolute one: measured on this
   corpus, gold and non-gold lexical scores overlap heavily (gold min 0.35 vs
   non-gold median 0.40), so an absolute threshold cannot separate them and
   tuning it harder just drops gold chunks.
4. **Grounded generation** — `rag/generate.py`, with an exact
   `NOT_IN_CORPUS` marker, a pre-generation gate, and `rag/groundedness.py`
   as a live guardrail (numeric always; Ragas-style faithfulness optionally).
   An answer containing a figure not in the source is withheld, not shown.
5. **Caching** — exact + semantic (`rag/cache.py`), plus `cache_control:
   ephemeral` on the static system prompts in `agents/llm.py`.

**Every threshold in this layer is swept, not guessed** — `python -m
rag.eval.tune` writes `rag/index/tuned_params.json`, and `rag/tuning.py`
reports whether the values in force are tuned or defaults. Two things the
sweep found that are worth not re-litigating:

- **The semantic cache threshold has a hard floor (0.60) that the sweep
  cannot override.** The sweep on 25 queries suggested 0.33; that is
  overfitting to the most-similar unrelated pair this particular eval set
  happens to contain, and real traffic will contain a closer one. A false
  cache hit serves a wrong answer silently, while a miss only costs latency,
  so the asymmetry is deliberate. The floor relaxes on its own as the eval
  set grows.
- **TF-IDF cannot connect some paraphrases at all.** "How cold does a
  delivery need to be" and "product must arrive at or below 4°C" share zero
  content words, so that pair's similarity is genuinely 0.0. This is
  reported, not papered over. `rag/query_expansion.py` is a domain-synonym
  rewrite that recovers retrieval for these (recall@5 went 0.80 → 0.95); it
  is explicitly a crutch for the lexical default and becomes inert with real
  dense embeddings (`OPS_COPILOT_EMBEDDINGS=voyage`).

**Offline eval numbers are a FLOOR, not the shipped behaviour** — the
harness prints which mode it ran in. Detecting an out-of-corpus question is a
semantic judgement; offline it leans on `rag/scope.py` (vocabulary absence),
which is a weak signal doing a job the model does properly.

### Step 7 — FastAPI backend (`api/`, `db/`)
`/simulate`, `/simulate/stream` (SSE, one event per agent as it finishes),
`/approve`, `/ask`, plus reference and dashboard endpoints. `db/schema.sql`
has the operational tables, a pgvector note, and a `decision_log` audit
table; `python -m db.load_csv` moves `data/generated/*.csv` into Postgres.
`simulation/ops_data.py` switches backends automatically and returns
identically-shaped DataFrames either way.

### Step 8 — Evaluation suite (`evals/`)
- `number_guard.py` — the structural enforcement of "the LLM never computes".
  Accepts a **closed list** of presentation transforms (rounding, percent
  display, thousands separators, absolute value) and nothing else. Do not add
  a percentage tolerance to make a failure go away: that silently licenses
  the model to do arithmetic, which is the thing being prevented. It runs in
  the API response, not only in CI.
- `narration_faithfulness.py` — the custom check this file asked for. Catches
  what the number guard structurally cannot: a real number attached to the
  wrong quantity, a reversed direction, a narration contradicting the verdict.
- `ragas_adapter.py` — real Ragas when `pip install -e ".[eval]"`. It
  **excludes correct declines from answer_relevancy** and reports their
  decline rate separately, because Ragas scores a correct "I don't know" near
  zero — the known blind spot. That is the metric applied to the population
  it is valid for, not the metric being massaged.
- `run_all.py` — exit 2 = the core invariant broke, 1 = a threshold drifted.
- `.github/workflows/ci.yml` — runs everything with `OPS_COPILOT_LLM=stub`
  (free, no flakes) on Python 3.10 and 3.12. **3.10 is in the matrix
  deliberately** because of the `interrupt()` behaviour noted above.

### Step 9 — Next.js frontend (`frontend/`)
Three pages: scenarios (SSE progress, verdict, approval, outcome cards),
dashboard (weekly revenue line, stock-coverage bars, tables), SOP lookup.
Next 16 + Recharts 3; both were upgraded from the initially-pinned versions
because the pinned Next had a published CVE and Recharts 2.x is deprecated.
Verified in a real browser, not just compiled.

Chart choices that are deliberate: the today-vs-scenario comparison is
**indexed to 100** rather than dual-axis (the four measures are dollars, kg,
minutes and a ratio); stock status uses the reserved status palette with the
status **spelled out in words** in the table and tooltip, never colour alone;
the legend is hand-rolled HTML because Recharts 3 dropped `Legend`'s
`payload` prop and its internal ordering reversed the series.

**A real bug this surfaced:** `weekly_revenue_series` returned a ragged
oldest bucket when the window was not a multiple of 7 days, which rendered as
a cliff at the left edge of the revenue chart and looked like a collapse in
trade. Partial buckets are now dropped.

## Question coverage — the retrieval fix

The system could simulate a scenario well and answer almost nothing else.
Reported symptoms: "how was ribeye last week" returned a top-seller line
about a different product, "best sellers last month" named only one product,
"how much should we increase on weekends" came back unsupported. Four
separate causes, all of them worth not reintroducing:

1. **No time-window parsing existed.** The history tools all took
   `start`/`end`; nothing ever passed `start`. Every history question got the
   default 28 days, so "yesterday", "last week" and "last month" returned the
   same answer and none of them said which window it was.
   `simulation/time_windows.py` now resolves the phrase, and the window is
   carried on `Plan["window"]` and **stated back in every answer**. It anchors
   to the last date in the DATA, never to today: the generated history ends
   on a fixed date, so a wall-clock anchor returns an empty window that reads
   as "we sold nothing".
2. **Data was fetched and then dropped.** `historical_agent` computed
   `sales_summary` on every run and `yield_analyzer` never copied it into the
   outcome; `inventory_agent` fetched `focus_primal` and the same thing
   happened. The narration could only see `top_movers` and `primals_at_risk`,
   which is exactly what it printed. If you add a tool call to an agent, add
   the corresponding line to `yield_analyzer` in the same commit.
3. **The offline narration had no history or inventory branch**, and printed
   `top_movers[0]` — one name where the question asked for a ranking.
4. **Intent vocabulary gaps** sent ordinary questions to `unsupported`.
   Inventory is now tested BEFORE the scenario branch ("how many kg of chuck
   roast do we have" named a product with no percentage and was being
   simulated as a what-if), and any question carrying a time expression is
   treated as history even with no history keyword.

5. **No whole-catalog total existed.** `sales_summary` takes a required
   `sku`, and `historical_agent` only called it inside `if sku:`. A question
   naming no product ("what were our sales yesterday", "how much did we sell
   last week") therefore had no total to report, and the narration fell back
   to what it did have: rankings. It answered *which cuts led* when the
   question asked *how much the shop sold*. `history_analysis.catalog_sales_summary`
   is the catalog-wide counterpart — total kg and revenue, products sold, open
   days, weekday/weekend split, best day — aggregated per DAY across every SKU
   so two products selling on one day is one open day, not two. It is called
   UNGATED, so it runs whether or not a product was named.

   Two things worth not re-introducing. The outcome key is
   `catalog_sales_totals`, **not** `catalog_totals`: the production agent
   already publishes `catalog_totals` for a scenario's extra volume, and
   `yield_analyzer` copies production's version in AFTER the historical block,
   so the first naming silently clobbered the sales figures on any question
   that dispatched both. And kg is summed through `units_to_kg` per SKU, never
   by adding `units_sold` — pack sizes differ, so raw units are not comparable
   across products. `tests/test_history_analysis.py` cross-checks the total
   against `catalog_performance` over the real history (two independent code
   paths) and asserts a 7-day total lands near the confirmed ~$140k/week rather
   than only being self-consistent — the primal-cost lesson.

New deterministic analysis, tested before being wrapped as tools, per the
usual convention: `catalog_performance` (per-product verdict: focus / reduce
/ watch / steady), `slow_movers`, `weekend_uplift`, and
`seasonal_planning.period_actuals` (what actually sold during the last
occurrence of a season).

Three judgement calls in there that are deliberate:

- **`MIN_COMPARISON_DAYS = 14` gates every recommendation.** Daily sales
  carry 12% noise, so a one-day window swings tens of percent on chance.
  "How did we do yesterday" is a fine question; answering it with "so order
  less short ribs" is advice built on noise. Below the threshold the analysis
  still reports what sold and sets `comparison_reliable=False`, but every
  verdict is downgraded to "watch".
- **Seasonality is checked before recommending a cut.** Telling an operator
  to pull back on ribeye when the real cause is that Christmas ended is a
  wrong call, not a vague one. A change the calendar explains never becomes
  "reduce".
- **Observed and confirmed multipliers are reported side by side, never
  blended.** `period_actuals` measures what happened; `PERIODS` is a business
  input. Note the last-long-weekend figure reads 1.877x observed against
  1.7x confirmed because that window abuts Boxing Day and its "normal"
  comparison days are contaminated — shown honestly rather than smoothed.

Two latent bugs in `evals/number_guard.py` surfaced once narrations carried
real data, and both were *tokenisation*, not tolerance — nothing about what
counts as a matching figure was widened:
- Only the year was stripped from an ISO date, leaving `-12-31` to be read as
  the numbers -12 and -31.
- The bare-year rule ate the leading digits of any 4-digit quantity, so
  `2044.71 kg` left `.71` to be checked as 71.
A hyphenated range in prose (`20-25 boxes`) genuinely does read as -25, so
that stays a violation and the narration writes "20 to 25" instead. There is
a test pinning each of these.

`frontend/components/AnswerDetails.tsx` renders the new blocks. `OutcomeCards`
only ever produced tiles for a scenario, so a history or inventory answer
showed a paragraph under an empty heading.

## Step 10 — decision layer, supply/execution data, provenance

Built in response to an external review of the backend. The review's summary
of the gap was right and worth restating: the problem was never that the
system needed more AI, it was that **business reasoning and orchestration
were not explicit or testable enough**. Nothing below moves arithmetic
toward the model; several things move it further away.

`pytest tests/ -q` shows **574 passing** (was 356); `python -m evals.run_all`
exits 0 and now includes a controlled-fault scenario suite.

### 10.1 Three new datasets, causally linked to the old ones

`data/generate_history.py` now writes seven files plus two derived exports.

- **`purchase_orders.csv`** (2,832 rows) — incoming supply, with lead times,
  late deliveries and short shipments.
- **`actual_production.csv`** (6,052 rows) — planned vs actual per cutting
  run, with a `shortfall_reason`.
- **`production_schedule.csv`** gains `status` and `priority`.
- **`product_catalog.csv`** and **`labor_requirements.csv`** are DERIVED
  REFERENCE EXPORTS of `data/catalog.py`, not new sources of truth. Nothing
  reads them back. The review asked for a SKU→primal bridge as "probably
  more important than adding random new datasets"; that bridge already
  existed and was already validated (`YieldProfile.source_primal`,
  `check_cut_plan_shares`, `check_primal_costs`) — it was just invisible to
  anyone reading only the CSVs. Exporting it makes it legible. Adding a
  second, hand-editable copy would have been a regression.

**Purchase orders are DERIVED from the stock series, not generated beside
it.** `primal_stock.csv` already encodes every delivery that ever happened —
any day a box count rises, something arrived. A second independent random
stream would have contradicted it everywhere, and every "on hand + incoming"
calculation would then rest on two files that disagree. So deliveries are
read out of the stock series and a PO is back-filled for each. The two
cannot disagree, because one is computed from the other. Only the *open*
orders past the end of history are genuinely new — that part has not
happened and cannot be derived.

Execution rates carry real signal because they are CAUSED, not noised: a
cancelled run traces to a day the stock series says the cooler was empty.
Shop-wide execution is **90%**; Blade Eyes, the highest-volume primal, is
**83%** precisely because it runs closest to empty.

Three generation bugs found and fixed by eyeballing outputs, per the
project's convention:
- Status was ratio-driven, so natural weight variance made 4,373 of 6,052
  runs "partial" and the column carried no signal. Status is now
  CAUSE-driven: a run is partial because something went wrong.
- Forward orders were sized from `implied_deliveries`, which returns NET
  day-over-day increases (delivery minus that day's consumption). Every
  primal came out under-supplied — incoming ran at 37-84% of forecast draw
  and eight of seventeen primals projected a stockout inside two weeks.
  Orders are now sized from measured consumption.
- That consumption then had to be DE-SEASONALISED, for the same reason
  `seasonal_plan` de-seasonalises: the history ends 31 December, so trailing
  consumption is the Christmas run-up. Sizing January's orders off it
  inflated every cooler by 24-62%.

### 10.2 New deterministic calculators (tests first, as always)

- `simulation/execution_rates.py` — planned vs actual; turns a plan into a
  forecast. Keyed by primal, not shop-wide, because the 83-98% spread is
  caused, not noise.
- `simulation/supply_calc.py` — forward inventory projection, walked
  DAY BY DAY. A closing balance hides the shape: a primal can finish the
  horizon comfortably and still hit zero on day four waiting for a day-five
  delivery. **Trim waste is reported, never subtracted** — the ledger is in
  raw primal kg and demand is in finished kg, so dividing by `yield_pct`
  already contains the waste. The review's equation
  (`+ production − demand − waste`) double-counts it, and production is not
  a separate addition either: it converts primal, it does not create it.
- `simulation/uncertainty.py` — prediction intervals and P(stockout).
  Volatility is MEASURED off residuals, never imported from
  `SALES_NOISE_CV`; `test_measured_volatility_recovers_injected_noise`
  asserts the estimator recovers ~0.12, which stays meaningful against real
  data. Lognormal, so a lower bound cannot go negative.
- `simulation/bottlenecks.py` — what binds on a whole-catalog plan, replacing
  the `days_of_cover: None` hole. "Meaningless to average" does not mean
  "nothing to say"; it means the answer is a RANKING.
- `simulation/scenario_sweep.py` — a ladder of levels with a stated
  selection policy.
- `simulation/decision_engine.py` — six weighted factors, any one able to veto.

### 10.3 Judgement calls worth not re-litigating

- **Thresholds land exactly on the caution line.** `_ramp_through` anchors
  each named threshold to score 0.5, so crossing `EXECUTION_CONCERN` means
  precisely what the constant's name says. With arbitrary ramp endpoints, a
  rate sitting exactly on the 0.85 line scored 0.60 and raised no caution —
  the constant named the wrong place.
- **An unassessed factor is EXCLUDED from the blend, never scored 1.0.**
  "We did not look" and "we looked and it is fine" must not produce the same
  number.
- **A blocking factor cannot be outvoted by a weighted average.** A capacity
  overrun is not compensated for by a good margin.
- **Waste is judged on PROPORTIONALITY.** Cutting 15% more meat produces
  ~15% more trim; that is arithmetic, not a finding.
- **Feasibility excludes margin.** A less profitable option is a worse
  option, not an impossible one; ruling it out would remove a decision the
  operator is entitled to make.
- **Catalog stockout risk is the WORST primal's, not an average.** The shop
  stops cutting when any one primal it needs is empty.

### 10.4 Bugs this surfaced (all fixed, all with tests)

- `bottlenecks.catalog_impact` did not de-seasonalise and reported **119.1%**
  cutter utilization at Christmas — the exact wrong number
  `seasonal_planning.py`'s docstring records having hit. It now agrees with
  `seasonal_plan` to four decimals (0.9178), from two independent code paths.
  `test_matches_seasonal_plan_utilization` pins it.
- The sweep passed a PRODUCT multiplier into a PRIMAL's stockout model. Blade
  Eyes feeds two products, so +90% chuck roast was modelled as +90% Blade
  Eyes draw and reported a 99.8% chance of running dry. True figures: +19%
  and 1.3%.
- `stockout_probability` compared 14-day TOTALS and returned 0.000 for every
  primal — including ones whose daily ledger runs dry repeatedly. A total
  cannot express timing. It now walks the ladder and takes the worst day.
- The waste factor read `demand_multiplier` (1.0 on a seasonal plan, where
  the real uplift lives on `seasonal.multiplier`) and concluded waste was
  rising "190% faster than volume" when it was exactly proportional. That
  scored waste 0.00 and named it the limiting factor on a plan whose real
  constraint was capacity.
- Inventory/supply/economics all reported "not assessed" on whole-catalog
  questions, so a seasonal plan was decided on three factors.
- The supply outlook was projected at 1.0x while risk was at 1.9x — the same
  answer reported 157% coverage and a certain stockout in one breath.
- The ladder only evaluated a target when the OPERATOR named one, so "how
  much for Christmas" swept a generic ladder and never checked the 1.9x the
  shop actually plans to. A business rule is a target too; the
  operator-supplied flag controls whether the system SUBSTITUTES a size, not
  whether the ladder evaluates one.

### 10.5 Contracts and provenance (`models/`)

- `models/provenance.py` — `Evidence[T]` with a source. Four things that used
  to look identical now do not: the operator said "christmas"; the calendar
  inferred it; 1.9 is a confirmed business rule; 1.877 was measured.
  **A numeric field tagged `llm_inference` is a structural violation** and
  `claim_guard` raises on it — the core rule, enforced at the type layer.
- `models/request.py` — `UserRequest` splits intent / entities / parameters.
  `explicit_multiplier` is `Optional[float]`, so "keep production flat"
  (1.0) and "how much should we?" (None) stay distinguishable. Collapsing
  both to 1.0 is why seasonal planning once needed special-casing.
  `Plan` is still produced, now DERIVED via `to_plan()`, so no existing node
  changed.
- `models/decision.py` — structured `Recommendation`, `DecisionFactor`,
  `ScenarioOption`.

### 10.6 Orchestration (`agents/planner.py`, rewired graph)

Dispatch is a function of the QUESTION. "What were our best sellers last
month" now runs ONE agent; the other three are never invoked.

**Skipped agents still report.** The previous design ran every specialist and
had idle ones return `{"skipped": True}` so the SSE stream showed three
agents reporting in. That rationale was about the STREAM, and is satisfiable
without doing the work: the supervisor emits a skip event naming why. The
stream is now strictly more informative. `test_agent_graph.py`'s old
assertion was rewritten to the new contract rather than worked around.

An **evidence sufficiency loop** (`evidence_check` → `gather_supplementary`,
capped at ONE pass) catches the failure this project has shipped twice —
data computed and never copied into the outcome, invisible because nothing
errored. Coverage below 1.0 caps the recommendation's confidence.

The **production agent now does real work**: it called `list_products()` and
stopped whenever no SKU was named, so a seasonal question got a catalog
listing while the inventory agent did the seasonal arithmetic alone. It now
supplies shop-wide capacity, bottlenecks and the option ladder.

### 10.7 Evaluation

- `evals/claim_guard.py` — the number guard's semantic counterpart. Catches a
  business-rule figure presented as the system's own calculation, a
  measurement claim with no measurement behind it, a narration contradicting
  its verdict, and any number sourced from the model. Runs in the API
  response, not only in CI.
- `evals/scenarios.py` — **controlled operational faults with known correct
  diagnoses.** Perturbs the data layer (cut labour 40%, cancel every open
  order, delay deliveries five days, quarter the cooler, halve execution)
  and scores SIX dimensions separately: intent, dispatch, evidence,
  diagnosis, decision, grounding. Separate because "right answer, wrong
  reason" is the interesting failure and an average would hide it.
  `tests/test_claim_guard.py` asserts each dimension can FAIL — a
  ground-truth suite that always passes is worse than none.

Currently 9/9 cases, 100% on every dimension.


## Not done

- **Nothing is deployed.** `frontend/vercel.json` exists and the API is
  deployment-ready, but no Vercel project, no hosted backend, and no Neon
  instance was created — that needs the project owner's accounts.
- **No live-LLM numbers have been measured.** Every figure in the README is
  from the offline deterministic path. Set `ANTHROPIC_API_KEY` and run
  `python -m evals.run_all --deep --ragas` to get the live figures; expect
  reranking and declining to improve, and the arithmetic to be unchanged.
- **The 4 reranker-drop cases** listed by `rag.eval.run_eval` are a known
  limit of the lexical fallback, not of the design. Check them again with
  credentials before tuning anything.
- **The step-10 thresholds are unconfirmed assumptions.** Every constant in
  `decision_engine.py`, the `FACTOR_WEIGHTS`, the sweep's selection policy,
  `DEMAND_AUTOCORRELATION`, and the vendor/execution knobs in
  `supply_generator.py` are stated as module constants precisely so they are
  easy to correct — but none has been validated against a real operation.
  They decide what the system recommends.
- **Demand autocorrelation is assumed, not measured.** The generator draws
  each day independently, so measuring it here would return ~0 and encode
  the generator's simplification as a fact about butchery. 0.25 is a
  deliberately conservative correction in the direction real demand errs.
- **The reviewer's remaining asks not built:** a finished-product inventory
  layer (this models primal boxes and finished packs are a thin
  short-cycle buffer), multi-step remediation search ("could we add labour /
  move production earlier / reallocate primal?"), and per-role labour
  requirements beyond the cutter role.

## Working conventions to keep

- Every new deterministic function gets a pytest test before being wrapped
  as a tool — this project has had multiple real bugs caught precisely
  because numbers were run and eyeballed before being trusted (Blade Eyes
  cut-plan-share, tenderloin box-weight, the seasonality ISO-week boundary
  bug, the ragged first bucket in `weekly_revenue_series`, and the
  retail-anchored primal cost). Don't skip this step to move faster.
  Note what the primal-cost bug says about the *kind* of test to write:
  57 tests asserted the math was internally consistent and all passed,
  because none of them asserted the shop could pay rent. Assert on the
  magnitude of an outcome, not only on its self-consistency.
- Flag assumptions inline in code comments AND in this file when adding new
  estimated numbers, the same way `PRIMAL_COST_PER_KG`, `SALES_NOISE_CV`,
  and `STOCKOUT_PROB` are flagged — so they're easy to find and correct
  later against real data.
- Prefer small, targeted diffs over regenerating whole files/directories —
  the project owner has specifically asked for this workflow (changes
  described precisely, not wholesale file dumps) to keep iteration fast and
  reviewable.
- **Thresholds get swept against an eval set, not chosen by feel.** Anything
  in `rag/tuning.py` is set by `python -m rag.eval.tune`; if you add a new
  threshold there, add it to the sweep too. Where a sweep result is
  untrustworthy (small eval set), say so in the code and guard it, as the
  semantic-cache floor does.
- **Report which mode a measurement came from.** Offline (no API key)
  numbers and live numbers are not comparable, and an unlabelled metric
  invites someone to compare them anyway. `rag.eval.run_eval` and
  `evals.run_all` both print the mode.
- The test suite must keep running with no credentials and no database.
  That is only possible because no number comes from the model — if a change
  makes a test need an API key to pass, the change has probably moved
  arithmetic into the LLM layer.
