-- Operational + vector schema for the Meat-Cutting Operations Copilot.
--
-- Target: a single Neon Postgres instance holding BOTH the operational
-- tables and the SOP vector index. That co-location is the reason pgvector
-- was chosen over a separate Pinecone index at this data scale: one
-- connection string, one backup, and the option of joining SOP chunks
-- against operational rows later without a cross-service call.
--
-- Apply with:  psql "$DATABASE_URL" -f db/schema.sql
-- Then load:   python -m db.load_csv

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Operational history (previously data/generated/*.csv)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS historical_sales (
    id          BIGSERIAL PRIMARY KEY,
    sku         TEXT        NOT NULL,
    sale_date   DATE        NOT NULL,
    units_sold  NUMERIC(12, 3) NOT NULL CHECK (units_sold >= 0),
    revenue     NUMERIC(12, 2) NOT NULL CHECK (revenue >= 0),
    UNIQUE (sku, sale_date)
);
CREATE INDEX IF NOT EXISTS historical_sales_date_idx ON historical_sales (sale_date);
CREATE INDEX IF NOT EXISTS historical_sales_sku_date_idx ON historical_sales (sku, sale_date);

CREATE TABLE IF NOT EXISTS primal_stock (
    id             BIGSERIAL PRIMARY KEY,
    source_primal  TEXT NOT NULL,
    as_of          DATE NOT NULL,
    boxes_on_hand  NUMERIC(10, 2) NOT NULL CHECK (boxes_on_hand >= 0),
    velocity_tier  TEXT,
    UNIQUE (source_primal, as_of)
);
CREATE INDEX IF NOT EXISTS primal_stock_as_of_idx ON primal_stock (as_of);

CREATE TABLE IF NOT EXISTS labor_availability (
    id                BIGSERIAL PRIMARY KEY,
    shift_date        DATE NOT NULL,
    role              TEXT NOT NULL,
    available_minutes NUMERIC(10, 1) NOT NULL CHECK (available_minutes >= 0),
    headcount         INTEGER NOT NULL CHECK (headcount >= 0),
    UNIQUE (shift_date, role)
);
CREATE INDEX IF NOT EXISTS labor_availability_date_idx ON labor_availability (shift_date);

CREATE TABLE IF NOT EXISTS production_schedule (
    id                BIGSERIAL PRIMARY KEY,
    schedule_date     DATE NOT NULL,
    source_primal     TEXT NOT NULL,
    planned_primal_kg NUMERIC(12, 2) NOT NULL CHECK (planned_primal_kg > 0),
    -- A schedule with no status can only express intent, so every forward
    -- projection had to assume the plan executes in full. See
    -- actual_production for the other half.
    status            TEXT NOT NULL DEFAULT 'planned'
                      CHECK (status IN ('planned', 'in_progress', 'completed',
                                        'partial', 'cancelled')),
    priority          TEXT NOT NULL DEFAULT 'normal'
                      CHECK (priority IN ('high', 'normal', 'low')),
    notes             TEXT,
    UNIQUE (schedule_date, source_primal)
);
CREATE INDEX IF NOT EXISTS production_schedule_date_idx ON production_schedule (schedule_date);
CREATE INDEX IF NOT EXISTS production_schedule_status_idx ON production_schedule (status);

-- ---------------------------------------------------------------------------
-- Supply and execution
-- ---------------------------------------------------------------------------
-- What was ORDERED vs what actually arrived, and what was PLANNED vs what was
-- actually cut. Without these two tables every forward projection silently
-- assumes vendors deliver in full and plans execute at 100%. Measured on the
-- generated history neither is true: vendors fill 98.9% and the shop
-- completes 90% of planned cutting (83% for its highest-volume primal).

CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id             TEXT PRIMARY KEY,
    source_primal     TEXT NOT NULL,
    order_date        DATE NOT NULL,
    expected_arrival  DATE NOT NULL,
    quantity_kg       NUMERIC(12, 2) NOT NULL CHECK (quantity_kg > 0),
    status            TEXT NOT NULL
                      CHECK (status IN ('ordered', 'in_transit', 'delivered',
                                        'delayed', 'short_shipped', 'cancelled')),
    supplier          TEXT NOT NULL,
    -- NULL while in transit. Coercing these to a real date would turn
    -- "has not arrived" into "arrived", which is exactly the error the
    -- projected-inventory equation exists to avoid.
    actual_arrival    DATE,
    received_kg       NUMERIC(12, 2) CHECK (received_kg IS NULL OR received_kg >= 0),
    data_source       TEXT NOT NULL DEFAULT 'synthetic'
                      CHECK (data_source IN ('real', 'synthetic')),
    -- An order cannot have arrived without a received quantity, or carry a
    -- received quantity without having arrived.
    CHECK ((actual_arrival IS NULL) = (received_kg IS NULL))
);
CREATE INDEX IF NOT EXISTS purchase_orders_primal_idx ON purchase_orders (source_primal);
CREATE INDEX IF NOT EXISTS purchase_orders_arrival_idx ON purchase_orders (expected_arrival);
CREATE INDEX IF NOT EXISTS purchase_orders_open_idx ON purchase_orders (status)
    WHERE actual_arrival IS NULL;

CREATE TABLE IF NOT EXISTS actual_production (
    id                BIGSERIAL PRIMARY KEY,
    production_date   DATE NOT NULL,
    source_primal     TEXT NOT NULL,
    planned_primal_kg NUMERIC(12, 2) NOT NULL CHECK (planned_primal_kg >= 0),
    actual_primal_kg  NUMERIC(12, 2) NOT NULL CHECK (actual_primal_kg >= 0),
    status            TEXT NOT NULL
                      CHECK (status IN ('planned', 'in_progress', 'completed',
                                        'partial', 'cancelled')),
    -- 'stock_short' | 'labor_short' | 'equipment' | 'quality_hold' | NULL.
    -- Free text rather than an enum: the useful list of reasons a cutting run
    -- fell short is exactly the thing a real shop would extend.
    shortfall_reason  TEXT,
    data_source       TEXT NOT NULL DEFAULT 'synthetic'
                      CHECK (data_source IN ('real', 'synthetic')),
    UNIQUE (production_date, source_primal)
);
CREATE INDEX IF NOT EXISTS actual_production_date_idx ON actual_production (production_date);
CREATE INDEX IF NOT EXISTS actual_production_reason_idx ON actual_production (shortfall_reason)
    WHERE shortfall_reason IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Decision audit trail
-- ---------------------------------------------------------------------------
-- Every scenario the copilot answered and what the operator decided about it.
-- The projected outcome is stored as JSONB verbatim: it is the tool output
-- the narration was checked against, so it is the evidence for the decision,
-- not a summary of it.

CREATE TABLE IF NOT EXISTS decision_log (
    id                 BIGSERIAL PRIMARY KEY,
    thread_id          TEXT        NOT NULL,
    asked_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    question           TEXT        NOT NULL,
    plan               JSONB       NOT NULL,
    projected_outcome  JSONB       NOT NULL,
    verdict            TEXT        NOT NULL,
    narration          TEXT,
    narrated_by        TEXT,
    approval_status    TEXT,
    approved_by        TEXT,
    approved_at        TIMESTAMPTZ,
    note               TEXT
);
CREATE INDEX IF NOT EXISTS decision_log_thread_idx ON decision_log (thread_id);
CREATE INDEX IF NOT EXISTS decision_log_asked_at_idx ON decision_log (asked_at DESC);

-- ---------------------------------------------------------------------------
-- SOP vector index
-- ---------------------------------------------------------------------------
-- The embedding column's dimension depends on the active provider (TF-IDF
-- vocabulary size, or 1024 for voyage-3), so rag/vectorstore/pgvector.py
-- creates these tables itself at build time with the right width rather than
-- hardcoding one here. They are documented here for completeness:
--
--   sop_chunks      (chunk_id PK, doc_id, doc_title, document_type,
--                    data_source, heading_path JSONB, text, char_count,
--                    metadata JSONB, embedding vector(N))
--                   + ivfflat index on embedding vector_cosine_ops
--   sop_index_state (index_name PK, state JSONB, updated_at)
--
-- Build with: python -m rag.index build --backend pgvector
