/**
 * Typed client for the FastAPI backend (api/main.py).
 *
 * The types here mirror api/schemas.py. They are written out rather than
 * inferred so a backend shape change surfaces as a TypeScript error at build
 * time instead of an undefined at runtime on the shop floor.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/** Where a value came from. A confirmed business rule and a figure the
 * system inferred carry different authority, and the UI must not present
 * them with the same weight. */
export type ValueSource =
  | "user"
  | "business_rule"
  | "historical_data"
  | "derived"
  | "system_default"
  | "inferred"
  | "llm_inference";

export interface EvidenceRef {
  value: unknown;
  source: ValueSource;
  basis: string;
  confidence?: number;
  as_of?: string;
  sample_size?: number;
}

export interface Plan {
  intent: string;
  product_sku: string | null;
  source_primal: string | null;
  demand_multiplier: number;
  reasoning: string;
  resolved_by: string;
  /** True when the OPERATOR named a size; false when they asked the system
   * to supply one. The two render differently: "here is what your 15% does"
   * versus "here is the size we suggest". */
  operator_supplied_a_size: boolean;
  provenance: Record<string, EvidenceRef>;
  required_evidence: string[];
}

export type FactorName =
  | "capacity"
  | "inventory"
  | "supply"
  | "economics"
  | "waste"
  | "execution";

export interface DecisionFactor {
  name: FactorName;
  /** 1.0 = no concern, 0.0 = worst case. Higher is always better. */
  score: number;
  weight: number;
  /** False means the factor could not be evaluated at all. Render that
   * differently from a factor that scored well - "we did not look" and "we
   * looked and it is fine" are not the same statement. */
  assessed: boolean;
  is_blocking: boolean;
  headline: string;
  detail: Record<string, unknown>;
}

export interface ScenarioOption {
  label: string;
  demand_multiplier: number;
  extra_weekly_product_kg: number;
  extra_weekly_primal_kg: number;
  extra_weekly_boxes: number;
  cutter_utilization_pct: number;
  exceeds_cutter_capacity: boolean;
  extra_weekly_trim_waste_kg: number;
  weekly_margin: number | null;
  delta_weekly_margin: number | null;
  stockout_probability: number | null;
  days_of_cover: number | null;
  feasible: boolean;
  infeasible_reasons: string[];
}

export interface Recommendation {
  verdict: "proceed" | "proceed_with_caution" | "do_not_proceed" | "informational";
  blockers: string[];
  cautions: string[];
  narration: string;
  narrated_by: string;
  requires_approval: boolean;

  action: string;
  decision_score: number | null;
  confidence: number;
  limiting_factor: FactorName | null;
  factors: DecisionFactor[];
  options: ScenarioOption[];
  chosen_option: string | null;
  recommended_multiplier: number | null;
  recommended_change_pct: number | null;
  reasons: string[];
  risks: string[];
  evidence_coverage: number | null;
  evidence_gaps: string[];
}

export interface MarginBlock {
  baseline_weekly_margin?: number;
  scenario_weekly_margin?: number;
  delta_weekly_margin?: number;
  baseline_weekly_revenue?: number;
  scenario_weekly_revenue?: number;
}

export interface InventoryBlock {
  extra_weekly_primal_kg?: number;
  extra_weekly_boxes?: number;
  boxes_on_hand?: number;
  // Null for a whole-catalog seasonal plan: one days-of-cover figure across
  // seventeen primals would be meaningless. Per-primal ordering is in
  // SeasonalBlock.top_primals_to_order instead.
  baseline_days_of_cover?: number | null;
  scenario_days_of_cover?: number | null;
}

export interface LaborBlock {
  extra_weekly_cutting_minutes: number;
  weekly_cutter_minutes_available: number;
  baseline_cutter_utilization_pct: number;
  scenario_cutter_utilization_pct: number;
  exceeds_cutter_capacity: boolean;
}

export interface WasteBlock {
  baseline_weekly_trim_waste_kg?: number;
  scenario_weekly_trim_waste_kg?: number;
  extra_weekly_trim_waste_kg?: number;
}

export interface SeasonalBlock {
  period: string;
  multiplier: number;
  multiplier_source: string;
  window_start: string | null;
  window_end: string | null;
  days_until: number | null;
  total_extra_weekly_kg: number;
  total_extra_primal_kg: number;
  total_extra_boxes: number;
  total_extra_trim_waste_kg: number;
  top_primals_to_order: {
    source_primal: string;
    extra_boxes: number;
    boxes_on_hand: number;
    boxes_to_order: number;
  }[];
}

export interface ProjectedOutcome {
  intent: string;
  product_sku: string | null;
  source_primal: string | null;
  demand_multiplier: number;
  margin?: MarginBlock;
  seasonal?: SeasonalBlock;
  seasonal_calendar?: {
    period: string;
    multiplier: number;
    starts: string;
    ends: string;
    days_until: number;
  }[];
  inventory?: InventoryBlock;
  labor?: LaborBlock;
  waste?: WasteBlock;
  primals_at_risk?: StockPosition[];
  top_movers?: TopMover[];
  sales_trend?: SalesTrend;
  [key: string]: unknown;
}

export interface ClaimGuard {
  passed: boolean;
  claims_checked: number;
  violations: { kind: string; detail: string; excerpt: string }[];
  attributed_claims: number;
  note: string;
}

export interface NumberGuard {
  passed: boolean;
  numbers_checked: number;
  violations: { stated: string; value: number; context: string }[];
  source_values: number;
  note: string;
}

export interface TraceEvent {
  node: string;
  message: string;
  data: Record<string, unknown>;
}

export interface SimulateResponse {
  thread_id: string;
  question: string;
  plan: Plan;
  projected_outcome: ProjectedOutcome;
  recommendation: Recommendation;
  narration: string;
  awaiting_approval: boolean;
  approval: Record<string, unknown> | null;
  trace: TraceEvent[];
  errors: string[];
  number_guard: NumberGuard | null;
  claim_guard: ClaimGuard | null;
}

export interface StockPosition {
  source_primal: string;
  as_of: string;
  boxes_on_hand: number;
  velocity_tier: string;
  target_boxes_low: number;
  target_boxes_high: number;
  kg_on_hand: number;
  avg_daily_primal_kg: number;
  days_of_cover: number | null;
  status: "stockout" | "critical" | "below_target" | "ok" | "over_target";
}

export interface TopMover {
  sku: string;
  product_name: string;
  total_kg: number;
  total_revenue: number;
  share_of_revenue_pct: number;
}

export interface SalesTrend {
  sku: string;
  recent_avg_kg_per_day: number;
  prior_avg_kg_per_day: number;
  change_pct: number | null;
  direction: string;
  seasonality_explains_change: boolean;
}

export interface WeeklyRevenuePoint {
  week_ending: string;
  revenue: number;
}

export interface AskSource {
  chunk_id: string;
  document: string;
  doc_id: string;
  heading: string;
  data_source: string;
  relevance: number;
  cited: boolean;
}

export interface AskResponse {
  query: string;
  answer: string;
  answered: boolean;
  sources: AskSource[];
  cache: string;
  latency_ms: number;
  groundedness: Record<string, unknown> | null;
  stats: Record<string, unknown>;
}

export interface Health {
  status: string;
  llm: string;
  vector_store: Record<string, unknown>;
  data_backend: string;
  rag_params: Record<string, unknown>;
  catalog_products: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* response had no JSON body; the status text will do */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  stock: (asOf?: string) =>
    request<StockPosition[]>(`/stock${asOf ? `?as_of=${asOf}` : ""}`),
  weeklyRevenue: (start?: string, end?: string) => {
    const params = new URLSearchParams();
    if (start) params.set("start", start);
    if (end) params.set("end", end);
    const query = params.toString();
    return request<WeeklyRevenuePoint[]>(
      `/history/weekly-revenue${query ? `?${query}` : ""}`,
    );
  },
  topMovers: (limit = 5, by: "revenue" | "kg" = "revenue") =>
    request<TopMover[]>(`/history/top-movers?limit=${limit}&by=${by}`),
  simulate: (question: string, threadId?: string) =>
    request<SimulateResponse>("/simulate", {
      method: "POST",
      body: JSON.stringify({ question, thread_id: threadId }),
    }),
  approve: (threadId: string, approved: boolean, approvedBy?: string, note?: string) =>
    request<{ thread_id: string; approval: Record<string, unknown> }>("/approve", {
      method: "POST",
      body: JSON.stringify({
        thread_id: threadId,
        approved,
        approved_by: approvedBy,
        note,
      }),
    }),
  ask: (query: string) =>
    request<AskResponse>("/ask", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
};

/**
 * Stream a scenario run, calling back as each agent reports in.
 *
 * Uses EventSource (hence the GET variant of /simulate/stream) so the browser
 * handles reconnection and framing. Returns a cancel function.
 */
export function streamSimulation(
  question: string,
  threadId: string,
  handlers: {
    onProgress?: (event: TraceEvent) => void;
    onApprovalRequired?: (payload: Record<string, unknown>) => void;
    onResult?: (result: SimulateResponse) => void;
    onError?: (message: string) => void;
    onDone?: () => void;
  },
): () => void {
  const params = new URLSearchParams({ question, thread_id: threadId });
  const source = new EventSource(`${API_URL}/simulate/stream?${params}`);

  const parse = <T,>(event: MessageEvent): T | null => {
    try {
      return JSON.parse(event.data) as T;
    } catch {
      return null;
    }
  };

  source.addEventListener("progress", (event) => {
    const payload = parse<TraceEvent>(event as MessageEvent);
    if (payload) handlers.onProgress?.(payload);
  });
  source.addEventListener("approval_required", (event) => {
    const payload = parse<Record<string, unknown>>(event as MessageEvent);
    if (payload) handlers.onApprovalRequired?.(payload);
  });
  source.addEventListener("result", (event) => {
    const payload = parse<SimulateResponse>(event as MessageEvent);
    if (payload) handlers.onResult?.(payload);
  });
  source.addEventListener("error", (event) => {
    const payload = parse<{ errors?: string[] }>(event as MessageEvent);
    handlers.onError?.(payload?.errors?.join("; ") ?? "Connection to the backend failed.");
  });
  source.addEventListener("done", () => {
    handlers.onDone?.();
    source.close();
  });

  return () => source.close();
}
