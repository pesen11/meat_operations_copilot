"""Request/response models for the API.

These are the contract the frontend codes against, so they are explicit
rather than `dict[str, Any]` everywhere - a projected outcome whose shape is
only knowable by reading agents/nodes.py is not a usable API.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000,
                          description="Plain-language operations question")
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation/session id. Required to approve the result "
                    "later; generated if omitted.")
    as_of: Optional[str] = Field(
        default=None, description="ISO date to evaluate against (default: latest data)")


class PlanOut(BaseModel):
    intent: str
    product_sku: Optional[str] = None
    source_primal: Optional[str] = None
    demand_multiplier: float = 1.0
    reasoning: str = ""
    resolved_by: str = "keyword"
    # True when the OPERATOR named a size, False when they asked the system
    # to supply one. The frontend renders those two differently: one is
    # "here is what your 15% does", the other "here is the size we suggest".
    operator_supplied_a_size: bool = True
    # Per-field origin: {"period": {"value": "christmas", "source": "user",
    # "basis": "named in the question"}}. Lets the UI show WHERE a figure
    # came from instead of presenting every number with equal authority.
    provenance: dict[str, Any] = {}
    required_evidence: list[str] = []


class DecisionFactorOut(BaseModel):
    """One scored axis of the decision. `assessed=False` means the factor
    could not be evaluated at all - rendered differently from a factor that
    was evaluated and scored well, because "we did not look" and "we looked
    and it is fine" are not the same statement."""
    name: str
    score: float
    weight: float
    assessed: bool = True
    is_blocking: bool = False
    headline: str = ""
    detail: dict[str, Any] = {}


class ScenarioOptionOut(BaseModel):
    """One rung on the swept ladder of production levels."""
    label: str
    demand_multiplier: float
    extra_weekly_product_kg: float
    extra_weekly_primal_kg: float
    extra_weekly_boxes: float
    cutter_utilization_pct: float
    exceeds_cutter_capacity: bool
    extra_weekly_trim_waste_kg: float
    weekly_margin: Optional[float] = None
    delta_weekly_margin: Optional[float] = None
    stockout_probability: Optional[float] = None
    days_of_cover: Optional[float] = None
    feasible: bool = True
    infeasible_reasons: list[str] = []


class RecommendationOut(BaseModel):
    verdict: str
    blockers: list[str] = []
    cautions: list[str] = []
    narration: str = ""
    narrated_by: str = "template"
    requires_approval: bool = False

    # --- added with the decision engine -------------------------------
    # All optional with defaults, so an existing client that ignores them
    # keeps working unchanged.
    action: str = "no_action"
    decision_score: Optional[float] = None
    confidence: float = 0.0
    limiting_factor: Optional[str] = None
    factors: list[DecisionFactorOut] = []
    options: list[ScenarioOptionOut] = []
    chosen_option: Optional[str] = None
    recommended_multiplier: Optional[float] = None
    recommended_change_pct: Optional[float] = None
    reasons: list[str] = []
    risks: list[str] = []
    evidence_coverage: Optional[float] = None
    evidence_gaps: list[str] = []


class SimulateResponse(BaseModel):
    thread_id: str
    question: str
    plan: PlanOut
    projected_outcome: dict[str, Any] = {}
    recommendation: RecommendationOut
    narration: str = ""
    awaiting_approval: bool = False
    approval: Optional[dict[str, Any]] = None
    trace: list[dict[str, Any]] = []
    errors: list[str] = []
    # Populated by evals/number_guard.py: every number in the narration is
    # checked against the tool output it claims to describe.
    number_guard: Optional[dict[str, Any]] = None
    # Populated by evals/claim_guard.py: the number guard's semantic
    # counterpart - is each figure attached to the right quantity, and
    # attributed to the right source. Both run in the response, not only in
    # CI, because a violation in production is worse than one in a build.
    claim_guard: Optional[dict[str, Any]] = None


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    approved_by: Optional[str] = None
    note: Optional[str] = Field(default=None, max_length=500)


class ApprovalResponse(BaseModel):
    thread_id: str
    approval: dict[str, Any]
    trace: list[dict[str, Any]] = []


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000,
                       description="Question about the shop's SOPs")
    use_cache: bool = True


class SourceOut(BaseModel):
    chunk_id: str
    document: str
    doc_id: str
    heading: str
    data_source: str
    relevance: float
    cited: bool


class AskResponse(BaseModel):
    query: str
    answer: str
    # False means the system declined because the corpus does not cover the
    # question. That is a correct outcome, not an error - see CLAUDE.md 6.4.
    answered: bool
    sources: list[SourceOut] = []
    cache: str = "miss"
    latency_ms: float = 0.0
    groundedness: Optional[dict[str, Any]] = None
    stats: dict[str, Any] = {}


class ProductOut(BaseModel):
    sku: str
    name: str
    source_primal: str
    price_per_kg: float
    data_source: str


class StockOut(BaseModel):
    source_primal: str
    as_of: str
    boxes_on_hand: float
    velocity_tier: str
    target_boxes_low: float
    target_boxes_high: float
    kg_on_hand: float
    avg_daily_primal_kg: float
    days_of_cover: Optional[float] = None
    status: str


class HealthResponse(BaseModel):
    status: str
    llm: str
    # Which checkpointer is actually in use. default_checkpointer() falls back
    # to InMemorySaver on ANY failure (missing driver, bad URL), which loses
    # the approval interrupt on restart without raising. A deployment must be
    # able to see that rather than discover it when /approve 404s.
    checkpointer: str = "unknown"
    vector_store: dict[str, Any]
    data_backend: str
    rag_params: dict[str, Any]
    catalog_products: int
