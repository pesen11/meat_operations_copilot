"""
Domain models for the Meat-Cutting Operations Copilot.

Design principle: these models describe the *shape* of the data. All
yield/labor/waste/margin arithmetic lives in simulation/*.py as plain,
unit-tested Python functions — never in the LLM layer. Agents call those
functions as LangChain @tool wrappers; they never compute numbers themselves.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Unit(str, Enum):
    KG = "kg"
    EACH = "each"
    CASE = "case"


class GradeLabel(str, Enum):
    """Grade label, since yield/labor can differ by grade
    (e.g., Wagyu trims differently than a standard Choice cut)."""
    PRIME = "prime"
    CHOICE = "choice"
    SELECT = "select"
    WAGYU = "wagyu"
    UNGRADED = "ungraded"


class LaborBasis(str, Enum):
    """
    How labor time was actually measured for a given yield path:

    - PER_BOX: the whole box gets opened and processed in one continuous
      run (open -> trim -> tie/portion -> cut -> tray). Labor and yield
      math are both anchored to the full box weight.
    - PER_PIECE: only a few individual primal pieces get pulled out of a
      box at a time (e.g. tenderloin — a whole 40kg box doesn't get cut
      at once because it wouldn't sell before spoiling). Labor and yield
      math are anchored to a single piece's weight, not the box.
    """
    PER_BOX = "per_box"
    PER_PIECE = "per_piece"


class DataSource(str, Enum):
    """Provenance tag: was this row transcribed from real workplace data,
    or generated as plausible synthetic data modeled on industry patterns?
    Keep this explicit end-to-end — the project's README states which
    numbers are real vs synthetic, and this field is how that claim stays
    auditable rather than becoming an unverifiable assertion."""
    REAL = "real"
    SYNTHETIC = "synthetic"


# ---------------------------------------------------------------------------
# Product & Yield
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """A sellable end product (e.g., 'Chuck Roast')."""
    sku: str
    name: str
    unit: Unit = Unit.KG
    price_per_kg: float = Field(gt=0, description="Sell price per kg")
    avg_pack_weight_kg: float = Field(gt=0, description="Average finished, sold-pack weight in kg")
    data_source: DataSource = DataSource.SYNTHETIC


class YieldProfile(BaseModel):
    """
    Core simulation math model: describes how much of a given product
    comes out of a given primal, how much is lost to trim/waste, and how
    long it takes to process — anchored either to a whole box or to a
    single piece pulled from a box, per `labor_basis`.

    A single primal can have MULTIPLE yield paths (e.g. Blade Eyes ->
    Chuck Roast on one path, Blade Eyes -> Thinly Sliced Blades on
    another) — model this as multiple YieldProfile rows sharing the same
    source_primal, not a single fixed ratio. `cut_plan_share` expresses
    "for every processing unit of this primal, X% is routed down this
    path."

    Per-pack labor/yield numbers are *derived*, not raw inputs — see
    simulation/yield_calc.py — because the box or piece labor time gets
    spread across however many finished packs it actually yields.
    """
    source_primal: str = Field(description="Primal or sub-primal name, e.g. 'Blade Eyes', 'Pork Belly'")
    product_sku: str
    grade: GradeLabel = GradeLabel.UNGRADED

    labor_basis: LaborBasis = LaborBasis.PER_BOX

    yield_pct: float = Field(
        gt=0, le=1, description="Fraction of the processing unit's weight (box or piece) "
                                 "that becomes finished product"
    )
    trim_waste_pct: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Fraction of the processing unit's weight lost as trim/waste on this path. "
                    "If not supplied, assumed to be (1 - yield_pct).",
    )
    cut_plan_share: float = Field(
        default=1.0, gt=0, le=1,
        description="Share of this primal's volume routed down this particular yield path "
                    "(multiple YieldProfiles per primal should sum to ~1.0 across paths)",
    )
    aging_days: Optional[int] = Field(
        default=None, description="Wet/dry aging days required before this path is cuttable, if any"
    )
    primal_cost_per_kg: Optional[float] = Field(
        default=None, gt=0,
        description="Estimated vendor cost per kg of the RAW primal this path is cut from, as "
                    "delivered (trim and bone included). Set from data/primal_costs.py, which "
                    "keys cost by primal so every path off one primal shares one cost. This "
                    "replaces the old primal_discount_pct, which expressed cost as a discount "
                    "off the finished product's RETAIL price and so modelled the shop as buying "
                    "primals at near shelf price — see data/primal_costs.py for the full story. "
                    "If unset, cost_calc.py falls back to back-solving from "
                    "DEFAULT_TARGET_GROSS_MARGIN_PCT.",
    )
    physical_delivery_box_weight_kg: Optional[float] = Field(
        default=None, gt=0,
        description="The REAL weight of the box this primal is delivered in, for stock-tracking "
                    "purposes only — decoupled from processing_unit_weight_kg, which drives labor/"
                    "yield math. These differ for PER_PIECE primals like tenderloin: the box you "
                    "receive is ~40kg, but labor/yield math is anchored to a single ~4.5kg piece "
                    "pulled from it. If unset, stock tracking falls back to processing_unit_weight_kg "
                    "(correct for PER_BOX primals, where the two are the same thing).",
    )
    data_source: DataSource = DataSource.SYNTHETIC

    # --- PER_BOX fields ---
    avg_box_weight_kg: Optional[float] = Field(
        default=None, gt=0, description="Required if labor_basis == PER_BOX: incoming box weight, kg"
    )
    labor_minutes_per_box: Optional[float] = Field(
        default=None, gt=0, description="Required if labor_basis == PER_BOX: total labor minutes "
                                         "to open, trim, cut, and tray/pack one full box"
    )

    # --- PER_PIECE fields ---
    avg_piece_weight_kg: Optional[float] = Field(
        default=None, gt=0, description="Required if labor_basis == PER_PIECE: weight of a single "
                                         "primal piece pulled from the box, kg"
    )
    labor_minutes_per_piece: Optional[float] = Field(
        default=None, gt=0, description="Required if labor_basis == PER_PIECE: labor minutes to "
                                         "trim, cut, and tray/pack a single primal piece"
    )

    @property
    def processing_unit_weight_kg(self) -> float:
        """Weight of whatever labor is actually anchored to: a box, or a single piece."""
        return self.avg_box_weight_kg if self.labor_basis == LaborBasis.PER_BOX else self.avg_piece_weight_kg

    @property
    def labor_minutes_per_processing_unit(self) -> float:
        return self.labor_minutes_per_box if self.labor_basis == LaborBasis.PER_BOX else self.labor_minutes_per_piece

    @model_validator(mode="after")
    def check_basis_fields_and_default_waste(self):
        if self.labor_basis == LaborBasis.PER_BOX:
            missing = [f for f in ("avg_box_weight_kg", "labor_minutes_per_box") if getattr(self, f) is None]
        else:
            missing = [f for f in ("avg_piece_weight_kg", "labor_minutes_per_piece") if getattr(self, f) is None]
        if missing:
            raise ValueError(f"labor_basis={self.labor_basis.value} requires: {', '.join(missing)}")

        if self.trim_waste_pct is None:
            self.trim_waste_pct = round(1.0 - self.yield_pct, 6)
        elif self.yield_pct + self.trim_waste_pct > 1.0 + 1e-6:
            raise ValueError("yield_pct + trim_waste_pct cannot exceed 1.0")
        return self


class DemandProfile(BaseModel):
    """
    Reference demand pattern for a product, used to seed/shape synthetic
    HistoricalSale generation. Kept separate from YieldProfile since it's
    a sales-side fact, not a cutting-side one.
    """
    product_sku: str
    avg_sales_weekday_kg: float = Field(ge=0)
    avg_sales_weekend_kg: float = Field(ge=0)
    data_source: DataSource = DataSource.SYNTHETIC


# ---------------------------------------------------------------------------
# Inventory / Sales / Labor / Schedule
# ---------------------------------------------------------------------------

class InventoryLevel(BaseModel):
    """Finished-product PACK inventory (post-cutting, ready to sell) — not primal boxes.
    Since production is planned 'today + tomorrow', this is normally a thin, short-cycle
    buffer, not a deep stockpile. See PrimalStockLevel for raw box inventory."""
    sku: str
    on_hand_units: float = Field(ge=0)
    as_of: datetime
    location: str = "main"


class VelocityTier(str, Enum):
    """How fast a primal's products move, which drives how many boxes get kept on hand.
    Confirmed with user: low ~5-6 boxes, medium ~8-12 boxes, best ~20-25 boxes."""
    LOW = "low"
    MEDIUM = "medium"
    BEST = "best"


class PrimalStockLevel(BaseModel):
    """
    Raw primal BOX inventory — what's actually sitting in the cooler before cutting.
    Keyed by primal (not product sku), since a box of a primal may feed multiple
    downstream products. Distinct from InventoryLevel, which tracks finished packs.
    """
    source_primal: str
    as_of: date
    boxes_on_hand: float = Field(ge=0)
    velocity_tier: Optional[VelocityTier] = None


class HistoricalSale(BaseModel):
    sku: str
    sale_date: date
    units_sold: float = Field(ge=0)
    revenue: float = Field(ge=0)


class LaborAvailability(BaseModel):
    shift_date: date
    role: str = Field(description="e.g., 'cutter', 'wrapper', 'lead'")
    available_minutes: float = Field(ge=0)
    headcount: int = Field(ge=0)


class ProductionScheduleEntry(BaseModel):
    """One planned cutting run.

    `status` and `priority` were added after the schedule turned out to be
    unable to express the most common real state of a cutting day: a run
    that is half done. Without a status every projection had to assume the
    plan executes in full, so "400 kg planned" and "400 kg available to
    sell" were the same number — see ActualProduction for the other half.
    """
    schedule_date: date
    source_primal: str
    planned_primal_kg: float = Field(gt=0)
    status: "ProductionStatus" = Field(
        default_factory=lambda: ProductionStatus.PLANNED,
        description="Lifecycle state. Historic rows are resolved (completed/"
                    "partial/cancelled); rows dated after the end of the "
                    "history stay 'planned'.",
    )
    priority: "SchedulePriority" = Field(
        default_factory=lambda: SchedulePriority.NORMAL,
        description="Which runs get the cutters first on a short day. "
                    "Derived from the primal's velocity tier.",
    )
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Supply & production execution
#
# These three models close the loop between what the shop PLANNED and what
# actually happened. Before they existed the simulation implicitly assumed
# the schedule executes perfectly and that stock appears in the cooler by
# magic, which made every forward projection optimistic in a way nothing
# measured.
# ---------------------------------------------------------------------------

class ProductionStatus(str, Enum):
    """Lifecycle of one scheduled cutting run.

    PARTIAL is the operationally interesting one: the run happened but did
    not finish, so some planned primal is still uncut and the finished
    product it would have produced does not exist. Treating PARTIAL as
    COMPLETED is how a plan silently overstates supply.
    """
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class SchedulePriority(str, Enum):
    """Which runs get the cutters first when the day is short.

    Derived from velocity tier, not hand-set: the primals that carry the
    shop's volume are the ones that must not slip.
    """
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class PurchaseOrderStatus(str, Enum):
    ORDERED = "ordered"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    SHORT_SHIPPED = "short_shipped"
    CANCELLED = "cancelled"


class PurchaseOrder(BaseModel):
    """
    One vendor order for raw primal.

    `quantity_kg` is what was ORDERED; `received_kg` is what actually turned
    up, and the two differ on a short shipment. Downstream supply math must
    read `received_kg` for history and `quantity_kg` for anything still in
    transit -- see simulation/supply_calc.py, which is the only place that
    distinction is allowed to be made.
    """
    po_id: str
    source_primal: str
    order_date: date
    expected_arrival: date
    quantity_kg: float = Field(gt=0, description="Ordered quantity of raw primal, kg")
    status: PurchaseOrderStatus = PurchaseOrderStatus.ORDERED
    supplier: str
    actual_arrival: Optional[date] = None
    received_kg: Optional[float] = Field(
        default=None, ge=0,
        description="Kg actually received. None while in transit. Less than "
                    "quantity_kg on a short shipment, which is a real vendor "
                    "behaviour the food-safety SOP already has a procedure for.",
    )
    data_source: DataSource = DataSource.SYNTHETIC

    @property
    def is_open(self) -> bool:
        """True while this order still represents incoming supply."""
        return self.status in (PurchaseOrderStatus.ORDERED,
                               PurchaseOrderStatus.IN_TRANSIT,
                               PurchaseOrderStatus.DELAYED)

    @property
    def days_late(self) -> Optional[int]:
        if self.actual_arrival is None:
            return None
        return (self.actual_arrival - self.expected_arrival).days

    @property
    def fill_rate(self) -> Optional[float]:
        """Received / ordered. None while in transit."""
        if self.received_kg is None or self.quantity_kg <= 0:
            return None
        return self.received_kg / self.quantity_kg


class ActualProduction(BaseModel):
    """
    What a cutting run actually produced, against what it was scheduled to.

    The ratio actual/planned is the shop's EXECUTION RATE, and it is the
    single number that turns a plan into a forecast. A plan of 400kg at an
    83% historical execution rate is a 334kg expectation, not a 400kg one.
    """
    production_date: date
    source_primal: str
    planned_primal_kg: float = Field(ge=0)
    actual_primal_kg: float = Field(ge=0)
    status: ProductionStatus = ProductionStatus.COMPLETED
    shortfall_reason: Optional[str] = Field(
        default=None,
        description="Why actual fell short: 'stock_short', 'labor_short', "
                    "'equipment', 'quality_hold', or None when it did not.",
    )
    data_source: DataSource = DataSource.SYNTHETIC

    @property
    def execution_rate(self) -> Optional[float]:
        if self.planned_primal_kg <= 0:
            return None
        return self.actual_primal_kg / self.planned_primal_kg
