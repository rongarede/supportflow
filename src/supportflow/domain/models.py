from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TicketState(str, Enum):
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_CUSTOMER = "WAITING_CUSTOMER"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


class Intent(str, Enum):
    DUPLICATE_CHARGE = "DUPLICATE_CHARGE"


class ActionType(str, Enum):
    CREATE_REFUND_REQUEST = "CREATE_REFUND_REQUEST"
    SEND_REPLY = "SEND_REPLY"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(UTC)


class Ticket(StrictModel):
    ticket_id: str
    customer_id: str
    subject: str
    body: str
    order_id: str
    amount: str
    currency: str
    created_at: datetime

    _normalise_created_at = model_validator(mode="after")(
        lambda self: self._with_utc_created_at()
    )

    def _with_utc_created_at(self) -> Ticket:
        self.created_at = _utc(self.created_at)
        return self


class TriageResult(StrictModel):
    ticket_id: str
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    rationale: str
    missing_fields: list[str] = Field(default_factory=list)


class EvidenceItem(StrictModel):
    evidence_id: str
    document_id: str
    version: str
    heading: str
    content: str
    active: bool
    score: float


class EvidenceBundle(StrictModel):
    items: list[EvidenceItem] = Field(min_length=1)
    audit_items: list[EvidenceItem] = Field(default_factory=list)


class ActionProposal(StrictModel):
    action_type: ActionType
    params: dict[str, Any]

    @model_validator(mode="after")
    def validate_action_parameters(self) -> ActionProposal:
        if self.action_type == ActionType.CREATE_REFUND_REQUEST:
            required = {"order_id", "amount", "currency"}
            missing = sorted(key for key in required if not self.params.get(key))
            if missing:
                raise ValueError(f"CREATE_REFUND_REQUEST requires {', '.join(missing)}")
        if self.action_type == ActionType.SEND_REPLY and not self.params.get("message"):
            raise ValueError("SEND_REPLY requires message")
        return self


class ResolutionProposal(StrictModel):
    ticket_id: str
    evidence_refs: list[str] = Field(min_length=1)
    actions: list[ActionProposal] = Field(min_length=1)
    created_at: datetime
    proposal_hash: str = ""

    @model_validator(mode="after")
    def validate_canonical_hash(self) -> ResolutionProposal:
        from supportflow.domain.hashing import proposal_hash

        self.created_at = _utc(self.created_at)
        expected = proposal_hash(self)
        if not self.proposal_hash:
            self.proposal_hash = expected
        elif self.proposal_hash != expected:
            raise ValueError("proposal_hash does not match the canonical proposal")
        return self


class RiskReview(StrictModel):
    escalated: bool
    rationale: str


class PolicyDecision(StrictModel):
    outcome: Literal["allow", "block", "escalate", "revise"]
    passed_rules: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    rationale: str


class ApprovalRecord(StrictModel):
    run_id: str
    proposal_hash: str
    reviewer: str
    approved_at: datetime
    status: Literal["approved", "superseded"] = "approved"

    @model_validator(mode="after")
    def normalise_approved_at(self) -> ApprovalRecord:
        self.approved_at = _utc(self.approved_at)
        return self


class ApprovalInput(StrictModel):
    proposal_hash: str
    reviewer: str


class ExecutionResult(StrictModel):
    action_type: ActionType
    status: Literal["SIMULATED_SUCCESS"]
    reference: str


class TraceEvent(StrictModel):
    stage: str
    occurred_at: datetime
    detail: str

    @model_validator(mode="after")
    def normalise_occurred_at(self) -> TraceEvent:
        self.occurred_at = _utc(self.occurred_at)
        return self


class RunError(StrictModel):
    code: str
    message: str


class CheckpointBrief(StrictModel):
    artifact: str
    rationale: str
    evidence: list[str]
    risks: list[str]
    decision_needed: str
    artifact_location: str


class RunSnapshot(StrictModel):
    run_id: str
    current_state: TicketState
    ticket: Ticket
    triage: TriageResult | None = None
    evidence: EvidenceBundle | None = None
    proposal: ResolutionProposal | None = None
    risk_review: RiskReview | None = None
    policy_decision: PolicyDecision | None = None
    approval: ApprovalRecord | None = None
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    execution_results: list[ExecutionResult] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    errors: list[RunError] = Field(default_factory=list)


class ApprovalMismatch(ValueError):
    """Raised when a human tries to approve anything other than the reviewed proposal."""
