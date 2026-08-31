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
    BILLING_QUESTION = "BILLING_QUESTION"
    REFUND_REQUEST = "REFUND_REQUEST"
    DUPLICATE_CHARGE = "DUPLICATE_CHARGE"
    REFUND_STATUS = "REFUND_STATUS"


class ActionType(str, Enum):
    SEND_REPLY = "SEND_REPLY"
    ADD_TAG = "ADD_TAG"
    REQUEST_INFORMATION = "REQUEST_INFORMATION"
    CREATE_REFUND_REQUEST = "CREATE_REFUND_REQUEST"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"


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
    urgency: Literal["low", "medium", "high"]
    extracted_facts: dict[str, str]
    missing_information: list[str]
    risk_flags: list[str]
    route: Literal["continue", "request_information", "escalate_human"]

    @property
    def missing_fields(self) -> list[str]:
        """Compatibility name for the pre-contract workflow implementation."""
        return self.missing_information


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
    parameters: dict[str, Any]
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    risk_level: Literal["low", "medium", "high"]

    @property
    def params(self) -> dict[str, Any]:
        """Compatibility accessor; serialized contracts use `parameters`."""
        return self.parameters

    @model_validator(mode="after")
    def validate_action_parameters(self) -> ActionProposal:
        if self.action_type == ActionType.CREATE_REFUND_REQUEST:
            required = {"order_id", "amount", "currency"}
            missing = sorted(key for key in required if not self.parameters.get(key))
            if missing:
                raise ValueError(f"CREATE_REFUND_REQUEST requires {', '.join(missing)}")
        required_by_action = {
            ActionType.SEND_REPLY: {"message"},
            ActionType.ADD_TAG: {"tag"},
            ActionType.REQUEST_INFORMATION: {"message"},
            ActionType.ESCALATE_HUMAN: {"queue", "summary"},
        }
        required = required_by_action.get(self.action_type, set())
        missing = sorted(key for key in required if not self.parameters.get(key))
        if missing:
            raise ValueError(f"{self.action_type.value} requires {', '.join(missing)}")
        return self


class ResolutionProposal(StrictModel):
    ticket_id: str
    reply_text: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    actions: list[ActionProposal] = Field(min_length=1)
    uncertainties: list[str]
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
    decision: Literal["pass", "revise", "escalate"]
    risk_flags: list[str]
    unsupported_claims: list[str]
    required_changes: list[str]
    explanation: str = Field(min_length=1)

    @property
    def escalated(self) -> bool:
        return self.decision == "escalate"

    @property
    def rationale(self) -> str:
        return self.explanation


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
    idempotency_key: str
    action_type: ActionType
    status: Literal["succeeded", "failed", "skipped_duplicate"]
    reference: str
    simulated_payload: dict[str, Any]
    executed_at: datetime
    error: str | None = None

    @model_validator(mode="after")
    def normalise_executed_at(self) -> ExecutionResult:
        self.executed_at = _utc(self.executed_at)
        return self


class TraceEvent(StrictModel):
    stage: str
    occurred_at: datetime
    detail: str

    @model_validator(mode="after")
    def normalise_occurred_at(self) -> TraceEvent:
        self.occurred_at = _utc(self.occurred_at)
        return self


class RunError(StrictModel):
    stage: str
    error_type: str
    message: str
    attempt: int = Field(ge=1)
    retryable: bool
    occurred_at: datetime

    @property
    def code(self) -> str:
        """Compatibility name used by the workbench and earlier tests."""
        return self.error_type

    @model_validator(mode="after")
    def normalise_occurred_at(self) -> RunError:
        self.occurred_at = _utc(self.occurred_at)
        return self


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
    node_attempts: dict[str, int] = Field(default_factory=dict)


class ApprovalMismatch(ValueError):
    """Raised when a human tries to approve anything other than the reviewed proposal."""
