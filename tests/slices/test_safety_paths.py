from __future__ import annotations

from datetime import UTC, datetime

import pytest

from supportflow.domain.models import (
    ActionProposal,
    ApprovalMismatch,
    ResolutionProposal,
    RiskReview,
    Ticket,
    TriageResult,
)
from supportflow.workflow.service import SupportFlowService


@pytest.fixture
def safety_service() -> SupportFlowService:
    return SupportFlowService.demo(as_of=datetime(2026, 8, 31, tzinfo=UTC))


@pytest.fixture
def ticket_factory():
    def build(ticket_id: str) -> Ticket:
        return Ticket(
            ticket_id=ticket_id,
            customer_id="customer-safety-001",
            subject="Please help with my payment",
            body="I need help with order order-100 and a charge of USD 29.00.",
            order_id="order-100",
            amount="29.00",
            currency="USD",
            created_at=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
        )

    return build


@pytest.mark.parametrize(
    ("ticket_id", "expected_state"),
    [
        ("T-MISSING-001", "WAITING_CUSTOMER"),
        ("T-CONFLICT-001", "ESCALATED"),
        ("T-RISK-001", "ESCALATED"),
    ],
)
def test_unsafe_paths_never_execute(safety_service, ticket_factory, ticket_id, expected_state) -> None:
    result = safety_service.submit(ticket_factory(ticket_id))

    assert result.current_state == expected_state
    assert result.execution_results == []


def test_expired_policy_citation_escalates_without_execution(safety_service, ticket_factory) -> None:
    result = safety_service.submit(ticket_factory("T-EXPIRED-001"))

    assert result.current_state == "ESCALATED"
    assert "PG-003" in result.policy_decision.failed_rules
    assert result.execution_results == []


def test_modification_invalidates_old_approval(safety_service, duplicate_ticket) -> None:
    first = safety_service.submit(duplicate_ticket)
    old_hash = first.proposal.proposal_hash

    revised = safety_service.modify(
        first.run_id,
        edits={"reply_text": "We opened a duplicate-charge review."},
        reviewer="portfolio-owner",
    )

    assert revised.current_state == "WAITING_APPROVAL"
    assert revised.proposal.proposal_hash != old_hash
    assert revised.approvals[-1].status == "superseded"
    with pytest.raises(ApprovalMismatch):
        safety_service.approve(revised.run_id, old_hash, "portfolio-owner")


def test_modification_rejects_identical_reply_without_superseding_approval(
    safety_service, duplicate_ticket
) -> None:
    waiting = safety_service.submit(duplicate_ticket)
    original_message = next(
        action.params["message"]
        for action in waiting.proposal.actions
        if action.action_type == "SEND_REPLY"
    )

    with pytest.raises(ValueError, match="change the proposal"):
        safety_service.modify(waiting.run_id, {"reply_text": original_message}, "portfolio-owner")

    unchanged = safety_service.snapshot(waiting.run_id)
    assert unchanged.current_state == "WAITING_APPROVAL"
    assert unchanged.proposal.proposal_hash == waiting.proposal.proposal_hash
    assert unchanged.approvals == []


def test_modification_rejects_proposal_without_editable_reply(
    safety_service, ticket_factory
) -> None:
    model = safety_service.graph.triage.model
    ticket_id = "T-NO-REPLY-001"
    evidence_ids_by_heading = {
        chunk.heading: chunk.evidence_id
        for chunk in safety_service.graph.retriever.chunks
        if chunk.document.active_at(datetime(2026, 8, 31, tzinfo=UTC))
    }
    model.responses[("triage", ticket_id, 1)] = TriageResult(
        ticket_id=ticket_id,
        intent="DUPLICATE_CHARGE",
        confidence=0.99,
        rationale="The ticket describes a duplicate charge.",
        urgency="medium",
        extracted_facts={"order_id": "order-100"},
        missing_information=[],
        risk_flags=[],
        route="continue",
    )
    model.responses[("resolution", ticket_id, 1)] = ResolutionProposal(
        ticket_id=ticket_id,
        reply_text="A refund request requires a customer-facing status update.",
        evidence_refs=[
            evidence_ids_by_heading["duplicate-charge-verification"],
            evidence_ids_by_heading["duplicate-charge-refund-request"],
        ],
        actions=[
            ActionProposal(
                action_type="CREATE_REFUND_REQUEST",
                parameters={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
                reason="Verified duplicate charges may enter refund review.",
                evidence_refs=[
                    evidence_ids_by_heading["duplicate-charge-verification"],
                    evidence_ids_by_heading["duplicate-charge-refund-request"],
                ],
                risk_level="medium",
            )
        ],
        uncertainties=[],
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    model.responses[("reviewer", ticket_id, 1)] = RiskReview(
        decision="pass",
        risk_flags=[],
        unsupported_claims=[],
        required_changes=[],
        explanation="The proposal is constrained to the active duplicate-charge rules.",
    )
    waiting = safety_service.submit(ticket_factory(ticket_id))

    with pytest.raises(ValueError, match="editable reply"):
        safety_service.modify(waiting.run_id, {"reply_text": "A revised reply."}, "portfolio-owner")

    unchanged = safety_service.snapshot(waiting.run_id)
    assert unchanged.current_state == "WAITING_APPROVAL"
    assert unchanged.proposal.proposal_hash == waiting.proposal.proposal_hash
    assert unchanged.approvals == []


def test_rejection_and_escalation_never_execute(safety_service, duplicate_ticket) -> None:
    waiting = safety_service.submit(duplicate_ticket)
    rejected = safety_service.reject(waiting.run_id, "Need customer confirmation.", "portfolio-owner")

    assert rejected.current_state == "REJECTED"
    assert rejected.execution_results == []

    second = safety_service.submit(duplicate_ticket, input_revision="second-review")
    escalated = safety_service.escalate(second.run_id, "Potential account takeover.", "portfolio-owner")

    assert escalated.current_state == "ESCALATED"
    assert escalated.execution_results == []


def test_model_exhaustion_needs_attention(safety_service, ticket_factory) -> None:
    result = safety_service.submit(ticket_factory("T-EXHAUST-001"))

    assert result.current_state == "NEEDS_ATTENTION"
    assert result.execution_results == []


def test_third_revision_escalates(safety_service, duplicate_ticket) -> None:
    first = safety_service.submit(duplicate_ticket)
    second = safety_service.modify(first.run_id, {"reply_text": "First revision."}, "portfolio-owner")
    third = safety_service.modify(second.run_id, {"reply_text": "Second revision."}, "portfolio-owner")
    fourth = safety_service.modify(third.run_id, {"reply_text": "Third revision."}, "portfolio-owner")

    assert fourth.current_state == "ESCALATED"
    assert fourth.execution_results == []
