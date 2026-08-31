from __future__ import annotations

from datetime import UTC, datetime

import pytest

from supportflow.domain.models import ApprovalMismatch, Ticket
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


def test_rejection_and_escalation_never_execute(safety_service, duplicate_ticket) -> None:
    waiting = safety_service.submit(duplicate_ticket)
    rejected = safety_service.reject(waiting.run_id, "Need customer confirmation.", "portfolio-owner")

    assert rejected.current_state == "REJECTED"
    assert rejected.execution_results == []

    second = safety_service.submit(duplicate_ticket)
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
