from datetime import UTC, datetime

from supportflow.domain.models import (
    ActionProposal,
    EvidenceBundle,
    EvidenceItem,
    ResolutionProposal,
    RiskReview,
    Ticket,
)
from supportflow.policy.gate import PolicyGate


def _ticket() -> Ticket:
    return Ticket(
        ticket_id="ticket-001",
        customer_id="customer-001",
        subject="Duplicate charge",
        body="Please help with a duplicate charge.",
        order_id="order-100",
        amount="29.00",
        currency="USD",
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


def _proposal(evidence_refs: list[str]) -> ResolutionProposal:
    return ResolutionProposal(
        ticket_id="ticket-001",
        evidence_refs=evidence_refs,
        actions=[
            ActionProposal(
                action_type="CREATE_REFUND_REQUEST",
                params={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
            ),
            ActionProposal(action_type="SEND_REPLY", params={"message": "We submitted your refund request."}),
        ],
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


def _evidence(active: bool = True) -> EvidenceBundle:
    return EvidenceBundle(
        items=[
            EvidenceItem(
                evidence_id="policy-duplicate-charge-001",
                document_id="policy-duplicate-charge",
                version="1.0",
                heading="duplicate-charge-verification",
                content="Verify the duplicate charge before a refund request.",
                active=active,
                score=0.9,
            )
        ]
    )


def test_gate_allows_reviewed_evidence_backed_refund_request() -> None:
    decision = PolicyGate().evaluate(
        _ticket(),
        _evidence(),
        _proposal(["policy-duplicate-charge-001"]),
        RiskReview(escalated=False, rationale="Evidence supports a simulated refund request."),
    )

    assert decision.outcome == "allow"
    assert decision.failed_rules == []


def test_gate_blocks_missing_evidence_reference() -> None:
    decision = PolicyGate().evaluate(
        _ticket(),
        _evidence(),
        _proposal(["missing-evidence"]),
        RiskReview(escalated=False, rationale="Evidence review completed."),
    )

    assert decision.outcome == "block"
    assert decision.failed_rules == ["PG-002"]
