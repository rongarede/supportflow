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
            ),
            EvidenceItem(
                evidence_id="policy-refund-request-001",
                document_id="policy-refund-request",
                version="1.0",
                heading="duplicate-charge-refund-request",
                content="Create a refund request for a verified duplicate charge.",
                active=active,
                score=0.8,
            ),
            EvidenceItem(
                evidence_id="policy-refund-request-002",
                document_id="policy-refund-request",
                version="1.0",
                heading="refund-timing",
                content="Tell the customer when a submitted request may be resolved.",
                active=active,
                score=0.7,
            ),
        ]
    )


def test_gate_allows_reviewed_evidence_backed_refund_request() -> None:
    decision = PolicyGate().evaluate(
        _ticket(),
        _evidence(),
        _proposal(
            [
                "policy-duplicate-charge-001",
                "policy-refund-request-001",
                "policy-refund-request-002",
            ]
        ),
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


def test_gate_blocks_active_but_irrelevant_evidence_for_duplicate_charge_refund() -> None:
    evidence = EvidenceBundle(
        items=[
            EvidenceItem(
                evidence_id="policy-password-reset-001",
                document_id="policy-password-reset",
                version="1.0",
                heading="password-reset",
                content="Reset a password after ownership verification.",
                active=True,
                score=0.99,
            )
        ]
    )
    decision = PolicyGate().evaluate(
        _ticket(),
        evidence,
        _proposal(["policy-password-reset-001"]),
        RiskReview(escalated=False, rationale="Review completed."),
    )

    assert decision.outcome == "block"
    assert decision.failed_rules == ["PG-004"]


def test_gate_escalates_inactive_evidence_instead_of_treating_it_as_executable() -> None:
    decision = PolicyGate().evaluate(
        _ticket(),
        _evidence(active=False),
        _proposal(
            [
                "policy-duplicate-charge-001",
                "policy-refund-request-001",
                "policy-refund-request-002",
            ]
        ),
        RiskReview(escalated=False, rationale="The expired rule is retained for audit."),
    )

    assert decision.outcome == "escalate"
    assert "PG-003" in decision.failed_rules


def test_gate_escalates_unresolved_eligibility_and_exclusion_conflict() -> None:
    evidence = _evidence().model_copy(
        update={
            "items": [
                *_evidence().items,
                EvidenceItem(
                    evidence_id="policy-refund-eligibility-001",
                    document_id="policy-refund-eligibility",
                    version="1.0",
                    heading="refund-eligibility",
                    content="The duplicate charge is eligible.",
                    active=True,
                    score=0.6,
                ),
                EvidenceItem(
                    evidence_id="policy-refund-eligibility-002",
                    document_id="policy-refund-eligibility",
                    version="1.0",
                    heading="refund-exclusion",
                    content="An exclusion might apply.",
                    active=True,
                    score=0.5,
                ),
            ]
        }
    )
    decision = PolicyGate().evaluate(
        _ticket(),
        evidence,
        _proposal([item.evidence_id for item in evidence.items]),
        RiskReview(escalated=False, rationale="The exclusion has not been resolved."),
    )

    assert decision.outcome == "escalate"
    assert "PG-009" in decision.failed_rules
