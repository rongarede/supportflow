from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from supportflow.domain.hashing import proposal_hash
from supportflow.domain.models import (
    ActionProposal,
    ActionType,
    ApprovalMismatch,
    ApprovalRecord,
    Intent,
    ResolutionProposal,
    RiskReview,
    TriageResult,
)
from supportflow.execution.executor import InMemoryExecutor


def _proposal(reply_text: str = "We have submitted your refund request.") -> ResolutionProposal:
    return ResolutionProposal(
        ticket_id="ticket-001",
        reply_text=reply_text,
        evidence_refs=["policy-duplicate-charge-001"],
        uncertainties=[],
        actions=[
            ActionProposal(
                action_type="CREATE_REFUND_REQUEST",
                parameters={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
                reason="A verified duplicate charge may enter refund review.",
                evidence_refs=["policy-duplicate-charge-001"],
                risk_level="medium",
            ),
            ActionProposal(
                action_type="SEND_REPLY",
                parameters={"message": reply_text},
                reason="The customer needs the review status.",
                evidence_refs=["policy-duplicate-charge-001"],
                risk_level="low",
            ),
        ],
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


def test_action_proposal_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        ActionProposal(
            action_type="ISSUE_CASH",
            parameters={},
            reason="Unsafe.",
            evidence_refs=["policy-1"],
            risk_level="high",
        )


def test_approved_contract_exposes_four_intents_and_five_actions() -> None:
    assert {item.value for item in Intent} == {
        "BILLING_QUESTION",
        "REFUND_REQUEST",
        "DUPLICATE_CHARGE",
        "REFUND_STATUS",
    }
    assert {item.value for item in ActionType} == {
        "SEND_REPLY",
        "ADD_TAG",
        "REQUEST_INFORMATION",
        "CREATE_REFUND_REQUEST",
        "ESCALATE_HUMAN",
    }


def test_triage_contract_keeps_extracted_facts_missing_information_and_risk_flags() -> None:
    triage = TriageResult(
        ticket_id="ticket-001",
        intent="REFUND_STATUS",
        confidence=0.91,
        rationale="The customer asks when an existing refund will arrive.",
        urgency="medium",
        extracted_facts={"order_id": "order-100"},
        missing_information=["refund_request_id"],
        risk_flags=["identity_not_verified"],
        route="request_information",
    )

    assert triage.extracted_facts == {"order_id": "order-100"}
    assert triage.missing_information == ["refund_request_id"]
    assert triage.risk_flags == ["identity_not_verified"]


def test_review_contract_requires_auditable_risk_fields() -> None:
    with pytest.raises(ValidationError):
        RiskReview(decision="pass", explanation="Looks safe.")


@pytest.mark.parametrize("missing", ["order_id", "amount", "currency"])
def test_refund_request_requires_each_complete_parameter(missing: str) -> None:
    params = {"order_id": "order-100", "amount": "29.00", "currency": "USD"}
    params.pop(missing)

    with pytest.raises(ValidationError):
        ActionProposal(
            action_type="CREATE_REFUND_REQUEST",
            parameters=params,
            reason="Verified duplicate charge.",
            evidence_refs=["policy-duplicate-charge-001"],
            risk_level="medium",
        )


@pytest.mark.parametrize(
    ("action_type", "parameters"),
    [
        ("SEND_REPLY", {"message": "We are reviewing this."}),
        ("ADD_TAG", {"tag": "refund-status"}),
        ("REQUEST_INFORMATION", {"message": "Please provide the refund request ID."}),
        (
            "CREATE_REFUND_REQUEST",
            {"order_id": "order-100", "amount": "29.00", "currency": "USD"},
        ),
        ("ESCALATE_HUMAN", {"queue": "billing-risk", "summary": "Conflicting policy."}),
    ],
)
def test_every_approved_action_has_reason_evidence_and_risk(
    action_type: str, parameters: dict[str, str]
) -> None:
    action = ActionProposal(
        action_type=action_type,
        parameters=parameters,
        reason="The cited policy supports this bounded action.",
        evidence_refs=["policy-duplicate-charge-001"],
        risk_level="low",
    )

    assert action.parameters == parameters


def test_reply_text_change_changes_canonical_proposal_hash() -> None:
    first = _proposal("The refund request was submitted.")
    second = _proposal("The refund request is pending.")

    assert first.proposal_hash == proposal_hash(first)
    assert second.proposal_hash == proposal_hash(second)
    assert first.proposal_hash != second.proposal_hash


def test_executor_rejects_proposal_mutated_after_exact_hash_approval() -> None:
    proposal = _proposal()
    approval = ApprovalRecord(
        run_id="run-001",
        proposal_hash=proposal.proposal_hash,
        reviewer="portfolio-owner",
        approved_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    proposal.actions[1].params["message"] = "A changed reply must require another approval."

    with pytest.raises(ApprovalMismatch, match="exact reviewed proposal"):
        InMemoryExecutor().execute("run-001", proposal, approval)
