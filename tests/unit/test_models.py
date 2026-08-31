from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from supportflow.domain.hashing import proposal_hash
from supportflow.domain.models import (
    ActionProposal,
    ApprovalMismatch,
    ApprovalRecord,
    ResolutionProposal,
)
from supportflow.execution.executor import InMemoryExecutor


def _proposal(reply_text: str = "We have submitted your refund request.") -> ResolutionProposal:
    return ResolutionProposal(
        ticket_id="ticket-001",
        evidence_refs=["policy-duplicate-charge-001"],
        actions=[
            ActionProposal(
                action_type="CREATE_REFUND_REQUEST",
                params={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
            ),
            ActionProposal(action_type="SEND_REPLY", params={"message": reply_text}),
        ],
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


def test_action_proposal_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        ActionProposal(action_type="ISSUE_CASH", params={})


@pytest.mark.parametrize("missing", ["order_id", "amount", "currency"])
def test_refund_request_requires_each_complete_parameter(missing: str) -> None:
    params = {"order_id": "order-100", "amount": "29.00", "currency": "USD"}
    params.pop(missing)

    with pytest.raises(ValidationError):
        ActionProposal(action_type="CREATE_REFUND_REQUEST", params=params)


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
