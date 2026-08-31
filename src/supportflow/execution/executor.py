from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from supportflow.domain.hashing import proposal_hash
from supportflow.domain.models import (
    ActionProposal,
    ApprovalMismatch,
    ApprovalRecord,
    ExecutionResult,
    ResolutionProposal,
)
from supportflow.storage.repositories import SupportFlowRepository


def action_idempotency_key(
    ticket_id: str,
    canonical_proposal_hash: str,
    action: ActionProposal,
) -> str:
    canonical_parameters = json.dumps(
        action.params, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    value = (
        ticket_id
        + canonical_proposal_hash
        + action.action_type.value
        + canonical_parameters
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_approval(
    run_id: str, proposal: ResolutionProposal, approval: ApprovalRecord | None
) -> str:
    canonical_hash = proposal_hash(proposal)
    if (
        approval is None
        or approval.run_id != run_id
        or approval.proposal_hash != canonical_hash
        or proposal.proposal_hash != canonical_hash
    ):
        raise ApprovalMismatch("Execution requires approval for the exact reviewed proposal")
    return canonical_hash


def _result(
    run_id: str,
    proposal: ResolutionProposal,
    action: ActionProposal,
    index: int,
) -> ExecutionResult:
    return ExecutionResult(
        idempotency_key=action_idempotency_key(
            proposal.ticket_id, proposal.proposal_hash, action
        ),
        action_type=action.action_type,
        status="succeeded",
        reference=f"simulated-{run_id}-{index + 1}",
        simulated_payload=action.params,
        executed_at=datetime.now(UTC),
    )


class InMemoryExecutor:
    def __init__(self) -> None:
        self._completed: dict[str, ExecutionResult] = {}

    def execute(
        self, run_id: str, proposal: ResolutionProposal, approval: ApprovalRecord | None
    ) -> list[ExecutionResult]:
        _verify_approval(run_id, proposal, approval)
        results: list[ExecutionResult] = []
        for index, action in enumerate(proposal.actions):
            candidate = _result(run_id, proposal, action, index)
            existing = self._completed.get(candidate.idempotency_key)
            if existing is not None:
                results.append(existing.model_copy(update={"status": "skipped_duplicate"}))
            else:
                self._completed[candidate.idempotency_key] = candidate
                results.append(candidate)
        return results


class DurableExecutor:
    def __init__(self, repository: SupportFlowRepository) -> None:
        self.repository = repository

    def execute(
        self, run_id: str, proposal: ResolutionProposal, approval: ApprovalRecord | None
    ) -> list[ExecutionResult]:
        canonical_hash = _verify_approval(run_id, proposal, approval)
        results: list[ExecutionResult] = []
        for index, action in enumerate(proposal.actions):
            candidate = _result(run_id, proposal, action, index)
            stored, _ = self.repository.store_execution(
                run_id, canonical_hash, candidate
            )
            results.append(stored)
        return results
