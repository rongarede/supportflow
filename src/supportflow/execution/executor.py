from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from typing import Callable

from supportflow.domain.hashing import proposal_hash
from supportflow.domain.models import (
    ActionProposal,
    ApprovalMismatch,
    ApprovalRecord,
    ExecutionResult,
    ResolutionProposal,
)
from supportflow.storage.repositories import SupportFlowRepository


class TransientExecutionError(RuntimeError):
    """A retryable failure from the simulated executor boundary."""


class ExecutionExhausted(RuntimeError):
    """Carries the auditable results accumulated before the retry budget ended."""

    def __init__(self, results: list[ExecutionResult], attempts: int) -> None:
        self.results = results
        self.attempts = attempts
        super().__init__(f"simulated execution exhausted after {attempts} attempts")


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
    def __init__(
        self,
        repository: SupportFlowRepository,
        *,
        action_runner: Callable[[ExecutionResult, int, int], ExecutionResult]
        | None = None,
        crash_hook: Callable[[str, int, ExecutionResult], None] | None = None,
    ) -> None:
        self.repository = repository
        self.action_runner = action_runner or (
            lambda candidate, _index, _attempt: candidate
        )
        self.crash_hook = crash_hook

    def execute(
        self, run_id: str, proposal: ResolutionProposal, approval: ApprovalRecord | None
    ) -> list[ExecutionResult]:
        canonical_hash = _verify_approval(run_id, proposal, approval)
        results: list[ExecutionResult] = []
        for index, action in enumerate(proposal.actions):
            candidate = _result(run_id, proposal, action, index)
            existing = self.repository.get_execution(candidate.idempotency_key)
            if existing is not None:
                results.append(existing.model_copy(update={"status": "skipped_duplicate"}))
                continue
            if self.crash_hook is not None:
                self.crash_hook("before_action", index, candidate)
            operation_key = f"execute:{candidate.idempotency_key}"
            while True:
                attempt = self.repository.claim_operation_attempt(
                    run_id, operation_key, max_attempts=3
                )
                if attempt is None:
                    attempt = max(
                        3, self.repository.operation_attempts(run_id, operation_key)
                    )
                    failed = candidate.model_copy(
                        update={
                            "status": "failed",
                            "error": "Transient simulated execution failed after bounded retries.",
                        }
                    )
                    raise ExecutionExhausted([*results, failed], attempt)
                try:
                    attempted = self.action_runner(candidate, index, attempt)
                    stored, _ = self.repository.store_execution(
                        run_id, canonical_hash, attempted
                    )
                except (TransientExecutionError, sqlite3.OperationalError) as error:
                    error_type = type(error).__name__
                    self.repository.mark_operation_result(
                        run_id,
                        operation_key,
                        status="failed",
                        error_type=error_type,
                    )
                    self.repository.trace.record_operation_retry(
                        run_id, "execution", attempt, error_type
                    )
                    if attempt < 3:
                        continue
                    failed = candidate.model_copy(
                        update={
                            "status": "failed",
                            "error": "Transient simulated execution failed after bounded retries.",
                        }
                    )
                    raise ExecutionExhausted([*results, failed], attempt) from error
                self.repository.mark_operation_result(
                    run_id, operation_key, status="succeeded"
                )
                results.append(stored)
                if self.crash_hook is not None:
                    self.crash_hook("after_action", index, stored)
                break
        return results
