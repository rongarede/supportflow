from supportflow.domain.models import (
    ApprovalMismatch,
    ApprovalRecord,
    ExecutionResult,
    ResolutionProposal,
)


class InMemoryExecutor:
    def __init__(self) -> None:
        self._completed: dict[tuple[str, str], list[ExecutionResult]] = {}

    def execute(
        self, run_id: str, proposal: ResolutionProposal, approval: ApprovalRecord | None
    ) -> list[ExecutionResult]:
        if approval is None or approval.run_id != run_id or approval.proposal_hash != proposal.proposal_hash:
            raise ApprovalMismatch("Execution requires approval for the exact reviewed proposal")
        key = (run_id, proposal.proposal_hash)
        if key not in self._completed:
            self._completed[key] = [
                ExecutionResult(
                    action_type=action.action_type,
                    status="SIMULATED_SUCCESS",
                    reference=f"simulated-{run_id}-{index + 1}",
                )
                for index, action in enumerate(proposal.actions)
            ]
        return self._completed[key]
