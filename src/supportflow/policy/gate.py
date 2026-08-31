from supportflow.domain.hashing import proposal_hash
from supportflow.domain.models import (
    ActionType,
    EvidenceBundle,
    PolicyDecision,
    ResolutionProposal,
    RiskReview,
    Ticket,
)


class PolicyGate:
    def evaluate(
        self, ticket: Ticket, evidence: EvidenceBundle, proposal: ResolutionProposal, review: RiskReview
    ) -> PolicyDecision:
        passed: list[str] = ["PG-001"]
        failed: list[str] = []
        evidence_ids = {item.evidence_id for item in evidence.items}
        if set(proposal.evidence_refs) <= evidence_ids:
            passed.append("PG-002")
        else:
            failed.append("PG-002")
        referenced = [item for item in evidence.items if item.evidence_id in proposal.evidence_refs]
        if not set(proposal.evidence_refs) <= evidence_ids:
            # An absent reference is already PG-002; PG-003 only evaluates existing citations.
            passed.append("PG-003")
        elif referenced and all(item.active for item in referenced):
            passed.append("PG-003")
        else:
            failed.append("PG-003")
        if proposal.evidence_refs:
            passed.append("PG-004")
        else:
            failed.append("PG-004")
        if all(action.action_type in set(ActionType) for action in proposal.actions):
            passed.append("PG-005")
        else:
            failed.append("PG-005")
        refund_actions = [action for action in proposal.actions if action.action_type == ActionType.CREATE_REFUND_REQUEST]
        if all(all(action.params.get(key) for key in ("order_id", "amount", "currency")) for action in refund_actions):
            passed.append("PG-006")
        else:
            failed.append("PG-006")
        if not review.escalated:
            passed.append("PG-007")
        else:
            failed.append("PG-007")
        if proposal.proposal_hash == proposal_hash(proposal):
            passed.append("PG-008")
        else:
            failed.append("PG-008")
        return PolicyDecision(
            outcome="allow" if not failed else "block",
            passed_rules=passed,
            failed_rules=failed,
            rationale="All deterministic policy checks passed." if not failed else "Policy checks blocked execution.",
        )
