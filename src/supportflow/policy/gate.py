from supportflow.domain.hashing import proposal_hash
from supportflow.domain.models import (
    ActionType,
    EvidenceBundle,
    PolicyDecision,
    ResolutionProposal,
    RiskReview,
    Ticket,
)


def _has_required_headings(action_type: ActionType, cited_headings: set[str]) -> bool:
    if action_type == ActionType.CREATE_REFUND_REQUEST:
        return (
            {"duplicate-charge-verification", "duplicate-charge-refund-request"}
            <= cited_headings
            or "refund-eligibility" in cited_headings
        )
    if action_type == ActionType.SEND_REPLY:
        return bool(
            cited_headings
            & {
                "duplicate-charge-verification",
                "refund-timing",
                "refund-timing-details",
                "refund-eligibility",
            }
        )
    if action_type == ActionType.REQUEST_INFORMATION:
        return bool(
            cited_headings
            & {"required-customer-verification", "payment-failure-required-details"}
        )
    if action_type == ActionType.ESCALATE_HUMAN:
        return bool(
            cited_headings
            & {"human-escalation-conditions", "refund-escalation-conditions"}
        )
    return action_type == ActionType.ADD_TAG and bool(cited_headings)


class PolicyGate:
    def evaluate(
        self, ticket: Ticket, evidence: EvidenceBundle, proposal: ResolutionProposal, review: RiskReview
    ) -> PolicyDecision:
        passed: list[str] = ["PG-001"]
        failed: list[str] = []
        available_evidence = [*evidence.items, *evidence.audit_items]
        evidence_ids = {item.evidence_id for item in available_evidence}
        references_exist = set(proposal.evidence_refs) <= evidence_ids
        if references_exist:
            passed.append("PG-002")
        else:
            failed.append("PG-002")
        referenced = [item for item in available_evidence if item.evidence_id in proposal.evidence_refs]
        if references_exist:
            if all(item.active for item in referenced):
                passed.append("PG-003")
            else:
                failed.append("PG-003")
            heading_by_id = {item.evidence_id: item.heading for item in referenced}
            if all(
                set(action.evidence_refs) <= set(proposal.evidence_refs)
                and set(action.evidence_refs) <= evidence_ids
                and _has_required_headings(
                    action_type=action.action_type,
                    cited_headings={heading_by_id[ref] for ref in action.evidence_refs},
                )
                for action in proposal.actions
            ):
                passed.append("PG-004")
            else:
                failed.append("PG-004")
            cited_headings = {item.heading for item in referenced}
            if {"refund-eligibility", "refund-exclusion"} <= cited_headings:
                failed.append("PG-009")
        if all(action.action_type in set(ActionType) for action in proposal.actions):
            passed.append("PG-005")
        else:
            failed.append("PG-005")
        refund_actions = [action for action in proposal.actions if action.action_type == ActionType.CREATE_REFUND_REQUEST]
        if all(all(action.params.get(key) for key in ("order_id", "amount", "currency")) for action in refund_actions):
            passed.append("PG-006")
        else:
            failed.append("PG-006")
        if review.decision == "pass":
            passed.append("PG-007")
        else:
            failed.append("PG-007")
        if proposal.proposal_hash == proposal_hash(proposal):
            passed.append("PG-008")
        else:
            failed.append("PG-008")
        reply_actions = [
            action for action in proposal.actions if action.action_type == ActionType.SEND_REPLY
        ]
        if not reply_actions or all(
            action.parameters.get("message") == proposal.reply_text
            for action in reply_actions
        ):
            passed.append("PG-010")
        else:
            failed.append("PG-010")
        escalation_failures = {"PG-003", "PG-007", "PG-009"}
        outcome = "allow" if not failed else "escalate" if escalation_failures.intersection(failed) else "block"
        return PolicyDecision(
            outcome=outcome,
            passed_rules=passed,
            failed_rules=failed,
            rationale=(
                "All deterministic policy checks passed."
                if not failed
                else "Policy checks require a human escalation."
                if outcome == "escalate"
                else "Policy checks blocked execution."
            ),
        )
