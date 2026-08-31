from supportflow.agents.protocols import StructuredModel
from supportflow.domain.models import EvidenceBundle, ResolutionProposal, RiskReview, Ticket


class RiskReviewerAgent:
    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def run(self, ticket: Ticket, proposal: ResolutionProposal, evidence: EvidenceBundle) -> RiskReview:
        return self.model.generate(
            "reviewer",
            {
                "ticket_id": ticket.ticket_id,
                "proposal_hash": proposal.proposal_hash,
                "evidence_ids": [item.evidence_id for item in evidence.items],
                "attempt": 1,
            },
            RiskReview,
        )
