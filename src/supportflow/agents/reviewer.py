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
                "ticket": ticket.model_dump(mode="json"),
                "proposal": proposal.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "attempt": 1,
            },
            RiskReview,
        )
