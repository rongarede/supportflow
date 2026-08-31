from supportflow.agents.protocols import StructuredModel
from supportflow.domain.models import EvidenceBundle, ResolutionProposal, Ticket


class ResolutionAgent:
    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def run(self, ticket: Ticket, evidence: EvidenceBundle) -> ResolutionProposal:
        return self.model.generate(
            "resolution",
            {
                "ticket_id": ticket.ticket_id,
                "ticket": ticket.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "attempt": 1,
            },
            ResolutionProposal,
        )
