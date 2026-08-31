from supportflow.agents.protocols import StructuredModel
from supportflow.domain.models import EvidenceBundle, ResolutionProposal, Ticket


class ResolutionAgent:
    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def run(self, ticket: Ticket, evidence: EvidenceBundle) -> ResolutionProposal:
        return self.model.generate(
            "resolution",
            {"ticket_id": ticket.ticket_id, "evidence_ids": [item.evidence_id for item in evidence.items], "attempt": 1},
            ResolutionProposal,
        )
