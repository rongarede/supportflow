from supportflow.agents.protocols import StructuredModel
from supportflow.domain.models import Ticket, TriageResult


class TriageAgent:
    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def run(self, ticket: Ticket) -> TriageResult:
        return self.model.generate(
            "triage",
            {"ticket_id": ticket.ticket_id, "subject": ticket.subject, "body": ticket.body, "attempt": 1},
            TriageResult,
        )
