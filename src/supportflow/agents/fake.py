from __future__ import annotations

from pydantic import BaseModel

from supportflow.agents.protocols import ModelOutput


class ModelExhausted(RuntimeError):
    """Raised when a deterministic fixture has no response for a model attempt."""


class FakeStructuredModel:
    """Deterministic portfolio substitute for an external structured-output LLM."""

    def __init__(self, responses: dict[tuple[str, str, int], BaseModel]) -> None:
        self.responses = responses

    def generate(self, role: str, input_payload: dict, output_type: type[ModelOutput]) -> ModelOutput:
        ticket_id = input_payload["ticket_id"]
        attempt = input_payload.get("attempt", 1)
        try:
            result = self.responses[(role, ticket_id, attempt)]
        except KeyError as error:
            raise ModelExhausted(
                f"No fake response remains for {role} on ticket {ticket_id} attempt {attempt}"
            ) from error
        if not isinstance(result, output_type):
            raise TypeError(f"Fake response for {role} is not {output_type.__name__}")
        return result
