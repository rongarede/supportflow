from typing import Protocol, TypeVar

from pydantic import BaseModel


ModelOutput = TypeVar("ModelOutput", bound=BaseModel)


class ModelTimeout(RuntimeError):
    """Raised when a model adapter cannot return before its configured timeout."""


class InvalidStructuredOutput(RuntimeError):
    """Raised when a model response cannot satisfy the requested Pydantic schema."""


class ModelExhausted(RuntimeError):
    """Raised after the workflow has used its bounded model-failure budget."""

    def __init__(self, node_name: str, attempts: int) -> None:
        self.node_name = node_name
        self.attempts = attempts
        super().__init__(f"{node_name} model output exhausted after {attempts} attempt(s)")


class StructuredModel(Protocol):
    def generate(self, role: str, input_payload: dict, output_type: type[ModelOutput]) -> ModelOutput: ...
