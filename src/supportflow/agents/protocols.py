from typing import Protocol, TypeVar

from pydantic import BaseModel


ModelOutput = TypeVar("ModelOutput", bound=BaseModel)


class StructuredModel(Protocol):
    def generate(self, role: str, input_payload: dict, output_type: type[ModelOutput]) -> ModelOutput: ...
