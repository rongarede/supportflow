from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from supportflow.agents.protocols import (
    InvalidStructuredOutput,
    ModelOutput,
    ModelTimeout,
)


class OpenAICompatibleStructuredModel:
    """Optional OpenAI-compatible structured-output adapter for the three agents only."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.environ.get("SUPPORTFLOW_LLM_MODEL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not self.model:
            raise ValueError("SUPPORTFLOW_LLM_MODEL is required for the OpenAI-compatible adapter")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI-compatible adapter")
        self.client = client or self._create_client()

    def _create_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the real-llm extra before using the OpenAI-compatible adapter") from error
        options: dict[str, str] = {"api_key": self.api_key}
        if self.base_url:
            options["base_url"] = self.base_url
        return OpenAI(**options)

    def generate(
        self, role: str, input_payload: dict, output_type: type[ModelOutput]
    ) -> ModelOutput:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are the SupportFlow {role} agent. Return only JSON that matches "
                            "the requested schema; do not make or authorize external actions."
                        ),
                    },
                    {"role": "user", "content": json.dumps(input_payload, sort_keys=True)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_type.__name__,
                        "schema": output_type.model_json_schema(),
                        "strict": True,
                    },
                },
            )
        except TimeoutError as error:
            raise ModelTimeout("provider request timed out") from error
        except Exception as error:
            if type(error).__name__ == "APITimeoutError":
                raise ModelTimeout("provider request timed out") from error
            raise
        try:
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("provider response did not include JSON content")
            return output_type.model_validate_json(content)
        except (AttributeError, IndexError, TypeError, ValidationError, ValueError) as error:
            raise InvalidStructuredOutput("provider response did not match the requested schema") from error
