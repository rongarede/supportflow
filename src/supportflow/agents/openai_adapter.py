from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from supportflow.agents.protocols import (
    InvalidAction,
    InvalidStructuredOutput,
    ModelOutput,
    ModelTimeout,
)
from supportflow.domain.models import ResolutionProposal, RiskReview, TriageResult


class ProviderDTO(BaseModel):
    """Provider-facing DTOs: every field is required and every object is closed."""

    model_config = ConfigDict(extra="forbid")


class TriageOutput(ProviderDTO):
    ticket_id: str
    intent: Literal["DUPLICATE_CHARGE"]
    confidence: float
    rationale: str
    missing_fields: list[str]


class RefundParams(ProviderDTO):
    order_id: str
    amount: str
    currency: str


class ReplyParams(ProviderDTO):
    message: str


class RefundAction(ProviderDTO):
    action_type: Literal["CREATE_REFUND_REQUEST"]
    params: RefundParams


class ReplyAction(ProviderDTO):
    action_type: Literal["SEND_REPLY"]
    params: ReplyParams


ClosedAction = Annotated[Union[RefundAction, ReplyAction], Field(discriminator="action_type")]


class ResolutionOutput(ProviderDTO):
    ticket_id: str
    evidence_refs: list[str]
    actions: list[ClosedAction]
    created_at: str


class RiskReviewOutput(ProviderDTO):
    escalated: bool
    rationale: str


OUTPUT_DTOS: dict[type[ModelOutput], type[ProviderDTO]] = {
    TriageResult: TriageOutput,
    ResolutionProposal: ResolutionOutput,
    RiskReview: RiskReviewOutput,
}


def strict_schema_for(output_type: type[ModelOutput]) -> dict[str, Any]:
    """Return the subset of output schemas that is strict-JSON-Schema compatible."""
    try:
        dto = OUTPUT_DTOS[output_type]
    except KeyError as error:
        raise ValueError(f"Unsupported structured output type: {output_type.__name__}") from error
    return dto.model_json_schema()


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
        options: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.base_url:
            options["base_url"] = self.base_url
        return OpenAI(**options)

    @staticmethod
    def _is_provider_timeout(error: Exception) -> bool:
        try:
            from openai import APITimeoutError
        except ImportError:
            return False
        return isinstance(error, APITimeoutError)

    @staticmethod
    def _parse_output(content: str, output_type: type[ModelOutput]) -> ModelOutput:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise InvalidStructuredOutput("provider response was not valid JSON") from error
        if output_type is ResolutionProposal:
            actions = payload.get("actions") if isinstance(payload, dict) else None
            if not isinstance(actions, list) or any(
                not isinstance(action, dict)
                or action.get("action_type") not in {"CREATE_REFUND_REQUEST", "SEND_REPLY"}
                for action in actions
            ):
                raise InvalidAction("provider response proposed an unsupported action")
        dto_type = OUTPUT_DTOS[output_type]
        try:
            dto = dto_type.model_validate(payload)
            return output_type.model_validate(dto.model_dump(mode="json"))
        except ValidationError as error:
            if output_type is ResolutionProposal and any(
                issue["loc"] and issue["loc"][0] == "actions" for issue in error.errors()
            ):
                raise InvalidAction("provider response proposed an invalid action") from error
            raise InvalidStructuredOutput("provider response did not match the requested schema") from error

    def generate(
        self, role: str, input_payload: dict, output_type: type[ModelOutput]
    ) -> ModelOutput:
        schema = strict_schema_for(output_type)
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
                        "name": OUTPUT_DTOS[output_type].__name__,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except TimeoutError as error:
            raise ModelTimeout("provider request timed out") from error
        except Exception as error:
            if self._is_provider_timeout(error):
                raise ModelTimeout("provider request timed out") from error
            raise
        try:
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("provider response did not include JSON content")
        except (AttributeError, IndexError, TypeError) as error:
            raise InvalidStructuredOutput("provider response did not include JSON content") from error
        return self._parse_output(content, output_type)
