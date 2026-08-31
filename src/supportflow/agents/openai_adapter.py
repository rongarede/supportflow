from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

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


class TriageFactOutput(ProviderDTO):
    key: str
    value: str


class TriageOutput(ProviderDTO):
    ticket_id: str
    intent: Literal[
        "BILLING_QUESTION", "REFUND_REQUEST", "DUPLICATE_CHARGE", "REFUND_STATUS"
    ]
    confidence: float
    rationale: str
    urgency: Literal["low", "medium", "high"]
    extracted_facts: list[TriageFactOutput]
    missing_information: list[str]
    risk_flags: list[str]
    route: Literal["continue", "request_information", "escalate_human"]


class ResolutionActionOutput(ProviderDTO):
    """Flat action DTO avoids provider-incompatible oneOf/discriminator schemas."""

    action_type: str
    order_id: str
    amount: str
    currency: str
    message: str
    tag: str
    queue: str
    summary: str
    reason: str
    evidence_refs: list[str]
    risk_level: Literal["low", "medium", "high"]


class ResolutionOutput(ProviderDTO):
    ticket_id: str
    reply_text: str
    evidence_refs: list[str]
    actions: list[ResolutionActionOutput]
    uncertainties: list[str]
    created_at: str


class RiskReviewOutput(ProviderDTO):
    decision: Literal["pass", "revise", "escalate"]
    risk_flags: list[str]
    unsupported_claims: list[str]
    required_changes: list[str]
    explanation: str


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
    raw = dto.model_json_schema()
    definitions = raw.pop("$defs", {})

    def inline(schema: dict[str, Any]) -> dict[str, Any]:
        reference = schema.get("$ref")
        if reference:
            return inline(definitions[reference.removeprefix("#/$defs/")])
        result = {
            key: value
            for key, value in schema.items()
            if key in {"additionalProperties", "const", "required", "title", "type"}
        }
        if "properties" in schema:
            result["properties"] = {
                name: inline(child) for name, child in schema["properties"].items()
            }
        if isinstance(schema.get("items"), dict):
            result["items"] = inline(schema["items"])
        return result

    return inline(raw)


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
        if output_type is ResolutionProposal and (
            not isinstance(payload, dict) or not isinstance(payload.get("actions"), list)
        ):
            raise InvalidStructuredOutput("provider response did not include an actions list")
        dto_type = OUTPUT_DTOS[output_type]
        try:
            dto = dto_type.model_validate(payload)
        except ValidationError as error:
            raise InvalidStructuredOutput("provider response did not match the requested schema") from error
        if output_type is TriageResult:
            triage = TriageOutput.model_validate(dto)
            return TriageResult(
                ticket_id=triage.ticket_id,
                intent=triage.intent,
                confidence=triage.confidence,
                rationale=triage.rationale,
                urgency=triage.urgency,
                extracted_facts={item.key: item.value for item in triage.extracted_facts},
                missing_information=triage.missing_information,
                risk_flags=triage.risk_flags,
                route=triage.route,
            )
        if output_type is ResolutionProposal:
            return OpenAICompatibleStructuredModel._resolution_from_dto(dto)
        return output_type.model_validate(dto.model_dump(mode="json"))

    @staticmethod
    def _resolution_from_dto(dto: ProviderDTO) -> ResolutionProposal:
        resolution = ResolutionOutput.model_validate(dto)
        actions = []
        for action in resolution.actions:
            if action.action_type == "CREATE_REFUND_REQUEST":
                if not all((action.order_id, action.amount, action.currency)):
                    raise InvalidAction("provider response proposed an incomplete refund action")
                actions.append(
                    {
                        "action_type": action.action_type,
                        "parameters": {
                            "order_id": action.order_id,
                            "amount": action.amount,
                            "currency": action.currency,
                        },
                        "reason": action.reason,
                        "evidence_refs": action.evidence_refs,
                        "risk_level": action.risk_level,
                    }
                )
            elif action.action_type == "SEND_REPLY":
                if not action.message:
                    raise InvalidAction("provider response proposed an empty reply action")
                actions.append(
                    {
                        "action_type": action.action_type,
                        "parameters": {"message": action.message},
                        "reason": action.reason,
                        "evidence_refs": action.evidence_refs,
                        "risk_level": action.risk_level,
                    }
                )
            elif action.action_type == "ADD_TAG":
                actions.append(
                    {
                        "action_type": action.action_type,
                        "parameters": {"tag": action.tag},
                        "reason": action.reason,
                        "evidence_refs": action.evidence_refs,
                        "risk_level": action.risk_level,
                    }
                )
            elif action.action_type == "REQUEST_INFORMATION":
                actions.append(
                    {
                        "action_type": action.action_type,
                        "parameters": {"message": action.message},
                        "reason": action.reason,
                        "evidence_refs": action.evidence_refs,
                        "risk_level": action.risk_level,
                    }
                )
            elif action.action_type == "ESCALATE_HUMAN":
                actions.append(
                    {
                        "action_type": action.action_type,
                        "parameters": {"queue": action.queue, "summary": action.summary},
                        "reason": action.reason,
                        "evidence_refs": action.evidence_refs,
                        "risk_level": action.risk_level,
                    }
                )
            else:
                raise InvalidAction("provider response proposed an unsupported action")
        try:
            return ResolutionProposal(
                ticket_id=resolution.ticket_id,
                reply_text=resolution.reply_text,
                evidence_refs=resolution.evidence_refs,
                actions=actions,
                uncertainties=resolution.uncertainties,
                created_at=resolution.created_at,
            )
        except ValidationError as error:
            raise InvalidAction("provider response proposed an invalid action") from error

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
