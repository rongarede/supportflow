from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from supportflow.agents.protocols import (
    InvalidAction,
    InvalidStructuredOutput,
    ModelTimeout,
)
from supportflow.domain.models import ResolutionProposal, RiskReview, TriageResult
from supportflow.workflow.service import SupportFlowService


class TimingOutModel:
    """A local model double: no network or credentials are involved."""

    def generate(self, role, input_payload, output_type):
        raise ModelTimeout("simulated timeout")


class MalformedModel:
    """A local model double that reports a failed structured-output validation."""

    def generate(self, role, input_payload, output_type):
        raise InvalidStructuredOutput("simulated malformed JSON")


class InvalidActionModel:
    """A local double that rejects the proposed action without a retry."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def generate(self, role, input_payload, output_type):
        if role == "resolution":
            raise InvalidAction("simulated invalid action")
        return self.delegate.generate(role, input_payload, output_type)


class ReviewTimingOutModel:
    """A local double that only exhausts the final model-backed node."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def generate(self, role, input_payload, output_type):
        if role == "reviewer":
            raise ModelTimeout("simulated review timeout")
        return self.delegate.generate(role, input_payload, output_type)


class CountingTimeoutModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, role, input_payload, output_type):
        self.calls += 1
        raise ModelTimeout("simulated durable timeout")


class MalformedResolutionModel:
    """A model double that fails only resolution with retryable malformed output."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.resolution_calls = 0

    def generate(self, role, input_payload, output_type):
        if role == "resolution":
            self.resolution_calls += 1
            raise InvalidStructuredOutput("simulated missing action list")
        return self.delegate.generate(role, input_payload, output_type)


@pytest.fixture
def service_factory():
    def build(model):
        service = SupportFlowService.demo(as_of=datetime(2026, 8, 31, tzinfo=UTC))
        if model is not None:
            service.graph.triage.model = model
            service.graph.resolution.model = model
            service.graph.reviewer.model = model
        return service

    return build


@pytest.fixture
def fake_model(service_factory):
    return service_factory(None).graph.triage.model


def test_model_adapter_does_not_replace_safety_components(
    service_factory, fake_model, duplicate_ticket
) -> None:
    """Catches an adapter path that swaps deterministic retrieval, policy, or execution."""
    service = service_factory(fake_model)

    result = service.submit(duplicate_ticket)

    assert result.current_state == "WAITING_APPROVAL"
    assert service.graph.retriever.__class__.__name__ == "RagRetriever"
    assert service.graph.gate.__class__.__name__ == "PolicyGate"
    assert service.graph.executor.__class__.__name__ == "DurableExecutor"


@pytest.mark.parametrize("model", [TimingOutModel(), MalformedModel()])
def test_two_retryable_model_failures_stop_without_execution(
    service_factory, model, duplicate_ticket
) -> None:
    """Catches an exhausted retry route that advances to retrieval or execution."""
    result = service_factory(model).submit(duplicate_ticket)

    assert result.current_state == "NEEDS_ATTENTION"
    assert result.node_attempts["triage"] == 2
    assert result.execution_results == []
    assert result.errors[-1].code == "MODEL_EXHAUSTED"


def test_retrieval_failure_is_not_reported_as_model_exhaustion(
    service_factory, fake_model, duplicate_ticket
) -> None:
    """Catches a retrieval ValueError mislabeled as a model failure."""
    service = service_factory(fake_model)
    service.graph.retriever.chunks = []

    result = service.submit(duplicate_ticket)

    assert result.current_state == "NEEDS_ATTENTION"
    assert result.errors[-1].code == "RETRIEVAL_UNAVAILABLE"
    assert result.execution_results == []


def test_invalid_action_stops_without_retry_or_execution(
    service_factory, fake_model, duplicate_ticket
) -> None:
    """Catches invalid action output being retried as if it were malformed JSON."""
    result = service_factory(InvalidActionModel(fake_model)).submit(duplicate_ticket)

    assert result.current_state == "NEEDS_ATTENTION"
    assert result.node_attempts["resolve"] == 1
    assert result.errors[-1].code == "INVALID_ACTION"
    assert result.execution_results == []


def test_structurally_malformed_resolution_retries_once_before_stopping(
    service_factory, fake_model, duplicate_ticket
) -> None:
    """Catches malformed resolution output being treated as a non-retryable action error."""
    model = MalformedResolutionModel(fake_model)
    result = service_factory(model).submit(duplicate_ticket)

    assert model.resolution_calls == 2
    assert result.node_attempts["resolve"] == 2
    assert result.errors[-1].code == "MODEL_EXHAUSTED"
    assert result.execution_results == []


def test_review_model_exhaustion_stops_before_policy(
    service_factory, fake_model, duplicate_ticket
) -> None:
    """Catches a terminal review failure that still enters the policy node."""
    result = service_factory(ReviewTimingOutModel(fake_model)).submit(duplicate_ticket)

    assert result.current_state == "NEEDS_ATTENTION"
    assert result.node_attempts["review"] == 2
    assert result.policy_decision is None
    assert result.execution_results == []


def test_durable_retry_budget_is_cumulative_after_reopen(tmp_path, monkeypatch, duplicate_ticket) -> None:
    """Catches a reopened run resetting a partially used model retry budget."""
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    runtime = tmp_path / "runtime"
    model = CountingTimeoutModel()
    first = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC), runtime_directory=runtime, model=model
    )
    run_id = "retry-budget-reopen"
    first.repository.create_run(run_id, duplicate_ticket)
    assert first.repository.claim_model_attempt(run_id, "triage") == 1

    reopened = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC), runtime_directory=runtime, model=model
    )
    reopened.graph.compiled.invoke(
        {"run_id": run_id, "ticket": duplicate_ticket, "trace": []}, reopened._config(run_id)
    )
    result = reopened.snapshot(run_id)

    assert model.calls == 1
    assert result.node_attempts["triage"] == 2
    assert result.current_state == "NEEDS_ATTENTION"


def test_openai_compatible_adapter_validates_structured_response_without_network() -> None:
    """Catches an adapter that does not return a typed result from provider JSON."""
    from supportflow.agents.openai_adapter import OpenAICompatibleStructuredModel

    class FakeCompletions:
        def create(self, **_kwargs):
            class Message:
                content = '{"ticket_id":"ticket-duplicate-001","intent":"DUPLICATE_CHARGE","confidence":0.99,"rationale":"duplicate charge","urgency":"medium","extracted_facts":[{"key":"order_id","value":"order-100"}],"missing_information":[],"risk_flags":[],"route":"continue"}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    adapter = OpenAICompatibleStructuredModel(
        model="local-test-model", api_key="test-key", client=FakeClient()
    )

    result = adapter.generate(
        "triage",
        {"ticket_id": "ticket-duplicate-001", "subject": "Charged twice", "body": "Same order billed twice."},
        TriageResult,
    )

    assert result.intent == "DUPLICATE_CHARGE"


def test_openai_adapter_emits_closed_required_schemas_for_all_agent_outputs() -> None:
    """Catches schemas outside the advertised OpenAI strict structured-output subset."""
    from supportflow.agents.openai_adapter import strict_schema_for

    def assert_strict(schema):
        allowed_keywords = {
            "additionalProperties",
            "const",
            "items",
            "properties",
            "required",
            "title",
            "type",
        }
        assert set(schema) <= allowed_keywords
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        for child in schema.get("properties", {}).values():
            assert_strict(child)
        if isinstance(schema.get("items"), dict):
            assert_strict(schema["items"])

    for output_type in (TriageResult, ResolutionProposal, RiskReview):
        assert_strict(strict_schema_for(output_type))


def test_openai_adapter_classifies_missing_resolution_actions_as_retryable_malformed_output() -> None:
    """Catches a missing or non-list actions field being mislabeled as an invalid action."""
    from supportflow.agents.openai_adapter import OpenAICompatibleStructuredModel

    class FakeCompletions:
        def create(self, **_kwargs):
            class Message:
                content = '{"ticket_id":"ticket-duplicate-001","evidence_refs":["policy-duplicate-charge-001"],"created_at":"2026-08-31T00:00:00Z"}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    adapter = OpenAICompatibleStructuredModel(
        model="local-test-model", api_key="test-key", client=FakeClient()
    )

    with pytest.raises(InvalidStructuredOutput):
        adapter.generate("resolution", {"ticket": {}}, ResolutionProposal)


def test_openai_adapter_rejects_an_invalid_action_without_reclassifying_it_as_malformed() -> None:
    """Catches a closed action violation that enters the retryable malformed-output path."""
    from supportflow.agents.openai_adapter import OpenAICompatibleStructuredModel

    class FakeCompletions:
        def create(self, **_kwargs):
            class Message:
                content = '{"ticket_id":"ticket-duplicate-001","reply_text":"Unsafe action.","evidence_refs":["policy-duplicate-charge-001"],"actions":[{"action_type":"DELETE_ACCOUNT","order_id":"","amount":"","currency":"","message":"","tag":"","queue":"","summary":"","reason":"Unsupported action.","evidence_refs":["policy-duplicate-charge-001"],"risk_level":"high"}],"uncertainties":[],"created_at":"2026-08-31T00:00:00Z"}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    adapter = OpenAICompatibleStructuredModel(
        model="local-test-model", api_key="test-key", client=FakeClient()
    )

    with pytest.raises(InvalidAction):
        adapter.generate("resolution", {"ticket": {}}, ResolutionProposal)


def test_openai_adapter_passes_full_ticket_evidence_and_proposal_to_all_three_calls(
    duplicate_ticket
) -> None:
    """Catches resolution or review prompts that omit the information needed for a valid proposal."""
    from supportflow.agents.openai_adapter import OpenAICompatibleStructuredModel

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            schema_name = kwargs["response_format"]["json_schema"]["name"]
            content_by_schema = {
                "TriageOutput": '{"ticket_id":"ticket-duplicate-001","intent":"DUPLICATE_CHARGE","confidence":0.99,"rationale":"duplicate charge","urgency":"medium","extracted_facts":[{"key":"order_id","value":"order-100"},{"key":"amount","value":"29.00"},{"key":"currency","value":"USD"}],"missing_information":[],"risk_flags":[],"route":"continue"}',
                "ResolutionOutput": '{"ticket_id":"ticket-duplicate-001","reply_text":"A refund request was created.","evidence_refs":["policy-duplicate-charge-001","policy-refund-request-001","policy-refund-request-002"],"actions":[{"action_type":"CREATE_REFUND_REQUEST","order_id":"order-100","amount":"29.00","currency":"USD","message":"","tag":"","queue":"","summary":"","reason":"Verified duplicate charge.","evidence_refs":["policy-duplicate-charge-001","policy-refund-request-001"],"risk_level":"medium"},{"action_type":"SEND_REPLY","order_id":"","amount":"","currency":"","message":"A refund request was created.","tag":"","queue":"","summary":"","reason":"Share a bounded status.","evidence_refs":["policy-refund-request-002"],"risk_level":"low"}],"uncertainties":[],"created_at":"2026-08-31T00:00:00Z"}',
                "RiskReviewOutput": '{"decision":"pass","risk_flags":[],"unsupported_claims":[],"required_changes":[],"explanation":"Evidence and actions are constrained."}',
            }

            class Message:
                content = content_by_schema[schema_name]

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    adapter = OpenAICompatibleStructuredModel(
        model="local-test-model", api_key="test-key", client=FakeClient()
    )
    service = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC), model=adapter
    )

    result = service.submit(duplicate_ticket)
    payloads = [json.loads(call["messages"][1]["content"]) for call in calls]

    assert result.current_state == "WAITING_APPROVAL"
    assert [call["response_format"]["json_schema"]["name"] for call in calls] == [
        "TriageOutput",
        "ResolutionOutput",
        "RiskReviewOutput",
    ]
    assert payloads[0]["ticket"]["body"] == duplicate_ticket.body
    assert payloads[1]["ticket"]["order_id"] == "order-100"
    assert payloads[1]["evidence"]["items"]
    assert payloads[2]["ticket"]["amount"] == "29.00"
    assert payloads[2]["evidence"]["items"]
    assert payloads[2]["proposal"]["actions"]


def test_restart_demo_accepts_a_dedicated_final_evidence_runtime(tmp_path) -> None:
    """Catches a final verification command that is rejected despite using an isolated runtime."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "supportflow.cli",
            "demo-restart",
            "--runtime",
            str(tmp_path / "final-restart"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "stored_side_effects: 2" in completed.stdout
