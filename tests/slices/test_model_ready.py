from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys

import pytest

from supportflow.agents.protocols import InvalidStructuredOutput, ModelTimeout
from supportflow.domain.models import TriageResult
from supportflow.workflow.service import SupportFlowService


class TimingOutModel:
    """A local model double: no network or credentials are involved."""

    def generate(self, role, input_payload, output_type):
        raise ModelTimeout("simulated timeout")


class MalformedModel:
    """A local model double that reports a failed structured-output validation."""

    def generate(self, role, input_payload, output_type):
        raise InvalidStructuredOutput("simulated malformed JSON")


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


def test_openai_compatible_adapter_validates_structured_response_without_network() -> None:
    """Catches an adapter that does not return a typed result from provider JSON."""
    from supportflow.agents.openai_adapter import OpenAICompatibleStructuredModel

    class FakeCompletions:
        def create(self, **_kwargs):
            class Message:
                content = '{"ticket_id":"ticket-duplicate-001","intent":"DUPLICATE_CHARGE","confidence":0.99,"rationale":"duplicate charge"}'

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
