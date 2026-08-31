from datetime import UTC, datetime
from typing import Callable, TypeVar

from pydantic import BaseModel

from supportflow.agents.protocols import (
    InvalidStructuredOutput,
    ModelExhausted,
    ModelTimeout,
)
from supportflow.domain.models import TraceEvent
from supportflow.storage.repositories import SupportFlowRepository


ModelResult = TypeVar("ModelResult", bound=BaseModel)


def trace(stage: str, detail: str) -> TraceEvent:
    return TraceEvent(stage=stage, detail=detail, occurred_at=datetime.now(UTC))


def run_model_node(
    call: Callable[[], ModelResult],
    *,
    node_name: str,
    run_id: str,
    repository: SupportFlowRepository | None,
) -> ModelResult:
    """Run one model-backed agent with one bounded retry and sanitized audit data."""
    for attempt in (1, 2):
        if repository is not None:
            repository.record_model_attempt(run_id, node_name, attempt)
        try:
            return call()
        except (ModelTimeout, InvalidStructuredOutput) as error:
            if repository is not None:
                repository.trace.record_retry(run_id, node_name, attempt, type(error).__name__)
    raise ModelExhausted(node_name, attempts=2)
