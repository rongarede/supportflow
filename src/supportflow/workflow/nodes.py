from datetime import UTC, datetime
from typing import Callable, TypeVar

from pydantic import BaseModel

from supportflow.agents.protocols import (
    InvalidAction,
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
    local_attempts = 0
    while True:
        if repository is not None:
            attempt = repository.claim_model_attempt(run_id, node_name)
            if attempt is None:
                raise ModelExhausted(node_name, attempts=2)
        else:
            local_attempts += 1
            attempt = local_attempts
            if attempt > 2:
                raise ModelExhausted(node_name, attempts=2)
        try:
            return call()
        except InvalidAction:
            raise
        except (ModelTimeout, InvalidStructuredOutput) as error:
            if repository is not None:
                repository.trace.record_retry(run_id, node_name, attempt, type(error).__name__)
