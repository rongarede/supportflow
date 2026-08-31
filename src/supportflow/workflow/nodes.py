from datetime import UTC, datetime

from supportflow.domain.models import TraceEvent


def trace(stage: str, detail: str) -> TraceEvent:
    return TraceEvent(stage=stage, detail=detail, occurred_at=datetime.now(UTC))
