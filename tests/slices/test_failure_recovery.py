from __future__ import annotations

from datetime import UTC, datetime

import pytest

from supportflow.execution.executor import TransientExecutionError
from supportflow.workflow.service import SupportFlowService


@pytest.fixture
def durable_service(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    runtime = tmp_path / "runtime"

    def build() -> SupportFlowService:
        return SupportFlowService.demo(
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
            runtime_directory=runtime,
        )

    return build


def test_retrieval_retries_once_then_succeeds_with_sanitized_trace(
    durable_service, duplicate_ticket, monkeypatch
) -> None:
    service = durable_service()
    original = service.graph.retriever.retrieve
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("secret customer token must not enter trace")
        return original(*args, **kwargs)

    monkeypatch.setattr(service.graph.retriever, "retrieve", fail_once)
    result = service.submit(duplicate_ticket)

    assert result.current_state == "WAITING_APPROVAL"
    assert calls == 2
    assert [event.stage for event in result.trace].count("retrieval_retry") == 1
    assert "secret customer token" not in " ".join(event.detail for event in result.trace)


def test_retrieval_exhaustion_persists_terminal_error_across_reopen(
    durable_service, duplicate_ticket, monkeypatch
) -> None:
    service = durable_service()

    def always_fail(*_args, **_kwargs):
        raise ValueError("provider secret must be sanitized")

    monkeypatch.setattr(service.graph.retriever, "retrieve", always_fail)
    failed = service.submit(duplicate_ticket)
    reopened = durable_service().snapshot(failed.run_id)

    assert failed.current_state == "NEEDS_ATTENTION"
    assert reopened.current_state == "NEEDS_ATTENTION"
    assert reopened.errors[-1].stage == "retrieve"
    assert reopened.errors[-1].error_type == "RETRIEVAL_UNAVAILABLE"
    assert reopened.errors[-1].attempt == 2
    assert reopened.errors[-1].retryable is True
    assert "provider secret" not in " ".join(event.detail for event in reopened.trace)
    assert reopened.execution_results == []


@pytest.mark.parametrize("failed_action", [0, 1])
def test_executor_retries_each_action_with_one_persisted_budget(
    durable_service, duplicate_ticket, failed_action
) -> None:
    service = durable_service()
    waiting = service.submit(duplicate_ticket)
    failed_once: set[int] = set()

    def fail_first_attempt(candidate, index, attempt):
        if index == failed_action and index not in failed_once:
            failed_once.add(index)
            raise TransientExecutionError("temporary secret provider failure")
        return candidate

    service.graph.executor.action_runner = fail_first_attempt
    completed = service.approve(
        waiting.run_id, waiting.proposal.proposal_hash, "owner"
    )

    assert completed.current_state == "COMPLETED"
    assert [result.status for result in completed.execution_results] == [
        "succeeded",
        "succeeded",
    ]
    assert [event.stage for event in completed.trace].count("execution_retry") == 1
    assert "temporary secret" not in " ".join(event.detail for event in completed.trace)


def test_executor_exhaustion_persists_failed_result_and_recovery_state(
    durable_service, duplicate_ticket
) -> None:
    service = durable_service()
    waiting = service.submit(duplicate_ticket)

    def always_fail(candidate, index, attempt):
        raise TransientExecutionError("secret provider payload")

    service.graph.executor.action_runner = always_fail
    failed = service.approve(
        waiting.run_id, waiting.proposal.proposal_hash, "owner"
    )
    reopened = durable_service().snapshot(failed.run_id)

    assert reopened.current_state == "NEEDS_ATTENTION"
    assert reopened.approval is not None
    assert reopened.execution_results[-1].status == "failed"
    assert reopened.errors[-1].stage == "execute"
    assert reopened.errors[-1].attempt == 3
    assert service.repository.count_execution_side_effects() == 0
    assert "secret provider payload" not in " ".join(
        event.detail for event in reopened.trace
    )


@pytest.mark.parametrize(
    ("phase", "action_index"),
    [
        ("before_action", 0),
        ("after_action", 0),
        ("before_action", 1),
        ("after_action", 1),
    ],
)
def test_executor_crash_before_or_after_each_action_recovers_without_duplication(
    durable_service, duplicate_ticket, phase, action_index
) -> None:
    class InjectedCrash(RuntimeError):
        pass

    service = durable_service()
    waiting = service.submit(duplicate_ticket)

    def crash_hook(observed_phase, observed_index, _result):
        if observed_phase == phase and observed_index == action_index:
            raise InjectedCrash(f"crash {phase} {action_index}")

    service.graph.executor.crash_hook = crash_hook
    with pytest.raises(InjectedCrash):
        service.approve(waiting.run_id, waiting.proposal.proposal_hash, "owner")

    recovered = durable_service().resume(waiting.run_id)

    assert recovered.current_state == "COMPLETED"
    assert recovered.approval is not None
    assert recovered.execution_results
    assert recovered.execution_results[-1].status in {"succeeded", "skipped_duplicate"}
    assert service.repository.count_execution_side_effects() == 2
