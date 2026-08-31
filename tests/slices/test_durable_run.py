from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys

import pytest

from supportflow.cli import demo_restart
from supportflow.settings import checkpoint_database_path, runtime_database_path
from supportflow.workflow.service import SupportFlowService


@pytest.fixture
def sqlite_service_factory(tmp_path):
    runtime_directory = tmp_path / "runtime"

    def build() -> SupportFlowService:
        return SupportFlowService.demo(
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
            runtime_directory=runtime_directory,
        )

    return build


def test_restart_resumes_without_rerunning_agents(sqlite_service_factory, duplicate_ticket) -> None:
    first = sqlite_service_factory().submit(duplicate_ticket)

    restarted = sqlite_service_factory().resume(first.run_id)

    assert restarted.current_state == "WAITING_APPROVAL"
    assert restarted.node_attempts == {
        "triage": 1,
        "retrieve": 1,
        "resolve": 1,
        "review": 1,
        "policy": 1,
    }


def test_repeated_approval_is_idempotent_per_action(
    sqlite_service_factory, duplicate_ticket
) -> None:
    service = sqlite_service_factory()
    waiting = service.submit(duplicate_ticket)

    first = service.approve(waiting.run_id, waiting.proposal.proposal_hash, "owner")
    second = service.approve(waiting.run_id, waiting.proposal.proposal_hash, "owner")

    assert [result.status for result in first.execution_results] == [
        "succeeded",
        "succeeded",
    ]
    assert [result.status for result in second.execution_results] == [
        "skipped_duplicate",
        "skipped_duplicate",
    ]
    assert service.repository.count_execution_side_effects() == 2


def test_duplicate_intake_reuses_the_same_run_in_process(
    sqlite_service_factory, duplicate_ticket
) -> None:
    """Catches a random default revision that reruns every model-backed node."""
    service = sqlite_service_factory()
    first = service.submit(duplicate_ticket)
    repeated = service.submit(duplicate_ticket)

    assert repeated.run_id == first.run_id
    assert repeated.node_attempts == first.node_attempts
    assert [event.stage for event in repeated.trace] == [
        "triage",
        "retrieve",
        "resolve",
        "review",
        "policy",
    ]


def test_duplicate_intake_reuses_the_same_run_after_reopen(
    sqlite_service_factory, duplicate_ticket
) -> None:
    first = sqlite_service_factory().submit(duplicate_ticket, input_revision="source-v7")
    reopened = sqlite_service_factory().submit(duplicate_ticket, input_revision="source-v7")

    assert reopened.run_id == first.run_id
    assert reopened.current_state == "WAITING_APPROVAL"
    assert reopened.node_attempts == first.node_attempts


def test_duplicate_completed_intake_does_not_repeat_side_effects(
    sqlite_service_factory, duplicate_ticket
) -> None:
    service = sqlite_service_factory()
    waiting = service.submit(duplicate_ticket)
    completed = service.approve(
        waiting.run_id, waiting.proposal.proposal_hash, "owner"
    )
    repeated = sqlite_service_factory().submit(duplicate_ticket)

    assert repeated.run_id == completed.run_id
    assert repeated.current_state == "COMPLETED"
    assert service.repository.count_execution_side_effects() == 2


def test_derived_input_revision_changes_only_when_ticket_input_changes(
    sqlite_service_factory, duplicate_ticket
) -> None:
    service = sqlite_service_factory()
    first = service.submit(duplicate_ticket)
    changed = service.submit(
        duplicate_ticket.model_copy(update={"body": "A corrected customer message."})
    )

    assert changed.run_id != first.run_id


def test_restart_preserves_approval_and_execution_idempotency(
    sqlite_service_factory, duplicate_ticket
) -> None:
    waiting = sqlite_service_factory().submit(duplicate_ticket)
    restarted = sqlite_service_factory()

    completed = restarted.approve(
        waiting.run_id, waiting.proposal.proposal_hash, "owner"
    )
    replayed = sqlite_service_factory().approve(
        waiting.run_id, waiting.proposal.proposal_hash, "owner"
    )

    assert completed.current_state == "COMPLETED"
    assert [result.status for result in replayed.execution_results] == [
        "skipped_duplicate",
        "skipped_duplicate",
    ]
    assert restarted.repository.count_execution_side_effects() == 2


def test_demo_restart_uses_fresh_processes_and_reports_two_stored_actions(tmp_path) -> None:
    runtime_directory = tmp_path / "demo-restart"
    environment = {**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "supportflow.cli",
            "demo-restart",
            "--runtime",
            str(runtime_directory),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "resumed_state: WAITING_APPROVAL" in completed.stdout
    assert "first_execution_statuses: succeeded, succeeded" in completed.stdout
    assert "repeated_execution_statuses: skipped_duplicate, skipped_duplicate" in completed.stdout
    assert "stored_side_effects: 2" in completed.stdout


@pytest.mark.parametrize("crashed_node", ["triage", "retrieve", "resolve", "review", "policy"])
def test_resume_reconciles_committed_node_output_without_rerunning_agent(
    tmp_path, monkeypatch, duplicate_ticket, crashed_node
) -> None:
    runtime_directory = tmp_path / "runtime"
    service = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=runtime_directory,
    )
    run_id = f"run-crash-after-{crashed_node}"
    service.repository.create_run(run_id, duplicate_ticket)
    original_record = service.repository.record_node_result

    def commit_then_crash(
        persisted_run_id,
        node_name,
        output,
        event,
        *,
        current_state,
        next_node,
    ) -> None:
        original_record(
            persisted_run_id,
            node_name,
            output,
            event,
            current_state=current_state,
            next_node=next_node,
        )
        if node_name == crashed_node:
            raise RuntimeError(f"injected crash after {node_name} journal commit")

    monkeypatch.setattr(service.repository, "record_node_result", commit_then_crash)

    with pytest.raises(RuntimeError, match=f"after {crashed_node} journal commit"):
        service.graph.compiled.invoke(
            {"run_id": run_id, "ticket": duplicate_ticket, "trace": []},
            service._config(run_id),
        )

    partial = service.snapshot(run_id)
    assert partial.current_state != "WAITING_APPROVAL"

    restarted = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=runtime_directory,
    )
    recovered = restarted.resume(run_id)

    assert recovered.current_state == "WAITING_APPROVAL"
    assert recovered.proposal is not None
    assert recovered.policy_decision.outcome == "allow"
    assert recovered.node_attempts == {
        "triage": 1,
        "retrieve": 1,
        "resolve": 1,
        "review": 1,
        "policy": 1,
    }
    assert [event.stage for event in recovered.trace] == [
        "triage",
        "retrieve",
        "resolve",
        "review",
        "policy",
    ]


@pytest.mark.parametrize("crashed_node", ["triage", "retrieve", "resolve", "review", "policy"])
def test_resume_safely_replays_a_pending_node_when_crash_precedes_journal_commit(
    tmp_path, monkeypatch, duplicate_ticket, crashed_node
) -> None:
    """A checkpointed pending node must not depend on already having a journal output."""
    runtime_directory = tmp_path / "runtime"
    service = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=runtime_directory,
    )
    run_id = f"run-crash-before-{crashed_node}"
    service.repository.create_run(run_id, duplicate_ticket)
    original_record = service.repository.record_node_result

    def crash_before_commit(
        persisted_run_id,
        node_name,
        output,
        event,
        *,
        current_state,
        next_node,
    ) -> None:
        if node_name == crashed_node:
            raise RuntimeError(f"injected crash before {node_name} journal commit")
        original_record(
            persisted_run_id,
            node_name,
            output,
            event,
            current_state=current_state,
            next_node=next_node,
        )

    monkeypatch.setattr(service.repository, "record_node_result", crash_before_commit)

    with pytest.raises(RuntimeError, match=f"before {crashed_node} journal commit"):
        service.graph.compiled.invoke(
            {"run_id": run_id, "ticket": duplicate_ticket, "trace": []},
            service._config(run_id),
        )

    restarted = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=runtime_directory,
    )
    recovered = restarted.resume(run_id)

    assert recovered.current_state == "WAITING_APPROVAL"
    assert recovered.proposal is not None
    assert recovered.policy_decision.outcome == "allow"
    assert restarted.repository.count_execution_side_effects() == 0
    assert [event.stage for event in recovered.trace] == [
        "triage",
        "retrieve",
        "resolve",
        "review",
        "policy",
    ]
    if crashed_node in {"triage", "resolve", "review"}:
        assert recovered.node_attempts[crashed_node] == 2
    if crashed_node == "retrieve":
        assert restarted.repository.operation_attempts(run_id, "retrieve") == 2


@pytest.mark.parametrize("strict_value", [None, "false"])
def test_sqlite_checkpoint_refuses_non_strict_serializer_mode(
    tmp_path, monkeypatch, strict_value
) -> None:
    if strict_value is None:
        monkeypatch.delenv("LANGGRAPH_STRICT_MSGPACK", raising=False)
    else:
        monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", strict_value)
    runtime_directory = tmp_path / "runtime"

    with pytest.raises(RuntimeError, match="LANGGRAPH_STRICT_MSGPACK=true"):
        SupportFlowService.demo(runtime_directory=runtime_directory)

    assert not checkpoint_database_path(runtime_directory).exists()


def test_demo_restart_defaults_to_dedicated_runtime_without_deleting_general_database(
    tmp_path,
) -> None:
    general_runtime = tmp_path / ".supportflow"
    general_runtime.mkdir()
    general_database = runtime_database_path(general_runtime)
    general_database.write_bytes(b"general-runtime-must-survive")
    environment = {
        **os.environ,
        "LANGGRAPH_STRICT_MSGPACK": "true",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }

    completed = subprocess.run(
        [sys.executable, "-m", "supportflow.cli", "demo-restart"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert general_database.read_bytes() == b"general-runtime-must-survive"
    assert runtime_database_path(general_runtime / "demo-restart").exists()


def test_demo_restart_refuses_to_clear_general_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    general_runtime = tmp_path / ".supportflow"
    general_runtime.mkdir()
    general_database = runtime_database_path(general_runtime)
    general_database.write_bytes(b"general-runtime-must-survive")

    with pytest.raises(ValueError, match="dedicated demo-restart runtime"):
        demo_restart(general_runtime)

    assert general_database.read_bytes() == b"general-runtime-must-survive"
