from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys

import pytest

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
    runtime_directory = tmp_path / "cli-runtime"
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
