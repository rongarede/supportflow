from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from supportflow.workflow.service import SupportFlowService


@pytest.fixture
def durable_service(tmp_path, monkeypatch) -> SupportFlowService:
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    return SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=tmp_path / "runtime",
    )


def test_workbench_exposes_approval_ready_brief(durable_service, duplicate_ticket) -> None:
    """Catches a workbench that hides evidence, policy, or human choices."""
    from supportflow.ui.app import build_view_model

    snapshot = durable_service.submit(duplicate_ticket)
    view = build_view_model(snapshot)

    assert view.state == "WAITING_APPROVAL"
    assert view.proposal_hash == snapshot.proposal.proposal_hash
    assert view.evidence_rows
    assert view.policy_outcome == "allow"
    assert {action.label for action in view.available_actions} == {
        "Approve",
        "Modify and re-review",
        "Reject",
        "Escalate",
    }


def test_ui_never_bypasses_application_service() -> None:
    """Catches a page that reaches persistence or execution below the service boundary."""
    source = Path("src/supportflow/ui/app.py").read_text()

    assert "sqlite3.connect" not in source
    assert "SimulatedExecutor(" not in source
    assert "SupportFlowService" in source


def test_workbench_runtime_is_durable_and_can_reopen_a_run(
    tmp_path, monkeypatch, duplicate_ticket
) -> None:
    from supportflow.ui.app import build_workbench_service

    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    runtime = tmp_path / "workbench"
    waiting = build_workbench_service(runtime).submit(duplicate_ticket)
    reopened = build_workbench_service(runtime).resume(waiting.run_id)

    assert reopened.run_id == waiting.run_id
    assert reopened.current_state == "WAITING_APPROVAL"
    assert reopened.node_attempts == waiting.node_attempts


def test_terminal_workbench_explains_that_no_human_actions_remain(
    durable_service, duplicate_ticket
) -> None:
    from supportflow.ui.app import available_actions_message, build_view_model

    waiting = durable_service.submit(duplicate_ticket)
    terminal = durable_service.reject(waiting.run_id, "Reviewed.", "owner")
    view = build_view_model(terminal)

    assert view.available_actions == ()
    assert available_actions_message(view) == "No human actions are available in REJECTED."
