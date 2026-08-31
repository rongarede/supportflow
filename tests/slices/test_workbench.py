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


def test_independent_form_submissions_reuse_waiting_and_completed_run_across_reopen(
    tmp_path, monkeypatch
) -> None:
    """UI-generated receipt times must not create new runs or repeat side effects."""
    from supportflow.ui.app import build_workbench_service, submit_workbench_ticket

    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    runtime = tmp_path / "workbench"
    fields = {
        "ticket_id": "ticket-duplicate-001",
        "customer_id": "customer-001",
        "subject": "I was charged twice",
        "body": "My order order-100 was charged twice for USD 29.00.",
        "order_id": "order-100",
        "amount": "29.00",
        "currency": "USD",
    }
    first = submit_workbench_ticket(
        build_workbench_service(runtime),
        **fields,
        received_at=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )
    duplicate_waiting = submit_workbench_ticket(
        build_workbench_service(runtime),
        **dict(fields),
        received_at=datetime(2026, 8, 31, 8, 5, tzinfo=UTC),
    )
    completed = build_workbench_service(runtime).approve(
        first.run_id, first.proposal.proposal_hash, "owner"
    )
    reopened_service = build_workbench_service(runtime)
    duplicate_completed = submit_workbench_ticket(
        reopened_service,
        **dict(fields),
        received_at=datetime(2026, 8, 31, 8, 10, tzinfo=UTC),
    )

    assert duplicate_waiting.run_id == first.run_id
    assert duplicate_waiting.current_state == "WAITING_APPROVAL"
    assert duplicate_completed.run_id == completed.run_id
    assert duplicate_completed.current_state == "COMPLETED"
    assert reopened_service.repository.count_execution_side_effects() == 2


def test_workbench_preserves_explicit_source_revisions(tmp_path, monkeypatch) -> None:
    from supportflow.ui.app import build_workbench_service, submit_workbench_ticket

    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    runtime = tmp_path / "workbench"
    fields = {
        "ticket_id": "ticket-duplicate-001",
        "customer_id": "customer-001",
        "subject": "I was charged twice",
        "body": "My order order-100 was charged twice for USD 29.00.",
        "order_id": "order-100",
        "amount": "29.00",
        "currency": "USD",
    }

    source_v1 = submit_workbench_ticket(
        build_workbench_service(runtime), **fields, input_revision="source-v1"
    )
    source_v1_reopened = submit_workbench_ticket(
        build_workbench_service(runtime), **dict(fields), input_revision="source-v1"
    )
    source_v2 = submit_workbench_ticket(
        build_workbench_service(runtime), **dict(fields), input_revision="source-v2"
    )

    assert source_v1_reopened.run_id == source_v1.run_id
    assert source_v2.run_id != source_v1.run_id
