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
