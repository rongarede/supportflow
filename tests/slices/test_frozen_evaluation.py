from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from supportflow.workflow.service import SupportFlowService


@pytest.fixture
def service_factory(tmp_path, monkeypatch):
    """Catches an evaluator that bypasses the durable public service boundary."""
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    sequence = 0

    def build(*, shared_index):
        nonlocal sequence
        sequence += 1
        service = SupportFlowService.demo(
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
            runtime_directory=tmp_path / f"runtime-{sequence}",
        )
        service.graph.retriever = shared_index
        return service

    return build


def test_frozen_dataset_runs_complete_workflow(tmp_path, service_factory) -> None:
    """Catches an evaluator that reports safety without driving complete service journeys."""
    from supportflow.eval.runner import run_evaluation

    rows = [
        json.loads(line)
        for line in Path("data/eval/tickets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 30
    assert Counter(row["case_type"] for row in rows) == {
        "billing_question": 12,
        "refund_request": 6,
        "missing_information": 4,
        "policy_conflict": 4,
        "duplicate_submission": 4,
    }

    summary = run_evaluation(Path("data/eval/tickets.jsonl"), service_factory, tmp_path)

    assert summary.unapproved_execution_count == 0
    assert summary.duplicate_side_effect_count == 0
    assert summary.recovery_failure_count == 0
    assert (tmp_path / f"eval-{summary.dataset_sha256}.json").exists()
    assert (tmp_path / f"eval-{summary.dataset_sha256}.md").exists()


def test_cli_evaluate_runs_the_same_frozen_journey(tmp_path) -> None:
    """Catches a CLI entry point that cannot forward evaluation arguments to the full journey."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "supportflow.cli",
            "evaluate",
            "--dataset",
            "data/eval/tickets.jsonl",
            "--policies",
            "data/policies",
            "--output",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "\"case_count\": 30" in completed.stdout


def test_evaluator_refuses_a_non_durable_service(tmp_path, monkeypatch) -> None:
    """Catches an evaluator that silently scores an in-memory workflow instead of recovery behavior."""
    from supportflow.eval.runner import run_evaluation

    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")

    def in_memory_factory(*, shared_index):
        service = SupportFlowService.demo(as_of=datetime(2026, 8, 31, tzinfo=UTC))
        service.graph.retriever = shared_index
        return service

    with pytest.raises(RuntimeError, match="durable service"):
        run_evaluation(Path("data/eval/tickets.jsonl"), in_memory_factory, tmp_path)
