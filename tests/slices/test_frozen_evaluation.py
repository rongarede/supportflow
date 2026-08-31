from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import os
import subprocess
import sys
from copy import deepcopy

import pytest

from supportflow.workflow.service import SupportFlowService
from supportflow.domain.models import Ticket


@pytest.fixture
def service_factory(tmp_path, monkeypatch):
    """Catches an evaluator that bypasses the durable public service boundary."""
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    sequence = 0

    def build(*, shared_index, runtime_directory=None):
        nonlocal sequence
        sequence += 1
        service = SupportFlowService.demo(
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
            runtime_directory=runtime_directory or tmp_path / f"runtime-{sequence}",
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

    def in_memory_factory(*, shared_index, runtime_directory):
        service = SupportFlowService.demo(as_of=datetime(2026, 8, 31, tzinfo=UTC))
        service.graph.retriever = shared_index
        return service

    with pytest.raises(RuntimeError, match="durable service"):
        run_evaluation(Path("data/eval/tickets.jsonl"), in_memory_factory, tmp_path)


def _write_dataset(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (path.parent / "model_fixtures.json").write_bytes(Path("data/eval/model_fixtures.json").read_bytes())
    return path


def test_expected_label_mutation_does_not_change_observed_journeys_but_degrades_scores(
    tmp_path, service_factory
) -> None:
    """Catches model fixtures derived from score labels instead of independent frozen outputs."""
    from supportflow.eval.runner import run_evaluation

    original = Path("data/eval/tickets.jsonl")
    baseline = run_evaluation(original, service_factory, tmp_path / "baseline")
    rows = [json.loads(line) for line in original.read_text(encoding="utf-8").splitlines()]
    mutated = deepcopy(rows)
    missing = next(row for row in mutated if row["case_id"] == "missing-01")
    conflict = next(row for row in mutated if row["case_id"] == "conflict-01")
    missing["case_type"], conflict["case_type"] = conflict["case_type"], missing["case_type"]
    missing["expected_intent"] = "NOT_DUPLICATE"
    missing["required_evidence_ids"] = ["not-an-evidence-id"]
    missing["expected_terminal_state"] = "COMPLETED"
    modified = run_evaluation(
        _write_dataset(tmp_path / "mutated.jsonl", mutated), service_factory, tmp_path / "mutated", Path("data/policies")
    )
    baseline_cases = json.loads(next((tmp_path / "baseline").glob("*.json")).read_text())["cases"]
    modified_cases = json.loads(next((tmp_path / "mutated").glob("*.json")).read_text())["cases"]

    assert [case["observed"] for case in baseline_cases] == [case["observed"] for case in modified_cases]
    assert modified.route_accuracy < baseline.route_accuracy
    assert modified.required_evidence_hit_rate < baseline.required_evidence_hit_rate
    assert modified.terminal_state_accuracy < baseline.terminal_state_accuracy


def test_forbidden_proposed_action_lowers_action_accuracy_even_when_rejected(
    tmp_path, service_factory
) -> None:
    """Catches action scoring that inspects execution only and misses an unsafe blocked proposal."""
    from supportflow.eval.runner import run_evaluation

    rows = [json.loads(line) for line in Path("data/eval/tickets.jsonl").read_text().splitlines()]
    rejected = next(row for row in rows if row["case_id"] == "billing-03")
    rejected["allowed_actions"] = []
    rejected["forbidden_actions"] = ["CREATE_REFUND_REQUEST", "SEND_REPLY"]
    summary = run_evaluation(
        _write_dataset(tmp_path / "forbidden.jsonl", rows), service_factory, tmp_path / "output", Path("data/policies")
    )

    assert summary.action_accuracy < 1.0


def test_reports_are_trackable_deterministic_and_include_fixture_audit_context(
    tmp_path, service_factory
) -> None:
    """Catches reports that omit adapter/provenance/observed evidence or vary for identical inputs."""
    from supportflow.eval.runner import run_evaluation

    first = run_evaluation(Path("data/eval/tickets.jsonl"), service_factory, tmp_path)
    json_path = tmp_path / f"eval-{first.dataset_sha256}.json"
    markdown_path = tmp_path / f"eval-{first.dataset_sha256}.md"
    first_hashes = (json_path.read_bytes(), markdown_path.read_bytes())
    second = run_evaluation(Path("data/eval/tickets.jsonl"), service_factory, tmp_path)
    payload = json.loads(json_path.read_text())

    assert first_hashes == (json_path.read_bytes(), markdown_path.read_bytes())
    assert first == second
    assert payload["adapters"] == {"model": "FakeStructuredModel", "embedding": "FixedEmbeddingProvider"}
    assert payload["cases"][0]["observed"]["proposed_actions"]
    assert payload["cases"][0]["expected_vs_observed"]
    markdown = markdown_path.read_text()
    assert "not real-LLM quality" in markdown
    assert "## Per-case audit" in markdown
    assert "| billing-01 |" in markdown
    tracked = subprocess.run(
        ["git", "check-ignore", str(Path("artifacts") / json_path.name)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode != 0


def test_evaluation_reopens_a_partial_durable_run_and_audits_recovery_equivalence(
    tmp_path, service_factory
) -> None:
    """Catches recovery scored from a same-instance no-op instead of reopened durable snapshots."""
    from supportflow.eval.runner import run_evaluation

    summary = run_evaluation(Path("data/eval/tickets.jsonl"), service_factory, tmp_path)
    payload = json.loads((tmp_path / f"eval-{summary.dataset_sha256}.json").read_text())
    recovery = next(case["recovery"] for case in payload["cases"] if case["case_id"] == "billing-01")

    assert recovery["interrupted_partial_checkpoint"] is True
    assert recovery["reopened_snapshot_matches"] is True
    assert recovery["node_attempts"] == {"triage": 1, "retrieve": 1, "resolve": 1, "review": 1, "policy": 1}
    assert summary.recovery_failure_count == 0


def test_recovery_comparison_detects_changed_reopened_state(tmp_path, service_factory) -> None:
    """Catches a recovery audit that reports success after reopened state has diverged."""
    from supportflow.eval.runner import _recovery_matches, _snapshot_signature
    from supportflow.rag.embeddings import FixedEmbeddingProvider
    from supportflow.rag.index import build_policy_chunks
    from supportflow.rag.documents import load_policy_documents
    from supportflow.rag.retriever import RagRetriever

    documents = load_policy_documents(Path("data/policies"))
    chunks = build_policy_chunks(documents)
    index = RagRetriever(chunks, FixedEmbeddingProvider({chunk.text: [1.0, 0.0] for chunk in chunks}))
    runtime = tmp_path / "runtime"
    first = service_factory(shared_index=index, runtime_directory=runtime)
    waiting = first.submit(Ticket(
        ticket_id="ticket-duplicate-001", customer_id="eval-account", subject="Duplicate charge",
        body="Order eval-order was charged twice for USD 29.00.", order_id="eval-order",
        amount="29.00", currency="USD", created_at=datetime(2026, 8, 31, tzinfo=UTC),
    ))
    reopened = service_factory(shared_index=index, runtime_directory=runtime)
    recovered = reopened.resume(waiting.run_id)
    before = _snapshot_signature(recovered, reopened)
    changed = reopened.reject(recovered.run_id, "test divergence", "reviewer")

    assert _recovery_matches(before, _snapshot_signature(changed, reopened)) is False
