from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np

from supportflow.domain.hashing import canonical_input_revision
from supportflow.domain.models import Ticket
from supportflow.rag.documents import PolicyDocument
from supportflow.rag.index import build_persisted_policy_index
from supportflow.storage.database import SupportFlowDatabase
from supportflow.storage.repositories import IndexManifest, SupportFlowRepository


def test_repository_derives_stable_revision_without_ingestion_timestamp(tmp_path) -> None:
    """The persistence boundary must deduplicate even when callers omit a revision."""
    repository = SupportFlowRepository(SupportFlowDatabase(tmp_path / "supportflow.db"))
    first_ticket = Ticket(
        ticket_id="T-FORM-1",
        customer_id="C-1",
        subject="Charged twice",
        body="Order A was charged twice for USD 10.00.",
        order_id="A",
        amount="10.00",
        currency="USD",
        created_at=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )
    independently_constructed = first_ticket.model_copy(
        update={"created_at": first_ticket.created_at + timedelta(minutes=5)}
    )

    first_run_id = repository.create_run("run-first", first_ticket)
    duplicate_run_id = repository.create_run(
        "run-duplicate", independently_constructed
    )
    explicit_run_id = repository.create_run(
        "run-explicit", independently_constructed, input_revision="source-v2"
    )

    assert canonical_input_revision(first_ticket) == canonical_input_revision(
        independently_constructed
    )
    assert duplicate_run_id == first_run_id == "run-first"
    assert explicit_run_id == "run-explicit"
    revisions = {
        row["run_id"]: row["input_revision"]
        for row in repository.database.connection.execute(
            "SELECT run_id, input_revision FROM runs"
        )
    }
    assert revisions["run-first"] == canonical_input_revision(first_ticket)
    assert revisions["run-explicit"] == "source-v2"


def test_schema_enforces_ticket_run_and_execution_deduplication(tmp_path) -> None:
    database = SupportFlowDatabase(tmp_path / "supportflow.db")
    table_names = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert {
        "tickets",
        "runs",
        "node_outputs",
        "approvals",
        "executions",
        "trace_events",
        "index_manifest",
    } <= table_names

    database.connection.execute(
        "INSERT INTO tickets (ticket_row_id, source, ticket_id, ticket_json, created_at) "
        "VALUES ('ticket-row-1', 'demo', 'T-1', '{}', '2026-08-31T00:00:00Z')"
    )
    database.connection.execute(
        "INSERT INTO runs (run_id, ticket_id, input_revision, created_at, updated_at) "
        "VALUES ('run-1', 'T-1', 'rev-1', '2026-08-31T00:00:00Z', '2026-08-31T00:00:00Z')"
    )
    database.connection.execute(
        "INSERT INTO executions "
        "(execution_id, run_id, proposal_hash, idempotency_key, result_json, created_at) "
        "VALUES ('execution-1', 'run-1', 'hash-1', 'key-1', '{}', '2026-08-31T00:00:00Z')"
    )

    duplicate_writes = [
        (
            "INSERT INTO tickets (ticket_row_id, source, ticket_id, ticket_json, created_at) "
            "VALUES ('ticket-row-2', 'demo', 'T-1', '{}', '2026-08-31T00:00:00Z')",
            (),
        ),
        (
            "INSERT INTO runs (run_id, ticket_id, input_revision, created_at, updated_at) "
            "VALUES ('run-2', 'T-1', 'rev-1', '2026-08-31T00:00:00Z', '2026-08-31T00:00:00Z')",
            (),
        ),
        (
            "INSERT INTO executions "
            "(execution_id, run_id, proposal_hash, idempotency_key, result_json, created_at) "
            "VALUES ('execution-2', 'run-1', 'hash-1', 'key-1', '{}', '2026-08-31T00:00:00Z')",
            (),
        ),
    ]
    for statement, parameters in duplicate_writes:
        try:
            database.connection.execute(statement, parameters)
        except sqlite3.IntegrityError:
            continue
        raise AssertionError("expected SQLite to reject a duplicate key")


def test_index_manifest_round_trip_and_removed_document_cleanup(tmp_path) -> None:
    repository = SupportFlowRepository(SupportFlowDatabase(tmp_path / "supportflow.db"))
    current = IndexManifest(
        document_id="billing-policy",
        content_hash="content-v1",
        chunk_ids=["billing-policy-001", "billing-policy-002"],
        embedding_model="fixed-test-v1",
        built_at="2026-08-31T00:00:00+00:00",
    )
    removed = IndexManifest(
        document_id="removed-policy",
        content_hash="removed-v1",
        chunk_ids=["removed-policy-001"],
        embedding_model="fixed-test-v1",
        built_at="2026-08-31T00:00:00+00:00",
    )

    repository.save_index_manifest(current)
    repository.save_index_manifest(removed)
    repository.delete_removed_index_documents({"billing-policy"})

    assert repository.get_index_manifest("billing-policy") == current
    assert repository.get_index_manifest("removed-policy") is None


def test_persisted_policy_index_reuses_unchanged_vectors_and_rebuilds_changed_documents(
    tmp_path,
) -> None:
    class CountingEmbeddingProvider:
        name = "counting-v1"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts: list[str]) -> np.ndarray:
            self.calls.append(texts)
            return np.asarray([[float(len(text)), 1.0] for text in texts])

    repository = SupportFlowRepository(SupportFlowDatabase(tmp_path / "supportflow.db"))
    provider = CountingEmbeddingProvider()
    billing = PolicyDocument(
        document_id="billing",
        version="1",
        effective_from="2026-01-01T00:00:00Z",
        effective_to=None,
        policy_type="billing",
        title="Billing",
        body="# duplicate-charge-verification\nVerify the order and payment identifiers.",
    )
    refund = PolicyDocument(
        document_id="refund",
        version="1",
        effective_from="2026-01-01T00:00:00Z",
        effective_to=None,
        policy_type="refund",
        title="Refund",
        body="# refund-timing\nRefunds take five business days.",
    )

    first_chunks, first_vectors = build_persisted_policy_index(
        [billing, refund], provider, repository
    )
    first_call_count = len(provider.calls)
    cached_chunks, cached_vectors = build_persisted_policy_index(
        [billing, refund], provider, repository
    )
    changed_refund = refund.model_copy(
        update={"body": "# refund-timing\nRefunds take seven business days."}
    )
    changed_chunks, changed_vectors = build_persisted_policy_index(
        [changed_refund], provider, repository
    )

    assert first_call_count == 2
    assert len(provider.calls) == 3
    assert [chunk.evidence_id for chunk in cached_chunks] == [
        chunk.evidence_id for chunk in first_chunks
    ]
    assert np.array_equal(cached_vectors, first_vectors)
    assert [chunk.document.document_id for chunk in changed_chunks] == ["refund"]
    assert changed_vectors.tolist() == [[33.0, 1.0]]
    assert repository.get_index_manifest("billing") is None
