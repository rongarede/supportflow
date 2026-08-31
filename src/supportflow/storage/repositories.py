from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
from pydantic import BaseModel

from supportflow.domain.models import (
    ApprovalRecord,
    EvidenceBundle,
    ExecutionResult,
    PolicyDecision,
    ResolutionProposal,
    RiskReview,
    Ticket,
    TraceEvent,
    TriageResult,
)
from supportflow.rag.documents import PolicyChunk
from supportflow.storage.database import SupportFlowDatabase


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dumps(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


@dataclass(frozen=True)
class IndexManifest:
    document_id: str
    content_hash: str
    chunk_ids: list[str]
    embedding_model: str
    built_at: str


class TraceRepository:
    def __init__(self, database: SupportFlowDatabase) -> None:
        self.database = database

    @staticmethod
    def _insert(connection, run_id: str, event: TraceEvent) -> None:
        connection.execute(
            "INSERT INTO trace_events "
            "(event_id, run_id, stage, detail, occurred_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid4()), run_id, event.stage, event.detail, event.occurred_at.isoformat()),
        )

    def append(self, run_id: str, event: TraceEvent) -> None:
        with self.database.immediate() as connection:
            self._insert(connection, run_id, event)

    def list_for_run(self, run_id: str) -> list[TraceEvent]:
        rows = self.database.connection.execute(
            "SELECT stage, detail, occurred_at FROM trace_events "
            "WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        return [
            TraceEvent(stage=row["stage"], detail=row["detail"], occurred_at=row["occurred_at"])
            for row in rows
        ]


class SupportFlowRepository:
    def __init__(self, database: SupportFlowDatabase) -> None:
        self.database = database
        self.trace = TraceRepository(database)

    def create_run(
        self,
        run_id: str,
        ticket: Ticket,
        *,
        source: str = "supportflow",
        input_revision: str | None = None,
    ) -> None:
        timestamp = _now()
        revision = input_revision or run_id
        with self.database.immediate() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO tickets "
                "(ticket_row_id, source, ticket_id, ticket_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(uuid4()), source, ticket.ticket_id, ticket.model_dump_json(), timestamp),
            )
            connection.execute(
                "INSERT INTO runs "
                "(run_id, ticket_id, input_revision, current_state, created_at, updated_at) "
                "VALUES (?, ?, ?, 'RECEIVED', ?, ?)",
                (run_id, ticket.ticket_id, revision, timestamp, timestamp),
            )

    def run_exists(self, run_id: str) -> bool:
        return (
            self.database.connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            is not None
        )

    def record_node_result(
        self,
        run_id: str,
        node_name: str,
        output: dict[str, Any],
        event: TraceEvent,
        *,
        current_state: str,
        next_node: str | None,
    ) -> None:
        timestamp = _now()
        with self.database.immediate() as connection:
            connection.execute(
                "INSERT INTO node_outputs "
                "(run_id, node_name, output_json, attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(run_id, node_name) DO UPDATE SET "
                "output_json = excluded.output_json, attempts = attempts + 1, "
                "updated_at = excluded.updated_at",
                (run_id, node_name, _dumps(output), timestamp, timestamp),
            )
            TraceRepository._insert(connection, run_id, event)
            connection.execute(
                "UPDATE runs SET current_state = ?, last_node = ?, next_node = ?, updated_at = ? "
                "WHERE run_id = ?",
                (current_state, node_name, next_node, timestamp, run_id),
            )

    def node_attempts(self, run_id: str) -> dict[str, int]:
        rows = self.database.connection.execute(
            "SELECT node_name, attempts FROM node_outputs WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        return {row["node_name"]: row["attempts"] for row in rows}

    def load_node_result(
        self, run_id: str, node_name: str
    ) -> tuple[dict[str, BaseModel], TraceEvent] | None:
        output_row = self.database.connection.execute(
            "SELECT output_json FROM node_outputs WHERE run_id = ? AND node_name = ?",
            (run_id, node_name),
        ).fetchone()
        if output_row is None:
            return None
        event_row = self.database.connection.execute(
            "SELECT stage, detail, occurred_at FROM trace_events "
            "WHERE run_id = ? AND stage = ? ORDER BY rowid DESC LIMIT 1",
            (run_id, node_name),
        ).fetchone()
        if event_row is None:
            raise RuntimeError(
                f"Persisted node output has no matching Trace event: {run_id}/{node_name}"
            )
        output_types: dict[str, tuple[str, type[BaseModel]]] = {
            "triage": ("triage", TriageResult),
            "retrieve": ("evidence", EvidenceBundle),
            "resolve": ("proposal", ResolutionProposal),
            "review": ("risk_review", RiskReview),
            "policy": ("policy_decision", PolicyDecision),
        }
        try:
            output_key, output_type = output_types[node_name]
        except KeyError as error:
            raise ValueError(f"Node output cannot be reconciled: {node_name}") from error
        raw_output = json.loads(output_row["output_json"])
        output = {output_key: output_type.model_validate(raw_output[output_key])}
        event = TraceEvent(
            stage=event_row["stage"],
            detail=event_row["detail"],
            occurred_at=event_row["occurred_at"],
        )
        return output, event

    def mark_run_state(self, run_id: str, state: str, next_node: str | None = None) -> None:
        with self.database.immediate() as connection:
            connection.execute(
                "UPDATE runs SET current_state = ?, next_node = ?, updated_at = ? WHERE run_id = ?",
                (state, next_node, _now(), run_id),
            )

    def save_approval(self, approval: ApprovalRecord) -> ApprovalRecord:
        with self.database.immediate() as connection:
            existing = connection.execute(
                "SELECT record_json FROM approvals WHERE run_id = ? AND proposal_hash = ?",
                (approval.run_id, approval.proposal_hash),
            ).fetchone()
            if existing is not None:
                return ApprovalRecord.model_validate_json(existing["record_json"])
            connection.execute(
                "INSERT INTO approvals "
                "(approval_id, run_id, proposal_hash, reviewer, status, approved_at, record_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    approval.run_id,
                    approval.proposal_hash,
                    approval.reviewer,
                    approval.status,
                    approval.approved_at.isoformat(),
                    approval.model_dump_json(),
                ),
            )
            return approval

    def get_approval(self, run_id: str, proposal_hash: str) -> ApprovalRecord | None:
        row = self.database.connection.execute(
            "SELECT record_json FROM approvals WHERE run_id = ? AND proposal_hash = ?",
            (run_id, proposal_hash),
        ).fetchone()
        return ApprovalRecord.model_validate_json(row["record_json"]) if row else None

    def store_execution(
        self,
        run_id: str,
        proposal_hash: str,
        result: ExecutionResult,
    ) -> tuple[ExecutionResult, bool]:
        with self.database.immediate() as connection:
            existing = connection.execute(
                "SELECT result_json FROM executions WHERE idempotency_key = ?",
                (result.idempotency_key,),
            ).fetchone()
            if existing is not None:
                original = ExecutionResult.model_validate_json(existing["result_json"])
                return original.model_copy(update={"status": "skipped_duplicate"}), False
            connection.execute(
                "INSERT INTO executions "
                "(execution_id, run_id, proposal_hash, idempotency_key, result_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    run_id,
                    proposal_hash,
                    result.idempotency_key,
                    result.model_dump_json(),
                    result.executed_at.isoformat(),
                ),
            )
            return result, True

    def count_execution_side_effects(self) -> int:
        row = self.database.connection.execute(
            "SELECT COUNT(*) AS count FROM executions"
        ).fetchone()
        return int(row["count"])

    def get_index_manifest(self, document_id: str) -> IndexManifest | None:
        row = self.database.connection.execute(
            "SELECT * FROM index_manifest WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return IndexManifest(
            document_id=row["document_id"],
            content_hash=row["content_hash"],
            chunk_ids=json.loads(row["chunk_ids_json"]),
            embedding_model=row["embedding_model"],
            built_at=row["built_at"],
        )

    def save_index_manifest(self, manifest: IndexManifest) -> None:
        with self.database.immediate() as connection:
            connection.execute(
                "INSERT INTO index_manifest "
                "(document_id, content_hash, chunk_ids_json, embedding_model, built_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(document_id) DO UPDATE SET "
                "content_hash = excluded.content_hash, chunk_ids_json = excluded.chunk_ids_json, "
                "embedding_model = excluded.embedding_model, built_at = excluded.built_at",
                (
                    manifest.document_id,
                    manifest.content_hash,
                    _dumps(manifest.chunk_ids),
                    manifest.embedding_model,
                    manifest.built_at,
                ),
            )

    def replace_index_document(
        self,
        manifest: IndexManifest,
        chunks: list[PolicyChunk],
        vectors: np.ndarray,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each persisted policy chunk requires exactly one vector")
        with self.database.immediate() as connection:
            connection.execute(
                "DELETE FROM index_chunks WHERE document_id = ?", (manifest.document_id,)
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                connection.execute(
                    "INSERT INTO index_chunks "
                    "(chunk_id, document_id, chunk_json, vector_json) VALUES (?, ?, ?, ?)",
                    (
                        chunk.evidence_id,
                        manifest.document_id,
                        chunk.model_dump_json(),
                        _dumps(vector.tolist()),
                    ),
                )
            connection.execute(
                "INSERT INTO index_manifest "
                "(document_id, content_hash, chunk_ids_json, embedding_model, built_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(document_id) DO UPDATE SET "
                "content_hash = excluded.content_hash, chunk_ids_json = excluded.chunk_ids_json, "
                "embedding_model = excluded.embedding_model, built_at = excluded.built_at",
                (
                    manifest.document_id,
                    manifest.content_hash,
                    _dumps(manifest.chunk_ids),
                    manifest.embedding_model,
                    manifest.built_at,
                ),
            )

    def load_index_chunks(
        self, chunk_ids: Iterable[str]
    ) -> tuple[list[PolicyChunk], np.ndarray]:
        chunks: list[PolicyChunk] = []
        vectors: list[list[float]] = []
        for chunk_id in chunk_ids:
            row = self.database.connection.execute(
                "SELECT chunk_json, vector_json FROM index_chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                return [], np.asarray([])
            chunks.append(PolicyChunk.model_validate_json(row["chunk_json"]))
            vectors.append(json.loads(row["vector_json"]))
        return chunks, np.asarray(vectors, dtype=float)

    def delete_removed_index_documents(self, active_document_ids: set[str]) -> None:
        rows = self.database.connection.execute(
            "SELECT document_id FROM index_manifest"
        ).fetchall()
        removed = {row["document_id"] for row in rows} - active_document_ids
        if not removed:
            return
        with self.database.immediate() as connection:
            for document_id in removed:
                connection.execute(
                    "DELETE FROM index_chunks WHERE document_id = ?", (document_id,)
                )
                connection.execute(
                    "DELETE FROM index_manifest WHERE document_id = ?", (document_id,)
                )
