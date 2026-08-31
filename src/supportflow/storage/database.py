from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_row_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    ticket_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (source, ticket_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    input_revision TEXT NOT NULL,
    current_state TEXT NOT NULL DEFAULT 'RECEIVED',
    last_node TEXT,
    next_node TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (ticket_id, input_revision)
);

CREATE TABLE IF NOT EXISTS node_outputs (
    run_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    output_json TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, node_name)
);

CREATE TABLE IF NOT EXISTS model_attempts (
    run_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, node_name)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    proposal_hash TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    status TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE (run_id, proposal_hash)
);

CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    proposal_hash TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    detail TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_manifest (
    document_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_ids_json TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    built_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_json TEXT NOT NULL,
    vector_json TEXT NOT NULL
);
"""


class SupportFlowDatabase:
    def __init__(self, path: str | Path) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(path),
            check_same_thread=False,
            isolation_level=None,
            timeout=30,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.connection.executescript(SCHEMA)

    @contextmanager
    def immediate(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()
