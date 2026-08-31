from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


HASH_ALGORITHM = "SHA-256 of UTF-8 canonical JSON (sorted keys, compact separators)"
MANIFEST_VERSION = "supportflow-evidence-manifest/v1"
SOURCE_REVISION_PREFIX = "supportflow-source-sha256:"


def canonical_json(payload: Any) -> str:
    """Serialize a JSON-safe, sanitized payload in one reproducible form."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def hashed_record(source: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a verifiable digest to a payload that has already been sanitized."""
    return {"source": source, "payload": payload, "sha256": canonical_sha256(payload)}


def build_manifest(
    *,
    source_revision: str,
    run: dict[str, Any],
    records: dict[str, dict[str, Any]],
    evaluation_path: str,
    evaluation_payload: Any,
) -> dict[str, Any]:
    """Build a portable manifest without copying secrets or raw customer content."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "sanitized": True,
        "hash_algorithm": HASH_ALGORITHM,
        "source_revision": source_revision,
        "run": run,
        "records": records,
        "evaluation_report": {
            "path": evaluation_path,
            "sha256": canonical_sha256(evaluation_payload),
        },
    }


def export_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the deterministic, reviewable JSON artifact after callers sanitize inputs."""
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def source_revision(repo_root: Path) -> str:
    """Hash executable source and frozen inputs, excluding generated evidence artifacts."""
    candidates = [repo_root / "pyproject.toml", repo_root / "uv.lock"]
    candidates.extend((repo_root / "src" / "supportflow").rglob("*.py"))
    candidates.extend((repo_root / "data").rglob("*"))
    digest = hashlib.sha256()
    for path in sorted(path for path in candidates if path.is_file()):
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return SOURCE_REVISION_PREFIX + digest.hexdigest()


def _checkpoint_count(runtime_directory: Path) -> int:
    connection = sqlite3.connect(runtime_directory / "checkpoints.sqlite")
    try:
        row = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
        return int(row[0])
    finally:
        connection.close()


def generate_manifest_from_deterministic_run(
    *,
    repo_root: Path,
    runtime_directory: Path,
    evaluation_path: Path,
    supplied_source_revision: str | None = None,
) -> dict[str, Any]:
    """Run the public journey, then export sanitized records from that run."""
    from supportflow.domain.models import Ticket
    from supportflow.settings import checkpoint_database_path, runtime_database_path
    from supportflow.workflow.service import SupportFlowService

    computed_revision = source_revision(repo_root)
    if supplied_source_revision is not None and supplied_source_revision != computed_revision:
        raise ValueError(
            "supplied source revision does not match the executable source bundle"
        )
    if runtime_directory.name != "evidence-export":
        raise ValueError("evidence export requires a dedicated evidence-export runtime")
    runtime_directory.mkdir(parents=True, exist_ok=True)
    for database_path in (
        runtime_database_path(runtime_directory),
        checkpoint_database_path(runtime_directory),
    ):
        database_path.unlink(missing_ok=True)
    service = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=runtime_directory,
    )
    ticket = Ticket(
        ticket_id="ticket-duplicate-001",
        customer_id="evidence-customer",
        subject="I was charged twice",
        body="My order order-100 was charged twice for USD 29.00.",
        order_id="order-100",
        amount="29.00",
        currency="USD",
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    waiting = service.submit(ticket, input_revision="evidence-v1")
    completed = service.approve(
        waiting.run_id, waiting.proposal.proposal_hash, "portfolio-owner"
    )
    replayed = service.approve(
        waiting.run_id, waiting.proposal.proposal_hash, "portfolio-owner"
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    proposal = replayed.proposal
    triage = replayed.triage
    review = replayed.risk_review
    policy = replayed.policy_decision
    approval = replayed.approval
    if None in (proposal, triage, review, policy, approval):
        raise RuntimeError("deterministic evidence run did not produce the complete chain")
    trace = service.repository.trace.list_for_run(replayed.run_id)
    records = {
        "ticket": hashed_record(
            "runtime ticket (sanitized)",
            {
                "ticket_id": ticket.ticket_id,
                "amount": ticket.amount,
                "currency": ticket.currency,
                "customer_identifier_redacted": True,
                "order_identifier_redacted": True,
                "ticket_text_redacted": True,
            },
        ),
        "triage": hashed_record(
            "runtime node_outputs/triage",
            {
                "ticket_id": triage.ticket_id,
                "intent": triage.intent.value,
                "urgency": triage.urgency,
                "extracted_fact_keys": sorted(triage.extracted_facts),
                "missing_information": triage.missing_information,
                "risk_flags": triage.risk_flags,
                "route": triage.route,
            },
        ),
        "evidence_bundle": hashed_record(
            "runtime node_outputs/retrieve",
            {
                "query": replayed.evidence.query,
                "sufficient": replayed.evidence.sufficient,
                "unresolved_questions": replayed.evidence.unresolved_questions,
                "active_evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "document_id": item.document_id,
                        "version": item.version,
                        "heading": item.heading,
                    }
                    for item in replayed.evidence.items
                ]
            },
        ),
        "proposal": hashed_record(
            "runtime node_outputs/resolve",
            {
                "proposal_hash": proposal.proposal_hash,
                "reply_text": proposal.reply_text,
                "evidence_refs": proposal.evidence_refs,
                "uncertainties": proposal.uncertainties,
                "actions": [
                    {
                        "action_type": action.action_type.value,
                        "parameters": action.parameters,
                        "reason": action.reason,
                        "evidence_refs": action.evidence_refs,
                        "risk_level": action.risk_level,
                    }
                    for action in proposal.actions
                ],
            },
        ),
        "risk_review": hashed_record(
            "runtime node_outputs/review",
            {
                "decision": review.decision,
                "risk_flags": review.risk_flags,
                "unsupported_claims": review.unsupported_claims,
                "required_changes": review.required_changes,
                "explanation": review.explanation,
            },
        ),
        "policy_decision": hashed_record(
            "runtime node_outputs/policy",
            policy.model_dump(mode="json"),
        ),
        "approval": hashed_record(
            "runtime approvals",
            {
                "proposal_hash": approval.proposal_hash,
                "reviewer_role": approval.reviewer,
                "status": approval.status,
            },
        ),
        "execution_results": hashed_record(
            "runtime executions plus replay",
            {
                "action_types": [
                    item.action_type.value for item in completed.execution_results
                ],
                "first_statuses": [
                    item.status for item in completed.execution_results
                ],
                "replay_statuses": [
                    item.status for item in replayed.execution_results
                ],
                "stored_side_effect_count": service.repository.count_execution_side_effects(),
            },
        ),
        "checkpoint": hashed_record(
            "runtime checkpoints.sqlite",
            {
                "checkpoint_row_count": _checkpoint_count(runtime_directory),
                "strict_msgpack": os.environ.get("LANGGRAPH_STRICT_MSGPACK", "").lower()
                == "true",
            },
        ),
        "trace": hashed_record(
            "runtime trace_events",
            {
                "stages": [event.stage for event in trace],
                "contains_model_retry": any(
                    event.stage == "model_retry" for event in trace
                ),
            },
        ),
    }
    relative_evaluation = evaluation_path.resolve().relative_to(repo_root.resolve()).as_posix()
    return build_manifest(
        source_revision=computed_revision,
        run={
            "run_id": replayed.run_id,
            "ticket_id": ticket.ticket_id,
            "input_revision": "evidence-v1",
            "terminal_state": replayed.current_state.value,
            "runtime_note": "Portable evidence is rebuilt from a dedicated deterministic run; mutable runtime databases are not versioned.",
        },
        records=records,
        evaluation_path=relative_evaluation,
        evaluation_payload=evaluation,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="supportflow-evidence")
    parser.add_argument("command", choices=["export"])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision")
    args = parser.parse_args(argv)
    repo_root = Path.cwd().resolve()
    manifest = generate_manifest_from_deterministic_run(
        repo_root=repo_root,
        runtime_directory=args.runtime.resolve(),
        evaluation_path=args.evaluation.resolve(),
        supplied_source_revision=args.source_revision,
    )
    export_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "run_id": manifest["run"]["run_id"],
                "source_revision": manifest["source_revision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
