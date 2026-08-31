from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from supportflow.agents.fake import FakeStructuredModel
from supportflow.domain.models import (
    ActionProposal,
    ResolutionProposal,
    RiskReview,
    Ticket,
    TriageResult,
)
from supportflow.eval.report import EvaluationSummary, write_reports
from supportflow.rag.documents import load_policy_documents
from supportflow.rag.embeddings import FixedEmbeddingProvider
from supportflow.rag.index import build_policy_chunks
from supportflow.rag.retriever import RagRetriever
from supportflow.workflow.service import SupportFlowService

_EXPECTED_DISTRIBUTION = {
    "billing_question": 12,
    "refund_request": 6,
    "missing_information": 4,
    "policy_conflict": 4,
    "duplicate_submission": 4,
}
_NORMAL_EVIDENCE_HEADINGS = (
    "duplicate-charge-verification",
    "duplicate-charge-refund-request",
    "refund-timing",
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _policy_sha256(policy_directory: Path) -> str:
    content = b"".join(
        path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0"
        for path in sorted(policy_directory.glob("*.md"))
    )
    return _sha256_bytes(content)


def _build_frozen_index(policy_directory: Path) -> RagRetriever:
    documents = load_policy_documents(policy_directory)
    chunks = build_policy_chunks(documents)
    provider = FixedEmbeddingProvider({chunk.text: [1.0, 0.0] for chunk in chunks})
    return RagRetriever(chunks, provider)


def _load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line]
    required = {
        "case_id", "source_id", "source_row_or_pattern", "rewrite_note", "case_type",
        "ticket", "expected_intent", "required_evidence_ids", "allowed_actions",
        "forbidden_actions", "expected_terminal_state", "human_action", "risk_flags",
    }
    if len(cases) != 30 or Counter(case.get("case_type") for case in cases) != _EXPECTED_DISTRIBUTION:
        raise ValueError("Frozen evaluation dataset must contain the specified 30-case distribution")
    if any(not required <= set(case) for case in cases):
        raise ValueError("Frozen evaluation case is missing required provenance or expectation fields")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Frozen evaluation case_id values must be unique")
    return cases


def _configure_case_model(service: SupportFlowService, case: dict[str, Any], index: RagRetriever) -> None:
    model = service.graph.triage.model
    if not isinstance(model, FakeStructuredModel):
        raise TypeError("Frozen evaluation requires the deterministic structured model")
    ticket = Ticket.model_validate(case["ticket"])
    current = datetime(2026, 8, 31, tzinfo=UTC)
    evidence_by_heading = {chunk.heading: chunk.evidence_id for chunk in index.chunks if chunk.document.active_at(current)}
    model.responses[("triage", ticket.ticket_id, 1)] = TriageResult(
        ticket_id=ticket.ticket_id,
        intent="DUPLICATE_CHARGE",
        confidence=0.99,
        rationale="Deterministic frozen evaluation fixture.",
        missing_fields=["provider failure time"] if case["case_type"] == "missing_information" else [],
    )
    if case["case_type"] == "missing_information":
        return
    headings = list(_NORMAL_EVIDENCE_HEADINGS)
    if case["case_type"] == "policy_conflict":
        headings.extend(["refund-eligibility", "refund-exclusion"])
    evidence_refs = [evidence_by_heading[heading] for heading in headings]
    model.responses[("resolution", ticket.ticket_id, 1)] = ResolutionProposal(
        ticket_id=ticket.ticket_id,
        evidence_refs=evidence_refs,
        actions=[
            ActionProposal(
                action_type="CREATE_REFUND_REQUEST",
                params={"order_id": ticket.order_id, "amount": ticket.amount, "currency": ticket.currency},
            ),
            ActionProposal(
                action_type="SEND_REPLY",
                params={"message": "A fictional duplicate-charge request was recorded for review."},
            ),
        ],
        created_at=current,
    )
    model.responses[("reviewer", ticket.ticket_id, 1)] = RiskReview(
        escalated=False,
        rationale="Deterministic frozen evaluation review.",
    )


def _drive_expected_human_action(
    service: SupportFlowService, snapshot, case: dict[str, Any]
):
    action = case["human_action"]
    if action == "resume":
        return snapshot
    if action == "approve":
        return service.approve(snapshot.run_id, snapshot.proposal.proposal_hash, "evaluation-reviewer")
    if action == "modify_approve":
        revised = service.modify(
            snapshot.run_id,
            {"reply_text": "A fictional duplicate-charge request was revised for review."},
            "evaluation-reviewer",
        )
        return service.approve(revised.run_id, revised.proposal.proposal_hash, "evaluation-reviewer")
    if action == "reject":
        return service.reject(snapshot.run_id, "Fixture requires customer confirmation.", "evaluation-reviewer")
    if action == "escalate":
        return service.escalate(snapshot.run_id, "Fixture requires a human decision.", "evaluation-reviewer")
    if action == "approve_replay":
        approved = service.approve(snapshot.run_id, snapshot.proposal.proposal_hash, "evaluation-reviewer")
        return service.approve(approved.run_id, approved.proposal.proposal_hash, "evaluation-reviewer")
    raise ValueError(f"Unknown frozen evaluation human_action: {action}")


def _score_case(case: dict[str, Any], initial, resumed, final, service: SupportFlowService) -> dict[str, Any]:
    proposal = initial.proposal
    referenced = set(proposal.evidence_refs) if proposal is not None else set()
    available = [*initial.evidence.items, *initial.evidence.audit_items] if initial.evidence else []
    available_by_id = {item.evidence_id: item for item in available}
    required = set(case["required_evidence_ids"])
    evidence_hits = sum(evidence_id in referenced for evidence_id in required)
    citations_valid = bool(proposal) and all(
        evidence_id in available_by_id and available_by_id[evidence_id].active
        for evidence_id in referenced
    )
    actual_actions = {result.action_type.value for result in final.execution_results}
    action_correct = actual_actions <= set(case["allowed_actions"]) and not (
        actual_actions & set(case["forbidden_actions"])
    )
    unapproved = int(bool(final.execution_results) and final.approval is None)
    duplicate_side_effects = 0
    if case["human_action"] == "approve_replay":
        duplicate_side_effects = sum(result.status != "skipped_duplicate" for result in final.execution_results)
        duplicate_side_effects += max(0, service.repository.count_execution_side_effects() - 2)
    return {
        "case_id": case["case_id"],
        "route_correct": bool(initial.triage) and initial.triage.intent.value == case["expected_intent"],
        "required_evidence_hits": evidence_hits,
        "required_evidence_total": len(required),
        "citations_valid": citations_valid if proposal is not None else True,
        "action_correct": action_correct,
        "terminal_state_correct": final.current_state.value == case["expected_terminal_state"],
        "unapproved_execution_count": unapproved,
        "duplicate_side_effect_count": duplicate_side_effects,
        "recovery_failure_count": int(resumed.current_state != initial.current_state),
        "actual_terminal_state": final.current_state.value,
    }


def run_evaluation(
    dataset_path: Path,
    service_factory: Callable[..., SupportFlowService],
    output_dir: Path,
    policy_directory: Path | None = None,
) -> EvaluationSummary:
    """Evaluate every frozen case through submit/resume and public human-action methods."""
    if os.environ.get("LANGGRAPH_STRICT_MSGPACK", "").strip().lower() != "true":
        raise RuntimeError("Frozen evaluation requires LANGGRAPH_STRICT_MSGPACK=true")
    dataset_path = dataset_path.resolve()
    cases = _load_cases(dataset_path)
    policy_directory = (policy_directory or dataset_path.parent.parent / "policies").resolve()
    shared_index = _build_frozen_index(policy_directory)
    results: list[dict[str, Any]] = []
    for case in cases:
        service = service_factory(shared_index=shared_index)
        if service._checkpoint_connection is None:
            raise RuntimeError("Frozen evaluation requires a durable service runtime")
        service.graph.retriever = shared_index
        _configure_case_model(service, case, shared_index)
        initial = service.submit(Ticket.model_validate(case["ticket"]))
        resumed = service.resume(initial.run_id)
        final = _drive_expected_human_action(service, resumed, case)
        results.append(_score_case(case, initial, resumed, final, service))
    case_count = len(results)
    evidence_total = sum(result["required_evidence_total"] for result in results)
    summary = EvaluationSummary(
        dataset_sha256=_sha256_bytes(dataset_path.read_bytes()),
        route_accuracy=sum(result["route_correct"] for result in results) / case_count,
        required_evidence_hit_rate=(sum(result["required_evidence_hits"] for result in results) / evidence_total if evidence_total else 1.0),
        citation_validity=sum(result["citations_valid"] for result in results) / case_count,
        action_accuracy=sum(result["action_correct"] for result in results) / case_count,
        terminal_state_accuracy=sum(result["terminal_state_correct"] for result in results) / case_count,
        unapproved_execution_count=sum(result["unapproved_execution_count"] for result in results),
        duplicate_side_effect_count=sum(result["duplicate_side_effect_count"] for result in results),
        recovery_failure_count=sum(result["recovery_failure_count"] for result in results),
        policy_sha256=_policy_sha256(policy_directory),
        case_count=case_count,
    )
    write_reports(summary, results, output_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="supportflow-eval")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with TemporaryDirectory(prefix="supportflow-eval-") as temporary_directory:
        runtime_root = Path(temporary_directory)
        sequence = 0

        def service_factory(*, shared_index: RagRetriever) -> SupportFlowService:
            nonlocal sequence
            sequence += 1
            service = SupportFlowService.demo(
                as_of=datetime(2026, 8, 31, tzinfo=UTC),
                policy_directory=args.policies,
                runtime_directory=runtime_root / f"case-{sequence}",
            )
            service.graph.retriever = shared_index
            return service

        summary = run_evaluation(args.dataset, service_factory, args.output, args.policies)
    print(json.dumps(summary.__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
