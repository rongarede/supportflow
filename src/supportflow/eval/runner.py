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


class _EvaluationInterruption(RuntimeError):
    pass


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
        "forbidden_actions", "expected_actions", "expected_terminal_state", "human_action",
        "risk_flags", "input_revision",
    }
    if len(cases) != 30 or Counter(case.get("case_type") for case in cases) != _EXPECTED_DISTRIBUTION:
        raise ValueError("Frozen evaluation dataset must contain the specified 30-case distribution")
    if any(not required <= set(case) for case in cases):
        raise ValueError("Frozen evaluation case is missing required provenance or expectation fields")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Frozen evaluation case_id values must be unique")
    return cases


def _load_model_fixtures(dataset_path: Path) -> dict[str, Any]:
    fixtures = json.loads((dataset_path.parent / "model_fixtures.json").read_text(encoding="utf-8"))
    if set(fixtures) != {"profiles", "ticket_profiles"}:
        raise ValueError("Frozen model fixtures have an unexpected schema")
    return fixtures


def _evaluation_input_sha256(dataset_sha256: str, policy_sha256: str, fixture_sha256: str) -> str:
    canonical = json.dumps(
        {"fixture_sha256": fixture_sha256, "policy_sha256": policy_sha256, "tickets_sha256": dataset_sha256},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(canonical.encode("utf-8"))


def _configure_case_model(
    service: SupportFlowService, ticket: Ticket, fixtures: dict[str, Any], index: RagRetriever
) -> None:
    model = service.graph.triage.model
    if not isinstance(model, FakeStructuredModel):
        raise TypeError("Frozen evaluation requires the deterministic structured model")
    profile_name = fixtures["ticket_profiles"].get(ticket.ticket_id)
    if profile_name is None:
        raise ValueError(f"No frozen model fixture for {ticket.ticket_id}")
    profile = fixtures["profiles"][profile_name]
    current = datetime(2026, 8, 31, tzinfo=UTC)
    evidence_by_heading = {chunk.heading: chunk.evidence_id for chunk in index.chunks if chunk.document.active_at(current)}
    triage = profile["triage"]
    model.responses[("triage", ticket.ticket_id, 1)] = TriageResult(
        ticket_id=ticket.ticket_id,
        intent=triage["intent"],
        confidence=triage["confidence"],
        rationale=triage["rationale"],
        urgency=triage["urgency"],
        extracted_facts={
            field: getattr(ticket, field) for field in triage["extracted_fact_fields"]
        },
        missing_information=triage["missing_information"],
        risk_flags=triage["risk_flags"],
        route=triage["route"],
    )
    if triage["missing_information"]:
        return

    def resolve_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
        replacements = {
            "$order_id": ticket.order_id,
            "$amount": ticket.amount,
            "$currency": ticket.currency,
        }
        return {key: replacements.get(value, value) for key, value in parameters.items()}

    proposal_fixture = profile["proposal"]
    evidence_refs = [evidence_by_heading[heading] for heading in profile["evidence_headings"]]
    model.responses[("resolution", ticket.ticket_id, 1)] = ResolutionProposal(
        ticket_id=ticket.ticket_id,
        reply_text=proposal_fixture["reply_text"],
        evidence_refs=evidence_refs,
        actions=[
            ActionProposal(
                action_type=action["action_type"],
                parameters=resolve_parameters(action["parameters"]),
                reason=action["reason"],
                evidence_refs=[evidence_by_heading[heading] for heading in action["evidence_headings"]],
                risk_level=action["risk_level"],
            )
            for action in proposal_fixture["actions"]
        ],
        uncertainties=proposal_fixture["uncertainties"],
        created_at=current,
    )
    review = profile["review"]
    model.responses[("reviewer", ticket.ticket_id, 1)] = RiskReview(
        decision=review["decision"],
        risk_flags=review["risk_flags"],
        unsupported_claims=review["unsupported_claims"],
        required_changes=review["required_changes"],
        explanation=review["explanation"],
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


def _snapshot_signature(snapshot, service: SupportFlowService) -> dict[str, Any]:
    return {
        "state": snapshot.current_state.value,
        "evidence": snapshot.evidence.model_dump(mode="json") if snapshot.evidence else None,
        "proposal": snapshot.proposal.model_dump(mode="json") if snapshot.proposal else None,
        "node_attempts": snapshot.node_attempts,
        "trace_stages": [event.stage for event in snapshot.trace],
        "trace_count": len(snapshot.trace),
        "approvals": [approval.model_dump(mode="json") for approval in snapshot.approvals],
        "execution_results": [result.model_dump(mode="json") for result in snapshot.execution_results],
        "side_effect_count": service.repository.count_execution_side_effects(),
    }


def _recovery_matches(expected: dict[str, Any], reopened: dict[str, Any]) -> bool:
    return expected == reopened


def _submit_with_partial_checkpoint(
    service: SupportFlowService, ticket: Ticket, input_revision: str
):
    original = service.repository.record_node_result
    interrupted: dict[str, str] = {}

    def commit_then_interrupt(run_id, node_name, output, event, *, current_state, next_node):
        original(run_id, node_name, output, event, current_state=current_state, next_node=next_node)
        if node_name == "resolve":
            interrupted["run_id"] = run_id
            raise _EvaluationInterruption("frozen evaluator interruption after resolve journal commit")

    service.repository.record_node_result = commit_then_interrupt
    try:
        service.submit(ticket, input_revision=input_revision)
    except _EvaluationInterruption:
        return interrupted["run_id"]
    finally:
        service.repository.record_node_result = original
    raise AssertionError("Expected the frozen partial checkpoint interruption")


def _score_case(
    case: dict[str, Any],
    initial,
    final,
    service: SupportFlowService,
    recovery: dict[str, Any],
    *,
    duplicate_intake_reused: bool,
) -> dict[str, Any]:
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
    proposed_action_records = [
        {
            "action_type": action.action_type.value,
            "parameters": action.parameters,
        }
        for action in (proposal.actions if proposal else [])
    ]
    proposed_actions = [action["action_type"] for action in proposed_action_records]
    canonical_actions = Counter(
        json.dumps(action, sort_keys=True, separators=(",", ":"))
        for action in proposed_action_records
    )
    expected_actions = Counter(
        json.dumps(action, sort_keys=True, separators=(",", ":"))
        for action in case["expected_actions"]
    )
    action_correct = canonical_actions == expected_actions and not (
        set(proposed_actions) & set(case["forbidden_actions"])
    )
    unapproved = int(bool(final.execution_results) and final.approval is None)
    duplicate_side_effects = 0
    if case["human_action"] == "approve_replay":
        duplicate_side_effects = sum(result.status != "skipped_duplicate" for result in final.execution_results)
        duplicate_side_effects += max(
            0,
            service.repository.count_execution_side_effects()
            - len(case["expected_actions"]),
        )
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
        "recovery_failure_count": int(not recovery["reopened_snapshot_matches"]),
        "recovery": recovery,
        "observed": {
            "route": initial.triage.intent.value if initial.triage else None,
            "evidence_ids": sorted(referenced),
            "proposed_actions": proposed_action_records,
            "terminal_state": final.current_state.value,
            "trace_stages": [event.stage for event in final.trace],
            "trace_count": len(final.trace),
            "duplicate_intake_reused": duplicate_intake_reused,
        },
        "expected_vs_observed": {
            "intent": {"expected": case["expected_intent"], "observed": initial.triage.intent.value if initial.triage else None},
            "required_evidence_ids": {"expected": sorted(required), "observed": sorted(referenced)},
            "proposed_actions": {
                "expected": case["expected_actions"],
                "allowed": sorted(case["allowed_actions"]),
                "forbidden": sorted(case["forbidden_actions"]),
                "observed": proposed_action_records,
            },
            "terminal_state": {"expected": case["expected_terminal_state"], "observed": final.current_state.value},
        },
    }


def run_evaluation(
    dataset_path: Path,
    service_factory: Callable[..., SupportFlowService],
    output_dir: Path,
    policy_directory: Path | None = None,
    recovery_mutator: Callable[[SupportFlowService, dict[str, Any], Any], None] | None = None,
) -> EvaluationSummary:
    """Evaluate every frozen case through submit/resume and public human-action methods."""
    if os.environ.get("LANGGRAPH_STRICT_MSGPACK", "").strip().lower() != "true":
        raise RuntimeError("Frozen evaluation requires LANGGRAPH_STRICT_MSGPACK=true")
    dataset_path = dataset_path.resolve()
    cases = _load_cases(dataset_path)
    fixtures = _load_model_fixtures(dataset_path)
    fixture_sha256 = _sha256_bytes((dataset_path.parent / "model_fixtures.json").read_bytes())
    policy_directory = (policy_directory or dataset_path.parent.parent / "policies").resolve()
    policy_sha256 = _policy_sha256(policy_directory)
    dataset_sha256 = _sha256_bytes(dataset_path.read_bytes())
    shared_index = _build_frozen_index(policy_directory)
    results: list[dict[str, Any]] = []
    adapters: dict[str, str] | None = None
    with TemporaryDirectory(prefix="supportflow-eval-runtime-") as temporary_directory:
        runtime_root = Path(temporary_directory)
        for case in cases:
            runtime_directory = runtime_root / case["case_id"]
            ticket = Ticket.model_validate(case["ticket"])
            control = service_factory(shared_index=shared_index, runtime_directory=runtime_root / f"{case['case_id']}-control")
            control.graph.retriever = shared_index
            _configure_case_model(control, ticket, fixtures, shared_index)
            control_snapshot = control.submit(
                ticket, input_revision=case["input_revision"]
            )
            service = service_factory(shared_index=shared_index, runtime_directory=runtime_directory)
            if service._checkpoint_connection is None:
                raise RuntimeError("Frozen evaluation requires a durable service runtime")
            service.graph.retriever = shared_index
            _configure_case_model(service, ticket, fixtures, shared_index)
            runtime_adapters = {
                "model": type(service.graph.triage.model).__name__,
                "embedding": type(shared_index.embedding_provider).__name__,
            }
            if adapters is None:
                adapters = runtime_adapters
            elif adapters != runtime_adapters:
                raise RuntimeError("Frozen evaluation adapter provenance changed between cases")
            interrupted = case["case_id"] == "billing-01"
            if interrupted:
                run_id = _submit_with_partial_checkpoint(
                    service, ticket, case["input_revision"]
                )
                initial = None
            else:
                initial = service.submit(
                    ticket, input_revision=case["input_revision"]
                )
                run_id = initial.run_id
            duplicate_intake_reused = True
            if case["case_type"] == "duplicate_submission":
                duplicate = service.submit(
                    ticket, input_revision=case["input_revision"]
                )
                duplicate_intake_reused = duplicate.run_id == run_id
            reopened = service_factory(shared_index=shared_index, runtime_directory=runtime_directory)
            reopened.graph.retriever = shared_index
            _configure_case_model(reopened, ticket, fixtures, shared_index)
            recovered = reopened.resume(run_id)
            if recovery_mutator is not None:
                recovery_mutator(reopened, case, recovered)
                recovered = reopened.snapshot(run_id)
            expected_signature = _snapshot_signature(control_snapshot, control)
            observed_signature = _snapshot_signature(recovered, reopened)
            initial = recovered
            recovery = {
                "interrupted_partial_checkpoint": interrupted,
                "reopened_snapshot_matches": _recovery_matches(expected_signature, observed_signature),
                "node_attempts": observed_signature["node_attempts"],
                "trace_stages": observed_signature["trace_stages"],
                "trace_count": observed_signature["trace_count"],
                "approvals": observed_signature["approvals"],
                "execution_results": observed_signature["execution_results"],
                "side_effect_count": observed_signature["side_effect_count"],
            }
            final = _drive_expected_human_action(reopened, recovered, case)
            results.append(
                _score_case(
                    case,
                    initial,
                    final,
                    reopened,
                    recovery,
                    duplicate_intake_reused=duplicate_intake_reused,
                )
            )
    case_count = len(results)
    evidence_total = sum(result["required_evidence_total"] for result in results)
    summary = EvaluationSummary(
        dataset_sha256=dataset_sha256,
        fixture_sha256=fixture_sha256,
        evaluation_input_sha256=_evaluation_input_sha256(dataset_sha256, policy_sha256, fixture_sha256),
        route_accuracy=sum(result["route_correct"] for result in results) / case_count,
        required_evidence_hit_rate=(sum(result["required_evidence_hits"] for result in results) / evidence_total if evidence_total else 1.0),
        citation_validity=sum(result["citations_valid"] for result in results) / case_count,
        action_accuracy=sum(result["action_correct"] for result in results) / case_count,
        terminal_state_accuracy=sum(result["terminal_state_correct"] for result in results) / case_count,
        unapproved_execution_count=sum(result["unapproved_execution_count"] for result in results),
        duplicate_side_effect_count=sum(result["duplicate_side_effect_count"] for result in results),
        recovery_failure_count=sum(result["recovery_failure_count"] for result in results),
        policy_sha256=policy_sha256,
        case_count=case_count,
    )
    write_reports(summary, results, output_dir, adapters or {})
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="supportflow-eval")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def service_factory(*, shared_index: RagRetriever, runtime_directory: Path) -> SupportFlowService:
        service = SupportFlowService.demo(
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
            policy_directory=args.policies,
            runtime_directory=runtime_directory,
        )
        service.graph.retriever = shared_index
        return service

    summary = run_evaluation(args.dataset, service_factory, args.output, args.policies)
    print(json.dumps(summary.__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
