from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from supportflow.agents.fake import FakeStructuredModel
from supportflow.agents.protocols import StructuredModel
from supportflow.agents.resolution import ResolutionAgent
from supportflow.agents.reviewer import RiskReviewerAgent
from supportflow.agents.triage import TriageAgent
from supportflow.domain.hashing import proposal_hash as canonical_proposal_hash
from supportflow.domain.models import (
    ActionProposal,
    ApprovalInput,
    ApprovalMismatch,
    ApprovalRecord,
    ResolutionProposal,
    RiskReview,
    RunSnapshot,
    Ticket,
    TicketState,
    TriageResult,
)
from supportflow.execution.executor import DurableExecutor
from supportflow.policy.gate import PolicyGate
from supportflow.rag.documents import load_policy_documents
from supportflow.rag.embeddings import FixedEmbeddingProvider, SentenceTransformerEmbeddingProvider
from supportflow.rag.index import build_persisted_policy_index, build_policy_chunks
from supportflow.rag.retriever import RagRetriever
from supportflow.settings import (
    DEFAULT_POLICY_DIRECTORY,
    checkpoint_database_path,
    runtime_database_path,
)
from supportflow.storage.database import SupportFlowDatabase
from supportflow.storage.repositories import SupportFlowRepository
from supportflow.workflow.graph import SupportFlowGraph


class SupportFlowService:
    def __init__(
        self,
        graph: SupportFlowGraph,
        repository: SupportFlowRepository,
        checkpoint_connection: sqlite3.Connection | None = None,
    ) -> None:
        self.graph = graph
        self.repository = repository
        self._checkpoint_connection = checkpoint_connection

    @classmethod
    def demo(
        cls,
        as_of: datetime | None = None,
        policy_directory: Path = DEFAULT_POLICY_DIRECTORY,
        use_sentence_transformer: bool = False,
        runtime_directory: Path | None = None,
        model: StructuredModel | None = None,
    ) -> SupportFlowService:
        current = (as_of or datetime.now(UTC)).astimezone(UTC)
        if (
            runtime_directory is not None
            and os.environ.get("LANGGRAPH_STRICT_MSGPACK", "").strip().lower()
            != "true"
        ):
            raise RuntimeError(
                "Durable checkpoints require LANGGRAPH_STRICT_MSGPACK=true"
            )
        documents = load_policy_documents(policy_directory)
        uncached_chunks = build_policy_chunks(documents)
        provider = (
            SentenceTransformerEmbeddingProvider()
            if use_sentence_transformer
            else FixedEmbeddingProvider({chunk.text: [1.0, 0.0] for chunk in uncached_chunks})
        )
        checkpoint_connection = None
        checkpointer = None
        if runtime_directory is None:
            database = SupportFlowDatabase(":memory:")
        else:
            runtime_directory.mkdir(parents=True, exist_ok=True)
            database = SupportFlowDatabase(runtime_database_path(runtime_directory))
            checkpoint_connection = sqlite3.connect(
                checkpoint_database_path(runtime_directory),
                check_same_thread=False,
                timeout=30,
            )
            checkpoint_connection.execute("PRAGMA busy_timeout = 30000")
            domain_type_names = (
                "Ticket",
                "TraceEvent",
                "Intent",
                "TriageResult",
                "EvidenceItem",
                "EvidenceBundle",
                "ActionType",
                "ActionProposal",
                "ResolutionProposal",
                "RiskReview",
                "PolicyDecision",
                "ApprovalRecord",
                "ExecutionResult",
                "RunError",
            )
            checkpointer = SqliteSaver(
                checkpoint_connection,
                serde=JsonPlusSerializer(
                    pickle_fallback=False,
                    allowed_msgpack_modules=[
                        ("supportflow.domain.models", type_name)
                        for type_name in domain_type_names
                    ],
                ),
            )
        repository = SupportFlowRepository(database)
        chunks, embeddings = build_persisted_policy_index(documents, provider, repository)
        retriever = RagRetriever(chunks, provider, embeddings=embeddings)
        evidence_ids_by_heading = {
            chunk.heading: chunk.evidence_id for chunk in chunks if chunk.document.active_at(current)
        }
        audit_evidence_ids_by_heading = {
            chunk.heading: chunk.evidence_id for chunk in chunks if not chunk.document.active_at(current)
        }
        duplicate_charge_refs = [
            evidence_ids_by_heading["duplicate-charge-verification"],
            evidence_ids_by_heading["duplicate-charge-refund-request"],
            evidence_ids_by_heading["refund-timing"],
        ]
        sample_ticket_id = "ticket-duplicate-001"
        if model is None:
            model = FakeStructuredModel(
                {
                ("triage", sample_ticket_id, 1): TriageResult(
                    ticket_id=sample_ticket_id,
                    intent="DUPLICATE_CHARGE",
                    confidence=0.99,
                    rationale="The customer describes a duplicate charge for one order.",
                ),
                ("resolution", sample_ticket_id, 1): ResolutionProposal(
                    ticket_id=sample_ticket_id,
                    evidence_refs=duplicate_charge_refs,
                    actions=[
                        ActionProposal(
                            action_type="CREATE_REFUND_REQUEST",
                            params={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
                        ),
                        ActionProposal(
                            action_type="SEND_REPLY",
                            params={"message": "We verified the duplicate charge and submitted a refund request."},
                        ),
                    ],
                    created_at=current,
                ),
                ("reviewer", sample_ticket_id, 1): RiskReview(
                    escalated=False,
                    rationale="The proposed simulated actions are constrained to active billing policies.",
                ),
                ("triage", "T-MISSING-001", 1): TriageResult(
                    ticket_id="T-MISSING-001",
                    intent="DUPLICATE_CHARGE",
                    confidence=0.99,
                    rationale="The order details needed for a payment investigation are incomplete.",
                    missing_fields=["payment provider failure details"],
                ),
                ("triage", "T-CONFLICT-001", 1): TriageResult(
                    ticket_id="T-CONFLICT-001",
                    intent="DUPLICATE_CHARGE",
                    confidence=0.99,
                    rationale="The ticket describes a duplicate charge with an unresolved exclusion.",
                ),
                ("resolution", "T-CONFLICT-001", 1): ResolutionProposal(
                    ticket_id="T-CONFLICT-001",
                    evidence_refs=[
                        *duplicate_charge_refs,
                        evidence_ids_by_heading["refund-eligibility"],
                        evidence_ids_by_heading["refund-exclusion"],
                    ],
                    actions=[
                        ActionProposal(
                            action_type="CREATE_REFUND_REQUEST",
                            params={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
                        ),
                        ActionProposal(action_type="SEND_REPLY", params={"message": "We are reviewing the request."}),
                    ],
                    created_at=current,
                ),
                ("reviewer", "T-CONFLICT-001", 1): RiskReview(
                    escalated=False,
                    rationale="Eligibility and exclusion evidence conflict and need human resolution.",
                ),
                ("triage", "T-RISK-001", 1): TriageResult(
                    ticket_id="T-RISK-001",
                    intent="DUPLICATE_CHARGE",
                    confidence=0.99,
                    rationale="The ticket describes a duplicate charge with a risk indicator.",
                ),
                ("resolution", "T-RISK-001", 1): ResolutionProposal(
                    ticket_id="T-RISK-001",
                    evidence_refs=duplicate_charge_refs,
                    actions=[
                        ActionProposal(
                            action_type="CREATE_REFUND_REQUEST",
                            params={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
                        ),
                        ActionProposal(action_type="SEND_REPLY", params={"message": "We are reviewing the request."}),
                    ],
                    created_at=current,
                ),
                ("reviewer", "T-RISK-001", 1): RiskReview(
                    escalated=True,
                    rationale="Potential account takeover requires a human investigator.",
                ),
                ("triage", "T-EXPIRED-001", 1): TriageResult(
                    ticket_id="T-EXPIRED-001",
                    intent="DUPLICATE_CHARGE",
                    confidence=0.99,
                    rationale="The ticket cites a previously applicable duplicate-charge rule.",
                ),
                ("resolution", "T-EXPIRED-001", 1): ResolutionProposal(
                    ticket_id="T-EXPIRED-001",
                    evidence_refs=[
                        evidence_ids_by_heading["duplicate-charge-verification"],
                        audit_evidence_ids_by_heading["duplicate-charge-refund-request"],
                        evidence_ids_by_heading["refund-timing"],
                    ],
                    actions=[
                        ActionProposal(
                            action_type="CREATE_REFUND_REQUEST",
                            params={"order_id": "order-100", "amount": "29.00", "currency": "USD"},
                        ),
                        ActionProposal(action_type="SEND_REPLY", params={"message": "We are reviewing the request."}),
                    ],
                    created_at=current,
                ),
                ("reviewer", "T-EXPIRED-001", 1): RiskReview(
                    escalated=False,
                    rationale="The cited refund rule has expired and needs a human decision.",
                ),
                }
            )
        graph = SupportFlowGraph(
            triage=TriageAgent(model),
            retriever=retriever,
            resolution=ResolutionAgent(model),
            reviewer=RiskReviewerAgent(model),
            gate=PolicyGate(),
            executor=DurableExecutor(repository),
            as_of=current,
            checkpointer=checkpointer,
            repository=repository,
        )
        return cls(graph, repository, checkpoint_connection)

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def _snapshot(self, run_id: str) -> RunSnapshot:
        values = self.graph.compiled.get_state(self._config(run_id)).values
        decision = values.get("policy_decision")
        if values.get("terminal_state"):
            state = TicketState(values["terminal_state"])
        elif values.get("triage") and values["triage"].missing_fields:
            state = TicketState.WAITING_CUSTOMER
        elif values.get("risk_review") and values["risk_review"].escalated:
            state = TicketState.ESCALATED
        elif decision is not None and decision.outcome == "escalate":
            state = TicketState.ESCALATED
        elif values.get("execution_results"):
            state = TicketState.COMPLETED
        elif decision is not None and decision.outcome == "block":
            state = TicketState.BLOCKED
        elif (
            decision is not None
            and decision.outcome == "allow"
            and values.get("proposal") is not None
            and values.get("risk_review") is not None
        ):
            state = TicketState.WAITING_APPROVAL
        else:
            state = TicketState.NEEDS_ATTENTION
        return RunSnapshot(
            run_id=run_id,
            current_state=state,
            ticket=values["ticket"],
            triage=values.get("triage"),
            evidence=values.get("evidence"),
            proposal=values.get("proposal"),
            risk_review=values.get("risk_review"),
            policy_decision=decision,
            approval=values.get("approval"),
            approvals=values.get("approvals", []),
            execution_results=values.get("execution_results", []),
            trace=values.get("trace", []),
            errors=values.get("errors", []),
            node_attempts=self.repository.node_attempts(run_id),
        )

    def submit(self, ticket: Ticket) -> RunSnapshot:
        run_id = str(uuid4())
        self.repository.create_run(run_id, ticket)
        self.graph.compiled.invoke({"run_id": run_id, "ticket": ticket, "trace": []}, self._config(run_id))
        snapshot = self._snapshot(run_id)
        self.repository.mark_run_state(snapshot.run_id, snapshot.current_state.value)
        return snapshot

    def approve(self, run_id: str, proposal_hash: str, reviewer: str) -> RunSnapshot:
        waiting = self._snapshot(run_id)
        if waiting.proposal is None:
            raise ApprovalMismatch("Run is not waiting for an approvable proposal")
        canonical_hash = canonical_proposal_hash(waiting.proposal)
        if (
            waiting.proposal.proposal_hash != canonical_hash
            or proposal_hash != canonical_hash
        ):
            raise ApprovalMismatch("Approval proposal hash does not match the reviewed proposal")
        if waiting.current_state == TicketState.COMPLETED:
            approval = self.repository.get_approval(run_id, canonical_hash)
            results = self.graph.executor.execute(run_id, waiting.proposal, approval)
            self.graph.record_execution_replay(self._config(run_id), results)
            return self._snapshot(run_id)
        if waiting.current_state != TicketState.WAITING_APPROVAL:
            raise ApprovalMismatch("Run is not waiting for an approvable proposal")
        approval = ApprovalInput(proposal_hash=proposal_hash, reviewer=reviewer)
        self.graph.compiled.invoke(Command(resume=approval.model_dump(mode="json")), self._config(run_id))
        snapshot = self._snapshot(run_id)
        self.repository.mark_run_state(snapshot.run_id, snapshot.current_state.value)
        return snapshot

    def modify(self, run_id: str, edits: dict[str, str], reviewer: str) -> RunSnapshot:
        waiting = self._snapshot(run_id)
        if waiting.current_state != TicketState.WAITING_APPROVAL or waiting.proposal is None:
            raise ApprovalMismatch("Run is not waiting for a modifiable proposal")
        reply_text = edits.get("reply_text")
        if not reply_text or set(edits) != {"reply_text"}:
            raise ValueError("Only a non-empty reply_text edit is supported")
        reply_actions = [action for action in waiting.proposal.actions if action.action_type == "SEND_REPLY"]
        if not reply_actions:
            raise ValueError("Proposal has no editable reply")
        if all(action.params["message"] == reply_text for action in reply_actions):
            raise ValueError("reply_text must change the proposal")
        revised_actions = [
            ActionProposal(
                action_type=action.action_type,
                params={**action.params, "message": reply_text}
                if action.action_type == "SEND_REPLY"
                else action.params,
            )
            for action in waiting.proposal.actions
        ]
        revised = ResolutionProposal(
            ticket_id=waiting.proposal.ticket_id,
            evidence_refs=waiting.proposal.evidence_refs,
            actions=revised_actions,
            created_at=waiting.proposal.created_at,
        )
        if revised.proposal_hash == waiting.proposal.proposal_hash:
            raise ValueError("reply_text must change the proposal")
        superseded = ApprovalRecord(
            run_id=run_id,
            proposal_hash=waiting.proposal.proposal_hash,
            reviewer=reviewer,
            approved_at=datetime.now(UTC),
            status="superseded",
        )
        revision_count = self.graph.compiled.get_state(self._config(run_id)).values.get("revision_count", 0) + 1
        self.graph.revise(
            self._config(run_id),
            revised,
            [*waiting.approvals, superseded],
            revision_count,
            reviewer,
        )
        return self._snapshot(run_id)

    def reject(self, run_id: str, reason: str, reviewer: str) -> RunSnapshot:
        waiting = self._snapshot(run_id)
        if waiting.current_state != TicketState.WAITING_APPROVAL:
            raise ApprovalMismatch("Run is not waiting for rejection")
        self.graph.set_terminal(self._config(run_id), "REJECTED", reason, reviewer)
        return self._snapshot(run_id)

    def escalate(self, run_id: str, reason: str, reviewer: str) -> RunSnapshot:
        waiting = self._snapshot(run_id)
        if waiting.current_state != TicketState.WAITING_APPROVAL:
            raise ApprovalMismatch("Run is not waiting for escalation")
        self.graph.set_terminal(self._config(run_id), "ESCALATED", reason, reviewer)
        return self._snapshot(run_id)

    def snapshot(self, run_id: str) -> RunSnapshot:
        return self._snapshot(run_id)

    def resume(self, run_id: str) -> RunSnapshot:
        if not self.repository.run_exists(run_id):
            raise KeyError(f"Unknown run_id: {run_id}")
        config = self._config(run_id)
        checkpoint = self.graph.compiled.get_state(config)
        recoverable_nodes = {"triage", "retrieve", "resolve", "review", "policy"}
        if any(
            node_name in recoverable_nodes
            and self.repository.load_node_result(run_id, node_name) is not None
            for node_name in checkpoint.next
        ):
            self.graph.compiled.invoke(None, config)
        snapshot = self._snapshot(run_id)
        self.repository.mark_run_state(snapshot.run_id, snapshot.current_state.value)
        return snapshot
