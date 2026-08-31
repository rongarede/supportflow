from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from supportflow.agents.fake import FakeStructuredModel
from supportflow.agents.resolution import ResolutionAgent
from supportflow.agents.reviewer import RiskReviewerAgent
from supportflow.agents.triage import TriageAgent
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
from supportflow.execution.executor import InMemoryExecutor
from supportflow.policy.gate import PolicyGate
from supportflow.rag.documents import load_policy_documents
from supportflow.rag.embeddings import FixedEmbeddingProvider, SentenceTransformerEmbeddingProvider
from supportflow.rag.index import build_policy_chunks
from supportflow.rag.retriever import RagRetriever
from supportflow.settings import DEFAULT_POLICY_DIRECTORY
from supportflow.workflow.graph import SupportFlowGraph


class SupportFlowService:
    def __init__(self, graph: SupportFlowGraph) -> None:
        self.graph = graph

    @classmethod
    def demo(
        cls,
        as_of: datetime | None = None,
        policy_directory: Path = DEFAULT_POLICY_DIRECTORY,
        use_sentence_transformer: bool = False,
    ) -> SupportFlowService:
        current = (as_of or datetime.now(UTC)).astimezone(UTC)
        chunks = build_policy_chunks(load_policy_documents(policy_directory))
        provider = (
            SentenceTransformerEmbeddingProvider()
            if use_sentence_transformer
            else FixedEmbeddingProvider({chunk.text: [1.0, 0.0] for chunk in chunks})
        )
        retriever = RagRetriever(chunks, provider)
        evidence_ids_by_heading = {
            chunk.heading: chunk.evidence_id for chunk in chunks if chunk.document.active_at(current)
        }
        duplicate_charge_refs = [
            evidence_ids_by_heading["duplicate-charge-verification"],
            evidence_ids_by_heading["duplicate-charge-refund-request"],
            evidence_ids_by_heading["refund-timing"],
        ]
        sample_ticket_id = "ticket-duplicate-001"
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
            }
        )
        graph = SupportFlowGraph(
            triage=TriageAgent(model),
            retriever=retriever,
            resolution=ResolutionAgent(model),
            reviewer=RiskReviewerAgent(model),
            gate=PolicyGate(),
            executor=InMemoryExecutor(),
            as_of=current,
        )
        return cls(graph)

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
        else:
            state = TicketState.WAITING_APPROVAL
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
        )

    def submit(self, ticket: Ticket) -> RunSnapshot:
        run_id = str(uuid4())
        self.graph.compiled.invoke({"run_id": run_id, "ticket": ticket, "trace": []}, self._config(run_id))
        return self._snapshot(run_id)

    def approve(self, run_id: str, proposal_hash: str, reviewer: str) -> RunSnapshot:
        waiting = self._snapshot(run_id)
        if waiting.current_state != TicketState.WAITING_APPROVAL or waiting.proposal is None:
            raise ApprovalMismatch("Run is not waiting for an approvable proposal")
        if waiting.proposal.proposal_hash != proposal_hash:
            raise ApprovalMismatch("Approval proposal hash does not match the reviewed proposal")
        approval = ApprovalInput(proposal_hash=proposal_hash, reviewer=reviewer)
        self.graph.compiled.invoke(Command(resume=approval.model_dump(mode="json")), self._config(run_id))
        return self._snapshot(run_id)

    def modify(self, run_id: str, edits: dict[str, str], reviewer: str) -> RunSnapshot:
        waiting = self._snapshot(run_id)
        if waiting.current_state != TicketState.WAITING_APPROVAL or waiting.proposal is None:
            raise ApprovalMismatch("Run is not waiting for a modifiable proposal")
        reply_text = edits.get("reply_text")
        if not reply_text or set(edits) != {"reply_text"}:
            raise ValueError("Only a non-empty reply_text edit is supported")
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
