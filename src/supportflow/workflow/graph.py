from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from supportflow.agents.resolution import ResolutionAgent
from supportflow.agents.reviewer import RiskReviewerAgent
from supportflow.agents.triage import TriageAgent
from supportflow.domain.models import ApprovalInput, ApprovalRecord
from supportflow.execution.executor import InMemoryExecutor
from supportflow.policy.gate import PolicyGate
from supportflow.rag.retriever import RagRetriever
from supportflow.workflow.nodes import trace


class WorkflowState(TypedDict, total=False):
    run_id: str
    ticket: object
    triage: object
    evidence: object
    proposal: object
    risk_review: object
    policy_decision: object
    approval: object
    execution_results: list
    trace: list


class SupportFlowGraph:
    def __init__(
        self,
        triage: TriageAgent,
        retriever: RagRetriever,
        resolution: ResolutionAgent,
        reviewer: RiskReviewerAgent,
        gate: PolicyGate,
        executor: InMemoryExecutor,
        as_of: datetime | None = None,
    ) -> None:
        self.triage = triage
        self.retriever = retriever
        self.resolution = resolution
        self.reviewer = reviewer
        self.gate = gate
        self.executor = executor
        self.as_of = (as_of or datetime.now(UTC)).astimezone(UTC)
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
        )
        self.checkpointer = MemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("supportflow.domain.models", type_name) for type_name in domain_type_names
                ]
            )
        )
        self.compiled = self._compile()

    @staticmethod
    def _append(state: WorkflowState, stage: str, detail: str) -> list:
        return [*state.get("trace", []), trace(stage, detail)]

    def _compile(self):
        builder = StateGraph(WorkflowState)

        def triage_node(state: WorkflowState) -> dict:
            result = self.triage.run(state["ticket"])
            return {"triage": result, "trace": self._append(state, "triage", result.intent.value)}

        def retrieve_node(state: WorkflowState) -> dict:
            ticket = state["ticket"]
            evidence = self.retriever.retrieve(
                f"{ticket.subject} {ticket.body}", state["triage"].intent.value, self.as_of
            )
            return {"evidence": evidence, "trace": self._append(state, "retrieve", "active policy evidence retrieved")}

        def resolve_node(state: WorkflowState) -> dict:
            proposal = self.resolution.run(state["ticket"], state["evidence"])
            return {"proposal": proposal, "trace": self._append(state, "resolve", "resolution proposal created")}

        def review_node(state: WorkflowState) -> dict:
            review = self.reviewer.run(state["ticket"], state["proposal"], state["evidence"])
            return {"risk_review": review, "trace": self._append(state, "review", "risk review completed")}

        def policy_node(state: WorkflowState) -> dict:
            decision = self.gate.evaluate(state["ticket"], state["evidence"], state["proposal"], state["risk_review"])
            return {"policy_decision": decision, "trace": self._append(state, "policy", decision.outcome)}

        def human_approval_node(state: WorkflowState) -> dict:
            payload = interrupt({"run_id": state["run_id"], "proposal_hash": state["proposal"].proposal_hash})
            input_value = ApprovalInput.model_validate(payload)
            approval = ApprovalRecord(
                run_id=state["run_id"], proposal_hash=input_value.proposal_hash, reviewer=input_value.reviewer, approved_at=datetime.now(UTC)
            )
            return {"approval": approval, "trace": self._append(state, "human_approval", input_value.reviewer)}

        def execute_node(state: WorkflowState) -> dict:
            results = self.executor.execute(state["run_id"], state["proposal"], state.get("approval"))
            return {"execution_results": results, "trace": self._append(state, "execute", "simulated actions completed")}

        def route_policy(state: WorkflowState) -> str:
            return "human_approval" if state["policy_decision"].outcome == "allow" else END

        builder.add_node("triage", triage_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("resolve", resolve_node)
        builder.add_node("review", review_node)
        builder.add_node("policy", policy_node)
        builder.add_node("human_approval", human_approval_node)
        builder.add_node("execute", execute_node)
        builder.add_edge(START, "triage")
        builder.add_edge("triage", "retrieve")
        builder.add_edge("retrieve", "resolve")
        builder.add_edge("resolve", "review")
        builder.add_edge("review", "policy")
        builder.add_conditional_edges("policy", route_policy)
        builder.add_edge("human_approval", "execute")
        builder.add_edge("execute", END)
        return builder.compile(checkpointer=self.checkpointer)
