from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from supportflow.agents.fake import ModelExhausted
from supportflow.agents.resolution import ResolutionAgent
from supportflow.agents.reviewer import RiskReviewerAgent
from supportflow.agents.triage import TriageAgent
from supportflow.domain.models import ApprovalInput, ApprovalRecord, RunError
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
    approvals: list
    revision_count: int
    terminal_state: str
    errors: list


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
            "RunError",
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
            try:
                result = self.triage.run(state["ticket"])
            except ModelExhausted as error:
                return self._needs_attention(state, "triage", error)
            return {"triage": result, "trace": self._append(state, "triage", result.intent.value)}

        def retrieve_node(state: WorkflowState) -> dict:
            ticket = state["ticket"]
            try:
                evidence = self.retriever.retrieve(
                    f"{ticket.subject} {ticket.body}",
                    state["triage"].intent.value,
                    self.as_of,
                    top_k=len(self.retriever.chunks),
                )
            except ValueError as error:
                return self._needs_attention(state, "retrieve", error)
            return {"evidence": evidence, "trace": self._append(state, "retrieve", "active policy evidence retrieved")}

        def resolve_node(state: WorkflowState) -> dict:
            try:
                proposal = self.resolution.run(state["ticket"], state["evidence"])
            except ModelExhausted as error:
                return self._needs_attention(state, "resolve", error)
            return {"proposal": proposal, "trace": self._append(state, "resolve", "resolution proposal created")}

        def review_node(state: WorkflowState) -> dict:
            try:
                review = self.reviewer.run(state["ticket"], state["proposal"], state["evidence"])
            except ModelExhausted as error:
                return self._needs_attention(state, "review", error)
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
            return {
                "approval": approval,
                "approvals": [*state.get("approvals", []), approval],
                "trace": self._append(state, "human_approval", input_value.reviewer),
            }

        def execute_node(state: WorkflowState) -> dict:
            results = self.executor.execute(state["run_id"], state["proposal"], state.get("approval"))
            return {"execution_results": results, "trace": self._append(state, "execute", "simulated actions completed")}

        def route_after_triage(state: WorkflowState) -> str:
            if state.get("terminal_state") or state["triage"].missing_fields:
                return END
            return "retrieve"

        def route_after_retrieve(state: WorkflowState) -> str:
            return END if state.get("terminal_state") else "resolve"

        def route_after_resolve(state: WorkflowState) -> str:
            return END if state.get("terminal_state") else "review"

        def route_after_review(state: WorkflowState) -> str:
            if state.get("terminal_state") or state["risk_review"].escalated:
                return END
            if state["policy_decision"].outcome == "escalate":
                return END
            if state.get("revision_count", 0) >= 2 and state["policy_decision"].outcome == "revise":
                return END
            return "human_approval" if state["policy_decision"].outcome == "allow" else END

        builder.add_node("triage", triage_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("resolve", resolve_node)
        builder.add_node("review", review_node)
        builder.add_node("policy", policy_node)
        builder.add_node("human_approval", human_approval_node)
        builder.add_node("execute", execute_node)
        builder.add_edge(START, "triage")
        builder.add_conditional_edges("triage", route_after_triage)
        builder.add_conditional_edges("retrieve", route_after_retrieve)
        builder.add_conditional_edges("resolve", route_after_resolve)
        builder.add_edge("review", "policy")
        builder.add_conditional_edges("policy", route_after_review)
        builder.add_edge("human_approval", "execute")
        builder.add_edge("execute", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _needs_attention(self, state: WorkflowState, stage: str, error: Exception) -> dict:
        return {
            "terminal_state": "NEEDS_ATTENTION",
            "errors": [*state.get("errors", []), RunError(code="MODEL_EXHAUSTED", message=str(error))],
            "trace": self._append(state, stage, "model output exhausted"),
        }

    def revise(
        self,
        config: dict,
        proposal: object,
        approvals: list[ApprovalRecord],
        revision_count: int,
        reviewer: str,
    ) -> None:
        state = self.compiled.get_state(config).values
        try:
            review = self.reviewer.run(state["ticket"], proposal, state["evidence"])
            decision = self.gate.evaluate(state["ticket"], state["evidence"], proposal, review)
            terminal_state = "ESCALATED" if revision_count > 2 else None
            detail = "revision limit reached" if terminal_state else f"revision requested by {reviewer}"
            self.compiled.update_state(
                config,
                {
                    "proposal": proposal,
                    "risk_review": review,
                    "policy_decision": decision,
                    "approvals": approvals,
                    "revision_count": revision_count,
                    "terminal_state": terminal_state,
                    "trace": self._append(state, "modify", detail),
                },
            )
        except ModelExhausted as error:
            self.compiled.update_state(config, self._needs_attention(state, "modify", error))

    def set_terminal(self, config: dict, state: str, reason: str, reviewer: str) -> None:
        values = self.compiled.get_state(config).values
        self.compiled.update_state(
            config,
            {
                "terminal_state": state,
                "trace": self._append(values, state.lower(), f"{reviewer}: {reason}"),
            },
        )
