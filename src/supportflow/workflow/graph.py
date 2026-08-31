from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from supportflow.agents.protocols import ModelExhausted
from supportflow.agents.resolution import ResolutionAgent
from supportflow.agents.reviewer import RiskReviewerAgent
from supportflow.agents.triage import TriageAgent
from supportflow.domain.models import ApprovalInput, ApprovalRecord, RunError
from supportflow.execution.executor import DurableExecutor, InMemoryExecutor
from supportflow.policy.gate import PolicyGate
from supportflow.rag.retriever import RagRetriever
from supportflow.storage.repositories import SupportFlowRepository
from supportflow.workflow.nodes import run_model_node, trace


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
        executor: InMemoryExecutor | DurableExecutor,
        as_of: datetime | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        repository: SupportFlowRepository | None = None,
    ) -> None:
        self.triage = triage
        self.retriever = retriever
        self.resolution = resolution
        self.reviewer = reviewer
        self.gate = gate
        self.executor = executor
        self.repository = repository
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
        self.checkpointer = checkpointer or MemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("supportflow.domain.models", type_name) for type_name in domain_type_names
                ]
            )
        )
        self.compiled = self._compile()

    def _node_result(
        self,
        state: WorkflowState,
        node_name: str,
        output: dict,
        detail: str,
        *,
        current_state: str,
        next_node: str | None,
    ) -> dict:
        event = trace(node_name, detail)
        if self.repository is not None:
            self.repository.record_node_result(
                state["run_id"],
                node_name,
                output,
                event,
                current_state=current_state,
                next_node=next_node,
            )
        return {**output, "trace": [*state.get("trace", []), event]}

    def _cached_node_result(
        self, state: WorkflowState, node_name: str
    ) -> dict | None:
        if self.repository is None:
            return None
        persisted = self.repository.load_node_result(state["run_id"], node_name)
        if persisted is None:
            return None
        output, event = persisted
        existing_trace = state.get("trace", [])
        recovered_trace = (
            existing_trace
            if event in existing_trace
            else [*existing_trace, event]
        )
        return {**output, "trace": recovered_trace}

    def _compile(self):
        builder = StateGraph(WorkflowState)

        def triage_node(state: WorkflowState) -> dict:
            cached = self._cached_node_result(state, "triage")
            if cached is not None:
                return cached
            try:
                result = run_model_node(
                    lambda: self.triage.run(state["ticket"]),
                    node_name="triage",
                    run_id=state["run_id"],
                    repository=self.repository,
                )
            except ModelExhausted:
                return self._needs_attention(
                    state,
                    "triage",
                    code="MODEL_EXHAUSTED",
                    message="Model output remained unavailable after bounded retries.",
                    trace_detail="model output exhausted",
                )
            return self._node_result(
                state,
                "triage",
                {"triage": result},
                result.intent.value,
                current_state="TRIAGED",
                next_node=None if result.missing_fields else "retrieve",
            )

        def retrieve_node(state: WorkflowState) -> dict:
            cached = self._cached_node_result(state, "retrieve")
            if cached is not None:
                return cached
            ticket = state["ticket"]
            try:
                evidence = self.retriever.retrieve(
                    f"{ticket.subject} {ticket.body}",
                    state["triage"].intent.value,
                    self.as_of,
                    top_k=len(self.retriever.chunks),
                )
            except ValueError:
                return self._needs_attention(
                    state,
                    "retrieve",
                    code="RETRIEVAL_UNAVAILABLE",
                    message="Active policy evidence could not be retrieved.",
                    trace_detail="retrieval unavailable",
                )
            return self._node_result(
                state,
                "retrieve",
                {"evidence": evidence},
                "active policy evidence retrieved",
                current_state="EVIDENCE_READY",
                next_node="resolve",
            )

        def resolve_node(state: WorkflowState) -> dict:
            cached = self._cached_node_result(state, "resolve")
            if cached is not None:
                return cached
            try:
                proposal = run_model_node(
                    lambda: self.resolution.run(state["ticket"], state["evidence"]),
                    node_name="resolve",
                    run_id=state["run_id"],
                    repository=self.repository,
                )
            except ModelExhausted:
                return self._needs_attention(
                    state,
                    "resolve",
                    code="MODEL_EXHAUSTED",
                    message="Model output remained unavailable after bounded retries.",
                    trace_detail="model output exhausted",
                )
            return self._node_result(
                state,
                "resolve",
                {"proposal": proposal},
                "resolution proposal created",
                current_state="RESOLUTION_PROPOSED",
                next_node="review",
            )

        def review_node(state: WorkflowState) -> dict:
            cached = self._cached_node_result(state, "review")
            if cached is not None:
                return cached
            try:
                review = run_model_node(
                    lambda: self.reviewer.run(state["ticket"], state["proposal"], state["evidence"]),
                    node_name="review",
                    run_id=state["run_id"],
                    repository=self.repository,
                )
            except ModelExhausted:
                return self._needs_attention(
                    state,
                    "review",
                    code="MODEL_EXHAUSTED",
                    message="Model output remained unavailable after bounded retries.",
                    trace_detail="model output exhausted",
                )
            return self._node_result(
                state,
                "review",
                {"risk_review": review},
                "risk review completed",
                current_state="RISK_REVIEWED",
                next_node="policy",
            )

        def policy_node(state: WorkflowState) -> dict:
            cached = self._cached_node_result(state, "policy")
            if cached is not None:
                return cached
            decision = self.gate.evaluate(state["ticket"], state["evidence"], state["proposal"], state["risk_review"])
            next_node = (
                "human_approval"
                if decision.outcome == "allow" and not state["risk_review"].escalated
                else None
            )
            return self._node_result(
                state,
                "policy",
                {"policy_decision": decision},
                decision.outcome,
                current_state="POLICY_CHECKED",
                next_node=next_node,
            )

        def human_approval_node(state: WorkflowState) -> dict:
            payload = interrupt({"run_id": state["run_id"], "proposal_hash": state["proposal"].proposal_hash})
            input_value = ApprovalInput.model_validate(payload)
            approval = ApprovalRecord(
                run_id=state["run_id"], proposal_hash=input_value.proposal_hash, reviewer=input_value.reviewer, approved_at=datetime.now(UTC)
            )
            event = trace("human_approval", input_value.reviewer)
            if self.repository is not None:
                approval = self.repository.save_approval(approval)
                self.repository.trace.append(state["run_id"], event)
                self.repository.mark_run_state(state["run_id"], "APPROVED", "execute")
            return {
                "approval": approval,
                "approvals": [*state.get("approvals", []), approval],
                "trace": [*state.get("trace", []), event],
            }

        def execute_node(state: WorkflowState) -> dict:
            results = self.executor.execute(state["run_id"], state["proposal"], state.get("approval"))
            event = trace("execute", "simulated actions completed")
            if self.repository is not None:
                self.repository.trace.append(state["run_id"], event)
                self.repository.mark_run_state(state["run_id"], "COMPLETED")
            return {"execution_results": results, "trace": [*state.get("trace", []), event]}

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

    def _needs_attention(
        self,
        state: WorkflowState,
        stage: str,
        *,
        code: str,
        message: str,
        trace_detail: str,
    ) -> dict:
        event = trace(stage, trace_detail)
        if self.repository is not None:
            self.repository.trace.append(state["run_id"], event)
            self.repository.mark_run_state(state["run_id"], "NEEDS_ATTENTION")
        return {
            "terminal_state": "NEEDS_ATTENTION",
            "errors": [*state.get("errors", []), RunError(code=code, message=message)],
            "trace": [*state.get("trace", []), event],
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
            review = run_model_node(
                lambda: self.reviewer.run(state["ticket"], proposal, state["evidence"]),
                node_name="modify",
                run_id=state["run_id"],
                repository=self.repository,
            )
            decision = self.gate.evaluate(state["ticket"], state["evidence"], proposal, review)
            terminal_state = "ESCALATED" if revision_count > 2 else None
            detail = "revision limit reached" if terminal_state else f"revision requested by {reviewer}"
            event = trace("modify", detail)
            if self.repository is not None:
                for approval in approvals:
                    self.repository.save_approval(approval)
                self.repository.trace.append(state["run_id"], event)
                self.repository.mark_run_state(
                    state["run_id"], terminal_state or "WAITING_APPROVAL"
                )
            self.compiled.update_state(
                config,
                {
                    "proposal": proposal,
                    "risk_review": review,
                    "policy_decision": decision,
                    "approvals": approvals,
                    "revision_count": revision_count,
                    "terminal_state": terminal_state,
                    "trace": [*state.get("trace", []), event],
                },
            )
        except ModelExhausted:
            self.compiled.update_state(
                config,
                self._needs_attention(
                    state,
                    "modify",
                    code="MODEL_EXHAUSTED",
                    message="Model output remained unavailable after bounded retries.",
                    trace_detail="model output exhausted",
                ),
            )

    def set_terminal(self, config: dict, state: str, reason: str, reviewer: str) -> None:
        values = self.compiled.get_state(config).values
        event = trace(state.lower(), f"{reviewer}: {reason}")
        if self.repository is not None:
            self.repository.trace.append(values["run_id"], event)
            self.repository.mark_run_state(values["run_id"], state)
        self.compiled.update_state(
            config,
            {
                "terminal_state": state,
                "trace": [*values.get("trace", []), event],
            },
        )

    def record_execution_replay(self, config: dict, results: list) -> None:
        values = self.compiled.get_state(config).values
        event = trace("execute", "duplicate simulated actions skipped")
        if self.repository is not None:
            self.repository.trace.append(values["run_id"], event)
            self.repository.mark_run_state(values["run_id"], "COMPLETED")
        self.compiled.update_state(
            config,
            {
                "execution_results": results,
                "trace": [*values.get("trace", []), event],
            },
        )
