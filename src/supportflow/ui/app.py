from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import streamlit as st

from supportflow.domain.models import RunSnapshot, Ticket, TicketState
from supportflow.workflow.service import SupportFlowService


@dataclass(frozen=True)
class WorkbenchAction:
    label: str


@dataclass(frozen=True)
class EvidenceRow:
    evidence_id: str
    version: str
    heading: str
    status: str
    score: float


@dataclass(frozen=True)
class TraceRow:
    stage: str
    occurred_at: str
    detail: str


@dataclass(frozen=True)
class ExecutionRow:
    action: str
    status: str
    reference: str
    error: str | None


@dataclass(frozen=True)
class WorkbenchViewModel:
    run_id: str
    state: str
    result: str
    rationale: str
    policy_outcome: str | None
    policy_rationale: str | None
    evidence_rows: tuple[EvidenceRow, ...]
    risks: tuple[str, ...]
    proposal_hash: str | None
    displayed_hash: str | None
    proposal_actions: tuple[str, ...]
    available_actions: tuple[WorkbenchAction, ...]
    unresolved_items: tuple[str, ...]
    trace_location: str
    trace_rows: tuple[TraceRow, ...]
    execution_rows: tuple[ExecutionRow, ...]


def build_view_model(snapshot: RunSnapshot) -> WorkbenchViewModel:
    """Translate one service snapshot into the information a human must review."""
    evidence = snapshot.evidence.items if snapshot.evidence is not None else []
    audit_evidence = snapshot.evidence.audit_items if snapshot.evidence is not None else []
    evidence_rows = tuple(
        EvidenceRow(
            evidence_id=item.evidence_id,
            version=item.version,
            heading=item.heading,
            status="active" if item.active else "inactive",
            score=item.score,
        )
        for item in [*evidence, *audit_evidence]
    )
    risks = tuple(
        item
        for item in (
            snapshot.risk_review.rationale if snapshot.risk_review is not None else None,
            *(snapshot.policy_decision.failed_rules if snapshot.policy_decision is not None else []),
        )
        if item
    )
    unresolved_items = tuple(
        [
            *(snapshot.triage.missing_fields if snapshot.triage is not None else []),
            *(f"{error.code}: {error.message}" for error in snapshot.errors),
        ]
    )
    proposal = snapshot.proposal
    action_labels = tuple(
        f"{action.action_type.value}: {action.params}"
        for action in (proposal.actions if proposal is not None else [])
    )
    available_actions = (
        (
            WorkbenchAction("Approve"),
            WorkbenchAction("Modify and re-review"),
            WorkbenchAction("Reject"),
            WorkbenchAction("Escalate"),
        )
        if snapshot.current_state == TicketState.WAITING_APPROVAL
        else ()
    )
    execution_rows = tuple(
        ExecutionRow(
            action=result.action_type.value,
            status=result.status,
            reference=result.reference,
            error=result.error,
        )
        for result in snapshot.execution_results
    )
    return WorkbenchViewModel(
        run_id=snapshot.run_id,
        state=snapshot.current_state.value,
        result=(snapshot.triage.intent.value if snapshot.triage is not None else "No resolution result yet"),
        rationale=(snapshot.triage.rationale if snapshot.triage is not None else "No triage rationale available."),
        policy_outcome=(snapshot.policy_decision.outcome if snapshot.policy_decision else None),
        policy_rationale=(snapshot.policy_decision.rationale if snapshot.policy_decision else None),
        evidence_rows=evidence_rows,
        risks=risks,
        proposal_hash=(proposal.proposal_hash if proposal is not None else None),
        displayed_hash=(proposal.proposal_hash if proposal is not None else None),
        proposal_actions=action_labels,
        available_actions=available_actions,
        unresolved_items=unresolved_items,
        trace_location=f"SupportFlow run {snapshot.run_id} / Trace",
        trace_rows=tuple(
            TraceRow(
                stage=event.stage,
                occurred_at=event.occurred_at.isoformat(),
                detail=event.detail,
            )
            for event in snapshot.trace
        ),
        execution_rows=execution_rows,
    )


def _service() -> SupportFlowService:
    if "supportflow_service" not in st.session_state:
        st.session_state.supportflow_service = SupportFlowService.demo()
    return st.session_state.supportflow_service


def _ticket_form(service: SupportFlowService) -> None:
    st.subheader("Submit a customer ticket")
    with st.form("ticket-form"):
        ticket_id = st.text_input("Ticket ID", value="ticket-duplicate-001")
        customer_id = st.text_input("Customer ID", value="customer-001")
        subject = st.text_input("Subject", value="I was charged twice")
        body = st.text_area(
            "Customer message",
            value="My order order-100 was charged twice for USD 29.00.",
        )
        order_id = st.text_input("Order ID", value="order-100")
        amount = st.text_input("Amount", value="29.00")
        currency = st.text_input("Currency", value="USD")
        submitted = st.form_submit_button("Review ticket")
    if not submitted:
        return
    snapshot = service.submit(
        Ticket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            subject=subject,
            body=body,
            order_id=order_id,
            amount=amount,
            currency=currency,
            created_at=datetime.now(UTC),
        )
    )
    st.session_state.run_id = snapshot.run_id
    st.session_state.displayed_proposal_hash = (
        snapshot.proposal.proposal_hash if snapshot.proposal is not None else None
    )


def _render_run(service: SupportFlowService, run_id: str) -> None:
    snapshot = service.snapshot(run_id)
    view = build_view_model(snapshot)
    displayed_hash = st.session_state.get("displayed_proposal_hash", view.proposal_hash)
    view = replace(view, displayed_hash=displayed_hash)

    with st.status(f"Run status: {view.state}", expanded=True):
        st.write(f"Result: {view.result}")
        st.write(f"Rationale: {view.rationale}")
        if view.policy_outcome:
            st.write(f"Policy Gate: {view.policy_outcome}")
            st.write(view.policy_rationale)
        if view.risks:
            st.write("Risks: " + "; ".join(view.risks))
        if view.unresolved_items:
            st.write("Unresolved items: " + "; ".join(view.unresolved_items))

    with st.expander("Evidence"):
        st.write("Evidence IDs and versions used for this decision")
        st.table([row.__dict__ for row in view.evidence_rows])

    if view.proposal_hash:
        st.subheader("Proposal")
        st.code(view.proposal_hash, language=None)
        for action in view.proposal_actions:
            st.write(action)
        st.caption("Available actions: " + ", ".join(action.label for action in view.available_actions))
        reply = next(
            (
                action.params["message"]
                for action in snapshot.proposal.actions
                if action.action_type.value == "SEND_REPLY"
            ),
            "",
        )
        with st.form("proposal-form"):
            reviewer = st.text_input("Reviewer", value="portfolio-owner")
            revised_reply = st.text_area("Reply text for modification", value=reply)
            reason = st.text_input("Reason for rejection or escalation", value="")
            approve_enabled = (
                view.policy_outcome == "allow"
                and view.displayed_hash == view.proposal_hash
                and view.state == TicketState.WAITING_APPROVAL.value
            )
            approve = st.form_submit_button("Approve", disabled=not approve_enabled)
            modify = st.form_submit_button(
                "Modify and re-review", disabled=view.state != TicketState.WAITING_APPROVAL.value
            )
            reject = st.form_submit_button("Reject", disabled=view.state != TicketState.WAITING_APPROVAL.value)
            escalate = st.form_submit_button("Escalate", disabled=view.state != TicketState.WAITING_APPROVAL.value)
        try:
            if approve:
                updated = service.approve(run_id, view.proposal_hash, reviewer)
            elif modify:
                updated = service.modify(run_id, {"reply_text": revised_reply}, reviewer)
            elif reject:
                updated = service.reject(run_id, reason or "Rejected by reviewer.", reviewer)
            elif escalate:
                updated = service.escalate(run_id, reason or "Escalated by reviewer.", reviewer)
            else:
                updated = None
        except (ValueError, KeyError) as error:
            st.error(str(error))
            updated = None
        if updated is not None:
            st.session_state.displayed_proposal_hash = (
                updated.proposal.proposal_hash if updated.proposal is not None else None
            )
            st.rerun()

    if view.execution_rows:
        st.subheader("Simulated execution")
        st.table([row.__dict__ for row in view.execution_rows])
    with st.expander("Trace"):
        st.write(f"Location: {view.trace_location}")
        st.table([row.__dict__ for row in view.trace_rows])


def main() -> None:
    st.set_page_config(page_title="SupportFlow Workbench", layout="wide")
    st.title("SupportFlow Workbench")
    st.caption("Submit, review, decide, and inspect a support run in one place.")
    service = _service()
    _ticket_form(service)
    run_id = st.session_state.get("run_id")
    if run_id:
        _render_run(service, run_id)


if __name__ == "__main__":
    main()
