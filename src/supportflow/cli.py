from __future__ import annotations

import argparse
from datetime import UTC, datetime

from supportflow.domain.models import Ticket
from supportflow.workflow.service import SupportFlowService


def demo_golden() -> None:
    ticket = Ticket(
        ticket_id="ticket-duplicate-001",
        customer_id="customer-001",
        subject="I was charged twice",
        body="My order order-100 was charged twice for USD 29.00.",
        order_id="order-100",
        amount="29.00",
        currency="USD",
        created_at=datetime.now(UTC),
    )
    service = SupportFlowService.demo(use_sentence_transformer=True)
    waiting = service.submit(ticket)
    completed = service.approve(waiting.run_id, waiting.proposal.proposal_hash, "portfolio-owner")
    print(f"run_id: {completed.run_id}")
    print(f"evidence_ids: {', '.join(item.evidence_id for item in completed.evidence.items)}")
    print(f"proposal_hash: {completed.proposal.proposal_hash}")
    print(f"approval: {completed.approval.reviewer}")
    print(f"final_state: {completed.current_state.value}")
    print("simulated_actions: " + ", ".join(f"{item.action_type.value}={item.status}" for item in completed.execution_results))


def main() -> None:
    parser = argparse.ArgumentParser(prog="supportflow")
    parser.add_argument("command", choices=["demo-golden"])
    args = parser.parse_args()
    if args.command == "demo-golden":
        demo_golden()


if __name__ == "__main__":
    main()
