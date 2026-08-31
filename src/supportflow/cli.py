from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from supportflow.agents.openai_adapter import OpenAICompatibleStructuredModel
from supportflow.domain.models import Ticket
from supportflow.eval.runner import main as evaluation_main
from supportflow.settings import (
    DEFAULT_RESTART_DEMO_DIRECTORY,
    checkpoint_database_path,
    runtime_database_path,
)
from supportflow.workflow.service import SupportFlowService


def _golden_ticket() -> Ticket:
    return Ticket(
        ticket_id="ticket-duplicate-001",
        customer_id="customer-001",
        subject="I was charged twice",
        body="My order order-100 was charged twice for USD 29.00.",
        order_id="order-100",
        amount="29.00",
        currency="USD",
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


def demo_golden(model_adapter: str = "fake") -> None:
    ticket = _golden_ticket()
    model = OpenAICompatibleStructuredModel() if model_adapter == "openai" else None
    service = SupportFlowService.demo(use_sentence_transformer=True, model=model)
    waiting = service.submit(ticket)
    completed = service.approve(waiting.run_id, waiting.proposal.proposal_hash, "portfolio-owner")
    print(f"run_id: {completed.run_id}")
    print(f"evidence_ids: {', '.join(item.evidence_id for item in completed.evidence.items)}")
    print(f"proposal_hash: {completed.proposal.proposal_hash}")
    print(f"approval: {completed.approval.reviewer}")
    print(f"final_state: {completed.current_state.value}")
    print(f"model_adapter: {model.__class__.__name__ if model else 'FakeStructuredModel'}")
    print("simulated_actions: " + ", ".join(f"{item.action_type.value}={item.status}" for item in completed.execution_results))


def _restart_submit(runtime_directory: Path) -> None:
    service = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=runtime_directory,
    )
    waiting = service.submit(_golden_ticket())
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "run_id": waiting.run_id,
                "proposal_hash": waiting.proposal.proposal_hash,
            }
        )
    )


def _restart_approve(
    runtime_directory: Path, run_id: str, approved_proposal_hash: str
) -> None:
    service = SupportFlowService.demo(
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
        runtime_directory=runtime_directory,
    )
    resumed = service.resume(run_id)
    result = service.approve(run_id, approved_proposal_hash, "portfolio-owner")
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "resumed_state": resumed.current_state.value,
                "statuses": [item.status for item in result.execution_results],
                "stored_side_effects": service.repository.count_execution_side_effects(),
            }
        )
    )


def demo_restart(runtime_directory: Path) -> None:
    runtime_directory = runtime_directory.resolve()
    if runtime_directory.name not in {"demo-restart", "final-restart"}:
        raise ValueError(
            "demo-restart may only clear a dedicated demo-restart runtime or a dedicated final-restart runtime"
        )
    runtime_directory.mkdir(parents=True, exist_ok=True)
    for database_path in (
        runtime_database_path(runtime_directory),
        checkpoint_database_path(runtime_directory),
    ):
        database_path.unlink(missing_ok=True)
    environment = {**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"}

    def run_child(command: str, *arguments: str) -> dict:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "supportflow.cli",
                command,
                "--runtime",
                str(runtime_directory),
                *arguments,
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(completed.stdout)

    submitted = run_child("_restart-submit")
    approval_arguments = (
        "--run-id",
        submitted["run_id"],
        "--proposal-hash",
        submitted["proposal_hash"],
    )
    first = run_child("_restart-approve", *approval_arguments)
    repeated = run_child("_restart-approve", *approval_arguments)

    print(f"submit_process: {submitted['pid']}")
    print(f"approval_process: {first['pid']}")
    print(f"repeat_process: {repeated['pid']}")
    print(f"resumed_state: {first['resumed_state']}")
    print("first_execution_statuses: " + ", ".join(first["statuses"]))
    print("repeated_execution_statuses: " + ", ".join(repeated["statuses"]))
    print(f"stored_side_effects: {repeated['stored_side_effects']}")


def demo_safety() -> None:
    service = SupportFlowService.demo(as_of=datetime(2026, 8, 31, tzinfo=UTC))

    def ticket(ticket_id: str) -> Ticket:
        return Ticket(
            ticket_id=ticket_id,
            customer_id="customer-safety-001",
            subject="Please help with my payment",
            body="I need help with order order-100 and a charge of USD 29.00.",
            order_id="order-100",
            amount="29.00",
            currency="USD",
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )

    missing = service.submit(ticket("T-MISSING-001"))
    conflict = service.submit(ticket("T-CONFLICT-001"))
    revision = service.submit(
        Ticket(
            ticket_id="ticket-duplicate-001",
            customer_id="customer-001",
            subject="I was charged twice",
            body="My order order-100 was charged twice for USD 29.00.",
            order_id="order-100",
            amount="29.00",
            currency="USD",
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )
    )
    revised = service.modify(
        revision.run_id,
        {"reply_text": "We opened a duplicate-charge review."},
        "portfolio-owner",
    )
    rejected = service.reject(revised.run_id, "Need customer confirmation.", "portfolio-owner")
    print(f"missing_information: {missing.current_state.value}")
    print(f"policy_conflict: {conflict.current_state.value}")
    print(f"superseded_approvals: {sum(item.status == 'superseded' for item in revised.approvals)}")
    print(f"rejection: {rejected.current_state.value}")
    print(f"unsafe_executions: {sum(len(run.execution_results) for run in [missing, conflict, rejected])}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluation_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(prog="supportflow")
    parser.add_argument(
        "command",
        choices=[
            "demo-golden",
            "demo-safety",
            "demo-restart",
            "_restart-submit",
            "_restart-approve",
        ],
    )
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RESTART_DEMO_DIRECTORY)
    parser.add_argument("--run-id")
    parser.add_argument("--proposal-hash")
    parser.add_argument("--model-adapter", choices=["fake", "openai"], default="fake")
    args = parser.parse_args()
    if args.command == "demo-golden":
        demo_golden(args.model_adapter)
    elif args.command == "demo-safety":
        demo_safety()
    elif args.command == "demo-restart":
        demo_restart(args.runtime)
    elif args.command == "_restart-submit":
        _restart_submit(args.runtime)
    elif args.command == "_restart-approve":
        if not args.run_id or not args.proposal_hash:
            parser.error("_restart-approve requires --run-id and --proposal-hash")
        _restart_approve(args.runtime, args.run_id, args.proposal_hash)


if __name__ == "__main__":
    main()
