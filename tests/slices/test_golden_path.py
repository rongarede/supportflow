from supportflow.workflow.service import SupportFlowService


def test_duplicate_charge_reaches_approved_simulated_execution(
    demo_service: SupportFlowService, duplicate_ticket
) -> None:
    waiting = demo_service.submit(duplicate_ticket)

    assert waiting.current_state == "WAITING_APPROVAL"
    assert waiting.evidence.items
    assert all(item.active for item in waiting.evidence.items)
    assert set(waiting.proposal.evidence_refs) <= {
        item.evidence_id for item in waiting.evidence.items
    }
    assert waiting.policy_decision.outcome == "allow"

    completed = demo_service.approve(
        waiting.run_id, waiting.proposal.proposal_hash, reviewer="portfolio-owner"
    )

    assert completed.current_state == "COMPLETED"
    assert [result.action_type for result in completed.execution_results] == [
        "CREATE_REFUND_REQUEST",
        "SEND_REPLY",
    ]
    assert [event.stage for event in completed.trace] == [
        "triage",
        "retrieve",
        "resolve",
        "review",
        "policy",
        "human_approval",
        "execute",
    ]
