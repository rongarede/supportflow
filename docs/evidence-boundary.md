# Evidence boundary

This document records executable prototype evidence, with its limits.

## Committed evidence manifest

The versioned, sanitized record is [artifacts/evidence-manifest-v1.json](../artifacts/evidence-manifest-v1.json). It preserves the run identifier and canonical sanitized payloads for the ticket, all three agent outputs, evidence, policy decision, approval, simulated executions, checkpoint, trace, and frozen evaluation report without committing the mutable runtime database.

Every record hash is SHA-256 of its UTF-8 canonical JSON payload: recursively sorted keys with compact `,` and `:` separators, emitted without ASCII escaping. The evaluation-report hash is computed from the parsed JSON using the same algorithm. `supportflow.evidence.build_manifest` and `export_manifest` are the deterministic exporter primitives; `tests/slices/test_evidence_manifest.py` rebuilds the committed manifest, recomputes every record hash, and computes the current committed evaluation-artifact hash.

## Durable duplicate-charge run

The restart demo ran on 2026-08-31 with run ID `d57c0628-a0c6-4788-bdc0-b622b07b2801` in `.supportflow/final-restart/`.

- **Ticket:** `ticket-duplicate-001`, persisted in `supportflow.db:runs` and `tickets`.
- **Three agent outputs:** `node_outputs` contains one result each for `triage`, `resolve`, and `review`; their adapter attempts are separately persisted as one each in `model_attempts`.
- **EvidenceBundle and PolicyDecision:** the same run has `retrieve` and `policy` outputs in `node_outputs`, with the policy outcome `allow`.
- **Approval binding:** `approvals` records reviewer `portfolio-owner` for proposal hash `8aaa5b6bbb05f8cac4479a103e4b9cddc45636a1a02afc94c9ee88c645f49028`.
- **ExecutionResult:** `executions` records exactly two simulated side effects, `CREATE_REFUND_REQUEST` and `SEND_REPLY`. Re-approval produced `skipped_duplicate` results without adding side effects.
- **Checkpoint and trace:** `.supportflow/final-restart/checkpoints.sqlite` contains 10 checkpoint rows. `trace_events` records triage, retrieval, resolution, review, policy, human approval, simulated execution, and the duplicate-execution skip.

The same command output showed `resumed_state: WAITING_APPROVAL`, then two first-run successes, two replay skips, and `stored_side_effects: 2`. The durable runtime is a local demonstration artifact; its executor is simulated. It is an export source, not the portable evidence itself; use the committed manifest above when reviewing the run.

## Frozen evaluation evidence

The current combined evaluation report is [artifacts/eval-870d4b3f9d2e2df3d42a22e9f92a22548f2a19c2d5dc545be6689451343c3f9d.md](../artifacts/eval-870d4b3f9d2e2df3d42a22e9f92a22548f2a19c2d5dc545be6689451343c3f9d.md), with machine-readable companion [JSON](../artifacts/eval-870d4b3f9d2e2df3d42a22e9f92a22548f2a19c2d5dc545be6689451343c3f9d.json).

The verified report covers 30 frozen cases with `unapproved_execution_count: 0`, `duplicate_side_effect_count: 0`, and `recovery_failure_count: 0`. Its identity is bound to the combined evaluation input SHA-256 `870d4b3f9d2e2df3d42a22e9f92a22548f2a19c2d5dc545be6689451343c3f9d`.

## What this does not prove

- No real-model demonstration was run: credentials were not supplied.
- Fixed fake results are not real-LLM evaluation.
- No external payment action, customer communication, or production policy decision occurred.
- No business baseline or improvement metric was measured.
