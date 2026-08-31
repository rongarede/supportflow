# Evidence boundary

This document records executable prototype evidence, with its limits.

## Committed evidence manifest

The versioned, sanitized record is [artifacts/evidence-manifest-v1.json](../artifacts/evidence-manifest-v1.json). It preserves the run identifier and canonical sanitized payloads for the ticket, all three agent outputs, evidence, policy decision, approval, simulated executions, checkpoint, trace, and frozen evaluation report without committing the mutable runtime database.

Every record hash is SHA-256 of its UTF-8 canonical JSON payload: recursively sorted keys with compact `,` and `:` separators, emitted without ASCII escaping. The evaluation-report hash is computed from parsed JSON using the same algorithm. `supportflow.evidence generate_manifest_from_deterministic_run` creates a fresh dedicated runtime, drives submit, approval, execution, and replay through the public service, then exports the sanitized runtime records. The test rebuilds from that run source; it never treats the committed manifest payload as its oracle.

The manifest is bound to `supportflow-source-sha256:4fe31d979c29d9e929de209943a852dd4275bc47d12ed68cc7f601af58b2de7b`. This is an exact content digest over executable Python source, frozen data, dependency metadata, and the lockfile. Generated artifacts are excluded from the digest. This is the deliberate ruling that avoids requiring a manifest to contain the hash of a Git commit that itself contains the manifest; `supportflow-evidence-manifest/v1` separately fixes the manifest schema identity.

## Durable duplicate-charge run

The deterministic evidence export ran with run ID `81543838-7614-5f65-a331-23aac8e88baf` and input revision `evidence-v1` in the dedicated `.supportflow/evidence-export/` runtime.

- **Ticket:** `ticket-duplicate-001`, persisted in `supportflow.db:runs` and `tickets`.
- **Three agent outputs:** `node_outputs` contains one result each for `triage`, `resolve`, and `review`; their adapter attempts are separately persisted as one each in `model_attempts`.
- **EvidenceBundle and PolicyDecision:** the same run has `retrieve` and `policy` outputs in `node_outputs`, with the policy outcome `allow`.
- **Approval binding:** `approvals` records reviewer `portfolio-owner` for proposal hash `5d79a037c4788a53798e48f25d86bae6795e7ff2c97213d7ecd2fd5872b55982`.
- **ExecutionResult:** `executions` records exactly two simulated side effects, `CREATE_REFUND_REQUEST` and `SEND_REPLY`. Re-approval produced `skipped_duplicate` results without adding side effects.
- **Checkpoint and trace:** `.supportflow/evidence-export/checkpoints.sqlite` contains 10 checkpoint rows. `trace_events` records triage, retrieval, resolution, review, policy, human approval, simulated execution, and the duplicate-execution skip.

The exported runtime records show two first-run successes, two replay skips, and `stored_side_effect_count: 2`. The durable runtime is a local demonstration artifact; its executor is simulated. It is an export source, not the portable evidence itself; use the committed manifest above when reviewing the run.

## Frozen evaluation evidence

The current combined evaluation report is [artifacts/eval-87eced57d8de89b24b3d6b1f470220761ffc443f48145bff011fc7735649e014.md](../artifacts/eval-87eced57d8de89b24b3d6b1f470220761ffc443f48145bff011fc7735649e014.md), with machine-readable companion [JSON](../artifacts/eval-87eced57d8de89b24b3d6b1f470220761ffc443f48145bff011fc7735649e014.json).

The verified report covers 30 independent frozen cases across all four intents and five actions. Exact action scoring checks multiplicity and parameters. It reports `unapproved_execution_count: 0`, `duplicate_side_effect_count: 0`, and `recovery_failure_count: 0`; its combined evaluation-input SHA-256 is `87eced57d8de89b24b3d6b1f470220761ffc443f48145bff011fc7735649e014`.

## What this does not prove

- No real-model demonstration was run: credentials were not supplied.
- Fixed fake results are not real-LLM evaluation.
- No external payment action, customer communication, or production policy decision occurred.
- No business baseline or improvement metric was measured.
