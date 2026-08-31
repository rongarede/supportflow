# SupportFlow reviewer demo

Run this from the repository root after `uv sync --extra dev`.

## 1. Complete a duplicate-charge journey

```bash
uv run python -m supportflow.cli demo-golden
```

Confirm the printed `final_state: COMPLETED`, proposal hash, approved reviewer, and two `succeeded` simulated actions. The default is `FakeStructuredModel`; it is reproducible and is not a claim about real-model quality.

## 2. Show approval invalidation after a modification

```bash
uv run pytest tests/slices/test_safety_paths.py::test_modification_invalidates_old_approval -v
```

The original proposal hash is superseded after the reply text changes, so it cannot approve the revised proposal.

## 3. Show duplicate execution prevention and restart recovery

```bash
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.cli demo-restart --runtime .supportflow/final-restart
```

The first approval runs in a fresh process and reports two `succeeded` actions. Replaying that approval reports `skipped_duplicate` for both actions and `stored_side_effects: 2`.

The restart tests also inject a stop immediately before each triage, retrieval, resolution, review, and policy journal commit. Reopening safely replays a missing result or reconciles an existing result, without executing an unapproved action.

## 4. Show bounded model failure

```bash
uv run pytest tests/slices/test_model_ready.py::test_two_retryable_model_failures_stop_without_execution -v
```

Timeout and malformed structured-output doubles each make two local attempts, persist only a safe error category, finish at `NEEDS_ATTENTION`, and leave `execution_results` empty. A semantically invalid action is not retried. No external request is made.

## 5. Reproduce the frozen evaluation

```bash
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.eval.runner --dataset data/eval/tickets.jsonl --policies data/policies --output artifacts
```

Read the output `evaluation_input_sha256` and open the matching JSON and Markdown report under `artifacts/`. The JSON records each retrieved evidence ID and the sufficiency result; every successful retrieval is constrained to Top-5. These reports verify fixed fixtures and deterministic controls; they do not establish real-model performance or business improvement.

## 6. Rebuild the evidence manifest from a fresh run

```bash
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.evidence export --runtime .supportflow/evidence-export --evaluation artifacts/eval-87eced57d8de89b24b3d6b1f470220761ffc443f48145bff011fc7735649e014.json --output artifacts/evidence-manifest-v1.json
```

The command creates a dedicated durable run, approves and replays its exact proposal, and exports sanitized runtime records. It also computes an executable-source content digest; a supplied mismatched `--source-revision` is rejected.
