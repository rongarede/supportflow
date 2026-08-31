# SupportFlow

SupportFlow is a bounded portfolio prototype for English billing and refund-support tickets. It turns a ticket into three constrained agent outputs (triage, resolution, and risk review), then keeps retrieval, policy validation, approval binding, and execution deterministic.

## Run the fixed-model journey

```bash
uv sync --extra dev
LANGGRAPH_STRICT_MSGPACK=true uv run pytest -q
uv run python -m supportflow.cli demo-golden
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.cli demo-restart --runtime .supportflow/final-restart
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.eval.runner --dataset data/eval/tickets.jsonl --policies data/policies --output artifacts
uv run streamlit run src/supportflow/ui/app.py
```

`demo-golden` completes the duplicate-charge path with fixed local model fixtures. `demo-restart` is intentionally allowed to clear only a directory named `demo-restart` or `final-restart`; it demonstrates a durable pause, a cross-process approval, and duplicate-execution prevention.

## Optional real-model adapter

Only the `StructuredModel` behind the three agents is replaceable. Retrieval stays `RagRetriever`; the `PolicyGate`, approval hash check, and `DurableExecutor` stay deterministic and simulated.

```bash
uv sync --extra real-llm
export SUPPORTFLOW_LLM_MODEL="your-compatible-model"
export OPENAI_API_KEY="your-key"
# Optional for an OpenAI-compatible provider:
export OPENAI_BASE_URL="https://provider.example/v1"
uv run --extra real-llm python -m supportflow.cli demo-golden --model-adapter openai
```

The adapter converts the three agent result types into provider-compatible strict JSON Schemas: all DTO fields are required, objects are closed, and action variants are limited to the two allowlisted simulated actions. It validates the provider result back into the existing Pydantic domain type. A timeout or malformed structured output is retried once; SDK retries are disabled so this is the complete two-call budget, persisted atomically across reopen/re-entry. An invalid action is not retried. Any exhausted or invalid result records a sanitized error and stops at `NEEDS_ATTENTION`; it does not enter retrieval, approval, or execution. Provider keys and authorization headers are never logged. Automated tests use local doubles only and do not make network requests. The real-model demo is **not run** in this repository without user-supplied credentials.

## Evidence and limits

The linked walkthrough is in [docs/demo-script.md](docs/demo-script.md), the concrete evidence chain is in [docs/evidence-boundary.md](docs/evidence-boundary.md), and its committed sanitized record is [artifacts/evidence-manifest-v1.json](artifacts/evidence-manifest-v1.json).

This is a simulated executor, not a connected payment or support system. The frozen evaluation uses fixed fake model outputs and is **not real-model quality evaluation**. There is no business baseline and no measured business-improvement metric; its reported scores only audit the supplied 30-ticket portfolio dataset and deterministic safety/recovery invariants.
