# SupportFlow

SupportFlow is a bounded portfolio prototype for English billing and refund-support tickets. It supports four structured intents (`BILLING_QUESTION`, `REFUND_REQUEST`, `DUPLICATE_CHARGE`, and `REFUND_STATUS`) and five simulated actions. Three constrained agent outputs (triage, resolution, and risk review) carry extracted facts, risks, reasons, evidence, and uncertainty; retrieval, policy validation, approval binding, and execution remain deterministic.

## Run the fixed-model journey

```bash
uv sync --extra dev
LANGGRAPH_STRICT_MSGPACK=true uv run pytest -q
uv run python -m supportflow.cli demo-golden
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.cli demo-restart --runtime .supportflow/final-restart
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.eval.runner --dataset data/eval/tickets.jsonl --policies data/policies --output artifacts
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.evidence export --runtime .supportflow/evidence-export --evaluation artifacts/eval-87eced57d8de89b24b3d6b1f470220761ffc443f48145bff011fc7735649e014.json --output artifacts/evidence-manifest-v1.json
uv run streamlit run src/supportflow/ui/app.py
```

`demo-golden` completes the duplicate-charge path with a fixed local model and deterministic token-hash embedding. This embedding is the reproducible default for tests, CLI demos, evaluation, and the workbench. Retrieval uses the Triage intent to prioritize the relevant curated policy sections, fuses exact cosine and BM25 rankings, and returns at most five active items. `EvidenceBundle.sufficient` is true only when the selected evidence covers the intent's required policy type; otherwise generation stops before a proposal. `SentenceTransformerEmbeddingProvider` remains an opt-in application-service choice and loads `sentence-transformers/all-MiniLM-L6-v2` from an already populated local cache (`local_files_only=True`); it never downloads a model implicitly. `demo-restart` is intentionally allowed to clear only a directory named `demo-restart` or `final-restart`; it demonstrates a durable pause, a cross-process approval, and duplicate-execution prevention.

When callers do not supply an upstream `input_revision`, the service and repository derive one from normalized ticket content while excluding the UI receipt timestamp. Independently constructed identical workbench submissions therefore reopen the same waiting or completed run. A supplied source revision is preserved and remains authoritative.

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

The adapter converts the three agent result types into provider-compatible strict JSON Schemas: all DTO fields are required, objects are closed, nested definitions are inlined, and the emitted subset contains no references, unions, or discriminators. Resolution uses one flat closed action DTO, then maps only the five allowlisted action forms into the domain variants after parsing. A timeout or malformed structured output is retried once; SDK retries are disabled so this is the complete two-call budget, persisted atomically across reopen/re-entry. Retrieval also retries once. A process stop after a safe model or retrieval result but before its repository journal commit replays that pending node within the same durable budget; an already journaled result is reconciled without re-running it. Each simulated action has a durable three-attempt transient-failure budget. All retry traces and terminal errors contain only safe categories, not provider payloads. An invalid action is not retried. Any exhausted or invalid result stops at `NEEDS_ATTENTION`; it cannot bypass approval. Provider keys and authorization headers are never logged. Automated tests use local doubles only and do not make network requests. The real-model demo is **not run** in this repository without user-supplied credentials.

## Evidence and limits

The linked walkthrough is in [docs/demo-script.md](docs/demo-script.md), the concrete evidence chain is in [docs/evidence-boundary.md](docs/evidence-boundary.md), and its committed sanitized record is [artifacts/evidence-manifest-v1.json](artifacts/evidence-manifest-v1.json). The manifest is rebuilt from a fresh deterministic public-service run. Its `source_revision` is a SHA-256 over the executable `src/supportflow` tree, frozen `data` inputs, `pyproject.toml`, and `uv.lock`; generated artifacts are deliberately excluded, so the binding is exact without a self-referential commit-hash paradox.

This is a simulated executor, not a connected payment or support system. The frozen evaluation uses fixed fake model outputs and is **not real-model quality evaluation**. There is no business baseline and no measured business-improvement metric; its reported scores only audit the supplied 30-ticket portfolio dataset and deterministic safety/recovery invariants.
