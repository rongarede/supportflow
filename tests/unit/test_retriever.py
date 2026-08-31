from datetime import UTC, datetime
from pathlib import Path

from supportflow.rag.documents import PolicyDocument, chunk_policy_document
from supportflow.rag.documents import load_policy_documents
from supportflow.rag.embeddings import FixedEmbeddingProvider
from supportflow.rag.retriever import RagRetriever


def test_retriever_filters_inactive_policy_before_rank_and_cites_active_evidence() -> None:
    active = PolicyDocument(
        document_id="policy-active",
        version="1.0",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        effective_to=None,
        policy_type="billing",
        title="Duplicate charge policy",
        body="# duplicate-charge-verification\nVerify a duplicate charge before refunding.",
    )
    expired = PolicyDocument(
        document_id="policy-expired",
        version="1.0",
        effective_from=datetime(2025, 1, 1, tzinfo=UTC),
        effective_to=datetime(2025, 12, 31, tzinfo=UTC),
        policy_type="billing",
        title="Expired duplicate charge policy",
        body="# duplicate-charge-verification\nRefund immediately without verification.",
    )
    chunks = chunk_policy_document(active) + chunk_policy_document(expired)
    retriever = RagRetriever(
        chunks,
        FixedEmbeddingProvider(
            {
                "duplicate charge": [1.0, 0.0],
                **{chunk.text: [1.0, 0.0] for chunk in chunks},
            }
        ),
    )

    evidence = retriever.retrieve(
        "duplicate charge", intent="DUPLICATE_CHARGE", as_of=datetime(2026, 8, 31, tzinfo=UTC)
    )

    assert [item.document_id for item in evidence.items] == ["policy-active"]
    assert evidence.items[0].active is True


def test_expired_policy_fixture_is_auditable_but_never_returned_as_active_evidence() -> None:
    policy_directory = Path(__file__).parents[2] / "data" / "policies"
    documents = load_policy_documents(policy_directory)
    chunks = [chunk for document in documents for chunk in chunk_policy_document(document)]
    expired = next(chunk for chunk in chunks if chunk.document.document_id == "policy-duplicate-charge-legacy")
    retriever = RagRetriever(
        chunks,
        FixedEmbeddingProvider(
            {
                "duplicate charge": [1.0, 0.0],
                **{chunk.text: [1.0, 0.0] for chunk in chunks},
            }
        ),
    )

    evidence = retriever.retrieve(
        "duplicate charge", intent="DUPLICATE_CHARGE", as_of=datetime(2026, 8, 31, tzinfo=UTC)
    )

    assert expired.document.active_at(datetime(2026, 8, 31, tzinfo=UTC)) is False
    assert expired.evidence_id not in [item.evidence_id for item in evidence.items]


def test_retrieval_uses_intent_and_returns_a_sufficient_top_five_bundle() -> None:
    """Top-5 must cover each approved intent without returning the whole policy pack."""
    policy_directory = Path(__file__).parents[2] / "data" / "policies"
    documents = load_policy_documents(policy_directory)
    chunks = [chunk for document in documents for chunk in chunk_policy_document(document)]
    retriever = RagRetriever(chunks, FixedEmbeddingProvider({}))
    current = datetime(2026, 8, 31, tzinfo=UTC)
    cases = {
        "BILLING_QUESTION": (
            "How do you handle a repeated card charge?",
            {"duplicate-charge-verification"},
        ),
        "DUPLICATE_CHARGE": (
            "Order A-10 was charged twice for USD 29.00.",
            {
                "duplicate-charge-verification",
                "duplicate-charge-refund-request",
                "refund-timing",
            },
        ),
        "REFUND_REQUEST": (
            "Please open a refund request; tell me when it will finish.",
            {"refund-eligibility", "refund-timing-details"},
        ),
        "REFUND_STATUS": (
            "What is the status of my existing refund?",
            {"required-customer-verification", "refund-timing-details"},
        ),
    }

    bundles = {
        intent: retriever.retrieve(query, intent=intent, as_of=current)
        for intent, (query, _) in cases.items()
    }

    for intent, (query, required_headings) in cases.items():
        bundle = bundles[intent]
        assert bundle.query == query
        assert bundle.sufficient is True
        assert bundle.unresolved_questions == []
        assert 1 <= len(bundle.items) <= 5
        assert required_headings <= {item.heading for item in bundle.items}
    assert [item.evidence_id for item in bundles["REFUND_REQUEST"].items] != [
        item.evidence_id for item in bundles["REFUND_STATUS"].items
    ]


def test_retrieval_marks_bundle_insufficient_when_policy_type_misses_intent() -> None:
    support_only = PolicyDocument(
        document_id="policy-support-only",
        version="1.0",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        effective_to=None,
        policy_type="support",
        title="General support",
        body="# general-support\nEscalate unusual support questions to a human.",
    )
    chunks = chunk_policy_document(support_only)
    retriever = RagRetriever(chunks, FixedEmbeddingProvider({}))

    evidence = retriever.retrieve(
        "I have a billing question",
        intent="BILLING_QUESTION",
        as_of=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert evidence.items
    assert evidence.sufficient is False
    assert evidence.unresolved_questions == [
        "No active billing policy evidence covers intent BILLING_QUESTION."
    ]
