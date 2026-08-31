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
