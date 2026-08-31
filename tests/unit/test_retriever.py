from datetime import UTC, datetime

from supportflow.rag.documents import PolicyDocument, chunk_policy_document
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
