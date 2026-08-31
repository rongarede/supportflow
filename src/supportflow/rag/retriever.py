from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import re

import numpy as np
from rank_bm25 import BM25Okapi

from supportflow.domain.models import EvidenceBundle, EvidenceItem
from supportflow.rag.documents import PolicyChunk
from supportflow.rag.embeddings import EmbeddingProvider


_INTENT_QUERY_HINTS = {
    "BILLING_QUESTION": "billing duplicate charge verification repeated charge",
    "REFUND_REQUEST": (
        "refund request eligibility exclusion conflict human escalation final timing payment provider"
    ),
    "DUPLICATE_CHARGE": (
        "duplicate charge verification refund request order amount currency final timing payment provider"
    ),
    "REFUND_STATUS": (
        "refund status request identifier required customer verification final timing payment provider"
    ),
}
_INTENT_POLICY_TYPES = {
    intent: frozenset({"billing"}) for intent in _INTENT_QUERY_HINTS
}
_INTENT_PRIORITY_HEADINGS = {
    "BILLING_QUESTION": frozenset({"duplicate-charge-verification"}),
    "REFUND_REQUEST": frozenset(
        {
            "refund-eligibility",
            "refund-exclusion",
            "refund-timing-details",
            "human-escalation-conditions",
        }
    ),
    "DUPLICATE_CHARGE": frozenset(
        {
            "duplicate-charge-verification",
            "duplicate-charge-refund-request",
            "refund-timing",
        }
    ),
    "REFUND_STATUS": frozenset(
        {"required-customer-verification", "refund-timing-details"}
    ),
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(vector_ids: list[str], bm25_ids: list[str], limit: int = 5) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in (vector_ids, bm25_ids):
        for rank, evidence_id in enumerate(ranked_ids, start=1):
            scores[evidence_id] += 1 / (60 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:limit]


class RagRetriever:
    def __init__(
        self,
        chunks: list[PolicyChunk],
        embedding_provider: EmbeddingProvider,
        embeddings: np.ndarray | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self._embeddings = (
            embeddings
            if embeddings is not None
            else embedding_provider.embed([chunk.text for chunk in chunks])
        )

    def retrieve(self, query: str, intent: str, as_of: datetime, top_k: int = 5) -> EvidenceBundle:
        try:
            intent_hint = _INTENT_QUERY_HINTS[intent]
            required_policy_types = _INTENT_POLICY_TYPES[intent]
            priority_headings = _INTENT_PRIORITY_HEADINGS[intent]
        except KeyError as error:
            raise ValueError(f"Unsupported retrieval intent: {intent}") from error
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        active_pairs = [
            (index, chunk) for index, chunk in enumerate(self.chunks) if chunk.document.active_at(as_of)
        ]
        if not active_pairs:
            raise ValueError("No active policy evidence is available")
        indices, active_chunks = zip(*active_pairs, strict=True)
        ranking_query = f"{query} {intent_hint}"
        query_vector = self.embedding_provider.embed([ranking_query])[0]
        vectors = self._embeddings[list(indices)]
        denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        scores = np.divide(vectors @ query_vector, denominator, out=np.zeros_like(denominator), where=denominator != 0)
        vector_order = sorted(
            range(len(active_chunks)), key=lambda index: (-float(scores[index]), active_chunks[index].evidence_id)
        )
        tokenized = [
            _tokens(
                f"{chunk.heading} {chunk.document.title} "
                f"{chunk.document.policy_type} {chunk.text}"
            )
            for chunk in active_chunks
        ]
        bm25_scores = BM25Okapi(tokenized).get_scores(_tokens(ranking_query))
        bm25_order = sorted(
            range(len(active_chunks)), key=lambda index: (-float(bm25_scores[index]), active_chunks[index].evidence_id)
        )
        chunk_by_id = {chunk.evidence_id: chunk for chunk in active_chunks}
        fused_ranking = reciprocal_rank_fusion(
            [active_chunks[index].evidence_id for index in vector_order],
            [active_chunks[index].evidence_id for index in bm25_order],
            limit=len(active_chunks),
        )
        # The curated pack has a deterministic intent-to-policy routing layer.
        # Preserve RRF order inside each tier, then reserve the small Top-5 budget
        # for the policy sections that can govern the approved intent.
        fused = [
            evidence_id
            for evidence_id in fused_ranking
            if chunk_by_id[evidence_id].heading in priority_headings
        ]
        fused.extend(
            evidence_id
            for evidence_id in fused_ranking
            if chunk_by_id[evidence_id].heading not in priority_headings
        )
        fused = fused[:top_k]
        selected_chunks = [chunk_by_id[evidence_id] for evidence_id in fused]
        covered_policy_types = {
            chunk.document.policy_type for chunk in selected_chunks
        }
        sufficient = bool(selected_chunks) and bool(
            required_policy_types & covered_policy_types
        )
        unresolved_questions = [] if sufficient else [
            "No active "
            f"{' or '.join(sorted(required_policy_types))} policy evidence covers intent {intent}."
        ]
        return EvidenceBundle(
            query=query,
            items=[
                EvidenceItem(
                    evidence_id=chunk_by_id[evidence_id].evidence_id,
                    document_id=chunk_by_id[evidence_id].document.document_id,
                    version=chunk_by_id[evidence_id].document.version,
                    heading=chunk_by_id[evidence_id].heading,
                    content=chunk_by_id[evidence_id].text,
                    active=True,
                    score=float(scores[list(active_chunks).index(chunk_by_id[evidence_id])]),
                )
                for evidence_id in fused
            ],
            sufficient=sufficient,
            unresolved_questions=unresolved_questions,
            audit_items=[
                EvidenceItem(
                    evidence_id=chunk.evidence_id,
                    document_id=chunk.document.document_id,
                    version=chunk.document.version,
                    heading=chunk.heading,
                    content=chunk.text,
                    active=False,
                    score=0.0,
                )
                for chunk in self.chunks
                if not chunk.document.active_at(as_of)
            ],
        )
