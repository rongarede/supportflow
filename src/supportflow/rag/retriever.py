from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

from supportflow.domain.models import EvidenceBundle, EvidenceItem
from supportflow.rag.documents import PolicyChunk
from supportflow.rag.embeddings import EmbeddingProvider


def reciprocal_rank_fusion(vector_ids: list[str], bm25_ids: list[str]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in (vector_ids, bm25_ids):
        for rank, evidence_id in enumerate(ranked_ids, start=1):
            scores[evidence_id] += 1 / (60 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:5]


class RagRetriever:
    def __init__(self, chunks: list[PolicyChunk], embedding_provider: EmbeddingProvider) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self._embeddings = embedding_provider.embed([chunk.text for chunk in chunks])

    def retrieve(self, query: str, intent: str, as_of: datetime, top_k: int = 5) -> EvidenceBundle:
        active_pairs = [
            (index, chunk) for index, chunk in enumerate(self.chunks) if chunk.document.active_at(as_of)
        ]
        if not active_pairs:
            raise ValueError("No active policy evidence is available")
        indices, active_chunks = zip(*active_pairs, strict=True)
        query_vector = self.embedding_provider.embed([query])[0]
        vectors = self._embeddings[list(indices)]
        denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        scores = np.divide(vectors @ query_vector, denominator, out=np.zeros_like(denominator), where=denominator != 0)
        vector_order = sorted(
            range(len(active_chunks)), key=lambda index: (-float(scores[index]), active_chunks[index].evidence_id)
        )
        tokenized = [chunk.text.lower().split() for chunk in active_chunks]
        bm25_scores = BM25Okapi(tokenized).get_scores(query.lower().split())
        bm25_order = sorted(
            range(len(active_chunks)), key=lambda index: (-float(bm25_scores[index]), active_chunks[index].evidence_id)
        )
        chunk_by_id = {chunk.evidence_id: chunk for chunk in active_chunks}
        fused = reciprocal_rank_fusion(
            [active_chunks[index].evidence_id for index in vector_order],
            [active_chunks[index].evidence_id for index in bm25_order],
        )[:top_k]
        return EvidenceBundle(
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
            ]
        )
