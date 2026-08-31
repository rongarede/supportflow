from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import numpy as np

from supportflow.rag.documents import PolicyChunk, PolicyDocument
from supportflow.rag.embeddings import EmbeddingProvider
from supportflow.storage.repositories import IndexManifest, SupportFlowRepository


def build_policy_chunks(documents: list) -> list[PolicyChunk]:
    from supportflow.rag.documents import chunk_policy_document

    return [chunk for document in documents for chunk in chunk_policy_document(document)]


def policy_content_hash(document: PolicyDocument) -> str:
    canonical = json.dumps(
        document.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_persisted_policy_index(
    documents: list[PolicyDocument],
    provider: EmbeddingProvider,
    repository: SupportFlowRepository,
) -> tuple[list[PolicyChunk], np.ndarray]:
    from supportflow.rag.documents import chunk_policy_document

    repository.delete_removed_index_documents(
        {document.document_id for document in documents}
    )
    all_chunks: list[PolicyChunk] = []
    all_vectors: list[np.ndarray] = []
    for document in documents:
        content_hash = policy_content_hash(document)
        existing = repository.get_index_manifest(document.document_id)
        if (
            existing is not None
            and existing.content_hash == content_hash
            and existing.embedding_model == provider.name
        ):
            chunks, vectors = repository.load_index_chunks(existing.chunk_ids)
            if len(chunks) == len(existing.chunk_ids):
                all_chunks.extend(chunks)
                all_vectors.append(vectors)
                continue
        chunks = chunk_policy_document(document)
        vectors = provider.embed([chunk.text for chunk in chunks])
        manifest = IndexManifest(
            document_id=document.document_id,
            content_hash=content_hash,
            chunk_ids=[chunk.evidence_id for chunk in chunks],
            embedding_model=provider.name,
            built_at=datetime.now(UTC).isoformat(),
        )
        repository.replace_index_document(manifest, chunks, vectors)
        all_chunks.extend(chunks)
        all_vectors.append(vectors)
    if not all_vectors:
        return all_chunks, np.empty((0, 0), dtype=float)
    return all_chunks, np.concatenate(all_vectors, axis=0)
