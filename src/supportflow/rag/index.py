from supportflow.rag.documents import PolicyChunk


def build_policy_chunks(documents: list) -> list[PolicyChunk]:
    from supportflow.rag.documents import chunk_policy_document

    return [chunk for document in documents for chunk in chunk_policy_document(document)]
