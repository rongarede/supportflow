from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FixedEmbeddingProvider:
    def __init__(self, vectors: dict[str, list[float]], name: str = "fixed-demo-v1") -> None:
        self.name = name
        self.vectors = {key: np.asarray(value, dtype=float) for key, value in vectors.items()}
        self.dimension = len(next(iter(self.vectors.values()))) if self.vectors else 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [self.vectors.get(text, np.zeros(self.dimension, dtype=float)) for text in texts], dtype=float
        )


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=float)
