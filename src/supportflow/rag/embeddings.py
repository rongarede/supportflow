from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FixedEmbeddingProvider:
    def __init__(
        self,
        vectors: dict[str, list[float]],
        name: str = "fixed-token-hash-v1",
        dimensions: int = 128,
    ) -> None:
        self.name = name
        self.vectors = {key: np.asarray(value, dtype=float) for key, value in vectors.items()}
        self.dimension = (
            len(next(iter(self.vectors.values()))) if self.vectors else dimensions
        )

    def _fixed_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=float)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                self.vectors.get(text, self._fixed_vector(text))
                for text in texts
            ],
            dtype=float,
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
