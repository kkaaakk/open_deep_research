"""Embedding backends for the local RAG pipeline."""

import hashlib
import math
import re
from abc import ABC, abstractmethod

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class HashEmbeddingBackend(EmbeddingBackend):
    """A deterministic, zero-dependency embedding backend for local development and tests."""

    def __init__(self, dimensions: int = 256):
        if dimensions <= 0:
            raise ValueError("Hash embedding dimensions must be greater than 0.")
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        """Hash tokens into a normalized dense vector."""
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + math.log1p(len(token))
            vector[position] += sign * weight

        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0:
            return vector
        return [component / magnitude for component in vector]


def create_embedding_backend(provider: str, hash_dimensions: int = 256) -> EmbeddingBackend:
    """Create the configured embedding backend."""
    normalized_provider = provider.lower().strip()
    if normalized_provider == "hash":
        return HashEmbeddingBackend(dimensions=hash_dimensions)
    raise ValueError(
        f"Unsupported RAG embedding provider '{provider}'. "
        "Add a new backend implementation in open_deep_research.rag.embeddings."
    )
