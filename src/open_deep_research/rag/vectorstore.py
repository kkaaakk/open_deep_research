"""Vector store backends for the local RAG pipeline."""

import math
from abc import ABC, abstractmethod

from open_deep_research.rag.types import RAGChunk, RetrievalResult


class VectorStoreBackend(ABC):
    """Abstract vector store backend."""

    @abstractmethod
    def add(self, chunks: list[RAGChunk], vectors: list[list[float]]) -> None:
        """Index chunks and their vectors."""

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        """Search the vector store."""


class InMemoryVectorStore(VectorStoreBackend):
    """A simple in-memory vector store for local development and testing."""

    def __init__(self):
        self._records: list[tuple[RAGChunk, list[float]]] = []

    def add(self, chunks: list[RAGChunk], vectors: list[list[float]]) -> None:
        """Store vectors alongside their source chunks."""
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match.")
        self._records.extend(zip(chunks, vectors))

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        """Return the highest-scoring chunks by cosine similarity."""
        if not self._records or not query_vector:
            return []

        scored_results = [
            RetrievalResult(chunk=chunk, score=_cosine_similarity(query_vector, vector))
            for chunk, vector in self._records
        ]
        scored_results.sort(key=lambda result: result.score, reverse=True)
        return scored_results[:top_k]


def create_vectorstore_backend(provider: str) -> VectorStoreBackend:
    """Create the configured vector store backend."""
    normalized_provider = provider.lower().strip()
    if normalized_provider == "memory":
        return InMemoryVectorStore()
    raise ValueError(
        f"Unsupported RAG vector store provider '{provider}'. "
        "Add a new backend implementation in open_deep_research.rag.vectorstore."
    )


def _cosine_similarity(left_vector: list[float], right_vector: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(left_vector) != len(right_vector):
        raise ValueError("Vector dimensions must match for similarity search.")

    left_norm = math.sqrt(sum(component * component for component in left_vector))
    right_norm = math.sqrt(sum(component * component for component in right_vector))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(left * right for left, right in zip(left_vector, right_vector))
    return dot_product / (left_norm * right_norm)
