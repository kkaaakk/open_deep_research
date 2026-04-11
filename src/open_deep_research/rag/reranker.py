"""Reranker implementations for the local RAG pipeline."""

import re
from abc import ABC, abstractmethod

from open_deep_research.rag.types import RetrievalResult

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Rerank retrieved results."""


class NoOpReranker(Reranker):
    """A no-op reranker that preserves vector search order."""

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        return results[:top_k]


class KeywordOverlapReranker(Reranker):
    """A lightweight reranker based on normalized keyword overlap."""

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        query_terms = set(TOKEN_PATTERN.findall(query.lower()))
        reranked_results = []

        for result in results:
            content_terms = set(
                TOKEN_PATTERN.findall(f"{result.chunk.title or ''} {result.chunk.content}".lower())
            )
            overlap_score = 0.0
            if query_terms and content_terms:
                overlap_score = len(query_terms & content_terms) / len(query_terms)
            reranked_results.append(
                result.model_copy(update={"rerank_score": overlap_score})
            )

        reranked_results.sort(
            key=lambda item: ((item.rerank_score or 0.0), item.score),
            reverse=True,
        )
        return reranked_results[:top_k]


def create_reranker(provider: str) -> Reranker:
    """Create the configured reranker backend."""
    normalized_provider = provider.lower().strip()
    if normalized_provider in {"none", "disabled"}:
        return NoOpReranker()
    if normalized_provider == "simple":
        return KeywordOverlapReranker()
    raise ValueError(
        f"Unsupported RAG reranker provider '{provider}'. "
        "Add a new backend implementation in open_deep_research.rag.reranker."
    )
