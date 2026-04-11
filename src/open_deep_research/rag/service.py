"""Independent local RAG pipeline orchestration."""

import json
import threading
from typing import Optional

from pydantic import BaseModel

from open_deep_research.rag.citations import build_answer_ready_context
from open_deep_research.rag.embeddings import create_embedding_backend
from open_deep_research.rag.loaders import load_documents_from_paths
from open_deep_research.rag.reranker import KeywordOverlapReranker, create_reranker
from open_deep_research.rag.retriever import ChunkRetriever
from open_deep_research.rag.splitter import split_documents
from open_deep_research.rag.types import AnswerReadyContext, RAGChunk, RAGDocument, RetrievalResult
from open_deep_research.rag.vectorstore import create_vectorstore_backend


class RAGPipelineConfig(BaseModel):
    """Configuration for the standalone local RAG pipeline."""

    knowledge_base_paths: list[str]
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 4
    rerank_top_n: int = 6
    embedding_provider: str = "hash"
    vectorstore_provider: str = "memory"
    reranker_provider: str = "simple"
    json_text_fields: Optional[list[str]] = None
    hash_embedding_dimensions: int = 256


class RAGPipeline:
    """End-to-end local RAG pipeline with lazy ingestion and in-memory indexing."""

    def __init__(self, config: RAGPipelineConfig):
        self.config = config
        self.embedding_backend = create_embedding_backend(
            provider=config.embedding_provider,
            hash_dimensions=config.hash_embedding_dimensions,
        )
        self.vectorstore = create_vectorstore_backend(config.vectorstore_provider)
        self.retriever = ChunkRetriever(self.vectorstore)
        self.reranker = create_reranker(config.reranker_provider)
        self.documents: list[RAGDocument] = []
        self.chunks: list[RAGChunk] = []
        self._indexed = False
        self._index_lock = threading.Lock()

    def query(self, query: str) -> AnswerReadyContext:
        """Retrieve grounded context for a query."""
        self._ensure_indexed()

        if not self.documents or not self.chunks:
            return AnswerReadyContext(
                query=query,
                context=(
                    "No local knowledge documents were loaded. "
                    "Check rag_knowledge_base_paths and supported file types before retrying."
                ),
            )

        query_vector = self.embedding_backend.embed_query(query)
        candidate_count = max(self.config.top_k, self.config.rerank_top_n)
        retrieval_results = self.retriever.retrieve(query_vector, top_k=candidate_count)
        retrieval_results = self._rerank_if_needed(query=query, results=retrieval_results)
        filtered_results = self._filter_results(retrieval_results)

        return build_answer_ready_context(
            query=query,
            matched_chunks=filtered_results[: self.config.top_k],
        )

    def _ensure_indexed(self) -> None:
        """Build the local index once, lazily."""
        if self._indexed:
            return

        with self._index_lock:
            if self._indexed:
                return

            self.documents = load_documents_from_paths(
                self.config.knowledge_base_paths,
                json_text_fields=self.config.json_text_fields,
            )
            self.chunks = split_documents(
                self.documents,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            if self.chunks:
                vectors = self.embedding_backend.embed_texts(
                    [chunk.content for chunk in self.chunks]
                )
                self.vectorstore.add(self.chunks, vectors)

            self._indexed = True

    def _rerank_if_needed(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply reranking to the candidate set when configured."""
        if not results:
            return []
        if isinstance(self.reranker, KeywordOverlapReranker):
            return self.reranker.rerank(query=query, results=results, top_k=len(results))
        return self.reranker.rerank(query=query, results=results, top_k=self.config.top_k)

    def _filter_results(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Filter clearly irrelevant results while preserving strong matches."""
        return [
            result
            for result in results
            if (result.rerank_score or 0.0) > 0.0 or result.score > 0.15
        ]


_PIPELINE_CACHE: dict[str, RAGPipeline] = {}
_PIPELINE_CACHE_LOCK = threading.Lock()


def get_or_create_rag_pipeline(config: RAGPipelineConfig) -> RAGPipeline:
    """Cache local RAG pipelines by effective configuration."""
    cache_key = json.dumps(config.model_dump(), sort_keys=True)
    with _PIPELINE_CACHE_LOCK:
        cached_pipeline = _PIPELINE_CACHE.get(cache_key)
        if cached_pipeline is not None:
            return cached_pipeline

        created_pipeline = RAGPipeline(config)
        _PIPELINE_CACHE[cache_key] = created_pipeline
        return created_pipeline


def reset_rag_pipeline_cache() -> None:
    """Clear the in-memory RAG pipeline cache."""
    with _PIPELINE_CACHE_LOCK:
        _PIPELINE_CACHE.clear()
