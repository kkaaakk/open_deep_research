"""Retriever implementation for the local RAG pipeline."""

from open_deep_research.rag.types import RetrievalResult
from open_deep_research.rag.vectorstore import VectorStoreBackend


class ChunkRetriever:
    """Thin retrieval wrapper around the configured vector store backend."""

    def __init__(self, vectorstore: VectorStoreBackend):
        self.vectorstore = vectorstore

    def retrieve(self, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        """Retrieve top-k results and deduplicate them by chunk identifier."""
        raw_results = self.vectorstore.search(query_vector, top_k=top_k)
        deduplicated_results: list[RetrievalResult] = []
        seen_chunk_ids: set[str] = set()

        for result in raw_results:
            if result.chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk.chunk_id)
            deduplicated_results.append(result)

        return deduplicated_results
