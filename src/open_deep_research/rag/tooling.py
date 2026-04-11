"""Tool adapters for exposing the local RAG pipeline to the main research flow."""

import asyncio

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from open_deep_research.configuration import Configuration
from open_deep_research.rag.service import RAGPipelineConfig, get_or_create_rag_pipeline

RAG_SEARCH_DESCRIPTION = (
    "Search the configured local knowledge base built from txt, md, json, or pdf files. "
    "Use this when the answer may exist in local documents, notes, manuals, or project files."
)


@tool(description=RAG_SEARCH_DESCRIPTION)
async def rag_search(query: str, config: RunnableConfig = None) -> str:
    """Search the local RAG knowledge base and return grounded context with citations."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_enabled:
        return "Local RAG retrieval is disabled in configuration."
    if not configurable.rag_knowledge_base_paths:
        return "No local RAG knowledge base paths are configured."

    try:
        pipeline_config = RAGPipelineConfig(
            knowledge_base_paths=configurable.rag_knowledge_base_paths,
            chunk_size=configurable.rag_chunk_size,
            chunk_overlap=configurable.rag_chunk_overlap,
            top_k=configurable.rag_top_k,
            rerank_top_n=configurable.rag_rerank_top_n,
            embedding_provider=configurable.rag_embedding_provider,
            vectorstore_provider=configurable.rag_vectorstore_provider,
            reranker_provider=configurable.rag_reranker_provider,
            json_text_fields=configurable.rag_json_text_fields,
            hash_embedding_dimensions=configurable.rag_hash_embedding_dimensions,
        )
        pipeline = get_or_create_rag_pipeline(pipeline_config)
        answer_ready_context = await asyncio.to_thread(pipeline.query, query)
        return answer_ready_context.context
    except Exception as exc:
        return f"Local RAG search failed: {exc}"
