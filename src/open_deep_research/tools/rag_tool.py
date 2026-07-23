"""Expose RAG retrieval as an agent tool."""

import asyncio

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from open_deep_research.configuration import Configuration
from open_deep_research.memory.context import get_conversation_id, get_user_id
from open_deep_research.rag.query_rewriter import (
    api_key_for_model,
    maybe_rewrite_query_with_model,
)
from open_deep_research.rag.service import RAGPipelineConfig, get_or_create_rag_pipeline

RAG_SEARCH_DESCRIPTION = (
    "Search the configured local RAG sources, including local txt/md/json/pdf/image files "
    "and optional chat memory from json/jsonl or MySQL records. Use this when the answer may exist "
    "in local documents, notes, manuals, project files, or remembered user context."
)


@tool(description=RAG_SEARCH_DESCRIPTION)
async def rag_search(query: str, config: RunnableConfig = None) -> str:
    """Search local knowledge and memory, returning cited context only."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_enabled:
        return "Local RAG retrieval is disabled in configuration."

    has_document_paths = bool(configurable.rag_knowledge_base_paths)
    has_memory_paths = bool(configurable.rag_memory_enabled and configurable.rag_memory_paths)
    has_mysql_memory = bool(configurable.rag_memory_enabled and configurable.rag_memory_mysql_url)
    if not has_document_paths and not has_memory_paths and not has_mysql_memory:
        return "No local RAG document paths or memory paths are configured."

    try:
        pipeline_config = RAGPipelineConfig(
            knowledge_base_paths=configurable.rag_knowledge_base_paths or [],
            chunk_size=configurable.rag_chunk_size,
            chunk_overlap=configurable.rag_chunk_overlap,
            top_k=configurable.rag_top_k,
            rerank_top_n=configurable.rag_rerank_top_n,
            embedding_provider=configurable.rag_embedding_provider,
            embedding_model=configurable.rag_embedding_model,
            embedding_device=configurable.rag_embedding_device,
            vectorstore_provider=configurable.rag_vectorstore_provider,
            vectorstore_path=configurable.rag_vectorstore_path,
            collection_name=configurable.rag_collection_name,
            milvus_uri=configurable.rag_milvus_uri,
            milvus_token=configurable.rag_milvus_token,
            milvus_db_name=configurable.rag_milvus_db_name,
            milvus_metric_type=configurable.rag_milvus_metric_type,
            reranker_provider=configurable.rag_reranker_provider,
            reranker_model=configurable.rag_reranker_model,
            reranker_device=configurable.rag_reranker_device,
            json_text_fields=configurable.rag_json_text_fields,
            multimodal_enabled=configurable.rag_multimodal_enabled,
            multimodal_provider=configurable.rag_multimodal_provider,
            ocr_languages=configurable.rag_ocr_languages,
            vision_enabled=configurable.rag_vision_enabled,
            vision_model=configurable.rag_vision_model,
            vision_prompt=configurable.rag_vision_prompt,
            vision_max_tokens=configurable.rag_vision_max_tokens,
            memory_enabled=configurable.rag_memory_enabled,
            memory_paths=configurable.rag_memory_paths,
            memory_json_text_fields=configurable.rag_memory_json_text_fields,
            memory_mysql_url=configurable.rag_memory_mysql_url,
            memory_mysql_table=configurable.rag_memory_mysql_table,
            memory_mysql_limit=configurable.rag_memory_mysql_limit,
            memory_mysql_index_record_types=configurable.rag_memory_mysql_index_record_types,
            memory_conversation_id=get_conversation_id(config),
            memory_user_id=get_user_id(config),
            hash_embedding_dimensions=configurable.rag_hash_embedding_dimensions,
            keyword_top_k=configurable.rag_keyword_top_k,
            keyword_backend=configurable.rag_keyword_backend,
            elasticsearch_url=configurable.rag_elasticsearch_url,
            elasticsearch_index=configurable.rag_elasticsearch_index,
            hybrid_alpha=configurable.rag_hybrid_alpha,
            rrf_rank_constant=configurable.rag_rrf_rank_constant,
            structured_metadata_weight=configurable.rag_structured_metadata_weight,
            graph_enabled=configurable.rag_graph_enabled,
            graph_backend=configurable.rag_graph_backend,
            graph_max_neighbors=configurable.rag_graph_max_neighbors,
            graph_weight=configurable.rag_graph_weight,
            graph_ner_enabled=configurable.rag_graph_ner_enabled,
            graph_idf_enabled=configurable.rag_graph_idf_enabled,
            graph_idf_threshold_percentile=configurable.rag_graph_idf_threshold_percentile,
            graph_confidence_threshold=configurable.rag_graph_confidence_threshold,
            structural_edges_enabled=configurable.rag_structural_edges_enabled,
            neo4j_uri=configurable.rag_neo4j_uri,
            neo4j_username=configurable.rag_neo4j_username,
            neo4j_password=configurable.rag_neo4j_password,
            neo4j_database=configurable.rag_neo4j_database,
            authority_rerank_enabled=configurable.rag_authority_rerank_enabled,
        )
        pipeline = get_or_create_rag_pipeline(pipeline_config)
        retrieval_query = await asyncio.to_thread(
            maybe_rewrite_query_with_model,
            query,
            enabled=configurable.rag_query_rewrite_enabled,
            model_name=configurable.rag_query_rewrite_model,
            max_tokens=configurable.rag_query_rewrite_max_tokens,
            api_key=api_key_for_model(configurable.rag_query_rewrite_model, config or {}),
        )
        answer_ready_context = await asyncio.to_thread(
            pipeline.query,
            retrieval_query,
            original_query=query,
        )
        return answer_ready_context.context
    except Exception as exc:
        return f"Local RAG search failed: {exc}"
