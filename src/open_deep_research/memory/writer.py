"""Conversation memory write workflow.

This path owns raw MySQL writes. When configured, it also triggers a background
index refresh after the write; query-time `RAGIndexer.ensure_ready()` still
remains the fallback if that background refresh has not completed.
"""

import logging
import threading
from typing import Any, Mapping, Sequence

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.memory.context import get_conversation_id, get_user_id
from open_deep_research.memory.extractor import extract_conversation_memories
from open_deep_research.memory.store import MySQLChatMemoryStore

LOGGER = logging.getLogger(__name__)


def persist_conversation_memory(
    *,
    configurable: Configuration,
    runtime_config: RunnableConfig,
    chat_content: str,
    summary: str,
    memories: Sequence[str | Mapping[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Persist chat memory to MySQL and mark rows as pending for indexing."""
    if not configurable.rag_memory_write_enabled:
        return 0
    if not configurable.rag_memory_mysql_url:
        raise ValueError("rag_memory_mysql_url is required when memory writing is enabled.")

    conversation_id = get_conversation_id(runtime_config)
    user_id = get_user_id(runtime_config)
    records = extract_conversation_memories(
        conversation_id=conversation_id,
        user_id=user_id,
        chat_content=chat_content,
        summary=summary,
        memories=memories,
        metadata=metadata,
    )
    if not records:
        return 0

    store = MySQLChatMemoryStore(
        database_url=configurable.rag_memory_mysql_url,
        table_name=configurable.rag_memory_mysql_table,
    )
    written_count = store.upsert_records(records)
    if written_count and configurable.rag_memory_write_sync_index:
        trigger_memory_index_refresh(
            configurable=configurable,
            conversation_id=conversation_id,
            user_id=user_id,
        )
    return written_count


def trigger_memory_index_refresh(
    *,
    configurable: Configuration,
    conversation_id: str,
    user_id: str | None = None,
) -> None:
    """Start a non-blocking refresh that syncs pending memory into the vector store."""
    thread = threading.Thread(
        target=_refresh_memory_index,
        kwargs={
            "configurable": configurable,
            "conversation_id": conversation_id,
            "user_id": user_id,
        },
        name="rag-memory-index-refresh",
        daemon=True,
    )
    thread.start()


def _refresh_memory_index(
    *,
    configurable: Configuration,
    conversation_id: str,
    user_id: str | None = None,
) -> None:
    try:
        from open_deep_research.rag.service import (
            RAGPipelineConfig,
            get_or_create_rag_pipeline,
        )

        pipeline = get_or_create_rag_pipeline(
            RAGPipelineConfig(
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
                memory_enabled=True,
                memory_paths=configurable.rag_memory_paths,
                memory_json_text_fields=configurable.rag_memory_json_text_fields,
                memory_mysql_url=configurable.rag_memory_mysql_url,
                memory_mysql_table=configurable.rag_memory_mysql_table,
                memory_mysql_limit=configurable.rag_memory_mysql_limit,
                memory_mysql_index_record_types=(
                    configurable.rag_memory_mysql_index_record_types
                ),
                memory_conversation_id=conversation_id,
                memory_user_id=user_id,
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
                neo4j_uri=configurable.rag_neo4j_uri,
                neo4j_username=configurable.rag_neo4j_username,
                neo4j_password=configurable.rag_neo4j_password,
                neo4j_database=configurable.rag_neo4j_database,
                authority_rerank_enabled=configurable.rag_authority_rerank_enabled,
            )
        )
        pipeline.index_pending_memories()
    except Exception as exc:  # pragma: no cover - depends on external DB/vector DB
        LOGGER.warning("Failed to refresh memory RAG index: %s", exc)
