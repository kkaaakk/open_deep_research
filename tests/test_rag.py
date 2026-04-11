import asyncio
import textwrap

import pytest

from open_deep_research.rag.service import RAGPipeline, RAGPipelineConfig, reset_rag_pipeline_cache
from open_deep_research.rag.splitter import split_documents
from open_deep_research.rag.types import RAGDocument
from open_deep_research.utils import get_all_tools


@pytest.fixture(autouse=True)
def clear_rag_cache():
    reset_rag_pipeline_cache()
    yield
    reset_rag_pipeline_cache()


def write_text_file(path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")


def test_split_documents_creates_chunk_ids_and_preserves_metadata():
    document = RAGDocument(
        content=(
            "Atlas Launch requires quarterly risk reviews.\n\n"
            "The release process includes two approvals, a rollback plan, and a release note. "
            * 8
        ),
        source="memory://atlas",
        title="Atlas handbook",
        metadata={"team": "atlas"},
    )

    chunks = split_documents([document], chunk_size=120, chunk_overlap=20)

    assert len(chunks) >= 2
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.source == "memory://atlas" for chunk in chunks)
    assert all(chunk.metadata["team"] == "atlas" for chunk in chunks)
    assert all(len(chunk.content) <= 120 for chunk in chunks)


def test_rag_pipeline_retrieves_relevant_local_context(tmp_path):
    write_text_file(
        tmp_path / "handbook.md",
        """
        # Atlas Launch Handbook

        Atlas Launch requires quarterly risk reviews and two approvals before production releases.
        """,
    )
    write_text_file(
        tmp_path / "notes.txt",
        """
        The cafeteria serves noodles on Tuesdays.
        """,
    )

    pipeline = RAGPipeline(
        RAGPipelineConfig(
            knowledge_base_paths=[str(tmp_path)],
            chunk_size=180,
            chunk_overlap=20,
            top_k=2,
            rerank_top_n=3,
        )
    )

    result = pipeline.query("Which team requires quarterly risk reviews before releases?")

    assert result.citations
    assert "quarterly risk reviews" in result.context.lower()
    assert any("handbook.md" in citation.source for citation in result.citations)


def test_rag_pipeline_handles_empty_documents(tmp_path):
    write_text_file(tmp_path / "empty.md", "")

    pipeline = RAGPipeline(
        RAGPipelineConfig(
            knowledge_base_paths=[str(tmp_path)],
            chunk_size=100,
            chunk_overlap=10,
        )
    )

    result = pipeline.query("anything")

    assert "No local knowledge documents were loaded" in result.context
    assert result.citations == []


def test_rag_pipeline_handles_empty_results(tmp_path):
    write_text_file(
        tmp_path / "policy.txt",
        """
        Budget approvals are reviewed by the finance committee every month.
        """,
    )

    pipeline = RAGPipeline(
        RAGPipelineConfig(
            knowledge_base_paths=[str(tmp_path)],
            chunk_size=140,
            chunk_overlap=10,
            top_k=2,
            rerank_top_n=3,
        )
    )

    result = pipeline.query("quantum entanglement experiment schedule")

    assert "No relevant local knowledge base results were found" in result.context
    assert result.citations == []


def test_hybrid_mode_exposes_web_and_rag_tools(tmp_path):
    write_text_file(
        tmp_path / "handbook.md",
        """
        Atlas Launch requires quarterly risk reviews.
        """,
    )

    hybrid_tools = asyncio.run(
        get_all_tools(
            {
                "configurable": {
                    "search_api": "tavily",
                    "rag_enabled": True,
                    "retrieval_mode": "hybrid",
                    "rag_knowledge_base_paths": [str(tmp_path)],
                }
            }
        )
    )
    hybrid_names = [
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        for tool in hybrid_tools
    ]

    rag_only_tools = asyncio.run(
        get_all_tools(
            {
                "configurable": {
                    "search_api": "tavily",
                    "rag_enabled": True,
                    "retrieval_mode": "rag_only",
                    "rag_knowledge_base_paths": [str(tmp_path)],
                }
            }
        )
    )
    rag_only_names = [
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        for tool in rag_only_tools
    ]

    assert "web_search" in hybrid_names
    assert "rag_search" in hybrid_names
    assert "rag_search" in rag_only_names
    assert "web_search" not in rag_only_names
