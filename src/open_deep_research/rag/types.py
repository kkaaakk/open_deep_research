"""Shared types for the local RAG pipeline."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class RAGDocument(BaseModel):
    """A source document that can be chunked and indexed."""

    content: str
    source: str
    title: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGChunk(BaseModel):
    """A chunk derived from a source document."""

    content: str
    source: str
    title: Optional[str] = None
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A retrieved chunk and its ranking information."""

    chunk: RAGChunk
    score: float
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    """A formatted citation for a retrieved chunk."""

    citation_id: int
    title: str
    source: str
    chunk_id: str
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerReadyContext(BaseModel):
    """Grounded context ready to be injected into the research loop."""

    query: str
    context: str
    citations: list[Citation] = Field(default_factory=list)
    matched_chunks: list[RetrievalResult] = Field(default_factory=list)
