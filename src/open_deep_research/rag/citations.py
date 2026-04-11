"""Citation helpers for the local RAG pipeline."""

from open_deep_research.rag.types import AnswerReadyContext, Citation, RetrievalResult


def build_answer_ready_context(
    query: str,
    matched_chunks: list[RetrievalResult],
) -> AnswerReadyContext:
    """Format retrieved chunks into a grounded context block with citations."""
    if not matched_chunks:
        empty_context = (
            "No relevant local knowledge base results were found for this query. "
            "Try a broader query, add more local documents, or fall back to web search."
        )
        return AnswerReadyContext(query=query, context=empty_context)

    citations = [
        Citation(
            citation_id=index,
            title=result.chunk.title or result.chunk.source,
            source=result.chunk.source,
            chunk_id=result.chunk.chunk_id,
            excerpt=_build_excerpt(result.chunk.content),
            metadata=result.chunk.metadata,
        )
        for index, result in enumerate(matched_chunks, start=1)
    ]

    lines = ["Local knowledge base results:\n"]
    for citation in citations:
        lines.append(f"--- SOURCE {citation.citation_id}: {citation.title} ---")
        lines.append(f"SOURCE: {citation.source}")
        lines.append(f"CHUNK ID: {citation.chunk_id}")
        lines.append(f"EXCERPT:\n{citation.excerpt}\n")

    lines.append("### Sources")
    for citation in citations:
        lines.append(
            f"[{citation.citation_id}] {citation.title}: {citation.source} (chunk: {citation.chunk_id})"
        )

    return AnswerReadyContext(
        query=query,
        context="\n".join(lines).strip(),
        citations=citations,
        matched_chunks=matched_chunks,
    )


def _build_excerpt(text: str, max_length: int = 500) -> str:
    """Create a compact excerpt for citation display."""
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_length:
        return normalized_text
    return normalized_text[: max_length - 3].rstrip() + "..."
