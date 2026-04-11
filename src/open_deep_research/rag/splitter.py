"""Text splitting utilities for the local RAG pipeline."""

from open_deep_research.rag.types import RAGChunk, RAGDocument


def split_documents(
    documents: list[RAGDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[RAGChunk]:
    """Split documents into chunks with stable chunk identifiers."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunked_documents: list[RAGChunk] = []
    for document_index, document in enumerate(documents):
        split_texts = _split_text(
            document.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk_index, chunk_text in enumerate(split_texts):
            chunked_documents.append(
                RAGChunk(
                    content=chunk_text,
                    source=document.source,
                    title=document.title,
                    chunk_id=f"doc-{document_index + 1}-chunk-{chunk_index + 1}",
                    metadata={
                        **document.metadata,
                        "document_index": document_index,
                        "chunk_index": chunk_index,
                    },
                )
            )
    return chunked_documents


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks while preferring sentence and paragraph boundaries."""
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []

    chunks: list[str] = []
    start_index = 0
    while start_index < len(normalized_text):
        chunk_end = min(start_index + chunk_size, len(normalized_text))

        if chunk_end < len(normalized_text):
            preferred_end = _find_preferred_break(normalized_text, start_index, chunk_end)
            if preferred_end is not None and preferred_end > start_index:
                chunk_end = preferred_end

        chunk_text = normalized_text[start_index:chunk_end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if chunk_end >= len(normalized_text):
            break

        next_start = max(chunk_end - chunk_overlap, start_index + 1)
        if next_start >= chunk_end:
            next_start = chunk_end
        start_index = next_start

    return chunks


def _find_preferred_break(text: str, start_index: int, chunk_end: int) -> int | None:
    """Find a natural boundary near the end of a candidate chunk."""
    minimum_boundary = start_index + max(1, (chunk_end - start_index) // 2)
    for separator in ("\n\n", "\n", ". ", " "):
        separator_index = text.rfind(separator, minimum_boundary, chunk_end)
        if separator_index != -1:
            return separator_index + len(separator)
    return None
