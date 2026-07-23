"""Compatibility exports for query-time RAG image helpers."""

from open_deep_research.rag.loaders import query_images as _query_images
from open_deep_research.rag.loaders.knowledge import extract_routed_image_bytes_text


def build_query_image_context(*args, **kwargs):
    """Recognize user query images while preserving legacy monkeypatch hooks."""
    _query_images.extract_routed_image_bytes_text = extract_routed_image_bytes_text
    return _query_images.build_query_image_context(*args, **kwargs)


__all__ = [
    "build_query_image_context",
    "extract_routed_image_bytes_text",
]
