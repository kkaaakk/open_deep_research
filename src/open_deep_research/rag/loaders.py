"""Document loaders for the local RAG pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from open_deep_research.rag.types import RAGDocument

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def load_documents_from_paths(
    knowledge_base_paths: Sequence[str],
    json_text_fields: Sequence[str] | None = None,
) -> list[RAGDocument]:
    """Load supported documents from configured files or directories."""
    loaded_documents: list[RAGDocument] = []
    for configured_path in knowledge_base_paths:
        path = Path(configured_path).expanduser()
        if not path.exists():
            LOGGER.warning("Skipping missing RAG path: %s", configured_path)
            continue
        for candidate_path in _iter_supported_paths(path):
            try:
                loaded_documents.extend(
                    _load_document(candidate_path, json_text_fields=json_text_fields)
                )
            except Exception as exc:  # pragma: no cover - defensive logging path
                LOGGER.warning("Failed to load RAG document %s: %s", candidate_path, exc)
    return loaded_documents


def _iter_supported_paths(path: Path) -> Iterable[Path]:
    """Yield supported files for a file or directory input path."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path.resolve()
        return

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate.resolve()


def _load_document(
    path: Path,
    json_text_fields: Sequence[str] | None = None,
) -> list[RAGDocument]:
    """Load a document based on its extension."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        return [
            RAGDocument(
                content=text,
                source=path.as_posix(),
                title=_guess_title(text=text, path=path),
                metadata={"extension": suffix, "path": path.as_posix()},
            )
        ]

    if suffix == ".json":
        return _load_json_documents(path, json_text_fields=json_text_fields)

    if suffix == ".pdf":
        return _load_pdf_document(path)

    return []


def _load_json_documents(
    path: Path,
    json_text_fields: Sequence[str] | None = None,
) -> list[RAGDocument]:
    """Load one or more documents from a JSON file."""
    raw_data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(raw_data, list):
        loaded_documents = []
        for item_index, item in enumerate(raw_data):
            text = _extract_json_text(item, json_text_fields=json_text_fields)
            if not text:
                continue
            title = _extract_json_title(item) or f"{path.stem} item {item_index + 1}"
            loaded_documents.append(
                RAGDocument(
                    content=text,
                    source=f"{path.as_posix()}#item-{item_index + 1}",
                    title=title,
                    metadata={
                        "extension": ".json",
                        "path": path.as_posix(),
                        "item_index": item_index,
                    },
                )
            )
        return loaded_documents

    text = _extract_json_text(raw_data, json_text_fields=json_text_fields)
    if not text:
        return []

    return [
        RAGDocument(
            content=text,
            source=path.as_posix(),
            title=_extract_json_title(raw_data) or path.stem,
            metadata={"extension": ".json", "path": path.as_posix()},
        )
    ]


def _load_pdf_document(path: Path) -> list[RAGDocument]:
    """Load text from a PDF document when PyMuPDF is available."""
    try:
        import fitz
    except ImportError:  # pragma: no cover - dependency exists in this repository
        LOGGER.warning("PyMuPDF is unavailable; skipping PDF file %s", path)
        return []

    pdf_text_parts: list[str] = []
    pdf_title: str | None = None
    page_count = 0

    with fitz.open(path) as pdf_document:
        metadata = pdf_document.metadata or {}
        pdf_title = metadata.get("title") or None
        page_count = pdf_document.page_count
        for page_index in range(pdf_document.page_count):
            page_text = pdf_document.load_page(page_index).get_text("text").strip()
            if page_text:
                pdf_text_parts.append(page_text)

    joined_text = "\n\n".join(pdf_text_parts).strip()
    if not joined_text:
        return []

    return [
        RAGDocument(
            content=joined_text,
            source=path.as_posix(),
            title=pdf_title or path.stem,
            metadata={
                "extension": ".pdf",
                "path": path.as_posix(),
                "page_count": page_count,
            },
        )
    ]


def _extract_json_text(
    data: Any,
    json_text_fields: Sequence[str] | None = None,
) -> str:
    """Extract text content from JSON structures."""
    fragments: list[str] = []

    if json_text_fields:
        for field_path in json_text_fields:
            extracted_value = _extract_field_path(data, field_path)
            fragments.extend(_collect_text_fragments(extracted_value))
    else:
        fragments.extend(_collect_text_fragments(data))

    deduplicated_fragments = []
    seen_fragments = set()
    for fragment in fragments:
        normalized_fragment = fragment.strip()
        if not normalized_fragment or normalized_fragment in seen_fragments:
            continue
        seen_fragments.add(normalized_fragment)
        deduplicated_fragments.append(normalized_fragment)

    return "\n\n".join(deduplicated_fragments)


def _extract_json_title(data: Any) -> str | None:
    """Infer a title from common JSON fields."""
    if not isinstance(data, dict):
        return None
    for title_key in ("title", "name", "headline", "subject"):
        title_value = data.get(title_key)
        if isinstance(title_value, str) and title_value.strip():
            return title_value.strip()
    return None


def _extract_field_path(data: Any, field_path: str) -> Any:
    """Resolve a dotted field path within a JSON structure."""
    current_value = data
    for part in field_path.split("."):
        if isinstance(current_value, list):
            if part.isdigit():
                index = int(part)
                if 0 <= index < len(current_value):
                    current_value = current_value[index]
                    continue
            return None
        if not isinstance(current_value, dict):
            return None
        current_value = current_value.get(part)
        if current_value is None:
            return None
    return current_value


def _collect_text_fragments(value: Any) -> list[str]:
    """Recursively collect meaningful string fragments from arbitrary JSON values."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments = []
        for nested_value in value.values():
            fragments.extend(_collect_text_fragments(nested_value))
        return fragments
    return []


def _guess_title(text: str, path: Path) -> str:
    """Infer a title from content or fallback to the filename stem."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line.startswith("#"):
        stripped_heading = first_line.lstrip("#").strip()
        if stripped_heading:
            return stripped_heading
    if 0 < len(first_line) <= 120:
        return first_line
    return path.stem
