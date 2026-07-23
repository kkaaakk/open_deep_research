"""Query-time image recognition helpers for RAG-aware questions."""

import base64
import binascii
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse

from open_deep_research.rag.loaders.knowledge import (
    SUPPORTED_IMAGE_EXTENSIONS,
    extract_routed_image_bytes_text,
)

LOGGER = logging.getLogger(__name__)
DATA_IMAGE_RE = re.compile(
    r"data:(?P<mime>image/[^;]+);base64,(?P<data>[A-Za-z0-9+/=]+)",
    re.IGNORECASE,
)
TEXT_IMAGE_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:\\|/|\.{1,2}[/\\])[^<>:\"|?*\n\r]+?\."
    r"(?:png|jpe?g|webp|bmp|tiff?))",
    re.IGNORECASE,
)


def build_query_image_context(
    messages: Sequence[Any],
    *,
    provider: str,
    ocr_languages: str,
    vision_enabled: bool,
    vision_model: str,
    vision_prompt: str,
    vision_max_tokens: int,
    max_images: int,
    max_bytes: int = 5_000_000,
) -> str:
    """Recognize images attached to user messages and return temporary query context."""
    if max_images <= 0:
        return ""

    sections = []
    for image_index, image_ref in enumerate(_iter_message_image_refs(messages), start=1):
        if len(sections) >= max_images:
            break
        try:
            image_bytes, source_label = _read_image_ref(image_ref, max_bytes=max_bytes)
        except Exception as exc:
            LOGGER.warning("Skipping query image %s: %s", image_index, exc)
            continue

        text, metadata = extract_routed_image_bytes_text(
            image_bytes,
            provider,
            ocr_languages,
            vision_enabled,
            vision_model,
            vision_prompt,
            vision_max_tokens,
        )
        if not text.strip():
            continue

        sections.append(
            "\n".join(
                [
                    f"User image {len(sections) + 1}",
                    f"Source: {source_label}",
                    f"Route: {metadata.get('image_route', 'unknown')}",
                    f"Reason: {metadata.get('image_route_reason', 'unknown')}",
                    text.strip(),
                ]
            )
        )

    return "\n\n".join(sections)


def _iter_message_image_refs(messages: Sequence[Any]) -> Iterable[Any]:
    for message in messages:
        content = _message_content(message)
        yield from _iter_content_image_refs(content)


def _message_content(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("content")
    return getattr(message, "content", None)


def _iter_content_image_refs(content: Any) -> Iterable[Any]:
    if isinstance(content, str):
        yield from _iter_text_image_refs(content)
        return

    if not isinstance(content, list):
        return

    for item in content:
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in {"image_url", "input_image"}:
            image_url = item.get("image_url") or item.get("url")
            if isinstance(image_url, Mapping):
                image_url = image_url.get("url")
            if isinstance(image_url, str) and image_url.strip():
                yield image_url.strip()
            continue

        if item_type in {"image", "input_image"}:
            source = item.get("source")
            if isinstance(source, Mapping) and source.get("type") == "base64":
                data = source.get("data")
                media_type = source.get("media_type") or "image/png"
                if isinstance(data, str) and data.strip():
                    yield f"data:{media_type};base64,{data.strip()}"
                continue

        for key in ("path", "image_path", "file_path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                yield value.strip()


def _iter_text_image_refs(text: str) -> Iterable[str]:
    for data_url in DATA_IMAGE_RE.finditer(text):
        yield data_url.group(0)
    for match in TEXT_IMAGE_PATH_RE.finditer(text):
        yield match.group("path").strip().strip("'\"")


def _read_image_ref(image_ref: Any, *, max_bytes: int) -> tuple[bytes, str]:
    if isinstance(image_ref, bytes):
        return _validate_size(image_ref, max_bytes=max_bytes), "message image bytes"
    if not isinstance(image_ref, str):
        raise ValueError("unsupported query image reference")

    stripped_ref = image_ref.strip()
    data_match = DATA_IMAGE_RE.fullmatch(stripped_ref)
    if data_match:
        try:
            image_bytes = base64.b64decode(data_match.group("data"), validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid base64 data URL image") from exc
        return (
            _validate_size(image_bytes, max_bytes=max_bytes),
            data_match.group("mime"),
        )

    parsed = urlparse(stripped_ref)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        if re.match(r"^/[A-Za-z]:/", str(path)):
            path = Path(str(path)[1:])
        return _read_image_path(path, max_bytes=max_bytes)

    if parsed.scheme in {"http", "https"}:
        raise ValueError("remote query image URLs are not fetched locally")

    return _read_image_path(Path(stripped_ref).expanduser(), max_bytes=max_bytes)


def _read_image_path(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported query image extension: {path.suffix}")
    resolved_path = path.resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"query image does not exist: {resolved_path}")
    return (
        _validate_size(resolved_path.read_bytes(), max_bytes=max_bytes),
        resolved_path.as_posix(),
    )


def _validate_size(image_bytes: bytes, *, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max query image bytes must be greater than 0")
    if len(image_bytes) > max_bytes:
        raise ValueError("query image exceeds configured byte limit")
    return image_bytes
