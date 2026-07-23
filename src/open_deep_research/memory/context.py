"""Runtime identity helpers for memory.

All memory writers, readers, and RAG tools should call this module so the same
conversation is used when writing MySQL rows and when retrieving indexed memory.
"""

from typing import Any, Mapping, Optional

from langchain_core.runnables import RunnableConfig


def get_conversation_id(config: RunnableConfig | Mapping[str, Any] | None) -> str:
    """Return the stable conversation/thread id used to scope memory."""
    configurable, metadata = _config_parts(config)
    return (
        _first_text(
            configurable.get("conversation_id"),
            configurable.get("thread_id"),
            metadata.get("conversation_id"),
            metadata.get("thread_id"),
        )
        or get_user_id(config)
        or "default"
    )


def get_user_id(config: RunnableConfig | Mapping[str, Any] | None) -> Optional[str]:
    """Return the user id when the runtime config exposes one.

    Older flows used `metadata.owner` as the closest user-level scope, so it is
    kept as a compatibility fallback.
    """
    configurable, metadata = _config_parts(config)
    return _first_text(
        configurable.get("user_id"),
        metadata.get("user_id"),
        metadata.get("owner"),
        configurable.get("owner"),
    )


def _config_parts(
    config: RunnableConfig | Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    config = config or {}
    return (
        config.get("configurable", {}) or {},
        config.get("metadata", {}) or {},
    )


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None

