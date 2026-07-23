"""Budget accounting helpers for bounded deep research runs."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from langchain_core.messages import AIMessage

from open_deep_research.configuration import Configuration

BUDGET_COUNTER_FIELDS = (
    "model_calls",
    "tool_calls",
    "search_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)
BUDGET_LIST_FIELDS = ("budget_warnings", "degradation_reasons")
SEARCH_TOOL_NAMES = {"web_search", "rag_search"}
MIN_MODEL_CALLS_PER_RESEARCH_UNIT = 2
TOKEN_CHAR_RATIO = 4
_BUDGET_CAPTURE: ContextVar[dict[str, Any] | None] = ContextVar(
    "open_deep_research_budget_capture",
    default=None,
)


def empty_budget_usage() -> dict[str, Any]:
    """Return an empty budget usage payload."""
    return {
        "model_calls": 0,
        "tool_calls": 0,
        "search_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "budget_warnings": [],
        "degradation_reasons": [],
    }


def normalize_budget_usage(value: Any) -> dict[str, Any]:
    """Normalize partial or absent budget usage into the canonical dict shape."""
    usage = empty_budget_usage()
    if value is None:
        return usage
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return usage

    for field in BUDGET_COUNTER_FIELDS:
        try:
            usage[field] = int(value.get(field, 0) or 0)
        except (TypeError, ValueError):
            usage[field] = 0

    for field in BUDGET_LIST_FIELDS:
        items = value.get(field, []) or []
        if isinstance(items, str):
            items = [items]
        usage[field] = [str(item) for item in items if str(item)]

    return usage


def merge_budget_usage(current_value: Any, new_value: Any = None) -> dict[str, Any]:
    """Merge two budget usage payloads by summing counters and de-duplicating notes."""
    merged = normalize_budget_usage(current_value)
    incoming = normalize_budget_usage(new_value)

    for field in BUDGET_COUNTER_FIELDS:
        merged[field] += incoming[field]

    for field in BUDGET_LIST_FIELDS:
        seen = set(merged[field])
        for item in incoming[field]:
            if item not in seen:
                merged[field].append(item)
                seen.add(item)

    return merged


def diff_budget_usage(total_value: Any, baseline_value: Any) -> dict[str, Any]:
    """Return the non-negative usage delta between a total payload and baseline."""
    total = normalize_budget_usage(total_value)
    baseline = normalize_budget_usage(baseline_value)
    diff = empty_budget_usage()

    for field in BUDGET_COUNTER_FIELDS:
        diff[field] = max(0, total[field] - baseline[field])

    for field in BUDGET_LIST_FIELDS:
        baseline_items = set(baseline[field])
        diff[field] = [item for item in total[field] if item not in baseline_items]

    return diff


def start_budget_capture() -> Token:
    """Start capturing nested budget usage in the current async context."""
    return _BUDGET_CAPTURE.set(empty_budget_usage())


def stop_budget_capture(token: Token) -> dict[str, Any]:
    """Stop nested budget capture and return the captured usage."""
    captured_usage = normalize_budget_usage(_BUDGET_CAPTURE.get())
    _BUDGET_CAPTURE.reset(token)
    return captured_usage


def capture_budget_usage(usage: Any) -> None:
    """Record nested budget usage when a capture context is active."""
    captured_usage = _BUDGET_CAPTURE.get()
    if captured_usage is None:
        return
    merged_usage = merge_budget_usage(captured_usage, usage)
    captured_usage.clear()
    captured_usage.update(merged_usage)


def capture_model_response(response: Any) -> None:
    """Record a nested model response when a capture context is active."""
    capture_budget_usage(budget_from_model_response(response))


def budget_usage_with_reason(reason: str) -> dict[str, Any]:
    """Create a budget payload that only records a degradation reason."""
    usage = empty_budget_usage()
    usage["degradation_reasons"] = [reason]
    return usage


def budget_usage_with_warning(warning: str) -> dict[str, Any]:
    """Create a budget payload that only records a warning."""
    usage = empty_budget_usage()
    usage["budget_warnings"] = [warning]
    return usage


def extract_token_usage(message: Any) -> dict[str, int]:
    """Extract token usage from LangChain message metadata when providers expose it."""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not isinstance(message, AIMessage):
        return usage

    usage_metadata = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}

    input_tokens = (
        usage_metadata.get("input_tokens")
        or token_usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or 0
    )
    output_tokens = (
        usage_metadata.get("output_tokens")
        or token_usage.get("output_tokens")
        or token_usage.get("completion_tokens")
        or 0
    )
    total_tokens = (
        usage_metadata.get("total_tokens")
        or token_usage.get("total_tokens")
        or (input_tokens + output_tokens)
        or 0
    )

    usage["input_tokens"] = int(input_tokens or 0)
    usage["output_tokens"] = int(output_tokens or 0)
    usage["total_tokens"] = int(total_tokens or 0)
    return usage


def budget_from_model_response(response: Any) -> dict[str, Any]:
    """Create a budget payload for one model response."""
    usage = empty_budget_usage()
    usage["model_calls"] = 1
    token_usage = extract_token_usage(response)
    for field, value in token_usage.items():
        usage[field] = value
    return usage


def budget_from_native_search() -> dict[str, Any]:
    """Create a budget payload for provider-side native web search."""
    usage = empty_budget_usage()
    usage["tool_calls"] = 1
    usage["search_calls"] = 1
    return usage


def budget_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    tools_by_name: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a budget payload for executed tool calls."""
    usage = empty_budget_usage()
    usage["tool_calls"] = len(tool_calls)
    usage["search_calls"] = sum(
        1 for tool_call in tool_calls if is_search_tool_call(tool_call, tools_by_name)
    )
    return usage


def is_search_tool_call(
    tool_call: dict[str, Any],
    tools_by_name: dict[str, Any] | None = None,
) -> bool:
    """Return whether a tool call is a retrieval/search call."""
    name = tool_call.get("name")
    if name in SEARCH_TOOL_NAMES:
        return True

    if not tools_by_name or name not in tools_by_name:
        return False

    tool = tools_by_name[name]
    if isinstance(tool, dict):
        return (
            tool.get("type") in {"search", "web_search_20250305", "web_search_preview"}
            or tool.get("name") in SEARCH_TOOL_NAMES
        )

    metadata = getattr(tool, "metadata", None) or {}
    return metadata.get("type") == "search" or metadata.get("name") in SEARCH_TOOL_NAMES


def can_spend_model_call(
    configurable: Configuration,
    usage: Any,
    *,
    reserve_final_report_call: bool = False,
) -> bool:
    """Return whether another model call can be made."""
    if not configurable.budget_enabled:
        return True
    limit = configurable.max_model_calls
    if limit is None:
        return True

    reserve = 1 if reserve_final_report_call and configurable.reserve_final_report_call else 0
    effective_limit = max(0, limit - reserve)
    return normalize_budget_usage(usage)["model_calls"] < effective_limit


def can_spend_tool_call(
    configurable: Configuration,
    usage: Any,
    increment: dict[str, Any],
) -> bool:
    """Return whether the tool/search budget can absorb the increment."""
    if not configurable.budget_enabled:
        return True

    current = normalize_budget_usage(usage)
    incoming = normalize_budget_usage(increment)
    if (
        configurable.max_tool_calls is not None
        and current["tool_calls"] + incoming["tool_calls"] > configurable.max_tool_calls
    ):
        return False
    if (
        configurable.max_search_calls is not None
        and current["search_calls"] + incoming["search_calls"] > configurable.max_search_calls
    ):
        return False
    return True


def filter_tool_calls_for_budget(
    configurable: Configuration,
    usage: Any,
    tool_calls: list[dict[str, Any]],
    tools_by_name: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split tool calls into executable and skipped calls under the configured budget."""
    if not configurable.budget_enabled:
        return tool_calls, []

    allowed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    projected_usage = normalize_budget_usage(usage)

    for tool_call in tool_calls:
        increment = budget_from_tool_calls([tool_call], tools_by_name)
        if can_spend_tool_call(configurable, projected_usage, increment):
            allowed.append(tool_call)
            projected_usage = merge_budget_usage(projected_usage, increment)
        else:
            skipped.append(tool_call)

    return allowed, skipped


def available_research_unit_slots(configurable: Configuration, usage: Any) -> int | None:
    """Return how many new researcher subgraphs can start, or None when unlimited."""
    if not configurable.budget_enabled or configurable.max_model_calls is None:
        return None

    current = normalize_budget_usage(usage)["model_calls"]
    reserve = 1 if configurable.reserve_final_report_call else 0
    remaining = max(0, configurable.max_model_calls - reserve - current)
    return remaining // MIN_MODEL_CALLS_PER_RESEARCH_UNIT


def is_over_budget(configurable: Configuration, usage: Any) -> bool:
    """Return whether any configured budget has already been exceeded."""
    if not configurable.budget_enabled:
        return False
    normalized = normalize_budget_usage(usage)
    limits = {
        "model_calls": configurable.max_model_calls,
        "tool_calls": configurable.max_tool_calls,
        "search_calls": configurable.max_search_calls,
        "input_tokens": configurable.max_input_tokens,
        "output_tokens": configurable.max_output_tokens,
    }
    return any(limit is not None and normalized[field] >= limit for field, limit in limits.items())


def get_budget_warnings(configurable: Configuration, usage: Any) -> list[str]:
    """Return warning strings for budgets at or above the configured warning ratio."""
    if not configurable.budget_enabled:
        return []

    normalized = normalize_budget_usage(usage)
    warning_ratio = configurable.budget_warning_ratio
    limits = {
        "model_calls": configurable.max_model_calls,
        "tool_calls": configurable.max_tool_calls,
        "search_calls": configurable.max_search_calls,
        "input_tokens": configurable.max_input_tokens,
        "output_tokens": configurable.max_output_tokens,
    }
    warnings = []
    for field, limit in limits.items():
        if limit is None or limit <= 0:
            continue
        if normalized[field] / limit >= warning_ratio:
            warnings.append(f"{field} is at {normalized[field]}/{limit}.")
    return warnings


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length for provider-agnostic budget shaping."""
    if not text:
        return 0
    return max(1, len(text) // TOKEN_CHAR_RATIO)


def truncate_text_to_token_budget(text: str, token_budget: int) -> tuple[str, bool]:
    """Approximate-truncate text to fit within a token budget."""
    if token_budget <= 0:
        return "", bool(text)
    if estimate_tokens(text) <= token_budget:
        return text, False
    return text[: token_budget * TOKEN_CHAR_RATIO].rstrip(), True


def remaining_input_tokens(configurable: Configuration, usage: Any) -> int | None:
    """Return remaining input tokens, if the user configured that budget."""
    if not configurable.budget_enabled or configurable.max_input_tokens is None:
        return None
    return configurable.max_input_tokens - normalize_budget_usage(usage)["input_tokens"]


def remaining_output_tokens(configurable: Configuration, usage: Any) -> int | None:
    """Return remaining output tokens, if the user configured that budget."""
    if not configurable.budget_enabled or configurable.max_output_tokens is None:
        return None
    return configurable.max_output_tokens - normalize_budget_usage(usage)["output_tokens"]


def format_budget_summary(configurable: Configuration, usage: Any) -> str:
    """Format a concise budget summary for final reports."""
    if not configurable.budget_enabled:
        return ""

    normalized = normalize_budget_usage(usage)
    lines = [
        "### Budget usage",
        (
            f"- Model calls: {normalized['model_calls']}"
            + _format_limit(configurable.max_model_calls)
        ),
        (
            f"- Tool calls: {normalized['tool_calls']}"
            + _format_limit(configurable.max_tool_calls)
        ),
        (
            f"- Search calls: {normalized['search_calls']}"
            + _format_limit(configurable.max_search_calls)
        ),
        (
            f"- Input tokens: {normalized['input_tokens']}"
            + _format_limit(configurable.max_input_tokens)
        ),
        (
            f"- Output tokens: {normalized['output_tokens']}"
            + _format_limit(configurable.max_output_tokens)
        ),
    ]

    warnings = normalized["budget_warnings"] + get_budget_warnings(configurable, normalized)
    reasons = normalized["degradation_reasons"]
    if warnings:
        lines.append("- Warnings: " + "; ".join(dict.fromkeys(warnings)))
    if reasons:
        lines.append("- Degradation: " + "; ".join(dict.fromkeys(reasons)))

    return "\n".join(lines)


def append_budget_summary(report: str, configurable: Configuration, usage: Any) -> str:
    """Append budget usage details to a final report when budgeting is enabled."""
    summary = format_budget_summary(configurable, usage)
    if not summary:
        return report
    return f"{report.rstrip()}\n\n---\n\n{summary}"


def _format_limit(limit: int | None) -> str:
    if limit is None:
        return ""
    return f" / {limit}"
