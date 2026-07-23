"""MCP (Model Context Protocol) server integration package.

Provides multi-server MCP tool loading with fault isolation, tool name
conflict resolution, scene-based domain filtering, and backward
compatibility with the legacy single-server ``mcp_config`` configuration.

All domain metadata lives in :data:`domain_filter.DOMAIN_REGISTRY` — the
single source of truth for domain names, labels, descriptions, and keywords.
"""

from open_deep_research.mcp.domain_filter import (
    DOMAIN_REGISTRY,
    DomainDef,
    build_domain_classifier_prompt,
    classify_tools,
    detect_active_domains,
    filter_tools_by_domain,
    get_domain,
    get_domain_description,
    get_domain_label,
    get_filtered_tools,
    iter_domain_labels,
    tag_builtin_tools,
    tool_domain_summary,
)
from open_deep_research.mcp.tools import load_mcp_tools

__all__ = [
    "DOMAIN_REGISTRY",
    "DomainDef",
    "load_mcp_tools",
    "build_domain_classifier_prompt",
    "classify_tools",
    "detect_active_domains",
    "filter_tools_by_domain",
    "get_domain",
    "get_domain_description",
    "get_domain_label",
    "get_filtered_tools",
    "iter_domain_labels",
    "tag_builtin_tools",
    "tool_domain_summary",
]
