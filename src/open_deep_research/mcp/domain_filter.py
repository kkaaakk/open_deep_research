"""Scene-based tool domain filtering and role-based tool routing.

All domain metadata lives in :data:`DOMAIN_REGISTRY` — the single source of
truth.  Keyword lists, prompt labels, and classifier descriptions are all
derived from it, so adding a new domain only requires one update.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain Registry — single source of truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainDef:
    """Definition of one tool domain.

    Attributes
    ----------
    name:
        Unique key used in metadata and code (e.g. ``"database"``).
    label:
        Human-readable heading for system prompts
        (e.g. ``"Database Tools (DBHub)"``).
    description:
        One-line summary for LLM domain classification in
        ``write_research_brief``.
    always_active:
        If ``True``, this domain is never filtered out (always included).
    keywords:
        Trigger words for keyword-based fallback detection.  Case-insensitive
        match against the research topic.
    """

    name: str
    label: str
    description: str
    always_active: bool = False
    keywords: list[str] = field(default_factory=list)


# fmt: off
DOMAIN_REGISTRY: list[DomainDef] = [
    DomainDef(
        name="core",
        label="Core Tools (always available)",
        description="Strategic thinking and research completion — always active, never filtered.",
        always_active=True,
    ),
    DomainDef(
        name="web_search",
        label="Web Search",
        description="Live web search for external or up-to-date information — always active by default.",
        always_active=True,
    ),
    DomainDef(
        name="rag",
        label="Local Knowledge Base (RAG)",
        description="Search internal company documents, chat memory, or configured knowledge bases.",
        keywords=[
            "rag", "knowledge base", "internal", "company document",
            "chat memory", "local document", "知识库", "内部文档", "本地检索",
        ],
    ),
    DomainDef(
        name="database",
        label="Database Tools (DBHub)",
        description="SQL queries, schema exploration, table/index inspection for PostgreSQL, MySQL, SQLite, SQL Server, MariaDB.",
        keywords=[
            "sql", "query", "select", "insert", "update", "delete",
            "database", "db", "table", "schema", "postgres", "mysql",
            "sqlite", "mariadb", "column", "index", "ddl", "dml",
            "join", "transaction", "stored procedure", "view",
            "数据库", "查询", "表", "字段", "索引", "SQL",
        ],
    ),
    DomainDef(
        name="document",
        label="Document Conversion (MarkItDown)",
        description="Convert PDFs, Word documents, Excel files, PowerPoints, images, audio, HTML, and other formats to Markdown.",
        keywords=[
            "convert", "markdown", "pdf", "word", "excel", "ppt",
            "docx", "xlsx", "pptx", "file", "document", "extract",
            "ocr", "epub", "html to",
            "转换", "文件", "文档", "导出", "提取",
        ],
    ),
    DomainDef(
        name="feishu",
        label="Feishu / Lark",
        description="Send messages, create/edit documents, search knowledge bases, manage calendars, manipulate Bitable (多维表格) — all via Feishu/Lark Open API.",
        keywords=[
            "feishu", "lark", "飞书",
            "发送消息", "创建文档", "多维表格", "bitable",
            "日历", "消息", "云文档", "知识库", "wiki",
            "会议", "审批", "通讯录",
        ],
    ),
    DomainDef(
        name="social_media",
        label="Social Media & Public Opinion",
        description="Search social media posts, aggregate sentiment, track complaints, and monitor brand risk across platforms.",
        keywords=[
            "social media", "twitter", "x post", "complaint", "sentiment",
            "public opinion", "brand",
            "舆情", "投诉", "社交媒体", "品牌",
            "舆论", "负面", "好评",
        ],
    ),
    DomainDef(
        name="external_mcp",
        label="External MCP Tools",
        description="Additional tools from legacy or custom MCP server configurations.",
    ),
]
# fmt: on

# Fast lookups derived from the registry (kept in sync automatically)
_DOMAIN_BY_NAME: dict[str, DomainDef] = {d.name: d for d in DOMAIN_REGISTRY}
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    d.name: d.keywords for d in DOMAIN_REGISTRY if d.keywords
}
_ALWAYS_ACTIVE_DOMAINS: set[str] = {
    d.name for d in DOMAIN_REGISTRY if d.always_active
}
_DOMAIN_PROMPT_ORDER: list[tuple[str, str]] = [
    (d.name, d.label) for d in DOMAIN_REGISTRY
]

# Built-in (non-MCP) tool name → domain mapping
_BUILTIN_NAME_TO_DOMAIN: dict[str, str] = {
    "think_tool": "core",
    "ResearchComplete": "core",
    "web_search": "web_search",
    "rag_search": "rag",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_domain(name: str) -> DomainDef | None:
    """Return the :class:`DomainDef` for *name*, or ``None``."""
    return _DOMAIN_BY_NAME.get(name)


def get_domain_description(name: str) -> str:
    """Return the one-line description for *name*."""
    dom = _DOMAIN_BY_NAME.get(name)
    return dom.description if dom else ""


def get_domain_label(name: str) -> str:
    """Return the human-readable label for *name*."""
    dom = _DOMAIN_BY_NAME.get(name)
    return dom.label if dom else name


def iter_domain_labels(domains: set[str]) -> list[tuple[str, str]]:
    """Return ordered ``(name, label)`` pairs for the given *domains*."""
    seen = {name for name, _ in _DOMAIN_PROMPT_ORDER if name in domains}
    # Also include any domains not in the order list
    for name in sorted(domains - seen):
        seen.add(name)
        _DOMAIN_PROMPT_ORDER.append((name, get_domain_label(name)))
    return [(name, label) for name, label in _DOMAIN_PROMPT_ORDER if name in domains]


def build_domain_classifier_prompt() -> str:
    """Generate the domain-classification section for the LLM prompt.

    Used by ``write_research_brief`` so the LLM can output ``relevant_domains``.
    Dynamically built from the registry — no hardcoded domain lists.
    """
    lines: list[str] = []
    for dom in DOMAIN_REGISTRY:
        if dom.name in ("core", "web_search"):
            continue  # always active, LLM doesn't need to list them
        kw_hint = ""
        if dom.keywords:
            kw_hint = f" (keywords: {', '.join(dom.keywords[:6])})"
        lines.append(f"- **{dom.name}**: {dom.description}{kw_hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool domain resolution
# ---------------------------------------------------------------------------


def _tool_domain(tool) -> str:
    """Resolve the domain of a single tool (LangChain tool or native-search dict).

    Priority:
    1. ``tool.metadata["tool_domain"]`` — set by MCP loader
    2. ``tool["name"]`` / ``tool.name`` → ``_BUILTIN_NAME_TO_DOMAIN``
    3. ``"web_search"`` — fallback for anonymous dict tools
    4. ``"external_mcp"`` — fallback for everything else
    """
    # 1) MCP-loaded tools have explicit domain metadata
    if not isinstance(tool, dict):
        metadata = getattr(tool, "metadata", None) or {}
        if isinstance(metadata, dict) and "tool_domain" in metadata:
            return str(metadata["tool_domain"])

    # 2) Name-based matching — works for both BaseTool and dict tools
    name = ""
    if isinstance(tool, dict):
        name = tool.get("name", "")
    else:
        name = getattr(tool, "name", "")
    if name in _BUILTIN_NAME_TO_DOMAIN:
        return _BUILTIN_NAME_TO_DOMAIN[name]

    # 3) Anonymous dict tools → assume web_search
    if isinstance(tool, dict):
        return "web_search"

    # 4) Fallback
    return "external_mcp"


def tag_builtin_tools(tools: list) -> list:
    """Ensure every built-in tool carries ``tool_domain`` metadata.

    Call this once after ``get_all_tools()`` assembles the full tool list.
    Tools that already have domain metadata (MCP tools) are left untouched.
    """
    for tool in tools:
        if isinstance(tool, dict):
            continue
        metadata = getattr(tool, "metadata", None)
        if metadata is None:
            tool.metadata = {}
        if isinstance(tool.metadata, dict) and "tool_domain" not in tool.metadata:
            name = getattr(tool, "name", "")
            domain = _BUILTIN_NAME_TO_DOMAIN.get(name, "external_mcp")
            tool.metadata["tool_domain"] = domain
    return tools


# ---------------------------------------------------------------------------
# Domain detection & filtering
# ---------------------------------------------------------------------------


def classify_tools(tools: list[BaseTool]) -> dict[str, list[BaseTool]]:
    """Group *tools* by their domain category."""
    buckets: dict[str, list[BaseTool]] = {}
    for tool in tools:
        domain = _tool_domain(tool)
        buckets.setdefault(domain, []).append(tool)
    return buckets


def detect_active_domains(
    research_topic: str,
    *,
    agent_role: str = "general_research",
) -> set[str]:
    """Return the set of domains relevant to *research_topic* (keyword fallback)."""
    active = set(_ALWAYS_ACTIVE_DOMAINS)
    text = research_topic.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            active.add(domain)
    logger.debug(
        "Keyword domain detection for role=%s topic=%.120s → %s",
        agent_role, research_topic, sorted(active),
    )
    return active


def filter_tools_by_domain(
    tools: list[BaseTool],
    active_domains: set[str],
) -> list[BaseTool]:
    """Return only the tools whose domain is in *active_domains*."""
    filtered: list[BaseTool] = []
    dropped: list[str] = []
    for tool in tools:
        domain = _tool_domain(tool)
        if domain in active_domains:
            filtered.append(tool)
        else:
            dropped.append(tool.name)
    if dropped:
        logger.info(
            "Domain filter: dropped %d tool(s) from inactive domains: %s",
            len(dropped), ", ".join(dropped),
        )
    return filtered


def get_filtered_tools(
    tools: list[BaseTool],
    research_topic: str,
    *,
    agent_role: str = "general_research",
) -> list[BaseTool]:
    """Convenience: detect active domains and filter *tools* in one call."""
    active = detect_active_domains(research_topic, agent_role=agent_role)
    return filter_tools_by_domain(tools, active)


def tool_domain_summary(tools: list[BaseTool]) -> str:
    """Return a human-readable summary of tools grouped by domain."""
    buckets = classify_tools(tools)
    lines: list[str] = []
    for domain in sorted(buckets):
        names = sorted(t.name for t in buckets[domain])
        lines.append(f"  [{domain}] {', '.join(names)}")
    return "\n".join(lines)
