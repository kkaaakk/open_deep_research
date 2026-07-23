"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from open_deep_research.budget import (
    append_budget_summary,
    available_research_unit_slots,
    budget_from_model_response,
    budget_from_native_search,
    budget_from_tool_calls,
    budget_usage_with_reason,
    can_spend_model_call,
    diff_budget_usage,
    estimate_tokens,
    filter_tool_calls_for_budget,
    is_over_budget,
    merge_budget_usage,
    remaining_input_tokens,
    remaining_output_tokens,
    start_budget_capture,
    stop_budget_capture,
    truncate_text_to_token_budget,
)
from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.mcp.domain_filter import (
    build_domain_classifier_prompt,
    get_filtered_tools,
    tag_builtin_tools,
)
from open_deep_research.memory.writer import persist_conversation_memory
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    final_section_writer_instructions,
    lead_researcher_prompt,
    public_opinion_final_report_generation_prompt,
    public_opinion_researcher_prompt,
    public_opinion_supervisor_prompt,
    report_planner_instructions,
    research_system_prompt,
    section_writer_from_role_reports_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.public_opinion_agents import (
    get_public_opinion_agent_spec,
    public_opinion_role_channels,
    public_opinion_role_expectations,
)
from open_deep_research.rag import query_images
from open_deep_research.social_media.tools import (
    SOCIAL_MEDIA_TOOL_NAMES,
    get_social_media_tools,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    PublicOpinionState,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    Section,
    Sections,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    build_dynamic_tool_prompt,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_research_tool_prompt,
    get_today_str,
    has_external_research_tool,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)
LOGGER = logging.getLogger(__name__)


def _public_opinion_mode(configurable: Configuration) -> bool:
    """Return whether the public-opinion business workflow is enabled."""
    return str(configurable.business_scenario).strip().lower() == "public_opinion_risk"


def _business_context(configurable: Configuration) -> str:
    """Build business context for business-scenario prompts."""
    context = (configurable.organization_context or "").strip()
    if context:
        return context
    return (
        "No additional organization context was configured. Use the user's request, "
        "local RAG evidence, and cited public sources without inventing company facts."
    )


def _enabled_business_agents(configurable: Configuration) -> str:
    """Format enabled business agents for prompt injection."""
    agents = configurable.enabled_business_agents or []
    return ", ".join(str(agent).strip() for agent in agents if str(agent).strip())


def _supervisor_system_prompt(configurable: Configuration) -> str:
    """Select the supervisor prompt for the configured workflow."""
    if _public_opinion_mode(configurable):
        return public_opinion_supervisor_prompt.format(
            date=get_today_str(),
            organization_context=_business_context(configurable),
            monitoring_window=configurable.public_opinion_monitoring_window,
            enabled_business_agents=_enabled_business_agents(configurable),
            max_concurrent_research_units=configurable.max_concurrent_research_units,
            max_researcher_iterations=configurable.max_researcher_iterations,
        )
    return lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )


def _researcher_assignment(tool_call: dict, configurable: Configuration) -> dict[str, str]:
    """Extract and normalize a delegated research assignment."""
    args = tool_call.get("args", {})
    research_topic = args.get("research_topic", "")
    agent_role = args.get("agent_role", "general_research") or "general_research"
    expected_output = args.get("expected_output", "") or ""

    if not _public_opinion_mode(configurable):
        agent_role = "general_research"
        expected_output = ""

    assignment_message = research_topic
    if _public_opinion_mode(configurable):
        assignment_message = (
            f"Assigned role: {agent_role}\n"
            f"Expected output: {expected_output or 'Role-appropriate evidence report'}\n\n"
            f"Task:\n{research_topic}"
        )

    return {
        "research_topic": research_topic,
        "agent_role": agent_role,
        "expected_output": expected_output,
        "message": assignment_message,
    }


PUBLIC_OPINION_ROLE_EXPECTATIONS = public_opinion_role_expectations()
PUBLIC_OPINION_ROLE_CHANNELS = public_opinion_role_channels()


def _tool_name(available_tool) -> str:
    """Return a stable name for LangChain tools and provider-native tool dicts."""
    if isinstance(available_tool, dict):
        return available_tool.get("name") or "web_search"
    return getattr(available_tool, "name", "")


def _role_enabled(configurable: Configuration, role: str) -> bool:
    """Return whether a public-opinion role should run for this configuration."""
    enabled_agents = configurable.enabled_business_agents or []
    return role in {str(agent).strip().lower() for agent in enabled_agents}


def _role_context(role_reports: dict[str, str]) -> str:
    """Format upstream role reports for downstream public-opinion agents."""
    if not role_reports:
        return "No upstream role reports are available yet."
    return "\n\n".join(
        f"## {role}\n{report}"
        for role, report in role_reports.items()
        if report
    )


def _agent_private_memory_context(agent_memories: dict[str, list[dict[str, Any]]], role: str) -> str:
    """Format one agent's private short-term memory for prompt injection."""
    memories = list((agent_memories or {}).get(role, []) or [])
    if not memories:
        return "No private memory has been recorded for this agent yet."
    formatted_entries = []
    for index, memory in enumerate(memories[-5:], start=1):
        if isinstance(memory, dict):
            title = str(memory.get("title") or f"Memory {index}")
            content = str(memory.get("content") or "")
            source = str(memory.get("source") or "agent_private_memory")
            formatted_entries.append(f"{index}. {title} [{source}]\n{content}")
        else:
            formatted_entries.append(f"{index}. {memory}")
    return "\n\n".join(formatted_entries)


def _build_agent_private_memory(role: str, report: str, raw_notes: list[str]) -> dict[str, Any]:
    """Create a compact private memory entry from an agent's completed report."""
    report_text = str(report or "").strip()
    if len(report_text) > 1800:
        report_text = report_text[:1800].rstrip() + "\n[truncated]"
    return {
        "title": f"{role} completed role report",
        "content": report_text or "No report content was produced.",
        "source": "current_public_opinion_run",
        "raw_note_count": len(raw_notes or []),
    }


def _role_tool_prompt(configurable: Configuration, role: str) -> str:
    """Describe the role-specific tool whitelist for the researcher prompt."""
    allowed_channels = get_public_opinion_agent_spec(role).tool_channels
    allowed_tools = ["think_tool", "ResearchComplete"]
    if "web" in allowed_channels:
        allowed_tools.append("web_search")
    if "rag" in allowed_channels:
        allowed_tools.append("rag_search")
    if "mcp" in allowed_channels:
        allowed_tools.extend(sorted(SOCIAL_MEDIA_TOOL_NAMES))
    return (
        f"{get_research_tool_prompt(configurable)}\n\n"
        "Role-specific tool whitelist: "
        f"{', '.join(allowed_tools)}. "
        "Do not attempt to use tools outside this whitelist."
    )


async def _business_agent_tools(config: RunnableConfig, role: str):
    """Return the role-specific tool whitelist for an explicit business agent."""
    allowed_channels = get_public_opinion_agent_spec(role).tool_channels
    all_tools = await get_all_tools(config)
    if "mcp" in allowed_channels:
        all_tools.extend(get_social_media_tools())
    filtered_tools = []

    for available_tool in all_tools:
        name = _tool_name(available_tool)
        if name in {"ResearchComplete", "think_tool"}:
            filtered_tools.append(available_tool)
        elif name == "web_search" and "web" in allowed_channels:
            filtered_tools.append(available_tool)
        elif name == "rag_search" and "rag" in allowed_channels:
            filtered_tools.append(available_tool)
        elif name in SOCIAL_MEDIA_TOOL_NAMES and "mcp" in allowed_channels:
            filtered_tools.append(available_tool)
        elif name not in {"ResearchComplete", "think_tool", "web_search", "rag_search"}:
            if "mcp" in allowed_channels:
                filtered_tools.append(available_tool)

    return filtered_tools


async def _get_filtered_researcher_tools(
    config: RunnableConfig,
    research_topic: str,
    agent_role: str = "general_research",
    *,
    extra_context: str = "",
    llm_domains: list[str] | None = None,
) -> list[Any]:
    """Get tools for a researcher, optionally filtered by domain relevance.

    Domain detection priority:
    1. **LLM-detected domains** from ``write_research_brief`` (highest
       accuracy — the LLM that already reads the full user question
       explicitly classifies relevant domains).
    2. **Keyword fallback** — when LLM domains are empty (budget guard
       skipped the brief generation, or the model returned no domains),
       fast keyword matching on the research topic + extra context.
    3. **Disabled** — when ``tool_domain_filtering_enabled`` is False,
       all tools are returned unfiltered.
    """
    configurable = Configuration.from_runnable_config(config)
    all_tools = await get_all_tools(config)

    # Tag built-in tools (think_tool, ResearchComplete, web_search, …)
    # with domain metadata so MCP and non-MCP tools are uniformly classified.
    all_tools = tag_builtin_tools(all_tools)

    if not configurable.tool_domain_filtering_enabled:
        return all_tools

    # LLM-detected domains → highest priority (already classified by the
    # model that read the full user question in write_research_brief).
    if llm_domains:
        # Always include core + web_search even if LLM omitted them
        active_domains = set(llm_domains) | {"core", "web_search"}
        logger.info(
            "Domain filter: using LLM-detected domains %s → active=%s",
            llm_domains, sorted(active_domains),
        )
        return filter_tools_by_domain(all_tools, active_domains)

    # Keyword fallback — LLM didn't produce domains (budget guard or edge case)
    composite_text = research_topic
    if extra_context and extra_context != research_topic:
        composite_text = f"{research_topic}\n{extra_context}"
    logger.info(
        "Domain filter: LLM domains empty, falling back to keyword match on %.120s",
        composite_text,
    )
    return get_filtered_tools(all_tools, composite_text, agent_role=agent_role)


def _has_query_image_context(messages) -> bool:
    return any(
        bool(getattr(message, "additional_kwargs", {}).get("rag_query_image_context"))
        for message in messages
    )


def _messages_without_query_image_context(messages):
    return [
        message
        for message in messages
        if not bool(getattr(message, "additional_kwargs", {}).get("rag_query_image_context"))
    ]


async def enrich_query_images(state: AgentState, config: RunnableConfig) -> dict:
    """Convert user-question images into temporary text context before research planning."""
    configurable = Configuration.from_runnable_config(config)
    if (
        not configurable.rag_query_image_enabled
        or not configurable.rag_multimodal_enabled
    ):
        return {}

    messages = state.get("messages", [])
    if _has_query_image_context(messages):
        return {}

    context = await asyncio.to_thread(
        query_images.build_query_image_context,
        messages,
        provider=configurable.rag_multimodal_provider,
        ocr_languages=configurable.rag_ocr_languages,
        vision_enabled=configurable.rag_vision_enabled,
        vision_model=configurable.rag_vision_model,
        vision_prompt=configurable.rag_vision_prompt,
        vision_max_tokens=configurable.rag_vision_max_tokens,
        max_images=configurable.rag_query_image_max_images,
        max_bytes=configurable.rag_query_image_max_bytes,
    )
    if not context.strip():
        return {}

    return {
        "messages": [
            HumanMessage(
                content=(
                    "Recognized image context from the user's question. "
                    "Use it as temporary query context; it is not persisted as local "
                    "knowledge or memory.\n\n"
                    f"{context.strip()}"
                ),
                additional_kwargs={"rag_query_image_context": True},
            )
        ]
    }


def budget_skip_tool_message(tool_call: dict, reason: str) -> ToolMessage:
    """Create a synthetic tool result when Budget Guard skips a tool call."""
    return ToolMessage(
        content=f"Budget guard skipped `{tool_call['name']}`: {reason}",
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    budget_usage = state.get("budget_usage", {})
    if not can_spend_model_call(
        configurable,
        budget_usage,
        reserve_final_report_call=True,
    ):
        return Command(
            goto="write_research_brief",
            update={
                "budget_usage": budget_usage_with_reason(
                    "Skipped clarification to preserve the final report model call."
                )
            },
        )
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    budget_update = budget_from_model_response(response)
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={
                "messages": [AIMessage(content=response.question)],
                "budget_usage": budget_update,
            }
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={
                "messages": [AIMessage(content=response.verification)],
                "budget_usage": budget_update,
            }
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    budget_usage = state.get("budget_usage", {})
    if not can_spend_model_call(
        configurable,
        budget_usage,
        reserve_final_report_call=True,
    ):
        research_brief = get_buffer_string(state.get("messages", []))
        supervisor_system_prompt = _supervisor_system_prompt(configurable)
        return Command(
            goto="research_supervisor",
            update={
                "research_brief": research_brief,
                "relevant_domains": [],  # budget guard — fall back to keyword match
                "supervisor_messages": {
                    "type": "override",
                    "value": [
                        SystemMessage(content=supervisor_system_prompt),
                        HumanMessage(content=research_brief)
                    ]
                },
                "budget_usage": budget_usage_with_reason(
                    "Skipped research brief generation to preserve the final report model call."
                ),
            },
        )

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages.
    # The domain classifier section is injected dynamically from the
    # domain registry — no hardcoded domain list in the prompt template.
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
        domain_classifier_section=build_domain_classifier_prompt(),
    )
    if _public_opinion_mode(configurable):
        prompt_content += (
            "\n\nBusiness scenario: Treat this request as enterprise public-opinion "
            "and brand-risk monitoring. Preserve the target organization, product, "
            "issue, geography, monitoring window, suspected claims, stakeholders, "
            "and any internal-knowledge requirements. If a detail is unspecified, "
            "state that it is unspecified rather than inventing it."
        )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    budget_update = budget_from_model_response(response)

    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = _supervisor_system_prompt(configurable)

    return Command(
        goto="plan_report_sections",
        update={
            "research_brief": response.research_brief,
            "relevant_domains": response.relevant_domains,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            },
            "budget_usage": budget_update,
        }
    )


async def plan_report_sections(state: AgentState, config: RunnableConfig) -> Command[Literal["plan_report_sections", "research_supervisor"]]:
    """Plan report sections for public-opinion mode using Plan-and-Execute pattern.
    
    In non-public-opinion mode, this is a passthrough to research_supervisor.
    In public-opinion mode, it generates structured Sections from the research brief,
    with optional human feedback via LangGraph interrupt().
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Passthrough for non-public-opinion mode
    if not _public_opinion_mode(configurable):
        return Command(goto="research_supervisor")
    
    budget_usage = state.get("budget_usage", {})
    
    # Budget guard: degrade to single section if we can't reserve final report call
    if not can_spend_model_call(configurable, budget_usage, reserve_final_report_call=True):
        budget_update = budget_usage_with_reason(
            "Skipped section planning due to budget constraints; falling back to single section."
        )
        single_section = Section(
            name="Research Report",
            description=state.get("research_brief", ""),
            research=True,
            agent_role="public_signal,internal_knowledge,risk_assessment,response_strategy",
            status="pending",
        )
        return Command(
            goto="research_supervisor",
            update={
                "sections": [single_section],
                "budget_usage": budget_update,
            },
        )
    
    # Plan sections using planner model
    planner_model_name = configurable.planner_model or configurable.research_model
    planner_max_tokens = configurable.planner_model_max_tokens
    
    planner_model_config: dict[str, Any] = {
        "model": planner_model_name,
        "api_key": get_api_key_for_model(planner_model_name, config),
        "tags": ["langsmith:nostream"],
    }
    if planner_max_tokens is not None:
        planner_model_config["max_tokens"] = planner_max_tokens
    
    feedback_text = "\n///\n".join(state.get("feedback_on_report_plan", [])) or "No feedback yet."
    
    prompt = report_planner_instructions.format(
        topic=state.get("research_brief", ""),
        report_organization=configurable.report_structure,
        feedback=feedback_text,
        date=get_today_str(),
    )
    
    planner = (
        configurable_model
        .with_structured_output(Sections)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(planner_model_config)
    )
    
    try:
        response = await planner.ainvoke([HumanMessage(content=prompt)])
        budget_update = budget_from_model_response(response)
    except Exception as exc:
        LOGGER.warning("Section planning failed: %s. Falling back to single section.", exc)
        budget_update = budget_usage_with_reason(
            f"Section planning failed: {exc}. Falling back to single section."
        )
        single_section = Section(
            name="Research Report",
            description=state.get("research_brief", ""),
            research=True,
            agent_role="public_signal,internal_knowledge,risk_assessment,response_strategy",
            status="pending",
        )
        return Command(
            goto="research_supervisor",
            update={
                "sections": [single_section],
                "budget_usage": budget_update,
            },
        )
    
    # Budget-aware section count limit with section_writer reservation
    available_slots = available_research_unit_slots(configurable, budget_usage)
    max_sections = available_slots if available_slots is not None else 5
    
    all_sections = list(response.sections)
    research_section_count = sum(1 for s in all_sections if s.research)
    slots_after_reservation = max_sections - research_section_count
    if slots_after_reservation <= 0:
        LOGGER.warning(
            "Not enough budget slots for section_writer after planning %d research sections. "
            "Falling back to single section.",
            research_section_count,
        )
        budget_update = merge_budget_usage(
            budget_update,
            budget_usage_with_reason(
                "Insufficient budget slots for section_writer; falling back to single section."
            ),
        )
        single_section = Section(
            name="Research Report",
            description=state.get("research_brief", ""),
            research=True,
            agent_role="public_signal,internal_knowledge,risk_assessment,response_strategy",
            status="pending",
        )
        return Command(
            goto="research_supervisor",
            update={
                "sections": [single_section],
                "budget_usage": budget_update,
            },
        )
    
    sections = all_sections[:max_sections]
    
    total_budget_update = merge_budget_usage(
        budget_update,
        budget_usage_with_reason(f"Planned {len(sections)} report sections (public-opinion mode)."),
    )
    
    # Human feedback via interrupt when enabled
    if configurable.allow_plan_feedback:
        feedback = interrupt({"sections": [s.model_dump() for s in sections]})
        
        if feedback is True:
            return Command(
                goto="research_supervisor",
                update={
                    "sections": sections,
                    "budget_usage": total_budget_update,
                },
            )
        elif isinstance(feedback, str):
            return Command(
                goto="plan_report_sections",
                update={
                    "feedback_on_report_plan": [feedback],
                    "budget_usage": total_budget_update,
                },
            )
        else:
            raise TypeError(
                f"Expected resume value to be True (bool) or str, got {type(feedback).__name__}. "
                f"Value: {feedback!r}"
            )
    
    return Command(
        goto="research_supervisor",
        update={
            "sections": sections,
            "budget_usage": total_budget_update,
        },
    )


# Public-opinion role names for validation
_PUBLIC_OPINION_ROLE_NAMES = frozenset({
    "public_signal", "internal_knowledge", "risk_assessment", "response_strategy",
})


def _extract_section_evidence(section, role_report_content: dict[str, str]) -> str:
    """Extract role evidence for a section based on its agent_role field.
    
    Parses comma-separated agent_role, validates against known roles,
    and extracts corresponding role reports.
    """
    agent_role_str = (section.agent_role or "").strip()
    if not agent_role_str:
        # No agent_role specified, use all available role reports
        return "\n\n".join(
            f"## {role}\n{content}"
            for role, content in role_report_content.items()
            if content
        ) or "No role evidence is available."
    
    role_names = [r.strip() for r in agent_role_str.split(",")]
    evidence_parts = []
    valid_roles_found = False
    
    for role_name in role_names:
        if not role_name:
            continue
        if role_name not in _PUBLIC_OPINION_ROLE_NAMES:
            LOGGER.warning("Invalid agent_role '%s' in section '%s'; ignoring.", role_name, section.name)
            continue
        content = role_report_content.get(role_name, "")
        if content:
            evidence_parts.append(f"## {role_name}\n{content}")
            valid_roles_found = True
        else:
            evidence_parts.append(f"## {role_name}\n(No evidence was collected by this role.)")
            valid_roles_found = True
    
    if not valid_roles_found:
        # Fallback: use all available role reports
        return "\n\n".join(
            f"## {role}\n{content}"
            for role, content in role_report_content.items()
            if content
        ) or "No role evidence is available."
    
    return "\n\n".join(evidence_parts)


async def section_writer(state: AgentState, config: RunnableConfig) -> dict:
    """Write report sections from role evidence in public-opinion mode.
    
    For each research=True section, extract evidence from role_reports based on 
    the section's agent_role field, then write the section content in parallel.
    """
    configurable = Configuration.from_runnable_config(config)
    sections = state.get("sections", [])
    role_reports = state.get("agent_memories", {})
    
    # Find role reports from public opinion subgraph output
    role_report_content: dict[str, str] = {}
    for role in _PUBLIC_OPINION_ROLE_NAMES:
        memories = list(role_reports.get(role, []) or [])
        for mem in memories:
            if isinstance(mem, dict) and mem.get("source") == "current_public_opinion_run":
                role_report_content[role] = str(mem.get("content") or "")
                break
    
    research_sections = [s for s in sections if s.research and s.status != "done"]
    if not research_sections:
        return {}
    
    # Budget guard: if no budget, copy role reports directly as content
    budget_usage = state.get("budget_usage", {})
    if not can_spend_model_call(configurable, budget_usage, reserve_final_report_call=True):
        budget_update = budget_usage_with_reason(
            "Skipped section_writer model calls due to budget constraints; using role reports as section content."
        )
        completed = []
        for section in research_sections:
            evidence = _extract_section_evidence(section, role_report_content)
            section.content = evidence
            section.status = "done"
            completed.append(section)
        return {
            "completed_sections": completed,
            "budget_usage": budget_update,
        }
    
    # Write sections in parallel using section_writer_model
    writer_model_name = configurable.section_writer_model or configurable.final_report_model
    writer_max_tokens = configurable.section_writer_model_max_tokens
    
    writer_model_config: dict[str, Any] = {
        "model": writer_model_name,
        "api_key": get_api_key_for_model(writer_model_name, config),
        "tags": ["langsmith:nostream"],
    }
    if writer_max_tokens is not None:
        writer_model_config["max_tokens"] = writer_max_tokens
    
    async def _write_one(section: Section) -> Section:
        try:
            evidence = _extract_section_evidence(section, role_report_content)
            prompt = section_writer_from_role_reports_prompt.format(
                section_name=section.name,
                section_description=section.description,
                evidence=evidence,
            )
            writer = configurable_model.with_config(writer_model_config)
            response = await writer.ainvoke([HumanMessage(content=prompt)])
            section.content = str(response.content)
            section.status = "done"
        except Exception as exc:
            LOGGER.warning("Failed to write section '%s': %s", section.name, exc)
            section.content = ""
        return section
    
    results = await asyncio.gather(*[_write_one(s) for s in research_sections], return_exceptions=True)
    
    completed = []
    for result in results:
        if isinstance(result, Exception):
            LOGGER.warning("Section writer task raised exception: %s", result)
            continue
        if isinstance(result, Section):
            completed.append(result)
    
    budget_update = budget_usage_with_reason(f"Wrote {len(completed)} sections via section_writer.")
    return {
        "completed_sections": completed,
        "budget_usage": budget_update,
    }


async def write_final_sections(state: AgentState, config: RunnableConfig) -> dict:
    """Write non-research sections (intro, conclusion) in parallel.
    
    Uses completed research sections as context for writing intro/conclusion.
    """
    configurable = Configuration.from_runnable_config(config)
    sections = state.get("sections", [])
    completed_sections = state.get("completed_sections", [])
    
    final_sections = [s for s in sections if not s.research]
    if not final_sections:
        return {}
    
    # Build context from completed research sections (deduped)
    completed_map: dict[str, str] = {}
    for cs in completed_sections:
        if cs.name and cs.content:
            completed_map[cs.name] = cs.content
    
    context = "\n\n".join(
        f"## {name}\n{content}"
        for name, content in completed_map.items()
    )
    
    if not context:
        context = "No completed research sections are available yet."
    
    budget_usage = state.get("budget_usage", {})
    
    # Budget guard
    if not can_spend_model_call(configurable, budget_usage, reserve_final_report_call=True):
        budget_update = budget_usage_with_reason(
            "Skipped final section writing due to budget constraints."
        )
        return {"budget_usage": budget_update}
    
    writer_model_config: dict[str, Any] = {
        "model": configurable.final_report_model,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }
    
    async def _write_one(section: Section) -> Section:
        try:
            prompt = final_section_writer_instructions.format(
                section_name=section.name,
                section_description=section.description,
                context=context,
            )
            writer = configurable_model.with_config(writer_model_config)
            response = await writer.ainvoke([HumanMessage(content=prompt)])
            section.content = str(response.content)
            section.status = "done"
        except Exception as exc:
            LOGGER.warning("Failed to write final section '%s': %s", section.name, exc)
            section.content = ""
        return section
    
    results = await asyncio.gather(*[_write_one(s) for s in final_sections], return_exceptions=True)
    
    completed = []
    for result in results:
        if isinstance(result, Exception):
            LOGGER.warning("Final section writer task raised exception: %s", result)
            continue
        if isinstance(result, Section):
            completed.append(result)
    
    budget_update = budget_usage_with_reason(f"Wrote {len(completed)} final sections.")
    return {
        "completed_sections": completed,
        "budget_usage": budget_update,
    }


async def compile_final_report(state: AgentState, config: RunnableConfig):
    """Compile the final report by assembling sections in planned order.
    
    In public-opinion mode, assembles sections in their original planned order.
    In general mode, delegates to final_report_generation.
    Falls back to notes-based synthesis when all sections are missing.
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Delegate to final_report_generation for non-public-opinion mode
    if not _public_opinion_mode(configurable):
        return await final_report_generation(state, config)
    
    sections = state.get("sections", [])
    completed_sections = state.get("completed_sections", [])
    budget_usage = state.get("budget_usage", {})
    cleared_state = {"notes": {"type": "override", "value": []}}
    
    # Build deduped content map: take the latest content for each section name
    content_map: dict[str, str] = {}
    for cs in completed_sections:
        if cs.name and cs.content:
            content_map[cs.name] = cs.content
    
    # Assemble in planned order
    parts = []
    missing_names = []
    for section in sections:
        content = content_map.get(section.name, "")
        if content:
            parts.append(content)
        else:
            missing_names.append(section.name)
    
    if parts:
        report = "\n\n".join(parts)
        
        # Handle missing sections
        if missing_names:
            if can_spend_model_call(configurable, budget_usage):
                # Try to fill missing sections with final_report_model
                try:
                    notes = state.get("notes", [])
                    findings = "\n".join(notes)
                    
                    final_report_prompt = public_opinion_final_report_generation_prompt.format(
                        research_brief=state.get("research_brief", ""),
                        organization_context=_business_context(configurable),
                        messages=get_buffer_string(state.get("messages", [])),
                        findings=f"已有报告内容:\n{report}\n\n补充证据:\n{findings}",
                        date=get_today_str(),
                    )
                    
                    writer_config: dict[str, Any] = {
                        "model": configurable.final_report_model,
                        "api_key": get_api_key_for_model(configurable.final_report_model, config),
                        "tags": ["langsmith:nostream"],
                    }
                    fill_response = await configurable_model.with_config(writer_config).ainvoke([
                        HumanMessage(content=final_report_prompt)
                    ])
                    budget_update = merge_budget_usage(
                        budget_from_model_response(fill_response),
                        budget_usage_with_reason(f"Filled {len(missing_names)} missing sections via final_report_model."),
                    )
                    final_usage = merge_budget_usage(budget_usage, budget_update)
                    report = append_budget_summary(
                        str(fill_response.content),
                        configurable,
                        final_usage,
                    )
                    await maybe_persist_chat_memory(state, config, report)
                    return {
                        "final_report": report,
                        "messages": [AIMessage(content=report)],
                        "budget_usage": budget_update,
                        **cleared_state,
                    }
                except Exception as exc:
                    LOGGER.warning("Failed to fill missing sections: %s", exc)
                    report += f"\n\n> Note: 以下 section 因预算限制未完成: {', '.join(missing_names)}"
            else:
                report += f"\n\n> Note: 以下 section 因预算限制未完成: {', '.join(missing_names)}"
        
        await maybe_persist_chat_memory(state, config, report)
        return {
            "final_report": report,
            "messages": [AIMessage(content=report)],
            "budget_usage": {},
            **cleared_state,
        }
    
    # All sections missing: full degradation to notes-based synthesis
    LOGGER.warning("No sections completed; falling back to notes-based report generation.")
    return await final_report_generation(state, config)


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools", "__end__"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    budget_usage = state.get("budget_usage", {})
    if not can_spend_model_call(
        configurable,
        budget_usage,
        reserve_final_report_call=True,
    ):
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(state.get("supervisor_messages", [])),
                "research_brief": state.get("research_brief", ""),
                "budget_usage": budget_usage_with_reason(
                    "Stopped supervisor planning to preserve the final report model call."
                ),
            },
        )

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    budget_update = budget_from_model_response(response)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            "budget_usage": budget_update,
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    budget_usage = state.get("budget_usage", {})
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    if is_over_budget(configurable, budget_usage):
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "budget_usage": budget_usage_with_reason(
                    "Stopped supervisor tool execution because a configured budget was reached."
                ),
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    supervisor_budget_update = {}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    allowed_think_tool_calls, skipped_think_tool_calls = filter_tool_calls_for_budget(
        configurable,
        budget_usage,
        think_tool_calls,
    )
    
    for tool_call in allowed_think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    if allowed_think_tool_calls:
        supervisor_budget_update = merge_budget_usage(
            supervisor_budget_update,
            budget_from_tool_calls(allowed_think_tool_calls),
        )

    for tool_call in skipped_think_tool_calls:
        all_tool_messages.append(
            budget_skip_tool_message(tool_call, "tool call budget exhausted")
        )

    if skipped_think_tool_calls:
        supervisor_budget_update = merge_budget_usage(
            supervisor_budget_update,
            budget_usage_with_reason("Skipped supervisor reflection because the tool call budget was exhausted."),
        )
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    projected_usage = merge_budget_usage(budget_usage, supervisor_budget_update)
    allowed_conduct_research_calls, skipped_conduct_research_calls = filter_tool_calls_for_budget(
        configurable,
        projected_usage,
        conduct_research_calls,
    )

    for tool_call in skipped_conduct_research_calls:
        all_tool_messages.append(
            budget_skip_tool_message(tool_call, "tool call budget exhausted")
        )

    if skipped_conduct_research_calls:
        supervisor_budget_update = merge_budget_usage(
            supervisor_budget_update,
            budget_usage_with_reason("Skipped research delegation because the tool call budget was exhausted."),
        )

    projected_usage = merge_budget_usage(budget_usage, supervisor_budget_update)
    research_unit_slots = available_research_unit_slots(configurable, projected_usage)
    model_budget_skipped_calls = []
    if research_unit_slots is not None:
        model_budget_skipped_calls = allowed_conduct_research_calls[research_unit_slots:]
        allowed_conduct_research_calls = allowed_conduct_research_calls[:research_unit_slots]

    for tool_call in model_budget_skipped_calls:
        all_tool_messages.append(
            budget_skip_tool_message(tool_call, "model call budget reserved for final report")
        )

    if model_budget_skipped_calls:
        supervisor_budget_update = merge_budget_usage(
            supervisor_budget_update,
            budget_usage_with_reason("Stopped new research units to preserve the final report model call."),
        )

    # Limit concurrent research units to prevent resource exhaustion
    overflow_conduct_research_calls = allowed_conduct_research_calls[configurable.max_concurrent_research_units:]
    allowed_conduct_research_calls = allowed_conduct_research_calls[:configurable.max_concurrent_research_units]

    for overflow_call in overflow_conduct_research_calls:
        all_tool_messages.append(ToolMessage(
            content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
            name="ConductResearch",
            tool_call_id=overflow_call["id"]
        ))

    if allowed_conduct_research_calls:
        supervisor_budget_update = merge_budget_usage(
            supervisor_budget_update,
            budget_from_tool_calls(allowed_conduct_research_calls),
        )
    
    if conduct_research_calls and not allowed_conduct_research_calls:
        combined_supervisor_messages = supervisor_messages + all_tool_messages
        return Command(
            goto=END,
            update={
                "supervisor_messages": all_tool_messages,
                "notes": get_notes_from_tool_calls(combined_supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "budget_usage": supervisor_budget_update,
            },
        )

    if allowed_conduct_research_calls:
        try:
            projected_usage = merge_budget_usage(budget_usage, supervisor_budget_update)
            assignments = [
                _researcher_assignment(tool_call, configurable)
                for tool_call in allowed_conduct_research_calls
            ]
            
            # Execute research tasks in parallel
            llm_domains = state.get("relevant_domains", [])
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=assignment["message"])
                    ],
                    "research_topic": assignment["research_topic"],
                    "agent_role": assignment["agent_role"],
                    "expected_output": assignment["expected_output"],
                    "relevant_domains": llm_domains,
                    "budget_usage": projected_usage,
                }, config) 
                for assignment in assignments
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call, assignment in zip(
                tool_results,
                allowed_conduct_research_calls,
                assignments,
            ):
                observation_content = observation.get(
                    "compressed_research",
                    "Error synthesizing research report: Maximum retries exceeded",
                )
                if _public_opinion_mode(configurable):
                    observation_content = (
                        f"Agent role: {assignment['agent_role']}\n"
                        f"Expected output: {assignment['expected_output']}\n\n"
                        f"{observation_content}"
                    )
                all_tool_messages.append(ToolMessage(
                    content=observation_content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            subgraph_budget_update = {}
            for observation in tool_results:
                subgraph_budget_update = merge_budget_usage(
                    subgraph_budget_update,
                    diff_budget_usage(
                        observation.get("budget_usage", {}),
                        projected_usage,
                    ),
                )
            supervisor_budget_update = merge_budget_usage(
                supervisor_budget_update,
                subgraph_budget_update,
            )
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                        "budget_usage": merge_budget_usage(
                            supervisor_budget_update,
                            budget_usage_with_reason("Stopped research after an error while running delegated researchers."),
                        ),
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    if supervisor_budget_update:
        update_payload["budget_usage"] = supervisor_budget_update
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    budget_usage = state.get("budget_usage", {})
    if not can_spend_model_call(
        configurable,
        budget_usage,
        reserve_final_report_call=True,
    ):
        return Command(
            goto="compress_research",
            update={
                "budget_usage": budget_usage_with_reason(
                    "Stopped researcher reasoning to preserve the final report model call."
                )
            },
        )
    
    # Get tools filtered by domain relevance to the research topic.
    # Pass researcher_messages content as extra_context so that the
    # supervisor's full assignment (which may contain keywords the
    # topic alone misses) contributes to domain detection.
    agent_role = state.get("agent_role", "general_research")
    research_topic = state.get("research_topic", "")
    tools = await _get_filtered_researcher_tools(
        config, research_topic, agent_role=agent_role,
        extra_context=get_buffer_string(state.get("researcher_messages", [])),
        llm_domains=state.get("relevant_domains", []),
    )
    if not has_external_research_tool(tools):
        raise ValueError(
            "No external research tools found to conduct research: Please configure "
            "web search, local RAG knowledge base paths, or MCP tools."
        )

    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # Build a dynamic tool prompt based on the ACTUAL tools bound,
    # not a static template.  This ensures the LLM sees an accurate
    # list of available tools grouped by domain.
    dynamic_tool_prompt = build_dynamic_tool_prompt(
        tools, configurable,
        active_domains=set(state.get("relevant_domains", [])),
    )
    # Still include user-supplied MCP prompt text if provided
    user_mcp_prompt = configurable.mcp_prompt or ""

    # Prepare system prompt with MCP context if available
    if _public_opinion_mode(configurable):
        if agent_role in PUBLIC_OPINION_ROLE_EXPECTATIONS:
            researcher_prompt = get_public_opinion_agent_spec(agent_role).format_system_prompt(
                retrieval_tool_prompt=dynamic_tool_prompt,
                mcp_prompt=user_mcp_prompt,
                date=get_today_str(),
                organization_context=_business_context(configurable),
            )
        else:
            researcher_prompt = public_opinion_researcher_prompt.format(
                retrieval_tool_prompt=dynamic_tool_prompt,
                mcp_prompt=user_mcp_prompt,
                date=get_today_str(),
                agent_role=agent_role,
                expected_output=state.get("expected_output", ""),
                organization_context=_business_context(configurable),
            )
    else:
        researcher_prompt = research_system_prompt.format(
            retrieval_tool_prompt=dynamic_tool_prompt,
            mcp_prompt=user_mcp_prompt,
            date=get_today_str()
        )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    budget_update = budget_from_model_response(response)
    if openai_websearch_called(response) or anthropic_websearch_called(response):
        budget_update = merge_budget_usage(budget_update, budget_from_native_search())
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
            "budget_usage": budget_update,
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    capture_token = start_budget_capture()
    try:
        observation = await tool.ainvoke(args, config)
        captured_budget = stop_budget_capture(capture_token)
        return observation, captured_budget
    except Exception as e:
        captured_budget = stop_budget_capture(capture_token)
        return f"Error executing tool: {str(e)}", captured_budget


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    if is_over_budget(configurable, state.get("budget_usage", {})):
        return Command(
            goto="compress_research",
            update={
                "budget_usage": budget_usage_with_reason(
                    "Stopped researcher tool execution because a configured budget was reached."
                )
            },
        )
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    # Use the same domain-filtered tool set as researcher() for consistency
    tools = await _get_filtered_researcher_tools(
        config,
        state.get("research_topic", ""),
        agent_role=state.get("agent_role", "general_research"),
        extra_context=get_buffer_string(state.get("researcher_messages", [])),
        llm_domains=state.get("relevant_domains", []),
    )
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }
    
    tool_calls = most_recent_message.tool_calls
    allowed_tool_calls, skipped_tool_calls = filter_tool_calls_for_budget(
        configurable,
        state.get("budget_usage", {}),
        tool_calls,
        tools_by_name,
    )

    skipped_tool_outputs = [
        budget_skip_tool_message(tool_call, "tool or search call budget exhausted")
        for tool_call in skipped_tool_calls
    ]
    budget_update = {}
    if skipped_tool_calls:
        budget_update = merge_budget_usage(
            budget_update,
            budget_usage_with_reason("Skipped researcher tools because the tool or search budget was exhausted."),
        )

    if not allowed_tool_calls:
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": skipped_tool_outputs,
                "budget_usage": budget_update,
            },
        )

    # Execute allowed tool calls in parallel
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in allowed_tool_calls
    ]
    tool_results = await asyncio.gather(*tool_execution_tasks)
    observations = [observation for observation, _ in tool_results]
    budget_update = merge_budget_usage(
        budget_update,
        budget_from_tool_calls(allowed_tool_calls, tools_by_name),
    )
    for _, captured_budget in tool_results:
        budget_update = merge_budget_usage(budget_update, captured_budget)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, allowed_tool_calls)
    ] + skipped_tool_outputs
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    budget_limit_reached = is_over_budget(
        configurable,
        merge_budget_usage(state.get("budget_usage", {}), budget_update),
    )
    
    if budget_limit_reached:
        budget_update = merge_budget_usage(
            budget_update,
            budget_usage_with_reason("Stopped researcher loop because a configured budget was reached after tool execution."),
        )

    if exceeded_iterations or research_complete_called or budget_limit_reached:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "budget_usage": budget_update,
            }
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "budget_usage": budget_update,
        }
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    budget_usage = state.get("budget_usage", {})
    if not can_spend_model_call(
        configurable,
        budget_usage,
        reserve_final_report_call=True,
    ):
        researcher_messages = state.get("researcher_messages", [])
        raw_notes_content = "\n".join([
            str(message.content) 
            for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
        ])
        return {
            "compressed_research": raw_notes_content or "Budget guard skipped research compression before any research notes were collected.",
            "raw_notes": [raw_notes_content],
            "budget_usage": budget_usage_with_reason(
                "Skipped research compression to preserve the final report model call."
            ),
        }

    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            budget_update = budget_from_model_response(response)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
                "budget_usage": budget_update,
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
        "budget_usage": budget_usage_with_reason(
            "Research compression failed after maximum retries."
        ),
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


async def _run_public_opinion_agent(
    state: PublicOpinionState,
    config: RunnableConfig,
    role: str,
) -> dict:
    """Run one explicit public-opinion business agent with a role-specific toolset."""
    configurable = Configuration.from_runnable_config(config)
    if not _role_enabled(configurable, role):
        return {}

    agent_spec = get_public_opinion_agent_spec(role)
    budget_usage = state.get("budget_usage", {})
    role_budget_update = {}
    expected_output = agent_spec.expected_output
    upstream_context = _role_context(state.get("role_reports", {}))
    private_memory_context = _agent_private_memory_context(
        state.get("agent_memories", {}),
        role,
    )
    assignment = (
        f"Overall research brief:\n{state.get('research_brief', '')}\n\n"
        f"Upstream role reports:\n{upstream_context}\n\n"
        f"Your private memory:\n{private_memory_context}\n\n"
        f"Input contract:\n{chr(10).join(f'- {item}' for item in agent_spec.input_contract)}\n\n"
        f"Your role-specific objective:\n{expected_output}"
    )

    tools = await _business_agent_tools(config, role)
    if not has_external_research_tool(tools):
        spec = get_public_opinion_agent_spec(role)
        required = [ch for ch in spec.tool_channels if ch not in ("mcp",)]
        raise ValueError(
            f"Public Opinion agent '{role}' ({spec.display_name}) requires "
            f"tool channels: {', '.join(sorted(required))}. "
            f"Missing tools — ensure RAG is enabled (rag_enabled=true) "
            f"and/or web search is configured (search_api=tavily)."
        )

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    agent_prompt = agent_spec.format_system_prompt(
        retrieval_tool_prompt=_role_tool_prompt(configurable, role),
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str(),
        organization_context=_business_context(configurable),
        private_memory_context=private_memory_context,
    )
    agent_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    role_messages = [HumanMessage(content=assignment)]
    tools_by_name = {_tool_name(available_tool): available_tool for available_tool in tools}

    for _ in range(configurable.max_react_tool_calls):
        projected_budget = merge_budget_usage(budget_usage, role_budget_update)
        if not can_spend_model_call(
            configurable,
            projected_budget,
            reserve_final_report_call=True,
        ):
            role_budget_update = merge_budget_usage(
                role_budget_update,
                budget_usage_with_reason(
                    f"Stopped {role} agent reasoning to preserve the final report model call."
                ),
            )
            break

        response = await agent_model.ainvoke(
            [SystemMessage(content=agent_prompt)] + role_messages
        )
        role_messages.append(response)
        response_budget = budget_from_model_response(response)
        if openai_websearch_called(response) or anthropic_websearch_called(response):
            response_budget = merge_budget_usage(response_budget, budget_from_native_search())
        role_budget_update = merge_budget_usage(role_budget_update, response_budget)

        has_tool_calls = bool(response.tool_calls)
        has_native_search = openai_websearch_called(response) or anthropic_websearch_called(response)
        if not has_tool_calls and not has_native_search:
            break

        if is_over_budget(configurable, merge_budget_usage(budget_usage, role_budget_update)):
            role_budget_update = merge_budget_usage(
                role_budget_update,
                budget_usage_with_reason(f"Stopped {role} agent because a budget was reached."),
            )
            break

        allowed_tool_calls, skipped_tool_calls = filter_tool_calls_for_budget(
            configurable,
            merge_budget_usage(budget_usage, role_budget_update),
            response.tool_calls,
            tools_by_name,
        )
        skipped_tool_outputs = [
            budget_skip_tool_message(tool_call, "tool or search call budget exhausted")
            for tool_call in skipped_tool_calls
        ]
        if skipped_tool_calls:
            role_budget_update = merge_budget_usage(
                role_budget_update,
                budget_usage_with_reason(
                    f"Skipped {role} agent tools because a tool or search budget was exhausted."
                ),
            )

        if allowed_tool_calls:
            # Split into known vs unknown tools so a missing tool doesn't crash
            known_calls: list[dict[str, Any]] = []
            unknown_calls: list[dict[str, Any]] = []
            for tool_call in allowed_tool_calls:
                if tool_call["name"] in tools_by_name:
                    known_calls.append(tool_call)
                else:
                    unknown_calls.append(tool_call)

            for tool_call in unknown_calls:
                role_messages.append(ToolMessage(
                    content=(
                        f"Tool '{tool_call['name']}' is not available in the current "
                        "configuration. Use one of the available tools instead."
                    ),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                ))

            if known_calls:
                tool_execution_tasks = [
                    execute_tool_safely(
                        tools_by_name[tool_call["name"]],
                        tool_call["args"],
                        config,
                    )
                    for tool_call in known_calls
                ]
                tool_results = await asyncio.gather(*tool_execution_tasks)
                observations = [observation for observation, _ in tool_results]
                role_budget_update = merge_budget_usage(
                    role_budget_update,
                    budget_from_tool_calls(known_calls, tools_by_name),
                )
                for _, captured_budget in tool_results:
                    role_budget_update = merge_budget_usage(role_budget_update, captured_budget)

                role_messages.extend([
                    ToolMessage(
                    content=observation,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                for observation, tool_call in zip(observations, known_calls)
            ])

        role_messages.extend(skipped_tool_outputs)

        if any(tool_call["name"] == "ResearchComplete" for tool_call in response.tool_calls):
            break

        if not allowed_tool_calls:
            break

    compression_state = {
        "researcher_messages": role_messages,
        "research_topic": assignment,
        "agent_role": role,
        "expected_output": expected_output,
        "budget_usage": merge_budget_usage(budget_usage, role_budget_update),
    }
    compressed = await compress_research(compression_state, config)
    role_budget_update = merge_budget_usage(
        role_budget_update,
        compressed.get("budget_usage", {}),
    )
    report = (
        f"Agent role: {role}\n"
        f"Agent name: {agent_spec.display_name}\n"
        f"Expected output: {expected_output}\n\n"
        f"{compressed.get('compressed_research', '')}"
    )
    private_memory = _build_agent_private_memory(
        role,
        report,
        compressed.get("raw_notes", []),
    )
    return {
        "role_reports": {role: report},
        "agent_memories": {role: [private_memory]},
        "notes": [report],
        "raw_notes": compressed.get("raw_notes", []),
        "budget_usage": role_budget_update,
    }


async def public_signal_agent(state: PublicOpinionState, config: RunnableConfig) -> dict:
    """Collect integrated news, social, complaint, competitor, and spread evidence."""
    return await _run_public_opinion_agent(state, config, "public_signal")


async def internal_knowledge_agent(state: PublicOpinionState, config: RunnableConfig) -> dict:
    """Collect internal RAG evidence from company knowledge, playbooks, and memory."""
    return await _run_public_opinion_agent(state, config, "internal_knowledge")


async def risk_assessment_agent(state: PublicOpinionState, config: RunnableConfig) -> dict:
    """Verify claims and assess compliance, legal, and product-risk signals."""
    return await _run_public_opinion_agent(state, config, "risk_assessment")


async def response_strategy_agent(state: PublicOpinionState, config: RunnableConfig) -> dict:
    """Create PR response posture, FAQ points, actions, and monitoring keywords."""
    return await _run_public_opinion_agent(state, config, "response_strategy")


public_opinion_builder = StateGraph(PublicOpinionState, config_schema=Configuration)
public_opinion_builder.add_node("public_signal_agent", public_signal_agent)
public_opinion_builder.add_node("internal_knowledge_agent", internal_knowledge_agent)
public_opinion_builder.add_node("risk_assessment_agent", risk_assessment_agent)
public_opinion_builder.add_node("response_strategy_agent", response_strategy_agent)

public_opinion_builder.add_edge(START, "public_signal_agent")
public_opinion_builder.add_edge(START, "internal_knowledge_agent")
public_opinion_builder.add_edge(
    [
        "public_signal_agent",
        "internal_knowledge_agent",
    ],
    "risk_assessment_agent",
)
public_opinion_builder.add_edge("risk_assessment_agent", "response_strategy_agent")
public_opinion_builder.add_edge("response_strategy_agent", END)

public_opinion_subgraph = public_opinion_builder.compile()


async def research_phase(state: AgentState, config: RunnableConfig) -> dict:
    """Run the configured research phase, using explicit business agents when enabled."""
    configurable = Configuration.from_runnable_config(config)
    if not _public_opinion_mode(configurable):
        return await supervisor_subgraph.ainvoke(state, config)

    input_budget = state.get("budget_usage", {})
    result = await public_opinion_subgraph.ainvoke(
        {
            "messages": state.get("messages", []),
            "research_brief": state.get("research_brief", ""),
            "relevant_domains": state.get("relevant_domains", []),
            "role_reports": {},
            "agent_memories": state.get("agent_memories", {}),
            "notes": [],
            "raw_notes": [],
            "budget_usage": input_budget,
        },
        config,
    )
    budget_update = diff_budget_usage(result.get("budget_usage", {}), input_budget)
    
    # Build state after public_opinion_subgraph for section_writer consumption
    post_opinion_state = {
        **state,
        "sections": state.get("sections", []),
        "completed_sections": [],
        "agent_memories": result.get("agent_memories", state.get("agent_memories", {})),
        "notes": result.get("notes", []),
        "raw_notes": result.get("raw_notes", []),
        "budget_usage": merge_budget_usage(input_budget, budget_update),
    }
    
    # Run section_writer to convert role evidence into section content
    writer_result = await section_writer(post_opinion_state, config)
    
    total_budget_update = merge_budget_usage(
        budget_update,
        writer_result.get("budget_usage", {}),
    )
    
    return {
        "research_brief": result.get("research_brief", state.get("research_brief", "")),
        "notes": result.get("notes", []),
        "raw_notes": result.get("raw_notes", []),
        "completed_sections": writer_result.get("completed_sections", []),
        "agent_memories": {
            "type": "override",
            "value": result.get("agent_memories", state.get("agent_memories", {})),
        },
        "budget_usage": total_budget_update,
    }


async def maybe_persist_chat_memory(
    state: AgentState,
    config: RunnableConfig,
    final_report_content: str,
) -> None:
    """Persist chat transcript, generated summary, and durable memory into MySQL RAG memory."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_memory_write_enabled:
        return

    messages_text = get_buffer_string(
        _messages_without_query_image_context(state.get("messages", []))
    )
    research_brief = (state.get("research_brief") or "").strip()
    memories = [research_brief] if research_brief else []
    try:
        await asyncio.to_thread(
            persist_conversation_memory,
            configurable=configurable,
            runtime_config=config,
            chat_content=messages_text,
            summary=final_report_content,
            memories=memories,
            metadata={
                "workflow": "deep_researcher",
                "date": get_today_str(),
            },
        )
    except Exception as exc:  # pragma: no cover - depends on external MySQL/vector DB
        LOGGER.warning("Failed to persist chat memory: %s", exc)


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    configurable = Configuration.from_runnable_config(config)
    budget_usage = state.get("budget_usage", {})
    budget_update = {}

    if not can_spend_model_call(configurable, budget_usage):
        budget_update = budget_usage_with_reason(
            "Skipped final report generation because no model call budget remained."
        )
        fallback_report = findings or "Budget guard stopped before enough research notes were collected to generate a final report."
        final_usage = merge_budget_usage(budget_usage, budget_update)
        final_report_content = append_budget_summary(
            fallback_report,
            configurable,
            final_usage,
        )
        return {
            "final_report": final_report_content,
            "messages": [AIMessage(content=final_report_content)],
            "budget_usage": budget_update,
            **cleared_state,
        }

    messages_text = get_buffer_string(state.get("messages", []))
    remaining_input_budget = remaining_input_tokens(configurable, budget_usage)
    if remaining_input_budget is not None:
        reserved_prompt_tokens = (
            estimate_tokens(state.get("research_brief", ""))
            + estimate_tokens(messages_text)
            + 1000
        )
        findings_budget = max(0, remaining_input_budget - reserved_prompt_tokens)
        findings, findings_truncated = truncate_text_to_token_budget(
            findings,
            findings_budget,
        )
        if findings_truncated:
            budget_update = merge_budget_usage(
                budget_update,
                budget_usage_with_reason(
                    "Truncated research findings to fit the remaining input token budget."
                ),
            )
    
    # Step 2: Configure the final report generation model
    final_report_max_tokens = configurable.final_report_model_max_tokens
    remaining_output_budget = remaining_output_tokens(
        configurable,
        merge_budget_usage(budget_usage, budget_update),
    )
    if remaining_output_budget is not None:
        if remaining_output_budget > 0:
            final_report_max_tokens = min(final_report_max_tokens, remaining_output_budget)
        elif configurable.reserve_final_report_call:
            final_report_max_tokens = min(final_report_max_tokens, 512)
            budget_update = merge_budget_usage(
                budget_update,
                budget_usage_with_reason(
                    "Generated a compact final report after the output token budget was exhausted."
                ),
            )
        else:
            budget_update = merge_budget_usage(
                budget_update,
                budget_usage_with_reason(
                    "Skipped final report generation because no output token budget remained."
                ),
            )
            fallback_report = findings or "Budget guard stopped before enough research notes were collected to generate a final report."
            final_usage = merge_budget_usage(budget_usage, budget_update)
            final_report_content = append_budget_summary(
                fallback_report,
                configurable,
                final_usage,
            )
            return {
                "final_report": final_report_content,
                "messages": [AIMessage(content=final_report_content)],
                "budget_usage": budget_update,
                **cleared_state,
            }

    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": final_report_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            if _public_opinion_mode(configurable):
                final_report_prompt = public_opinion_final_report_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    organization_context=_business_context(configurable),
                    messages=messages_text,
                    findings=findings,
                    date=get_today_str(),
                )
            else:
                final_report_prompt = final_report_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    messages=messages_text,
                    findings=findings,
                    date=get_today_str()
                )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            final_budget_update = merge_budget_usage(
                budget_update,
                budget_from_model_response(final_report),
            )
            final_usage = merge_budget_usage(budget_usage, final_budget_update)
            final_report_content = append_budget_summary(
                str(final_report.content),
                configurable,
                final_usage,
            )
            await maybe_persist_chat_memory(state, config, final_report_content)
            
            # Return successful report generation
            return {
                "final_report": final_report_content,
                "messages": [AIMessage(content=final_report_content)],
                "budget_usage": final_budget_update,
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            "budget_usage": budget_update,
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    "budget_usage": budget_update,
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        "budget_usage": merge_budget_usage(
            budget_update,
            budget_usage_with_reason("Final report generation failed after maximum retries."),
        ),
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("enrich_query_images", enrich_query_images)       # Query-time image recognition
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("plan_report_sections", plan_report_sections)     # Section planning (Plan-and-Execute)
deep_researcher_builder.add_node("research_supervisor", research_phase)            # Research execution phase
deep_researcher_builder.add_node("write_final_sections", write_final_sections)     # Non-research section writing
deep_researcher_builder.add_node("compile_final_report", compile_final_report)     # Final report assembly

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "enrich_query_images")                     # Entry point
deep_researcher_builder.add_edge("enrich_query_images", "clarify_with_user")       # Image context to planning
deep_researcher_builder.add_edge("research_supervisor", "write_final_sections")    # Research to final sections
deep_researcher_builder.add_edge("write_final_sections", "compile_final_report")   # Final sections to compilation
deep_researcher_builder.add_edge("compile_final_report", END)                     # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
