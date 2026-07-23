"""Tests for public-opinion business-agent encapsulation."""

from open_deep_research.configuration import Configuration
from open_deep_research.public_opinion_agents import (
    PUBLIC_OPINION_AGENT_ORDER,
    PUBLIC_OPINION_AGENT_SPECS,
    get_public_opinion_agent_spec,
)
from open_deep_research.state import agent_memories_reducer


def test_public_opinion_agent_registry_has_compact_roles() -> None:
    """Public-opinion workflow exposes four compact business agents."""
    assert PUBLIC_OPINION_AGENT_ORDER == (
        "public_signal",
        "internal_knowledge",
        "risk_assessment",
        "response_strategy",
    )
    assert set(PUBLIC_OPINION_AGENT_SPECS) == set(PUBLIC_OPINION_AGENT_ORDER)


def test_legacy_public_opinion_roles_are_mapped_to_compact_roles() -> None:
    """Old seven-agent role configs remain compatible."""
    config = Configuration(
        business_scenario="public_opinion_risk",
        enabled_business_agents=[
            "news_intelligence",
            "social_sentiment",
            "internal_knowledge",
            "fact_verification",
            "competitor_impact",
            "compliance_risk",
            "pr_strategy",
        ],
    )

    assert config.enabled_business_agents == [
        "public_signal",
        "internal_knowledge",
        "risk_assessment",
        "response_strategy",
    ]


def test_each_public_opinion_agent_owns_prompt_contract_and_policy() -> None:
    """Each agent spec renders a dedicated prompt with contract and policy sections."""
    for role in PUBLIC_OPINION_AGENT_ORDER:
        spec = get_public_opinion_agent_spec(role)
        prompt = spec.format_system_prompt(
            retrieval_tool_prompt="Role-specific tool whitelist: test.",
            mcp_prompt="",
            date="June 3, 2026",
            organization_context="Test organization context.",
            private_memory_context="Previous private note for this agent.",
        )

        assert spec.role in prompt
        assert spec.display_name in prompt
        assert "<Private Agent Memory>" in prompt
        assert "Previous private note for this agent." in prompt
        assert "<Input Contract>" in prompt
        assert "<Tool Policy>" in prompt
        assert "<Memory Policy>" in prompt
        assert "<Execution Strategy>" in prompt
        assert "<Output Schema>" in prompt
        assert spec.expected_output in prompt


def test_agent_memories_reducer_keeps_private_memory_by_role() -> None:
    """Private agent memory is appended under each role without cross-role mixing."""
    current = {
        "public_signal": [{"content": "old public memory"}],
        "risk_assessment": [{"content": "old risk memory"}],
    }
    update = {
        "public_signal": [{"content": "new public memory"}],
        "response_strategy": {"content": "new response memory"},
    }

    merged = agent_memories_reducer(current, update)

    assert [entry["content"] for entry in merged["public_signal"]] == [
        "old public memory",
        "new public memory",
    ]
    assert [entry["content"] for entry in merged["risk_assessment"]] == [
        "old risk memory",
    ]
    assert [entry["content"] for entry in merged["response_strategy"]] == [
        "new response memory",
    ]
