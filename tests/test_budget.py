from types import SimpleNamespace

from langchain_core.messages import AIMessage

from open_deep_research.budget import (
    available_research_unit_slots,
    budget_from_model_response,
    capture_model_response,
    diff_budget_usage,
    filter_tool_calls_for_budget,
    merge_budget_usage,
    start_budget_capture,
    stop_budget_capture,
)
from open_deep_research.configuration import Configuration


def test_budget_from_ai_message_usage_metadata():
    message = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 13,
            "output_tokens": 7,
            "total_tokens": 20,
        },
    )

    usage = budget_from_model_response(message)

    assert usage["model_calls"] == 1
    assert usage["input_tokens"] == 13
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 20


def test_budget_from_ai_message_response_metadata_token_usage():
    message = AIMessage(
        content="done",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "total_tokens": 16,
            }
        },
    )

    usage = budget_from_model_response(message)

    assert usage["model_calls"] == 1
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 5
    assert usage["total_tokens"] == 16


def test_merge_and_diff_budget_usage_for_parallel_researchers():
    baseline = {
        "model_calls": 2,
        "tool_calls": 1,
        "search_calls": 1,
        "degradation_reasons": ["started compact mode"],
    }
    researcher_total = {
        "model_calls": 4,
        "tool_calls": 3,
        "search_calls": 2,
        "input_tokens": 80,
        "output_tokens": 30,
        "degradation_reasons": ["started compact mode", "skipped extra search"],
    }

    delta = diff_budget_usage(researcher_total, baseline)
    merged = merge_budget_usage(baseline, delta)

    assert delta["model_calls"] == 2
    assert delta["tool_calls"] == 2
    assert delta["search_calls"] == 1
    assert delta["degradation_reasons"] == ["skipped extra search"]
    assert merged["model_calls"] == 4
    assert merged["tool_calls"] == 3
    assert merged["search_calls"] == 2


def test_budget_capture_records_nested_model_calls():
    token = start_budget_capture()
    capture_model_response(
        AIMessage(
            content="nested",
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
        )
    )

    captured_usage = stop_budget_capture(token)

    assert captured_usage["model_calls"] == 1
    assert captured_usage["input_tokens"] == 3
    assert captured_usage["output_tokens"] == 2


def test_filter_tool_calls_skips_after_tool_budget():
    configurable = Configuration(budget_enabled=True, max_tool_calls=1)
    tool_calls = [
        {"name": "web_search", "args": {"query": "alpha"}, "id": "call_1"},
        {"name": "rag_search", "args": {"query": "beta"}, "id": "call_2"},
    ]
    tools_by_name = {
        "web_search": SimpleNamespace(name="web_search", metadata={"type": "search"}),
        "rag_search": SimpleNamespace(name="rag_search", metadata={"type": "search"}),
    }

    allowed, skipped = filter_tool_calls_for_budget(
        configurable,
        {},
        tool_calls,
        tools_by_name,
    )

    assert [tool_call["id"] for tool_call in allowed] == ["call_1"]
    assert [tool_call["id"] for tool_call in skipped] == ["call_2"]


def test_filter_tool_calls_skips_after_search_budget():
    configurable = Configuration(budget_enabled=True, max_search_calls=1)
    usage = {"search_calls": 1}
    tool_calls = [
        {"name": "rag_search", "args": {"query": "alpha"}, "id": "call_1"},
    ]
    tools_by_name = {
        "rag_search": SimpleNamespace(name="rag_search", metadata={"type": "search"}),
    }

    allowed, skipped = filter_tool_calls_for_budget(
        configurable,
        usage,
        tool_calls,
        tools_by_name,
    )

    assert allowed == []
    assert skipped == tool_calls


def test_available_research_units_reserves_final_report_call():
    configurable = Configuration(
        budget_enabled=True,
        max_model_calls=5,
        reserve_final_report_call=True,
    )

    assert available_research_unit_slots(configurable, {"model_calls": 2}) == 1
    assert available_research_unit_slots(configurable, {"model_calls": 3}) == 0


def test_budget_disabled_has_unlimited_research_unit_slots():
    configurable = Configuration(budget_enabled=False, max_model_calls=1)

    assert available_research_unit_slots(configurable, {"model_calls": 100}) is None
