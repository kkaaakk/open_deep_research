"""Tests for social media MCP primitives and thin LangChain tool wrappers."""

from open_deep_research.social_media.tools import (
    SOCIAL_MEDIA_TOOL_NAMES,
    get_social_media_tools,
)
from open_deep_research.social_media_mcp import (
    _dedupe_records,
    _matches_query,
    _query_groups,
    check_duplicate,
    extract_images,
    fetch_comments as mcp_fetch_comments,
    fetch_x_thread,
    normalize_post,
    search_complaints,
    search_posts,
    search_x_posts,
)

SAMPLE_CONFIG = {"data_paths": ["data/social_media/sample_posts.jsonl"]}


# ── low-level MCP primitives ────────────────────────────────────────────

def test_search_x_posts_supports_cursor_pagination() -> None:
    first_page = search_x_posts("battery", limit=2, config=SAMPLE_CONFIG)
    assert first_page["ok"] is True
    assert first_page["total_matches"] == 3
    assert first_page["returned_count"] == 2
    assert first_page["next_cursor"] == "2"

    second_page = search_x_posts("battery", limit=2, cursor=first_page["next_cursor"], config=SAMPLE_CONFIG)
    assert second_page["ok"] is True
    assert second_page["returned_count"] == 1
    assert second_page["next_cursor"] is None


def test_low_level_thread_comments_images_and_normalization() -> None:
    thread = fetch_x_thread("x-root-001", config=SAMPLE_CONFIG)
    comments = mcp_fetch_comments("x-root-001", config=SAMPLE_CONFIG)
    images = extract_images("x-root-001", config=SAMPLE_CONFIG)
    normalized = normalize_post({"id": "raw-1", "platform": "x", "text": "raw post"})
    duplicate = check_duplicate({"id": "x-root-001"}, config=SAMPLE_CONFIG)

    assert thread["ok"] is True
    assert thread["post_count"] >= 2
    assert comments["ok"] is True
    assert comments["returned_count"] >= 1
    assert images["ok"] is True
    assert images["image_count"] == 1
    assert normalized["post"]["content"] == "raw post"
    assert duplicate["is_duplicate"] is True


def test_alias_matching_and_deduplication() -> None:
    records = [
        {"id": "dup-1", "platform": "weibo", "content": "A-EV battery discussion is increasing.",
         "published_at": "2026-06-02T08:00:00", "sentiment": "negative",
         "engagement": {"views": 100}, "tags": ["battery"]},
        {"id": "dup-1", "platform": "weibo", "content": "A-EV battery discussion is increasing.",
         "published_at": "2026-06-02T08:00:00", "sentiment": "negative",
         "engagement": {"views": 100}, "tags": ["battery"]},
    ]
    config = {"entity_aliases": {"Acme EV": ["A-EV", "Acme electric vehicle"]}}
    query_groups = _query_groups("Acme EV", config)
    assert len(_dedupe_records(records)) == 1
    assert _matches_query(records[0], query_groups) is True


def test_search_posts_mcp() -> None:
    result = search_posts("battery", limit=5, config=SAMPLE_CONFIG)
    assert result["ok"] is True
    assert result["total_matches"] >= 3


def test_fetch_thread_mcp() -> None:
    result = search_x_posts("x-root-001", config=SAMPLE_CONFIG)
    # search_x_posts resolves by thread_id; result contains matching posts
    assert result.get("ok") is True


def test_search_complaints_mcp() -> None:
    result = search_complaints("battery", limit=10, config=SAMPLE_CONFIG)
    assert result["ok"] is True


# ── thin @tool wrappers ────────────────────────────────────────────────

def test_tool_names_match_expected_set() -> None:
    names = {tool.name for tool in get_social_media_tools()}
    assert names == SOCIAL_MEDIA_TOOL_NAMES
    assert "search_posts" in names
    assert "fetch_thread" in names
    assert "fetch_comments" in names
    assert "search_complaints" in names
    assert "fetch_author_profile" in names
    assert "get_trending_keywords" in names
    assert "get_public_opinion_snapshot" in names


