"""Configuration management for the Open Deep Research system."""

import json
import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator, model_validator


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


class RetrievalMode(Enum):
    """Enumeration of available research retrieval modes."""

    WEB_ONLY = "web_only"
    RAG_ONLY = "rag_only"
    HYBRID = "hybrid"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    rag_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to enable the optional local RAG retrieval pipeline."
            }
        }
    )
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.WEB_ONLY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": RetrievalMode.WEB_ONLY.value,
                "description": "Research retrieval mode. Use web_only to preserve the current behavior, rag_only to use only the local knowledge base, or hybrid to combine both.",
                "options": [
                    {"label": "Web only", "value": RetrievalMode.WEB_ONLY.value},
                    {"label": "RAG only", "value": RetrievalMode.RAG_ONLY.value},
                    {"label": "Hybrid", "value": RetrievalMode.HYBRID.value}
                ]
            }
        }
    )
    rag_top_k: int = Field(
        default=4,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 4,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of local RAG chunks to return for each query."
            }
        }
    )
    rag_chunk_size: int = Field(
        default=1200,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1200,
                "min": 200,
                "max": 8000,
                "description": "Chunk size for local RAG document splitting."
            }
        }
    )
    rag_chunk_overlap: int = Field(
        default=200,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 200,
                "min": 0,
                "max": 2000,
                "description": "Chunk overlap for local RAG document splitting."
            }
        }
    )
    rag_rerank_top_n: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "How many initial local retrieval results to rerank before keeping the final top-k."
            }
        }
    )
    rag_embedding_provider: str = Field(
        default="hash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "hash",
                "description": "Embedding backend for local RAG. The built-in default is 'hash' for a zero-dependency local baseline."
            }
        }
    )
    rag_vectorstore_provider: str = Field(
        default="memory",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "memory",
                "description": "Vector store backend for local RAG. The built-in default is the in-memory vector store."
            }
        }
    )
    rag_reranker_provider: str = Field(
        default="simple",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "simple",
                "description": "Reranker backend for local RAG. Use 'simple' for the built-in keyword overlap reranker or 'none' to disable reranking."
            }
        }
    )
    rag_knowledge_base_paths: Optional[List[str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Comma-separated paths or a JSON array of local files/directories to index for RAG."
            }
        }
    )
    rag_json_text_fields: Optional[List[str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Optional comma-separated JSON field paths to extract for RAG ingestion. Leave empty to recursively collect string fields."
            }
        }
    )
    rag_hash_embedding_dimensions: int = Field(
        default=256,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 256,
                "min": 32,
                "max": 2048,
                "description": "Embedding vector width for the built-in hash embedding backend."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    @field_validator("rag_knowledge_base_paths", "rag_json_text_fields", mode="before")
    @classmethod
    def normalize_list_settings(cls, value: Any) -> Optional[list[str]]:
        """Normalize list-like configuration provided through env vars or runnable config."""
        if value in (None, ""):
            return None
        if isinstance(value, list):
            normalized_items = [str(item).strip() for item in value if str(item).strip()]
            return normalized_items or None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    normalized_items = [str(item).strip() for item in parsed if str(item).strip()]
                    return normalized_items or None
            normalized_items = [item.strip() for item in stripped.split(",") if item.strip()]
            return normalized_items or None
        return value

    @model_validator(mode="after")
    def validate_rag_chunk_settings(self) -> "Configuration":
        """Validate local RAG chunking settings."""
        if self.rag_chunk_size <= 0:
            raise ValueError("rag_chunk_size must be greater than 0.")
        if self.rag_chunk_overlap < 0:
            raise ValueError("rag_chunk_overlap must be greater than or equal to 0.")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("rag_chunk_overlap must be smaller than rag_chunk_size.")
        return self


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
