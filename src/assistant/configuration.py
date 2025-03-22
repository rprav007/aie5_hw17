import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

from enum import Enum

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""
    max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
    local_llm: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))  # Default to DUCKDUCKGO
    fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")
    
    # Vector store configuration
    qdrant_host: str = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.environ.get("QDRANT_PORT", "6333"))
    vector_dimension: int = int(os.environ.get("VECTOR_DIMENSION", "4096"))  # Default for mxbai-embed-large
    embeddings_model: str = os.environ.get("OLLAMA_EMBEDDINGS_MODEL", "mxbai-embed-large")
    collection_name: str = os.environ.get("QDRANT_COLLECTION", "research_vectors")
    
    # JSON document configuration
    json_data_dir: str = os.environ.get("JSON_DATA_DIR", "./data/data")
    json_glob_pattern: str = os.environ.get("JSON_GLOB_PATTERN", "**/*.json")
    json_jq_schema: str = os.environ.get("JSON_JQ_SCHEMA", "..")
    json_text_content: bool = os.environ.get("JSON_TEXT_CONTENT", "False").lower() in ("true", "1", "t")
    load_json_on_start: bool = os.environ.get("LOAD_JSON_ON_START", "True").lower() in ("true", "1", "t")

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})