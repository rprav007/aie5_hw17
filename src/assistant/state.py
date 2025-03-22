from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain_qdrant import QdrantVectorStore

@dataclass
class SummaryState:
    """The state of the research assistant."""
    research_topic: str
    search_query: Optional[str] = None
    web_research_results: List[str] = field(default_factory=list)
    sources_gathered: List[str] = field(default_factory=list)
    research_loop_count: int = 0
    running_summary: str = ""
    vector_store: Optional[QdrantVectorStore] = None

@dataclass
class SummaryStateInput:
    """The input to the research assistant."""
    research_topic: str

@dataclass
class SummaryStateOutput:
    """The output of the research assistant."""
    running_summary: str