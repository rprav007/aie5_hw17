import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from assistant.configuration import Configuration, SearchAPI
from assistant.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, duckduckgo_search
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions
from assistant.vector_store import initialize_vector_store, get_embeddings_model, store_documents_in_vectorstore
from assistant.document_loader import load_json_documents

# Initialize embeddings model globally since it's reused
embeddings_model = None

def initialize_state(state_input: SummaryStateInput, config: RunnableConfig) -> SummaryState:
    """Initialize the state with vector store."""
    global embeddings_model
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize embeddings model first
    embeddings_model = get_embeddings_model(configurable)
    
    # Initialize vector store with embeddings model
    vector_store = initialize_vector_store(configurable, embeddings_model)
    
    # Load and store JSON documents if enabled
    if configurable.load_json_on_start:
        try:
            json_docs = load_json_documents(
                data_dir=configurable.json_data_dir,
                glob_pattern=configurable.json_glob_pattern,
                jq_schema=configurable.json_jq_schema,
                text_content=False  # Set to False to handle dictionary content
            )
            if json_docs:
                # Convert dictionary content to string for storage
                processed_docs = []
                for doc in json_docs:
                    if isinstance(doc.page_content, dict):
                        # Convert dict to formatted string
                        content_str = json.dumps(doc.page_content, indent=2)
                        processed_docs.append(Document(
                            page_content=content_str,
                            metadata=doc.metadata
                        ))
                    else:
                        processed_docs.append(doc)
                
                store_documents_in_vectorstore(
                    documents=processed_docs,
                    vector_store=vector_store,
                    metadata={"research_topic": state_input.research_topic}
                )
        except Exception as e:
            print(f"Warning: Error loading JSON documents: {e}")
            # Continue without JSON documents
    
    return SummaryState(
        research_topic=state_input.research_topic,
        vector_store=vector_store
    )

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Check vector store for similar research first
    if state.vector_store:
        similar_docs = state.vector_store.similarity_search(state.research_topic, k=1)
        if similar_docs:
            # Use the most relevant result to inform the query
            context = similar_docs[0].page_content
            query_writer_instructions_formatted += f"\n\nConsider this related research:\n{context}"

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    query = json.loads(result.content)

    return {"search_query": query['query']}

def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    # Store research results in vector store
    if state.vector_store:
        doc = Document(
            page_content=search_str,
            metadata={
                "research_topic": state.research_topic,
                "search_query": state.search_query,
                "research_loop": state.research_loop_count,
                "type": "web_research"
            }
        )
        state.vector_store.add_documents([doc])

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0)
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= int(configurable.max_web_research_loops):
        return "web_research"
    else:
        return "finalize_summary"

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("initialize", initialize_state)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "initialize")
builder.add_edge("initialize", "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()