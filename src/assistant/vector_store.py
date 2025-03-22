from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from assistant.configuration import Configuration

def initialize_vector_store(config: Configuration, embeddings_model: OllamaEmbeddings) -> QdrantVectorStore:
    """Initialize the Qdrant vector store.
    
    Args:
        config: Configuration instance
        embeddings_model: OllamaEmbeddings instance
        
    Returns:
        QdrantVectorStore instance
    """
    url = f"{config.qdrant_host}:{config.qdrant_port}"
    
    # Create an empty document to initialize the store
    empty_doc = Document(
        page_content="",
        metadata={"type": "initialization"}
    )
    
    return QdrantVectorStore.from_documents(
        documents=[empty_doc],
        embedding=embeddings_model,
        url=url,
        prefer_grpc=True,
        collection_name="research_vectors"
    )

def get_embeddings_model(config: Configuration) -> OllamaEmbeddings:
    """Get the Ollama embeddings model."""
    return OllamaEmbeddings(
        model=config.embeddings_model,
        base_url=config.ollama_base_url
    )

def store_documents_in_vectorstore(
    documents: List[Document],
    vector_store: QdrantVectorStore,
    metadata: Optional[dict] = None
) -> None:
    """Store documents in the vector store.
    
    Args:
        documents: List of documents to store
        vector_store: QdrantVectorStore instance
        metadata: Additional metadata to store with documents
    """
    # Add metadata to documents if provided
    if metadata:
        for doc in documents:
            doc.metadata.update({
                **metadata,
                "source": doc.metadata.get("source", "local_json"),
                "type": "json_document"
            })
    
    # Store documents
    vector_store.add_documents(documents)

def search_similar_results(
    vector_store: QdrantVectorStore,
    query: str,
    limit: int = 5
) -> List[Document]:
    """Search for similar research results.
    
    Args:
        vector_store: QdrantVectorStore instance
        query: Query string
        limit: Maximum number of results to return
        
    Returns:
        List of relevant documents
    """
    return vector_store.similarity_search(query, k=limit) 