from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_core.documents import Document

def load_json_documents(
    data_dir: str = "./data/data",
    glob_pattern: str = "**/*.json",
    jq_schema: str = "..",
    text_content: bool = False
) -> List[Document]:
    """Load JSON documents from a directory.
    
    Args:
        data_dir: Directory containing JSON files
        glob_pattern: Pattern to match JSON files
        jq_schema: JQ schema for extracting content
        text_content: Whether to extract text content
        
    Returns:
        List of loaded documents
    """
    # Ensure data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True)
        return []
        
    # Initialize directory loader
    json_loader = DirectoryLoader(
        path=data_dir,
        glob=glob_pattern,
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": jq_schema,
            "text_content": text_content
        }
    )
    
    # Load documents
    return json_loader.load()

def store_documents_in_vectorstore(
    documents: List[Document],
    vector_store_client,
    embeddings_model,
    metadata: Optional[dict] = None
) -> None:
    """Store documents in the vector store.
    
    Args:
        documents: List of documents to store
        vector_store_client: Qdrant client instance
        embeddings_model: Model to generate embeddings
        metadata: Additional metadata to store with documents
    """
    for doc in documents:
        # Combine document metadata with additional metadata
        combined_metadata = {
            "source": doc.metadata.get("source", "local_json"),
            "type": "json_document",
            **(metadata or {}),
            **(doc.metadata or {})
        }
        
        # Store in vector store
        embedding = embeddings_model.embed_query(doc.page_content)
        vector_store_client.upsert(
            collection_name="research_vectors",
            points=[{
                "id": hash(doc.page_content),
                "vector": embedding,
                "payload": {
                    "text": doc.page_content,
                    **combined_metadata
                }
            }]
        ) 