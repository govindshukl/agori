"""Core functionality for the Agori package."""

import uuid
from typing import Dict, List, Optional, Union
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from .exceptions import ConfigurationError, ProcessingError, SearchError
from .utils import setup_logging, split_document

logger = setup_logging()

class Agori:
    """Main class for ChromaDB and Azure OpenAI embeddings integration."""
    
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str = "text-embedding-ada-002",
        api_version: str = "2024-02-15-preview",
        persist_directory: Optional[str] = None
    ):
        """Initialize Agori.
        
        Args:
            api_key: Azure OpenAI API key
            api_base: Azure OpenAI API base URL
            model_name: Name of the embedding model
            api_version: API version
            persist_directory: Optional directory for persistent storage
            
        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=api_base,
                api_type="azure",
                api_version=api_version,
                model_name=model_name
            )
            
            client_settings = {}
            if persist_directory:
                client_settings["persist_directory"] = persist_directory
                
            self.client = chromadb.Client(**client_settings)
            logger.info("Agori initialized successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Agori: {str(e)}")

    def process_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: Optional[str] = None
    ) -> Dict[str, Union[str, int]]:
        """Process a PDF document and store its contents in ChromaDB.
        
        Args:
            file_path: Path to the PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            collection_name: Optional name for the collection
            
        Returns:
            Dict containing processing results
            
        Raises:
            ProcessingError: If document processing fails
        """
        try:
            documents = split_document(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            collection_id = collection_name or str(uuid.uuid4())
            
            collection = self.client.create_collection(
                name=collection_id,
                embedding_function=self.embedding_function
            )
            
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
            collection.add(documents=documents, ids=doc_ids)
            
            logger.info(f"Successfully processed document into collection: {collection_id}")
            
            return {
                "collection_id": collection_id,
                "chunk_count": len(documents),
                "status": "success"
            }
            
        except Exception as e:
            raise ProcessingError(f"Failed to process document: {str(e)}")

    def search(
        self,
        collection_id: str,
        query: str,
        n_results: int = 3,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Search documents in a ChromaDB collection.
        
        Args:
            collection_id: The collection ID
            query: Search query text
            n_results: Number of results to return
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List of search results with similarity scores
            
        Raises:
            SearchError: If search fails
        """
        try:
            collection = self.client.get_collection(
                name=collection_id,
                embedding_function=self.embedding_function
            )
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results and results['documents']:
                for i, (doc, distance) in enumerate(zip(
                    results['documents'][0],
                    results['distances'][0]
                )):
                    similarity = 1 - distance
                    if similarity >= min_similarity:
                        formatted_results.append({
                            "text": doc,
                            "similarity": similarity,
                            "rank": i + 1
                        })
            
            logger.info(f"Search completed. Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            raise SearchError(f"Failed to search documents: {str(e)}")