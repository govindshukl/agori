"""Core functionality for the Agori package."""

import logging
import os
from typing import Any, Dict, List

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from cryptography.fernet import Fernet, InvalidToken

from .exceptions import ConfigurationError, ProcessingError, SearchError


class SecureChromaDB:
    """Main class for secure ChromaDB and Azure OpenAI embeddings integration."""

    def __init__(
        self,
        api_key: str,
        api_endpoint: str,
        encryption_key: str,
        api_version: str = "2024-02-15-preview",
        api_type: str = "azure",
        model_name: str = "text-embedding-ada-002",
        storage_dir: str = "./secure_chroma_storage",
    ):
        """Initialize SecureChromaDB."""
        try:
            self.logger = self._setup_logging()

            if not encryption_key:
                raise ConfigurationError("Encryption key is required")

            # Initialize encryption
            self.encryption_key = encryption_key
            try:
                self.cipher_suite = Fernet(self.encryption_key)
            except Exception as e:
                raise ConfigurationError(f"Invalid encryption key: {str(e)}")

            try:
                # Initialize Azure OpenAI embedding function
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    api_base=api_endpoint,
                    api_type=api_type,
                    api_version=api_version,
                    model_name=model_name,
                )
            except Exception as e:
                raise ConfigurationError(f"Invalid API configuration: {str(e)}")

            # Initialize ChromaDB client
            self.storage_dir = storage_dir
            self.client = chromadb.PersistentClient(path=storage_dir)

            self.logger.info("SecureChromaDB initialized successfully")

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize SecureChromaDB: {str(e)}")

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the package."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _encrypt_text(self, text: str) -> str:
        """Encrypt a string."""
        try:
            return self.cipher_suite.encrypt(text.encode()).decode()
        except Exception as e:
            raise ProcessingError(f"Encryption failed: {str(e)}")

    def _decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt a string."""
        try:
            return self.cipher_suite.decrypt(encrypted_text.encode()).decode()
        except InvalidToken:
            raise ProcessingError("Failed to decrypt: Invalid token")
        except Exception as e:
            raise ProcessingError(f"Decryption failed: {str(e)}")

    def create_collection(
        self, name: str, metadata: Dict[str, Any] = None
    ) -> chromadb.Collection:
        """Create a new collection with optional metadata."""
        try:
            collection_id = f"encrypted_{name}_{os.urandom(8).hex()}"

            collection = self.client.create_collection(
                name=collection_id,
                embedding_function=self.embedding_function,
                metadata={
                    "original_name": self._encrypt_text(name),
                    "encrypted": True,
                    **(
                        {k: self._encrypt_text(str(v)) for k, v in metadata.items()}
                        if metadata
                        else {}
                    ),
                },
            )

            self.logger.info(f"Created collection: {collection_id}")
            return collection

        except Exception as e:
            raise ProcessingError(f"Failed to create collection: {str(e)}")

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
    ) -> None:
        """Add documents to a collection with encryption."""
        try:
            collection = self.client.get_collection(name=collection_name)

            # Generate embeddings from original text
            embeddings = self.embedding_function(documents)

            # Encrypt documents after embedding calculation
            encrypted_docs = [self._encrypt_text(doc) for doc in documents]

            # Encrypt metadata if provided
            encrypted_metadatas = None
            if metadatas:
                encrypted_metadatas = [
                    {k: self._encrypt_text(str(v)) for k, v in meta.items()}
                    for meta in metadatas
                ]

            # Generate IDs if not provided
            if not ids:
                ids = [f"doc_{i}_{os.urandom(4).hex()}" for i in range(len(documents))]

            collection.add(
                embeddings=embeddings,
                documents=encrypted_docs,
                metadatas=encrypted_metadatas,
                ids=ids,
            )

            self.logger.info(
                f"Added {len(documents)} documents to collection: {collection_name}"
            )

        except Exception as e:
            raise ProcessingError(f"Failed to add documents: {str(e)}")

    def query_collection(
        self, collection_name: str, query_texts: List[str], n_results: int = 5
    ) -> Dict:
        """Query the collection and decrypt results."""
        try:
            collection = self.client.get_collection(name=collection_name)

            # Generate embeddings from raw query text
            query_embeddings = self.embedding_function(query_texts)

            # Query using embeddings
            results = collection.query(
                query_embeddings=query_embeddings, n_results=n_results
            )

            # Decrypt returned documents
            decrypted_results = {
                "documents": [
                    [self._decrypt_text(doc) for doc in docs]
                    for docs in results["documents"]
                ],
                "distances": results["distances"],
                "ids": results["ids"],
            }

            # Decrypt metadata if present
            if "metadatas" in results and results["metadatas"]:
                decrypted_results["metadatas"] = [
                    [
                        {k: self._decrypt_text(str(v)) for k, v in meta.items()}
                        for meta in metadata_list
                    ]
                    for metadata_list in results["metadatas"]
                ]

            self.logger.info(
                f"Query completed. Found {len(results['documents'])} results"
            )
            return decrypted_results

        except Exception as e:
            raise SearchError(f"Failed to query collection: {str(e)}")
