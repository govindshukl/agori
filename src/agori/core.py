"""Core functionality for the Agori package."""

import datetime
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        db_unique_id: str,
        api_version: str = "2024-02-15-preview",
        api_type: str = "azure",
        model_name: str = "text-embedding-ada-002",
        base_storage_path: str = "./secure_chroma_storage",
    ):
        """Initialize SecureChromaDB."""
        try:
            self.logger = self._setup_logging()
            self.db_unique_id = self._validate_db_id(db_unique_id)

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

            # Initialize ChromaDB client with unique storage path
            self.storage_path = Path(base_storage_path) / self.db_unique_id
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.storage_path))

            self.logger.info(f"SecureChromaDB initialized with ID: {self.db_unique_id}")

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize SecureChromaDB: {str(e)}")

    def _validate_db_id(self, db_id: str) -> str:
        """Validate database ID."""
        if not db_id or not db_id.strip():
            raise ConfigurationError("Database ID cannot be empty")

        clean_id = "".join(c for c in db_id if c.isalnum() or c in ["-", "_"])
        if not clean_id:
            raise ConfigurationError("Database ID must contain valid characters")

        return clean_id

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
            if not name.strip():
                raise ProcessingError("Collection name cannot be empty")

            collection_name = name.strip()

            collection_metadata = {
                "encrypted": True,
                "creation_time": str(datetime.datetime.utcnow()),
                **(
                    {k: self._encrypt_text(str(v)) for k, v in metadata.items()}
                    if metadata
                    else {}
                ),
            }

            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata=collection_metadata,
            )

            self.logger.info(f"Created collection: {collection_name}")
            return collection

        except Exception as e:
            raise ProcessingError(f"Failed to create collection: {str(e)}")

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the database."""
        try:
            collections = self.client.list_collections()
            collection_info = []

            for collection in collections:
                try:
                    metadata = collection.metadata or {}
                    collection_info.append(
                        {
                            "name": collection.name,
                            "creation_time": metadata.get("creation_time"),
                            "metadata": (
                                {
                                    k: self._decrypt_text(str(v))
                                    for k, v in metadata.items()
                                    if k not in ["encrypted", "creation_time"]
                                }
                                if metadata
                                else {}
                            ),
                        }
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Error processing collection {collection.name}: {str(e)}"
                    )

            return collection_info

        except Exception as e:
            raise ProcessingError(f"Failed to list collections: {str(e)}")

    def get_collection(self, name: str) -> chromadb.Collection:
        """Get a collection by its name."""
        try:
            return self.client.get_collection(name=name)
        except Exception as e:
            raise ProcessingError(f"Failed to get collection {name}: {str(e)}")

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents to a collection with encryption."""
        try:
            collection = self.get_collection(collection_name)

            embeddings = self.embedding_function(documents)
            encrypted_docs = [self._encrypt_text(doc) for doc in documents]

            encrypted_metadatas = None
            if metadatas:
                encrypted_metadatas = [
                    {k: self._encrypt_text(str(v)) for k, v in meta.items()}
                    for meta in metadatas
                ]

            final_ids = ids or [
                f"doc_{i}_{os.urandom(4).hex()}" for i in range(len(documents))
            ]

            collection.add(
                embeddings=embeddings,
                documents=encrypted_docs,
                metadatas=encrypted_metadatas,
                ids=final_ids,
            )

            self.logger.info(
                f"Added {len(documents)} documents to collection: {collection_name}"
            )
            return final_ids

        except Exception as e:
            raise ProcessingError(f"Failed to add documents: {str(e)}")

    def query_collection(
        self, collection_name: str, query_texts: List[str], n_results: int = 5
    ) -> Dict:
        """Query the collection and decrypt results."""
        try:
            collection = self.get_collection(collection_name)

            query_embeddings = self.embedding_function(query_texts)
            results = collection.query(
                query_embeddings=query_embeddings, n_results=n_results
            )

            decrypted_results = {
                "documents": [
                    [self._decrypt_text(doc) for doc in docs]
                    for docs in results["documents"]
                ],
                "distances": results["distances"],
                "ids": results["ids"],
            }

            if "metadatas" in results and results["metadatas"]:
                decrypted_results["metadatas"] = [
                    [
                        {k: self._decrypt_text(str(v)) for k, v in meta.items()}
                        for meta in metadata_list
                    ]
                    for metadata_list in results["metadatas"]
                ]

            self.logger.info(
                f"""Query completed for collection
                {collection_name}. Found {len(results['documents'])} results"""
            )
            return decrypted_results

        except Exception as e:
            raise SearchError(f"Failed to query collection: {str(e)}")

    def drop_collection(self, name: str) -> None:
        """
        Drop a collection from the database.

        Args:
            name (str): Name of the collection to drop

        Raises:
            ProcessingError: If the collection cannot be dropped
        """
        try:
            self.client.delete_collection(name=name)
            self.logger.info(f"Dropped collection: {name}")
        except Exception as e:
            raise ProcessingError(f"Failed to drop collection {name}: {str(e)}")

    def cleanup_database(self, force: bool = False) -> None:
        """
        Clean up the database by removing all collections and associated files.

        Args:
            force (bool): If True, forcefully remove files even if collections
                         cannot be dropped cleanly

        Raises:
            ProcessingError: If the database cleanup fails
        """
        try:
            # First try to cleanly delete all collections
            try:
                collections = self.client.list_collections()
                for collection in collections:
                    try:
                        self.drop_collection(collection.name)
                    except Exception as e:
                        if not force:
                            raise ProcessingError(
                                f"Failed to drop collection {collection.name}: {str(e)}"
                            )
                        self.logger.warning(
                            f"Failed to cleanly drop collection {collection.name}, "
                            f"continuing with force cleanup: {str(e)}"
                        )
            except Exception as e:
                if not force:
                    raise ProcessingError(f"Failed to list collections: {str(e)}")
                self.logger.warning(
                    f"""Failed to list collections,
                        continuing with force cleanup: {str(e)}"""
                )

            # Reset the ChromaDB client connection
            try:
                self.client.reset()
            except Exception as e:
                self.logger.warning(f"Failed to reset client connection: {str(e)}")

            # Remove the database directory
            if self.storage_path.exists():
                try:
                    shutil.rmtree(self.storage_path)
                    self.logger.info(
                        f"Removed database directory: {str(self.storage_path)}"
                    )
                except Exception as e:
                    raise ProcessingError(
                        f"Failed to remove database directory: {str(e)}"
                    )

            self.logger.info(f"Database cleanup completed for ID: {self.db_unique_id}")

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Database cleanup failed: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            self.cleanup_database(force=True)
        except Exception as e:
            self.logger.error(f"Failed to cleanup database on exit: {str(e)}")
