"""Tests for SecureChromaDB functionality."""

import base64
import datetime
import logging  # Added import for logging
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from cryptography.fernet import Fernet

from agori import ConfigurationError, ProcessingError, SecureChromaDB


@pytest.fixture
def encryption_key():
    """Fixture to provide a consistent encryption key for tests."""
    return base64.urlsafe_b64encode(os.urandom(32))


@pytest.fixture
def mock_embeddings():
    """Fixture to provide mock embeddings."""
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.fixture
def secure_db(encryption_key):
    """Fixture to create a SecureChromaDB instance with mocked dependencies."""
    with patch("chromadb.PersistentClient"), patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction"
    ) as mock_ef:
        # Configure mock embedding function
        mock_ef.return_value = Mock()
        mock_ef.return_value.side_effect = lambda x: [[0.1, 0.2, 0.3] for _ in x]

        db = SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key=encryption_key,
            db_unique_id="test-db",
            base_storage_path="./test_storage",
        )
        yield db


def test_initialization(encryption_key):
    """Test successful initialization of SecureChromaDB."""
    with patch("chromadb.PersistentClient"), patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction"
    ):
        db = SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key=encryption_key,
            db_unique_id="test-db",  # Added required db_unique_id
        )

        assert db.encryption_key == encryption_key
        assert isinstance(db.cipher_suite, Fernet)
        assert db.db_unique_id == "test-db"  # Added assertion for db_unique_id


def test_initialization_without_encryption_key():
    """Test that initialization fails without encryption key."""
    with pytest.raises(ConfigurationError) as excinfo:
        SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key="",
            db_unique_id="test-db",  # Added required db_unique_id
        )
    assert "Encryption key is required" in str(excinfo.value)


def test_initialization_without_db_id():
    """Test that initialization fails without db_unique_id."""
    with pytest.raises(ConfigurationError) as excinfo:
        SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key=base64.urlsafe_b64encode(os.urandom(32)),
            db_unique_id="",  # Empty db_unique_id
        )
    assert "Database ID cannot be empty" in str(excinfo.value)


def test_invalid_db_id():
    """Test initialization with invalid db_unique_id."""
    with pytest.raises(ConfigurationError) as excinfo:
        SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key=base64.urlsafe_b64encode(os.urandom(32)),
            db_unique_id="@#$%",  # Invalid characters
        )
    assert "Database ID must contain valid characters" in str(excinfo.value)


def test_encryption_decryption(secure_db):
    """Test text encryption and decryption."""
    original_text = "This is a secret message"
    encrypted = secure_db._encrypt_text(original_text)
    decrypted = secure_db._decrypt_text(encrypted)

    assert encrypted != original_text
    assert decrypted == original_text


def test_create_collection(secure_db):
    """Test collection creation with metadata."""
    metadata = {"description": "Test collection"}

    with patch.object(secure_db.client, "create_collection") as mock_create:
        mock_collection = Mock()
        mock_create.return_value = mock_collection

        collection = secure_db.create_collection("test_collection", metadata)
        assert collection is not None

        assert mock_create.called
        call_args = mock_create.call_args[1]
        assert "metadata" in call_args
        assert call_args["metadata"]["encrypted"] is True
        assert "creation_time" in call_args["metadata"]  # Added check for creation_time


def test_list_collections(secure_db):
    """Test listing collections."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"
    mock_collection.metadata = {
        "encrypted": True,
        "creation_time": str(datetime.datetime.utcnow()),
        "description": secure_db._encrypt_text("Test collection"),
    }

    with patch.object(secure_db.client, "list_collections") as mock_list:
        mock_list.return_value = [mock_collection]

        collections = secure_db.list_collections()

        assert len(collections) == 1
        assert collections[0]["name"] == "test_collection"
        assert "creation_time" in collections[0]
        assert collections[0]["metadata"]["description"] == "Test collection"


def test_add_documents(secure_db):
    """Test adding documents to a collection."""
    documents = ["Doc 1", "Doc 2"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    with patch.object(secure_db.client, "get_collection") as mock_get:
        mock_collection = Mock()
        mock_get.return_value = mock_collection

        doc_ids = secure_db.add_documents(
            collection_name="test_collection",
            documents=documents,
            metadatas=metadatas,
        )

        assert mock_collection.add.called
        call_args = mock_collection.add.call_args[1]

        # Verify documents were encrypted
        assert all(isinstance(doc, str) for doc in call_args["documents"])
        assert len(call_args["documents"]) == len(documents)

        # Verify metadata was encrypted
        assert all(
            isinstance(next(iter(m.values())), str) for m in call_args["metadatas"]
        )

        # Verify document IDs were returned
        assert len(doc_ids) == len(documents)


def test_query_collection(secure_db):
    """Test querying a collection."""
    test_doc = "Test document"
    encrypted_doc = secure_db._encrypt_text(test_doc)

    mock_results = {
        "documents": [[encrypted_doc, encrypted_doc]],
        "distances": [[0.1, 0.2]],
        "ids": [["id1", "id2"]],
        "metadatas": [[{"source": secure_db._encrypt_text("test")}]],
    }

    with patch.object(secure_db.client, "get_collection") as mock_get:
        mock_collection = Mock()
        mock_collection.query.return_value = mock_results
        mock_get.return_value = mock_collection

        results = secure_db.query_collection(
            collection_name="test_collection",
            query_texts=["test query"],
        )

        assert mock_collection.query.called
        assert "documents" in results
        assert "distances" in results
        assert "ids" in results
        assert "metadatas" in results

        # Verify results were decrypted properly
        assert results["documents"][0][0] == test_doc


def test_invalid_api_credentials():
    """Test initialization with invalid API credentials."""
    with pytest.raises(ConfigurationError) as excinfo:
        with patch(
            "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction"
        ) as mock_ef:
            mock_ef.side_effect = Exception("Invalid credentials")
            SecureChromaDB(
                api_key="invalid",
                api_endpoint="invalid",
                encryption_key=base64.urlsafe_b64encode(os.urandom(32)),
                db_unique_id="test-db",  # Added required db_unique_id
            )
    assert "Invalid API configuration" in str(excinfo.value)


def test_collection_not_found(secure_db):
    """Test handling of non-existent collection."""
    with patch.object(
        secure_db.client,
        "get_collection",
        side_effect=ValueError("Collection not found"),
    ):
        with pytest.raises(ProcessingError):
            secure_db.add_documents("nonexistent_collection", ["doc1"])


def test_drop_collection(secure_db):
    """Test dropping a collection."""
    with patch.object(secure_db.client, "delete_collection") as mock_delete:
        # Test successful drop
        secure_db.drop_collection("test_collection")
        mock_delete.assert_called_once_with(name="test_collection")


def test_drop_collection_error(secure_db):
    """Test error handling when dropping a collection fails."""
    with patch.object(
        secure_db.client, "delete_collection", side_effect=Exception("Failed to delete")
    ):
        with pytest.raises(ProcessingError) as excinfo:
            secure_db.drop_collection("test_collection")
        assert "Failed to drop collection" in str(excinfo.value)


def test_cleanup_database(secure_db):
    """Test database cleanup functionality."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"

    with patch.object(secure_db.client, "list_collections") as mock_list, patch.object(
        secure_db.client, "delete_collection"
    ) as mock_delete, patch.object(secure_db.client, "reset") as mock_reset, patch(
        "shutil.rmtree"
    ) as mock_rmtree, patch.object(
        Path, "exists", return_value=True
    ):

        # Mock collection listing
        mock_list.return_value = [mock_collection]

        # Test successful cleanup
        secure_db.cleanup_database()

        # Verify all cleanup steps were called
        mock_list.assert_called_once()
        mock_delete.assert_called_once_with(name="test_collection")
        mock_reset.assert_called_once()
        mock_rmtree.assert_called_once_with(secure_db.storage_path)


def test_cleanup_database_force(secure_db):
    """Test forced database cleanup when collection deletion fails."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"

    with patch.object(secure_db.client, "list_collections") as mock_list, patch.object(
        secure_db.client, "delete_collection", side_effect=Exception("Delete failed")
    ) as mock_delete, patch.object(secure_db.client, "reset") as mock_reset, patch(
        "shutil.rmtree"
    ) as mock_rmtree, patch.object(
        Path, "exists", return_value=True
    ):

        # Mock collection listing
        mock_list.return_value = [mock_collection]

        # Test forced cleanup
        secure_db.cleanup_database(force=True)

        # Verify cleanup steps were attempted
        mock_list.assert_called_once()
        mock_delete.assert_called_once_with(name="test_collection")
        mock_reset.assert_called_once()
        mock_rmtree.assert_called_once_with(secure_db.storage_path)


def test_cleanup_database_error_no_force(secure_db):
    """Test cleanup failure when not using force mode."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"

    with patch.object(secure_db.client, "list_collections") as mock_list, patch.object(
        secure_db.client, "delete_collection", side_effect=Exception("Delete failed")
    ):

        # Mock collection listing
        mock_list.return_value = [mock_collection]

        # Test cleanup without force should raise error
        with pytest.raises(ProcessingError) as excinfo:
            secure_db.cleanup_database(force=False)

        assert "Failed to drop collection" in str(excinfo.value)


def test_context_manager(encryption_key):
    """Test SecureChromaDB context manager functionality."""
    with patch("chromadb.PersistentClient"), patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction"
    ), patch.object(SecureChromaDB, "cleanup_database") as mock_cleanup:

        with SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key=encryption_key,
            db_unique_id="test-db",
        ) as db:
            # Perform some operation
            assert db.db_unique_id == "test-db"

        # Verify cleanup was called on exit
        mock_cleanup.assert_called_once_with(force=True)


def test_context_manager_error_handling(encryption_key):
    """Test context manager cleanup with error handling."""
    with patch("chromadb.PersistentClient"), patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction"
    ), patch.object(
        SecureChromaDB, "cleanup_database", side_effect=Exception("Cleanup failed")
    ), patch.object(
        logging.Logger, "error"
    ) as mock_log_error:

        with SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key=encryption_key,
            db_unique_id="test-db",
        ) as db:
            assert db.db_unique_id == "test-db"

        # Verify error was logged
        mock_log_error.assert_called_once()
        assert "Failed to cleanup database on exit" in mock_log_error.call_args[0][0]


if __name__ == "__main__":
    pytest.main([__file__])
