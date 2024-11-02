"""Tests for SecureChromaDB functionality."""

import base64
import os
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
            storage_dir="./test_storage",
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
        )

        assert db.encryption_key == encryption_key
        assert isinstance(db.cipher_suite, Fernet)


def test_initialization_without_encryption_key():
    """Test that initialization fails without encryption key."""
    with pytest.raises(ConfigurationError) as excinfo:
        SecureChromaDB(
            api_key="test-key",
            api_endpoint="https://test.openai.azure.com",
            encryption_key="",
        )
    assert "Encryption key is required" in str(excinfo.value)


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

        # Use _ to indicate unused return value
        _ = secure_db.create_collection("test_collection", metadata)

        assert mock_create.called
        call_args = mock_create.call_args[1]
        assert "metadata" in call_args
        assert call_args["metadata"]["encrypted"] is True
        assert "original_name" in call_args["metadata"]


def test_add_documents(secure_db):
    """Test adding documents to a collection."""
    documents = ["Doc 1", "Doc 2"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    with patch.object(secure_db.client, "get_collection") as mock_get:
        mock_collection = Mock()
        mock_get.return_value = mock_collection

        secure_db.add_documents(
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


def test_query_collection(secure_db):
    """Test querying a collection."""
    # Create encrypted mock data
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


if __name__ == "__main__":
    pytest.main([__file__])
