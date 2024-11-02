# Agori

A secure Python package for document storage and semantic search using ChromaDB and Azure OpenAI embeddings with built-in encryption.

## Features

- 🔐 End-to-end encryption for documents and metadata
- 🔍 Semantic search with Azure OpenAI embeddings
- 📚 Multiple collection support within a database
- 💾 Persistent storage with unique database IDs
- 🧹 Automatic resource cleanup
- ⚡ Simple and intuitive API

## Installation

```bash
pip install agori
```

## Quick Start

```python
import base64
import os
from agori import SecureChromaDB

# Generate a secure encryption key
encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()

# Initialize SecureChromaDB with context manager
with SecureChromaDB(
    api_key="your-azure-api-key",
    api_endpoint="https://your-instance.openai.azure.com/",
    encryption_key=encryption_key,
    db_unique_id="my_secure_db",
    base_storage_path="./secure_storage",
    model_name="text-embedding-ada-002",  # or your deployment name
    api_version="2024-02-15-preview",
    api_type="azure"
) as db:
    # Create a collection
    collection = db.create_collection(
        name="my_documents",
        metadata={"description": "Secure document storage"}
    )

    # Add documents with metadata
    documents = [
        "Important confidential document",
        "Sensitive information details"
    ]
    metadatas = [
        {"type": "confidential", "department": "HR"},
        {"type": "sensitive", "department": "Finance"}
    ]
    
    db.add_documents(
        collection_name="my_documents",
        documents=documents,
        metadatas=metadatas
    )

    # Query documents
    results = db.query_collection(
        collection_name="my_documents",
        query_texts=["confidential information"],
        n_results=2
    )

    # Process results
    for i, docs in enumerate(results["documents"]):
        print(f"\nResults:")
        for j, doc in enumerate(docs):
            print(f"Document {j+1}: {doc}")
            if "metadatas" in results:
                print(f"Metadata: {results['metadatas'][i][j]}")
            print(f"Similarity: {results['distances'][i][j]}")
```

## Key Features Explained

### Secure Document Storage
- All documents and metadata are encrypted before storage
- Uses Fernet symmetric encryption
- Secure key management required

### Collection Management
```python
# List all collections
collections = db.list_collections()

# Drop a specific collection
db.drop_collection("collection_name")

# Clean up entire database
db.cleanup_database()
```

### Multiple Collections Support
```python
# Create different collections for different purposes
hr_collection = db.create_collection(
    name="hr_documents",
    metadata={"department": "HR", "security_level": "high"}
)

finance_collection = db.create_collection(
    name="finance_documents",
    metadata={"department": "Finance", "security_level": "high"}
)
```

### Automatic Resource Management
```python
# Using context manager for automatic cleanup
with SecureChromaDB(...) as db:
    # Your operations here
    # Resources are automatically cleaned up after the block
```

## Error Handling

The package provides specific exceptions for different scenarios:
- `ConfigurationError`: For initialization and configuration issues
- `ProcessingError`: For document processing and collection operations
- `SearchError`: For query-related issues

```python
from agori import ConfigurationError, ProcessingError, SearchError

try:
    results = db.query_collection(
        collection_name="my_collection",
        query_texts=["search term"]
    )
except SearchError as e:
    print(f"Search failed: {str(e)}")
```

## Security Considerations

1. Key Management
   - Store encryption keys securely
   - Never hardcode keys in source code
   - Use environment variables or secure key management systems

2. API Credentials
   - Keep Azure OpenAI credentials secure
   - Use appropriate access controls
   - Monitor API usage

3. Storage
   - Secure the storage location
   - Regular backups if needed
   - Proper cleanup of sensitive data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

For more information, visit our [GitHub repository](https://github.com/govindshukl/agori).