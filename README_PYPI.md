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

## Quick Start

```python
from agori.core import WorkingMemory
from cryptography.fernet import Fernet

# Generate a new encryption key (in practice, store this securely)
encryption_key = Fernet.generate_key()

# Initialize the secure database
db = WorkingMemory(
    api_key="your-azure-openai-key",
    api_endpoint="your-azure-endpoint",
    encryption_key=encryption_key,
    db_unique_id="my_secure_db"
)

# Create a new collection
collection_metadata = {
    "description": "Research papers database",
    "owner": "research_team"
}
collection = db.create_collection("research_papers", metadata=collection_metadata)

# Add documents with metadata
documents = [
    "Advances in Neural Networks",
    "Quantum Computing Overview",
    "Machine Learning Applications"
]
metadata_list = [
    {"author": "John Doe", "year": "2023"},
    {"author": "Jane Smith", "year": "2023"},
    {"author": "Bob Wilson", "year": "2024"}
]

# Add documents - they will be automatically encrypted
doc_ids = db.add_documents(
    collection_name="research_papers",
    documents=documents,
    metadatas=metadata_list
)

# Query the collection - results will be automatically decrypted
results = db.query_collection(
    collection_name="research_papers",
    query_texts=["neural networks"],
    n_results=2
)

# Process results
for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"Result {i+1}:")
    print(f"Document: {doc}")
    print(f"Similarity Score: {1 - distance}")  # Convert distance to similarity
    if "metadatas" in results:
        print(f"Metadata: {results['metadatas'][0][i]}")
    print()

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
with WorkingMemory(...) as db:
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