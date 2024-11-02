# Agori

Agori is a secure Python package that provides encrypted document storage and semantic search capabilities using ChromaDB and Azure OpenAI embeddings. It focuses on secure storage and retrieval of sensitive documents while maintaining searchability through encrypted vector embeddings.

## Features

- 🔐 End-to-end encryption for documents and metadata
- 🔍 Semantic search using Azure OpenAI embeddings
- 📚 Multiple collection management within a database
- 💾 Persistent storage with database isolation
- 🚀 Simple and intuitive API
- 🛡️ Comprehensive error handling
- 📝 Detailed logging
- 🧹 Automatic resource cleanup

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

# Initialize SecureChromaDB with context manager for automatic cleanup
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
    # Create and manage collections
    db.create_collection(
        name="sensitive_docs",
        metadata={"security_level": "high", "department": "HR"}
    )

    # Add documents with metadata
    db.add_documents(
        collection_name="sensitive_docs",
        documents=["Confidential report 2024", "Employee records"],
        metadatas=[
            {"type": "report", "year": "2024"},
            {"type": "records", "department": "HR"}
        ]
    )

    # Search documents
    results = db.query_collection(
        collection_name="sensitive_docs",
        query_texts=["confidential information"],
        n_results=2
    )

    # Process results
    for i, docs in enumerate(results["documents"]):
        for j, doc in enumerate(docs):
            print(f"Document: {doc}")
            print(f"Similarity: {results['distances'][i][j]}")
            if "metadatas" in results:
                print(f"Metadata: {results['metadatas'][i][j]}\n")
```

## Advanced Usage

### Collection Management

```python
# List all collections
collections = db.list_collections()
for collection in collections:
    print(f"Name: {collection['name']}")
    print(f"Creation Time: {collection['creation_time']}")
    print(f"Metadata: {collection['metadata']}\n")

# Drop a specific collection
db.drop_collection("collection_name")

# Clean up entire database
db.cleanup_database(force=True)
```

### Error Handling

```python
from agori import ConfigurationError, ProcessingError, SearchError

try:
    # Attempt to create collection
    collection = db.create_collection(
        name="secure_docs",
        metadata={"department": "Legal"}
    )
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ProcessingError as e:
    print(f"Processing error: {e}")
```

## Security Features

### Encryption
- All documents and metadata are encrypted using Fernet symmetric encryption
- Secure key generation and management required
- Encrypted storage of documents and metadata

### Database Isolation
- Each database instance has a unique ID
- Separate storage paths for different databases
- Secure cleanup of resources

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/govindshukl/agori.git
cd agori

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .
```

### Testing and Quality Assurance

```bash
# Run tests
pytest tests -v --cov=agori

# Code formatting
black src/agori tests
isort src/agori tests

# Linting
flake8 src/agori tests
mypy src/agori tests
```

## Requirements

- Python 3.8 or higher
- Azure OpenAI API access
- Required packages:
  - chromadb
  - cryptography
  - azure-openai

## Best Practices

### Security
1. Never hardcode encryption keys or API credentials
2. Use environment variables for sensitive information
3. Implement proper key management
4. Regular cleanup of sensitive data

### Resource Management
1. Use context managers for automatic cleanup
2. Properly handle collection lifecycle
3. Implement error handling for all operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or need support, please:

1. Check the [documentation](https://github.com/govindshukl/agori/docs)
2. Search through [existing issues](https://github.com/govindshukl/agori/issues)
3. Open a new issue if needed

## Acknowledgments

- ChromaDB for vector database functionality
- Azure OpenAI for embeddings generation
- Cryptography.io for encryption capabilities