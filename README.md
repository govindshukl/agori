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

#cleanup the collection and db
db.drop_collection("research_papers")
db.cleanup_database()

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