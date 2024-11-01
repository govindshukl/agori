# Agori

Agori is a powerful and user-friendly Python package that simplifies document processing and semantic search using ChromaDB and Azure OpenAI embeddings. It provides an intuitive interface for processing PDF documents, storing their embeddings, and performing semantic searches.

## Features

- 📄 Easy PDF document processing
- 🔍 Semantic search capabilities using Azure OpenAI embeddings
- 💾 Optional persistent storage support
- 🎯 Customizable chunk sizes and overlap for document splitting
- 🚀 Simple and intuitive API
- 🛡️ Comprehensive error handling
- 📝 Detailed logging

## Installation

```bash
pip install agori
```

## Quick Start

```python
from agori import Agori

# Initialize Agori with your Azure OpenAI credentials
agori = Agori(
    api_key="your-azure-api-key",
    api_base="https://your-instance.openai.azure.com/"
)

# Process a PDF document
result = agori.process_document(
    file_path="path/to/your/document.pdf",
    chunk_size=1000,  # Optional: customize chunk size
    chunk_overlap=200  # Optional: customize overlap
)

# Get the collection ID from the result
collection_id = result["collection_id"]

# Search the processed document
results = agori.search(
    collection_id=collection_id,
    query="What is the main topic?",
    n_results=3,  # Optional: number of results to return
    min_similarity=0.7  # Optional: minimum similarity threshold
)

# Print search results
for result in results:
    print(f"Rank: {result['rank']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Text: {result['text']}\n")
```

## Advanced Usage

### Persistent Storage

You can enable persistent storage to maintain your document embeddings across sessions:

```python
agori = Agori(
    api_key="your-azure-api-key",
    api_base="https://your-instance.openai.azure.com/",
    persist_directory="path/to/storage"
)
```

### Custom Collection Names

You can specify custom collection names instead of using auto-generated IDs:

```python
result = agori.process_document(
    file_path="document.pdf",
    collection_name="my-custom-collection"
)
```

### Error Handling

Agori provides detailed error messages and custom exceptions:

```python
from agori import AgoriException, ConfigurationError, ProcessingError, SearchError

try:
    results = agori.search("collection-id", "query")
except SearchError as e:
    print(f"Search failed: {e}")
except AgoriException as e:
    print(f"An error occurred: {e}")
```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/agori.git
cd agori

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .
```

### Running Tests

```bash
pytest tests -v --cov=agori
```

### Code Formatting

```bash
black src/agori tests
isort src/agori tests
```

### Linting

```bash
flake8 src/agori tests
mypy src/agori tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

- Python 3.8 or higher
- Azure OpenAI API access
- Required packages:
  - chromadb
  - openai
  - langchain
  - pypdf

## Support

If you encounter any issues or need support, please:

1. Check the [documentation](https://github.com/yourusername/agori/docs)
2. Search through [existing issues](https://github.com/yourusername/agori/issues)
3. Open a new issue if needed

## Acknowledgments

- ChromaDB for the vector database functionality
- Azure OpenAI for embeddings generation
- LangChain for document processing utilities