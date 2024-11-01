# Agori

A user-friendly Python package for document processing and semantic search using ChromaDB and Azure OpenAI embeddings.

## Installation

```bash
pip install agori
```

## Quick Start

```python
from agori import Agori

# Initialize
agori = Agori(
    api_key="your-azure-api-key",
    api_base="https://your-instance.openai.azure.com/"
)

# Process a document
result = agori.process_document("document.pdf")

# Search
results = agori.search(
    collection_id=result["collection_id"],
    query="What is the main topic?"
)

# Print results
for result in results:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Text: {result['text']}\n")
```

## Features

- 📄 Easy PDF document processing
- 🔍 Semantic search with Azure OpenAI embeddings
- 💾 Optional persistent storage
- 🚀 Simple and intuitive API

For more information, visit our [GitHub repository](https://github.com/yourusername/agori).