# Agori

[![PyPI version](https://badge.fury.io/py/agori.svg)](https://badge.fury.io/py/agori)
[![Python Versions](https://img.shields.io/pypi/pyversions/agori.svg)](https://pypi.org/project/agori/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agori is a Python package that implements AI-powered decision-making frameworks. It enhances group decision-making processes by leveraging generative AI capabilities through structured frameworks.

## Features

- 🤖 AI-powered decision frameworks
- 🎯 Nominal Group Technique (NGT) implementation
- 📊 Multi-perspective analysis
- 🚀 Parallel processing for large documents
- 📝 Structured output with clear recommendations
- ⚙️ Highly configurable

## Installation

```bash
pip install agori
```

### Requirements

- Python 3.8+
- Azure OpenAI API access
- Required packages (automatically installed):
  - langchain
  - tiktoken
  - azure-openai
  - asyncio

## Quick Start

```python
import asyncio
from agori.decision_frameworks.ngt import NGTProcessor 
from langchain_community.document_loaders import PyPDFLoader

async def analyze_document(document_path: str, query: str):
    # Load document
    loader = PyPDFLoader(document_path)
    pages = loader.load()
    
    # Initialize NGT processor
    ngt = NGTProcessor(
        deployment="your-deployment",
        endpoint="your-endpoint", 
        api_key="your-api-key",
        api_version="your-api-version",
        verbose=True
    )

    # Add documents and run analysis
    ngt.add_documents(pages)
    result = await ngt.run_ngt_analysis(query)
    return result

# Run analysis
if __name__ == "__main__":
    document_path = "your_document.pdf"
    query = "What strategic decisions should we make based on this document?"
    
    # If running in Jupyter or similar environment
    import nest_asyncio
    nest_asyncio.apply()
    
    result = await analyze_document(document_path, query)
    print(result)
```

## Configuration

```python
from agori.decision_frameworks.ngt.models import Config

config = Config(
    MAX_TOKENS_PER_REQUEST=126000,
    CHUNK_SIZE=31500,
    CHUNK_OVERLAP=2000,
    MAX_WORKERS=3,
    MAX_PARALLEL_REQUESTS=3,
    VERBOSE=True
)

ngt = NGTProcessor(
    deployment="your-deployment",
    endpoint="your-endpoint", 
    api_key="your-api-key",
    api_version="your-api-version",
    config=config
)
```

## Key Components

### NGT Framework
The Nominal Group Technique framework includes:

```python
# Generate experts
experts = await ngt.generate_experts(query)

# Generate expert responses
expert_responses = await ngt.analyze_with_experts(experts, query)

# Synthesize results
synthesis = await ngt.synthesize_ngt_results(expert_responses)
```

### Error Handling

```python
from agori.decision_frameworks.ngt import (
    NGTError,
    ExpertGenerationError,
    IdeaGenerationError,
    SynthesisError
)

try:
    result = await ngt.run_ngt_analysis(query)
except ExpertGenerationError as e:
    print(f"Error generating experts: {e}")
except IdeaGenerationError as e:
    print(f"Error generating ideas: {e}")
except SynthesisError as e:
    print(f"Error synthesizing results: {e}")
```

## Data Classes

```python
from agori.decision_frameworks.ngt.models import (
    NGTExpert,
    NGTIdea,
    NGTResponse,
    Config
)
```

## Output Structure

The NGT analysis provides structured output including:
- Executive Summary
- Consolidated Ideas
- Implementation Plan
- Risk Mitigation
- Success Metrics
- Next Steps

## Documentation

For detailed documentation, visit our [GitHub repository](https://github.com/govind.shukl/agori).

## Examples

More examples are available in the [examples directory](https://github.com/govind.shukl/agori/tree/main/examples) of our GitHub repository.

## Contributing

We welcome contributions! Please visit our [GitHub repository](https://github.com/govind.shukl/agori) for contribution guidelines.

## Support

- Issue Tracker: [GitHub Issues](https://github.com/govind.shukl/agori/issues)
- Documentation: [GitHub Wiki](https://github.com/govind.shukl/agori/wiki)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/govind.shukl/agori/blob/main/LICENSE) file for details.

## Citation

```bibtex
@software{agori2024,
  title = {Agori: AI-Powered Decision Frameworks},
  year = {2024},
  url = {https://github.com/govind.shukl/agori}
}
```

## Changelog

See [CHANGELOG.md](https://github.com/govind.shukl/agori/blob/main/CHANGELOG.md) for a list of changes.