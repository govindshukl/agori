# Agori

Agori is a Python package that implements various decision-making frameworks powered by generative AI. It aims to enhance group decision-making processes by leveraging the capabilities of large language models.

## Currently Supported Frameworks

### 1. Nominal Group Technique (NGT)
The Nominal Group Technique is a structured decision-making method that helps groups reach consensus through a systematic process of idea generation and evaluation. Our implementation enhances this process using AI to:
- Generate diverse expert perspectives
- Analyze problems from multiple angles
- Synthesize insights into actionable recommendations

## Installation

```bash
pip install agori
```

## Quick Start

Here's a simple example using the NGT framework:

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

    # Add documents
    ngt.add_documents(pages)
    
    # Run analysis
    result = await ngt.run_ngt_analysis(query)
    return result

# Run analysis
document_path = "your_document.pdf"
query = "What strategic decisions should we make based on this document?"
result = await analyze_document(document_path, query)
```

## Features

### NGT Framework
- **Expert Role Generation**: Automatically identifies relevant expert perspectives based on the problem context
- **Multi-perspective Analysis**: Analyzes problems from different expert viewpoints
- **Parallel Processing**: Efficiently processes large documents using concurrent analysis
- **Structured Output**: Provides organized insights with clear recommendations
- **Configurable Parameters**: Allows customization of processing parameters
- **Detailed Logging**: Offers verbose mode for tracking the decision-making process

## Configuration

The NGT processor accepts various configuration parameters:

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

## Advanced Usage

### Step-by-Step Process

You can also run the NGT process step by step:

```python
# Generate experts
experts = await ngt.generate_experts(query)

# Generate expert responses
expert_responses = await ngt.analyze_with_experts(experts, query)

# Synthesize results
synthesis = await ngt.synthesize_ngt_results(expert_responses)
```

### Error Handling

The package includes custom exceptions for better error handling:

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
except NGTError as e:
    print(f"General NGT error: {e}")
```

## Dependencies

- langchain
- tiktoken
- azure-openai
- asyncio
- typing
- dataclasses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

## Project Status

This project is actively under development. The NGT framework is the first implementation, with more decision-making frameworks planned for future releases.

## Roadmap

- [ ] Additional decision frameworks
- [ ] Working memory for intermediate results caching
- [ ] Support for more LLM providers
- [ ] Integration with decision support tools


## Citation

If you use Agori in your research, please cite:

```bibtex
@software{agori2024,
  title = {Agori: AI-Powered Decision Frameworks},
  year = {2024},
  url = {https://github.com/govindshukl/agori}
}
```

## Contact

For questions and feedback, please open an issue on the GitHub repository.