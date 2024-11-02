"""Agori - A secure wrapper for ChromaDB with Azure OpenAI embeddings."""

from .core import SecureChromaDB
from .exceptions import AgoriException, ConfigurationError, ProcessingError, SearchError

__version__ = "0.1.0"
__all__ = [
    "SecureChromaDB",
    "AgoriException",
    "ConfigurationError",
    "ProcessingError",
    "SearchError",
]
