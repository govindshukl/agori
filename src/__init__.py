"""Agori - A user-friendly wrapper for ChromaDB with Azure OpenAI embeddings."""

from .core import Agori
from .exceptions import AgoriException

__version__ = "0.1.0"
__all__ = ["Agori", "AgoriException"]