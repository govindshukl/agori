from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class ContentWrapper:
    content: str


@dataclass(frozen=True)  # Make it immutable and hashable
class NGTExpert:
    """Expert for NGT process."""

    role: str
    description: str
    specialty: str
    metadata: Dict[str, Any] = field(
        default_factory=dict, hash=False
    )  # Exclude from hash


@dataclass
class Config:
    MAX_TOKENS_PER_REQUEST: int = 126000
    CHUNK_SIZE: int = 31500
    CHUNK_OVERLAP: int = 2000
    MAX_WORKERS: int = 3
    MAX_PARALLEL_REQUESTS: int = 3
    GPT4_COST_PER_1K_INPUT: float = 0.03
    GPT4_COST_PER_1K_OUTPUT: float = 0.06
    EMBEDDING_COST_PER_1K: float = 0.0001
    VERBOSE: bool = False


@dataclass
class NGTIdea:
    """Idea generated in NGT process."""

    content: str
    expert: NGTExpert
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NGTResponse:
    """Response from an expert in NGT process."""

    expert: NGTExpert
    ideas: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate response attributes."""
        if not isinstance(self.expert, NGTExpert):
            raise TypeError("expert must be an NGTExpert instance")
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary")
        if not isinstance(self.ideas, str):
            raise TypeError("ideas must be a string")
