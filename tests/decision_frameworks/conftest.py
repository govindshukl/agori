from unittest.mock import Mock

import pytest

from agori.decision_frameworks.ngt.models import NGTExpert, NGTResponse


@pytest.fixture
def sample_expert():
    """Create a sample NGTExpert for testing."""
    return NGTExpert(
        role="Technical Architect",
        description="System design expert",
        specialty="Enterprise Architecture",
        metadata={"experience": "10+ years"},
    )


@pytest.fixture
def sample_response(sample_expert):
    """Create a sample NGTResponse for testing."""
    return NGTResponse(
        expert=sample_expert,
        ideas="# Key Ideas\nIdea 1\n# Technical Details\nDetail 1",
        metrics={"total_ideas": 1},
        metadata={"processing_time": 1.0},
    )


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        Mock(content="First chunk content", metadata={"index": 0}),
        Mock(content="Second chunk content", metadata={"index": 1}),
    ]


@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    return "This is a test context for decision making about technology implementation."


@pytest.fixture
def sample_query():
    """Create a sample query for testing."""
    return "What's the best approach for implementing this system?"
