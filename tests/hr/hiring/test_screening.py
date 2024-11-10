# agori/tests/hr/hiring/test_screening.py
"""Tests for candidate screening functionality."""

from unittest.mock import Mock, patch

import pytest

from agori.core.db import WorkingMemory
from agori.hr.hiring.screening import CandidateScreening


@pytest.fixture
def mock_db():
    """Fixture to provide a mocked WorkingMemory instance."""
    return Mock(spec=WorkingMemory)


@pytest.fixture
def screening():
    """Fixture to provide a CandidateScreening instance."""
    with patch("agori.core.db.WorkingMemory") as mock_db:
        return CandidateScreening(mock_db)


# Placeholder for test cases
def test_screening_initialization(screening):
    """Test successful initialization of CandidateScreening."""
    assert screening is not None
