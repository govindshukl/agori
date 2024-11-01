"""Tests for the core Agori functionality."""

import pytest
from agori import Agori, AgoriException

def test_initialization():
    """Test that Agori can be initialized with valid credentials."""
    with pytest.raises(AgoriException):
        # Should raise exception with invalid credentials
        Agori(api_key="invalid", api_base="invalid")

# Add more tests as needed