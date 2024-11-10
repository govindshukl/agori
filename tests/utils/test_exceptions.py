# agori/tests/utils/test_exceptions.py
"""Tests for custom exceptions."""

import pytest

from agori.utils.exceptions import AgoriException, ConfigurationError


def test_agori_exception():
    """Test base AgoriException."""
    with pytest.raises(AgoriException) as exc_info:
        raise AgoriException("Test error")
    assert str(exc_info.value) == "Test error"


def test_configuration_error():
    """Test ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("Invalid config")
    assert str(exc_info.value) == "Invalid config"
