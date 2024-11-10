"""Custom exceptions for the Agori package."""


class AgoriException(Exception):
    """Base exception for Agori package."""

    pass


class ConfigurationError(AgoriException):
    """Raised when there's an error in configuration."""

    pass


class ProcessingError(AgoriException):
    """Raised when there's an error processing documents."""

    pass


class SearchError(AgoriException):
    """Raised when there's an error during document search."""

    pass
