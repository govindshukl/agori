"""Utility functions for the Agori package."""

import logging
from pathlib import Path

from .exceptions import ProcessingError


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the package.

    Args:
        level: Logging level (default: logging.INFO)

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("agori")


def validate_file_path(file_path: str) -> Path:
    """Validate that the file exists and is a PDF.

    Args:
        file_path: Path to the file

    Returns:
        Path object of validated file

    Raises:
        ProcessingError: If file doesn't exist or isn't a PDF
    """
    path = Path(file_path)
    if not path.exists():
        raise ProcessingError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ProcessingError(f"File must be a PDF: {file_path}")
    return path
