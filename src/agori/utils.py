"""Utility functions for the Agori package."""

import logging
from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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


def split_document(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """Split a PDF document into chunks.

    Args:
        file_path: Path to the PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks

    Raises:
        ProcessingError: If document processing fails
    """
    try:
        path = validate_file_path(file_path)
        loader = PyPDFLoader(str(path))
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(pages)
        return [chunk.page_content for chunk in chunks]

    except Exception as e:
        raise ProcessingError(f"Failed to split document: {str(e)}")
