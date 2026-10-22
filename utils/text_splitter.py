"""
text_splitter.py
-----------------
Splits large Document objects (full PDF pages) into smaller overlapping
chunks suitable for embedding and retrieval. Smaller chunks improve
retrieval precision; overlap preserves context across chunk boundaries.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default chunking configuration. These can be tuned based on document type.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Document objects into smaller chunks while preserving
    their original metadata (source filename, page number).

    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs -> sentences -> words) before falling back to
    a hard character cut, which keeps chunks semantically coherent.

    Args:
        documents: List of Document objects (e.g. one per PDF page).
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive
            chunks, to avoid losing context at chunk boundaries.

    Returns:
        List of smaller Document chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with a sequential id within its source — useful for
    # debugging and for stable citation display.
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks
