"""
vector_store.py
----------------
Handles embedding generation (HuggingFace Sentence Transformers) and the
FAISS vector database: building a fresh index, saving it locally, loading
it back, and merging new documents into an existing index (so users can
keep adding PDFs without losing previously indexed ones).
"""

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# A small, fast, high-quality general-purpose sentence embedding model.
# Runs locally on CPU — no API key required for embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Instantiate (and cache via Streamlit's resource cache, see app.py) the
    HuggingFace embedding model used to vectorize document chunks and
    user queries.

    Returns:
        A HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: List[Document], embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """
    Build a brand new FAISS vector store from document chunks.

    Args:
        chunks: List of Document chunks to embed and index.
        embeddings: The embedding model to use.

    Returns:
        A FAISS vector store containing the embedded chunks.
    """
    if not chunks:
        raise ValueError("No document chunks provided to build the vector store.")

    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(vector_store: FAISS, index_path: str) -> None:
    """
    Persist a FAISS index to local disk so it can be reloaded across
    sessions without re-embedding everything.

    Args:
        vector_store: The FAISS vector store to save.
        index_path: Directory path where the index files will be written.
    """
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)


def load_vector_store(
    index_path: str, embeddings: HuggingFaceEmbeddings
) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk, if it exists.

    Args:
        index_path: Directory path where the index files are stored.
        embeddings: The embedding model (must match the one used to build
            the index, since FAISS itself stores no info about it).

    Returns:
        The loaded FAISS vector store, or None if no index exists at the
        given path.
    """
    index_file = os.path.join(index_path, "index.faiss")
    if not os.path.exists(index_file):
        return None

    # allow_dangerous_deserialization=True is required by LangChain because
    # FAISS indexes are loaded via pickle. This is safe here because we are
    # only ever loading an index that *this application* previously wrote.
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )


def add_documents_to_store(vector_store: FAISS, chunks: List[Document]) -> FAISS:
    """
    Merge new document chunks into an existing FAISS vector store, enabling
    incremental indexing when a user uploads additional PDFs.

    Args:
        vector_store: The existing FAISS vector store.
        chunks: New Document chunks to add.

    Returns:
        The updated FAISS vector store (mutated in place and returned for
        convenience).
    """
    vector_store.add_documents(chunks)
    return vector_store


def similarity_search(
    vector_store: FAISS, query: str, k: int = 4
) -> List[Document]:
    """
    Retrieve the top-k most relevant document chunks for a given query
    using FAISS similarity search.

    Args:
        vector_store: The FAISS vector store to search.
        query: The user's natural language question.
        k: Number of top relevant chunks to retrieve.

    Returns:
        List of the k most similar Document chunks.
    """
    return vector_store.similarity_search(query, k=k)


def clear_vector_store(index_path: str) -> None:
    """Delete the saved FAISS index files from disk and reset the directory."""
    if os.path.exists(index_path):
        import shutil
        shutil.rmtree(index_path, ignore_errors=True)
    os.makedirs(index_path, exist_ok=True)


def get_indexed_filenames(vector_store: Optional[FAISS]) -> List[str]:
    """Extract unique source file names from a FAISS vector store's docstore."""
    if vector_store is None:
        return []
    try:
        filenames = set()
        for doc in vector_store.docstore._dict.values():
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                filenames.add(doc.metadata["source"])
            elif isinstance(doc, dict) and "metadata" in doc and "source" in doc["metadata"]:
                filenames.add(doc["metadata"]["source"])
        return sorted(list(filenames))
    except Exception:
        return []

