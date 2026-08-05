import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from typing import List, Optional
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Lazily initialize and cache the HuggingFace embedding model.
    Using a local sentence-transformers model avoids embedding API
    costs/quotas entirely — only the LLM calls hit an external API.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def build_vectorstore(chunks: List[Document], collection_name: str = "documind") -> Chroma:
    """
    Build (or rebuild) a Chroma vector store from document chunks,
    persisting it to disk at settings.chroma_persist_dir.
    """
    embeddings = get_embedding_model()
    chunks = filter_complex_metadata(chunks)  # strip list/dict metadata Chroma can't store

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_dir,
    )

    return vectorstore


def load_vectorstore(collection_name: str = "documind") -> Chroma:
    """
    Load an existing persisted Chroma vector store from disk
    without re-embedding anything.

    Args:
        collection_name: Logical name of the collection to load.

    Returns:
        The Chroma vector store instance backed by existing data.
    """
    embeddings = get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents_to_vectorstore(vectorstore: Chroma, chunks: List[Document]) -> None:
    """
    Add new chunks to an existing vector store (incremental ingestion).
    """
    chunks = filter_complex_metadata(chunks)  # strip list/dict metadata Chroma can't store
    vectorstore.add_documents(chunks)