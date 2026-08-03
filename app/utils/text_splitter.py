from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks suitable for embedding.

    Uses RecursiveCharacterTextSplitter, which tries to split on
    paragraph/sentence boundaries first before falling back to
    hard character limits — this keeps chunks semantically coherent.

    Args:
        documents: Raw Document objects from the loaders.

    Returns:
        List of chunked Document objects, each carrying a chunk_id
        in its metadata for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks