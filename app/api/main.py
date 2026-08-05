import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingestion.loaders import load_document, UnsupportedFileTypeError
from app.utils.text_splitter import split_documents
from app.core.vectorstore import (
    build_vectorstore,
    load_vectorstore,
    add_documents_to_vectorstore,
)
from app.core.rag_chain import ask_question
from app.core.memory import memory_store

app = FastAPI(
    title="Retrievault RAG API",
    description="Document ingestion and retrieval-augmented Q&A API",
    version="1.0.0",
)

from app.core.vectorstore import get_embedding_model

@app.on_event("startup")
def preload_models():
    """Warm up the embedding model at server startup, not on first request."""
    print("Preloading embedding model...")
    get_embedding_model()
    print("Embedding model ready.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production use
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Tracks whether at least one document has been ingested this run
_state = {"vectorstore": None, "documents_ingested": 0}


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health_check():
    """Basic liveness check."""
    return {"status": "ok", "documents_ingested": _state["documents_ingested"]}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_docs = load_document(str(file_path))
        chunks = split_documents(raw_docs)

        if _state["vectorstore"] is None:
            _state["vectorstore"] = build_vectorstore(chunks)
        else:
            add_documents_to_vectorstore(_state["vectorstore"], chunks)

        _state["documents_ingested"] += 1

        return {
            "filename": file.filename,
            "chunks_created": len(chunks),
            "status": "indexed",
        }

    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Answer a question using the RAG pipeline over previously
    ingested documents.
    """
    if _state["vectorstore"] is None:
        # Try loading a previously persisted store from disk
        try:
            _state["vectorstore"] = load_vectorstore()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="No documents have been ingested yet. Upload a document first.",
            )

    try:
        answer, source_docs = ask_question(
            vectorstore=_state["vectorstore"],
            question=request.question,
            session_id=request.session_id,
        )
        sources = list({doc.metadata.get("source_file", "unknown") for doc in source_docs})
        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/session/{session_id}/clear")
def clear_session(session_id: str):
    """Clear conversation memory for a given session."""
    memory_store.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}