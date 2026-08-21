import os
import sys
import uuid
import tempfile
from pathlib import Path

import streamlit as st

# Allow imports from the app/ package when run from the frontend/ directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from app.ingestion.loaders import load_document, UnsupportedFileTypeError
from app.utils.text_splitter import split_documents
from app.core.vectorstore import build_vectorstore, add_documents_to_vectorstore
from app.core.rag_chain import ask_question
from app.core.memory import memory_store
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Retrievault",
    page_icon="📚",
    layout="wide",
)

# --- Session state initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None



def build_langchain_history():
    """Convert Streamlit's chat_history into LangChain message objects."""
    messages = []
    for role, text, _ in st.session_state.chat_history:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages[-12:]

def process_uploaded_file(uploaded_file) -> dict:
    """
    Save the uploaded file to a temporary path, load and chunk it,
    then add it to the session's vector store.
    """
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        raw_docs = load_document(tmp_path)
        for doc in raw_docs:
            doc.metadata["source_file"] = uploaded_file.name  # use real filename, not temp path

        chunks = split_documents(raw_docs)

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = build_vectorstore(
                chunks, collection_name=st.session_state.session_id
            )
        else:
            add_documents_to_vectorstore(st.session_state.vectorstore, chunks)

        return {"filename": uploaded_file.name, "chunks_created": len(chunks)}

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --- Sidebar: document upload ---
with st.sidebar:
    st.title("📚 Retrievault")
    st.caption("Document-grounded Q&A")

    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Supported: PDF, DOCX, TXT, XLSX",
        type=["pdf", "docx", "txt", "xlsx"],
    )

    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    result = process_uploaded_file(uploaded_file)
                    st.session_state.documents_uploaded.append(uploaded_file.name)
                    st.success(
                        f"Indexed '{result['filename']}' "
                        f"({result['chunks_created']} chunks)"
                    )
                except UnsupportedFileTypeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.documents_uploaded:
        st.subheader("Indexed Documents")
        for doc_name in st.session_state.documents_uploaded:
            st.text(f"✓ {doc_name}")

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        memory_store.clear_session(st.session_state.session_id)
        st.session_state.chat_history = []
        st.rerun()


# --- Main panel: chat interface ---
st.header("Chat with your documents")

if not st.session_state.documents_uploaded:
    st.info("Upload a document from the sidebar to get started.")

for role, text, sources in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

question = st.chat_input("Ask a question about your documents...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and index a document first.")
    else:
        st.session_state.chat_history.append(("user", question, []))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = ask_question(
                        vectorstore=st.session_state.vectorstore,
                        question=question,
                        chat_history=build_langchain_history(),
                    )
                    sources = list({
                        doc.metadata.get("source_file", "unknown") for doc in source_docs
                    })

                    st.markdown(answer)
                    if sources:
                        st.caption(f"Sources: {', '.join(sources)}")

                    st.session_state.chat_history.append(("assistant", answer, sources))

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg, []))