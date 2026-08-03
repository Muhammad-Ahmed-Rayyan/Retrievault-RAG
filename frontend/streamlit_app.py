import uuid
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Retrievault",
    page_icon="📚",
    layout="wide",
)

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text, sources)
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = []


def upload_file_to_backend(uploaded_file) -> dict:
    """Send an uploaded file to the FastAPI /upload endpoint."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def send_chat_message(question: str) -> dict:
    """Send a question to the FastAPI /chat endpoint."""
    payload = {"question": question, "session_id": st.session_state.session_id}
    response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def clear_session():
    """Reset conversation memory both locally and on the backend."""
    requests.post(f"{API_BASE_URL}/session/{st.session_state.session_id}/clear", timeout=30)
    st.session_state.chat_history = []


# Sidebar: document upload
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
                    result = upload_file_to_backend(uploaded_file)
                    st.session_state.documents_uploaded.append(uploaded_file.name)
                    st.success(
                        f"Indexed '{result['filename']}' "
                        f"({result['chunks_created']} chunks)"
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"Upload failed: {e}")

    if st.session_state.documents_uploaded:
        st.subheader("Indexed Documents")
        for doc_name in st.session_state.documents_uploaded:
            st.text(f"✓ {doc_name}")

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        clear_session()
        st.rerun()


# Main panel: chat interface
st.header("Chat with your documents")

if not st.session_state.documents_uploaded:
    st.info("Upload a document from the sidebar to get started.")

# Render chat history
for role, text, sources in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.chat_history.append(("user", question, []))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = send_chat_message(question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")

                st.session_state.chat_history.append(("assistant", answer, sources))

            except requests.exceptions.RequestException as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg, []))