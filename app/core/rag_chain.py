from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from app.config import settings
from app.core.memory import memory_store


# System prompt: instructs the model to answer strictly from retrieved
# context, cite which source it drew from, and admit when it doesn't know.
SYSTEM_PROMPT = """You are DocuMind, a helpful assistant that answers \
questions strictly based on the provided document context.

Rules:
1. Only use information found in the "Context" section below to answer.
2. If the answer isn't in the context, say so clearly — do not guess \
or use outside knowledge.
3. When relevant, mention which source file the information came from.
4. Keep answers concise and directly relevant to the question.
5. If the user asks a follow-up question, use the conversation history \
to understand what they're referring to.
6. Do not use LaTeX formatting (no \\text{{}}, \\operatorname{{}}, or \\( \\) \
notation). Write mathematical expressions and formulas in plain readable \
text instead, e.g. "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O".

Context:
{context}
"""


def get_llm() -> ChatGroq:
    """Initialize the Groq-hosted LLM used for answer generation."""
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0.2,      # low temperature: favor factual, grounded answers
        max_tokens=1024,
    )


def format_docs(docs: List[Document]) -> str:
    """Format retrieved chunks into a single context string with source tags."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma):
    """
    Construct the RAG pipeline as a LangChain Runnable.

    Returns:
        A callable chain that accepts {"question": ..., "chat_history": [...]}
        and returns a string answer.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k_results},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = get_llm()

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(
    vectorstore: Chroma,
    question: str,
    session_id: str = "default",
) -> Tuple[str, List[Document]]:
    """
    High-level entry point: runs a question through the RAG chain,
    updates conversation memory, and returns the answer with sources.

    Args:
        vectorstore: The Chroma vector store to retrieve from.
        question: The user's natural language question.
        session_id: Identifier for the conversation session (for memory).

    Returns:
        Tuple of (answer_text, list of source Documents used).
    """
    chain, retriever = build_rag_chain(vectorstore)

    chat_history = memory_store.get_history(session_id)
    source_docs = retriever.invoke(question)

    answer = chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    memory_store.add_exchange(session_id, question, answer)

    return answer, source_docs