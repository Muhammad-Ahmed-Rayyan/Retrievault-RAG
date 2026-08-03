from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ConversationMemoryStore:
    """
    In-memory session store mapping session_id -> chat history.

    For a production system this would be backed by Redis or a database,
    but in-process memory is sufficient for a single-user demo/portfolio app.
    """

    def __init__(self, max_turns: int = 6):
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self.max_turns = max_turns  # number of Q&A pairs to retain

    def get_history(self, session_id: str) -> List[BaseMessage]:
        return self._sessions.get(session_id, [])

    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        """Append a Q&A turn and trim history to the max window size."""
        history = self._sessions.setdefault(session_id, [])
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        # Keep only the last `max_turns` exchanges (2 messages per turn)
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            self._sessions[session_id] = history[-max_messages:]

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Singleton store shared across the app's lifetime
memory_store = ConversationMemoryStore()