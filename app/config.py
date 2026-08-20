from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Central application configuration loaded from environment variables."""

    groq_api_key: str = ""
    gemini_api_key: str = ""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "/tmp/chroma_db"

    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    top_k_results: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure persistence directory exists
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)