import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / os.getenv("CHROMA_DB_PATH", "chroma_db")

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

class Settings:
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    # Embeddings Settings
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Re-ranker Settings
    RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "local").lower()
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Vector DB
    CHROMA_DB_PATH = str(DB_DIR)

    # Text Chunking Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval Settings
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "4"))

    @classmethod
    def validate(cls):
        """Validates that necessary API keys are present based on LLM configuration."""
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY is not set but openai is selected as LLM provider.")
        elif cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY is not set but gemini is selected as LLM provider.")
        elif cls.LLM_PROVIDER in ["xai", "grok"] and not cls.XAI_API_KEY:
            print("WARNING: XAI_API_KEY is not set but xai/grok is selected as LLM provider.")
        
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY is not set but openai is selected as embedding provider.")
        elif cls.EMBEDDING_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY is not set but gemini is selected as embedding provider.")
