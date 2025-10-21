"""Project configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get value from Streamlit secrets or environment variables."""
    if _STREAMLIT_AVAILABLE:
        try:
            if hasattr(st, "secrets") and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    
    load_dotenv()
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    groq_api_key: Optional[str]
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    vectorstore_path: Path
    metadata_store_path: Path
    api_base_url: str

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            groq_api_key=_get_secret("GROQ_API_KEY"),
            llm_model=_get_secret("LLM_MODEL", "llama-3.1-8b-instant"),
            chunk_size=int(_get_secret("CHUNK_SIZE", "800") or "800"),
            chunk_overlap=int(_get_secret("CHUNK_OVERLAP", "200") or "200"),
            vectorstore_path=VECTORSTORE_DIR / "faiss.index",
            metadata_store_path=VECTORSTORE_DIR / "metadata.pkl",
            api_base_url=_get_secret("API_BASE_URL", "https://api.groq.com/openai/v1"),
        )


settings = Settings.load()


def ensure_directories() -> None:
    for path in (VECTORSTORE_DIR, PROCESSED_DIR, RAW_DIR):
        path.mkdir(parents=True, exist_ok=True)
