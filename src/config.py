"""Project configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


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
        load_dotenv()
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            vectorstore_path=VECTORSTORE_DIR / "faiss.index",
            metadata_store_path=VECTORSTORE_DIR / "metadata.pkl",
            api_base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
        )


settings = Settings.load()


def ensure_directories() -> None:
    for path in (VECTORSTORE_DIR, PROCESSED_DIR, RAW_DIR):
        path.mkdir(parents=True, exist_ok=True)
