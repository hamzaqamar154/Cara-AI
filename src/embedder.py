"""Embedding generation service using local models."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model = None
        if SentenceTransformer is not None:
            try:
                import os
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                logger.info("Loaded local embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.model = None

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = [t.strip() for t in texts if t.strip()]
        if not texts_list:
            return np.zeros((0, 384), dtype=np.float32)

        if self.model:
            return np.array(self.model.encode(texts_list, convert_to_numpy=True))

        raise RuntimeError(
            "No embedding backend available. Please install sentence-transformers: "
            "pip install sentence-transformers"
        )
