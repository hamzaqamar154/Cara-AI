"""FAISS vector store for document retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from .config import ensure_directories, settings
from .data_processing import DocumentChunk
from .embedder import EmbeddingService


class VectorStore:
    def __init__(
        self,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        ensure_directories()
        self.index_path = index_path or settings.vectorstore_path
        self.metadata_path = metadata_path or settings.metadata_store_path
        self.embedding_service = EmbeddingService()
        self.metadata: List[DocumentChunk] = []
        self.index: faiss.IndexFlatL2 | None = None
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            with self.metadata_path.open("rb") as fp:
                self.metadata = pickle.load(fp)

    def _persist(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("wb") as fp:
            pickle.dump(self.metadata, fp)

    def add_documents(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedding_service.embed(chunk.text for chunk in chunks)
        if embeddings.size == 0:
            return
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        self._persist()

    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None:
            return []
        query_vec = self.embedding_service.embed([query])
        if query_vec.size == 0:
            return []
        
        distances, indices = self.index.search(query_vec, k * 2)
        results: List[Tuple[DocumentChunk, float]] = []
        seen_texts = set()
        seen_ids = set()
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx]
            
            if chunk.id in seen_ids:
                continue
            
            text_normalized = " ".join(chunk.text.split())
            if text_normalized in seen_texts:
                continue
            
            seen_ids.add(chunk.id)
            seen_texts.add(text_normalized)
            results.append((chunk, float(dist)))
            
            if len(results) >= k:
                break
        
        return results
