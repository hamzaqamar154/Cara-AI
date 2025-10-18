"""
Lightweight tests covering chunking and retrieval wiring.
"""

from pathlib import Path

import numpy as np

from src.data_processing import DocumentChunk, chunk_text
from src.retriever import EmbeddingService, VectorStore


def test_chunk_text_produces_overlap() -> None:
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert any("5" in chunk for chunk in chunks)


class DummyEmbedder(EmbeddingService):
    def __init__(self) -> None:
        self.dim = 8

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 997
            vec = np.array([(seed + i) % 97 for i in range(self.dim)], dtype=np.float32)
            vectors.append(vec)
        return np.vstack(vectors)


def test_vector_store_add_and_search(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.retriever.EmbeddingService", DummyEmbedder)
    store = VectorStore(
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "metadata.pkl",
    )
    chunks = [
        DocumentChunk(id="one", text="rapid brown fox", source="test"),
        DocumentChunk(id="two", text="slow blue whale", source="test"),
    ]
    store.add_documents(chunks)
    results = store.search("brown")
    assert results
