"""CLI interface for document assistant using Groq LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ensure_directories
from .data_processing import process_pdf
from .llm import LLMService
from .retriever import VectorStore


def embed_pdf(pdf_path: Path) -> None:
    """Process and embed a PDF file into the vector store."""
    chunks = process_pdf(pdf_path)
    store = VectorStore()
    store.add_documents(chunks)
    print(f"Embedded {len(chunks)} chunks from {pdf_path}")


def answer_query(query: str, top_k: int = 3) -> str:
    store = VectorStore()
    results = store.search(query, k=top_k)
    llm = LLMService()
    context = [chunk.text for chunk, _ in results]
    return llm.generate_answer(query, context)


def main() -> None:
    ensure_directories()
    parser = argparse.ArgumentParser(description="Intelligent Document Assistant")
    parser.add_argument("--pdf", type=Path, help="Path to PDF to embed")
    parser.add_argument("--query", type=str, help="Question to ask the assistant")
    args = parser.parse_args()

    if args.pdf:
        embed_pdf(args.pdf)
    if args.query:
        print(answer_query(args.query))


if __name__ == "__main__":
    main()
