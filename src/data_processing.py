"""PDF processing and text chunking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PyPDF2 import PdfReader

from .config import PROCESSED_DIR, RAW_DIR, ensure_directories, settings


@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    contents = []
    for page in reader.pages:
        contents.append(page.extract_text() or "")
    return "\n".join(contents)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    words = text.split()
    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def persist_chunks(chunks: Iterable[DocumentChunk], output_path: Path) -> None:
    serialized = [chunk.__dict__ for chunk in chunks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(serialized, fp, indent=2)


def process_pdf(pdf_file: Path) -> list[DocumentChunk]:
    ensure_directories()
    raw_target = RAW_DIR / pdf_file.name
    if pdf_file.resolve() != raw_target.resolve():
        raw_target.write_bytes(pdf_file.read_bytes())
    text = extract_text_from_pdf(raw_target)
    chunk_texts = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    chunks = [
        DocumentChunk(
            id=f"{raw_target.stem}_{idx}",
            text=chunk,
            source=str(raw_target),
        )
        for idx, chunk in enumerate(chunk_texts)
    ]
    persist_chunks(chunks, PROCESSED_DIR / f"{raw_target.stem}.json")
    return chunks
