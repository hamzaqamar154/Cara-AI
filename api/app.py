"""FastAPI backend for document assistant."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import RAW_DIR, ensure_directories
from src.data_processing import DocumentChunk, process_pdf
from src.llm import LLMService
from src.retriever import VectorStore

app = FastAPI(title="Document Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_directories()
vector_store = VectorStore()
llm_service = LLMService()


class EmbedRequest(BaseModel):
    chunks: list[str]
    source: str = "api"


class QueryRequest(BaseModel):
    question: str
    k: int = 3


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    dest = RAW_DIR / file.filename
    dest.write_bytes(await file.read())
    chunks = process_pdf(dest)
    vector_store.add_documents(chunks)
    return {"message": "Uploaded and embedded", "chunks": len(chunks)}


@app.post("/embed")
def embed_text(request: EmbedRequest) -> dict:
    chunks = [
        DocumentChunk(id=f"{request.source}_{idx}", text=text, source=request.source)
        for idx, text in enumerate(request.chunks)
    ]
    vector_store.add_documents(chunks)
    return {"message": "Text embedded", "chunks": len(chunks)}


@app.post("/query")
def query_documents(request: QueryRequest) -> dict:
    results = vector_store.search(request.question, k=request.k)
    context = [chunk.text for chunk, _ in results]
    answer = llm_service.generate_answer(request.question, context)
    return {
        "answer": answer,
        "context": context,
        "results": [
            {"chunk_id": chunk.id, "distance": distance, "source": chunk.source}
            for chunk, distance in results
        ],
    }
