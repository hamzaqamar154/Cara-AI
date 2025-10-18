# Architecture

## Pipeline

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Storage
                                                           ↓
Query → Embed Query → Search FAISS → Truncate Context → Groq LLM → Answer
```

## Components

**Data Processing** (`src/data_processing.py`)
- Extracts text from PDFs using PyPDF2
- Splits text into overlapping chunks (800 chars, 200 overlap default)
- Saves chunks as JSON

**Embedding Service** (`src/embedder.py`)
- Uses local SentenceTransformers model (all-MiniLM-L6-v2)
- Generates 384-dimensional vectors
- No external API calls

**Vector Store** (`src/retriever.py`)
- FAISS IndexFlatL2 for similarity search
- Stores embeddings and metadata separately
- Deduplicates search results

**LLM Service** (`src/llm.py`)
- Groq API integration via OpenAI-compatible client
- Truncates context to fit token limits
- Falls back to keyword matching if API unavailable

**UI** (`ui/app.py`)
- Streamlit interface for upload and query
- Displays answers with reference and supporting passages

## Token Management

Groq free tier: 6000 tokens/minute

The system limits context to stay under this:
- Max 4 chunks sent to LLM
- Each chunk truncated to 400 characters
- Total context ~2500 tokens
- Leaves room for prompt and response

## API

- `GET /health` - Status check
- `POST /upload` - Upload PDF, returns chunk count
- `POST /embed` - Embed text chunks directly
- `POST /query` - Query with `{"question": "...", "k": 3}`, returns answer and context

## Deployment

Backend: `uvicorn api.app:app --reload --host 127.0.0.1 --port 8000`
UI: `streamlit run ui/app.py --server.port 8501`

Keep `vectorstore/` directory to persist data between deployments.
