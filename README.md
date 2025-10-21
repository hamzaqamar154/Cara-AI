# Cara AI - Document Assistant

RAG system for querying PDF documents. Extracts text, generates embeddings, stores them in FAISS, and answers questions using Groq's LLM API.

## Technologies

- **Python** - Core language
- **Streamlit** - Web UI
- **FastAPI** - REST API backend
- **Groq API** - LLM inference (llama-3.1-8b-instant)
- **SentenceTransformers** - Local embeddings (all-MiniLM-L6-v2)
- **FAISS** - Vector similarity search
- **PyPDF2** - PDF text extraction

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure Groq API key in `.env`:
   ```
   GROQ_API_KEY=your_key_here
   API_BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=llama-3.1-8b-instant
   ```

   Get your API key from https://console.groq.com/keys

   Without the API key, the system falls back to offline mode using keyword matching.

## Usage

**Web UI:**
```bash
streamlit run ui/app.py --server.port 8501
```

**API:**
```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

**CLI:**
```bash
python -m src.main --pdf data/raw/sample.pdf
python -m src.main --query "What is this about?"
```

Note: The UI is intentionally minimal. Focus is on the RAG logic and functionality rather than design polish.

## How it works

1. PDFs are uploaded and text is extracted
2. Text is chunked (800 chars, 200 overlap by default)
3. Chunks are embedded using local SentenceTransformers
4. Embeddings stored in FAISS with metadata
5. Queries are embedded and matched against stored chunks
6. Top 3-5 chunks are sent to Groq LLM for answer generation
7. Context is automatically truncated to fit token limits

## Configuration

Environment variables:
- `GROQ_API_KEY` - Required for AI answers
- `LLM_MODEL` - Groq model (default: llama-3.1-8b-instant)
- `CHUNK_SIZE` - Chunk size in characters (default: 800)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)

## Token limits

Groq free tier has a 6000 tokens/minute limit. The system automatically:
- Limits to 4 chunks max
- Truncates each chunk to 400 characters
- Keeps total context under ~2500 tokens

## API endpoints

- `GET /health` - Health check
- `POST /upload` - Upload and process PDF
- `POST /embed` - Embed text chunks
- `POST /query` - Query documents

API docs at `http://127.0.0.1:8000/docs`

## Deployment

For Streamlit Cloud deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).

Quick steps:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set main file path: `ui/app.py`
4. Add secrets in Streamlit Cloud settings:
   - `GROQ_API_KEY`
   - `LLM_MODEL` (optional, defaults to llama-3.1-8b-instant)
   - `API_BASE_URL` (optional, defaults to https://api.groq.com/openai/v1)

---

Author: Mirza Noor Hamza
