"""Streamlit UI for document assistant."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st


def _resolve_project_root() -> Path:
    try:
        possible_roots = list(Path(__file__).resolve().parents)
        for candidate in possible_roots:
            if (candidate / "src").exists():
                return candidate
        cwd = Path.cwd()
        if (cwd / "src").exists():
            return cwd
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd()


try:
    ROOT = _resolve_project_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except Exception:
    ROOT = Path.cwd()

from src.config import RAW_DIR, ensure_directories
from src.data_processing import process_pdf
from src.llm import LLMService
from src.retriever import VectorStore


@st.cache_resource
def get_vector_store():
    ensure_directories()
    return VectorStore()


@st.cache_resource
def get_llm():
    return LLMService()

STYLES = """
<style>
    .header {
        background: #3C6E71;
        color: white;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 2rem;
    }
    .header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 500;
    }
    .header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .answer-box {
        background: #FAFAFA;
        border-left: 4px solid #3C6E71;
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    .answer-title {
        color: #284B63;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .answer-content {
        color: #2d3748;
        line-height: 1.7;
        font-size: 1rem;
        white-space: pre-wrap;
    }
    .reference-box {
        background: #FAFAFA;
        border-left: 4px solid #F2A365;
        border-radius: 6px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .reference-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #284B63;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .reference-text {
        color: #4a5568;
        line-height: 1.6;
        font-size: 0.9rem;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .reference-meta {
        font-size: 0.75rem;
        color: #718096;
    }
    .passages-section {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #D9D9D9;
    }
    .passage-card {
        background: white;
        border: 1px solid #D9D9D9;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .passage-header {
        font-weight: 600;
        color: #284B63;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    .passage-text {
        color: #4a5568;
        line-height: 1.6;
        font-size: 0.9rem;
    }
</style>
"""


def determine_top_k(store: VectorStore) -> int:
    total_chunks = len(getattr(store, "metadata", []))
    if total_chunks == 0:
        return 3
    return min(5, max(3, total_chunks // 20 + 2))


st.set_page_config(page_title="Cara AI", layout="wide")
st.markdown(STYLES, unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1 style="color: white; font-size: 2.5rem; font-weight: 600; text-align: center;">Cara AI - Document Assistant</h1>
    <p style="color: white; font-size: 1.2rem; font-weight: 400; text-align: center;">By Mirza Noor Hamza</p>
    <p style="color: white; font-size: 1.2rem; font-weight: 400; text-align: center;" class="description">Query your documents using RAG and Groq's LLM API</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Upload Document")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded:
        with st.spinner("Processing..."):
            temp_path = RAW_DIR / uploaded.name
            temp_path.write_bytes(uploaded.getvalue())
            chunks = process_pdf(temp_path)
            vector_store = get_vector_store()
            vector_store.add_documents(chunks)
            st.success(f"Embedded {len(chunks)} chunks from {uploaded.name}")
    
    st.markdown("---")
    st.markdown("### Stored Data")
    vector_store = get_vector_store()
    total_chunks = len(vector_store.metadata)
    if total_chunks > 0:
        st.info(f"{total_chunks} document chunks stored")
        
        if st.button("Clear All Data", use_container_width=True, type="secondary"):
            vector_store.index = None
            vector_store.metadata = []
            if vector_store.index_path.exists():
                vector_store.index_path.unlink()
            if vector_store.metadata_path.exists():
                vector_store.metadata_path.unlink()
            st.success("Data cleared")
            st.rerun()
    else:
        st.info("No documents stored")

st.markdown("### Ask a Question")
query = st.text_input(
    "Enter your question:",
    placeholder="What is this document about?",
    key="query_input",
    label_visibility="collapsed"
)

vector_store = get_vector_store()
llm = get_llm()
auto_k = determine_top_k(vector_store)

if st.button("Get Answer", type="primary", use_container_width=True) and query:
    with st.spinner("Searching documents..."):
        results = vector_store.search(query, k=auto_k)
        context = [chunk.text for chunk, _ in results]
        answer = llm.generate_answer(query, context)
    
    st.markdown("""
    <div class="answer-box">
        <div class="answer-title">Answer</div>
        <div class="answer-content">{}</div>
    </div>
    """.format(answer.replace("\n", "<br>").replace('"', '&quot;')), unsafe_allow_html=True)
    
    if results:
        top_chunk, top_distance = results[0]
        reference_text = top_chunk.text
        if len(reference_text) > 300:
            reference_text = reference_text[:300].rsplit(' ', 1)[0] + "..."
        
        source_name = Path(top_chunk.source).name
        reference_html = f"""
        <div class="reference-box">
            <div class="reference-label">Reference Passage</div>
            <div class="reference-text">"{html.escape(reference_text)}"</div>
            <div class="reference-meta">
                Source: {html.escape(source_name)} | Relevance: {top_distance:.3f}
            </div>
        </div>
        """
        st.markdown(reference_html, unsafe_allow_html=True)
    
    if results:
        st.markdown('<div class="passages-section">', unsafe_allow_html=True)
        st.markdown("### Supporting Passages")
        st.caption(f"{len(results)} relevant passages found")
        
        for idx, (chunk, distance) in enumerate(results, start=1):
            source_name = Path(chunk.source).name
            passage_html = f"""
            <div class="passage-card">
                <div class="passage-header">
                    Passage {idx} | Relevance: {distance:.3f} | Source: {source_name}
                </div>
                <div class="passage-text">
                    {chunk.text}
                </div>
            </div>
            """
            st.markdown(passage_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No supporting passages found. Upload a document first.")

elif query:
    st.info("Click 'Get Answer' to search your documents")
