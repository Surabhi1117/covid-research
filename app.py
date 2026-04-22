import streamlit as st
import json
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pypdf import PdfReader
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="COVID-19 Research Platform", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .search-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; color: black; }
    .paper-title { color: #0d6efd; font-size: 1.2rem; font-weight: bold; text-decoration: none; }
    .paper-meta { font-size: 0.85rem; color: #6c757d; margin-bottom: 10px; }
    .score-badge { background-color: #e9ceec; color: #6f42c1; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CACHED AI MODELS ---
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-1.5-flash')
    else:
        gemini = None
    return embed_model, gemini

embed_model, gemini = load_models()

# --- DATA & INDEXING ---
@st.cache_data
def load_papers():
    with open('papers.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def get_vector_db(_papers):
    client = chromadb.Client()
    collection = client.create_collection(name="papers")
    texts = [p['abstract'] for p in _papers if p.get('abstract')]
    ids = [p['cord_uid'] for p in _papers if p.get('abstract')]
    metadatas = [{"title": p['title'], "doi": p.get('doi', ''), "url": p.get('url', '')} for p in _papers if p.get('abstract')]
    embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection

papers_data = load_papers()
vector_index = get_vector_db(papers_data)
papers_dict = {p['cord_uid']: p for p in papers_data}

# --- UI LOGIC ---
st.sidebar.title("🧬 COVID-19 AI Research")
if st.sidebar.button("🔍 Search"): st.session_state.mode = "search"
if st.sidebar.button("📝 Draft"): st.session_state.mode = "draft"
if st.sidebar.button("📁 Analyze PDF"): st.session_state.mode = "upload"

if "mode" not in st.session_state: st.session_state.mode = "search"

if st.session_state.mode == "search":
    st.title("Search Engine")
    query = st.text_input("Ask a scientific question")
    if query:
        query_emb = embed_model.encode([query]).tolist()
        sem = vector_index.query(query_embeddings=query_emb, n_results=10)
        for i in range(len(sem['ids'][0])):
            uid = sem['ids'][0][i]
            p = papers_dict.get(uid)
            if p:
                st.markdown(f'<div class="search-card"><a class="paper-title" href="{p.get("url") or "#"}" target="_blank">{p["title"]}</a><br><p>{p["abstract"][:300]}...</p></div>', unsafe_allow_html=True)
                if st.button("Analyze with Gemini", key=uid):
                    res = gemini.generate_content(f"Summarize: {p['abstract']}")
                    st.info(res.text)

elif st.session_state.mode == "draft" and gemini:
    st.title("Research Drafter")
    topic = st.text_input("Topic")
    if topic and st.button("Generate"):
        res = gemini.generate_content(f"Draft a research paper on: {topic}")
        st.markdown(res.text)

elif st.session_state.mode == "upload" and gemini:
    st.title("PDF Analysis")
    f = st.file_uploader("Upload", type="pdf")
    if f and st.button("Analyze"):
        text = "".join([p.extract_text() for p in PdfReader(f).pages])[:30000]
        res = gemini.generate_content(f"Summarize this paper: {text}")
        st.markdown(res.text)
