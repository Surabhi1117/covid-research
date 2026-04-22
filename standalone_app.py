import streamlit as st
import json
import os
import torch
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import google.generativeai as genai
import spacy
from typing import List, Dict, Any
from pypdf import PdfReader
import io
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
ST_PORT = 8501
MODEL_NAME = "all-MiniLM-L6-v2"
SUMM_MODEL = "facebook/bart-large-cnn"
NER_MODEL = "en_ner_bc5cdr_md"

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
    embed_model = SentenceTransformer(MODEL_NAME, device=device)
    summ_pipe = pipeline("summarization", model=SUMM_MODEL, device=0 if torch.cuda.is_available() else -1)
    try:
        nlp = spacy.load(NER_MODEL)
    except:
        nlp = None
    
    # Setup Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-1.5-flash-8b')
    else:
        gemini = None
        
    return embed_model, summ_pipe, nlp, gemini

import time
def safe_generate(model, prompt):
    """Generate content with automatic retries for rate limits."""
    for attempt in range(3):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise e

embed_model, summ_pipe, nlp, gemini = load_models()

# --- DATA & INDEXING ---
@st.cache_data
def load_papers():
    with open('papers.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def get_vector_db(_papers):
    client = chromadb.Client() # In-memory for Cloud
    collection = client.create_collection(name="papers")
    
    texts = [p['abstract'] for p in _papers if p.get('abstract')]
    ids = [p['cord_uid'] for p in _papers if p.get('abstract')]
    metadatas = [{"title": p['title'], "doi": p.get('doi', ''), "url": p.get('url', '')} for p in _papers if p.get('abstract')]
    
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection

papers_data = load_papers()
vector_index = get_vector_db(papers_data)
papers_dict = {p['cord_uid']: p for p in papers_data}

# --- SEARCH LOGIC ---
def hybrid_search(query, limit=10):
    # Semantic Search
    query_emb = embed_model.encode([query]).tolist()
    sem_results = vector_index.query(query_embeddings=query_emb, n_results=limit*2)
    
    results = []
    for i in range(len(sem_results['ids'][0])):
        uid = sem_results['ids'][0][i]
        paper = papers_dict.get(uid)
        if paper:
            results.append({
                **paper,
                "score": 1.0 / (1.0 + sem_results['distances'][0][i])
            })
    return results[:limit]

# --- UI LOGIC ---
st.sidebar.title("🧬 COVID-19 AI Research")
st.sidebar.info("Self-Contained Deployment Version")

st.sidebar.divider()
st.sidebar.subheader("Tools")
if st.sidebar.button("🔍 Search Papers"): st.session_state.app_mode = "search"
if st.sidebar.button("📝 Draft Research Paper"): st.session_state.app_mode = "draft"
if st.sidebar.button("✍️ Paraphraser"): st.session_state.app_mode = "paraphrase"
if st.sidebar.button("📁 Upload & Analyze PDF"): st.session_state.app_mode = "upload"

if "app_mode" not in st.session_state: st.session_state.app_mode = "search"

if st.session_state.app_mode == "search":
    st.title("Search & Retrieval Engine")
    query = st.text_input("Ask a scientific question", placeholder="e.g., vaccine acceptance in Bangladesh")
    
    if query:
        results = hybrid_search(query)
        st.subheader(f"Found {len(results)} relevant papers")
        
        if st.checkbox("💡 Generate AI Answer", value=True) and gemini:
            context = "\n".join([f"- {p['title']}: {p['abstract'][:300]}" for p in results[:5]])
            prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer concisely with citations."
            try:
                answer = gemini.generate_content(prompt)
                st.info(answer.text)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    st.error("Error: The AI model 'gemini-flash-latest' was not found. Please check your API key and model access.")
                else:
                    st.error(f"AI Generation Error: {e}")
            
        for res in results:
            with st.container():
                st.markdown(f'<div class="search-card">', unsafe_allow_html=True)
                url = res.get('url') or (f"https://doi.org/{res['doi']}" if res.get('doi') else "#")
                st.markdown(f'<a class="paper-title" href="{url}" target="_blank">{res["title"]}</a>', unsafe_allow_html=True)
                st.markdown(f'<div class="paper-meta">DOI: {res.get("doi", "N/A")} | <span class="score-badge">Score: {res["score"]:.2f}</span></div>', unsafe_allow_html=True)
                if st.button("View Details", key=res['cord_uid']):
                    st.session_state.selected_paper = res['cord_uid']
                    st.rerun()
                st.write(res['abstract'][:300] + "...")
                st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.app_mode == "draft" and gemini:
    st.title("AI Research Paper Drafter")
    topic = st.text_input("Enter topic")
    if topic and st.button("Generate Draft"):
        results = hybrid_search(topic, limit=10)
        context = "\n".join([f"[{i}] {p['title']}: {p['abstract']}" for i, p in enumerate(results, 1)])
        prompt = f"Write a full research paper draft on '{topic}' using this context:\n{context}"
        try:
            draft = gemini.generate_content(prompt)
            st.markdown(draft.text)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                st.error("Error: The AI model 'gemini-flash-latest' was not found. Please check your API key and model access.")
            else:
                st.error(f"AI Generation Error: {e}")

elif st.session_state.app_mode == "upload" and gemini:
    st.title("PDF Analysis")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Analyze"):
        reader = PdfReader(uploaded_file)
        text = "".join([p.extract_text() for p in reader.pages])[:30000]
        prompt = f"Summarize and list key points for this paper:\n{text}"
        try:
            analysis = gemini.generate_content(prompt)
            st.markdown(analysis.text)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                st.error("Error: The AI model 'gemini-flash-latest' was not found. Please check your API key and model access.")
            else:
                st.error(f"AI Generation Error: {e}")

elif st.session_state.app_mode == "paraphrase":
    st.title("🚀 QuillBot-style AI Paraphraser")
    st.write("Professional-grade rewriting tool for scientific and academic text.")
    
    col_ctrl1, col_ctrl2 = st.columns([0.7, 0.3])
    with col_ctrl1:
        mode = st.selectbox("Select Mode", ["Standard", "Fluency", "Formal", "Simple", "Creative", "Expand", "Shorten"])
    with col_ctrl2:
        intensity = st.select_slider("Synonym Intensity", options=["Low", "Medium", "High"], value="Medium")
    
    text_to_paraphrase = st.text_area("Original Text", height=200, placeholder="Paste your text here...")
    
    if text_to_paraphrase:
        if st.button("Paraphrase Text"):
            with st.spinner(f"Paraphrasing in {mode} mode..."):
                try:
                    # In this self-contained context, we call the model directly
                    intensity_map = {"Low": "minimal changes to vocabulary", "Medium": "moderate use of synonyms", "High": "significant vocabulary overhaul"}
                    prompt = f"""
                    TASK: Paraphrase the text below.
                    MODE: {mode}
                    INTENSITY: {intensity} ({intensity_map[intensity]})
                    INSTRUCTIONS: Maintain 100% scientific accuracy, change sentence structure significantly, and provide TWO versions.
                    TEXT: {text_to_paraphrase}
                    """
                    response = safe_generate(gemini, prompt)
                    st.success("Paraphrasing complete!")
                    st.divider()
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.subheader("Original")
                        st.info(text_to_paraphrase)
                    with col_res2:
                        st.subheader(f"Paraphrased ({mode})")
                        st.success(response.text)
                        st.download_button("Download Result", response.text, file_name=f"paraphrased_{mode}.md")
                except Exception as e:
                    st.error(f"Error: {e}")

if "selected_paper" in st.session_state:
    p = papers_dict[st.session_state.selected_paper]
    st.divider()
    st.header(p['title'])
    if st.button("Close"): 
        del st.session_state.selected_paper
        st.rerun()
    
    with st.spinner("Summarizing..."):
        summ = summ_pipe(p['abstract'][:4000], max_length=150, min_length=40)[0]['summary_text']
        st.success(f"### AI Summary\n{summ}")
    
    st.write(f"**Authors:** {p['authors']}")
    st.write(f"**Abstract:** {p['abstract']}")
