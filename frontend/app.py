import streamlit as st
import httpx
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="COVID-19 Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0d6efd;
        color: white;
    }
    .search-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .paper-title {
        color: #0d6efd;
        font-size: 1.2rem;
        font-weight: bold;
        text-decoration: none;
    }
    .paper-title:hover {
        text-decoration: underline;
    }
    .paper-meta {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 10px;
    }
    .score-badge {
        background-color: #e9ceec;
        color: #6f42c1;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

def safe_api_call(method, endpoint, **kwargs):
    try:
        # Set a longer timeout for AI processing (60 seconds)
        timeout = httpx.Timeout(60.0, connect=10.0)
        if method == "GET":
            response = httpx.get(f"{API_URL}{endpoint}", timeout=timeout, **kwargs)
        else:
            response = httpx.post(f"{API_URL}{endpoint}", timeout=timeout, **kwargs)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def search_papers(query, mode, limit=10):
    return safe_api_call("GET", "/search", params={"q": query, "mode": mode, "limit": limit})

def ask_question(question, paper_ids=None):
    params = {"question": question}
    if paper_ids:
        params["paper_ids"] = paper_ids
    return safe_api_call("POST", "/ask", params=params)

def get_paper_details(paper_id):
    return safe_api_call("GET", f"/paper/{paper_id}")

def generate_draft(topic):
    return safe_api_call("POST", "/draft", params={"topic": topic})

def upload_and_analyze(file):
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    return safe_api_call("POST", "/analyze", files=files)

# Sidebar
st.sidebar.title("🧬 COVID-19 AI Research")
st.sidebar.info("Accelerating scientific discovery with AI.")

search_mode = st.sidebar.selectbox(
    "Search Mode",
    ["Hybrid (Recommended)", "Semantic", "Keyword"],
    index=0
).lower().split(" ")[0]

limit = st.sidebar.slider("Results Limit", 5, 50, 10)

st.sidebar.divider()
st.sidebar.subheader("Tools")
if st.sidebar.button("🔍 Search Papers"):
    st.session_state.app_mode = "search"
if st.sidebar.button("📝 Draft Research Paper"):
    st.session_state.app_mode = "draft"
if st.sidebar.button("📁 Upload & Analyze PDF"):
    st.session_state.app_mode = "upload"

st.sidebar.divider()
st.sidebar.subheader("About")
st.sidebar.write("This platform uses Deep Learning (SBERT, BART) and RAG (Gemini) to accelerate COVID-19 research.")

# App Mode State
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "search"

# Main UI
if st.session_state.app_mode == "search":
    st.title("Search & Retrieval Engine")
    query = st.text_input("Ask a scientific question or enter keywords", placeholder="e.g., What are the neurological symptoms of long COVID?")

    if query:
        with st.spinner("Searching papers..."):
            data = search_papers(query, search_mode, limit)
            
            if data and data.get("results"):
                st.subheader(f"Found {len(data['results'])} relevant papers")
                
                # Hybrid Answer (RAG)
                if st.checkbox("💡 Generate AI Answer with Citations", value=True):
                    with st.spinner("AI is analyzing papers..."):
                        qa_data = ask_question(query)
                        if qa_data and "answer" in qa_data:
                            st.info("### AI Summary Answer")
                            st.write(qa_data["answer"])
                            st.divider()

                # Results Display
                for res in data["results"]:
                    with st.container():
                        st.markdown(f'<div class="search-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([0.8, 0.2])
                        
                        # Link construction: Prioritize 'url' field, fallback to DOI
                        paper_url = res.get("url")
                        if not paper_url and res.get("doi"):
                            doi = res["doi"]
                            paper_url = f"https://doi.org/{doi}" if "http" not in doi else doi
                        
                        with col1:
                            if paper_url:
                                st.markdown(f'<a class="paper-title" href="{paper_url}" target="_blank">{res["title"]}</a>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="paper-title">{res["title"]}</span>', unsafe_allow_html=True)
                            
                            st.markdown(f'<div class="paper-meta">DOI: {res.get("doi", "N/A")} | <span class="score-badge">Relevance: {res["score"]:.2f}</span></div>', unsafe_allow_html=True)
                        
                        with col2:
                            if st.button("View Details", key=f"btn_{res['cord_uid']}"):
                                st.session_state.selected_paper = res['cord_uid']
                                st.rerun()
                        
                        st.write(res["abstract"][:300] + "..." if res["abstract"] else "No abstract available.")
                        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.app_mode == "draft":
    st.title("AI Research Paper Drafter")
    st.write("Generate a comprehensive research paper draft based on the latest scholarly articles.")
    
    topic = st.text_input("Enter the topic for your research paper", placeholder="e.g., The neurological impact of long COVID on young adults")
    
    if topic:
        if st.button("Generate Full Draft"):
            with st.spinner("AI is researching and writing your paper... this may take a minute."):
                draft_data = generate_draft(topic)
                if draft_data and "draft" in draft_data:
                    st.success("Draft Generated Successfully!")
                    st.markdown("---")
                    st.markdown(draft_data["draft"])
                    st.divider()
                    st.subheader("Source Papers Used")
                    for s in draft_data["sources"]:
                        st.write(f"- **{s['title']}** (DOI: {s.get('doi', 'N/A')})")
                    st.download_button("Download Draft as Text", draft_data["draft"], file_name="research_draft.md")

elif st.session_state.app_mode == "upload":
    st.title("PDF Upload & Analysis")
    st.write("Upload a research paper PDF to get an instant summary and key findings.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if st.button("Analyze Uploaded Paper"):
            with st.spinner("Analyzing document..."):
                analysis_data = upload_and_analyze(uploaded_file)
                if analysis_data and "analysis" in analysis_data:
                    st.success(f"Analysis of {uploaded_file.name} complete!")
                    st.markdown("---")
                    st.markdown(analysis_data["analysis"])
                    st.download_button("Download Analysis", analysis_data["analysis"], file_name=f"analysis_{uploaded_file.name}.txt")

# Paper Details Dialog/View
if "selected_paper" in st.session_state:
    cord_uid = st.session_state.selected_paper
    details = get_paper_details(cord_uid)
    
    if details:
        st.divider()
        st.header(details["metadata"]["title"])
        st.subheader("AI-Generated Summary")
        st.success(details["summary"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Entities Found")
            for label, ents in details["entities"].items():
                st.write(f"**{label}:** {', '.join(ents)}")
        
        with col2:
            st.subheader("Metadata")
            st.write(f"**Authors:** {details['metadata']['authors']}")
            st.write(f"**Published:** {details['metadata']['publish_time']}")
            st.write(f"**Journal:** {details['metadata']['journal']}")
            st.write(f"**DOI:** {details['metadata'].get('doi', 'N/A')}")
            
            # Direct link to paper
            paper_url = details['metadata'].get('url')
            if not paper_url and details['metadata'].get('doi'):
                doi = details['metadata']['doi']
                paper_url = f"https://doi.org/{doi}" if "http" not in doi else doi
            
            if paper_url:
                st.link_button("🌐 Read Full Paper Online", paper_url)
        
        st.subheader("Abstract")
        st.write(details["abstract"])
        
        if st.button("Back to Search"):
            del st.session_state.selected_paper
            st.rerun()
