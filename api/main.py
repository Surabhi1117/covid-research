from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from pypdf import PdfReader
import io
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient

from data_pipeline.db import init_mongo, get_mongo_db
from embeddings.engine import EmbeddingEngine
from search.hybrid_search import SearchEngine
from rag.pipeline import RAGPipeline
from models.summarizer import Summarizer
from models.ner import NERExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="COVID-19 Research Platform API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines (lazy initialized)
embedding_engine = None
rag_pipeline = None
summarizer = None
ner_extractor = None

@app.on_event("startup")
async def startup_event():
    global embedding_engine, rag_pipeline, summarizer, ner_extractor
    logger.info("Initializing engines on startup...")
    await init_mongo()
    embedding_engine = EmbeddingEngine()
    rag_pipeline = RAGPipeline()
    summarizer = Summarizer()
    ner_extractor = NERExtractor()
    logger.info("All engines initialized.")

@app.get("/")
async def root():
    return {"status": "online", "message": "COVID-19 Research API is running"}

@app.get("/search")
async def search_papers(q: str, mode: str = "hybrid", limit: int = 10, db = Depends(get_mongo_db)):
    """
    Search for papers using keyword, semantic, or hybrid search.
    """
    engine = SearchEngine(db, embedding_engine)
    if mode == "semantic":
        results = await engine.semantic_search(q, limit=limit)
    elif mode == "keyword":
        results = await engine.keyword_search(q, limit=limit)
    else:
        results = await engine.hybrid_search(q, limit=limit)
    return {"query": q, "results": results}

@app.get("/paper/{cord_uid}")
async def get_paper(cord_uid: str, db = Depends(get_mongo_db)):
    """
    Get detailed information about a single paper.
    """
    paper = await db.papers.find_one({"cord_uid": cord_uid})
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Generate summary and extract entities on the fly
    full_text = paper.get("abstract", "")
    summary = summarizer.summarize(full_text)
    entities = ner_extractor.extract_entities(full_text)
    
    return {
        "metadata": {
            "title": paper["title"],
            "authors": paper["authors"],
            "publish_time": paper.get("publish_time"),
            "doi": paper.get("doi"),
            "journal": paper.get("journal", "N/A"),
            "license": paper.get("license", "N/A"),
            "url": paper.get("url")
        },
        "abstract": full_text,
        "summary": summary,
        "entities": entities
    }

@app.post("/ask")
async def ask_question(question: str, paper_ids: Optional[List[str]] = None, db = Depends(get_mongo_db)):
    """
    Ask a question and get a RAG-based answer.
    """
    engine = SearchEngine(db, embedding_engine)
    if paper_ids:
        # Use specific papers as context
        context_papers = []
        for pid in paper_ids:
            p = await db.papers.find_one({"cord_uid": pid})
            if p:
                context_papers.append(p)
    else:
        # Perform search to get context
        context_papers = await engine.hybrid_search(question, limit=5)
    
    answer = rag_pipeline.answer_question(question, context_papers)
    return {"question": question, "answer": answer, "sources": context_papers}

@app.post("/draft")
async def generate_draft(topic: str, db = Depends(get_mongo_db)):
    """
    Generate a full research paper draft on a given topic.
    """
    engine = SearchEngine(db, embedding_engine)
    context_papers = await engine.hybrid_search(topic, limit=10)
    
    draft = rag_pipeline.draft_paper(topic, context_papers)
    return {"topic": topic, "draft": draft, "sources": context_papers}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Extract text from an uploaded PDF and generate a summary + key points using Gemini.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        content = await file.read()
        pdf = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        
        truncated_text = text[:30000] 
        prompt = f"""
Analyze the following research paper text and provide:
1. A concise executive summary (3-5 sentences).
2. A list of the most important points/findings (bullet points).
3. The main contribution of this work.

PAPER TEXT:
{truncated_text}
"""
        try:
            response = rag_pipeline.model.generate_content(prompt)
            return {"filename": file.filename, "analysis": response.text}
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="The AI model 'gemini-flash-latest' was not found. Please check your API key and model access.")
            raise e
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/paraphrase")
async def paraphrase_text(text: str, style: str = "Balanced"):
    """
    Paraphrase text in a specific style using Gemini.
    """
    prompt = f"Paraphrase the following scientific text in a {style} tone. Provide 2 distinct versions:\n\n{text}"
    try:
        response = rag_pipeline.model.generate_content(prompt)
        return {"original": text, "style": style, "paraphrased": response.text}
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="The AI model 'gemini-flash-latest' was not found.")
        raise HTTPException(status_code=500, detail=f"Paraphrasing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
