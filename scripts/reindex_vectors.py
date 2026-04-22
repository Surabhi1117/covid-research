import os
import asyncio
from pymongo import MongoClient
from embeddings.engine import EmbeddingEngine
from dotenv import load_dotenv

load_dotenv()

def reindex():
    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    client = MongoClient(mongo_url)
    db = client["covid_research"]
    papers_col = db.papers
    
    engine = EmbeddingEngine()
    
    print("Fetching papers from MongoDB...")
    # Fetch papers that have an abstract and haven't been indexed or just all of them
    papers = list(papers_col.find({"abstract": {"$exists": True, "$ne": ""}}))
    print(f"Found {len(papers)} papers to index.")
    
    batch_size = 50
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        ids = [p["cord_uid"] for p in batch]
        texts = [f"{p['title']} {p.get('abstract', '')}" for p in batch]
        metadatas = [{"title": p["title"], "doi": p.get("doi", "")} for p in batch]
        
        print(f"Indexing batch {i//batch_size + 1}...")
        engine.add_to_index(ids, texts, metadatas)

    print("Re-indexing complete.")

if __name__ == "__main__":
    reindex()
