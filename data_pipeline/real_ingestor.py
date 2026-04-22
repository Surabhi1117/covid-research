import os
import httpx
import logging
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivIngestor:
    def __init__(self):
        self.mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
        self.client = AsyncIOMotorClient(self.mongo_url)
        self.db = self.client.covid_research
        self.papers = self.db.papers

    async def fetch_arxiv_papers(self, query: str = "covid-19", limit: int = 50):
        """Fetch real research papers from arXiv API."""
        logger.info(f"Fetching arXiv papers for query: {query}...")
        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.text)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                
                papers_to_insert = []
                for entry in root.findall('atom:entry', namespace):
                    title = entry.find('atom:title', namespace).text.strip()
                    summary = entry.find('atom:summary', namespace).text.strip()
                    published = entry.find('atom:published', namespace).text
                    url = entry.find('atom:id', namespace).text
                    
                    # DOI is often in a specific link or not present in arXiv directly
                    # but we can use the arXiv ID as a fallback or construct a link
                    arxiv_id = url.split('/')[-1]
                    doi = f"10.48550/arXiv.{arxiv_id}"
                    
                    authors = [a.find('atom:name', namespace).text for a in entry.findall('atom:author', namespace)]
                    
                    paper = {
                        "cord_uid": arxiv_id,
                        "title": title,
                        "authors": ", ".join(authors),
                        "abstract": summary,
                        "doi": doi,
                        "publish_time": datetime.fromisoformat(published.replace('Z', '+00:00')),
                        "url": url,
                        "ingested_at": datetime.utcnow()
                    }
                    papers_to_insert.append(paper)
                
                if papers_to_insert:
                    # Clear old mock data
                    await self.papers.delete_many({"cord_uid": {"$regex": "^mock_"}})
                    
                    for p in papers_to_insert:
                        await self.papers.update_one(
                            {"cord_uid": p["cord_uid"]},
                            {"$set": p},
                            upsert=True
                        )
                    logger.info(f"Successfully ingested {len(papers_to_insert)} arXiv papers.")
                else:
                    logger.warning("No arXiv papers found.")
                    
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {e}")

async def main():
    ingestor = ArxivIngestor()
    topics = ["covid-19 psychological", "sars-cov-2 respiratory", "covid-19 vaccine"]
    for topic in topics:
        await ingestor.fetch_arxiv_papers(query=topic, limit=20)
        await asyncio.sleep(2) # Be nice to arXiv
    print("ArXiv data ingestion complete.")

if __name__ == "__main__":
    asyncio.run(main())
