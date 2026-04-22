import logging
import os
from typing import Dict, Any
from datasets import load_dataset
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CORD19Ingestor:
    def __init__(self, mongo_url: str = None):
        url = mongo_url or os.getenv("MONGO_URL", "mongodb://localhost:27017")
        self.client = MongoClient(url)
        self.db = self.client["covid_research"]
        self.papers = self.db.papers

    def fetch_cord19(self, limit: int = 1000):
        logger.info("Loading CORD-19 dataset from Hugging Face...")
        try:
            dataset = load_dataset("allenai/cord19", "metadata", split="train", streaming=True)
            
            count = 0
            for row in dataset:
                if count >= limit:
                    break
                
                self._process_row(row)
                count += 1
                if count % 100 == 0:
                    logger.info(f"Processed {count} papers...")
            
            logger.info(f"Successfully ingested {count} papers into MongoDB.")
            
        except Exception as e:
            logger.error(f"Error fetching CORD-19: {e}")
            logger.info("Generating mock data for testing...")
            self._generate_mock_data()

    def _generate_mock_data(self):
        mock_papers = [
            {
                "cord_uid": "mock_001",
                "title": "Impact of SARS-CoV-2 on Respiratory Health",
                "authors": "Smith, J. et al.",
                "abstract": "This study examines the long-term effects of COVID-19 on lung function and respiratory health in recovered patients.",
                "doi": "10.1001/mock.1",
                "publish_time": datetime(2021, 5, 20)
            },
            {
                "cord_uid": "mock_002",
                "title": "Vaccine Efficacy Against Variants",
                "authors": "Doe, A. et al.",
                "abstract": "We analyze the effectiveness of mRNA vaccines against the Omicron and Delta variants of SARS-CoV-2.",
                "doi": "10.1001/mock.2",
                "publish_time": datetime(2022, 1, 15)
            }
        ]
        for p in mock_papers:
            self.papers.update_one({"cord_uid": p["cord_uid"]}, {"$set": p}, upsert=True)
        logger.info("Mock data generated.")

    def _process_row(self, row: Dict[str, Any]):
        publish_time = None
        if row.get("publish_time"):
            try:
                date_str = row["publish_time"]
                if len(date_str) == 4:
                    publish_time = datetime.strptime(date_str, "%Y")
                else:
                    publish_time = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except Exception:
                pass

        paper = {
            "cord_uid": row.get("cord_uid"),
            "title": row.get("title", "No Title"),
            "authors": row.get("authors"),
            "abstract": row.get("abstract"),
            "publish_time": publish_time,
            "doi": row.get("doi"),
            "journal": row.get("journal"),
            "source_x": row.get("source_x"),
            "license": row.get("license"),
            "url": row.get("url"),
            "extra_metadata": {
                "pubmed_id": row.get("pubmed_id"),
                "pmcid": row.get("pmcid"),
                "who_covidence_id": row.get("who_covidence_id")
            }
        }
        
        # Upsert in MongoDB
        self.papers.update_one(
            {"cord_uid": paper["cord_uid"]},
            {"$set": paper},
            upsert=True
        )

if __name__ == "__main__":
    ingestor = CORD19Ingestor()
    ingestor.fetch_cord19(limit=20)
