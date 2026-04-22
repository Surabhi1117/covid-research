import logging
import torch
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        # Initialize ChromaDB (fast)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="papers")

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # Convert to list of lists if it's a numpy array
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return embeddings

    def add_to_index(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add texts and their embeddings to ChromaDB."""
        if not texts:
            return
        
        embeddings = self.generate_embeddings(texts)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Added {len(ids)} documents to ChromaDB.")

    def search_vectors(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """Search ChromaDB for similar vectors."""
        query_embeddings = self.generate_embeddings([query])
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        return results
