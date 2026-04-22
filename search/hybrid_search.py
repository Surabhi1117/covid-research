from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from embeddings.engine import EmbeddingEngine

class SearchEngine:
    def __init__(self, db: AsyncIOMotorDatabase, embedding_engine: EmbeddingEngine):
        self.db = db
        self.embedding_engine = embedding_engine

    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using ChromaDB."""
        vector_results = self.embedding_engine.search_vectors(query, n_results=limit)
        
        results = []
        ids = vector_results["ids"][0]
        distances = vector_results["distances"][0]
        
        # Convert distances to similarity scores (approximate)
        for doc_id, dist in zip(ids, distances):
            # Chroma distances can vary; we use a simple inverse mapping for score
            score = 1.0 / (1.0 + dist)
            
            # Fetch full metadata from MongoDB
            paper = await self.db.papers.find_one({"cord_uid": doc_id})
            if paper:
                results.append({
                    "id": str(paper["_id"]),
                    "cord_uid": paper["cord_uid"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "doi": paper.get("doi"),
                    "url": paper.get("url"),
                    "score": score
                })
        
        return results

    async def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform keyword search using MongoDB Text Search."""
        cursor = self.db.papers.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        results = []
        async for paper in cursor:
            results.append({
                "id": str(paper["_id"]),
                "cord_uid": paper["cord_uid"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "doi": paper.get("doi"),
                "url": paper.get("url"),
                "score": paper["score"]
            })
        return results

    async def hybrid_search(self, query: str, limit: int = 10, k: int = 60) -> List[Dict[str, Any]]:
        """Combine keyword and semantic search using RRF."""
        candidate_limit = limit * 2
        
        keyword_results = await self.keyword_search(query, candidate_limit)
        semantic_results = await self.semantic_search(query, candidate_limit)
        
        scores = {}
        paper_data = {}
        
        for rank, res in enumerate(keyword_results, 1):
            uid = res["cord_uid"]
            scores[uid] = scores.get(uid, 0) + 1.0 / (k + rank)
            paper_data[uid] = res

        for rank, res in enumerate(semantic_results, 1):
            uid = res["cord_uid"]
            scores[uid] = scores.get(uid, 0) + 1.0 / (k + rank)
            if uid not in paper_data:
                paper_data[uid] = res
        
        sorted_uids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        final_results = []
        for uid in sorted_uids[:limit]:
            res = paper_data[uid]
            res["rrf_score"] = scores[uid]
            final_results.append(res)
            
        return final_results
