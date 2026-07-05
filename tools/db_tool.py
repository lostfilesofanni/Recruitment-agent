import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class VectorDBManager:
    """
    Manages ChromaDB vector operations using SentenceTransformers local embeddings.
    """
    def __init__(self, db_path: str = "vectordb", collection_name: str = "resumes"):
        # Resolve absolute path to avoid running directory confusion
        abs_db_path = os.path.abspath(db_path)
        self.client = chromadb.PersistentClient(path=abs_db_path)
        # Load local lightweight embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_resume(self, file_name: str, content: str):
        """
        Embeds and adds a resume to ChromaDB.
        Overwrites if the resume ID already exists.
        """
        embedding = self.model.encode(content).tolist()
        # Delete first if it exists to overwrite cleanly
        try:
            self.collection.delete(ids=[file_name])
        except Exception:
            pass
            
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{"file_name": file_name}],
            ids=[file_name]
        )

    def resume_exists(self, file_name: str) -> bool:
        """
        Checks if a resume is already in the database.
        """
        try:
            res = self.collection.get(ids=[file_name])
            return len(res.get("ids", [])) > 0
        except Exception:
            return False

    def search_candidates(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search in ChromaDB.
        """
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        docs = []
        if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                # Cosine distance: smaller means more similar (0.0 = identical)
                score = results["distances"][0][i] if "distances" in results and results["distances"] else 1.0
                # Convert distance to similarity score
                similarity = round(1.0 - score, 4)
                docs.append({
                    "content": results["documents"][0][i],
                    "file_name": results["metadatas"][0][i]["file_name"],
                    "score": similarity
                })
        return docs
