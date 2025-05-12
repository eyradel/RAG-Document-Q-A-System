from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = self.model.get_sentence_embedding_dimension()

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=True)

    def build_index(self, texts: List[str]):
        """Build FAISS index from texts."""
        embeddings = self.create_embeddings(texts)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        self.documents = texts

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Ensure index is valid
                results.append({
                    'text': self.documents[idx],
                    'score': float(1 / (1 + distance))  # Convert distance to similarity score
                })
        
        return results

    def save_index(self, directory: str):
        """Save the index and documents to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self, directory: str):
        """Load the index and documents from disk."""
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f) 