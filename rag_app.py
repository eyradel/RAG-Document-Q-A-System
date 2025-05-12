from typing import List, Dict
import os
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from transformers import pipeline

class RAGApplication:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(model_name)
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )

    def process_documents(self, file_paths: List[str]):
        """Process multiple documents and build the search index."""
        all_chunks = []
        for file_path in file_paths:
            chunks = self.document_processor.process_file(file_path)
            all_chunks.extend(chunks)
        
        self.embedding_manager.build_index(all_chunks)

    def query(self, question: str, k: int = 3) -> Dict:
        """Query the RAG system with a question."""
        # First, find relevant document chunks
        relevant_chunks = self.embedding_manager.search(question, k=k)
        
        # Combine the chunks into a context
        context = " ".join([chunk['text'] for chunk in relevant_chunks])
        
        # Use the QA pipeline to answer the question
        answer = self.qa_pipeline(
            question=question,
            context=context
        )
        
        return {
            'answer': answer['answer'],
            'confidence': answer['score'],
            'relevant_chunks': relevant_chunks
        }

    def save_index(self, directory: str):
        """Save the current index and documents."""
        self.embedding_manager.save_index(directory)

    def load_index(self, directory: str):
        """Load a previously saved index and documents."""
        self.embedding_manager.load_index(directory) 