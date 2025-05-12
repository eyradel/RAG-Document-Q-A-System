# RAG Document Q&A System

This is a Retrieval-Augmented Generation (RAG) application that can process PDF and PowerPoint files, create embeddings using Hugging Face models, and answer questions about the content.

## Features

- Process PDF and PowerPoint (PPT/PPTX) files
- Create embeddings using Hugging Face's sentence-transformers
- Store and search embeddings using FAISS
- Question answering using Hugging Face's RoBERTa model
- REST API interface using FastAPI

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python app.py
```

2. The server will be available at `http://localhost:8000`

3. API Endpoints:
   - `POST /upload`: Upload PDF or PowerPoint files
   - `POST /query`: Ask questions about the uploaded documents
   - `GET /health`: Health check endpoint

4. Example usage with curl:
```bash
# Upload files
curl -X POST -F "files=@document.pdf" -F "files=@presentation.pptx" http://localhost:8000/upload

# Query the system
curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the main topic of the document?"}' http://localhost:8000/query
```

## Architecture

The application consists of several components:

1. `document_processor.py`: Handles PDF and PowerPoint file processing
2. `embedding_manager.py`: Manages text embeddings and vector search
3. `rag_app.py`: Main RAG application logic
4. `app.py`: FastAPI server implementation

## Models Used

- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- QA Model: `deepset/roberta-base-squad2`

## Notes

- The application creates two directories:
  - `uploads/`: For storing uploaded files
  - `index/`: For storing the FAISS index and document cache
- The system uses FAISS for efficient similarity search
- All models are from Hugging Face and are downloaded automatically on first use
