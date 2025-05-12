from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
from rag_app import RAGApplication

app = FastAPI(title="RAG Document Q&A System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = "uploads"
INDEX_DIR = "index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize RAG application
rag_app = RAGApplication()

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    saved_files = []
    for file in files:
        if not file.filename.lower().endswith(('.pdf', '.ppt', '.pptx')):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
    
    try:
        rag_app.process_documents(saved_files)
        rag_app.save_index(INDEX_DIR)
        return {"message": "Files processed successfully", "files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(question: str):
    """Query the RAG system."""
    try:
        result = rag_app.query(question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 