# from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import os
import logging
from pydantic import BaseModel
from rag_system import RAGSystem

logger = logging.getLogger(__name__)

# FastAPI application instance
app = FastAPI(title="Document RAG API")

# Allow requests from any origin (adjust if you need stricter CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate RAG system and prepare upload directory
rag_system = RAGSystem()
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Root route
@app.get("/", summary="Check API status")
def read_root():
    return {"status": "ok"}

# Upload endpoint
@app.post("/upload", summary="Upload and process a document")
async def upload_document(file: UploadFile = File(...)):
    supported_extensions = rag_system.document_processor.supported_extensions
    # Attempt to determine file extension – fall back to Content-Type when the filename is empty or malformed.
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in supported_extensions:
        # Infer extension from Content-Type header for common types
        content_type_map = {
            "application/pdf": ".pdf",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/json": ".json",
        }
        inferred_ext = content_type_map.get(file.content_type, "")

        if inferred_ext and inferred_ext in supported_extensions:
            file_extension = inferred_ext
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported or unknown file type. Filename extension: '{file_extension or 'none'}', Content-Type: '{file.content_type}'"
            )

    document_id = str(uuid.uuid4())
    saved_file_path = os.path.join(UPLOAD_DIR, f"{document_id}{file_extension}")

    try:
        # Save the uploaded file permanently
        with open(saved_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process and index the PDF, passing the original filename for metadata
        result = rag_system.index_file(
            file_path=saved_file_path,
            document_id=document_id,
            original_filename=file.filename
        )

        # Check if processing was successful
        if result.get("status") != "success":
            # Clean up the saved file if processing fails
            if os.path.exists(saved_file_path):
                os.remove(saved_file_path)
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {result.get('message', 'Unknown error')}")
        
    except Exception as e:
        # Clean up the saved file in case of an unexpected error
        if os.path.exists(saved_file_path):
            os.remove(saved_file_path)
        # Re-raise HTTPException directly
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    return {"document_id": document_id, "filename": file.filename}

# Download endpoint for original file (optional)
@app.get("/files/{document_id}", summary="Download an uploaded file")
async def get_file(document_id: str):
    # Attempt to locate the file with any supported extension
    for ext in rag_system.document_processor.supported_extensions:
        file_path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
        if os.path.exists(file_path):
            return FileResponse(
                path=file_path,
                media_type='application/octet-stream',
                filename=f"{document_id}{ext}"
            )

    raise HTTPException(status_code=404, detail="File not found.")

# Pydantic model for chat endpoint
class ChatRequest(BaseModel):
    query: str
    document_id: str

# Chat endpoint (RAG-based QA)
@app.post("/chat", summary="Get an answer from the RAG system")
async def chat(request: ChatRequest):
    try:
        # Generate a response using the RAG system
        response = rag_system.generate_response(
            query=request.query,
            document_id=request.document_id
        )
        # Add a download link to each source
        if "sources" in response:
            for source in response["sources"]:
                source["download_link"] = f"/files/{request.document_id}"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answer from RAG system: {e}") 
