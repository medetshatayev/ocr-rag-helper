#from fastapi import FastAPI
#from fastapi.middleware.cors import CORSMiddleware

#app = FastAPI(title="Document RAG API")

# Set up CORS middleware to allow requests from the frontend
#app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins to avoid CORS issues
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#@app.get("/", summary="Check API status")
def read_root():
    return {"status": "ok"}

# Add these imports to api_main.py
#import uuid
#import os
#from fastapi import UploadFile, File, HTTPException
#from fastapi.responses import FileResponse
#from rag_system import RAGSystem

#rag_system = RAGSystem()
#UPLOAD_DIR = "uploaded_files"
#os.makedirs(UPLOAD_DIR, exist_ok=True)

#@app.post("/upload", summary="Upload and process a document")
async def upload_document(file: UploadFile = File(...)):
    supported_extensions = rag_system.document_processor.supported_extensions
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_extension}'. Supported types: {', '.join(sorted(supported_extensions))}"
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

#@app.get("/files/{document_id}", summary="Download an uploaded file")
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

# Add these imports to api_main.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    document_id: str

#@app.post("/chat", summary="Get an answer from the RAG system")
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
