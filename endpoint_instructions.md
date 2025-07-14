# Instructions: How to Work with the Project's Endpoints

## 1. Check API Status (GET /)
   - **Purpose**: Verify the API is running.
   - **Example (curl)**:
     ```
     curl -k https://yourdomain.com/
     ```
   - **Expected Response** (JSON): `{"status": "ok"}`
   - **Tips**: Use this to debug if the service is up.

## 2. Upload a Document (POST /upload)
   - **Purpose**: Upload a PDF for processing and indexing. It extracts content (text, tables, OCR if needed), generates a unique `document_id`, stores the file in `uploaded_files/`, and indexes it in ChromaDB.
   - **Requirements**: Only PDFs accepted (content-type check). File is saved permanently.
   - **Example (curl)**:
     ```
     curl -k -X POST "https://yourdomain.com/upload" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@/path/to/your/document.pdf"
     ```
   - **Expected Response** (JSON on success):
     ```
     {
       "document_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
       "filename": "document.pdf"
     }
     ```
   - **Errors**: 400 for non-PDF, 500 for processing failures (file is cleaned up).
   - **Tips**: Save the `document_id` for querying. Uploads are persistent.

## 3. Ask a Question (POST /chat)
   - **Purpose**: Query the document with a natural language question. The RAG system retrieves relevant chunks, generates an answer using AI, and includes sources with a download link.
   - **Requirements**: Provide `query` (string) and `document_id` (from upload) in JSON body.
   - **Example (curl)**:
     ```
     curl -k -X POST "https://yourdomain.com/chat" \
       -H "Content-Type: application/json" \
       -d '{
         "query": "What is the main conclusion of this document?",
         "document_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
       }'
     ```
   - **Expected Response** (JSON): Something like:
     ```
     {
       "answer": "The main conclusion is...",
       "sources": [
         {
           "content": "Relevant chunk...",
           "download_link": "/files/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
         }
       ]
     }
     ```
   - **Errors**: 500 if RAG fails (e.g., invalid document_id).
   - **Tips**: Queries are document-specific. For multi-doc support, extend the code.

## 4. Download a File (GET /files/{document_id})
   - **Purpose**: Retrieve an uploaded file by its ID.
   - **Example (curl)**:
     ```
     curl -k -O "https://yourdomain.com/files/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
     ```
   - **Expected Response**: The file downloads (as `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.pdf`—original name isn't preserved yet; improve by storing metadata).
   - **Errors**: 404 if file not found.
   - **Tips**: Links are auto-added to chat responses for easy access.
