
# OCR RAG Helper

This project is a Retrieval-Augmented Generation (RAG) system with a Streamlit-based user interface and a FastAPI backend. It's designed to process and understand documents, including PDFs with text, tables, and images (via OCR), and answer questions based on their content.

## Key Features

- **Advanced Document Processing**: Can handle multiple file formats, including `.pdf`, `.txt`, `.md`, and various code files.
- **OCR and Table Extraction**: For PDFs, the system can extract text, tables, and even use OCR (Optical Character Recognition) to get text from images.
- **Vector-Based Retrieval**: Uses `ChromaDB` to store document chunks as vector embeddings for efficient similarity search.
- **Azure OpenAI Integration**: Leverages Azure OpenAI for generating high-quality embeddings and answers.
- **User-Friendly Web Interface**: A `Streamlit` application provides an intuitive chat interface for interacting with the RAG system.
- **API Backend**: A `FastAPI` backend exposes endpoints for uploading documents and asking questions.
- **Flexible PDF Processing**: Includes both an advanced PDF processor (with `PyMuPDF` and `Tesseract`) and a simple, fallback processor (`pdfplumber`) for environments where full OCR capabilities are not needed or available.

## Project Structure

```
.
├── Docs/                   # Directory where uploaded documents are stored
├── traefik/                # Traefik configuration for reverse proxy
├── uploaded_files/         # Directory for files uploaded via the API
├── vector_db/              # Directory for the ChromaDB vector store
├── api_main.py             # FastAPI application
├── document_processor.py   # Handles processing of different document types
├── main.py                 # Main Streamlit application
├── pdf_processor.py        # Advanced PDF processor with OCR
├── pdf_processor_simple.py # Simple PDF processor (fallback)
├── rag_system.py           # Core RAG system logic
├── docker-compose.yml      # Docker Compose for running the application
├── Dockerfile              # Dockerfile for building the application image
├── requirements.txt        # Python dependencies
└── README.md              
```

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- An Azure OpenAI account with API key and endpoint

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/medetshatayev/ocr-rag-helper.git
    cd ocr-rag-helper
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Azure OpenAI credentials:
    ```
    AZURE_OPENAI_API_KEY="your_api_key"
    AZURE_OPENAI_ENDPOINT="your_endpoint"
    AZURE_OPENAI_API_VERSION="your-deployment-version"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="your-embedding-deployment"
    AZURE_OPENAI_DEPLOYMENT="your-chat-deployment"
    ```

3.  **Build and run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the application:**
    -   **Streamlit UI**: Open your browser and go to `http://localhost:8501`
    -   **FastAPI Docs**: Open your browser and go to `http://localhost:8000/docs`

## API Usage

The application includes a FastAPI backend with the following key endpoints:

-   `POST /upload`: Upload a PDF file for processing.
-   `POST /chat`: Ask a question about an uploaded document.
-   `GET /files/{document_id}`: Download an uploaded file.

You can find detailed information about the API and test the endpoints by visiting the automatically generated documentation at `http://localhost:8000/docs`. 