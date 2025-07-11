# OCR-Powered Document Q&A with RAG

This project provides a powerful, containerized API for asking questions about your documents. It uses an advanced Retrieval-Augmented Generation (RAG) pipeline, featuring Optical Character Recognition (OCR) to understand text even in scanned PDFs. The system is built with FastAPI, managed with Docker, and fronted by a Traefik reverse proxy for secure and scalable deployment.

## System Architecture

The application is composed of several key components that work together to deliver a seamless document-to-answer pipeline:

1.  **Traefik Reverse Proxy**: Acts as the entry point, handling incoming HTTPS requests, managing TLS certificates, and routing traffic to the application container.
2.  **FastAPI Application (`api_main.py`)**: The core web service that exposes a RESTful API for uploading documents and asking questions.
3.  **RAG System (`rag_system.py`)**: The brain of the operation. It orchestrates the entire process, from sending content to be vectorized by an AI model (like Azure OpenAI) to storing and retrieving those vectors from a persistent ChromaDB database.
4.  **Document Processor (`document_processor.py`)**: A smart dispatcher that analyzes files and routes them to the correct content extraction module. It supports multiple file types and includes a fallback mechanism for environments where advanced dependencies are not available.
5.  **PDF Processors**:
    *   **Advanced (`pdf_processor.py`)**: The primary processor, which uses a combination of `PyMuPDF` for efficient text extraction, `pdfplumber` for accurate table parsing, and `Tesseract` for OCR on image-based content.
    *   **Simple (`pdf_processor_simple.py`)**: A fallback processor that uses `pdfplumber` alone, ensuring the system can still process standard PDFs even without the OCR dependencies.

## Key Features

- **Advanced Document Analysis**: Extracts not just text, but also tables from PDFs, preserving their structure.
- **OCR Integration**: Automatically performs OCR on scanned PDFs or pages containing images to extract text that would otherwise be inaccessible.
- **Flexible RAG Pipeline**: Leverages state-of-the-art AI models for generating embeddings and answering questions, with integration for services like Azure OpenAI.
- **Persistent Vector Storage**: Uses ChromaDB to store document vectors, allowing for efficient similarity searches and persistent memory of your document library.
- **Scalable and Secure Deployment**: Fully containerized with Docker and pre-configured for HTTPS with Traefik, making it suitable for both development and production environments.
- **Resilient by Design**: Includes a fallback PDF processor to ensure core functionality remains available even if certain dependencies fail to install.

## Setup and Deployment

Follow these instructions to get the entire stack up and running on your local machine.

### Prerequisites

- [Docker Engine](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- A code editor like VS Code or a terminal for running commands.

### Step 1: Clone and Configure

First, get the code and set up the necessary environment configuration.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ocr-rag-helper
    ```

2.  **Create the environment file (`.env`):**
    This file holds your secret keys and configuration. Create a file named `.env` in the project root and add the following, filling in your own credentials.

    ```env
    # --- RAG System Configuration ---
    # The size of text chunks for the RAG model
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200

    # --- Azure OpenAI Credentials ---
    # Fill these with your Azure service details
    AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT="https://your-service-name.openai.azure.com/"
    AZURE_OPENAI_API_VERSION="2024-02-01"
    AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini" # Your deployment for chat/completions
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small" # Your deployment for embeddings
    ```

### Step 2: Configure Local DNS

To allow Traefik to correctly route traffic, you need to map the domain name used in the configuration to your local machine.

1.  **Edit your `hosts` file** (requires administrator/sudo privileges):
    - **macOS/Linux**: `/etc/hosts`
    - **Windows**: `C:\Windows\System32\drivers\etc\hosts`

2.  **Add the following line** and save the file:
    ```
    127.0.0.1 yourdomain.com traefik.yourdomain.com
    ```
    This ensures that requests to both the API and the Traefik dashboard are routed to your local Docker environment.

### Step 3: Launch the Services

The services must be started in a specific order to ensure the shared Docker network is created correctly.

1.  **Start Traefik:**
    Navigate to the `traefik` directory and use Docker Compose to start it in detached mode. This creates the `traefik_proxy` network.
    ```bash
    cd traefik
    docker-compose up -d
    cd ..
    ```

2.  **Start the Application:**
    Now, from the project root, build and start the main application. It will automatically connect to the network created by Traefik.
    ```bash
    docker-compose up -d --build
    ```

3.  **Verify the setup:**
    Check that both containers are running.
    ```bash
    docker ps
    ```
    You should see both `traefik` and `ocr_rag_api` in the list.

## Usage Guide

You can interact with the API using `curl` or any other API client.

### 1. Upload a Document

Send a `POST` request to the `/upload` endpoint. The document will be processed, and its content will be indexed in the vector database.

- **Command:**
  ```bash
  curl -k -X POST "https://yourdomain.com/upload" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/document.pdf"
  ```
- **Successful Response:**
  ```json
  {
    "document_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "filename": "your_document.pdf"
  }
  ```
  **Save the `document_id`** for the next step.

### 2. Ask a Question

Send a `POST` request to the `/chat` endpoint with your query and the `document_id` of the file you want to ask about.

- **Command:**
  ```bash
  curl -k -X POST "https://yourdomain.com/chat" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "What is the main conclusion of this document?",
      "document_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    }'
  ```
- **Successful Response:**
  The API will return a detailed answer synthesized from the relevant parts of your document, along with the sources it used.

## Stopping the Application

To stop all services and remove the containers, run the `down` command in each directory.

1.  **Stop the application:**
    ```bash
    docker-compose down
    ```
2.  **Stop Traefik:**
    ```bash
    cd traefik
    docker-compose down
    cd ..
    ```
This ensures a clean shutdown of the entire environment. 