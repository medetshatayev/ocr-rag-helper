# RAG Chat Assistant

A Retrieval-Augmented Generation (RAG) chat assistant built with Streamlit that processes documents including PDFs with tables, scanned pages, and structured data.

## Features

### Advanced Document Processing
- **PDF Support**: Extract text from both text-based and scanned PDFs
- **OCR Integration**: Automatic OCR for scanned pages using Tesseract
- **Table Extraction**: Intelligent table detection and structured data extraction
- **Multi-format Support**: PDF, TXT, MD, PY, JS, HTML, CSS, JSON, XML, CSV, YAML

### Smart RAG Pipeline
- **Vector Search**: ChromaDB for efficient similarity search
- **Context-Aware Chunking**: Intelligent text splitting that preserves document structure
- **OpenAI Integration**: GPT-4o for high-quality responses
- **Source Attribution**: Shows sources with page numbers and confidence scores

### Modern Interface
- **Clean Chat UI**: Streamlit interface with message history
- **Real-time Processing**: Live document indexing with progress tracking
- **Database Management**: View stats, clear data, force reindexing
- **Responsive Design**: Works on desktop and mobile

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
3. **Tesseract OCR** (for scanned PDF processing)

### Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   
   **Windows:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH or set `TESSDATA_PREFIX`
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

4. **Configure OpenAI API Key:**
   
   Create a `.env` file in your project directory:
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key_here
   
   # Optional Configuration
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   MAX_RETRIEVAL_RESULTS=5
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## Document Processing

The system automatically scans your project directory for supported files:

### Supported File Types
- **PDFs**: Text-based and scanned documents with table extraction
- **Text Files**: .txt, .md, .py, .js, .html, .css
- **Data Files**: .json, .xml, .csv, .yml, .yaml

### Advanced PDF Features

#### Multi-Layer Text Extraction
1. **Direct Text**: Fast extraction from text-based PDFs
2. **Table Detection**: Automatic table identification and formatting
3. **OCR Fallback**: Tesseract OCR for scanned pages
4. **Structure Preservation**: Maintains document hierarchy and context

#### Table Processing
- Converts tables to searchable text and markdown
- Preserves headers and row relationships
- Enables queries like "What are the penalties in the table?"

#### OCR Integration
- Automatic detection of scanned vs. text-based pages
- Confidence scoring for OCR results
- Quality filtering to ensure accurate text extraction

## Usage

### 1. Index Documents
- Click "Index Documents" in the sidebar
- System scans current directory for supported files
- Processing progress shown in real-time

### 2. Chat with Your Documents
- Ask questions in natural language
- Get contextual answers with source citations
- View similarity scores and page references

### 3. Example Queries
```
"What are the main penalties mentioned?"
"Summarize the table on page 5"
"What requirements are listed in section 3?"
"Compare the data from different sections"
```

## Configuration

### Environment Variables
```env
# Required
OPENAI_API_KEY=your_key_here

# Optional
CHUNK_SIZE=1000              # Text chunk size for processing
CHUNK_OVERLAP=200            # Overlap between chunks
MAX_RETRIEVAL_RESULTS=5      # Number of similar chunks to retrieve
```

### OCR Configuration
For better OCR results, you can:
- Adjust confidence threshold in `pdf_processor.py`
- Configure Tesseract language packs
- Tune image preprocessing settings

## Technical Architecture

```
Documents → PDF Processor → Document Processor → RAG System → Streamlit UI
               ↓              ↓                    ↓
            OCR+Tables → Text Chunks → Vector Embeddings → GPT-4o
```

### Core Components

1. **`pdf_processor.py`**: Advanced PDF processing with OCR and table extraction
2. **`document_processor.py`**: Multi-format document handling
3. **`rag_system.py`**: Vector database and OpenAI integration
4. **`main.py`**: Streamlit interface

### Data Flow

1. **Document Scanning**: Recursive directory scan for supported files
2. **Content Extraction**: PDF processing with OCR fallback
3. **Intelligent Chunking**: Structure-aware text splitting
4. **Vector Storage**: ChromaDB with metadata
5. **Query Processing**: Similarity search + GPT-4o generation
6. **Response Delivery**: Formatted answers with source attribution

## Troubleshooting

### Common Issues

**1. Tesseract not found**
```
TesseractNotFoundError: tesseract is not installed
```
- Install Tesseract OCR and add to PATH
- Verify installation: `tesseract --version`

**2. OpenAI API errors**
```
AuthenticationError: Invalid API key
```
- Check your API key in `.env` file
- Ensure you have API credits available

**3. PDF processing fails**
```
Error processing PDF: ...
```
- Check if PDF is corrupted or password-protected
- Try with a simpler PDF first

**4. Memory issues with large PDFs**
- Split large PDFs into smaller files
- Adjust chunk size in configuration
- Monitor system resources

### Performance Tips

1. **Optimize Chunking**: Adjust `CHUNK_SIZE` based on your content
2. **Batch Processing**: System processes documents in batches
3. **Incremental Updates**: Only new files are processed on re-indexing
4. **Database Persistence**: Vector database persists between sessions

## Monitoring

The sidebar shows real-time statistics:
- **Total Documents**: Number of indexed chunks
- **Unique Sources**: Number of processed files
- **Content Types**: Breakdown by document type
- **Processing Status**: Success/error messages

## Security

- API keys stored in environment variables
- Local vector database (no data sent to external services except OpenAI)
- No file uploads to external servers
- All processing happens locally

## Contributing

Feel free to enhance the system:
- Add new document formats
- Improve OCR accuracy
- Enhance table extraction
- Add new features to the UI

## License

This project is open source. Use and modify as needed.

---

For questions or issues, please check the troubleshooting section. 