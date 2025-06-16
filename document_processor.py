import os
import logging
from typing import List, Dict, Set
from pathlib import Path
import mimetypes

try:
    from pdf_processor import PDFProcessor
    PYMUPDF_AVAILABLE = True
except ImportError as e:
    print(f"PyMuPDF not available, using simple PDF processor: {e}")
    from pdf_processor_simple import SimplePDFProcessor as PDFProcessor
    PYMUPDF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main document processor that handles multiple file types."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.supported_extensions = {
            '.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', 
            '.json', '.xml', '.csv', '.yml', '.yaml'
        }
        self.processed_files = set()
        
        # Log which PDF processor is being used
        if PYMUPDF_AVAILABLE:
            logger.info("Using advanced PDF processor with PyMuPDF + OCR support")
        else:
            logger.info("Using simple PDF processor with pdfplumber only (no OCR)")
        
    def scan_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """Scan directory for supported document files."""
        files = []
        directory_path = Path(directory_path)
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                # Skip hidden files and common excluded directories
                if not self._should_skip_file(file_path):
                    files.append(str(file_path))
        
        logger.info(f"Found {len(files)} supported files in {directory_path}")
        return files
    
    def process_documents(self, file_paths: List[str]) -> List[Dict]:
        """Process multiple documents and return chunks with metadata."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                if file_path not in self.processed_files:
                    logger.info(f"Processing: {file_path}")
                    chunks = self._process_single_file(file_path)
                    all_chunks.extend(chunks)
                    self.processed_files.add(file_path)
                else:
                    logger.info(f"Skipping already processed file: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        return all_chunks
    
    def _process_single_file(self, file_path: str) -> List[Dict]:
        """Process a single file based on its type."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._process_pdf(str(file_path))
        else:
            return self._process_text_file(str(file_path))
    
    def _process_pdf(self, file_path: str) -> List[Dict]:
        """Process PDF using specialized PDF processor."""
        return self.pdf_processor.process_pdf(file_path)
    
    def _process_text_file(self, file_path: str) -> List[Dict]:
        """Process text-based files."""
        chunks = []
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.error(f"Could not decode file: {file_path}")
                return chunks
            
            if content.strip():
                # Split content into chunks
                text_chunks = self._split_text(content)
                
                for i, chunk_text in enumerate(text_chunks):
                    if len(chunk_text.strip()) > 50:
                        chunk = {
                            'content': chunk_text.strip(),
                            'metadata': {
                                'source': file_path,
                                'content_type': 'text',
                                'file_type': Path(file_path).suffix,
                                'chunk_id': f"text_{Path(file_path).stem}_{i+1}",
                                'processing_method': 'text_file_processing'
                            }
                        }
                        chunks.append(chunk)
                        
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence ending
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            # Move start forward with overlap
            start = end - overlap
            
        return chunks
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if file should be skipped."""
        skip_patterns = {
            # Hidden files
            file_path.name.startswith('.'),
            # Common build/cache directories
            any(part in ['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env'] 
                for part in file_path.parts),
            # Large files (> 10MB)
            file_path.stat().st_size > 10 * 1024 * 1024 if file_path.exists() else False
        }
        
        return any(skip_patterns)
    
    def get_file_stats(self, directory_path: str) -> Dict:
        """Get statistics about files in directory."""
        files = self.scan_directory(directory_path)
        
        stats = {
            'total_files': len(files),
            'by_extension': {},
            'total_size_mb': 0
        }
        
        for file_path in files:
            path = Path(file_path)
            ext = path.suffix.lower()
            
            stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
            
            try:
                stats['total_size_mb'] += path.stat().st_size / (1024 * 1024)
            except:
                pass
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats 