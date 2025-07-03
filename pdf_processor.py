import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import io
import logging
from typing import List, Dict, Tuple, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Advanced PDF processor with OCR, table extraction, and structure-aware parsing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.confidence_threshold = 60  # OCR confidence threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_pdf(self, file_path: str) -> List[Dict]:
        """
        Process PDF with multiple extraction methods.
        Returns list of processed chunks with metadata.
        """
        logger.info(f"Processing PDF: {file_path}")
        chunks = []
        
        try:
            # Open with both libraries for different capabilities
            pymupdf_doc = fitz.open(file_path)
            
            with pdfplumber.open(file_path) as plumber_pdf:
                for page_num in range(len(pymupdf_doc)):
                    logger.info(f"Processing page {page_num + 1}")
                    
                    # Process with pdfplumber (better for tables)
                    plumber_page = plumber_pdf.pages[page_num]
                    page_chunks = self._process_page(
                        pymupdf_doc[page_num], 
                        plumber_page, 
                        page_num + 1, 
                        file_path
                    )
                    chunks.extend(page_chunks)
            
            pymupdf_doc.close()
            logger.info(f"Successfully processed {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def _process_page(self, pymupdf_page, plumber_page, page_num: int, file_path: str) -> List[Dict]:
        """Process a single page with multiple extraction methods."""
        chunks = []
        
        # 1. Extract tables first
        tables = self._extract_tables(plumber_page, page_num, file_path)
        chunks.extend(tables)
        
        # 2. Extract text (excluding table areas)
        text_chunks = self._extract_text_content(
            pymupdf_page, plumber_page, page_num, file_path
        )
        chunks.extend(text_chunks)
        
        # 3. OCR fallback for low-text pages
        if self._needs_ocr(pymupdf_page, chunks):
            ocr_chunks = self._perform_ocr(pymupdf_page, page_num, file_path)
            chunks.extend(ocr_chunks)
        
        return chunks
    
    def _extract_tables(self, page, page_num: int, file_path: str) -> List[Dict]:
        """Extract and format tables from page."""
        chunks = []
        
        try:
            tables = page.extract_tables()
            
            for i, table in enumerate(tables):
                if table and len(table) > 1:  # Ensure table has header + data
                    # Convert table to DataFrame for better handling
                    df = pd.DataFrame(table[1:], columns=table[0])
                    
                    # Create markdown representation
                    table_markdown = self._table_to_markdown(df)
                    
                    # Create searchable text version
                    table_text = self._table_to_text(df)
                    
                    chunk = {
                        'content': f"Table {i+1}:\n{table_markdown}\n\nSearchable content:\n{table_text}",
                        'metadata': {
                            'source': file_path,
                            'page': page_num,
                            'content_type': 'table',
                            'table_id': f"table_{page_num}_{i+1}",
                            'table_structure': df.to_dict(),
                            'processing_method': 'pdfplumber_table_extraction'
                        }
                    }
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_num}: {str(e)}")
        
        return chunks
    
    def _extract_text_content(self, pymupdf_page, plumber_page, page_num: int, file_path: str) -> List[Dict]:
        """Extract text content, excluding table areas."""
        chunks = []
        
        try:
            # Get text from PyMuPDF (generally more reliable for text)
            text = pymupdf_page.get_text()
            
            if text.strip():
                # Split into logical chunks
                text_chunks = self._intelligent_text_split(text)
                
                for i, chunk_text in enumerate(text_chunks):
                    if len(chunk_text.strip()) > 50:  # Only meaningful chunks
                        chunk = {
                            'content': chunk_text.strip(),
                            'metadata': {
                                'source': file_path,
                                'page': page_num,
                                'content_type': 'text',
                                'chunk_id': f"text_{page_num}_{i+1}",
                                'processing_method': 'pymupdf_text_extraction'
                            }
                        }
                        chunks.append(chunk)
                        
        except Exception as e:
            logger.warning(f"Text extraction failed on page {page_num}: {str(e)}")
        
        return chunks
    
    def _needs_ocr(self, page, existing_chunks: List[Dict]) -> bool:
        """Determine if OCR is needed for this page."""
        # Check if we have minimal text content
        text_chunks = [c for c in existing_chunks if c['metadata']['content_type'] == 'text']
        total_text_length = sum(len(c['content']) for c in text_chunks)
        
        # If very little text was extracted, page might be scanned
        return total_text_length < 100
    
    def _perform_ocr(self, page, page_num: int, file_path: str) -> List[Dict]:
        """Perform OCR on page image."""
        chunks = []
        
        try:
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR with confidence data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text with confidence filtering
            ocr_text = []
            for i in range(len(ocr_data['text'])):
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if confidence > self.confidence_threshold and text:
                    ocr_text.append(text)
            
            if ocr_text:
                full_ocr_text = ' '.join(ocr_text)
                
                chunk = {
                    'content': full_ocr_text,
                    'metadata': {
                        'source': file_path,
                        'page': page_num,
                        'content_type': 'ocr_text',
                        'chunk_id': f"ocr_{page_num}",
                        'processing_method': 'tesseract_ocr',
                        'avg_confidence': sum(int(c) for c in ocr_data['conf'] if int(c) > 0) / len([c for c in ocr_data['conf'] if int(c) > 0])
                    }
                }
                chunks.append(chunk)
                
        except Exception as e:
            logger.warning(f"OCR failed on page {page_num}: {str(e)}")
        
        return chunks
    
    def _table_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown table."""
        return df.to_markdown(index=False)
    
    def _table_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to searchable text."""
        text_parts = []
        
        # Add headers
        headers = " | ".join(str(col) for col in df.columns)
        text_parts.append(f"Headers: {headers}")
        
        # Add row data
        for idx, row in df.iterrows():
            row_text = " | ".join(str(val) for val in row.values)
            text_parts.append(f"Row {idx + 1}: {row_text}")
        
        return "\n".join(text_parts)
    
    def _intelligent_text_split(self, text: str) -> List[str]:
        """Split text intelligently, respecting paragraph and sentence boundaries."""
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap

        # First split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + '\n\n' + paragraph
            else:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks 