import pdfplumber
import pandas as pd
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePDFProcessor:
    """Simplified PDF processor using only pdfplumber - no PyMuPDF required."""
    
    def __init__(self):
        pass
        
    def process_pdf(self, file_path: str) -> List[Dict]:
        """
        Process PDF with pdfplumber only.
        Returns list of processed chunks with metadata.
        """
        logger.info(f"Processing PDF with simple processor: {file_path}")
        chunks = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"Processing page {page_num + 1}")
                    
                    page_chunks = self._process_page(page, page_num + 1, file_path)
                    chunks.extend(page_chunks)
            
            logger.info(f"Successfully processed {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def _process_page(self, page, page_num: int, file_path: str) -> List[Dict]:
        """Process a single page with pdfplumber."""
        chunks = []
        
        # 1. Extract tables first
        tables = self._extract_tables(page, page_num, file_path)
        chunks.extend(tables)
        
        # 2. Extract text content
        text_chunks = self._extract_text_content(page, page_num, file_path)
        chunks.extend(text_chunks)
        
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
    
    def _extract_text_content(self, page, page_num: int, file_path: str) -> List[Dict]:
        """Extract text content."""
        chunks = []
        
        try:
            # Get text from pdfplumber
            text = page.extract_text()
            
            if text and text.strip():
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
                                'processing_method': 'pdfplumber_text_extraction'
                            }
                        }
                        chunks.append(chunk)
                        
        except Exception as e:
            logger.warning(f"Text extraction failed on page {page_num}: {str(e)}")
        
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
    
    def _intelligent_text_split(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text intelligently, respecting paragraph and sentence boundaries."""
        
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
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks 