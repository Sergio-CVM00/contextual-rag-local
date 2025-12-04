"""Document Processing - Chunking and Metadata"""

import PyPDF2
from typing import List, Dict, Tuple
from pathlib import Path


class DocumentProcessor:
    """Process documents: load, chunk, and generate metadata"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: float = 0.2):
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap)
    
    def load_document(self, file_path: str) -> Tuple[str, str]:
        """Load document from file (PDF or TXT)"""
        path = Path(file_path)
        filename = path.name
        
        try:
            if file_path.lower().endswith('.pdf'):
                return self._load_pdf(file_path), filename
            else:
                return self._load_text(file_path), filename
        except Exception as e:
            print(f"Error loading document: {e}")
            return "", filename
    
    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def _load_text(self, file_path: str) -> str:
        """Load plain text file with fallback encodings"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all encodings fail, try binary with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Find sentence boundary if possible
            if end < len(text):
                # Look for period, newline, or question mark
                for boundary_char in ['. ', '\n', '? ']:
                    pos = text.rfind(boundary_char, start, end)
                    if pos > start:
                        end = pos + len(boundary_char)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def create_metadata(self, filename: str, chunk_index: int, total_chunks: int) -> Dict:
        """Create metadata for a chunk"""
        return {
            "source_file": filename,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks
        }
