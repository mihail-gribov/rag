"""
Document Parser Module
Handles PDF text extraction and chunking
"""

import os
import logging
from typing import List, Dict
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

from config import config


class DocumentParser:
    """Handles PDF document parsing and text chunking"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize DocumentParser with chunking parameters
        
        Args:
            chunk_size: Size of text chunks in characters (default from config)
            chunk_overlap: Overlap between chunks in characters (default from config)
        """
        self.chunk_size = chunk_size or config.document.chunk_size
        self.chunk_overlap = chunk_overlap or config.document.chunk_overlap
        
        # Initialize logger
        self.logger = logging.getLogger('arxiv_rag.document_parser')
        
        # Initialize text splitter
        if RecursiveCharacterTextSplitter:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
        else:
            self.logger.warning("langchain not available, using simple text splitting")
            self.text_splitter = None
    
    def parse_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a PDF or PyPDF2 is not available
            Exception: For other PDF parsing errors
        """
        if not PyPDF2:
            raise ValueError("PyPDF2 is required for PDF parsing. Install with: pip install pypdf2")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.document.max_file_size_mb:
            raise ValueError(f"PDF file too large: {file_size_mb:.1f}MB > {config.document.max_file_size_mb}MB")
        
        self.logger.info(f"Parsing PDF: {file_path.name} ({file_size_mb:.1f}MB)")
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    self.logger.warning(f"PDF is encrypted, skipping: {file_path.name}")
                    return ""
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
            
            # Clean up text
            text = self._clean_text(text)
            
            self.logger.info(f"Extracted {len(text)} characters from {file_path.name}")
            return text
            
        except PyPDF2.PdfReadError as e:
            error_msg = f"Error reading PDF {file_path.name}: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing PDF {file_path.name}: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and normalizing
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are just whitespace characters
            if line.isspace():
                continue
            
            cleaned_lines.append(line)
        
        # Join lines with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive newlines
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text
    
    def _split_text_simple(self, text: str) -> List[str]:
        """
        Simple text splitting when langchain is not available
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Find a good break point (end of sentence or paragraph)
            break_point = end
            
            # Look for sentence endings
            for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
                if text[i] in '.!?':
                    break_point = i + 1
                    break
            
            # Look for paragraph breaks
            if break_point == end:
                for i in range(end, max(start + self.chunk_size // 2, end - 50), -1):
                    if text[i] == '\n':
                        break_point = i
                        break
            
            chunk = text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = break_point - self.chunk_overlap
            if start < 0:
                start = break_point
        
        return chunks
    
    def parse_document(self, file_path: str) -> List[str]:
        """
        Parse PDF document and return text chunks
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of text chunks
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a PDF
            Exception: For other parsing errors
        """
        self.logger.info(f"Starting document parsing: {file_path}")
        
        try:
            # Extract text from PDF
            text = self.parse_pdf(file_path)
            
            if not text:
                self.logger.warning(f"No text extracted from {file_path}")
                return []
            
            # Split text into chunks
            if self.text_splitter:
                chunks = self.text_splitter.split_text(text)
            else:
                chunks = self._split_text_simple(text)
            
            self.logger.info(f"Created {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to parse document {file_path}: {e}")
            raise
    
    def get_document_metadata(self, file_path: str) -> Dict:
        """
        Get document metadata
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with document metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file statistics
        stat = file_path.stat()
        file_size_mb = stat.st_size / (1024 * 1024)
        
        metadata = {
            "file": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower().lstrip('.'),
            "file_size_bytes": stat.st_size,
            "file_size_mb": round(file_size_mb, 2),
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
        }
        
        # Add PDF-specific metadata if available
        if file_path.suffix.lower() == '.pdf' and PyPDF2:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata.update({
                        "page_count": len(pdf_reader.pages),
                        "is_encrypted": pdf_reader.is_encrypted,
                    })
                    
                    # Try to get PDF metadata
                    if pdf_reader.metadata:
                        pdf_meta = pdf_reader.metadata
                        metadata.update({
                            "title": pdf_meta.get('/Title', ''),
                            "author": pdf_meta.get('/Author', ''),
                            "subject": pdf_meta.get('/Subject', ''),
                            "creator": pdf_meta.get('/Creator', ''),
                            "producer": pdf_meta.get('/Producer', ''),
                            "creation_date": pdf_meta.get('/CreationDate', ''),
                            "modification_date": pdf_meta.get('/ModDate', ''),
                        })
            except Exception as e:
                self.logger.warning(f"Could not extract PDF metadata from {file_path.name}: {e}")
        
        return metadata
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
