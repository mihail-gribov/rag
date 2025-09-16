"""
arXiv Fetcher Module
Handles downloading and processing documents from arXiv.org
"""

import os
import requests
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
from config import config
from logging_config import app_logger, user_logger, error_logger


class ArxivFetcher:
    """Handles fetching documents from arXiv.org"""
    
    def __init__(self, papers_dir: Optional[str] = None):
        self.papers_dir = Path(papers_dir or config.paths.papers_dir)
        self.papers_dir.mkdir(exist_ok=True)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.metadata_dir.mkdir(exist_ok=True)
        self.logger = app_logger.getChild('arxiv_fetcher')
    
    def search_documents(self, query: str, max_results: int = None) -> List[Dict]:
        """Search for documents on arXiv"""
        max_results = max_results or config.arxiv.max_results
        
        self.logger.info(f"Searching arXiv for query: '{query}' (max_results: {max_results})")
        user_logger.info(f"User search query: '{query}' (max_results: {max_results})")
        
        try:
            # Use arXiv API directly via HTTP
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(base_url, params=params, timeout=config.arxiv.timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            documents = []
            for entry in root.findall('atom:entry', ns):
                document_info = self._parse_arxiv_entry(entry, ns)
                if document_info:
                    documents.append(document_info)
            
            self.logger.info(f"Found {len(documents)} documents for query: '{query}'")
            return documents
            
        except Exception as e:
            error_msg = f"Failed to search arXiv for query '{query}': {str(e)}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise
    
    def download_document(self, document_info: Dict) -> Optional[str]:
        """Download PDF document"""
        document_id = document_info["id"]
        pdf_url = document_info["pdf_url"]
        filename = f"{document_id}.pdf"
        filepath = self.papers_dir / filename
        
        self.logger.info(f"Starting download of document: {document_id}")
        user_logger.info(f"User downloading document: {document_info['title']}")
        
        # Check if file already exists
        if filepath.exists():
            self.logger.info(f"Document {document_id} already exists, skipping download")
            return str(filepath)
        
        try:
            response = requests.get(pdf_url, timeout=config.arxiv.timeout)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Save metadata
            self._save_metadata(document_info, str(filepath))
            
            self.logger.info(f"Successfully downloaded document: {document_id}")
            user_logger.info(f"Successfully downloaded: {document_info['title']}")
            
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Failed to download document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return None
    
    def fetch_and_download(self, query: str, max_results: int = None) -> List[str]:
        """Search and download documents"""
        max_results = max_results or config.arxiv.max_results
        
        self.logger.info(f"Starting fetch and download for query: '{query}' (max_results: {max_results})")
        user_logger.info(f"User fetch and download: '{query}' (max_results: {max_results})")
        
        try:
            # Search for documents
            documents = self.search_documents(query, max_results)
            
            if not documents:
                self.logger.warning(f"No documents found for query: '{query}'")
                return []
            
            # Download documents
            downloaded_files = []
            for document in documents:
                filepath = self.download_document(document)
                if filepath:
                    downloaded_files.append(filepath)
            
            self.logger.info(f"Successfully downloaded {len(downloaded_files)} out of {len(documents)} documents")
            user_logger.info(f"Downloaded {len(downloaded_files)} documents for query: '{query}'")
            
            return downloaded_files
            
        except Exception as e:
            error_msg = f"Failed to fetch and download documents for query '{query}': {str(e)}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise
    
    def _parse_arxiv_entry(self, entry, ns) -> Optional[Dict]:
        """Parse arXiv entry from XML"""
        try:
            # Extract ID
            entry_id = entry.find('atom:id', ns).text
            arxiv_id = self._extract_arxiv_id(entry_id)
            
            # Extract title
            title = entry.find('atom:title', ns).text.strip()
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # Extract abstract
            abstract = entry.find('atom:summary', ns)
            abstract_text = abstract.text.strip() if abstract is not None else ""
            
            # Extract published date
            published = entry.find('atom:published', ns)
            published_date = datetime.fromisoformat(published.text.replace('Z', '+00:00')) if published is not None else datetime.now()
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Construct PDF URL
            pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
            
            return {
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract_text,
                "published": published_date,
                "pdf_url": pdf_url,
                "categories": categories,
                "entry_id": entry_id
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse arXiv entry: {e}")
            return None
    
    def _extract_arxiv_id(self, entry_id: str) -> str:
        """Extract arXiv ID from entry ID"""
        # arXiv entry IDs are in format: http://arxiv.org/abs/2023.12345v1
        # We want just the ID part: 2023.12345
        return entry_id.split('/')[-1].split('v')[0]
    
    def _save_metadata(self, document_info: Dict, pdf_path: str) -> None:
        """Save document metadata to JSON file"""
        document_id = document_info["id"]
        metadata = {
            "id": document_id,
            "title": document_info["title"],
            "authors": document_info["authors"],
            "abstract": document_info["abstract"],
            "published_date": document_info["published"].isoformat(),
            "download_date": datetime.now().isoformat(),
            "download_url": document_info["pdf_url"],
            "categories": document_info["categories"],
            "pdf_path": pdf_path,
            "entry_id": document_info["entry_id"]
        }
        
        metadata_file = self.metadata_dir / f"{document_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved metadata for document {document_id}")
    
    def get_downloaded_documents(self) -> List[Dict]:
        """Get list of downloaded documents with metadata"""
        documents = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                documents.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        return documents
    
    def is_document_downloaded(self, document_id: str) -> bool:
        """Check if document is already downloaded"""
        pdf_path = self.papers_dir / f"{document_id}.pdf"
        metadata_path = self.metadata_dir / f"{document_id}.json"
        return pdf_path.exists() and metadata_path.exists()
