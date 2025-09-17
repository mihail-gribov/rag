"""
Markdown Formatter Module
Handles formatting responses and saving to files
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from config import config


class MarkdownFormatter:
    """Handles Markdown formatting and file operations"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize Markdown Formatter
        
        Args:
            output_dir: Directory to save output files (default from config)
        """
        self.output_dir = Path(output_dir or config.paths.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def format_response(self, query: str, result: Dict) -> str:
        """
        Format RAG response as Markdown
        
        Args:
            query: Original search query
            result: RAG result dictionary with 'answer' and 'sources'
            
        Returns:
            Formatted Markdown string
        """
        if "error" in result:
            return f"# Error\n\n{result['error']}"
        
        # Create markdown content
        md_content = []
        
        # Header
        md_content.append(f"# Research Query: {query}")
        md_content.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append("")
        
        # Answer section
        md_content.append("## Answer")
        md_content.append("")
        md_content.append(result["answer"])
        md_content.append("")
        
        # Sources section
        if result.get("sources"):
            md_content.append("## Sources")
            md_content.append("")
            
            for i, source in enumerate(result["sources"], 1):
                filename = os.path.basename(source.get("file", "Unknown"))
                md_content.append(f"### Source {i}: {filename}")
                md_content.append("")
                md_content.append(f"**Chunk ID:** {source.get('chunk_id', 0)}")
                md_content.append("")
                md_content.append("**Content:**")
                md_content.append("")
                md_content.append(f"> {source.get('content', 'No content available')}")
                md_content.append("")
        
        return "\n".join(md_content)
    
    def save_to_file(self, query: str, result: Dict, output_dir: str = None) -> str:
        """
        Save formatted response to file
        
        Args:
            query: Original search query
            result: RAG result dictionary
            output_dir: Override output directory (optional)
            
        Returns:
            Path to saved file
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = self.output_dir
        
        # Generate filename from query
        safe_query = self._generate_safe_filename(query)
        timestamp = datetime.now().strftime('%Y-%m-%d')
        filename = f"{safe_query}_{timestamp}.md"
        
        filepath = output_path / filename
        
        # Format and save
        md_content = self.format_response(query, result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(filepath)
    
    def format_sources_list(self, sources: List[Dict]) -> str:
        """
        Format sources as a simple list
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Formatted sources list in Markdown
        """
        if not sources:
            return "No sources found."
        
        md_content = []
        md_content.append("## Sources")
        md_content.append("")
        
        for i, source in enumerate(sources, 1):
            filename = os.path.basename(source.get("file", "Unknown"))
            md_content.append(f"{i}. **{filename}** (chunk {source.get('chunk_id', 0)})")
            md_content.append(f"   {source.get('content', 'No content')[:100]}...")
            md_content.append("")
        
        return "\n".join(md_content)
    
    def _generate_safe_filename(self, query: str) -> str:
        """
        Generate safe filename from query
        
        Args:
            query: Original query string
            
        Returns:
            Safe filename string
        """
        # Remove special characters and limit length
        safe_query = re.sub(r'[^\w\s-]', '', query)
        safe_query = re.sub(r'[-\s]+', '_', safe_query)
        safe_query = safe_query.strip('_').lower()
        
        # Limit to 50 characters
        if len(safe_query) > 50:
            safe_query = safe_query[:50].rstrip('_')
        
        # Ensure it's not empty
        if not safe_query:
            safe_query = "query"
        
        return safe_query