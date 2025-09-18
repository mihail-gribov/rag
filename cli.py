#!/usr/bin/env python3
"""
arXiv RAG System CLI Interface
Command-line interface for managing arXiv documents and RAG queries
"""

import click
import os
import sys
from pathlib import Path
from typing import Optional

# Import system components
from arxiv_fetcher import ArxivFetcher
from rag_engine import ArxivRAGEngine
from markdown_formatter import MarkdownFormatter
from config import config, validate_config
from logging_config import app_logger, user_logger, error_logger


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """arXiv RAG System - Search and analyze scientific documents"""
    # Store verbose flag in context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Validate configuration
    if not validate_config(config):
        click.echo("Configuration validation failed. Please check your settings.", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo("Configuration loaded successfully")
        click.echo(f"Papers directory: {config.paths.papers_dir}")
        click.echo(f"Output directory: {config.paths.output_dir}")
        click.echo(f"ChromaDB directory: {config.chromadb.persist_directory}")


@cli.command()
@click.argument('query')
@click.option('--max-results', default=None, type=int, help='Maximum number of documents to fetch')
@click.option('--papers-dir', default=None, help='Directory to store downloaded documents')
@click.pass_context
def fetch(ctx, query, max_results, papers_dir):
    """Fetch documents from arXiv.org"""
    verbose = ctx.obj.get('verbose', False)
    
    try:
        # Initialize fetcher
        fetcher = ArxivFetcher(papers_dir)
        
        if verbose:
            click.echo(f"Searching arXiv for: {query}")
            click.echo(f"Max results: {max_results or config.arxiv.max_results}")
            click.echo(f"Papers directory: {fetcher.papers_dir}")
        
        # Fetch and download documents
        downloaded_files = fetcher.fetch_and_download(query, max_results)
        
        if not downloaded_files:
            click.echo("No documents were downloaded.")
            return
        
        click.echo(f"Successfully downloaded {len(downloaded_files)} documents")
        
        # Add documents to RAG engine
        if verbose:
            click.echo("Adding documents to search database...")
        
        rag_engine = ArxivRAGEngine()
        rag_engine.add_documents(str(fetcher.papers_dir))
        
        click.echo("Documents added to search database successfully")
        
        if verbose:
            doc_count = rag_engine.get_document_count()
            click.echo(f"Total documents in database: {doc_count}")
        
    except Exception as e:
        error_msg = f"Failed to fetch documents: {str(e)}"
        click.echo(error_msg, err=True)
        error_logger.error(error_msg, exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument('question')
@click.option('--papers-dir', default=None, help='Directory with documents')
@click.option('--save-to-file', is_flag=True, help='Save response to Markdown file')
@click.option('--output-dir', default=None, help='Directory to save output files')
@click.pass_context
def search(ctx, question, papers_dir, save_to_file, output_dir):
    """Search through downloaded documents"""
    verbose = ctx.obj.get('verbose', False)
    
    try:
        # Check if documents exist
        papers_path = Path(papers_dir or config.paths.papers_dir)
        if not papers_path.exists() or not any(papers_path.glob("*.pdf")):
            click.echo("No documents found. Use 'fetch' command first.")
            return
        
        if verbose:
            click.echo(f"Searching for: {question}")
            click.echo(f"Papers directory: {papers_path}")
            if save_to_file:
                click.echo(f"Output directory: {output_dir or config.paths.output_dir}")
        
        # Initialize RAG engine
        rag_engine = ArxivRAGEngine()
        
        # Perform search
        result = rag_engine.search(question)
        
        if "error" in result:
            click.echo(f"Search failed: {result['error']}", err=True)
            return
        
        # Display or save result
        if save_to_file:
            # Save to file
            formatter = MarkdownFormatter(output_dir)
            filepath = formatter.save_to_file(question, result, output_dir)
            click.echo(f"Response saved to: {filepath}")
            
            # Show preview
            click.echo("\nPreview:")
            click.echo("=" * 50)
            answer_preview = result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"]
            click.echo(answer_preview)
        else:
            # Display in console
            click.echo("\nAnswer:")
            click.echo("=" * 50)
            click.echo(result["answer"])
            
            # Show sources
            if result.get("sources"):
                click.echo("\nSources:")
                click.echo("=" * 50)
                for i, source in enumerate(result["sources"], 1):
                    filename = os.path.basename(source.get("file", "Unknown"))
                    click.echo(f"{i}. {filename} (chunk {source.get('chunk_id', 0)})")
                    content_preview = source.get("content", "")[:100] + "..." if len(source.get("content", "")) > 100 else source.get("content", "")
                    click.echo(f"   {content_preview}")
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        click.echo(error_msg, err=True)
        error_logger.error(error_msg, exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--papers-dir', default=None, help='Directory with documents')
@click.option('--show-metadata', is_flag=True, help='Show detailed metadata for each document')
@click.pass_context
def list_docs(ctx, papers_dir, show_metadata):
    """List downloaded documents"""
    verbose = ctx.obj.get('verbose', False)
    
    try:
        papers_path = Path(papers_dir or config.paths.papers_dir)
        
        if not papers_path.exists():
            click.echo("Documents directory does not exist")
            return
        
        # Get documents
        if show_metadata:
            # Use fetcher to get metadata
            fetcher = ArxivFetcher(papers_dir)
            documents = fetcher.get_downloaded_documents()
            
            if not documents:
                click.echo("No documents found")
                return
            
            click.echo(f"Found {len(documents)} documents:")
            click.echo("=" * 80)
            
            for doc in documents:
                click.echo(f"ID: {doc['id']}")
                click.echo(f"Title: {doc['title']}")
                click.echo(f"Authors: {', '.join(doc['authors'])}")
                click.echo(f"Published: {doc['published_date']}")
                click.echo(f"Categories: {', '.join(doc['categories'])}")
                click.echo("-" * 80)
        else:
            # Simple list
            documents = [f for f in papers_path.glob("*.pdf")]
            
            if not documents:
                click.echo("No documents found")
            else:
                click.echo(f"Found {len(documents)} documents:")
                for document in documents:
                    click.echo(f"  - {document.name}")
        
        if verbose:
            click.echo(f"Papers directory: {papers_path}")
            if show_metadata:
                click.echo(f"Total files: {len(documents)}")
            else:
                click.echo(f"Total files: {len(documents)}")
        
    except Exception as e:
        error_msg = f"Failed to list documents: {str(e)}"
        click.echo(error_msg, err=True)
        error_logger.error(error_msg, exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear(ctx, force):
    """Clear the search database"""
    verbose = ctx.obj.get('verbose', False)
    
    try:
        if not force:
            if not click.confirm("Are you sure you want to clear the database?"):
                click.echo("Operation cancelled")
                return
        
        if verbose:
            click.echo("Clearing search database...")
        
        # Initialize RAG engine and clear database
        rag_engine = ArxivRAGEngine()
        rag_engine.clear_database()
        
        click.echo("Database cleared successfully")
        
        if verbose:
            click.echo(f"ChromaDB directory: {config.chromadb.persist_directory}")
        
    except Exception as e:
        error_msg = f"Failed to clear database: {str(e)}"
        click.echo(error_msg, err=True)
        error_logger.error(error_msg, exc_info=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and statistics"""
    verbose = ctx.obj.get('verbose', False)
    
    try:
        click.echo("arXiv RAG System Status")
        click.echo("=" * 40)
        
        # Check configuration
        click.echo("Configuration:")
        click.echo(f"  OpenAI API Key: {'✓ Set' if config.openai_api_key else '✗ Not set'}")
        click.echo(f"  Model: {config.rag.model_name}")
        click.echo(f"  Papers Directory: {config.paths.papers_dir}")
        click.echo(f"  Output Directory: {config.paths.output_dir}")
        click.echo(f"  ChromaDB Directory: {config.chromadb.persist_directory}")
        
        # Check documents
        papers_path = Path(config.paths.papers_dir)
        if papers_path.exists():
            pdf_files = list(papers_path.glob("*.pdf"))
            click.echo(f"\nDocuments: {len(pdf_files)} PDF files")
        else:
            click.echo("\nDocuments: No papers directory found")
        
        # Check database
        try:
            rag_engine = ArxivRAGEngine()
            doc_count = rag_engine.get_document_count()
            click.echo(f"Database: {doc_count} chunks indexed")
            
            if verbose:
                collection_info = rag_engine.get_collection_info()
                click.echo(f"  Collection: {collection_info.get('collection_name', 'Unknown')}")
                click.echo(f"  Embedding Model: {collection_info.get('embedding_model', 'Unknown')}")
                click.echo(f"  LLM Model: {collection_info.get('llm_model', 'Unknown')}")
        except Exception as e:
            click.echo(f"Database: Error - {str(e)}")
        
    except Exception as e:
        error_msg = f"Failed to get status: {str(e)}"
        click.echo(error_msg, err=True)
        error_logger.error(error_msg, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
