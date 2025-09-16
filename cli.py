#!/usr/bin/env python3
"""
arXiv RAG System CLI Interface
Command-line interface for managing arXiv documents and RAG queries
"""

import click
import os
from pathlib import Path

# Import will be available after implementing other modules
# from arxiv_fetcher import ArxivFetcher
# from rag_engine import ArxivRAGEngine
# from markdown_formatter import MarkdownFormatter


@click.group()
def cli():
    """arXiv RAG System - Search and analyze scientific documents"""
    pass


@cli.command()
@click.argument('query')
@click.option('--max-results', default=10, help='Maximum number of documents to fetch')
@click.option('--papers-dir', default='./papers', help='Directory to store downloaded documents')
def fetch(query, max_results, papers_dir):
    """Fetch documents from arXiv.org"""
    click.echo(f"Searching arXiv for: {query}")
    click.echo(f"Max results: {max_results}")
    click.echo(f"Papers directory: {papers_dir}")
    
    # TODO: Implement actual fetching logic
    click.echo("Fetch functionality will be implemented in task 004")


@cli.command()
@click.argument('question')
@click.option('--papers-dir', default='./papers', help='Directory with documents')
@click.option('--save-to-file', is_flag=True, help='Save response to Markdown file')
@click.option('--output-dir', default='./output', help='Directory to save output files')
def search(question, papers_dir, save_to_file, output_dir):
    """Search through downloaded documents"""
    if not os.path.exists(papers_dir) or not os.listdir(papers_dir):
        click.echo("No documents found. Use 'fetch' command first.")
        return
    
    click.echo(f"Searching for: {question}")
    click.echo(f"Save to file: {save_to_file}")
    click.echo(f"Output directory: {output_dir}")
    
    # TODO: Implement actual search logic
    click.echo("Search functionality will be implemented in task 006")


@cli.command()
@click.option('--papers-dir', default='./papers', help='Directory with documents')
def list(papers_dir):
    """List downloaded documents"""
    if not os.path.exists(papers_dir):
        click.echo("Documents directory does not exist")
        return
    
    documents = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
    
    if not documents:
        click.echo("No documents found")
    else:
        click.echo(f"Found {len(documents)} documents:")
        for document in documents:
            click.echo(f"  - {document}")


@cli.command()
def clear():
    """Clear the search database"""
    if click.confirm("Are you sure you want to clear the database?"):
        click.echo("Database cleared")
        # TODO: Implement actual clearing logic
        click.echo("Clear functionality will be implemented in task 006")


if __name__ == '__main__':
    cli()
