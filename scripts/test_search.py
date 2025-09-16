#!/usr/bin/env python3
"""
Test script for arXiv search functionality
Usage: python scripts/test_search.py "search query"
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_fetcher import ArxivFetcher


def main():
    parser = argparse.ArgumentParser(description='Test arXiv search functionality')
    parser.add_argument('query', help='Search query for arXiv')
    parser.add_argument('--max-results', type=int, default=10, help='Maximum number of results (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"🔍 Searching arXiv for: '{args.query}'")
    print(f"📊 Max results: {args.max_results}")
    print("=" * 80)
    
    try:
        # Initialize fetcher
        fetcher = ArxivFetcher()
        
        # Search for documents
        documents = fetcher.search_documents(args.query, args.max_results)
        
        if not documents:
            print("❌ No documents found")
            return
        
        print(f"✅ Found {len(documents)} documents:")
        print()
        
        # Display results
        for i, doc in enumerate(documents, 1):
            print(f"{i:2d}. 📄 {doc['title']}")
            print(f"    👥 Authors: {', '.join(doc['authors'][:3])}{'...' if len(doc['authors']) > 3 else ''}")
            print(f"    📅 Published: {doc['published'].strftime('%Y-%m-%d')}")
            print(f"    🏷️  Categories: {', '.join(doc['categories'][:3])}{'...' if len(doc['categories']) > 3 else ''}")
            print(f"    🔗 PDF URL: {doc['pdf_url']}")
            print(f"    📝 Abstract: {doc['abstract'][:150]}{'...' if len(doc['abstract']) > 150 else ''}")
            print()
            
            if args.verbose:
                print(f"    📋 Full Abstract:")
                print(f"    {doc['abstract']}")
                print()
        
        print("=" * 80)
        print(f"✅ Search completed successfully. Found {len(documents)} documents.")
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
