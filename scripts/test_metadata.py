#!/usr/bin/env python3
"""
Test script for viewing downloaded document metadata
Usage: python scripts/test_metadata.py [--document-id ID]
"""

import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_fetcher import ArxivFetcher


def main():
    parser = argparse.ArgumentParser(description='View downloaded document metadata')
    parser.add_argument('--document-id', help='Specific document ID to view')
    parser.add_argument('--list', action='store_true', help='List all downloaded documents')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize fetcher
        fetcher = ArxivFetcher()
        
        if args.list:
            # List all downloaded documents
            documents = fetcher.get_downloaded_documents()
            
            if not documents:
                print("❌ No downloaded documents found")
                return
            
            print(f"📚 Downloaded documents ({len(documents)}):")
            print("=" * 80)
            
            for i, doc in enumerate(documents, 1):
                print(f"{i:2d}. 📄 {doc['title']}")
                print(f"    🆔 ID: {doc['id']}")
                print(f"    👥 Authors: {', '.join(doc['authors'][:3])}{'...' if len(doc['authors']) > 3 else ''}")
                print(f"    📅 Published: {doc['published_date']}")
                print(f"    📥 Downloaded: {doc['download_date']}")
                print(f"    🏷️  Categories: {', '.join(doc['categories'][:3])}{'...' if len(doc['categories']) > 3 else ''}")
                print(f"    📁 PDF: {doc['pdf_path']}")
                print()
        
        elif args.document_id:
            # View specific document metadata
            metadata_file = Path("metadata") / f"{args.document_id}.json"
            
            if not metadata_file.exists():
                print(f"❌ Metadata file not found: {metadata_file}")
                return
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"📄 Document Metadata: {metadata['title']}")
            print("=" * 80)
            print(f"🆔 ID: {metadata['id']}")
            print(f"📅 Published: {metadata['published_date']}")
            print(f"📥 Downloaded: {metadata['download_date']}")
            print(f"🔗 PDF URL: {metadata['download_url']}")
            print(f"📁 Local PDF: {metadata['pdf_path']}")
            print(f"🏷️  Categories: {', '.join(metadata['categories'])}")
            print()
            print(f"👥 Authors:")
            for author in metadata['authors']:
                print(f"    • {author}")
            print()
            print(f"📝 Abstract:")
            print(metadata['abstract'])
            
        else:
            # Default: show summary
            documents = fetcher.get_downloaded_documents()
            
            if not documents:
                print("❌ No downloaded documents found")
                return
            
            print(f"📚 Downloaded Documents Summary")
            print("=" * 50)
            print(f"Total documents: {len(documents)}")
            
            # Count by category
            categories = {}
            for doc in documents:
                for category in doc['categories']:
                    categories[category] = categories.get(category, 0) + 1
            
            print(f"Categories:")
            for category, count in sorted(categories.items()):
                print(f"  • {category}: {count}")
            
            print()
            print("Use --list to see all documents or --document-id ID to view specific document")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
