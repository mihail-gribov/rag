#!/usr/bin/env python3
"""
Demo script showcasing arXiv RAG system functionality
Usage: python scripts/demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_fetcher import ArxivFetcher


def main():
    print("🚀 arXiv RAG System Demo")
    print("=" * 50)
    
    # Initialize fetcher
    print("🔧 Initializing ArxivFetcher...")
    fetcher = ArxivFetcher()
    print("✅ ArxivFetcher initialized")
    print()
    
    # Demo 1: Search
    print("🔍 Demo 1: Searching for 'RAG' articles...")
    documents = fetcher.search_documents("RAG", 3)
    print(f"✅ Found {len(documents)} documents")
    
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['title'][:60]}...")
    print()
    
    # Demo 2: Download
    print("📥 Demo 2: Downloading documents...")
    downloaded_files = fetcher.fetch_and_download("machine learning", 2)
    print(f"✅ Downloaded {len(downloaded_files)} files")
    
    for file_path in downloaded_files:
        file_size = Path(file_path).stat().st_size / (1024 * 1024)
        print(f"  📄 {Path(file_path).name} ({file_size:.2f} MB)")
    print()
    
    # Demo 3: Metadata
    print("📋 Demo 3: Viewing metadata...")
    all_docs = fetcher.get_downloaded_documents()
    print(f"✅ Total documents in system: {len(all_docs)}")
    
    # Count by category
    categories = {}
    for doc in all_docs:
        for category in doc['categories']:
            categories[category] = categories.get(category, 0) + 1
    
    print("📊 Documents by category:")
    for category, count in sorted(categories.items()):
        print(f"  • {category}: {count}")
    print()
    
    print("🎉 Demo completed successfully!")
    print("💡 Try these commands for more details:")
    print("  python scripts/test_search.py 'your query'")
    print("  python scripts/test_metadata.py --list")
    print("  python scripts/test_system.py")


if __name__ == "__main__":
    main()
