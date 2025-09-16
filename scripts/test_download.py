#!/usr/bin/env python3
"""
Test script for arXiv download functionality
Usage: python scripts/test_download.py "search query" [--max-results N]
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_fetcher import ArxivFetcher


def main():
    parser = argparse.ArgumentParser(description='Test arXiv download functionality')
    parser.add_argument('query', help='Search query for arXiv')
    parser.add_argument('--max-results', type=int, default=3, help='Maximum number of documents to download (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"🔍 Searching and downloading from arXiv: '{args.query}'")
    print(f"📊 Max results: {args.max_results}")
    print("=" * 80)
    
    try:
        # Initialize fetcher
        fetcher = ArxivFetcher()
        
        # Search and download documents
        downloaded_files = fetcher.fetch_and_download(args.query, args.max_results)
        
        if not downloaded_files:
            print("❌ No documents were downloaded")
            return
        
        print(f"✅ Successfully downloaded {len(downloaded_files)} documents:")
        print()
        
        # Display downloaded files
        for i, file_path in enumerate(downloaded_files, 1):
            file_size = Path(file_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            print(f"{i:2d}. 📄 {Path(file_path).name}")
            print(f"    📁 Path: {file_path}")
            print(f"    📊 Size: {file_size_mb:.2f} MB")
            print()
        
        # Show metadata files
        metadata_dir = Path("metadata")
        if metadata_dir.exists():
            metadata_files = list(metadata_dir.glob("*.json"))
            print(f"📋 Metadata files created: {len(metadata_files)}")
            for metadata_file in metadata_files:
                print(f"    📄 {metadata_file.name}")
        
        print("=" * 80)
        print(f"✅ Download completed successfully. {len(downloaded_files)} files downloaded.")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
