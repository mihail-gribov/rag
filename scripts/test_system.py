#!/usr/bin/env python3
"""
Test script for system health check
Usage: python scripts/test_system.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, validate_config
from logging_config import app_logger, perf_logger, user_logger, error_logger


def test_configuration():
    """Test configuration system"""
    print("🔧 Testing Configuration System...")
    
    try:
        # Test config loading
        print(f"  ✅ Config loaded successfully")
        print(f"  📊 arXiv max_results: {config.arxiv.max_results}")
        print(f"  📊 RAG model: {config.rag.model_name}")
        print(f"  📁 Papers dir: {config.paths.papers_dir}")
        
        # Test validation
        is_valid = validate_config(config)
        if is_valid:
            print(f"  ✅ Configuration validation passed")
        else:
            print(f"  ❌ Configuration validation failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_logging():
    """Test logging system"""
    print("📝 Testing Logging System...")
    
    try:
        # Test all loggers
        app_logger.info("Test message from app logger")
        perf_logger.info("Test message from performance logger")
        user_logger.info("Test message from user logger")
        error_logger.error("Test error message")
        
        print(f"  ✅ All loggers working correctly")
        
        # Check log files exist
        log_dir = Path(config.paths.log_dir)
        log_files = ['app.log', 'performance.log', 'user_actions.log', 'errors.log']
        
        for log_file in log_files:
            log_path = log_dir / log_file
            if log_path.exists():
                print(f"  ✅ {log_file} exists")
            else:
                print(f"  ❌ {log_file} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Logging test failed: {e}")
        return False


def test_directories():
    """Test directory structure"""
    print("📁 Testing Directory Structure...")
    
    try:
        required_dirs = [
            config.paths.papers_dir,
            config.paths.output_dir,
            config.paths.log_dir,
            config.paths.metadata_dir
        ]
        
        for dir_path in required_dirs:
            dir_obj = Path(dir_path)
            if dir_obj.exists() and dir_obj.is_dir():
                print(f"  ✅ {dir_path} exists")
            else:
                print(f"  ❌ {dir_path} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Directory test failed: {e}")
        return False


def test_arxiv_fetcher():
    """Test arXiv fetcher basic functionality"""
    print("🔍 Testing arXiv Fetcher...")
    
    try:
        from arxiv_fetcher import ArxivFetcher
        
        # Initialize fetcher
        fetcher = ArxivFetcher()
        print(f"  ✅ ArxivFetcher initialized")
        
        # Test search (small query)
        documents = fetcher.search_documents("test", 1)
        if documents:
            print(f"  ✅ Search functionality working")
            print(f"  📊 Found {len(documents)} test documents")
        else:
            print(f"  ⚠️  No test documents found (this might be normal)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ArXiv fetcher test failed: {e}")
        return False


def main():
    print("🚀 arXiv RAG System Health Check")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_logging,
        test_directories,
        test_arxiv_fetcher
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! System is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
