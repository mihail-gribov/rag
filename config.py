"""
Configuration Module
Handles loading and managing application configuration
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ArxivConfig:
    max_results: int = 10
    sort_by: str = "relevance"
    timeout: int = 30


@dataclass
class DocumentConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 50


@dataclass
class RAGConfig:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 1000
    top_k: int = 3


@dataclass
class ChromaDBConfig:
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "arxiv_documents"
    persist_directory: str = "./chroma_db"


@dataclass
class PathsConfig:
    papers_dir: str = "./papers"
    output_dir: str = "./output"
    log_dir: str = "./log"
    metadata_dir: str = "./metadata"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_rotation: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class AppConfig:
    arxiv: ArxivConfig
    document: DocumentConfig
    rag: RAGConfig
    chromadb: ChromaDBConfig
    paths: PathsConfig
    logging: LoggingConfig
    
    # Secret data from .env
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML file and environment variables"""
    
    # Load public configuration from YAML
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Create configuration objects with defaults
    arxiv_config = ArxivConfig(**config_data.get('arxiv', {}))
    document_config = DocumentConfig(**config_data.get('document', {}))
    rag_config = RAGConfig(**config_data.get('rag', {}))
    chromadb_config = ChromaDBConfig(**config_data.get('chromadb', {}))
    paths_config = PathsConfig(**config_data.get('paths', {}))
    logging_config = LoggingConfig(**config_data.get('logging', {}))
    
    # Override with environment variables if present
    _apply_env_overrides(arxiv_config, document_config, rag_config, 
                        chromadb_config, paths_config, logging_config)
    
    # Load secret data from environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_organization = os.getenv('OPENAI_ORGANIZATION')
    
    return AppConfig(
        arxiv=arxiv_config,
        document=document_config,
        rag=rag_config,
        chromadb=chromadb_config,
        paths=paths_config,
        logging=logging_config,
        openai_api_key=openai_api_key,
        openai_organization=openai_organization
    )


def _apply_env_overrides(arxiv_config, document_config, rag_config, 
                        chromadb_config, paths_config, logging_config):
    """Apply environment variable overrides to configuration objects"""
    
    # arXiv overrides
    if os.getenv('ARXIV_MAX_RESULTS'):
        arxiv_config.max_results = int(os.getenv('ARXIV_MAX_RESULTS'))
    if os.getenv('ARXIV_SORT_BY'):
        arxiv_config.sort_by = os.getenv('ARXIV_SORT_BY')
    if os.getenv('ARXIV_TIMEOUT'):
        arxiv_config.timeout = int(os.getenv('ARXIV_TIMEOUT'))
    
    # Document overrides
    if os.getenv('DOCUMENT_CHUNK_SIZE'):
        document_config.chunk_size = int(os.getenv('DOCUMENT_CHUNK_SIZE'))
    if os.getenv('DOCUMENT_CHUNK_OVERLAP'):
        document_config.chunk_overlap = int(os.getenv('DOCUMENT_CHUNK_OVERLAP'))
    if os.getenv('DOCUMENT_MAX_FILE_SIZE_MB'):
        document_config.max_file_size_mb = int(os.getenv('DOCUMENT_MAX_FILE_SIZE_MB'))
    
    # RAG overrides
    if os.getenv('RAG_MODEL_NAME'):
        rag_config.model_name = os.getenv('RAG_MODEL_NAME')
    if os.getenv('RAG_TEMPERATURE'):
        rag_config.temperature = float(os.getenv('RAG_TEMPERATURE'))
    if os.getenv('RAG_MAX_TOKENS'):
        rag_config.max_tokens = int(os.getenv('RAG_MAX_TOKENS'))
    if os.getenv('RAG_TOP_K'):
        rag_config.top_k = int(os.getenv('RAG_TOP_K'))
    
    # ChromaDB overrides
    if os.getenv('CHROMADB_HOST'):
        chromadb_config.host = os.getenv('CHROMADB_HOST')
    if os.getenv('CHROMADB_PORT'):
        chromadb_config.port = int(os.getenv('CHROMADB_PORT'))
    if os.getenv('CHROMADB_COLLECTION_NAME'):
        chromadb_config.collection_name = os.getenv('CHROMADB_COLLECTION_NAME')
    if os.getenv('CHROMADB_PERSIST_DIRECTORY'):
        chromadb_config.persist_directory = os.getenv('CHROMADB_PERSIST_DIRECTORY')
    
    # Paths overrides
    if os.getenv('PATHS_PAPERS_DIR'):
        paths_config.papers_dir = os.getenv('PATHS_PAPERS_DIR')
    if os.getenv('PATHS_OUTPUT_DIR'):
        paths_config.output_dir = os.getenv('PATHS_OUTPUT_DIR')
    if os.getenv('PATHS_LOG_DIR'):
        paths_config.log_dir = os.getenv('PATHS_LOG_DIR')
    if os.getenv('PATHS_METADATA_DIR'):
        paths_config.metadata_dir = os.getenv('PATHS_METADATA_DIR')
    
    # Logging overrides
    if os.getenv('LOGGING_LEVEL'):
        logging_config.level = os.getenv('LOGGING_LEVEL')
    if os.getenv('LOGGING_FORMAT'):
        logging_config.format = os.getenv('LOGGING_FORMAT')
    if os.getenv('LOGGING_FILE_ROTATION'):
        logging_config.file_rotation = os.getenv('LOGGING_FILE_ROTATION').lower() == 'true'
    if os.getenv('LOGGING_MAX_FILE_SIZE_MB'):
        logging_config.max_file_size_mb = int(os.getenv('LOGGING_MAX_FILE_SIZE_MB'))
    if os.getenv('LOGGING_BACKUP_COUNT'):
        logging_config.backup_count = int(os.getenv('LOGGING_BACKUP_COUNT'))


def validate_config(config: AppConfig) -> bool:
    """Validate configuration at startup"""
    errors = []
    
    # Check API key
    if not config.openai_api_key:
        errors.append("OPENAI_API_KEY is required")
    
    # Check directories
    required_dirs = [
        config.paths.papers_dir,
        config.paths.output_dir,
        config.paths.log_dir,
        config.paths.metadata_dir
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True)
    
    # Check ChromaDB connection (optional, don't fail if not available)
    try:
        import chromadb
        client = chromadb.HttpClient(
            host=config.chromadb.host,
            port=config.chromadb.port
        )
        client.heartbeat()
    except Exception as e:
        # Don't fail validation for ChromaDB connection issues
        # This allows the system to work even if ChromaDB is not running
        pass
    
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        return False
    
    return True


# Global configuration instance
config = load_config()
