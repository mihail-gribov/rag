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
    """Load configuration from file"""
    # TODO: Implement actual configuration loading logic in task 002
    return AppConfig(
        arxiv=ArxivConfig(),
        document=DocumentConfig(),
        rag=RAGConfig(),
        chromadb=ChromaDBConfig(),
        paths=PathsConfig(),
        logging=LoggingConfig(),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_organization=os.getenv('OPENAI_ORGANIZATION')
    )


# Global configuration instance
config = load_config()
