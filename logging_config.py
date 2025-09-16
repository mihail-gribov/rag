"""
Logging Configuration Module
Handles setup and configuration of logging system with file rotation
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Tuple
from config import config


def setup_logging() -> Tuple[logging.Logger, logging.Logger, logging.Logger, logging.Logger]:
    """Setup logging system with specialized loggers and file rotation"""
    
    # Create log directory
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Log format
    formatter = logging.Formatter(config.logging.format)
    
    # Main application logger
    app_logger = logging.getLogger('arxiv_rag')
    app_logger.setLevel(getattr(logging, config.logging.level))
    
    # Clear existing handlers to avoid duplicates
    app_logger.handlers.clear()
    
    # File handler with rotation for main app logs
    if config.logging.file_rotation:
        app_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'app.log',
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
    else:
        app_handler = logging.FileHandler(log_dir / 'app.log')
    
    app_handler.setFormatter(formatter)
    app_logger.addHandler(app_handler)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)
    
    # Performance logger
    perf_logger = logging.getLogger('arxiv_rag.performance')
    perf_logger.setLevel(logging.INFO)
    perf_logger.handlers.clear()
    
    if config.logging.file_rotation:
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'performance.log',
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
    else:
        perf_handler = logging.FileHandler(log_dir / 'performance.log')
    
    perf_handler.setFormatter(formatter)
    perf_logger.addHandler(perf_handler)
    
    # User actions logger
    user_logger = logging.getLogger('arxiv_rag.user_actions')
    user_logger.setLevel(logging.INFO)
    user_logger.handlers.clear()
    
    if config.logging.file_rotation:
        user_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'user_actions.log',
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
    else:
        user_handler = logging.FileHandler(log_dir / 'user_actions.log')
    
    user_handler.setFormatter(formatter)
    user_logger.addHandler(user_handler)
    
    # Error logger
    error_logger = logging.getLogger('arxiv_rag.errors')
    error_logger.setLevel(logging.ERROR)
    error_logger.handlers.clear()
    
    if config.logging.file_rotation:
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'errors.log',
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
    else:
        error_handler = logging.FileHandler(log_dir / 'errors.log')
    
    error_handler.setFormatter(formatter)
    error_logger.addHandler(error_handler)
    
    return app_logger, perf_logger, user_logger, error_logger


# Initialize loggers
app_logger, perf_logger, user_logger, error_logger = setup_logging()