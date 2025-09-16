"""
Error Logging Module
Handles logging of errors with context and traceback information
"""

import traceback
from typing import Dict, Any, Optional
from logging_config import error_logger


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None, 
              user_action: Optional[str] = None) -> None:
    """Log error with context and traceback information"""
    
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc()
    }
    
    if context:
        error_info["context"] = context
    
    if user_action:
        error_info["user_action"] = user_action
    
    error_logger.error(f"Error occurred: {error_info}")


def log_connection_error(error: Exception, document_id: str, url: str) -> None:
    """Log connection error during document download"""
    log_error(error, 
              context={"document_id": document_id, "url": url},
              user_action="download_document")


def log_rate_limit_error(error: Exception, query: str, model: str) -> None:
    """Log rate limit error during search"""
    log_error(error,
              context={"query": query, "model": model},
              user_action="search")


def log_parsing_error(error: Exception, file_path: str) -> None:
    """Log PDF parsing error"""
    log_error(error,
              context={"file_path": file_path},
              user_action="parse_document")


def log_api_error(error: Exception, api_name: str, endpoint: str) -> None:
    """Log API error"""
    log_error(error,
              context={"api_name": api_name, "endpoint": endpoint},
              user_action="api_call")
