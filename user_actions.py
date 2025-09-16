"""
User Actions Logging Module
Handles logging of user actions and interactions
"""

from datetime import datetime
from typing import Dict, Any, Optional
from logging_config import user_logger


def log_user_action(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log user action to user actions logger"""
    timestamp = datetime.now().isoformat()
    
    log_entry = f"Action: {action}"
    if details:
        # Format details as key-value pairs
        detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        log_entry += f" | Details: {detail_str}"
    
    user_logger.info(log_entry)


def log_fetch_action(query: str, max_results: int) -> None:
    """Log document fetch action"""
    log_user_action("fetch_documents", {
        "query": query,
        "max_results": max_results
    })


def log_search_action(query: str) -> None:
    """Log search action"""
    log_user_action("search", {
        "query": query
    })


def log_clear_action(confirmed: bool) -> None:
    """Log database clear action"""
    log_user_action("clear_database", {
        "confirmed": confirmed
    })


def log_save_action(filename: str, output_dir: Optional[str] = None) -> None:
    """Log save to file action"""
    details = {"filename": filename}
    if output_dir:
        details["output_dir"] = output_dir
    
    log_user_action("save_to_file", details)


def log_list_action(papers_dir: str) -> None:
    """Log list documents action"""
    log_user_action("list_documents", {
        "papers_dir": papers_dir
    })
