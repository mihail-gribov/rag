"""
Log Cleanup Utilities
Provides tools for cleaning up old log files and managing log storage
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict


def cleanup_old_logs(log_dir: str = "log", days_to_keep: int = 30) -> int:
    """Clean up old log files older than specified days"""
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_files = 0
    
    # Clean up archived log files (files with .1, .2, etc. extensions)
    for log_file in log_path.glob("*.log.*"):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            log_file.unlink()
            deleted_files += 1
            print(f"Deleted old log file: {log_file}")
    
    return deleted_files


def get_log_file_sizes(log_dir: str = "log") -> Dict[str, int]:
    """Get sizes of all log files in bytes"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return {}
    
    file_sizes = {}
    for log_file in log_path.glob("*.log*"):
        file_sizes[log_file.name] = log_file.stat().st_size
    
    return file_sizes


def print_log_storage_info(log_dir: str = "log") -> None:
    """Print information about log storage usage"""
    file_sizes = get_log_file_sizes(log_dir)
    
    if not file_sizes:
        print("No log files found")
        return
    
    print("=== Log Storage Information ===")
    total_size = 0
    
    for filename, size in sorted(file_sizes.items()):
        size_mb = size / (1024 * 1024)
        total_size += size
        print(f"{filename}: {size_mb:.2f} MB")
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"Total log storage: {total_size_mb:.2f} MB")


def cleanup_large_logs(log_dir: str = "log", max_size_mb: int = 100) -> int:
    """Clean up log files if total size exceeds threshold"""
    file_sizes = get_log_file_sizes(log_dir)
    total_size_mb = sum(file_sizes.values()) / (1024 * 1024)
    
    if total_size_mb <= max_size_mb:
        print(f"Log storage ({total_size_mb:.2f} MB) is within limit ({max_size_mb} MB)")
        return 0
    
    print(f"Log storage ({total_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)")
    print("Cleaning up old archived files...")
    
    return cleanup_old_logs(log_dir, days_to_keep=7)


def list_log_files(log_dir: str = "log") -> List[Path]:
    """List all log files in the directory"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return []
    
    return list(log_path.glob("*.log*"))


def print_log_files(log_dir: str = "log") -> None:
    """Print list of all log files with their sizes and modification times"""
    log_files = list_log_files(log_dir)
    
    if not log_files:
        print("No log files found")
        return
    
    print("=== Log Files ===")
    for log_file in sorted(log_files):
        stat = log_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        print(f"{log_file.name}: {size_mb:.2f} MB, modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
