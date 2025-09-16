"""
Log Analysis Utilities
Provides tools for analyzing and summarizing log files
"""

import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any


def analyze_logs(log_dir: str = "log", days: int = 7) -> Dict[str, Any]:
    """Analyze logs for the last N days"""
    
    log_path = Path(log_dir)
    cutoff_date = datetime.now() - timedelta(days=days)
    
    stats = {
        "total_actions": 0,
        "errors": 0,
        "slow_queries": 0,
        "total_cost": 0.0,
        "documents_downloaded": 0,
        "searches_performed": 0,
        "high_cost_queries": 0
    }
    
    # Analyze app.log
    app_log_file = log_path / "app.log"
    if app_log_file.exists():
        with open(app_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "Successfully downloaded document" in line:
                    stats["documents_downloaded"] += 1
                elif "ERROR" in line:
                    stats["errors"] += 1
                elif "Search completed successfully" in line:
                    stats["searches_performed"] += 1
    
    # Analyze performance.log
    perf_log_file = log_path / "performance.log"
    if perf_log_file.exists():
        with open(perf_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "Cost: $" in line:
                    cost_match = re.search(r'Cost: \$(\d+\.\d+)', line)
                    if cost_match:
                        stats["total_cost"] += float(cost_match.group(1))
                
                if "Slow" in line:
                    stats["slow_queries"] += 1
                
                if "High cost query" in line:
                    stats["high_cost_queries"] += 1
    
    # Analyze user_actions.log
    user_log_file = log_path / "user_actions.log"
    if user_log_file.exists():
        with open(user_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "Action:" in line:
                    stats["total_actions"] += 1
    
    return stats


def print_log_summary(log_dir: str = "log", days: int = 7) -> None:
    """Print summary of log analysis"""
    stats = analyze_logs(log_dir, days)
    
    print("=== Log Analysis Summary ===")
    print(f"Analysis period: Last {days} days")
    print(f"Total user actions: {stats['total_actions']}")
    print(f"Documents downloaded: {stats['documents_downloaded']}")
    print(f"Searches performed: {stats['searches_performed']}")
    print(f"Total errors: {stats['errors']}")
    print(f"Slow queries: {stats['slow_queries']}")
    print(f"High cost queries: {stats['high_cost_queries']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")


def get_error_summary(log_dir: str = "log", days: int = 7) -> Dict[str, int]:
    """Get summary of error types"""
    log_path = Path(log_dir)
    error_counts = defaultdict(int)
    
    error_log_file = log_path / "errors.log"
    if error_log_file.exists():
        with open(error_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract error type from log line
                error_match = re.search(r'"error_type": "([^"]+)"', line)
                if error_match:
                    error_type = error_match.group(1)
                    error_counts[error_type] += 1
    
    return dict(error_counts)


def get_performance_summary(log_dir: str = "log", days: int = 7) -> Dict[str, Any]:
    """Get performance metrics summary"""
    log_path = Path(log_dir)
    perf_stats = {
        "total_queries": 0,
        "avg_response_time": 0.0,
        "max_response_time": 0.0,
        "total_tokens": 0,
        "total_cost": 0.0
    }
    
    response_times = []
    
    perf_log_file = log_path / "performance.log"
    if perf_log_file.exists():
        with open(perf_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "Response time:" in line:
                    time_match = re.search(r'Response time: ([\d.]+)s', line)
                    if time_match:
                        response_time = float(time_match.group(1))
                        response_times.append(response_time)
                        perf_stats["max_response_time"] = max(perf_stats["max_response_time"], response_time)
                
                if "Total tokens:" in line:
                    tokens_match = re.search(r'Total tokens: (\d+)', line)
                    if tokens_match:
                        perf_stats["total_tokens"] += int(tokens_match.group(1))
                
                if "Cost: $" in line:
                    cost_match = re.search(r'Cost: \$(\d+\.\d+)', line)
                    if cost_match:
                        perf_stats["total_cost"] += float(cost_match.group(1))
    
    if response_times:
        perf_stats["total_queries"] = len(response_times)
        perf_stats["avg_response_time"] = sum(response_times) / len(response_times)
    
    return perf_stats


def print_performance_summary(log_dir: str = "log", days: int = 7) -> None:
    """Print performance metrics summary"""
    stats = get_performance_summary(log_dir, days)
    
    print("=== Performance Summary ===")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average response time: {stats['avg_response_time']:.2f}s")
    print(f"Max response time: {stats['max_response_time']:.2f}s")
    print(f"Total tokens processed: {stats['total_tokens']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")


def print_error_summary(log_dir: str = "log", days: int = 7) -> None:
    """Print error summary"""
    error_counts = get_error_summary(log_dir, days)
    
    print("=== Error Summary ===")
    if error_counts:
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{error_type}: {count}")
    else:
        print("No errors found in the specified period")
