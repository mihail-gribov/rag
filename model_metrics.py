"""
Model Metrics Logging Module
Handles logging of model performance metrics and costs
"""

from dataclasses import dataclass
from typing import Optional
from logging_config import perf_logger


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model_name: str
    response_time: float
    query: str


def log_model_metrics(metrics: ModelMetrics) -> None:
    """Log model metrics to performance logger"""
    perf_logger.info(f"Model: {metrics.model_name}")
    perf_logger.info(f"Query: {metrics.query[:100]}...")
    perf_logger.info(f"Tokens: {metrics.input_tokens} in, {metrics.output_tokens} out")
    perf_logger.info(f"Total tokens: {metrics.total_tokens}")
    perf_logger.info(f"Cost: ${metrics.cost_usd:.4f}")
    perf_logger.info(f"Response time: {metrics.response_time:.2f}s")
    
    # Warning for high cost queries
    if metrics.cost_usd > 0.01:
        perf_logger.warning(f"High cost query: ${metrics.cost_usd:.4f}")
    
    # Warning for slow responses
    if metrics.response_time > 15:
        perf_logger.warning(f"Slow model response: {metrics.response_time:.2f}s")


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """Calculate cost based on token usage and model pricing"""
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    model_pricing = pricing.get(model_name, pricing["gpt-3.5-turbo"])
    
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    return input_cost + output_cost
