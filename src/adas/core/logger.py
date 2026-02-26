"""Structured logging for ADAS system.

This module provides centralized logging configuration and utilities.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler with structured format
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Format: timestamp - name - level - message
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


def log_performance(logger: logging.Logger, operation: str, duration_ms: float) -> None:
    """Log performance metric.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
    """
    logger.info(f"PERF: {operation} took {duration_ms:.2f}ms")


def log_safety_event(logger: logging.Logger, event: str, severity: str, **kwargs: Any) -> None:
    """Log safety-critical event.
    
    Args:
        logger: Logger instance
        event: Event description
        severity: Severity level (INFO, WARNING, CRITICAL)
        **kwargs: Additional context
    """
    context = " ".join(f"{k}={v}" for k, v in kwargs.items())
    msg = f"SAFETY [{severity}]: {event}"
    if context:
        msg += f" | {context}"
    
    if severity == "CRITICAL":
        logger.critical(msg)
    elif severity == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)
