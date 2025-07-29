#!/usr/bin/env python3
"""
Logging utilities for NeMo Japanese fine-tuning.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logging(
    name: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for the package.
    
    Args:
        name: Logger name (use __name__ from calling module)
        level: Logging level
        format_string: Custom format string
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Create logger
    logger = logging.getLogger(name or __name__)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 